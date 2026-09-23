"""
kNN graph construction.

This module is the central hub that wires together PCA, batch correction,
and kNN search into a single ``build_knn_graph`` call.

Architecture
------------
All internal computation works on plain NumPy arrays; AnnData is only
accessed at the entry point to extract the logcounts matrix and batch
labels.  This keeps the internals fast and easy to unit-test in isolation.

Data flow (single batch, batch_key=None)
    logcounts (n_cells, n_genes)
        → PCA → embedding (n_cells, n_pcs)
        → KNNBackend.fit_query → indices (n_cells, k), distances (n_cells, k)

Data flow (per-batch, batch_method="per_batch" — geneBasisR default)
    For each batch b:
        logcounts[batch==b] (n_b, n_genes)
            → PCA → embedding (n_b, n_pcs)
            → KNNBackend.fit_query → local indices/distances
        remap local → global cell indices
    Stitch per-batch results into full (n_cells, k) arrays.

Data flow (cross-batch correction, batch_method="mnn"/"harmony")
  Prefer "harmony" — "mnn" is a from-scratch sklearn path that does not scale.
    logcounts (n_cells, n_genes)
        → PCA → embedding (n_cells, n_pcs)
        → BatchCorrector.fit_transform → corrected embedding (n_cells, n_pcs)
        → KNNBackend.fit_query → indices (n_cells, k), distances (n_cells, k)

Design decisions
----------------
- PCA uses sklearn.decomposition.PCA (centers data automatically, matching
  irlba::prcomp_irlba in geneBasisR which also centers).
- Default batch method is "per_batch" matching geneBasisR's actual behavior:
  separate PCA + within-batch kNN per batch, stitched into a global result.
  "mnn" and "harmony" are available for explicit cross-batch correction.
- Default kNN method is "approx" (PyNNDescent) for production use, but
  "exact" (sklearn) is available for correctness testing against R.
- All index arrays are 0-indexed (Python convention).
- Random state is fixed at 32 to match geneBasisR's internal set.seed(32).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
from anndata import AnnData

from ._backends import get_knn_backend, get_batch_corrector


def build_knn_graph(
    adata: AnnData,
    genes: list[str],
    *,
    batch_key: str | None = None,
    knn_method: str = "approx",
    batch_method: str = "per_batch",
    n_neighbors: int = 5,
    n_pcs: int | None = 50,
    layer: str | None = None,
    random_state: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a kNN graph over cells using a given gene subset.

    Parameters
    ----------
    adata : AnnData
        Single-cell dataset. Logcounts must be in ``adata.X`` or the layer
        specified by ``layer``.
    genes : list[str]
        Gene subset to use for graph construction.
    batch_key : str, optional
        Column in ``adata.obs`` identifying batches. If None, no batch
        handling is applied and all cells are embedded together.
    knn_method : {"approx", "exact"}
        kNN backend. "approx" (PyNNDescent) is the default production backend;
        "exact" (sklearn brute-force) is used for correctness testing vs R.
    batch_method : {"per_batch", "mnn", "harmony"}
        How batches are handled when ``batch_key`` is set.
        "per_batch" (default): separate PCA + within-batch kNN per batch,
            stitched into a global result. Matches geneBasisR's default behavior.
        "mnn": cross-batch MNN correction (custom sklearn implementation).
        "harmony": cross-batch Harmony correction (harmonypy package).
    n_neighbors : int
        Number of neighbours k. Default 5 matches geneBasisR.
    n_pcs : int or None
        Number of PCA components. None skips PCA (used when gene panel is
        very small). Default 50 matches geneBasisR's nPC.all.
    layer : str, optional
        AnnData layer holding logcounts. If None, uses ``adata.X``.
    random_state : int
        Random seed. Default 32 matches geneBasisR's internal set.seed(32).

    Returns
    -------
    indices : np.ndarray, shape (n_cells, n_neighbors), dtype int
        0-indexed neighbour indices ordered by ascending distance.
    distances : np.ndarray, shape (n_cells, n_neighbors), dtype float
        Corresponding Euclidean distances.
    """
    knn = get_knn_backend(knn_method, random_state=random_state)

    if batch_key is not None and batch_method == "per_batch":
        batch_labels = adata.obs[batch_key].values
        return _build_per_batch_knn(
            adata, genes, batch_labels, n_pcs, n_neighbors, knn, layer, random_state,
        )

    # Cross-batch correction paths (mnn, harmony) or no-batch path
    embedding = _build_embedding(
        adata, genes,
        batch_key=batch_key,
        batch_method=batch_method,
        n_pcs=n_pcs,
        layer=layer,
        random_state=random_state,
    )
    indices, distances = knn.fit_query(embedding.astype(np.float32), n_neighbors)
    return indices.astype(np.int32), distances.astype(np.float64)


# ---------------------------------------------------------------------------
# Internal helpers (prefixed with _ — not part of the public API)
# ---------------------------------------------------------------------------

def _build_per_batch_knn(
    adata: AnnData,
    genes: list[str],
    batch_labels: np.ndarray,
    n_pcs: int | None,
    n_neighbors: int,
    knn_backend,
    layer: str | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-batch PCA + within-batch kNN, matching geneBasisR's batch handling.

    For each batch: run PCA on that batch's cells, find kNN within the batch,
    map local indices to global cell indices.  Results are stitched into
    full (n_cells, n_neighbors) arrays — exactly what .get_mapping() in
    geneBasisR produces.

    Parameters
    ----------
    adata, genes, n_pcs, n_neighbors, knn_backend, layer, random_state
        Same semantics as in build_knn_graph.
    batch_labels : np.ndarray, shape (n_cells,)
        Batch identifier per cell.

    Returns
    -------
    indices   : np.ndarray, shape (n_cells, n_neighbors), dtype int32
    distances : np.ndarray, shape (n_cells, n_neighbors), dtype float64
    """
    n_cells = len(adata)
    all_indices = np.full((n_cells, n_neighbors), -1, dtype=np.int32)
    all_distances = np.full((n_cells, n_neighbors), np.nan, dtype=np.float64)

    for batch in np.unique(batch_labels):
        batch_mask = np.where(batch_labels == batch)[0]  # global cell indices

        X = _extract_logcounts(adata[batch_mask], genes, layer)

        if n_pcs is not None and n_pcs > 0 and X.shape[1] > n_pcs:
            emb = _pca(X, n_pcs, random_state).astype(np.float32)
        else:
            emb = X.astype(np.float32)

        # Local kNN: indices are in [0, len(batch_mask))
        local_idx, local_dist = knn_backend.fit_query(emb, n_neighbors)

        # Remap local → global cell indices
        all_indices[batch_mask] = batch_mask[local_idx]
        all_distances[batch_mask] = local_dist.astype(np.float64)

    return all_indices, all_distances


def _build_embedding(
    adata: AnnData,
    genes: list[str],
    *,
    batch_key: str | None = None,
    batch_method: str = "per_batch",
    n_pcs: int | None = 50,
    layer: str | None = None,
    random_state: int = 32,
) -> np.ndarray:
    """Extract logcounts, apply PCA and optional batch correction.

    This helper is shared by ``build_knn_graph`` and evaluation functions
    that need the embedding itself (e.g. for computing pairwise distances).

    Parameters
    ----------
    adata : AnnData
    genes : list[str]
    batch_key : str or None
    batch_method : {"per_batch", "mnn", "harmony"}
    n_pcs : int or None
        None skips PCA entirely.
    layer : str or None
    random_state : int

    Returns
    -------
    embedding : np.ndarray, shape (n_cells, n_features), dtype float64
        For "per_batch" mode with PCA: each batch's PCA is computed
        independently and stacked.  Within-batch Euclidean distances are
        valid; cross-batch distances are not meaningful (different PC bases).
    """
    X = _extract_logcounts(adata, genes, layer)

    if batch_key is not None and batch_method == "per_batch":
        # Per-batch PCA: compute separately per batch, then stack.
        # Raw gene space (n_pcs=None or n_pcs>=n_genes): consistent across
        # batches, so no special handling needed.
        if n_pcs is not None and n_pcs > 0 and X.shape[1] > n_pcs:
            batch_labels = adata.obs[batch_key].values
            batches = np.unique(batch_labels)
            batch_embs = {}
            for b in batches:
                mask = np.where(batch_labels == b)[0]
                batch_embs[b] = _pca(X[mask], n_pcs, random_state).astype(np.float64)
            n_out = min(e.shape[1] for e in batch_embs.values())
            embedding = np.zeros((X.shape[0], n_out), dtype=np.float64)
            for b in batches:
                mask = np.where(batch_labels == b)[0]
                embedding[mask] = batch_embs[b][:, :n_out]
        else:
            embedding = X.astype(np.float64)
        return embedding

    # Harmony requires a dense PCA embedding — it normalises each cell to unit
    # length, so raw sparse gene matrices with zero-expression cells produce NaN.
    # If n_pcs is None (caller requested raw gene space) but batch_method is
    # "harmony", silently apply a default PCA before correction.
    _effective_n_pcs = n_pcs
    if batch_key is not None and batch_method == "harmony" and (
        n_pcs is None or n_pcs <= 0
    ):
        _effective_n_pcs = min(50, X.shape[1] - 1)

    if _effective_n_pcs is not None and _effective_n_pcs > 0 and X.shape[1] > _effective_n_pcs:
        embedding = _pca(X, _effective_n_pcs, random_state).astype(np.float64)
    else:
        embedding = X.astype(np.float64)

    if batch_key is not None:
        batch_labels = adata.obs[batch_key].values
        corrector = get_batch_corrector(batch_method, random_state=random_state)
        embedding = corrector.fit_transform(embedding, batch_labels)

    return embedding


def _extract_logcounts(
    adata: AnnData,
    genes: list[str],
    layer: str | None,
) -> np.ndarray:
    """Extract a dense (n_cells, n_genes) logcounts matrix for the given genes.

    Parameters
    ----------
    adata : AnnData
    genes : list[str]
    layer : str or None

    Returns
    -------
    X : np.ndarray, shape (n_cells, n_genes), dtype float32
    """
    if layer is not None:
        X = adata[:, genes].layers[layer]
    else:
        X = adata[:, genes].X

    if scipy.sparse.issparse(X):
        X = X.toarray()

    return np.asarray(X, dtype=np.float32)


def _pca(X: np.ndarray, n_pcs: int, random_state: int) -> np.ndarray:
    """Apply PCA (centering + SVD) to a logcounts matrix.

    Uses sklearn.decomposition.PCA which centers the data automatically,
    matching irlba::prcomp_irlba in geneBasisR.  n_pcs is clamped to
    min(n_cells-1, n_genes-1) to avoid requesting more components than
    the matrix rank allows.

    Parameters
    ----------
    X : np.ndarray, shape (n_cells, n_genes)
    n_pcs : int
    random_state : int

    Returns
    -------
    embedding : np.ndarray, shape (n_cells, n_pcs_actual)
    """
    from sklearn.decomposition import PCA

    n_pcs_actual = min(n_pcs, X.shape[0] - 1, X.shape[1] - 1)
    pca = PCA(n_components=n_pcs_actual, random_state=random_state)
    return pca.fit_transform(X.astype(np.float64))
