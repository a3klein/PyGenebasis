"""
Graph-geometry metrics: how well the panel preserves transcriptome structure.

Nothing here knows about cell types.  These compare two kNN graphs -- one
built from the panel, one from the full feature set -- and ask how far the
panel's neighbours are from the true ones, and how well the panel's
neighbourhoods predict each gene.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def get_neighs_all_stat(
    adata: AnnData,
    *,
    genes_all: list[str] | None = None,
    batch_key: str | None = None,
    n_neighbors: int = 5,
    n_pcs_all: int = 50,
    option: str = "exact",
    knn_method: str = "approx",
    batch_method: str = "per_batch",
    layer: str | None = None,
    random_state: int = 32,
) -> dict:
    """Pre-compute true-graph neighbour statistics.

    Call this once and pass the result to ``get_neighborhood_preservation_scores``
    via ``neighs_all_stat`` to avoid recomputing the true graph on repeated
    calls (e.g. when evaluating multiple panel sizes).

    Returns
    -------
    dict with keys:
        "indices"       : np.ndarray (n_cells, k) — true-graph neighbour indices
        "embedding"     : np.ndarray (n_cells, n_pcs) — all-genes embedding
        "mean_dist_all" : np.ndarray (n_cells,)   — mean distance to all cells
                          in the all-genes embedding
    """
    from ..knn._graph import build_knn_graph, _build_embedding

    if genes_all is None:
        genes_all = adata.var_names.tolist()

    # Build the true-graph embedding once (used for both kNN and distances)
    all_embedding = _build_embedding(
        adata, genes_all,
        batch_key=batch_key, batch_method=batch_method,
        n_pcs=n_pcs_all, layer=layer, random_state=random_state,
    )

    true_indices, _ = build_knn_graph(
        adata, genes_all,
        batch_key=batch_key, knn_method=knn_method, batch_method=batch_method,
        n_neighbors=n_neighbors, n_pcs=n_pcs_all, layer=layer,
        random_state=random_state,
    )

    batch_labels_arr = adata.obs[batch_key].values if batch_key is not None else None
    per_batch_dists = (batch_method == "per_batch") and (batch_labels_arr is not None)
    mean_dist_all = _mean_dist_all(
        all_embedding, option=option, random_state=random_state,
        batch_labels=batch_labels_arr if per_batch_dists else None,
    )

    return {
        "indices": true_indices,
        "embedding": all_embedding,
        "mean_dist_all": mean_dist_all,
    }


def knn_overlap(indices_a, indices_b) -> np.ndarray:
    """Per-cell fraction of neighbours two kNN graphs agree on.

    Needs only the neighbour indices, so it works on any pair of graphs
    regardless of how their embeddings were built — which makes it the one
    preservation number comparable across modalities.  Bounded [0, 1], unlike
    the ratio score.

    Parameters
    ----------
    indices_a, indices_b : array-like, shape (n_cells, k)
        Neighbour indices.  ``k`` may differ between them; the fraction is
        taken over the smaller.

    Returns
    -------
    np.ndarray, shape (n_cells,)
    """
    a = np.asarray(indices_a)
    b = np.asarray(indices_b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected 2-D index arrays, got {a.ndim}-D and {b.ndim}-D")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"different cell counts: {a.shape[0]} vs {b.shape[0]}")
    k = min(a.shape[1], b.shape[1])
    return np.fromiter(
        (len(set(ra[:k]).intersection(rb[:k])) / k for ra, rb in zip(a, b)),
        dtype=np.float64, count=a.shape[0])


def preservation_from_embedding(
    reference_embedding,
    reference_indices,
    selection_indices,
    *,
    mean_dist_all=None,
    batch_labels=None,
    option: str = "exact",
    random_state: int = 32,
) -> np.ndarray:
    """Neighbourhood preservation from precomputed embeddings and graphs.

    The scoring maths, separated from how the inputs were built, so a pipeline
    that makes its own embedding — methylation via ALLCools, say — can use the
    same metric as the RNA path.  ``get_neighborhood_preservation_scores`` is
    this function plus the AnnData plumbing.

    Note the asymmetry: the reference side needs **coordinates**, because the
    score measures distances from each cell to the panel's chosen neighbours
    and those pairs are not edges of the reference graph.  The selection side
    needs only **indices** — the panel embedding's own geometry never enters.

    Parameters
    ----------
    reference_embedding : array-like, shape (n_cells, n_features)
        Coordinates of the reference (all-feature) embedding.  Every distance
        below is measured here.
    reference_indices : array-like, shape (n_cells, k)
        Neighbours from the reference graph.
    selection_indices : array-like, shape (n_cells, k)
        Neighbours from the panel graph.
    mean_dist_all : array-like, optional
        Per-cell mean distance to all other cells.  Derived from the embedding
        when omitted.
    batch_labels : array-like, optional
        Restricts ``mean_dist_all`` to within-batch distances, matching what
        ``batch_method="per_batch"`` does.  Ignored if ``mean_dist_all`` is
        supplied.
    option : {"exact", "approx"}
        Whether ``mean_dist_all`` uses all cells or a 10% sample.
    random_state : int
        Seeds the sample when ``option="approx"``.

    Returns
    -------
    np.ndarray, shape (n_cells,)
        Per-cell score; 1.0 means the panel recovers the reference
        neighbourhood exactly.  Unbounded in both directions — see
        ``docs/panel_evaluation.md``.
    """
    emb = np.asarray(reference_embedding, dtype=np.float64)
    ref_idx = np.asarray(reference_indices)
    sel_idx = np.asarray(selection_indices)
    if emb.ndim != 2:
        raise ValueError(f"reference_embedding must be 2-D, got {emb.ndim}-D")
    for name, idx in (("reference_indices", ref_idx), ("selection_indices", sel_idx)):
        if idx.shape[0] != emb.shape[0]:
            raise ValueError(
                f"{name} has {idx.shape[0]} rows, embedding has {emb.shape[0]}")

    if mean_dist_all is None:
        mean_dist_all = _mean_dist_all(
            emb, option=option, random_state=random_state,
            batch_labels=None if batch_labels is None else np.asarray(batch_labels))
    mean_dist_all = np.asarray(mean_dist_all, dtype=np.float64)

    dist_true = _median_neighbour_dist(emb, ref_idx)
    dist_selection = _median_neighbour_dist(emb, sel_idx)

    denom = mean_dist_all - dist_true
    # identical neighbourhoods make this zero; NaN is the honest answer
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    return (mean_dist_all - dist_selection) / denom


def get_neighborhood_preservation_scores(
    adata: AnnData,
    genes_selection: list[str],
    *,
    genes_all: list[str] | None = None,
    batch_key: str | None = None,
    n_neighbors: int = 5,
    n_pcs_all: int = 50,
    n_pcs_selection: int | None = None,
    knn_method: str = "approx",
    batch_method: str = "per_batch",
    option: str = "exact",
    neighs_all_stat: dict | None = None,
    layer: str | None = None,
    random_state: int = 32,
) -> pd.DataFrame:
    """Compute per-cell neighbourhood preservation scores.

    Parameters
    ----------
    adata : AnnData
    genes_selection : list[str]
        Selected gene panel.
    genes_all : list[str], optional
        Genes defining the true graph. Defaults to all genes in adata.
    batch_key : str, optional
    n_neighbors : int
        k. Default 5.
    n_pcs_all : int
        PCs for the true graph. Default 50.
    n_pcs_selection : int or None
        PCs for the selection graph. None skips PCA (default; use when
        panel is small).
    knn_method : {"approx", "exact"}
    batch_method : {"mnn", "harmony"}
    option : {"exact", "approx"}
        How to compute mean_dist_all. "exact" uses all cells; "approx"
        samples 10% (faster for large datasets).
    neighs_all_stat : dict, optional
        Pre-computed true-graph statistics from get_neighs_all_stat.
        Pass this to avoid recomputing the true graph on repeated calls.
    layer : str, optional
    random_state : int

    Returns
    -------
    pd.DataFrame
        Columns: ``cell`` (str), ``cell_score`` (float), ``knn_overlap``
        (float).  ``knn_overlap`` is the bounded [0, 1] alternative and is the
        one comparable across modalities.
    """
    from ..knn._graph import build_knn_graph

    if genes_all is None:
        genes_all = adata.var_names.tolist()

    # --- True graph statistics (pre-computed or computed here) ---
    if neighs_all_stat is None:
        neighs_all_stat = get_neighs_all_stat(
            adata,
            genes_all=genes_all,
            batch_key=batch_key,
            n_neighbors=n_neighbors,
            n_pcs_all=n_pcs_all,
            option=option,
            knn_method=knn_method,
            batch_method=batch_method,
            layer=layer,
            random_state=random_state,
        )

    true_indices = neighs_all_stat["indices"]        # (n_cells, k)
    mean_dist_all = neighs_all_stat["mean_dist_all"]  # (n_cells,)
    # All distances are computed in the all-genes embedding (same space as
    # mean_dist_all) to keep the cell_score formula dimensionally consistent.
    all_embedding = neighs_all_stat["embedding"]      # (n_cells, n_pcs_all)

    # --- Selection graph ---
    sel_indices, _ = build_knn_graph(
        adata, genes_selection,
        batch_key=batch_key, knn_method=knn_method, batch_method=batch_method,
        n_neighbors=n_neighbors, n_pcs=n_pcs_selection, layer=layer,
        random_state=random_state,
    )

    # mean_dist_all is already batch-restricted by get_neighs_all_stat when
    # batch_method="per_batch", so it is passed through rather than recomputed.
    cell_scores = preservation_from_embedding(
        all_embedding, true_indices, sel_indices,
        mean_dist_all=mean_dist_all,
    )

    return pd.DataFrame({
        "cell": adata.obs_names.tolist(),
        "cell_score": cell_scores,
        "knn_overlap": knn_overlap(true_indices, sel_indices),
    })


def get_gene_prediction_scores(
    adata: AnnData,
    genes_selection: list[str],
    *,
    genes_all: list[str] | None = None,
    batch_key: str | None = None,
    n_neighbors: int = 5,
    n_pcs_all: int = 50,
    n_pcs_selection: int | None = None,
    knn_method: str = "approx",
    batch_method: str = "per_batch",
    method: str = "spearman",
    layer: str | None = None,
    random_state: int = 32,
) -> pd.DataFrame:
    """Compute per-gene prediction scores.

    For each gene, measures how well its expression is predicted by the
    panel's kNN neighbourhood, normalised by how well the true graph predicts
    it.  Score ≈ 1 means the panel recovers gene expression as well as using
    all genes.

    Parameters
    ----------
    adata : AnnData
    genes_selection : list[str]
    genes_all : list[str], optional
    batch_key : str, optional
    n_neighbors : int
    n_pcs_all : int
    n_pcs_selection : int or None
    knn_method : {"approx", "exact"}
    batch_method : {"mnn", "harmony"}
    method : {"spearman", "pearson"}
        Correlation method.  Default "spearman" matches geneBasisR.
    layer : str, optional
    random_state : int

    Returns
    -------
    pd.DataFrame
        Columns: ``gene`` (str), ``corr`` (float), ``corr_all`` (float),
        ``gene_score`` (float = corr / corr_all).
    """
    from ..knn._graph import build_knn_graph, _extract_logcounts

    if genes_all is None:
        genes_all = adata.var_names.tolist()

    # True graph kNN
    true_indices, _ = build_knn_graph(
        adata, genes_all,
        batch_key=batch_key, knn_method=knn_method, batch_method=batch_method,
        n_neighbors=n_neighbors, n_pcs=n_pcs_all, layer=layer,
        random_state=random_state,
    )

    # Selection graph kNN
    sel_indices, _ = build_knn_graph(
        adata, genes_selection,
        batch_key=batch_key, knn_method=knn_method, batch_method=batch_method,
        n_neighbors=n_neighbors, n_pcs=n_pcs_selection, layer=layer,
        random_state=random_state,
    )

    # Logcounts for all genes
    X_all = _extract_logcounts(adata, genes_all, layer).astype(np.float64)

    corr_fn = _spearman_vec if method == "spearman" else _pearson_vec

    # Neighbour-averaged prediction from each graph
    corr_sel = corr_fn(X_all, sel_indices)
    corr_all = corr_fn(X_all, true_indices)

    gene_score = np.where(
        np.abs(corr_all) < 1e-12,
        np.nan,
        corr_sel / corr_all,
    )

    return pd.DataFrame({
        "gene": genes_all,
        "corr": corr_sel,
        "corr_all": corr_all,
        "gene_score": gene_score,
    })



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _median_neighbour_dist(
    embedding: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Compute the median Euclidean distance from each cell to its neighbours.

    Parameters
    ----------
    embedding : np.ndarray, shape (n_cells, n_features)
    indices   : np.ndarray, shape (n_cells, k)

    Returns
    -------
    median_dists : np.ndarray, shape (n_cells,)
    """
    # (n_cells, k, n_features)
    neighbour_embeddings = embedding[indices]
    # (n_cells, k)
    diffs = embedding[:, np.newaxis, :] - neighbour_embeddings
    dists = np.linalg.norm(diffs, axis=2)
    return np.median(dists, axis=1)


def _mean_dist_all(
    embedding: np.ndarray,
    option: str = "exact",
    random_state: int = 32,
    sample_frac: float = 0.10,
    batch_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Mean Euclidean distance from each cell to all (or sampled) other cells.

    Parameters
    ----------
    embedding : np.ndarray, shape (n_cells, n_features)
    option : {"exact", "approx"}
        "exact"  — pairwise distances to all cells (memory-intensive for large N)
        "approx" — sample 10% of cells as reference
    random_state : int
    sample_frac  : float
        Fraction of cells to sample in "approx" mode.
    batch_labels : np.ndarray, shape (n_cells,), optional
        If provided, distances are computed within each batch separately.
        Required when the embedding is from per-batch PCA (different PC bases
        across batches make cross-batch distances meaningless).

    Returns
    -------
    mean_dists : np.ndarray, shape (n_cells,)
    """
    if batch_labels is not None:
        # Compute per-batch and assemble into a single output vector.
        n_cells = embedding.shape[0]
        mean_dists = np.zeros(n_cells, dtype=np.float64)
        for batch in np.unique(batch_labels):
            mask = np.where(batch_labels == batch)[0]
            mean_dists[mask] = _mean_dist_all(
                embedding[mask], option=option,
                random_state=random_state, sample_frac=sample_frac,
            )
        return mean_dists

    n_cells = embedding.shape[0]

    if option == "approx":
        rng = np.random.default_rng(random_state)
        n_sample = max(1, int(n_cells * sample_frac))
        sample_idx = rng.choice(n_cells, size=n_sample, replace=False)
        ref = embedding[sample_idx]
    else:
        ref = embedding

    # Compute pairwise distances in chunks to limit memory usage
    chunk = 512
    mean_dists = np.zeros(n_cells, dtype=np.float64)
    for start in range(0, n_cells, chunk):
        end = min(start + chunk, n_cells)
        diff = embedding[start:end, np.newaxis, :] - ref[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)  # (chunk, n_ref)
        mean_dists[start:end] = dists.mean(axis=1)

    return mean_dists


def _spearman_vec(
    X: np.ndarray,
    knn_indices: np.ndarray,
) -> np.ndarray:
    """Vectorised Spearman correlation between each gene and its kNN prediction.

    For each gene g:
        predicted[i] = mean(X[neighbours_of_i, g])
        corr(g)      = spearman_r(X[:, g], predicted[:, g])

    Parameters
    ----------
    X           : np.ndarray, shape (n_cells, n_genes)
    knn_indices : np.ndarray, shape (n_cells, k)

    Returns
    -------
    corrs : np.ndarray, shape (n_genes,)
    """
    # Neighbour-averaged prediction for all genes at once
    predicted = X[knn_indices].mean(axis=1)  # (n_cells, n_genes)

    # Rank-transform each gene column, then compute Pearson (= Spearman)
    from scipy.stats import rankdata

    corrs = np.empty(X.shape[1], dtype=np.float64)
    n = X.shape[0]
    for g in range(X.shape[1]):
        r_x = rankdata(X[:, g])
        r_p = rankdata(predicted[:, g])
        # Pearson on ranks = Spearman
        mx = r_x - r_x.mean()
        mp = r_p - r_p.mean()
        denom = np.sqrt((mx ** 2).sum() * (mp ** 2).sum())
        corrs[g] = (mx * mp).sum() / denom if denom > 0 else 0.0

    return corrs


def _pearson_vec(
    X: np.ndarray,
    knn_indices: np.ndarray,
) -> np.ndarray:
    """Vectorised Pearson correlation between each gene and its kNN prediction."""
    predicted = X[knn_indices].mean(axis=1)  # (n_cells, n_genes)
    X_c = X - X.mean(axis=0)
    P_c = predicted - predicted.mean(axis=0)
    num = (X_c * P_c).sum(axis=0)
    denom = np.sqrt((X_c ** 2).sum(axis=0) * (P_c ** 2).sum(axis=0))
    return np.where(denom > 0, num / denom, 0.0)
