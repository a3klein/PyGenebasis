"""
Batch correction backend implementations.

Design decision: batch correction is abstracted so that graph.py can swap
between MNN and Harmony without any change to the calling code.  The user
selects a backend via the ``batch_method`` flag.

Backends
--------
mnn     → MNNCorrector     — Mutual Nearest Neighbours, implemented from scratch
                              using sklearn NearestNeighbors.  Matches
                              batchelor::reducedMNN() from Haghverdi et al. 2018.
                              Primary / default backend.  Does NOT rely on scanpy
                              or mnnpy packages.
harmony → HarmonyCorrector — Harmony integration via the standalone ``harmonypy``
                              package (NOT scanpy's wrapper).  We use harmonypy
                              directly because it exposes the full API and avoids
                              coupling to scanpy's version.

Both correctors accept a PCA embedding and batch labels, and return a
batch-corrected embedding of the same shape.

MNN algorithm (Haghverdi et al. 2018)
--------------------------------------
For each pair of batches (reference, query):
  1. Find mutual nearest neighbors (MNNs): cell a in reference and cell b in
     query are MNNs if a is among the k nearest reference cells to b AND b is
     among the k nearest query cells to a.
  2. Compute per-cell correction vectors: for each query cell q, the correction
     is the Gaussian-weighted sum of (ref_a - query_b) vectors from nearby MNN
     pairs, with sigma controlling the smoothing bandwidth.
  3. Apply corrections: corrected_b = b + correction(b).
Batches are merged sequentially: batch1 corrected toward batch0, then batch2
corrected toward the merged (batch0+corrected_batch1) set, etc.

This matches batchelor::reducedMNN() which also operates in PCA space and uses
sequential merging.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BatchCorrector(ABC):
    """Common interface for batch correction methods."""

    @abstractmethod
    def fit_transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
    ) -> np.ndarray:
        """Correct a PCA embedding for batch effects.

        Parameters
        ----------
        X : np.ndarray, shape (n_cells, n_pcs)
            PCA embedding before correction.
        batch_labels : np.ndarray, shape (n_cells,)
            Batch identifier per cell (string or integer).

        Returns
        -------
        X_corrected : np.ndarray, shape (n_cells, n_pcs)
            Batch-corrected embedding.
        """


class MNNCorrector(BatchCorrector):
    """MNN-based batch correction matching batchelor::reducedMNN().

    Implemented from scratch using sklearn NearestNeighbors — no mnnpy or
    scanpy dependency.

    For each pair of batches, finds mutual nearest neighbours in PCA space
    and computes Gaussian-smoothed correction vectors.  Batches are merged
    sequentially (batch[i] corrected toward the merged batch[0..i-1]).

    Parameters
    ----------
    k : int
        Number of nearest neighbours to use when finding MNN pairs.
        Default 20 matches batchelor::reducedMNN() default.
    sigma : float
        Gaussian kernel bandwidth for smoothing correction vectors.
        Default 1.0 matches batchelor::reducedMNN() default.
    random_state : int
        Kept for API symmetry; unused (MNN is deterministic).
    """

    def __init__(
        self,
        k: int = 20,
        sigma: float = 1.0,
        random_state: int = 32,
    ) -> None:
        self.k = k
        self.sigma = sigma
        self.random_state = random_state

    def fit_transform(self, X: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        batches = list(np.unique(batch_labels))

        if len(batches) == 1:
            return X.copy()

        # Track original cell positions
        batch_masks = {b: np.where(batch_labels == b)[0] for b in batches}

        corrected = X.copy()

        # Sequential merging: correct batch[i] toward union of batch[0..i-1]
        for i in range(1, len(batches)):
            ref_batches = batches[:i]
            query_batch = batches[i]

            ref_idx = np.concatenate([batch_masks[b] for b in ref_batches])
            query_idx = batch_masks[query_batch]

            X_ref = corrected[ref_idx]
            X_query = corrected[query_idx]

            correction = _mnn_correct_pair(X_ref, X_query, k=self.k, sigma=self.sigma)
            corrected[query_idx] = X_query + correction

        return corrected


class HarmonyCorrector(BatchCorrector):
    """Harmony batch correction via the standalone ``harmonypy`` package.

    Design decision: we call ``harmonypy.run_harmony()`` directly rather than
    ``scanpy.external.pp.harmony_integrate`` to avoid coupling to scanpy's
    internal AnnData representation and to access Harmony's full parameter set.

    Parameters
    ----------
    random_state : int
        Random seed passed to Harmony.
    max_iter_harmony : int
        Maximum number of Harmony iterations.
    """

    def __init__(self, random_state: int = 32, max_iter_harmony: int = 10) -> None:
        self.random_state = random_state
        self.max_iter_harmony = max_iter_harmony

    def fit_transform(self, X: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
        import pandas as pd
        import harmonypy

        X = np.asarray(X, dtype=np.float64)
        meta = pd.DataFrame({"batch": batch_labels})
        ho = harmonypy.run_harmony(
            X,
            meta,
            vars_use="batch",
            random_state=self.random_state,
            max_iter_harmony=self.max_iter_harmony,
        )
        # harmonypy Z_corr property already returns (n_cells, n_pcs) — no transpose needed.
        return np.asarray(ho.Z_corr, dtype=np.float64)


def get_batch_corrector(method: str, **kwargs) -> BatchCorrector:
    """Factory: return the batch corrector for the given ``batch_method`` flag.

    Parameters
    ----------
    method : {"mnn", "harmony"}
        "mnn"     → MNNCorrector     (default; matches geneBasisR)
        "harmony" → HarmonyCorrector (via harmonypy)
    **kwargs
        Forwarded to the corrector constructor.
    """
    if method == "mnn":
        return MNNCorrector(**kwargs)
    elif method == "harmony":
        return HarmonyCorrector(**kwargs)
    else:
        raise ValueError(
            f"Unknown batch_method: {method!r}. Choose 'per_batch', 'mnn', or 'harmony'."
        )


# ---------------------------------------------------------------------------
# MNN internals
# ---------------------------------------------------------------------------

def _mnn_correct_pair(
    X_ref: np.ndarray,
    X_query: np.ndarray,
    k: int = 20,
    sigma: float = 1.0,
) -> np.ndarray:
    """Compute correction vectors for X_query to align it with X_ref.

    Matches batchelor::reducedMNN() for a single reference-query pair.

    Parameters
    ----------
    X_ref   : np.ndarray, shape (n_ref, n_pcs)
    X_query : np.ndarray, shape (n_query, n_pcs)
    k       : number of MNN pairs per cell
    sigma   : Gaussian bandwidth for correction smoothing

    Returns
    -------
    corrections : np.ndarray, shape (n_query, n_pcs)
    """
    from sklearn.neighbors import NearestNeighbors

    k_ref = min(k, len(X_ref))
    k_query = min(k, len(X_query))

    # k nearest ref cells for each query cell
    nn_ref = NearestNeighbors(n_neighbors=k_ref, metric="euclidean", algorithm="brute")
    nn_ref.fit(X_ref)
    ref_neighbors_of_query = nn_ref.kneighbors(X_query, return_distance=False)  # (n_query, k)

    # k nearest query cells for each ref cell
    nn_query = NearestNeighbors(n_neighbors=k_query, metric="euclidean", algorithm="brute")
    nn_query.fit(X_query)
    query_neighbors_of_ref = nn_query.kneighbors(X_ref, return_distance=False)  # (n_ref, k)

    # Build: for each ref cell i, which query cells are its k-NN?
    # query_neighbors_of_ref[i] = set of query indices in kNN of ref i
    query_neigh_sets = [set(row) for row in query_neighbors_of_ref]

    # Find MNN pairs: (ref_i, query_j) where j ∈ kNN_ref(query_j) AND i ∈ kNN_query(ref_i)
    mnn_ref_idx = []
    mnn_query_idx = []
    for j, ref_neighbors in enumerate(ref_neighbors_of_query):
        for i in ref_neighbors:
            if j in query_neigh_sets[i]:
                mnn_ref_idx.append(i)
                mnn_query_idx.append(j)

    if len(mnn_ref_idx) == 0:
        # No MNN pairs found — fall back to global mean shift
        correction = X_ref.mean(axis=0) - X_query.mean(axis=0)
        return np.tile(correction, (len(X_query), 1))

    mnn_ref_idx = np.array(mnn_ref_idx)
    mnn_query_idx = np.array(mnn_query_idx)

    # Correction vector for each MNN pair: move query toward ref
    correction_vecs = X_ref[mnn_ref_idx] - X_query[mnn_query_idx]  # (n_pairs, n_pcs)

    # Position of the query cell in each MNN pair (for Gaussian weighting)
    mnn_query_pts = X_query[mnn_query_idx]   # (n_pairs, n_pcs)

    # For each query cell, compute Gaussian-weighted average of correction vectors
    # Using vectorised cdist for efficiency
    # dists[j, p] = distance from query cell j to MNN query point p
    # Chunked to avoid large memory use when n_query * n_pairs is large
    chunk = 256
    corrections = np.zeros_like(X_query)
    for start in range(0, len(X_query), chunk):
        end = min(start + chunk, len(X_query))
        diff = X_query[start:end, np.newaxis, :] - mnn_query_pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)                    # (chunk, n_pairs)
        weights = np.exp(-dists ** 2 / (2.0 * sigma ** 2))     # (chunk, n_pairs)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-12
        corrections[start:end] = weights @ correction_vecs      # (chunk, n_pcs)

    return corrections
