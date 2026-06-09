"""
Minkowski reconstruction distance scoring.

This module implements the core per-gene scoring function used during gene
selection.  For a given kNN graph and a set of candidate genes, it measures
how well each gene's expression can be reconstructed by averaging over the
k nearest neighbours in the graph.

A gene that scores *high* is poorly reconstructed → it carries information
not yet captured by the current panel → it is a good candidate to add next.

Key speedup over geneBasisR
---------------------------
geneBasisR loops over candidate genes one at a time in R.  Here the entire
candidate gene set is scored in a **single vectorised NumPy broadcast**:

    neighbor_expr  = logcounts[knn_indices]          # (n_cells, k, n_genes)
    neighbor_avg   = neighbor_expr.mean(axis=1)      # (n_cells, n_genes)
    diff           = logcounts - neighbor_avg         # (n_cells, n_genes)
    dist           = (|diff|^p).sum(axis=0)^(1/p)   # (n_genes,)

This replaces an O(n_genes) sequential loop with a single array operation,
giving a speedup proportional to the number of candidate genes (~thousands).

Memory note
-----------
The intermediate ``neighbor_expr`` tensor has shape (n_cells, k, n_genes).
For large datasets this may not fit in RAM; in that case ``calc_minkowski_distances``
can chunk over gene batches using joblib.  This is controlled by ``chunk_size``.
"""

from __future__ import annotations

import numpy as np


def calc_minkowski_distances(
    logcounts: np.ndarray,
    knn_indices: np.ndarray,
    *,
    p: float = 3.0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Score candidate genes by Minkowski reconstruction error.

    For each candidate gene g, measures how well its expression is
    reconstructed by averaging the k nearest neighbours' expression in the
    current graph.  Higher distance = more unique information = better
    candidate to add to the panel.

    Matches geneBasisR's ``calc_Minkowski_distances`` with p.minkowski=3.

    Parameters
    ----------
    logcounts : np.ndarray, shape (n_cells, n_genes)
        Log-normalised expression matrix.  Rows are cells, columns are genes.
    knn_indices : np.ndarray, shape (n_cells, k), dtype int
        0-indexed neighbour indices from ``build_knn_graph``.
    p : float
        Minkowski order. Default 3 matches geneBasisR.
    chunk_size : int or None
        If set, genes are processed in chunks of this size to limit peak
        memory usage.  None processes all genes at once (fastest).

    Returns
    -------
    distances : np.ndarray, shape (n_genes,)
        Per-gene Minkowski reconstruction distance. Non-negative.
    """
    logcounts = np.asarray(logcounts, dtype=np.float64)
    knn_indices = np.asarray(knn_indices, dtype=np.int32)
    n_genes = logcounts.shape[1]

    if chunk_size is None:
        return _minkowski_chunk(logcounts, knn_indices, slice(None), p)

    distances = np.empty(n_genes, dtype=np.float64)
    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        distances[start:end] = _minkowski_chunk(
            logcounts, knn_indices, slice(start, end), p
        )
    return distances


def _minkowski_chunk(
    logcounts: np.ndarray,
    knn_indices: np.ndarray,
    gene_slice: slice,
    p: float,
) -> np.ndarray:
    """Vectorised Minkowski distance for a slice of genes.

    Internal helper used by ``calc_minkowski_distances`` when chunking.

    Parameters
    ----------
    logcounts : np.ndarray, shape (n_cells, n_genes)
    knn_indices : np.ndarray, shape (n_cells, k)
    gene_slice : slice
        Column slice into logcounts to process.
    p : float

    Returns
    -------
    dist : np.ndarray, shape (slice_size,)
    """
    lc = logcounts[:, gene_slice]              # (n_cells, slice_size)
    neighbor_expr = lc[knn_indices]            # (n_cells, k, slice_size)
    neighbor_avg = neighbor_expr.mean(axis=1)  # (n_cells, slice_size)
    diff = lc - neighbor_avg                   # (n_cells, slice_size)
    dist = (np.abs(diff) ** p).sum(axis=0) ** (1.0 / p)  # (slice_size,)
    return dist
