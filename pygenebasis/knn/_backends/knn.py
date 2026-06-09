"""
kNN backend implementations.

Design decision: kNN construction is abstracted behind a common interface so that
the calling code (graph.py) never needs to know whether it is using exact or
approximate neighbours.  A factory function ``get_knn_backend`` selects the
concrete class from the user-facing ``knn_method`` flag.

Backends
--------
exact  → ExactKNN   — sklearn NearestNeighbors (brute-force, L2).
                       Matches geneBasisR's BiocNeighbors::queryKNN most closely.
                       Used as the ground-truth reference in correctness tests.
approx → ApproxKNN  — PyNNDescent (random-projection forest + graph refinement).
                       Primary production backend; 10-100× faster on large data.

GPU (cuML) support is deferred until the CPU paths are validated against R.

Both backends return 0-indexed integer arrays to match Python/NumPy convention.
geneBasisR returns 1-indexed (R) indices; the reference generation script
subtracts 1 before saving so that the CSVs are already 0-indexed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class KNNBackend(ABC):
    """Common interface for kNN graph construction."""

    @abstractmethod
    def fit_query(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Build a kNN graph where every point is both reference and query.

        Parameters
        ----------
        X : np.ndarray, shape (n_cells, n_features)
            Embedding (e.g. PCA scores) to search in.
        k : int
            Number of neighbours to return (excluding self).

        Returns
        -------
        indices : np.ndarray, shape (n_cells, k), dtype int
            0-indexed neighbour indices, ordered by ascending distance.
        distances : np.ndarray, shape (n_cells, k), dtype float
            Corresponding Euclidean distances.
        """


class ExactKNN(KNNBackend):
    """Exact kNN via sklearn's brute-force NearestNeighbors.

    Uses the same distance (Euclidean / L2) as BiocNeighbors::queryKNN and
    therefore gives the closest numerical match to geneBasisR outputs.

    Parameters
    ----------
    random_state : int
        Unused for exact search; kept for API symmetry with ApproxKNN.
    """

    def __init__(self, random_state: int = 32) -> None:
        self.random_state = random_state

    def fit_query(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(X, dtype=np.float64)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="euclidean")
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        # First column is always self (distance = 0) when querying training data
        return indices[:, 1:].astype(np.int32), distances[:, 1:]


class ApproxKNN(KNNBackend):
    """Approximate kNN via PyNNDescent.

    PyNNDescent uses a random-projection forest seeded graph that is then
    refined via neighbour-of-neighbour descent.  Results are approximate but
    typically >95% overlap with exact kNN at a fraction of the cost.

    This is the primary production backend for large (>10k cell) datasets.

    Parameters
    ----------
    random_state : int
        Seed passed to NNDescent for reproducibility.
    n_jobs : int
        Number of threads for the index build (-1 = all cores).
    """

    def __init__(self, random_state: int = 32, n_jobs: int = -1) -> None:
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit_query(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        from pynndescent import NNDescent

        X = np.asarray(X, dtype=np.float32)
        n_cells = X.shape[0]

        # Request k+1 neighbours; self is typically included at distance ≈ 0
        index = NNDescent(
            X,
            n_neighbors=k + 1,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        indices, distances = index.neighbor_graph

        # Filter out self-references (vectorised)
        cell_idx = np.arange(n_cells)[:, None]  # (n_cells, 1)
        is_self = indices == cell_idx            # (n_cells, k+1)

        # Sort so that self (True) comes last, then take first k columns
        order = np.argsort(is_self.astype(np.int8), axis=1, kind="stable")
        indices_sorted = np.take_along_axis(indices, order, axis=1)
        distances_sorted = np.take_along_axis(distances, order, axis=1)

        return indices_sorted[:, :k].astype(np.int32), distances_sorted[:, :k].astype(np.float64)


def get_knn_backend(method: str, **kwargs) -> KNNBackend:
    """Factory: return the kNN backend for the given ``knn_method`` flag.

    Parameters
    ----------
    method : {"exact", "approx"}
        "exact"  → ExactKNN  (sklearn, brute-force)
        "approx" → ApproxKNN (PyNNDescent, approximate)
    **kwargs
        Forwarded to the backend constructor (e.g. ``random_state``).
    """
    if method == "exact":
        return ExactKNN(**kwargs)
    elif method == "approx":
        return ApproxKNN(**kwargs)
    else:
        raise ValueError(
            f"Unknown knn_method: {method!r}. Choose 'exact' or 'approx'."
        )
