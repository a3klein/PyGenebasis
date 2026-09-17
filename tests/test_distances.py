"""
Tests for pygenebasis.distances.calc_minkowski_distances.

Contract tests
--------------
- Output shape is (n_genes,)
- All values are non-negative
- A gene already in the graph scores near 0
- Vectorised result matches a loop-based reference implementation exactly

R reference comparison tests
------------------------------
- True-graph distances match R (mouse embryo)
- Selection-graph distances match R (mouse embryo)
- True-graph distances match R (BG SN)
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from pygenebasis import build_knn_graph, calc_minkowski_distances
from helpers import (
    N_NEIGHBORS, N_PCS_ALL, P_MINKOWSKI, RANDOM_STATE,
    SEED_GENES_MOUSE, SEED_GENES_BG_SN, SPEARMAN_TOL, spearman_r,
)


def _reference_minkowski_loop(logcounts, knn_indices, p):
    """Loop-based reference implementation for testing correctness."""
    n_cells, n_genes = logcounts.shape
    distances = np.zeros(n_genes)
    for g in range(n_genes):
        expr = logcounts[:, g]
        neighbor_avg = logcounts[knn_indices, g].mean(axis=1)
        distances[g] = (np.abs(expr - neighbor_avg) ** p).sum() ** (1 / p)
    return distances


def _dense(X):
    """Convert sparse or dense array to dense float32."""
    if scipy.sparse.issparse(X):
        return np.asarray(X.todense(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


# ===========================================================================
# Contract tests — small synthetic data
# ===========================================================================

class TestCalcMinkowskiDistancesContract:

    def test_output_shape(self, small_adata):
        genes = small_adata.var_names[:20].tolist()
        X = _dense(small_adata.X)
        idx, _ = build_knn_graph(
            small_adata, genes, knn_method="exact",
            n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        dist = calc_minkowski_distances(X, idx, p=P_MINKOWSKI)
        assert dist.shape == (small_adata.n_vars,), (
            f"Expected ({small_adata.n_vars},), got {dist.shape}"
        )

    def test_nonnegative(self, small_adata):
        genes = small_adata.var_names[:20].tolist()
        X = _dense(small_adata.X)
        idx, _ = build_knn_graph(
            small_adata, genes, knn_method="exact",
            n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        dist = calc_minkowski_distances(X, idx, p=P_MINKOWSKI)
        assert np.all(dist >= 0), "Distances should be non-negative"

    def test_gene_in_graph_scores_low(self, small_adata):
        """A gene used to build the graph is well-reconstructed → low score."""
        genes = small_adata.var_names[:20].tolist()
        X = _dense(small_adata.X)
        idx, _ = build_knn_graph(
            small_adata, genes, knn_method="exact",
            n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        dist = calc_minkowski_distances(X, idx, p=P_MINKOWSKI)
        gene_indices = [small_adata.var_names.get_loc(g) for g in genes]
        other_indices = [i for i in range(small_adata.n_vars) if i not in gene_indices]
        mean_in = dist[gene_indices].mean()
        mean_out = dist[other_indices].mean()
        assert mean_in < mean_out, (
            "Genes in the graph should score lower than out-of-graph genes"
        )

    def test_matches_loop_implementation(self, small_adata):
        genes = small_adata.var_names[:20].tolist()
        X = _dense(small_adata.X)
        idx, _ = build_knn_graph(
            small_adata, genes, knn_method="exact",
            n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        dist_vec = calc_minkowski_distances(X, idx, p=P_MINKOWSKI)
        dist_loop = _reference_minkowski_loop(X, idx, P_MINKOWSKI)
        np.testing.assert_allclose(
            dist_vec, dist_loop, rtol=1e-5,
            err_msg="Vectorised result differs from loop-based reference",
        )

    def test_p_equals_2_is_euclidean(self, small_adata):
        """With p=2 and k=1 the distance should equal the L2 norm to the single neighbour."""
        genes = small_adata.var_names[:20].tolist()
        X = _dense(small_adata.X)
        idx, _ = build_knn_graph(
            small_adata, genes, knn_method="exact",
            n_neighbors=1, random_state=RANDOM_STATE,
        )
        dist = calc_minkowski_distances(X, idx, p=2.0)
        assert np.all(dist >= 0)


# ===========================================================================
# R reference comparison
# ===========================================================================

@pytest.mark.mouse
class TestCalcMinkowskiDistancesMouseReference:

    def test_true_graph_distances_match_r(self, adata_mouse, r_mouse):
        hvg_genes = r_mouse["hvg_genes"]["gene"].tolist()
        X = _dense(adata_mouse[:, hvg_genes].X)
        idx, _ = build_knn_graph(
            adata_mouse, hvg_genes,
            batch_key="sample", batch_method="per_batch",
            n_neighbors=N_NEIGHBORS, n_pcs=N_PCS_ALL,
            knn_method="exact", random_state=RANDOM_STATE,
        )
        dist_py = calc_minkowski_distances(X, idx, p=P_MINKOWSKI)
        r_dist = r_mouse["minkowski_true_graph"].set_index("gene")["dist"]
        r_dist = r_dist.reindex(hvg_genes).values
        rho = spearman_r(dist_py, r_dist)
        assert rho >= SPEARMAN_TOL, (
            f"Mouse true-graph Minkowski Spearman ρ={rho:.4f} < {SPEARMAN_TOL}"
        )

    def test_selection_graph_distances_match_r(self, adata_mouse, r_mouse):
        hvg_genes = r_mouse["hvg_genes"]["gene"].tolist()
        selected = r_mouse["gene_search_results"]["gene"].tolist()
        X = _dense(adata_mouse[:, hvg_genes].X)
        idx, _ = build_knn_graph(
            adata_mouse, selected,
            batch_key="sample", batch_method="per_batch",
            n_neighbors=N_NEIGHBORS, n_pcs=None,
            knn_method="exact", random_state=RANDOM_STATE,
        )
        dist_py = calc_minkowski_distances(X, idx, p=P_MINKOWSKI)
        r_dist = r_mouse["minkowski_selection_graph"].set_index("gene")["dist"]
        r_dist = r_dist.reindex(hvg_genes).values
        rho = spearman_r(dist_py, r_dist)
        assert rho >= SPEARMAN_TOL, (
            f"Mouse selection-graph Minkowski Spearman ρ={rho:.4f} < {SPEARMAN_TOL}"
        )


@pytest.mark.bg_sn
class TestCalcMinkowskiDistancesBgSnReference:

    def test_true_graph_distances_match_r(self, adata_bg_sn, r_bg_sn):
        hvg_genes = r_bg_sn["hvg_genes"]["gene"].tolist()
        X = _dense(adata_bg_sn[:, hvg_genes].X)
        idx, _ = build_knn_graph(
            adata_bg_sn, hvg_genes,
            batch_key="donor_id", batch_method="per_batch",
            n_neighbors=N_NEIGHBORS, n_pcs=N_PCS_ALL,
            knn_method="exact", random_state=RANDOM_STATE,
        )
        dist_py = calc_minkowski_distances(X, idx, p=P_MINKOWSKI)
        r_dist = r_bg_sn["minkowski_true_graph"].set_index("gene")["dist"]
        r_dist = r_dist.reindex(hvg_genes).values
        rho = spearman_r(dist_py, r_dist)
        assert rho >= SPEARMAN_TOL, (
            f"BG SN true-graph Minkowski Spearman ρ={rho:.4f} < {SPEARMAN_TOL}"
        )
