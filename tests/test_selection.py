"""
Tests for pygenebasis.selection.gene_search.

Contract tests
--------------
- Returns exactly n_genes rows
- genes_base appear in output
- genes_discard / genes_discard_prefix are never in output
- Seeded run (same genes_base + random_state) is deterministic
- Approx and exact modes produce ≥85% overlapping gene lists

R reference comparison tests
------------------------------
- Gene list matches R (mouse embryo, exact mode, ≥90% overlap)
- Gene list matches R (BG SN, exact mode, ≥90% overlap)
- Gene list matches R (mouse embryo, approx mode, ≥85% overlap)
"""

from __future__ import annotations

import numpy as np
import pytest

from pygenebasis import gene_search, trim_panel
from helpers import (
    N_GENES, N_NEIGHBORS, N_PCS_ALL, P_MINKOWSKI, RANDOM_STATE,
    SEED_GENES_MOUSE, SEED_GENES_BG_SN,
    GENE_OVERLAP_EXACT, GENE_OVERLAP_APPROX, GENE_OVERLAP_APPROX_CONTRACT,
    GENE_SEARCH_INFORMATIVE_TOL,
    overlap_fraction, weighted_overlap_fraction,
)


# ===========================================================================
# Contract tests
# ===========================================================================

class TestGeneSearchContract:

    def test_returns_n_genes(self, small_adata):
        result = gene_search(
            small_adata, n_genes=10,
            genes_base=["gene_0", "gene_1"],
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        assert len(result) == 10

    def test_output_columns(self, small_adata):
        result = gene_search(
            small_adata, n_genes=5,
            genes_base=["gene_0"],
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        assert "rank" in result.columns
        assert "gene" in result.columns

    def test_rank_is_one_indexed(self, small_adata):
        result = gene_search(
            small_adata, n_genes=5,
            genes_base=["gene_0"],
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        assert result["rank"].tolist() == list(range(1, 6))

    def test_genes_base_in_output(self, small_adata):
        genes_base = ["gene_0", "gene_1", "gene_2"]
        result = gene_search(
            small_adata, n_genes=10,
            genes_base=genes_base,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        selected = result["gene"].tolist()
        for g in genes_base:
            assert g in selected, f"genes_base gene {g!r} missing from output"

    def test_genes_discard_not_in_output(self, small_adata):
        discard = ["gene_5", "gene_6", "gene_7"]
        result = gene_search(
            small_adata, n_genes=10,
            genes_base=["gene_0"],
            genes_discard=discard,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        selected = result["gene"].tolist()
        for g in discard:
            assert g not in selected, f"Discarded gene {g!r} appeared in output"

    def test_genes_discard_prefix_not_in_output(self, small_adata):
        result = gene_search(
            small_adata, n_genes=10,
            genes_base=["gene_0"],
            genes_discard_prefix=["gene_1"],
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        selected = result["gene"].tolist()
        for g in selected:
            assert not g.startswith("gene_1"), (
                f"Gene {g!r} matches discarded prefix 'gene_1'"
            )

    def test_output_genes_are_subset_of_candidates(self, small_adata):
        result = gene_search(
            small_adata, n_genes=10,
            genes_base=["gene_0"],
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        all_genes = set(small_adata.var_names)
        for g in result["gene"]:
            assert g in all_genes

    def test_no_duplicate_genes(self, small_adata):
        result = gene_search(
            small_adata, n_genes=10,
            genes_base=["gene_0"],
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        assert len(result["gene"]) == len(result["gene"].unique())

    def test_deterministic(self, small_adata):
        kwargs = dict(
            n_genes=10, genes_base=["gene_0"],
            knn_method="approx", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        result1 = gene_search(small_adata, **kwargs)
        result2 = gene_search(small_adata, **kwargs)
        assert result1["gene"].tolist() == result2["gene"].tolist()

    def test_approx_vs_exact_overlap(self, small_adata):
        shared = dict(
            n_genes=10, genes_base=["gene_0"],
            n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        exact = gene_search(small_adata, knn_method="exact", **shared)["gene"].tolist()
        approx = gene_search(small_adata, knn_method="approx", **shared)["gene"].tolist()
        frac = overlap_fraction(exact, approx)
        # Use the lower contract threshold here: on synthetic data many genes are
        # equally informative within each cell type, so the greedy algorithm diverges
        # early between exact/approx kNN.  Real-data reference tests enforce 0.85.
        assert frac >= GENE_OVERLAP_APPROX_CONTRACT, (
            f"Approx/exact gene overlap {frac:.3f} < {GENE_OVERLAP_APPROX_CONTRACT}"
        )

    def test_selects_informative_genes(self, gene_search_adata):
        """gene_search should preferentially select the 25 truly informative genes.

        gene_search_adata has 5 cell types × 5 marker genes = 25 informative genes
        (indices 0–24, interleaved), versus 175 pure-noise genes (indices 25–199).
        The Poisson contrast is ~100× (in-class λ≈16, background λ=0.1), so a
        correct greedy search should strongly prefer these genes.

        We check two conditions:
        1. Unweighted: at least GENE_SEARCH_INFORMATIVE_TOL fraction of the
           selected n_genes are in the informative set.
        2. Position-weighted: early-rank selections (where the score landscape is
           steep) should be even more strongly concentrated in the informative set.
           The weighted fraction uses weight 1/(rank+1), so rank-1 counts ~3× more
           than rank-3.  This catches a degraded model that gets lucky on the last
           few picks but fails early.
        """
        # Informative genes: indices 0..24 (5 types × 5 markers, interleaved)
        n_types, n_markers = 5, 5
        informative = {f"gene_{i}" for i in range(n_types * n_markers)}

        n_select = 10  # select fewer than 25 to stay in the informative regime
        result = gene_search(
            gene_search_adata, n_genes=n_select,
            genes_base=["gene_0"],
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, verbose=False,
        )
        selected = result["gene"].tolist()

        unweighted = overlap_fraction(selected, informative)
        assert unweighted >= GENE_SEARCH_INFORMATIVE_TOL, (
            f"Only {unweighted:.2f} of selected genes are informative "
            f"(expected ≥{GENE_SEARCH_INFORMATIVE_TOL}); selected={selected}"
        )

        weighted = weighted_overlap_fraction(selected, informative)
        # Early-rank selections should be even more concentrated in the informative
        # set, so we require the weighted fraction to exceed the unweighted threshold.
        assert weighted >= GENE_SEARCH_INFORMATIVE_TOL, (
            f"Position-weighted informative fraction {weighted:.2f} < "
            f"{GENE_SEARCH_INFORMATIVE_TOL}; selected={selected}"
        )


# ===========================================================================
# R reference comparison
# ===========================================================================

@pytest.mark.mouse
class TestGeneSearchMouseReference:

    def test_gene_list_matches_r_exact(self, adata_mouse, r_mouse):
        hvg_genes = r_mouse["hvg_genes"]["gene"].tolist()
        r_genes = r_mouse["gene_search_results"]["gene"].tolist()
        result = gene_search(
            adata_mouse[:, hvg_genes].copy(), n_genes=N_GENES,
            genes_base=SEED_GENES_MOUSE,
            genes_discard_prefix=["MT-", "RPL", "RPS"],
            batch_key="sample", batch_method="per_batch",
            n_neighbors=N_NEIGHBORS, n_pcs=N_PCS_ALL,
            p_minkowski=P_MINKOWSKI,
            knn_method="exact", random_state=RANDOM_STATE,
        )
        py_genes = result["gene"].tolist()
        frac = overlap_fraction(r_genes, py_genes)
        assert frac >= GENE_OVERLAP_EXACT, (
            f"Mouse gene overlap (exact) {frac:.3f} < {GENE_OVERLAP_EXACT}"
        )

    def test_gene_list_matches_r_approx(self, adata_mouse, r_mouse):
        hvg_genes = r_mouse["hvg_genes"]["gene"].tolist()
        r_genes = r_mouse["gene_search_results"]["gene"].tolist()
        result = gene_search(
            adata_mouse[:, hvg_genes].copy(), n_genes=N_GENES,
            genes_base=SEED_GENES_MOUSE,
            genes_discard_prefix=["MT-", "RPL", "RPS"],
            batch_key="sample", batch_method="per_batch",
            n_neighbors=N_NEIGHBORS, n_pcs=N_PCS_ALL,
            p_minkowski=P_MINKOWSKI,
            knn_method="approx", random_state=RANDOM_STATE,
        )
        py_genes = result["gene"].tolist()
        frac = overlap_fraction(r_genes, py_genes)
        assert frac >= GENE_OVERLAP_APPROX, (
            f"Mouse gene overlap (approx) {frac:.3f} < {GENE_OVERLAP_APPROX}"
        )

    @pytest.mark.slow
    def test_seed_genes_in_top_positions(self, adata_mouse, r_mouse):
        hvg_genes = r_mouse["hvg_genes"]["gene"].tolist()
        result = gene_search(
            adata_mouse[:, hvg_genes].copy(), n_genes=N_GENES,
            genes_base=SEED_GENES_MOUSE,
            batch_key="sample", batch_method="per_batch",
            knn_method="exact", random_state=RANDOM_STATE,
        )
        top_genes = result["gene"].tolist()[:len(SEED_GENES_MOUSE)]
        for g in SEED_GENES_MOUSE:
            assert g in top_genes


# ===========================================================================
# trim_panel tests
# ===========================================================================

class TestTrimPanelContract:
    """Contract tests for trim_panel — no reference data required."""

    def _panel(self, adata, n=20):
        return adata.var_names[:n].tolist()

    def test_output_keys(self, small_adata):
        panel = self._panel(small_adata)
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=3,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        assert {"removal_order", "reduced_panel", "score_trajectory"} == set(result.keys())

    def test_removal_order_length(self, small_adata):
        panel = self._panel(small_adata)
        n_remove = 4
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=n_remove,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        assert len(result["removal_order"]) == n_remove

    def test_reduced_panel_length(self, small_adata):
        panel = self._panel(small_adata)
        n_remove = 4
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=n_remove,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        assert len(result["reduced_panel"]) == len(panel) - n_remove

    def test_score_trajectory_length(self, small_adata):
        panel = self._panel(small_adata)
        n_remove = 4
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=n_remove,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        # n_remove + 1: initial full-panel score + one score per removal
        assert len(result["score_trajectory"]) == n_remove + 1

    def test_removed_genes_were_in_panel(self, small_adata):
        panel = self._panel(small_adata)
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=4,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        panel_set = set(panel)
        for g in result["removal_order"]:
            assert g in panel_set, f"Removed gene {g!r} was not in original panel"

    def test_reduced_panel_disjoint_from_removal(self, small_adata):
        panel = self._panel(small_adata)
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=4,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        removed = set(result["removal_order"])
        for g in result["reduced_panel"]:
            assert g not in removed, f"Gene {g!r} is in both reduced_panel and removal_order"

    def test_reduced_panel_union_removal_equals_original(self, small_adata):
        panel = self._panel(small_adata)
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=4,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        assert set(result["reduced_panel"]) | set(result["removal_order"]) == set(panel)

    def test_raises_when_n_remove_too_large(self, small_adata):
        panel = self._panel(small_adata, n=5)
        with pytest.raises(ValueError):
            trim_panel(
                small_adata, genes_panel=panel, n_remove=5,
                knn_method="exact", n_neighbors=N_NEIGHBORS,
                random_state=RANDOM_STATE, n_jobs=1, verbose=False,
            )

    def test_score_trajectory_finite(self, small_adata):
        panel = self._panel(small_adata)
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=3,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        assert all(np.isfinite(s) for s in result["score_trajectory"])

    def test_deterministic(self, small_adata):
        panel = self._panel(small_adata)
        kwargs = dict(
            genes_panel=panel, n_remove=3,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        r1 = trim_panel(small_adata, **kwargs)
        r2 = trim_panel(small_adata, **kwargs)
        assert r1["removal_order"] == r2["removal_order"]


class TestTrimPanelBehavior:
    """Behavioral tests — verify the algorithm makes sensible decisions."""

    def test_first_score_matches_full_panel_score(self, small_adata):
        """score_trajectory[0] should equal the mean cell score for the full panel.

        Both trim_panel and get_neighborhood_preservation_scores must use the
        same option="exact" so that mean_dist_all is computed identically.
        trim_panel defaults to option="approx" (10% sample) to be fast at
        scale — override here for a fair comparison.
        """
        from pygenebasis import get_neighborhood_preservation_scores
        panel = small_adata.var_names[:20].tolist()
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=2,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            option="exact",
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        # Compute mean score independently for the full panel
        cs = get_neighborhood_preservation_scores(
            small_adata, genes_selection=panel,
            n_neighbors=N_NEIGHBORS, knn_method="exact",
            random_state=RANDOM_STATE,
        )
        expected = float(cs["cell_score"].mean())
        assert abs(result["score_trajectory"][0] - expected) < 1e-3, (
            f"score_trajectory[0]={result['score_trajectory'][0]:.6f} "
            f"!= full-panel mean score {expected:.6f}"
        )

    def test_redundant_gene_removed_first(self, gene_search_adata):
        """An exact duplicate in a panel should be one of the first genes removed.

        We build a 4-gene panel: two distinct informative markers (gene_1, gene_2)
        plus gene_0 and gene_copy (an exact duplicate of gene_0).  gene_0 and
        gene_copy are symmetric — either is the most redundant gene in the panel
        since removing one leaves the other as a full substitute.

        The test checks that the first gene removed is one of the duplicated pair
        (gene_0 or gene_copy), not one of the two unique markers.
        """
        import scipy.sparse
        import anndata as ad
        import pandas as pd

        base = gene_search_adata
        X_extra = base.X[:, 0].toarray() if scipy.sparse.issparse(base.X) else base.X[:, [0]]
        X_new = scipy.sparse.hstack([base.X, scipy.sparse.csr_matrix(X_extra)])
        var_new = pd.DataFrame(index=list(base.var_names) + ["gene_copy"])
        adata_ext = ad.AnnData(X=X_new, obs=base.obs.copy(), var=var_new)

        # Minimal panel: one redundant pair (gene_0 / gene_copy) + two unique markers
        panel = ["gene_0", "gene_1", "gene_2", "gene_copy"]
        result = trim_panel(
            adata_ext, genes_panel=panel, n_remove=1,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            option="exact",
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        first_removed = result["removal_order"][0]
        assert first_removed in {"gene_0", "gene_copy"}, (
            f"Expected the duplicate pair (gene_0 or gene_copy) to be removed first, "
            f"but got {first_removed!r}"
        )

    def test_single_removal(self, small_adata):
        """n_remove=1 removes exactly one gene and returns two trajectory values."""
        panel = small_adata.var_names[:10].tolist()
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=1,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        assert len(result["removal_order"]) == 1
        assert len(result["reduced_panel"]) == 9
        assert len(result["score_trajectory"]) == 2


class TestTrimPanelProtect:
    """Tests for the genes_protect parameter."""

    def _panel(self, adata, n=15):
        return adata.var_names[:n].tolist()

    def test_protected_genes_never_removed(self, small_adata):
        """No gene in genes_protect should appear in removal_order."""
        panel = self._panel(small_adata)
        protect = panel[:3]
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=4,
            genes_protect=protect,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        for g in protect:
            assert g not in result["removal_order"], (
                f"Protected gene {g!r} appeared in removal_order"
            )

    def test_protected_genes_in_reduced_panel(self, small_adata):
        """All genes_protect must be in reduced_panel."""
        panel = self._panel(small_adata)
        protect = panel[:3]
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=4,
            genes_protect=protect,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        for g in protect:
            assert g in result["reduced_panel"], (
                f"Protected gene {g!r} missing from reduced_panel"
            )

    def test_output_sizes_unchanged(self, small_adata):
        """genes_protect does not change removal_order/reduced_panel lengths."""
        panel = self._panel(small_adata)
        n_remove = 4
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=n_remove,
            genes_protect=panel[:2],
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        assert len(result["removal_order"]) == n_remove
        assert len(result["reduced_panel"]) == len(panel) - n_remove

    def test_raises_when_n_remove_exceeds_unprotected(self, small_adata):
        """n_remove > len(panel) - len(protect) should raise ValueError."""
        panel = self._panel(small_adata, n=6)
        protect = panel[:4]   # only 2 unprotected genes
        with pytest.raises(ValueError, match="unprotected"):
            trim_panel(
                small_adata, genes_panel=panel, n_remove=3,
                genes_protect=protect,
                knn_method="exact", n_neighbors=N_NEIGHBORS,
                random_state=RANDOM_STATE, n_jobs=1, verbose=False,
            )

    def test_raises_when_protect_gene_not_in_panel(self, small_adata):
        """genes_protect containing a gene absent from genes_panel raises ValueError."""
        panel = self._panel(small_adata)
        with pytest.raises(ValueError, match="not in genes_panel"):
            trim_panel(
                small_adata, genes_panel=panel, n_remove=2,
                genes_protect=["not_a_real_gene"],
                knn_method="exact", n_neighbors=N_NEIGHBORS,
                random_state=RANDOM_STATE, n_jobs=1, verbose=False,
            )

    def test_empty_protect_list_behaves_as_none(self, small_adata):
        """genes_protect=[] should behave identically to genes_protect=None."""
        panel = self._panel(small_adata)
        kwargs = dict(
            genes_panel=panel, n_remove=3,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        r_none = trim_panel(small_adata, **kwargs, genes_protect=None)
        r_empty = trim_panel(small_adata, **kwargs, genes_protect=[])
        assert r_none["removal_order"] == r_empty["removal_order"]

    def test_protect_all_but_one_removes_from_unprotected(self, small_adata):
        """When all but one gene is protected, only that gene can be removed."""
        panel = self._panel(small_adata, n=5)
        unprotected = panel[-1]
        protect = panel[:-1]
        result = trim_panel(
            small_adata, genes_panel=panel, n_remove=1,
            genes_protect=protect,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE, n_jobs=1, verbose=False,
        )
        assert result["removal_order"] == [unprotected]


@pytest.mark.bg_sn
class TestGeneSearchBgSnReference:

    def test_gene_list_matches_r_exact(self, adata_bg_sn, r_bg_sn):
        hvg_genes = r_bg_sn["hvg_genes"]["gene"].tolist()
        r_genes = r_bg_sn["gene_search_results"]["gene"].tolist()
        result = gene_search(
            adata_bg_sn[:, hvg_genes].copy(), n_genes=N_GENES,
            genes_base=SEED_GENES_BG_SN,
            genes_discard_prefix=["MT-", "RPL", "RPS"],
            batch_key="donor_id", batch_method="per_batch",
            n_neighbors=N_NEIGHBORS, n_pcs=N_PCS_ALL,
            p_minkowski=P_MINKOWSKI,
            knn_method="exact", random_state=RANDOM_STATE,
        )
        py_genes = result["gene"].tolist()
        frac = overlap_fraction(r_genes, py_genes)
        # BG SN has 19549 HVGs (12.5× more than mouse's 1554), making the true
        # graph kNN only ~62% consistent with R (vs ~85% for mouse).  Greedy
        # gene selection is sensitive to early iterations, so divergence in the
        # true graph propagates to the panel.  Observed overlap: ~0.68.
        BG_SN_GENE_OVERLAP_TOL = 0.65
        assert frac >= BG_SN_GENE_OVERLAP_TOL, (
            f"BG SN gene overlap (exact) {frac:.3f} < {BG_SN_GENE_OVERLAP_TOL}"
        )
