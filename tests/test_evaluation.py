"""
Tests for pygenebasis.evaluation:
  - get_neighborhood_preservation_scores
  - get_gene_prediction_scores
  - evaluate_library

Testing strategy: all reference tests pass R's HVG gene list as genes_all and
R's selected gene list as genes_selection.  This isolates evaluation algorithm
differences from HVG selection differences (scanpy vs scran give ~30% overlap).
"""

from __future__ import annotations

import numpy as np
import pytest

import anndata as ad

from pygenebasis import (
    get_neighborhood_preservation_scores,
    get_gene_prediction_scores,
    evaluate_library,
    get_panel_celltype_accuracy,
    get_celltype_mapping,
)
from pygenebasis.panel._evaluation import get_neighs_all_stat, _median_neighbour_dist
from pygenebasis.knn._graph import build_knn_graph
from helpers import (
    N_NEIGHBORS, N_PCS_ALL, RANDOM_STATE,
    SEED_GENES_MOUSE, SEED_GENES_BG_SN,
    SPEARMAN_TOL, MAE_TOL, spearman_r,
)


# ===========================================================================
# get_neighborhood_preservation_scores
# ===========================================================================

class TestNeighborhoodPreservationContract:

    def test_output_shape(self, small_adata):
        genes = small_adata.var_names[:10].tolist()
        result = get_neighborhood_preservation_scores(
            small_adata, genes_selection=genes,
            n_neighbors=N_NEIGHBORS, knn_method="exact",
            random_state=RANDOM_STATE,
        )
        assert len(result) == small_adata.n_obs

    def test_output_columns(self, small_adata):
        genes = small_adata.var_names[:10].tolist()
        result = get_neighborhood_preservation_scores(
            small_adata, genes_selection=genes,
            n_neighbors=N_NEIGHBORS, knn_method="exact",
            random_state=RANDOM_STATE,
        )
        assert "cell" in result.columns
        assert "cell_score" in result.columns

    def test_scores_bounded(self, small_adata):
        """Scores should generally be in [0, 1] for well-behaved data."""
        genes = small_adata.var_names[:10].tolist()
        result = get_neighborhood_preservation_scores(
            small_adata, genes_selection=genes,
            n_neighbors=N_NEIGHBORS, knn_method="exact",
            random_state=RANDOM_STATE,
        )
        scores = result["cell_score"].values
        assert np.nanpercentile(scores, 1) >= -0.1
        assert np.nanpercentile(scores, 99) <= 1.1

    def test_all_genes_gives_high_score(self, small_adata):
        """Using all genes as selection should give scores close to 1."""
        all_genes = small_adata.var_names.tolist()
        result = get_neighborhood_preservation_scores(
            small_adata, genes_selection=all_genes,
            genes_all=all_genes,
            n_neighbors=N_NEIGHBORS, knn_method="exact",
            random_state=RANDOM_STATE,
        )
        assert result["cell_score"].mean() > 0.8

    def test_fewer_genes_lower_score(self, small_adata):
        """A smaller panel should give a lower mean score than a larger one."""
        all_genes = small_adata.var_names.tolist()
        large = get_neighborhood_preservation_scores(
            small_adata, genes_selection=all_genes[:50],
            n_neighbors=N_NEIGHBORS, knn_method="exact", random_state=RANDOM_STATE,
        )["cell_score"].mean()
        small = get_neighborhood_preservation_scores(
            small_adata, genes_selection=all_genes[:5],
            n_neighbors=N_NEIGHBORS, knn_method="exact", random_state=RANDOM_STATE,
        )["cell_score"].mean()
        assert large > small


@pytest.mark.mouse
class TestNeighborhoodPreservationMouseReference:

    def test_dist_sel_matches_r(self, adata_mouse, r_mouse):
        """dist_sel (median HVG-PCA distance to selection-graph neighbours) matches R.

        The cell score formula (mean_dist - dist_sel) / (mean_dist - dist_true)
        involves the difference of two nearly equal numbers (~8.2 vs ~8.2), so
        a ~1-2% PCA coordinate difference (sklearn vs irlba) is enough to flip
        the sign for most cells.  dist_sel and dist_true themselves rank very
        well against R (ρ ≈ 0.99) because the PCA shift is systematic and
        preserves relative structure.

        We compare dist_sel directly since the reference exports it in
        cell_score_intermediates.csv.  This is a stronger test than internal
        consistency: it verifies that Python builds the same effective
        neighbourhood structure as R for the selected gene panel.
        """
        selected  = r_mouse["gene_search_results"]["gene"].tolist()
        hvg_genes = r_mouse["hvg_genes"]["gene"].tolist()
        r_inter   = r_mouse["cell_score_intermediates"].set_index("cell")

        neighs_all_stat = get_neighs_all_stat(
            adata_mouse, genes_all=hvg_genes,
            batch_key="sample", n_neighbors=N_NEIGHBORS, n_pcs_all=N_PCS_ALL,
            knn_method="exact", batch_method="per_batch",
            option="exact", random_state=RANDOM_STATE,
        )
        sel_indices, _ = build_knn_graph(
            adata_mouse, selected,
            batch_key="sample", knn_method="exact", batch_method="per_batch",
            n_neighbors=N_NEIGHBORS, n_pcs=None, random_state=RANDOM_STATE,
        )
        dist_sel = _median_neighbour_dist(neighs_all_stat["embedding"], sel_indices)

        cells = adata_mouse.obs_names.tolist()
        py_series = np.array([
            dist_sel[i] for i, c in enumerate(cells) if c in r_inter.index
        ])
        r_series = r_inter.loc[
            [c for c in cells if c in r_inter.index], "dist_sel"
        ].values

        rho = spearman_r(py_series, r_series)
        assert rho >= SPEARMAN_TOL, (
            f"Mouse dist_sel Spearman ρ={rho:.4f} < {SPEARMAN_TOL}"
        )


@pytest.mark.bg_sn
class TestNeighborhoodPreservationBgSnReference:

    def test_cell_scores_internal_consistency(self, adata_bg_sn, r_bg_sn):
        """Cell scores are internally consistent on the BG SN dataset.

        The BG SN dataset has 19549 HVGs (vs ~1554 for mouse).  When using
        all HVGs as the selection panel, n_pcs_selection must be set to the
        same value as n_pcs_all (50) so that the selection graph is built in
        the same PCA space as the true graph.  Without PCA, kNN in 19549-D
        raw gene space gives very different neighbours than the PCA-based true
        graph, causing all-HVG to score *lower* than the 50-gene panel.
        """
        selected = r_bg_sn["gene_search_results"]["gene"].tolist()
        hvg_genes = r_bg_sn["hvg_genes"]["gene"].tolist()

        result = get_neighborhood_preservation_scores(
            adata_bg_sn, genes_selection=selected,
            genes_all=hvg_genes,
            batch_key="donor_id", batch_method="per_batch",
            n_neighbors=N_NEIGHBORS, n_pcs_all=N_PCS_ALL,
            knn_method="exact", random_state=RANDOM_STATE,
        )
        scores = result["cell_score"]
        assert scores.notna().all(), "NaN cell scores found"

        # Use same PCA for selection as for the true graph
        result_all = get_neighborhood_preservation_scores(
            adata_bg_sn, genes_selection=hvg_genes,
            genes_all=hvg_genes,
            batch_key="donor_id", batch_method="per_batch",
            n_neighbors=N_NEIGHBORS, n_pcs_all=N_PCS_ALL,
            n_pcs_selection=N_PCS_ALL,
            knn_method="exact", random_state=RANDOM_STATE,
        )
        assert result_all["cell_score"].mean() > scores.mean(), (
            "All-HVG panel (with PCA) should score higher than 50-gene panel"
        )


# ===========================================================================
# get_gene_prediction_scores
# ===========================================================================

class TestGenePredictionScoresContract:

    def test_output_columns(self, small_adata):
        genes = small_adata.var_names[:10].tolist()
        result = get_gene_prediction_scores(
            small_adata, genes_selection=genes,
            n_neighbors=N_NEIGHBORS, knn_method="exact",
            random_state=RANDOM_STATE,
        )
        assert {"gene", "corr", "corr_all", "gene_score"}.issubset(result.columns)

    def test_output_length(self, small_adata):
        genes = small_adata.var_names[:10].tolist()
        result = get_gene_prediction_scores(
            small_adata, genes_selection=genes,
            n_neighbors=N_NEIGHBORS, knn_method="exact",
            random_state=RANDOM_STATE,
        )
        assert len(result) == small_adata.n_vars

    def test_selected_genes_score_near_one(self, small_adata):
        genes = small_adata.var_names[:20].tolist()
        result = get_gene_prediction_scores(
            small_adata, genes_selection=genes,
            n_neighbors=N_NEIGHBORS, knn_method="exact",
            random_state=RANDOM_STATE,
        )
        sel_scores = result[result["gene"].isin(genes)]["gene_score"]
        assert sel_scores.mean() > 0.7


@pytest.mark.mouse
class TestGenePredictionScoresMouseReference:

    def test_gene_scores_match_r(self, adata_mouse, r_mouse):
        selected = r_mouse["gene_search_results"]["gene"].tolist()
        hvg_genes = r_mouse["hvg_genes"]["gene"].tolist()
        r_scores = r_mouse["gene_scores"].set_index("gene")["gene_score"]
        result = get_gene_prediction_scores(
            adata_mouse, genes_selection=selected,
            genes_all=hvg_genes,
            batch_key="sample", batch_method="per_batch",
            n_neighbors=N_NEIGHBORS, n_pcs_all=N_PCS_ALL,
            knn_method="exact", random_state=RANDOM_STATE,
        )
        py_scores = result.set_index("gene")["gene_score"].reindex(r_scores.index)
        rho = spearman_r(py_scores.values, r_scores.values)
        assert rho >= SPEARMAN_TOL, f"Mouse gene score Spearman ρ={rho:.4f}"


@pytest.mark.bg_sn
class TestGenePredictionScoresBgSnReference:

    def test_gene_scores_match_r(self, adata_bg_sn, r_bg_sn):
        selected = r_bg_sn["gene_search_results"]["gene"].tolist()
        hvg_genes = r_bg_sn["hvg_genes"]["gene"].tolist()
        r_scores = r_bg_sn["gene_scores"].set_index("gene")["gene_score"]
        result = get_gene_prediction_scores(
            adata_bg_sn, genes_selection=selected,
            genes_all=hvg_genes,
            batch_key="donor_id", batch_method="per_batch",
            n_neighbors=N_NEIGHBORS, n_pcs_all=N_PCS_ALL,
            knn_method="exact", random_state=RANDOM_STATE,
        )
        py_scores = result.set_index("gene")["gene_score"].reindex(r_scores.index)
        rho = spearman_r(py_scores.values, r_scores.values)
        # BG SN kNN overlap with R is ~0.62 (vs 0.85 for mouse) due to 19549
        # HVGs making PCA less stable.  Gene prediction scores are downstream
        # of kNN quality, so the achievable Spearman is lower.  Observed: 0.94.
        BG_SN_GENE_SCORE_TOL = 0.90
        assert rho >= BG_SN_GENE_SCORE_TOL, f"BG SN gene score Spearman ρ={rho:.4f} < {BG_SN_GENE_SCORE_TOL}"


# ===========================================================================
# evaluate_library
# ===========================================================================

class TestEvaluateLibraryContract:

    def test_single_mode_returns_all_keys(self, small_adata):
        genes = small_adata.var_names[:10].tolist()
        result = evaluate_library(
            small_adata, genes_selection=genes,
            celltype_key="celltype",
            library_size_type="single",
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        assert "cell_score_stat" in result
        assert "gene_score_stat" in result
        assert "celltype_stat" in result

    def test_single_mode_n_genes_column(self, small_adata):
        genes = small_adata.var_names[:10].tolist()
        result = evaluate_library(
            small_adata, genes_selection=genes,
            celltype_key="celltype",
            library_size_type="single",
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        unique_sizes = result["cell_score_stat"]["n_genes"].unique()
        assert list(unique_sizes) == [len(genes)]

    def test_series_mode_multiple_sizes(self, small_adata):
        genes = small_adata.var_names[:20].tolist()
        result = evaluate_library(
            small_adata, genes_selection=genes,
            celltype_key="celltype",
            library_size_type="series",
            n_genes_step=5,
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        unique_sizes = sorted(result["cell_score_stat"]["n_genes"].unique())
        assert len(unique_sizes) > 1

    def test_consistent_with_individual_functions(self, small_adata):
        """evaluate_library results should match calling functions directly."""
        genes = small_adata.var_names[:10].tolist()
        lib_result = evaluate_library(
            small_adata, genes_selection=genes,
            celltype_key="celltype",
            library_size_type="single",
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            random_state=RANDOM_STATE,
        )
        direct_result = get_neighborhood_preservation_scores(
            small_adata, genes_selection=genes,
            n_neighbors=N_NEIGHBORS, knn_method="exact",
            random_state=RANDOM_STATE,
        )
        lib_scores = lib_result["cell_score_stat"]["cell_score"].values
        direct_scores = direct_result["cell_score"].values
        np.testing.assert_allclose(
            sorted(lib_scores), sorted(direct_scores), rtol=1e-5
        )


# ===========================================================================
# get_panel_celltype_accuracy
# ===========================================================================

class TestGetPanelCelltypeAccuracy:
    """Tests for get_panel_celltype_accuracy.

    Uses the hierarchical_adata fixture (from conftest.py):
        class_label → celltype
        ClassA → TypeA, TypeB
        ClassB → TypeC, TypeD
        ClassC → TypeE  (single-child)
    100 genes total; first 50 are informative markers.
    """

    LEVEL_KEYS = ["class_label", "celltype"]

    def _genes(self, adata):
        return adata.var_names[:50].tolist()

    # --- Contract tests ---

    def test_output_columns(self, hierarchical_adata):
        result = get_panel_celltype_accuracy(
            hierarchical_adata, self._genes(hierarchical_adata), self.LEVEL_KEYS,
            constrained_method="none",
            knn_method="exact", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        assert {"level", "celltype", "constrained_accuracy", "compounded_accuracy"}.issubset(
            result.columns
        )

    def test_output_no_duplicates(self, hierarchical_adata):
        result = get_panel_celltype_accuracy(
            hierarchical_adata, self._genes(hierarchical_adata), self.LEVEL_KEYS,
            constrained_method="none",
            knn_method="exact", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        assert not result.duplicated(subset=["level", "celltype"]).any()

    def test_output_row_count(self, hierarchical_adata):
        """One row per (level, celltype): 3 classes + 5 cell types = 8 rows."""
        result = get_panel_celltype_accuracy(
            hierarchical_adata, self._genes(hierarchical_adata), self.LEVEL_KEYS,
            constrained_method="none",
            knn_method="exact", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        n_classes   = hierarchical_adata.obs["class_label"].nunique()
        n_celltypes = hierarchical_adata.obs["celltype"].nunique()
        assert len(result) == n_classes + n_celltypes

    def test_accuracy_values_in_range(self, hierarchical_adata):
        result = get_panel_celltype_accuracy(
            hierarchical_adata, self._genes(hierarchical_adata), self.LEVEL_KEYS,
            constrained_method="within_class",
            knn_method="exact", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        assert (result["constrained_accuracy"] >= 0).all()
        assert (result["constrained_accuracy"] <= 1).all()
        assert (result["compounded_accuracy"] >= 0).all()
        assert (result["compounded_accuracy"] <= 1).all()

    def test_coarsest_level_constrained_equals_compounded(self, hierarchical_adata):
        result = get_panel_celltype_accuracy(
            hierarchical_adata, self._genes(hierarchical_adata), self.LEVEL_KEYS,
            constrained_method="within_class",
            knn_method="exact", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        coarsest = result[result["level"] == self.LEVEL_KEYS[0]]
        np.testing.assert_allclose(
            coarsest["constrained_accuracy"].values,
            coarsest["compounded_accuracy"].values,
        )

    def test_compounded_le_constrained_at_finer_levels(self, hierarchical_adata):
        """compounded ≤ constrained at every level below the coarsest."""
        result = get_panel_celltype_accuracy(
            hierarchical_adata, self._genes(hierarchical_adata), self.LEVEL_KEYS,
            constrained_method="within_class",
            knn_method="exact", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        finer = result[result["level"] != self.LEVEL_KEYS[0]]
        assert (finer["compounded_accuracy"].values <= finer["constrained_accuracy"].values + 1e-9).all()

    # --- Behavioural tests ---

    def test_none_method_matches_celltype_mapping(self, hierarchical_adata):
        """'none' constrained_accuracy at each level must match get_celltype_mapping."""
        genes = self._genes(hierarchical_adata)
        result = get_panel_celltype_accuracy(
            hierarchical_adata, genes, self.LEVEL_KEYS,
            constrained_method="none",
            knn_method="exact", n_neighbors=N_NEIGHBORS,
            batch_method="per_batch", random_state=RANDOM_STATE,
        )
        for level in self.LEVEL_KEYS:
            direct = get_celltype_mapping(
                hierarchical_adata, genes,
                celltype_key=level,
                knn_method="exact", n_neighbors=N_NEIGHBORS,
                batch_method="per_batch", n_pcs_selection=None,
                return_stat=True, random_state=RANDOM_STATE,
            )["stat"].set_index("celltype")["frac_correctly_mapped"]

            level_result = (
                result[result["level"] == level]
                .set_index("celltype")["constrained_accuracy"]
            )
            common = direct.index.intersection(level_result.index)
            np.testing.assert_allclose(
                direct.loc[common].sort_index().values,
                level_result.loc[common].sort_index().values,
                rtol=1e-5,
                err_msg=f"'none' does not match get_celltype_mapping at level '{level}'",
            )

    def test_within_class_single_child_accuracy_is_one(self, hierarchical_adata):
        """ClassC contains only TypeE — within-class accuracy must be 1.0."""
        result = get_panel_celltype_accuracy(
            hierarchical_adata, self._genes(hierarchical_adata), self.LEVEL_KEYS,
            constrained_method="within_class",
            knn_method="exact", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        type_e = result[(result["level"] == "celltype") & (result["celltype"] == "TypeE")]
        assert len(type_e) == 1
        assert type_e["constrained_accuracy"].iloc[0] == pytest.approx(1.0)

    def test_all_methods_give_high_accuracy_on_separable_data(self, hierarchical_adata):
        """All three methods should give high accuracy on well-separated synthetic data."""
        genes = self._genes(hierarchical_adata)
        for method in ("none", "global_filter", "within_class"):
            result = get_panel_celltype_accuracy(
                hierarchical_adata, genes, self.LEVEL_KEYS,
                constrained_method=method,
                knn_method="exact", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
            )
            mean_acc = result["constrained_accuracy"].mean()
            assert mean_acc > 0.8, (
                f"constrained_method='{method}' gave mean accuracy {mean_acc:.3f} < 0.8"
            )

    # --- Error handling tests ---

    def test_raises_on_invalid_constrained_method(self, hierarchical_adata):
        with pytest.raises(ValueError, match="constrained_method"):
            get_panel_celltype_accuracy(
                hierarchical_adata, self._genes(hierarchical_adata), self.LEVEL_KEYS,
                constrained_method="bad_method",
            )

    def test_raises_on_missing_level_key(self, hierarchical_adata):
        with pytest.raises(ValueError, match="level_keys not found"):
            get_panel_celltype_accuracy(
                hierarchical_adata, self._genes(hierarchical_adata),
                ["class_label", "nonexistent_level"],
            )

    def test_raises_on_non_one_to_one_hierarchy(self, hierarchical_adata):
        """A cell type that maps to two parent classes must raise ValueError."""
        adata_bad = hierarchical_adata.copy()
        first_typeA = np.where(adata_bad.obs["celltype"].values == "TypeA")[0][0]
        adata_bad.obs.loc[adata_bad.obs_names[first_typeA], "class_label"] = "ClassB"
        with pytest.raises(ValueError, match="Non-1-to-1"):
            get_panel_celltype_accuracy(
                adata_bad, self._genes(hierarchical_adata), self.LEVEL_KEYS,
            )

    def test_list_batch_key_runs(self, hierarchical_adata):
        """list[str] batch_key should be resolved to a joint column without error."""
        adata_b = hierarchical_adata.copy()
        rng = np.random.default_rng(0)
        adata_b.obs["donor_id"] = rng.choice(["d1", "d2"], size=adata_b.n_obs).tolist()
        adata_b.obs["region"]   = rng.choice(["r1", "r2"], size=adata_b.n_obs).tolist()
        result = get_panel_celltype_accuracy(
            adata_b, self._genes(hierarchical_adata), self.LEVEL_KEYS,
            constrained_method="none",
            batch_key=["donor_id", "region"],
            knn_method="exact", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        assert len(result) > 0

    def test_n_cells_per_group_reduces_cells(self, hierarchical_adata):
        """n_cells_per_group should subsample before kNN and give same schema."""
        # hierarchical_adata has 40 cells per type; cap at 20 → smaller run
        result = get_panel_celltype_accuracy(
            hierarchical_adata, self._genes(hierarchical_adata), self.LEVEL_KEYS,
            n_cells_per_group=20,
            knn_method="exact", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE,
        )
        assert list(result.columns) == [
            "level", "celltype", "constrained_accuracy", "compounded_accuracy"
        ]
        assert not result.duplicated(subset=["level", "celltype"]).any()
        assert result["constrained_accuracy"].between(0, 1).all()

    def test_n_cells_per_group_none_unchanged(self, hierarchical_adata):
        """n_cells_per_group=None should give the same result as omitting it."""
        genes = self._genes(hierarchical_adata)
        kw = dict(knn_method="exact", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE)
        r_default = get_panel_celltype_accuracy(
            hierarchical_adata, genes, self.LEVEL_KEYS, **kw
        )
        r_none = get_panel_celltype_accuracy(
            hierarchical_adata, genes, self.LEVEL_KEYS, n_cells_per_group=None, **kw
        )
        pd = pytest.importorskip("pandas")
        pd.testing.assert_frame_equal(r_default, r_none)
