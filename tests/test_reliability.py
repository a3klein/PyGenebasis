"""
Tests for pygenebasis.reliability.

Contract tests verify output shapes, types, and invariants that must hold
regardless of implementation details.  Behavioural tests verify that the
perturbations have the expected effect on the data.

All tests use the synthetic ``rel_adata`` and ``rel_results`` fixtures defined
below — no reference data files required.

Fixture design
--------------
rel_adata : 160 cells × 40 genes, 3-level hierarchy
    Class      : A, B          (2 classes)
    Subclass   : A1, A2, B1, B2  (2 per class)
    Group      : A1a, A1b, A2a, A2b, B1a, B1b, B2a, B2b  (2 per subclass)
    20 cells per Group → 160 cells total
    Each Group has 5 strong marker genes; remaining 15 are low-noise background.
    Two batch columns: batch_donor (4 donors) and batch_region (2 regions).

rel_results : 40-gene results DataFrame
    failure_mode distribution: ~20 reliable, ~7 probe_failure,
                               ~7 composition_mismatch, ~6 idiosyncratic_noise
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pygenebasis.reliability import (
    subsample_adata,
    perturb_expression,
    run_perturbation_analysis,
    plot_delta_ct_heatmap,
    plot_perturbation_summary,
    plot_removal_benefit,
)
from pygenebasis.io import prepare_batch_key
from pygenebasis.panel._mapping import _constrained_vote

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

LEVEL_KEYS  = ["Class", "Subclass", "Group"]
N_GENES     = 40
N_PER_GROUP = 20   # cells per finest-level group
N_GROUPS    = 8    # Group labels
N_CELLS     = N_PER_GROUP * N_GROUPS   # 160


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rel_adata() -> ad.AnnData:
    """Synthetic AnnData with a 3-level cell type hierarchy."""
    rng = np.random.default_rng(123)

    # Hierarchy: Class → Subclass → Group
    hierarchy = {
        "A1a": ("A", "A1"), "A1b": ("A", "A1"),
        "A2a": ("A", "A2"), "A2b": ("A", "A2"),
        "B1a": ("B", "B1"), "B1b": ("B", "B1"),
        "B2a": ("B", "B2"), "B2b": ("B", "B2"),
    }
    groups = list(hierarchy.keys())

    # Cell labels
    group_labels    = np.repeat(groups, N_PER_GROUP)
    subclass_labels = np.array([hierarchy[g][1] for g in group_labels])
    class_labels    = np.array([hierarchy[g][0] for g in group_labels])

    # Expression: 5 marker genes per Group (interleaved), rest background
    counts = rng.poisson(lam=0.2, size=(N_CELLS, N_GENES)).astype(np.float32)
    for i, grp in enumerate(groups):
        marker_cols = list(range(i, N_GENES, N_GROUPS))[:5]
        cell_rows   = np.where(group_labels == grp)[0]
        for col in marker_cols:
            counts[np.ix_(cell_rows, [col])] = rng.poisson(
                lam=15.0, size=(N_PER_GROUP, 1)
            ).astype(np.float32)
    counts = np.log1p(counts)

    obs = pd.DataFrame(
        {
            "Class":        class_labels,
            "Subclass":     subclass_labels,
            "Group":        group_labels,
            "batch_donor":  [f"donor{(i % 4) + 1}" for i in range(N_CELLS)],
            "batch_region": [f"region{(i % 2) + 1}" for i in range(N_CELLS)],
        },
        index=[f"cell_{i}" for i in range(N_CELLS)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(N_GENES)])
    return ad.AnnData(X=scipy.sparse.csr_matrix(counts), obs=obs, var=var)


@pytest.fixture(scope="module")
def rel_results(rel_adata: ad.AnnData) -> pd.DataFrame:
    """Synthetic results DataFrame with all four failure modes represented."""
    rng = np.random.default_rng(99)
    genes = rel_adata.var_names.tolist()
    n = len(genes)

    # Assign failure modes: ~half reliable, rest split across three modes
    modes = (
        ["reliable"]             * 20 +
        ["probe_failure"]        * 7  +
        ["composition_mismatch"] * 7  +
        ["idiosyncratic_noise"]  * 6
    )
    rng.shuffle(modes)

    return pd.DataFrame(
        {
            "profile_similarity":  rng.uniform(0.1, 0.9, size=n),
            "residual_variance":   rng.uniform(0.0, 0.01, size=n),
            "residual_z":          rng.standard_normal(size=n),
            "pc1_loading":         rng.standard_normal(size=n),
            "detection_log2_ratio": rng.uniform(-5, 5, size=n),
            "failure_mode":        modes,
        },
        index=genes,
    )


def _is_fig(obj) -> bool:
    return isinstance(obj, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# subsample_adata
# ---------------------------------------------------------------------------

class TestSubsampleAdata:

    def test_output_is_adata(self, rel_adata):
        out = subsample_adata(rel_adata, LEVEL_KEYS, n_cells_per_group=5)
        assert isinstance(out, ad.AnnData)

    def test_respects_per_group_limit(self, rel_adata):
        limit = 5
        out = subsample_adata(rel_adata, LEVEL_KEYS, n_cells_per_group=limit)
        counts = out.obs["Group"].value_counts()
        assert (counts <= limit).all()

    def test_keeps_all_when_limit_large(self, rel_adata):
        out = subsample_adata(rel_adata, LEVEL_KEYS, n_cells_per_group=N_PER_GROUP + 100)
        assert out.n_obs == rel_adata.n_obs

    def test_stratifies_at_finest_level(self, rel_adata):
        # Finest level is "Group"; all 8 groups should be represented
        out = subsample_adata(rel_adata, LEVEL_KEYS, n_cells_per_group=5)
        assert set(out.obs["Group"].unique()) == set(rel_adata.obs["Group"].unique())

    def test_coarser_levels_preserved(self, rel_adata):
        out = subsample_adata(rel_adata, LEVEL_KEYS, n_cells_per_group=5)
        assert "Class" in out.obs.columns
        assert "Subclass" in out.obs.columns

    def test_output_is_subset_of_input(self, rel_adata):
        out = subsample_adata(rel_adata, LEVEL_KEYS, n_cells_per_group=10)
        assert set(out.obs_names).issubset(set(rel_adata.obs_names))

    def test_deterministic(self, rel_adata):
        out1 = subsample_adata(rel_adata, LEVEL_KEYS, n_cells_per_group=10, random_state=7)
        out2 = subsample_adata(rel_adata, LEVEL_KEYS, n_cells_per_group=10, random_state=7)
        assert list(out1.obs_names) == list(out2.obs_names)

    def test_different_seeds_differ(self, rel_adata):
        out1 = subsample_adata(rel_adata, LEVEL_KEYS, n_cells_per_group=10, random_state=1)
        out2 = subsample_adata(rel_adata, LEVEL_KEYS, n_cells_per_group=10, random_state=2)
        assert list(out1.obs_names) != list(out2.obs_names)

    def test_single_level_key(self, rel_adata):
        out = subsample_adata(rel_adata, ["Group"], n_cells_per_group=5)
        assert (out.obs["Group"].value_counts() <= 5).all()

    def test_invalid_level_key_raises(self, rel_adata):
        with pytest.raises(ValueError):
            subsample_adata(rel_adata, ["Class", "NOT_A_COLUMN"], n_cells_per_group=5)

    def test_genes_unchanged(self, rel_adata):
        out = subsample_adata(rel_adata, LEVEL_KEYS, n_cells_per_group=5)
        assert list(out.var_names) == list(rel_adata.var_names)


# ---------------------------------------------------------------------------
# prepare_batch_key
# ---------------------------------------------------------------------------

class TestPrepareBatchKey:

    def test_string_key_unchanged(self, rel_adata):
        out_adata, out_key = prepare_batch_key(rel_adata, "batch_donor")
        assert out_key == "batch_donor"
        assert out_adata is rel_adata  # no copy for string key

    def test_none_key(self, rel_adata):
        out_adata, out_key = prepare_batch_key(rel_adata, None)
        assert out_key is None
        assert out_adata is rel_adata

    def test_list_key_creates_joint_column(self, rel_adata):
        out_adata, out_key = prepare_batch_key(rel_adata, ["batch_donor", "batch_region"])
        assert out_key is not None
        assert out_key in out_adata.obs.columns

    def test_joint_column_values(self, rel_adata):
        out_adata, out_key = prepare_batch_key(rel_adata, ["batch_donor", "batch_region"])
        expected = rel_adata.obs["batch_donor"].astype(str) + "_" + rel_adata.obs["batch_region"].astype(str)
        pd.testing.assert_series_equal(
            out_adata.obs[out_key].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_list_key_does_not_modify_input(self, rel_adata):
        original_cols = set(rel_adata.obs.columns)
        prepare_batch_key(rel_adata, ["batch_donor", "batch_region"])
        assert set(rel_adata.obs.columns) == original_cols

    def test_invalid_list_key_raises(self, rel_adata):
        with pytest.raises(ValueError):
            prepare_batch_key(rel_adata, ["batch_donor", "NOT_A_COLUMN"])

    def test_single_element_list(self, rel_adata):
        # A list with one element should still work (treated like a string key)
        out_adata, out_key = prepare_batch_key(rel_adata, ["batch_donor"])
        assert out_key is not None
        assert out_key in out_adata.obs.columns


# ---------------------------------------------------------------------------
# perturb_expression
# ---------------------------------------------------------------------------

class TestPerturbExpression:

    def _dense_X(self, adata: ad.AnnData) -> np.ndarray:
        X = adata.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        return X.astype(np.float64)

    def _genes_of_mode(self, rel_results, mode: str) -> list[str]:
        return rel_results.index[rel_results["failure_mode"] == mode].tolist()

    # --- Returns a copy ---

    def test_returns_adata(self, rel_adata, rel_results):
        genes = self._genes_of_mode(rel_results, "probe_failure")
        out = perturb_expression(rel_adata, genes, "probe_failure")
        assert isinstance(out, ad.AnnData)

    def test_does_not_modify_input(self, rel_adata, rel_results):
        genes = self._genes_of_mode(rel_results, "probe_failure")
        X_before = self._dense_X(rel_adata)
        perturb_expression(rel_adata, genes, "probe_failure")
        np.testing.assert_array_equal(self._dense_X(rel_adata), X_before)

    def test_output_shape_unchanged(self, rel_adata, rel_results):
        genes = self._genes_of_mode(rel_results, "composition_mismatch")
        out = perturb_expression(rel_adata, genes, "composition_mismatch")
        assert out.shape == rel_adata.shape

    # --- probe_failure ---

    def test_probe_failure_is_permutation(self, rel_adata, rel_results):
        genes = self._genes_of_mode(rel_results, "probe_failure")
        assert len(genes) >= 1
        out = perturb_expression(rel_adata, genes, "probe_failure", random_state=0)
        X_in  = self._dense_X(rel_adata).astype(np.float64)
        X_out = self._dense_X(out).astype(np.float64)
        gene_idx = [rel_adata.var_names.get_loc(g) for g in genes]
        for g in gene_idx:
            assert set(X_out[:, g].round(6)) == set(X_in[:, g].round(6)), \
                "probe_failure should permute values (same set, different order)"

    def test_probe_failure_changes_order(self, rel_adata, rel_results):
        genes = self._genes_of_mode(rel_results, "probe_failure")
        out = perturb_expression(rel_adata, genes, "probe_failure", random_state=0)
        X_in  = self._dense_X(rel_adata)
        X_out = self._dense_X(out)
        gene_idx = [rel_adata.var_names.get_loc(g) for g in genes]
        # At least one gene column should differ from original
        any_changed = any(not np.array_equal(X_out[:, g], X_in[:, g]) for g in gene_idx)
        assert any_changed

    def test_probe_failure_deterministic(self, rel_adata, rel_results):
        genes = self._genes_of_mode(rel_results, "probe_failure")
        out1 = perturb_expression(rel_adata, genes, "probe_failure", random_state=42)
        out2 = perturb_expression(rel_adata, genes, "probe_failure", random_state=42)
        np.testing.assert_array_equal(self._dense_X(out1), self._dense_X(out2))

    def test_probe_failure_different_seeds_differ(self, rel_adata, rel_results):
        genes = self._genes_of_mode(rel_results, "probe_failure")
        out1 = perturb_expression(rel_adata, genes, "probe_failure", random_state=1)
        out2 = perturb_expression(rel_adata, genes, "probe_failure", random_state=2)
        assert not np.array_equal(self._dense_X(out1), self._dense_X(out2))

    # --- composition_mismatch ---

    def test_composition_mismatch_zeros_genes(self, rel_adata, rel_results):
        genes = self._genes_of_mode(rel_results, "composition_mismatch")
        out = perturb_expression(rel_adata, genes, "composition_mismatch")
        X_out    = self._dense_X(out)
        gene_idx = [rel_adata.var_names.get_loc(g) for g in genes]
        for g in gene_idx:
            assert X_out[:, g].sum() == 0.0, f"gene {g} should be zeroed"

    def test_composition_mismatch_deterministic(self, rel_adata, rel_results):
        genes = self._genes_of_mode(rel_results, "composition_mismatch")
        out1 = perturb_expression(rel_adata, genes, "composition_mismatch", random_state=0)
        out2 = perturb_expression(rel_adata, genes, "composition_mismatch", random_state=99)
        np.testing.assert_array_equal(self._dense_X(out1), self._dense_X(out2))

    # --- idiosyncratic_noise ---

    def test_idiosyncratic_noise_changes_values(self, rel_adata, rel_results):
        genes = self._genes_of_mode(rel_results, "idiosyncratic_noise")
        out = perturb_expression(rel_adata, genes, "idiosyncratic_noise", random_state=0)
        X_in  = self._dense_X(rel_adata)
        X_out = self._dense_X(out)
        gene_idx = [rel_adata.var_names.get_loc(g) for g in genes]
        any_changed = any(not np.allclose(X_out[:, g], X_in[:, g]) for g in gene_idx)
        assert any_changed

    def test_idiosyncratic_noise_deterministic(self, rel_adata, rel_results):
        genes = self._genes_of_mode(rel_results, "idiosyncratic_noise")
        out1 = perturb_expression(rel_adata, genes, "idiosyncratic_noise", random_state=7)
        out2 = perturb_expression(rel_adata, genes, "idiosyncratic_noise", random_state=7)
        np.testing.assert_allclose(self._dense_X(out1), self._dense_X(out2))

    def test_idiosyncratic_noise_different_seeds_differ(self, rel_adata, rel_results):
        genes = self._genes_of_mode(rel_results, "idiosyncratic_noise")
        out1 = perturb_expression(rel_adata, genes, "idiosyncratic_noise", random_state=1)
        out2 = perturb_expression(rel_adata, genes, "idiosyncratic_noise", random_state=2)
        assert not np.allclose(self._dense_X(out1), self._dense_X(out2))

    # --- Unaffected genes are unchanged (all modes) ---

    @pytest.mark.parametrize("mode,key", [
        ("probe_failure",        "probe_failure"),
        ("composition_mismatch", "composition_mismatch"),
        ("idiosyncratic_noise",  "idiosyncratic_noise"),
    ])
    def test_unaffected_genes_unchanged(self, rel_adata, rel_results, mode, key):
        perturbed_genes = rel_results.index[rel_results["failure_mode"] == key].tolist()
        unaffected_genes = [g for g in rel_adata.var_names if g not in perturbed_genes]
        out = perturb_expression(rel_adata, perturbed_genes, mode, random_state=0)
        X_in  = self._dense_X(rel_adata)
        X_out = self._dense_X(out)
        idx = [rel_adata.var_names.get_loc(g) for g in unaffected_genes]
        np.testing.assert_array_equal(X_out[:, idx], X_in[:, idx])

    # --- Edge cases ---

    def test_empty_gene_list_returns_unchanged(self, rel_adata):
        out = perturb_expression(rel_adata, [], "probe_failure")
        np.testing.assert_array_equal(self._dense_X(out), self._dense_X(rel_adata))

    def test_genes_not_in_adata_silently_skipped(self, rel_adata):
        out = perturb_expression(rel_adata, ["FAKE_GENE"], "composition_mismatch")
        assert out.shape == rel_adata.shape

    def test_invalid_mode_raises(self, rel_adata):
        with pytest.raises(ValueError):
            perturb_expression(rel_adata, rel_adata.var_names[:2].tolist(), "reliable")

    def test_invalid_mode_unknown_raises(self, rel_adata):
        with pytest.raises(ValueError):
            perturb_expression(rel_adata, rel_adata.var_names[:2].tolist(), "totally_wrong_mode")


# ---------------------------------------------------------------------------
# run_perturbation_analysis
# ---------------------------------------------------------------------------

class TestRunPerturbationAnalysis:

    @pytest.fixture(scope="class")
    def results(self, rel_adata, rel_results):
        """Run the full analysis once and cache the output."""
        return run_perturbation_analysis(
            rel_adata,
            rel_results,
            level_keys=LEVEL_KEYS,
            n_neighbors=3,
            knn_method="exact",
            n_replicates=2,
            n_cells_per_group=N_PER_GROUP,  # no downsampling needed
            random_state=0,
            verbose=False,
        )

    # --- Return structure ---

    def test_returns_dict(self, results):
        assert isinstance(results, dict)

    def test_dict_has_required_keys(self, results):
        assert {"delta_ct", "baseline_ct", "summary"}.issubset(results.keys())

    # --- delta_ct ---

    def test_delta_ct_columns(self, results):
        assert {"failure_mode", "level", "celltype", "replicate", "delta_ct"}.issubset(
            results["delta_ct"].columns
        )

    def test_delta_ct_levels_match_level_keys(self, results):
        assert set(results["delta_ct"]["level"].unique()) == set(LEVEL_KEYS)

    def test_delta_ct_only_non_reliable_modes(self, results):
        modes = set(results["delta_ct"]["failure_mode"].unique())
        assert "reliable" not in modes

    def test_delta_ct_failure_modes_present(self, results, rel_results):
        expected = set(rel_results["failure_mode"].unique()) - {"reliable"}
        assert expected.issubset(set(results["delta_ct"]["failure_mode"].unique()))

    def test_delta_ct_values_are_numeric(self, results):
        assert pd.api.types.is_float_dtype(results["delta_ct"]["delta_ct"])
        assert results["delta_ct"]["delta_ct"].notna().all()

    def test_delta_ct_stochastic_modes_have_replicates(self, results):
        stochastic = results["delta_ct"][
            results["delta_ct"]["failure_mode"].isin(["probe_failure", "idiosyncratic_noise"])
        ]
        n_replicates = stochastic.groupby(["failure_mode", "level", "celltype"])["replicate"].nunique()
        assert (n_replicates == 2).all()

    def test_delta_ct_deterministic_mode_has_one_replicate(self, results):
        determ = results["delta_ct"][
            results["delta_ct"]["failure_mode"] == "composition_mismatch"
        ]
        n_replicates = determ.groupby(["failure_mode", "level", "celltype"])["replicate"].nunique()
        assert (n_replicates == 1).all()

    # --- baseline_ct ---

    def test_baseline_ct_columns(self, results):
        assert {"level", "celltype", "ct_accuracy"}.issubset(results["baseline_ct"].columns)

    def test_baseline_ct_levels_match(self, results):
        assert set(results["baseline_ct"]["level"].unique()) == set(LEVEL_KEYS)

    def test_baseline_ct_accuracy_in_range(self, results):
        acc = results["baseline_ct"]["ct_accuracy"]
        assert (acc >= 0).all() and (acc <= 1).all()

    def test_baseline_ct_accuracy_above_chance(self, results):
        # On well-separated synthetic data, baseline accuracy should be > chance
        mean_acc = results["baseline_ct"]["ct_accuracy"].mean()
        assert mean_acc > 0.5, f"Mean baseline CT accuracy too low: {mean_acc:.3f}"

    def test_baseline_ct_celltypes_at_finest_level(self, results, rel_adata):
        finest_rows = results["baseline_ct"][results["baseline_ct"]["level"] == "Group"]
        expected_groups = set(rel_adata.obs["Group"].unique())
        assert set(finest_rows["celltype"].unique()) == expected_groups

    # --- summary ---

    def test_summary_columns(self, results):
        assert {"failure_mode", "level", "mean_delta", "std_delta", "n_cell_types"}.issubset(
            results["summary"].columns
        )

    def test_summary_levels_match(self, results):
        assert set(results["summary"]["level"].unique()) == set(LEVEL_KEYS)

    def test_summary_modes_match_delta_ct(self, results):
        assert set(results["summary"]["failure_mode"].unique()) == \
               set(results["delta_ct"]["failure_mode"].unique())

    def test_summary_mean_delta_is_numeric(self, results):
        assert pd.api.types.is_float_dtype(results["summary"]["mean_delta"])

    # --- Determinism ---

    def test_deterministic(self, rel_adata, rel_results):
        kwargs = dict(
            level_keys=LEVEL_KEYS, n_neighbors=3, knn_method="exact",
            n_replicates=2, n_cells_per_group=N_PER_GROUP, random_state=0, verbose=False,
        )
        r1 = run_perturbation_analysis(rel_adata, rel_results, **kwargs)
        r2 = run_perturbation_analysis(rel_adata, rel_results, **kwargs)
        pd.testing.assert_frame_equal(
            r1["delta_ct"].reset_index(drop=True),
            r2["delta_ct"].reset_index(drop=True),
        )

    # --- Input validation ---

    def test_invalid_level_key_raises(self, rel_adata, rel_results):
        with pytest.raises(ValueError):
            run_perturbation_analysis(
                rel_adata, rel_results,
                level_keys=["Class", "NOT_A_COLUMN"],
                n_replicates=1, verbose=False,
            )

    def test_single_level(self, rel_adata, rel_results):
        r = run_perturbation_analysis(
            rel_adata, rel_results,
            level_keys=["Group"],
            n_neighbors=3, knn_method="exact",
            n_replicates=1, n_cells_per_group=N_PER_GROUP,
            verbose=False,
        )
        assert set(r["delta_ct"]["level"].unique()) == {"Group"}

    def test_panel_from_results_index(self, rel_adata, rel_results):
        # Subset results to half the genes; only those genes should be used
        subset = rel_results.iloc[:20]
        r = run_perturbation_analysis(
            rel_adata, subset,
            level_keys=["Group"],
            n_neighbors=3, knn_method="exact",
            n_replicates=1, n_cells_per_group=N_PER_GROUP,
            verbose=False,
        )
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# plot_delta_ct_heatmap
# ---------------------------------------------------------------------------

class TestPlotDeltaCtHeatmap:

    def _make_delta_ct(self, level: str = "Group") -> pd.DataFrame:
        rng = np.random.default_rng(0)
        failure_modes = ["probe_failure", "composition_mismatch", "idiosyncratic_noise"]
        celltypes     = [f"Group_{c}" for c in "abcdef"]
        rows = []
        for fm in failure_modes:
            n_rep = 1 if fm == "composition_mismatch" else 3
            for rep in range(n_rep):
                for ct in celltypes:
                    rows.append({
                        "failure_mode": fm,
                        "level":        level,
                        "celltype":     ct,
                        "replicate":    rep,
                        "delta_ct":     float(rng.uniform(-0.1, 0.2)),
                    })
        return pd.DataFrame(rows)

    def _make_baseline_ct(self, level: str = "Group") -> pd.DataFrame:
        celltypes = [f"Group_{c}" for c in "abcdef"]
        return pd.DataFrame({
            "level":       [level] * len(celltypes),
            "celltype":    celltypes,
            "ct_accuracy": np.linspace(0.7, 0.98, len(celltypes)),
        })

    def test_returns_fig_ax(self):
        delta = self._make_delta_ct()
        result = plot_delta_ct_heatmap(delta, level="Group")
        assert isinstance(result, tuple) and len(result) == 2
        assert _is_fig(result[0])

    def test_with_baseline_ct(self):
        delta    = self._make_delta_ct()
        baseline = self._make_baseline_ct()
        fig, ax  = plot_delta_ct_heatmap(delta, level="Group", baseline_ct=baseline)
        assert _is_fig(fig)

    def test_custom_delta_threshold(self):
        delta   = self._make_delta_ct()
        fig, ax = plot_delta_ct_heatmap(delta, level="Group", delta_threshold=0.01)
        assert _is_fig(fig)

    def test_many_celltypes(self):
        rng       = np.random.default_rng(1)
        celltypes = [f"ct_{i}" for i in range(30)]
        rows = [
            {"failure_mode": "probe_failure", "level": "Group",
             "celltype": ct, "replicate": 0, "delta_ct": float(rng.uniform(-0.1, 0.2))}
            for ct in celltypes
        ]
        fig, ax = plot_delta_ct_heatmap(pd.DataFrame(rows), level="Group")
        assert _is_fig(fig)


# ---------------------------------------------------------------------------
# plot_perturbation_summary
# ---------------------------------------------------------------------------

class TestPlotPerturbationSummary:

    def _make_summary(self) -> pd.DataFrame:
        rng    = np.random.default_rng(2)
        modes  = ["probe_failure", "composition_mismatch", "idiosyncratic_noise"]
        levels = ["Class", "Subclass", "Group"]
        rows   = []
        for fm in modes:
            for lv in levels:
                rows.append({
                    "failure_mode":  fm,
                    "level":         lv,
                    "mean_delta":    float(rng.uniform(-0.05, 0.15)),
                    "std_delta":     float(rng.uniform(0.0, 0.05)),
                    "n_cell_types":  rng.integers(2, 10),
                })
        return pd.DataFrame(rows)

    def test_returns_fig_axes(self):
        summary = self._make_summary()
        result  = plot_perturbation_summary(summary)
        assert isinstance(result, tuple) and len(result) == 2
        assert _is_fig(result[0])

    def test_one_level(self):
        summary = self._make_summary()
        summary = summary[summary["level"] == "Group"].copy()
        fig, axes = plot_perturbation_summary(summary)
        assert _is_fig(fig)

    def test_custom_figsize(self):
        summary  = self._make_summary()
        fig, axes = plot_perturbation_summary(summary, figsize=(12, 4))
        assert _is_fig(fig)


# ---------------------------------------------------------------------------
# run_perturbation_analysis — removal / benefit outputs
# ---------------------------------------------------------------------------

class TestRunPerturbationAnalysisRemoval:
    """Tests for the two new output keys: delta_ct_removal and removal_benefit."""

    @pytest.fixture(scope="class")
    def results(self, rel_adata, rel_results):
        """Run the full analysis once (with removal) and cache the output."""
        return run_perturbation_analysis(
            rel_adata,
            rel_results,
            level_keys=LEVEL_KEYS,
            n_neighbors=3,
            knn_method="exact",
            n_replicates=2,
            n_cells_per_group=N_PER_GROUP,
            random_state=0,
            verbose=False,
        )

    # --- delta_ct_removal present and well-formed ---

    def test_delta_ct_removal_key_present(self, results):
        assert "delta_ct_removal" in results

    def test_delta_ct_removal_is_dataframe(self, results):
        assert isinstance(results["delta_ct_removal"], pd.DataFrame)

    def test_delta_ct_removal_columns(self, results):
        assert {"failure_mode", "level", "celltype", "delta_ct"}.issubset(
            results["delta_ct_removal"].columns
        )

    def test_delta_ct_removal_no_replicate_column(self, results):
        # Removal is deterministic — there should be no replicate column
        assert "replicate" not in results["delta_ct_removal"].columns

    def test_delta_ct_removal_levels_match(self, results):
        assert set(results["delta_ct_removal"]["level"].unique()) == set(LEVEL_KEYS)

    def test_delta_ct_removal_only_non_reliable_modes(self, results):
        assert "reliable" not in results["delta_ct_removal"]["failure_mode"].unique()

    def test_delta_ct_removal_values_numeric(self, results):
        assert pd.api.types.is_float_dtype(results["delta_ct_removal"]["delta_ct"])
        assert results["delta_ct_removal"]["delta_ct"].notna().all()

    # --- removal_benefit present and well-formed ---

    def test_removal_benefit_key_present(self, results):
        assert "removal_benefit" in results

    def test_removal_benefit_is_dataframe(self, results):
        assert isinstance(results["removal_benefit"], pd.DataFrame)

    def test_removal_benefit_columns(self, results):
        assert {"failure_mode", "level", "celltype", "removal_benefit"}.issubset(
            results["removal_benefit"].columns
        )

    def test_removal_benefit_levels_match(self, results):
        assert set(results["removal_benefit"]["level"].unique()) == set(LEVEL_KEYS)

    def test_removal_benefit_values_numeric(self, results):
        assert pd.api.types.is_float_dtype(results["removal_benefit"]["removal_benefit"])
        assert results["removal_benefit"]["removal_benefit"].notna().all()

    def test_removal_benefit_formula(self, results):
        """benefit = mean(delta_ct_perturbed) - delta_ct_removal for each row."""
        mean_pert = (
            results["delta_ct"]
            .groupby(["failure_mode", "level", "celltype"])["delta_ct"]
            .mean()
            .reset_index()
            .rename(columns={"delta_ct": "mean_pert"})
        )
        rem = results["delta_ct_removal"].rename(columns={"delta_ct": "delta_rem"})
        merged = mean_pert.merge(rem, on=["failure_mode", "level", "celltype"])
        expected_benefit = merged["mean_pert"] - merged["delta_rem"]

        actual = results["removal_benefit"].merge(
            merged[["failure_mode", "level", "celltype"]],
            on=["failure_mode", "level", "celltype"],
        )
        np.testing.assert_allclose(
            actual["removal_benefit"].values,
            expected_benefit.values,
            atol=1e-10,
        )

    def test_removal_benefit_same_cell_types_as_perturbed(self, results):
        # Every (failure_mode, level, celltype) in removal_benefit should
        # also appear in delta_ct
        pert_keys = set(
            zip(
                results["delta_ct"]["failure_mode"],
                results["delta_ct"]["level"],
                results["delta_ct"]["celltype"],
            )
        )
        for _, row in results["removal_benefit"].iterrows():
            key = (row["failure_mode"], row["level"], row["celltype"])
            assert key in pert_keys, f"Unexpected key in removal_benefit: {key}"


# ---------------------------------------------------------------------------
# plot_removal_benefit
# ---------------------------------------------------------------------------

class TestPlotRemovalBenefit:

    def _make_removal_benefit(self, level: str = "Group") -> pd.DataFrame:
        rng = np.random.default_rng(3)
        failure_modes = ["probe_failure", "composition_mismatch", "idiosyncratic_noise"]
        celltypes     = [f"Group_{c}" for c in "abcde"]
        rows = []
        for fm in failure_modes:
            for ct in celltypes:
                rows.append({
                    "failure_mode":    fm,
                    "level":           level,
                    "celltype":        ct,
                    "removal_benefit": float(rng.uniform(-0.1, 0.15)),
                })
        return pd.DataFrame(rows)

    def test_returns_fig_ax_tuple(self):
        rb  = self._make_removal_benefit()
        result = plot_removal_benefit(rb, level="Group")
        assert isinstance(result, tuple) and len(result) == 2
        assert _is_fig(result[0])

    def test_custom_threshold(self):
        rb = self._make_removal_benefit()
        fig, ax = plot_removal_benefit(rb, level="Group", benefit_threshold=0.01)
        assert _is_fig(fig)

    def test_custom_figsize(self):
        rb = self._make_removal_benefit()
        fig, ax = plot_removal_benefit(rb, level="Group", figsize=(10, 8))
        assert _is_fig(fig)


# ---------------------------------------------------------------------------
# _constrained_vote (unit tests)
# ---------------------------------------------------------------------------

class TestConstrainedVote:
    """Unit tests for _constrained_vote.

    Builds small synthetic kNN results so tests are fast and deterministic.
    """

    def _simple_setup(self):
        """4 cells, k=2 neighbours each.

        true_labels:   [A1, A1, B1, B1]
        parent_labels: [A,  A,  B,  B ]
        indices: each cell points to its 2 closest (all within same parent)
        """
        true_labels   = np.array(["A1", "A1", "B1", "B1"])
        parent_labels = np.array(["A",  "A",  "B",  "B" ])
        indices   = np.array([[1, 0], [0, 1], [3, 2], [2, 3]])
        distances = np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
        return indices, distances, true_labels, parent_labels

    def test_returns_ndarray(self):
        indices, distances, true_labels, parent_labels = self._simple_setup()
        out = _constrained_vote(indices, distances, true_labels, parent_labels)
        assert isinstance(out, np.ndarray)

    def test_output_length(self):
        indices, distances, true_labels, parent_labels = self._simple_setup()
        out = _constrained_vote(indices, distances, true_labels, parent_labels)
        assert len(out) == len(true_labels)

    def test_perfect_separation_correct_predictions(self):
        """When neighbours are perfectly within-parent, predictions should be 100%."""
        indices, distances, true_labels, parent_labels = self._simple_setup()
        out = _constrained_vote(indices, distances, true_labels, parent_labels)
        np.testing.assert_array_equal(out, true_labels)

    def test_no_cross_lineage_predictions(self):
        """Constrained vote must not predict a label from a different parent class."""
        rng = np.random.default_rng(42)
        # 8 cells: 4 from class A (labels A1/A2), 4 from class B (labels B1/B2)
        true_labels   = np.array(["A1", "A1", "A2", "A2", "B1", "B1", "B2", "B2"])
        parent_labels = np.array(["A",  "A",  "A",  "A",  "B",  "B",  "B",  "B" ])
        # Each cell's 3 neighbours are all within the same parent class
        indices = np.array([
            [1, 2, 3],  # A cell → A neighbours
            [0, 2, 3],
            [3, 0, 1],
            [2, 0, 1],
            [5, 6, 7],  # B cell → B neighbours
            [4, 6, 7],
            [7, 4, 5],
            [6, 4, 5],
        ])
        distances = np.tile([0.1, 0.2, 0.3], (8, 1))

        out = _constrained_vote(indices, distances, true_labels, parent_labels)

        # A cells must predict an A sub-type; B cells must predict a B sub-type
        a_cell_mask = parent_labels == "A"
        b_cell_mask = parent_labels == "B"
        assert all(lbl.startswith("A") for lbl in out[a_cell_mask])
        assert all(lbl.startswith("B") for lbl in out[b_cell_mask])

    def test_coarsest_level_all_same_parent(self):
        """When all cells share one parent, constrained == unconstrained.

        At the coarsest level every cell has the same parent label so no
        neighbour is filtered out — constrained and unconstrained agree.
        """
        from pygenebasis.panel._mapping import _mode_with_tiebreak

        rng = np.random.default_rng(7)
        true_labels   = np.array(["A", "B", "A", "B", "A"])
        parent_labels = np.full(5, "root")  # single parent

        indices   = np.array([[1, 2], [0, 3], [0, 4], [1, 4], [0, 2]])
        distances = np.tile([0.1, 0.2], (5, 1))

        out_con = _constrained_vote(indices, distances, true_labels, parent_labels)
        out_unc = _mode_with_tiebreak(true_labels[indices], distances)

        np.testing.assert_array_equal(out_con, out_unc)

    def test_fallback_when_no_valid_neighbours(self):
        """Cells with zero same-parent neighbours fall back to unconstrained."""
        from pygenebasis.panel._mapping import _mode_with_tiebreak

        # 4 cells: cell 0 is class A; cells 1-3 are class B.
        # Cell 0's 3 neighbours (cells 1-3) are all class B → no valid constrained
        # neighbours → fallback to unconstrained vote.
        true_labels   = np.array(["A1", "B1", "B1", "B2"])
        parent_labels = np.array(["A",  "B",  "B",  "B" ])
        # Only cell 0 is a query cell; indices index into the full 4-cell array
        indices   = np.array([[1, 2, 3]])
        distances = np.array([[0.1, 0.2, 0.3]])

        out_con = _constrained_vote(indices, distances, true_labels, parent_labels)
        out_unc = _mode_with_tiebreak(
            true_labels[np.array([[1, 2, 3]])],
            distances,
        )
        # Fallback → same result as unconstrained for this cell
        np.testing.assert_array_equal(out_con, out_unc)

    def test_tiebreak_uses_distance_order(self):
        """Tie resolved by first occurrence in distance-sorted valid neighbour list."""
        # 5 cells: cell 0 is the query (label "Q"); cells 1-4 are its neighbours.
        # All cells share parent "P" so no filtering occurs.
        # Neighbour order (by distance): X2, X1, X2, X1 → X2 wins the tie.
        true_labels   = np.array(["Q", "X2", "X1", "X2", "X1"])
        parent_labels = np.full(5, "P")
        indices   = np.array([[1, 2, 3, 4]])   # 1 query cell, neighbours at 1-4
        distances = np.array([[0.1, 0.2, 0.3, 0.4]])

        # Pass full 5-cell arrays; function only predicts for the 1 query cell
        out = _constrained_vote(indices, distances, true_labels, parent_labels)
        assert out[0] == "X2"


# ---------------------------------------------------------------------------
# run_perturbation_analysis — vote parameter and constrained outputs
# ---------------------------------------------------------------------------

class TestRunPerturbationAnalysisConstrained:
    """Tests for constrained-vote outputs and the vote parameter."""

    @pytest.fixture(scope="class")
    def results_both(self, rel_adata, rel_results):
        """run_perturbation_analysis with vote='both' (default)."""
        return run_perturbation_analysis(
            rel_adata,
            rel_results,
            level_keys=LEVEL_KEYS,
            n_neighbors=3,
            knn_method="exact",
            n_replicates=2,
            n_cells_per_group=N_PER_GROUP,
            random_state=0,
            vote="both",
            verbose=False,
        )

    @pytest.fixture(scope="class")
    def results_unc(self, rel_adata, rel_results):
        """run_perturbation_analysis with vote='unconstrained'."""
        return run_perturbation_analysis(
            rel_adata,
            rel_results,
            level_keys=LEVEL_KEYS,
            n_neighbors=3,
            knn_method="exact",
            n_replicates=2,
            n_cells_per_group=N_PER_GROUP,
            random_state=0,
            vote="unconstrained",
            verbose=False,
        )

    # --- vote parameter validation ---

    def test_invalid_vote_raises(self, rel_adata, rel_results):
        with pytest.raises(ValueError, match="vote must be one of"):
            run_perturbation_analysis(
                rel_adata, rel_results,
                level_keys=LEVEL_KEYS,
                n_replicates=1, verbose=False,
                vote="magic",
            )

    # --- vote='unconstrained' omits constrained keys ---

    def test_vote_unconstrained_no_constrained_keys(self, results_unc):
        constrained_keys = {
            "delta_ct_constrained", "baseline_ct_constrained",
            "summary_constrained", "delta_ct_removal_constrained",
            "removal_benefit_constrained",
        }
        assert constrained_keys.isdisjoint(results_unc.keys())

    def test_vote_unconstrained_standard_keys_present(self, results_unc):
        assert {"delta_ct", "baseline_ct", "summary",
                "delta_ct_removal", "removal_benefit"}.issubset(results_unc.keys())

    # --- vote='both' / vote='constrained' add constrained keys ---

    def test_vote_both_constrained_keys_present(self, results_both):
        constrained_keys = {
            "delta_ct_constrained", "baseline_ct_constrained",
            "summary_constrained", "delta_ct_removal_constrained",
            "removal_benefit_constrained",
        }
        assert constrained_keys.issubset(results_both.keys())

    def test_vote_constrained_adds_constrained_keys(self, rel_adata, rel_results):
        r = run_perturbation_analysis(
            rel_adata, rel_results,
            level_keys=LEVEL_KEYS,
            n_neighbors=3, knn_method="exact",
            n_replicates=1, n_cells_per_group=N_PER_GROUP,
            random_state=0, vote="constrained", verbose=False,
        )
        assert "delta_ct_constrained" in r
        assert "baseline_ct_constrained" in r

    # --- Structural checks on constrained DataFrames ---

    def test_baseline_ct_constrained_columns(self, results_both):
        assert {"level", "celltype", "ct_accuracy"}.issubset(
            results_both["baseline_ct_constrained"].columns
        )

    def test_baseline_ct_constrained_levels_match(self, results_both):
        assert set(results_both["baseline_ct_constrained"]["level"].unique()) == set(LEVEL_KEYS)

    def test_baseline_ct_constrained_accuracy_in_range(self, results_both):
        acc = results_both["baseline_ct_constrained"]["ct_accuracy"]
        assert (acc >= 0).all() and (acc <= 1).all()

    def test_delta_ct_constrained_columns(self, results_both):
        assert {"failure_mode", "level", "celltype", "replicate", "delta_ct"}.issubset(
            results_both["delta_ct_constrained"].columns
        )

    def test_delta_ct_constrained_levels_match(self, results_both):
        assert set(results_both["delta_ct_constrained"]["level"].unique()) == set(LEVEL_KEYS)

    def test_delta_ct_constrained_only_non_reliable_modes(self, results_both):
        assert "reliable" not in results_both["delta_ct_constrained"]["failure_mode"].unique()

    def test_summary_constrained_columns(self, results_both):
        assert {"failure_mode", "level", "mean_delta", "std_delta", "n_cell_types"}.issubset(
            results_both["summary_constrained"].columns
        )

    def test_delta_ct_removal_constrained_columns(self, results_both):
        assert {"failure_mode", "level", "celltype", "delta_ct"}.issubset(
            results_both["delta_ct_removal_constrained"].columns
        )

    def test_delta_ct_removal_constrained_no_replicate(self, results_both):
        assert "replicate" not in results_both["delta_ct_removal_constrained"].columns

    def test_removal_benefit_constrained_columns(self, results_both):
        assert {"failure_mode", "level", "celltype", "removal_benefit"}.issubset(
            results_both["removal_benefit_constrained"].columns
        )

    # --- Hierarchy constraint: coarsest level constrained == unconstrained ---

    def test_coarsest_level_constrained_equals_unconstrained_baseline(self, results_both):
        """Class-level (coarsest) has no parent filter → identical to unconstrained."""
        coarsest = LEVEL_KEYS[0]
        unc = (
            results_both["baseline_ct"][results_both["baseline_ct"]["level"] == coarsest]
            .set_index("celltype")["ct_accuracy"]
            .sort_index()
        )
        con = (
            results_both["baseline_ct_constrained"][
                results_both["baseline_ct_constrained"]["level"] == coarsest
            ]
            .set_index("celltype")["ct_accuracy"]
            .sort_index()
        )
        pd.testing.assert_series_equal(unc, con, check_names=False)

    def test_coarsest_level_constrained_equals_unconstrained_delta(self, results_both):
        """At coarsest level, mean delta CT should be the same unc vs. con."""
        coarsest = LEVEL_KEYS[0]
        unc = results_both["delta_ct"][
            results_both["delta_ct"]["level"] == coarsest
        ].groupby(["failure_mode", "celltype"])["delta_ct"].mean().sort_index()
        con = results_both["delta_ct_constrained"][
            results_both["delta_ct_constrained"]["level"] == coarsest
        ].groupby(["failure_mode", "celltype"])["delta_ct"].mean().sort_index()
        pd.testing.assert_series_equal(unc, con, check_names=False)

    # --- Standard keys identical regardless of vote setting ---

    def test_standard_keys_unchanged_by_vote(self, rel_adata, rel_results):
        """Unconstrained keys must be identical whether vote='both' or vote='unconstrained'."""
        kwargs = dict(
            level_keys=LEVEL_KEYS, n_neighbors=3, knn_method="exact",
            n_replicates=2, n_cells_per_group=N_PER_GROUP, random_state=0, verbose=False,
        )
        r_both = run_perturbation_analysis(rel_adata, rel_results, vote="both",          **kwargs)
        r_unc  = run_perturbation_analysis(rel_adata, rel_results, vote="unconstrained", **kwargs)
        pd.testing.assert_frame_equal(
            r_both["delta_ct"].reset_index(drop=True),
            r_unc["delta_ct"].reset_index(drop=True),
        )
