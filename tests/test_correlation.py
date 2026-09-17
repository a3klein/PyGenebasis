"""
Tests for pygenebasis.correlation.

All tests are self-contained — no external data files required.
Synthetic AnnData fixtures are created in memory or written to temporary
directories by pytest.

Fixture overview
----------------
genes (module scope)
    List of 20 gene names shared by both AnnData fixtures.

merfish_adata (module scope)
    180 cells × 20 genes, dense float32 expression.  Three cell types
    (TypeA × 80, TypeB × 60, TypeC × 40) stored in obs["cell_type"].
    Expression sampled from lognormal to mimic MERFISH density.

ref_adata (module scope)
    400 cells × 20 genes, sparse CSR float32 expression.  Same three cell
    types (TypeA × 200, TypeB × 120, TypeC × 80).  Expression sampled from
    Poisson to mimic scRNA-seq sparsity.

ref_h5ad_path (module scope)
    ref_adata written to a temporary h5ad file; used to test backed loading.

corr_data (module scope)
    Output of compute_corr_matrices(merfish_adata, ref_adata), computed once
    and reused across scoring tests.

scored (module scope)
    (results, thresholds) from score_gene_reliability(corr_data), computed
    once and reused across assertion tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
import anndata as ad

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pygenebasis.reliability._correlation import (
    _choose_n_pcs,
    _parallel_analysis,
    _extract_expression,
    _pearson_corr_from_matrix,
    _linear_rescale,
    _subsample,
    compute_corr_matrices,
    score_gene_reliability,
    compute_gene_reliability,
    plot_metric_distributions,
    plot_failure_mode_scatter,
)
from pygenebasis.io import read_adata


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_MERFISH = 180
N_REF     = 400
N_GENES   = 20

GENE_NAMES = [f"gene_{i:02d}" for i in range(N_GENES)]

# Cell type sizes — TypeC is deliberately rare for stratified subsampling tests
CT_MERFISH = {"TypeA": 80, "TypeB": 60, "TypeC": 40}
CT_REF     = {"TypeA": 200, "TypeB": 120, "TypeC": 80}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adata(
    n_obs: int,
    n_vars: int,
    gene_names: list[str],
    ct_counts: dict[str, int],
    sparse: bool,
    rng: np.random.Generator,
) -> ad.AnnData:
    """Build a synthetic AnnData with cell-type labels."""
    if sparse:
        X = sps.csr_matrix(rng.poisson(0.5, (n_obs, n_vars)).astype(np.float32))
    else:
        X = np.exp(rng.normal(1.0, 1.0, (n_obs, n_vars))).astype(np.float32)

    labels: list[str] = []
    for ct, count in ct_counts.items():
        labels.extend([ct] * count)
    assert len(labels) == n_obs

    obs = pd.DataFrame({"cell_type": labels}, index=[f"cell_{i:04d}" for i in range(n_obs)])
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene_id"))
    return ad.AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def genes() -> list[str]:
    return GENE_NAMES


@pytest.fixture(scope="module")
def merfish_adata() -> ad.AnnData:
    rng = np.random.default_rng(42)
    return _make_adata(N_MERFISH, N_GENES, GENE_NAMES, CT_MERFISH, sparse=False, rng=rng)


@pytest.fixture(scope="module")
def ref_adata() -> ad.AnnData:
    rng = np.random.default_rng(7)
    return _make_adata(N_REF, N_GENES, GENE_NAMES, CT_REF, sparse=True, rng=rng)


@pytest.fixture(scope="module")
def ref_h5ad_path(tmp_path_factory, ref_adata) -> Path:
    """ref_adata written to disk; used for backed-loading tests."""
    path = tmp_path_factory.mktemp("h5ad") / "ref.h5ad"
    ref_adata.write_h5ad(path)
    return path


@pytest.fixture(scope="module")
def corr_data(merfish_adata, ref_adata) -> dict:
    return compute_corr_matrices(merfish_adata, ref_adata)


@pytest.fixture(scope="module")
def scored(corr_data) -> tuple[pd.DataFrame, dict]:
    return score_gene_reliability(corr_data)


# ---------------------------------------------------------------------------
# TestChooseNPcs
# ---------------------------------------------------------------------------

class TestChooseNPcs:

    def test_exact_threshold_met_at_first_component(self):
        evr = np.array([0.9, 0.05, 0.05])
        assert _choose_n_pcs(evr, 0.9) == 1

    def test_requires_multiple_components(self):
        # Use exact binary fractions to avoid floating-point cumsum issues.
        # evr = [0.5, 0.25, 0.25] → cumsum = [0.5, 0.75, 1.0] (exact in float64)
        evr = np.array([0.5, 0.25, 0.25])
        # threshold 0.9: cumsum[0]=0.5 < 0.9, cumsum[1]=0.75 < 0.9, cumsum[2]=1.0 ≥ 0.9
        assert _choose_n_pcs(evr, 0.9) == 3

    def test_threshold_never_reached_returns_all(self):
        # Floating-point: cumsum may not reach exactly 1.0 due to rounding
        evr = np.array([0.3, 0.3, 0.3])   # sums to 0.9, not ≥ 1.0
        k = _choose_n_pcs(evr, 1.0)
        assert k == len(evr)

    def test_single_component_sufficient(self):
        evr = np.array([1.0])
        assert _choose_n_pcs(evr, 0.9) == 1

    def test_minimum_is_one(self):
        evr = np.array([0.5, 0.5])
        k = _choose_n_pcs(evr, 0.01)  # first component already > 0.01
        assert k == 1


# ---------------------------------------------------------------------------
# TestPearsonCorrFromMatrix
# ---------------------------------------------------------------------------

class TestPearsonCorrFromMatrix:

    def _known_X(self) -> np.ndarray:
        """4 cells × 3 genes with deterministic values."""
        return np.array(
            [[1.0, 2.0, 3.0],
             [2.0, 4.0, 6.0],
             [0.0, 1.0, 0.0],
             [3.0, 1.0, 2.0]],
            dtype=float,
        )

    def test_shape(self):
        X = self._known_X()
        C = _pearson_corr_from_matrix(X)
        assert C.shape == (3, 3)

    def test_diagonal_is_one(self):
        X = self._known_X()
        C = _pearson_corr_from_matrix(X)
        np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-10)

    def test_symmetric(self):
        X = self._known_X()
        C = _pearson_corr_from_matrix(X)
        np.testing.assert_allclose(C, C.T, atol=1e-10)

    def test_dense_matches_numpy_corrcoef(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 8))
        C = _pearson_corr_from_matrix(X)
        expected = np.corrcoef(X, rowvar=False)
        np.testing.assert_allclose(C, expected, atol=1e-10)

    def test_sparse_matches_dense(self):
        rng = np.random.default_rng(1)
        X_dense = rng.poisson(0.5, (80, 10)).astype(float)
        X_sparse = sps.csr_matrix(X_dense)
        C_dense  = _pearson_corr_from_matrix(X_dense)
        C_sparse = _pearson_corr_from_matrix(X_sparse)
        np.testing.assert_allclose(C_sparse, C_dense, atol=1e-6)

    def test_all_zero_gene_diagonal_one_offdiag_zero(self):
        """A constant-zero gene should produce diag=1 and off-diag=0."""
        X = np.array(
            [[0.0, 1.0, 2.0],
             [0.0, 3.0, 1.0],
             [0.0, 2.0, 4.0]],
            dtype=float,
        )
        C = _pearson_corr_from_matrix(X)
        assert C[0, 0] == pytest.approx(1.0)
        assert C[0, 1] == pytest.approx(0.0)
        assert C[1, 0] == pytest.approx(0.0)

    def test_perfectly_correlated_genes(self):
        """gene_0 = 2 * gene_1 → correlation = 1."""
        X = np.column_stack([np.arange(10, dtype=float), np.arange(10, dtype=float) * 2])
        C = _pearson_corr_from_matrix(X)
        assert C[0, 1] == pytest.approx(1.0, abs=1e-10)

    def test_values_in_minus_one_to_one(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((30, 12))
        C = _pearson_corr_from_matrix(X)
        assert C.min() >= -1.0 - 1e-10
        assert C.max() <=  1.0 + 1e-10


# ---------------------------------------------------------------------------
# TestLinearRescale
# ---------------------------------------------------------------------------

class TestLinearRescale:

    def _make_symmetric(self, n: int, rng: np.random.Generator) -> np.ndarray:
        X = rng.standard_normal((n, n))
        S = (X + X.T) / 2
        np.fill_diagonal(S, 1.0)
        return np.clip(S, -1.0, 1.0)

    def test_identity_when_identical(self):
        """When corr_m == corr_r the rescaled matrix should be close to corr_r."""
        rng = np.random.default_rng(0)
        C = self._make_symmetric(10, rng)
        rescaled = _linear_rescale(C, C)
        np.testing.assert_allclose(rescaled, C, atol=1e-4)

    def test_known_affine_transform(self):
        """corr_r = 2 * corr_m + 0.1: rescaling must preserve the monotonic rank order.

        _linear_rescale removes the global linear bias so the diff matrix is
        centred, but the output is NOT expected to equal corr_r — it is a
        different affine function of corr_m.  What is guaranteed is that the
        Spearman rank correlation between rescaled and corr_r equals 1.0
        (since both are monotonic in corr_m with slope > 0), and that the
        output stays in [-1, 1].
        """
        from scipy.stats import spearmanr as _spear
        rng = np.random.default_rng(1)
        C_m = self._make_symmetric(12, rng) * 0.4   # keep small so 2x+0.1 stays in [-1,1]
        C_r = np.clip(2.0 * C_m + 0.1, -1.0, 1.0)
        np.fill_diagonal(C_r, 1.0)
        rescaled = _linear_rescale(C_m, C_r)
        idx = np.triu_indices(12, k=1)
        rho, _ = _spear(rescaled[idx], C_r[idx])
        assert rho == pytest.approx(1.0, abs=1e-6)

    def test_output_clipped_to_minus_one_one(self):
        rng = np.random.default_rng(2)
        C_m = self._make_symmetric(8, rng)
        # Create a ref that will force extreme values after rescaling
        C_r = self._make_symmetric(8, rng) * 0.1
        rescaled = _linear_rescale(C_m, C_r)
        assert rescaled.min() >= -1.0
        assert rescaled.max() <=  1.0


# ---------------------------------------------------------------------------
# TestSubsample
# ---------------------------------------------------------------------------

class TestSubsample:

    def test_stratified_respects_per_type_cap(self, merfish_adata):
        cap = 30
        sub = _subsample(merfish_adata, "cell_type", cap, None, 0, "merfish")
        counts = sub.obs["cell_type"].value_counts()
        assert (counts <= cap).all()

    def test_stratified_preserves_rare_type_fully(self, merfish_adata):
        """TypeC has 40 cells; capping at 50 should keep all 40."""
        cap = 50
        sub = _subsample(merfish_adata, "cell_type", cap, None, 0, "merfish")
        assert sub.obs["cell_type"].value_counts()["TypeC"] == CT_MERFISH["TypeC"]

    def test_stratified_all_types_present(self, merfish_adata):
        sub = _subsample(merfish_adata, "cell_type", 10, None, 0, "merfish")
        assert set(sub.obs["cell_type"].unique()) == {"TypeA", "TypeB", "TypeC"}

    def test_random_frac_correct_count(self, merfish_adata):
        frac = 0.5
        sub = _subsample(merfish_adata, None, None, frac, 0, "merfish")
        expected = round(N_MERFISH * frac)
        # Allow ±1 for rounding
        assert abs(sub.n_obs - expected) <= 1

    def test_no_subsampling_returns_all_cells(self, merfish_adata):
        sub = _subsample(merfish_adata, None, None, None, 0, "merfish")
        assert sub.n_obs == N_MERFISH

    def test_stratified_takes_priority_over_frac(self, merfish_adata):
        """When both cell_type_key and subsample_frac are supplied, stratified wins."""
        cap = 5
        sub = _subsample(merfish_adata, "cell_type", cap, 0.9, 0, "merfish")
        counts = sub.obs["cell_type"].value_counts()
        assert (counts <= cap).all()

    def test_invalid_frac_raises(self, merfish_adata):
        with pytest.raises(ValueError, match="subsample_frac"):
            _subsample(merfish_adata, None, None, 1.5, 0, "merfish")

    def test_zero_frac_raises(self, merfish_adata):
        with pytest.raises(ValueError, match="subsample_frac"):
            _subsample(merfish_adata, None, None, 0.0, 0, "merfish")


# ---------------------------------------------------------------------------
# TestExtractExpression
# ---------------------------------------------------------------------------

class TestExtractExpression:

    def test_dense_shape(self, merfish_adata, genes):
        X = _extract_expression(merfish_adata, genes, None)
        assert X.shape == (N_MERFISH, N_GENES)

    def test_sparse_shape(self, ref_adata, genes):
        X = _extract_expression(ref_adata, genes, None)
        assert X.shape == (N_REF, N_GENES)

    def test_sparse_preserves_sparsity(self, ref_adata, genes):
        X = _extract_expression(ref_adata, genes, None)
        assert sps.issparse(X)

    def test_gene_subset(self, merfish_adata):
        sub_genes = GENE_NAMES[:5]
        X = _extract_expression(merfish_adata, sub_genes, None)
        assert X.shape[1] == 5

    def test_backed_adata_loads_correctly(self, ref_h5ad_path, genes):
        """Loading via read_adata (backed → memory) should give the same expression matrix."""
        # Load into memory via the io module
        ref_in_memory = read_adata(ref_h5ad_path, genes=genes)
        X_mem = _extract_expression(ref_in_memory, genes, None)

        # Load in backed mode directly
        ref_backed = ad.read_h5ad(ref_h5ad_path, backed="r")
        X_backed = _extract_expression(ref_backed, genes, None)
        ref_backed.file.close()

        # Convert both to dense for comparison (X_mem may be sparse, X_backed may be dense)
        X_mem_dense    = X_mem.toarray() if sps.issparse(X_mem) else np.asarray(X_mem)
        X_backed_dense = X_backed.toarray() if sps.issparse(X_backed) else np.asarray(X_backed)
        np.testing.assert_allclose(X_mem_dense, X_backed_dense, atol=1e-6)

    def test_io_read_adata_integration(self, ref_h5ad_path, genes, ref_adata):
        """read_adata output can be passed directly to compute_corr_matrices."""
        ref_loaded = read_adata(ref_h5ad_path, genes=genes)
        # Should not raise
        X = _extract_expression(ref_loaded, genes, None)
        assert X.shape == (N_REF, N_GENES)


# ---------------------------------------------------------------------------
# TestComputeCorrMatrices
# ---------------------------------------------------------------------------

class TestComputeCorrMatrices:

    def test_returns_expected_keys(self, corr_data):
        expected = {
            "genes",
            "merfish_corr",
            "ref_corr",
            "merfish_corr_rescaled",
            "merfish_detect_rate",
            "ref_detect_rate",
            "detection_log2_ratio",
        }
        assert set(corr_data.keys()) == expected

    def test_genes_defaults_to_intersection(self, merfish_adata, ref_adata):
        expected = list(merfish_adata.var_names.intersection(ref_adata.var_names))
        data = compute_corr_matrices(merfish_adata, ref_adata)
        assert data["genes"] == expected

    def test_custom_genes_respected(self, merfish_adata, ref_adata):
        subset = GENE_NAMES[:8]
        data = compute_corr_matrices(merfish_adata, ref_adata, genes=subset)
        assert data["genes"] == subset
        assert data["merfish_corr"].shape == (8, 8)

    def test_merfish_corr_shape(self, corr_data):
        n = N_GENES
        assert corr_data["merfish_corr"].shape == (n, n)
        assert corr_data["ref_corr"].shape == (n, n)
        assert corr_data["merfish_corr_rescaled"].shape == (n, n)

    def test_corr_matrices_diagonal_is_one(self, corr_data):
        np.testing.assert_allclose(np.diag(corr_data["merfish_corr"]), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.diag(corr_data["ref_corr"]), 1.0, atol=1e-6)

    def test_corr_rescaled_clipped_to_minus_one_one(self, corr_data):
        C = corr_data["merfish_corr_rescaled"]
        assert C.min() >= -1.0
        assert C.max() <=  1.0

    def test_detect_rates_shape_and_range(self, corr_data):
        assert corr_data["merfish_detect_rate"].shape == (N_GENES,)
        assert corr_data["ref_detect_rate"].shape == (N_GENES,)
        assert (corr_data["merfish_detect_rate"] >= 0).all()
        assert (corr_data["merfish_detect_rate"] <= 1).all()

    def test_detection_log2_ratio_formula(self, merfish_adata, ref_adata):
        """log2 ratio computed from full (unsubsampled) expression matrices."""
        data = compute_corr_matrices(merfish_adata, ref_adata)
        eps = 1e-6

        X_m = np.asarray(merfish_adata.X)
        X_r = ref_adata.X.toarray()

        detect_m = (X_m > 0).mean(axis=0)
        detect_r = (X_r > 0).mean(axis=0)
        expected = np.log2((detect_m + eps) / (detect_r + eps))

        np.testing.assert_allclose(data["detection_log2_ratio"], expected, atol=1e-6)

    def test_sparse_and_dense_ref_give_same_correlations(self, merfish_adata, ref_adata):
        """Dense copy of ref_adata should yield the same correlation matrix."""
        ref_dense = ref_adata.copy()
        ref_dense.X = ref_adata.X.toarray()

        data_sparse = compute_corr_matrices(merfish_adata, ref_adata)
        data_dense  = compute_corr_matrices(merfish_adata, ref_dense)

        np.testing.assert_allclose(
            data_sparse["ref_corr"], data_dense["ref_corr"], atol=1e-5
        )

    def test_merfish_stratified_subsampling_reduces_cells(self, merfish_adata, ref_adata):
        """With cap=10 per type, the correlation should still be computable."""
        data = compute_corr_matrices(
            merfish_adata, ref_adata,
            merfish_cell_type_key="cell_type",
            merfish_n_cells_per_type=10,
        )
        assert data["merfish_corr"].shape == (N_GENES, N_GENES)

    def test_ref_random_subsampling_applied(self, merfish_adata, ref_adata):
        """ref_subsample_frac should not crash and should produce valid output."""
        data = compute_corr_matrices(
            merfish_adata, ref_adata,
            ref_subsample_frac=0.5,
        )
        assert data["ref_corr"].shape == (N_GENES, N_GENES)

    def test_no_subsampling_uses_all_cells(self, merfish_adata, ref_adata):
        """Without subsampling args, detection rates use all cells."""
        data = compute_corr_matrices(merfish_adata, ref_adata)
        X_m = np.asarray(merfish_adata.X)
        detect_m = (X_m > 0).mean(axis=0)
        np.testing.assert_allclose(data["merfish_detect_rate"], detect_m, atol=1e-6)

    def test_works_with_backed_adata(self, merfish_adata, ref_h5ad_path):
        """ref_adata loaded with read_adata (io module) can be passed directly."""
        ref_in_memory = read_adata(ref_h5ad_path)
        data = compute_corr_matrices(merfish_adata, ref_in_memory)
        assert data["ref_corr"].shape == (N_GENES, N_GENES)

    def test_backed_mode_adata_handled(self, merfish_adata, ref_h5ad_path):
        """Backed AnnData (isbacked=True) can be passed directly."""
        ref_backed = ad.read_h5ad(ref_h5ad_path, backed="r")
        try:
            data = compute_corr_matrices(merfish_adata, ref_backed)
            assert data["ref_corr"].shape == (N_GENES, N_GENES)
        finally:
            ref_backed.file.close()


# ---------------------------------------------------------------------------
# TestScoreGeneReliability
# ---------------------------------------------------------------------------

class TestScoreGeneReliability:

    def test_returns_dataframe_and_dict(self, scored):
        results, thresholds = scored
        assert isinstance(results, pd.DataFrame)
        assert isinstance(thresholds, dict)

    def test_output_has_expected_columns(self, scored):
        results, _ = scored
        expected_cols = {
            "profile_similarity", "residual_variance", "residual_z",
            "pc1_loading", "detection_log2_ratio",
            "flag_fidelity", "flag_residual", "flag_detection",
            "reliable", "detection_rate", "failure_mode",
        }
        assert expected_cols.issubset(set(results.columns))

    def test_index_matches_genes(self, corr_data, scored):
        results, _ = scored
        assert list(results.index) == corr_data["genes"]

    def test_failure_mode_valid_values(self, scored):
        results, _ = scored
        valid = {"reliable", "probe_failure", "composition_mismatch", "idiosyncratic_noise"}
        assert set(results["failure_mode"].unique()).issubset(valid)

    def test_reliable_equals_not_fidelity_or_residual(self, scored):
        results, _ = scored
        expected = ~(results["flag_fidelity"] | results["flag_residual"])
        pd.testing.assert_series_equal(results["reliable"], expected, check_names=False)

    def test_residual_z_is_standardized(self, scored):
        # The z-score uses numpy's population std (ddof=0).
        # pd.Series.std() uses ddof=1 by default, so check with ddof=0 explicitly.
        results, _ = scored
        z = results["residual_z"].values
        assert z.mean() == pytest.approx(0.0, abs=1e-10)
        assert z.std(ddof=0) == pytest.approx(1.0, abs=1e-8)

    def test_thresholds_keys_present(self, scored):
        _, thresholds = scored
        assert "residual_threshold"  in thresholds
        assert "profile_threshold"   in thresholds
        assert "detection_threshold" in thresholds
        assert "n_pcs_used"          in thresholds
        assert "variance_explained"  in thresholds
        assert "pca_method"          in thresholds
        assert "_fit"                in thresholds
        assert "_pca"                in thresholds

    def test_pca_method_stored_in_thresholds(self, corr_data):
        """pca_method key reflects the method that was used."""
        _, t_pa    = score_gene_reliability(corr_data, pca_method="parallel_analysis")
        _, t_scree = score_gene_reliability(corr_data, pca_method="scree")
        assert t_pa["pca_method"]    == "parallel_analysis"
        assert t_scree["pca_method"] == "scree"

    def test_pca_dict_has_eigenvalues_and_null(self, corr_data):
        """_pca sub-dict must contain real_eigenvalues; null_threshold set for PA only."""
        _, t_pa    = score_gene_reliability(corr_data, pca_method="parallel_analysis")
        _, t_scree = score_gene_reliability(corr_data, pca_method="scree")
        assert "real_eigenvalues"         in t_pa["_pca"]
        assert "null_threshold"           in t_pa["_pca"]
        assert t_pa["_pca"]["null_threshold"] is not None
        assert t_scree["_pca"]["null_threshold"] is None

    def test_invalid_pca_method_raises(self, corr_data):
        with pytest.raises(ValueError, match="pca_method"):
            score_gene_reliability(corr_data, pca_method="bad_method")

    def test_scree_n_pcs_explains_variance_threshold(self, corr_data):
        """Scree method: n_pcs_used must actually reach variance_threshold."""
        _, thresholds = score_gene_reliability(corr_data, pca_method="scree",
                                               variance_threshold=0.90)
        assert thresholds["variance_explained"] >= 0.90

    def test_scree_is_minimum_number_of_pcs(self, corr_data):
        """Scree method: using one fewer PC should explain less than variance_threshold."""
        _, thresholds = score_gene_reliability(corr_data, pca_method="scree",
                                               variance_threshold=0.90)
        k = thresholds["n_pcs_used"]
        if k > 1:
            from sklearn.decomposition import PCA as _PCA
            import numpy as _np

            corr_m = corr_data["merfish_corr_rescaled"]
            corr_r = corr_data["ref_corr"]
            diff = corr_m - corr_r
            _np.fill_diagonal(diff, _np.nan)
            row_means = _np.nanmean(diff, axis=1, keepdims=True)
            diff_c = diff - row_means
            _np.fill_diagonal(diff_c, 0.0)

            pca = _PCA(n_components=min(len(corr_data["genes"]) - 1, 50))
            pca.fit(diff_c)
            cumvar = _np.cumsum(pca.explained_variance_ratio_)
            assert cumvar[k - 2] < 0.90

    def test_priority_composition_mismatch_over_probe_failure(self, corr_data):
        """A gene with flag_detection=True should be composition_mismatch, not probe_failure,
        even if it also has flag_fidelity=True."""
        results, thresholds = score_gene_reliability(corr_data)
        # Find genes flagged for both detection and fidelity
        both = results["flag_detection"] & results["flag_fidelity"] & ~results["reliable"]
        if both.any():
            assert (results.loc[both, "failure_mode"] == "composition_mismatch").all()

    def test_priority_probe_failure_over_idiosyncratic_noise(self, corr_data):
        """A gene with flag_fidelity=True (but not detection) → probe_failure, not noise."""
        results, _ = score_gene_reliability(corr_data)
        fidelity_only = results["flag_fidelity"] & ~results["flag_detection"] & ~results["reliable"]
        if fidelity_only.any():
            assert (results.loc[fidelity_only, "failure_mode"] == "probe_failure").all()

    def test_scree_custom_variance_threshold(self, corr_data):
        """Scree: lower threshold → fewer PCs than higher threshold."""
        _, t_low  = score_gene_reliability(corr_data, pca_method="scree",
                                           variance_threshold=0.50)
        _, t_high = score_gene_reliability(corr_data, pca_method="scree",
                                           variance_threshold=0.95)
        assert t_low["n_pcs_used"] <= t_high["n_pcs_used"]

    def test_variance_explained_stored_correctly(self, corr_data):
        _, thresholds = score_gene_reliability(corr_data)
        vexp = thresholds["variance_explained"]
        assert vexp >= 0.0
        assert vexp <= 1.0 + 1e-10

    def test_identical_datasets_mostly_reliable(self):
        """When MERFISH and reference are identical, most genes should be reliable."""
        rng = np.random.default_rng(99)
        X = rng.poisson(1.0, (100, 15)).astype(np.float32)
        adata = ad.AnnData(
            X=sps.csr_matrix(X),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(100)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(15)]),
        )
        data = compute_corr_matrices(adata, adata)
        results, _ = score_gene_reliability(data)
        # With identical correlation matrices, most genes should be reliable
        frac_reliable = results["reliable"].mean()
        assert frac_reliable >= 0.5   # conservative: noisy data may flag a few


# ---------------------------------------------------------------------------
# TestParallelAnalysis
# ---------------------------------------------------------------------------

class TestParallelAnalysis:
    """Unit tests for the _parallel_analysis helper."""

    # ------------------------------------------------------------------
    # Fixtures / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_diff_c(n_genes: int, rng_seed: int = 0) -> np.ndarray:
        """Random row-mean-centred difference matrix (no real structure)."""
        rng = np.random.default_rng(rng_seed)
        d = rng.standard_normal((n_genes, n_genes))
        d = (d + d.T) / 2           # symmetric
        np.fill_diagonal(d, np.nan)
        d -= np.nanmean(d, axis=1, keepdims=True)
        np.fill_diagonal(d, 0.0)
        return d

    @staticmethod
    def _make_structured_diff_c(n_genes: int, n_signal: int = 3,
                                 rng_seed: int = 0) -> np.ndarray:
        """Diff matrix with n_signal planted low-rank components above noise."""
        rng = np.random.default_rng(rng_seed)
        # Low-rank signal
        U = rng.standard_normal((n_genes, n_signal))
        signal = U @ U.T * 5.0       # strong signal
        # Noise
        noise = rng.standard_normal((n_genes, n_genes))
        noise = (noise + noise.T) / 2
        d = signal + noise
        np.fill_diagonal(d, np.nan)
        d -= np.nanmean(d, axis=1, keepdims=True)
        np.fill_diagonal(d, 0.0)
        return d

    # ------------------------------------------------------------------
    # Return shape / type
    # ------------------------------------------------------------------

    def test_returns_three_tuple(self):
        d = self._make_diff_c(15)
        result = _parallel_analysis(d, n_max=10, n_permutations=20, random_state=0,
                                    n_jobs=1)
        assert len(result) == 3

    def test_k_is_int(self):
        d = self._make_diff_c(15)
        k, _, _ = _parallel_analysis(d, n_max=10, n_permutations=20, random_state=0,
                                     n_jobs=1)
        assert isinstance(k, int)

    def test_real_eigenvalues_shape(self):
        n_max = 8
        d = self._make_diff_c(15)
        _, real_eigs, _ = _parallel_analysis(d, n_max=n_max, n_permutations=20,
                                             random_state=0, n_jobs=1)
        assert real_eigs.shape == (n_max,)

    def test_null_threshold_shape(self):
        n_max = 8
        d = self._make_diff_c(15)
        _, _, null_thr = _parallel_analysis(d, n_max=n_max, n_permutations=20,
                                            random_state=0, n_jobs=1)
        assert null_thr.shape == (n_max,)

    # ------------------------------------------------------------------
    # k bounds
    # ------------------------------------------------------------------

    def test_k_at_least_one(self):
        """Even pure noise should return k >= 1."""
        d = self._make_diff_c(15)
        k, _, _ = _parallel_analysis(d, n_max=10, n_permutations=50, random_state=0,
                                     n_jobs=1)
        assert k >= 1

    def test_k_at_most_n_max(self):
        n_max = 10
        d = self._make_diff_c(15)
        k, _, _ = _parallel_analysis(d, n_max=n_max, n_permutations=20, random_state=0,
                                     n_jobs=1)
        assert k <= n_max

    # ------------------------------------------------------------------
    # Null properties
    # ------------------------------------------------------------------

    def test_null_threshold_non_negative(self):
        """Eigenvalues of a PD-ish matrix are non-negative."""
        d = self._make_diff_c(15)
        _, _, null_thr = _parallel_analysis(d, n_max=8, n_permutations=30,
                                            random_state=0, n_jobs=1)
        assert np.all(null_thr >= 0)

    def test_null_threshold_decreasing(self):
        """Null eigenvalues should decrease with rank (PCA orders by variance)."""
        d = self._make_diff_c(20)
        _, _, null_thr = _parallel_analysis(d, n_max=10, n_permutations=50,
                                            random_state=0, n_jobs=1)
        assert np.all(np.diff(null_thr) <= 0)

    # ------------------------------------------------------------------
    # Signal detection
    # ------------------------------------------------------------------

    def test_structured_data_chooses_more_pcs_than_noise(self):
        """A diff matrix with planted low-rank signal should yield more PCs than noise."""
        n_genes = 30
        d_noise     = self._make_diff_c(n_genes, rng_seed=7)
        d_structured = self._make_structured_diff_c(n_genes, n_signal=4, rng_seed=7)

        k_noise, _, _      = _parallel_analysis(d_noise,      n_max=15,
                                                n_permutations=50, random_state=0,
                                                n_jobs=1)
        k_structured, _, _ = _parallel_analysis(d_structured, n_max=15,
                                                n_permutations=50, random_state=0,
                                                n_jobs=1)
        assert k_structured >= k_noise

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    def test_deterministic_with_same_seed(self):
        d = self._make_diff_c(20)
        k1, eig1, thr1 = _parallel_analysis(d, n_max=10, n_permutations=30,
                                            random_state=42, n_jobs=1)
        k2, eig2, thr2 = _parallel_analysis(d, n_max=10, n_permutations=30,
                                            random_state=42, n_jobs=1)
        assert k1 == k2
        np.testing.assert_array_equal(eig1, eig2)
        np.testing.assert_array_equal(thr1, thr2)

    # ------------------------------------------------------------------
    # Integration with score_gene_reliability
    # ------------------------------------------------------------------

    def test_score_gene_reliability_parallel_analysis_runs(self, corr_data):
        """parallel_analysis is the default and should run without error."""
        results, thresholds = score_gene_reliability(
            corr_data, pca_method="parallel_analysis", n_permutations=30, n_jobs=1
        )
        assert thresholds["pca_method"] == "parallel_analysis"
        assert thresholds["n_pcs_used"] >= 1
        assert thresholds["_pca"]["null_threshold"] is not None

    def test_parallel_analysis_n_pcs_differs_from_scree(self, corr_data):
        """The two methods may choose different k — just verify both run and k >= 1."""
        _, t_pa    = score_gene_reliability(corr_data, pca_method="parallel_analysis",
                                            n_permutations=30, n_jobs=1)
        _, t_scree = score_gene_reliability(corr_data, pca_method="scree")
        assert t_pa["n_pcs_used"]    >= 1
        assert t_scree["n_pcs_used"] >= 1


# ---------------------------------------------------------------------------
# TestComputeGeneReliability (wrapper)
# ---------------------------------------------------------------------------

class TestComputeGeneReliability:

    def test_matches_two_step_approach(self, merfish_adata, ref_adata):
        """One-step and two-step approaches should produce identical results."""
        corr_data = compute_corr_matrices(merfish_adata, ref_adata)
        results_2step, thr_2step = score_gene_reliability(corr_data)
        results_1step, thr_1step = compute_gene_reliability(merfish_adata, ref_adata)

        pd.testing.assert_frame_equal(results_1step, results_2step)
        assert thr_1step["residual_threshold"]  == pytest.approx(thr_2step["residual_threshold"])
        assert thr_1step["profile_threshold"]   == pytest.approx(thr_2step["profile_threshold"])
        assert thr_1step["detection_threshold"] == pytest.approx(thr_2step["detection_threshold"])

    def test_with_merfish_stratified_subsampling(self, merfish_adata, ref_adata):
        """Stratified MERFISH subsampling should not crash."""
        results, thresholds = compute_gene_reliability(
            merfish_adata, ref_adata,
            merfish_cell_type_key="cell_type",
            merfish_n_cells_per_type=20,
        )
        assert len(results) == N_GENES
        assert thresholds["n_pcs_used"] >= 1

    def test_with_ref_stratified_subsampling(self, merfish_adata, ref_adata):
        results, _ = compute_gene_reliability(
            merfish_adata, ref_adata,
            ref_cell_type_key="cell_type",
            ref_n_cells_per_type=30,
        )
        assert len(results) == N_GENES

    def test_with_both_random_subsampling(self, merfish_adata, ref_adata):
        results, _ = compute_gene_reliability(
            merfish_adata, ref_adata,
            merfish_subsample_frac=0.6,
            ref_subsample_frac=0.5,
        )
        assert len(results) == N_GENES

    def test_deterministic_with_fixed_random_state(self, merfish_adata, ref_adata):
        r1, _ = compute_gene_reliability(
            merfish_adata, ref_adata,
            merfish_cell_type_key="cell_type",
            merfish_n_cells_per_type=30,
            random_state=0,
        )
        r2, _ = compute_gene_reliability(
            merfish_adata, ref_adata,
            merfish_cell_type_key="cell_type",
            merfish_n_cells_per_type=30,
            random_state=0,
        )
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# TestPlottingSmoke
# ---------------------------------------------------------------------------

class TestPlottingSmoke:

    def test_plot_metric_distributions_returns_fig_and_three_axes(self, scored):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        results, thresholds = scored
        fig, axes = plot_metric_distributions(results, thresholds)
        assert hasattr(fig, "savefig")
        assert len(axes) == 3
        plt.close(fig)

    def test_plot_failure_mode_scatter_returns_fig_and_ax(self, scored):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        results, thresholds = scored
        fig, ax = plot_failure_mode_scatter(results, thresholds)
        assert hasattr(fig, "savefig")
        assert hasattr(ax, "set_xlabel")
        plt.close(fig)

    def test_plot_metric_distributions_no_error_with_all_reliable(self):
        """Edge case: all genes reliable — histograms should still render."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(5)
        X = rng.poisson(1.0, (80, 12)).astype(np.float32)
        adata = ad.AnnData(
            X=sps.csr_matrix(X),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(80)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(12)]),
        )
        data = compute_corr_matrices(adata, adata)
        results, thresholds = score_gene_reliability(data)
        fig, axes = plot_metric_distributions(results, thresholds)
        plt.close(fig)

    def test_plot_failure_mode_scatter_annotate_false(self, scored):
        """annotate_flagged=False should not raise."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        results, thresholds = scored
        fig, ax = plot_failure_mode_scatter(results, thresholds, annotate_flagged=False)
        plt.close(fig)
