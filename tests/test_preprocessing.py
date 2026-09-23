"""
Tests for pygenebasis.preprocessing.retain_informative_genes.

Testing strategy for HVG reference comparison
----------------------------------------------
The default flavor="scran" uses scranpy.model_gene_variances, which reimplements
scran::modelGeneVar in Python.  This gives ≥80% overlap with R's HVG list on
both reference datasets (mouse: ~90%, BG SN: ~87%).

For ALL downstream tests (graph, evaluation, selection, mapping), we use R's
HVG list as input — not Python's.  This isolates algorithmic differences in
each downstream step from HVG selection differences, giving a cleaner comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from pygenebasis import retain_informative_genes, highly_variable_methylation_features
from helpers import SPEARMAN_TOL, overlap_fraction


@pytest.fixture
def meth_adata():
    """Synthetic mCH rate matrix with per-gene mean coverage in .var."""
    rng = np.random.default_rng(7)
    n_cell, n_gene = 200, 400
    gene_mean = rng.uniform(0.02, 0.95, size=n_gene)
    noise = rng.uniform(0.02, 0.35, size=n_gene)
    X = np.clip(rng.normal(gene_mean, noise, size=(n_cell, n_gene)), 0, 1)
    cov = rng.lognormal(3.2, 0.8, size=n_gene)
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cell)]),
        var=pd.DataFrame({"cov_mean": cov}, index=[f"g{i}" for i in range(n_gene)]),
    )


class TestRetainInformativeGenesContract:

    def test_returns_fewer_genes(self, small_adata):
        filtered = retain_informative_genes(small_adata)
        assert filtered.n_vars < small_adata.n_vars

    def test_no_mt_genes_when_discard_mt(self, small_adata):
        import anndata as ad
        import numpy as np
        import pandas as pd
        import scipy.sparse
        # inject some fake MT genes
        n = small_adata.n_obs
        mt_var = pd.DataFrame(index=["MT-gene1", "MT-gene2", "mt-gene3"])
        mt_X = scipy.sparse.csr_matrix(np.ones((n, 3), dtype=np.float32))
        mt_adata = ad.AnnData(X=mt_X, obs=small_adata.obs.copy(), var=mt_var)
        import anndata
        combined = anndata.concat([small_adata, mt_adata], axis=1)
        filtered = retain_informative_genes(combined, discard_mt=True)
        for gene in filtered.var_names:
            assert not gene.startswith(("MT-", "mt-", "Mt-")), (
                f"MT gene {gene!r} survived filtering"
            )

    def test_mt_genes_kept_when_not_discarding(self, small_adata):
        import anndata as ad
        import numpy as np
        import pandas as pd
        import scipy.sparse
        n = small_adata.n_obs
        mt_var = pd.DataFrame(index=["MT-gene1", "MT-gene2"])
        mt_X = scipy.sparse.csr_matrix(
            np.full((n, 2), fill_value=5.0, dtype=np.float32)
        )
        mt_adata = ad.AnnData(X=mt_X, obs=small_adata.obs.copy(), var=mt_var)
        import anndata
        combined = anndata.concat([small_adata, mt_adata], axis=1)
        # high expression → should be HVG; discard_mt=False keeps them
        filtered = retain_informative_genes(combined, discard_mt=False)
        mt_retained = [g for g in filtered.var_names if g.startswith("MT-")]
        assert len(mt_retained) > 0

    def test_inplace_false_returns_copy(self, small_adata):
        filtered = retain_informative_genes(small_adata, inplace=False)
        assert filtered is not small_adata

    def test_inplace_true_returns_none(self, small_adata):
        import copy
        adata_copy = copy.copy(small_adata)
        result = retain_informative_genes(adata_copy, inplace=True)
        assert result is None

    def test_n_parameter_respected(self, small_adata):
        filtered = retain_informative_genes(small_adata, n=10)
        assert filtered.n_vars <= 10

    def test_raises_without_X(self):
        import anndata as ad
        import pandas as pd
        empty = ad.AnnData(
            obs=pd.DataFrame(index=["c1"]),
            var=pd.DataFrame(index=["g1"]),
        )
        with pytest.raises(Exception):
            retain_informative_genes(empty)

    def test_retains_fewer_genes_than_full(self, gene_search_adata):
        """retain_informative_genes should reduce the gene count substantially."""
        filtered = retain_informative_genes(gene_search_adata)
        assert filtered.n_vars < gene_search_adata.n_vars, (
            "retain_informative_genes should discard at least some genes"
        )
        # Should retain less than 90% of genes (meaningful filtering)
        assert filtered.n_vars < int(0.90 * gene_search_adata.n_vars), (
            f"Too few genes discarded: retained {filtered.n_vars}/{gene_search_adata.n_vars}"
        )


@pytest.mark.mouse
class TestRetainInformativeGenesMouseReference:

    def test_high_overlap_with_r_hvgs(self, adata_mouse, r_mouse):
        r_hvgs = set(r_mouse["hvg_genes"]["gene"].tolist())
        filtered = retain_informative_genes(adata_mouse)
        py_hvgs = set(filtered.var_names.tolist())
        frac = len(r_hvgs & py_hvgs) / len(r_hvgs)
        # scranpy reimplements scran::modelGeneVar — expect ≥80% overlap with R.
        assert frac >= 0.80, (
            f"HVG overlap with R only {frac:.3f} (expected ≥0.80)"
        )


@pytest.mark.bg_sn
class TestRetainInformativeGenesBgSnReference:

    def test_high_overlap_with_r_hvgs(self, adata_bg_sn, r_bg_sn):
        r_hvgs = set(r_bg_sn["hvg_genes"]["gene"].tolist())
        filtered = retain_informative_genes(adata_bg_sn)
        py_hvgs = set(filtered.var_names.tolist())
        frac = len(r_hvgs & py_hvgs) / len(r_hvgs)
        assert frac >= 0.80, (
            f"BG SN HVG overlap with R only {frac:.3f} (expected ≥0.80)"
        )


class TestHighlyVariableMethylationFeatures:
    """Port of ALLCools highly_variable_methylation_feature.

    Verified column-for-column against the upstream function on identical input
    (including the merged bin assignments); these guard the port from drifting.
    """

    def test_returns_expected_columns(self, meth_adata):
        df = highly_variable_methylation_features(meth_adata)
        assert list(df.index) == list(meth_adata.var_names)
        assert {"mean", "dispersion", "cov", "mean_bin", "cov_bin",
                "dispersion_norm", "feature_select"}.issubset(df.columns)

    def test_missing_coverage_key_is_explicit(self, meth_adata):
        del meth_adata.var["cov_mean"]
        with pytest.raises(KeyError, match="add_feature_cov_mean"):
            highly_variable_methylation_features(meth_adata)

    def test_n_top_feature_is_honoured(self, meth_adata):
        """Upstream hardcodes 5000 and ignores the argument; ours must not."""
        for n in (50, 150):
            df = highly_variable_methylation_features(meth_adata, n_top_feature=n)
            assert df["feature_select"].sum() == n

    def test_n_top_feature_takes_the_highest_dispersion_norm(self, meth_adata):
        df = highly_variable_methylation_features(meth_adata, n_top_feature=40)
        expected = set(df["dispersion_norm"].sort_values(ascending=False).index[:40])
        assert set(df.index[df["feature_select"]]) == expected

    def test_cutoff_mode_respects_mean_bounds(self, meth_adata):
        df = highly_variable_methylation_features(meth_adata, min_mean=0.3, max_mean=0.6)
        sel = df[df["feature_select"]]
        assert (sel["mean"] > 0.3).all() and (sel["mean"] < 0.6).all()

    def test_cutoff_mode_respects_min_disp(self, meth_adata):
        df = highly_variable_methylation_features(meth_adata, min_disp=1.0)
        assert (df.loc[df["feature_select"], "dispersion_norm"] > 1.0).all()

    def test_bin_merging_leaves_no_unnormalisable_bin(self, meth_adata):
        """Merging sparse bins is what keeps dispersion_norm finite."""
        df = highly_variable_methylation_features(meth_adata)
        assert np.isfinite(df["dispersion_norm"]).all()

    def test_binsize_changes_the_grouping(self, meth_adata):
        coarse = highly_variable_methylation_features(meth_adata, mean_binsize=0.5)
        fine = highly_variable_methylation_features(meth_adata, mean_binsize=0.02)
        assert coarse["mean_bin"].nunique() < fine["mean_bin"].nunique()

    def test_raises_when_no_bin_is_populated_enough(self, meth_adata):
        with pytest.raises(ValueError, match="larger bin size"):
            highly_variable_methylation_features(meth_adata, bin_min_features=10_000)

    def test_layer_is_used(self, meth_adata):
        meth_adata.layers["other"] = meth_adata.X[:, ::-1].copy()
        a = highly_variable_methylation_features(meth_adata)
        b = highly_variable_methylation_features(meth_adata, layer="other")
        assert not np.allclose(a["mean"].values, b["mean"].values)

    def test_retain_informative_genes_methylation_flavor(self, meth_adata):
        out = retain_informative_genes(
            meth_adata, flavor="methylation", n=60, discard_mt=False,
        )
        assert out.n_vars == 60
        expected = highly_variable_methylation_features(meth_adata, n_top_feature=60)
        assert set(out.var_names) == set(expected.index[expected["feature_select"]])

    def test_nan_rates_are_skipped_not_propagated(self, meth_adata):
        """Real rate matrices are NaN where a gene has no coverage in a cell.

        ALLCools runs on xarray, whose mean() skips NaN. Using numpy's mean here
        would make every gene NaN and select nothing.
        """
        rng = np.random.default_rng(1)
        X = meth_adata.X.copy()
        X[rng.random(X.shape) < 0.15] = np.nan
        meth_adata.X = X

        df = highly_variable_methylation_features(meth_adata, n_top_feature=40)
        assert np.isfinite(df["mean"]).all()
        assert np.isfinite(df["dispersion_norm"]).all()
        assert df["feature_select"].sum() == 40

    def test_nan_mean_matches_nanmean(self, meth_adata):
        """The per-gene mean ignores NaN cells, as xarray does."""
        X = meth_adata.X.copy()
        X[0, :] = np.nan
        meth_adata.X = X
        df = highly_variable_methylation_features(meth_adata)
        np.testing.assert_allclose(df["mean"].to_numpy(), np.nanmean(X, axis=0))
