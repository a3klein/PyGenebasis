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

import pytest

from pygenebasis import retain_informative_genes
from helpers import SPEARMAN_TOL, overlap_fraction


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
