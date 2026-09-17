"""
Smoke tests for visualization functions.

Each test checks:
  - the function runs without error
  - it returns a 2-tuple whose first element is a matplotlib Figure

Uses the small_adata fixture (synthetic, 150 cells × 200 genes, 3 cell types)
so these tests run fast without reference data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for CI
import matplotlib.figure

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pygenebasis.pl._core import (
    plot_mapping_heatmap,
    plot_expression_heatmap,
    plot_coexpression,
    plot_redundancy_stat,
    plot_umaps_w_counts,
)


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

def _is_fig(obj) -> bool:
    return isinstance(obj, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# plot_mapping_heatmap
# ---------------------------------------------------------------------------

class TestPlotMappingHeatmap:
    def _make_mapping(self, n_celltypes: int = 3, n_cells: int = 90) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        celltypes = [f"Type{chr(65 + i)}" for i in range(n_celltypes)]
        true_ct = np.repeat(celltypes, n_cells // n_celltypes)
        # ~80% correct predictions
        predicted = true_ct.copy()
        flip = rng.choice(len(predicted), size=len(predicted) // 5, replace=False)
        predicted[flip] = rng.choice(celltypes, size=len(flip))
        return pd.DataFrame({
            "cell": [f"cell_{i}" for i in range(len(true_ct))],
            "celltype": true_ct,
            "mapped_celltype": predicted,
        })

    def test_returns_fig_ax(self):
        mapping = self._make_mapping()
        result = plot_mapping_heatmap(mapping)
        assert isinstance(result, tuple) and len(result) == 2
        assert _is_fig(result[0])

    def test_with_title(self):
        mapping = self._make_mapping()
        fig, ax = plot_mapping_heatmap(mapping, title="Test panel")
        assert _is_fig(fig)
        assert ax.get_title() == "Test panel"

    def test_many_celltypes(self):
        mapping = self._make_mapping(n_celltypes=20, n_cells=200)
        fig, ax = plot_mapping_heatmap(mapping)
        assert _is_fig(fig)


# ---------------------------------------------------------------------------
# plot_expression_heatmap
# ---------------------------------------------------------------------------

class TestPlotExpressionHeatmap:
    def test_mean_mode(self, small_adata):
        genes = small_adata.var_names[:20].tolist()
        result = plot_expression_heatmap(small_adata, genes)
        assert isinstance(result, tuple) and len(result) == 2
        assert _is_fig(result[0])

    def test_frac_mode(self, small_adata):
        genes = small_adata.var_names[:20].tolist()
        fig, ax = plot_expression_heatmap(small_adata, genes, value_type="frac")
        assert _is_fig(fig)

    def test_missing_genes_ignored(self, small_adata):
        genes = small_adata.var_names[:10].tolist() + ["NOT_A_GENE"]
        fig, ax = plot_expression_heatmap(small_adata, genes)
        assert _is_fig(fig)

    def test_custom_celltype_key(self, small_adata):
        genes = small_adata.var_names[:10].tolist()
        fig, ax = plot_expression_heatmap(
            small_adata, genes, celltype_key="celltype"
        )
        assert _is_fig(fig)


# ---------------------------------------------------------------------------
# plot_coexpression
# ---------------------------------------------------------------------------

class TestPlotCoexpression:
    def test_returns_fig_ax(self, small_adata):
        genes = small_adata.var_names[:15].tolist()
        result = plot_coexpression(small_adata, genes)
        assert isinstance(result, tuple) and len(result) == 2
        assert _is_fig(result[0])

    def test_with_title(self, small_adata):
        genes = small_adata.var_names[:10].tolist()
        fig, ax = plot_coexpression(small_adata, genes, title="Coex")
        assert ax.get_title() == "Coex"

    def test_missing_genes_ignored(self, small_adata):
        genes = small_adata.var_names[:10].tolist() + ["FAKE"]
        fig, ax = plot_coexpression(small_adata, genes)
        assert _is_fig(fig)


# ---------------------------------------------------------------------------
# plot_redundancy_stat
# ---------------------------------------------------------------------------

class TestPlotRedundancyStat:
    def _make_redundancy_df(
        self, n_genes: int = 10, n_celltypes: int = 3
    ) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        genes = [f"gene_{i}" for i in range(n_genes)]
        celltypes = [f"Type{chr(65 + i)}" for i in range(n_celltypes)]
        rows = []
        for g in genes:
            for ct in celltypes:
                ratio = float(rng.uniform(0.7, 1.1))
                rows.append({
                    "gene": g,
                    "celltype": ct,
                    "frac_correctly_mapped": ratio * 0.9,
                    "frac_correctly_mapped_all": 0.9,
                    "frac_correctly_mapped_ratio": ratio,
                })
        return pd.DataFrame(rows)

    def test_returns_fig_ax(self):
        df = self._make_redundancy_df()
        result = plot_redundancy_stat(df)
        assert isinstance(result, tuple) and len(result) == 2
        assert _is_fig(result[0])

    def test_many_genes(self):
        df = self._make_redundancy_df(n_genes=50, n_celltypes=8)
        fig, ax = plot_redundancy_stat(df)
        assert _is_fig(fig)


# ---------------------------------------------------------------------------
# plot_umaps_w_counts
# ---------------------------------------------------------------------------

class TestPlotUmapsWCounts:
    def _add_umap(self, adata):
        """Add a synthetic UMAP embedding to adata (in place)."""
        import anndata as ad
        rng = np.random.default_rng(99)
        adata = adata.copy()
        adata.obsm["X_umap"] = rng.standard_normal((adata.n_obs, 2)).astype(np.float32)
        return adata

    def test_returns_fig_axes(self, small_adata):
        adata = self._add_umap(small_adata)
        genes = small_adata.var_names[:6].tolist()
        result = plot_umaps_w_counts(adata, genes)
        assert isinstance(result, tuple) and len(result) == 2
        assert _is_fig(result[0])

    def test_ncol_param(self, small_adata):
        adata = self._add_umap(small_adata)
        genes = small_adata.var_names[:9].tolist()
        fig, axes = plot_umaps_w_counts(adata, genes, ncol=3)
        assert _is_fig(fig)
        assert axes.shape == (3, 3)

    def test_single_gene(self, small_adata):
        adata = self._add_umap(small_adata)
        genes = [small_adata.var_names[0]]
        fig, axes = plot_umaps_w_counts(adata, genes)
        assert _is_fig(fig)

    def test_missing_genes_raises(self, small_adata):
        adata = self._add_umap(small_adata)
        with pytest.raises(ValueError, match="No requested genes"):
            plot_umaps_w_counts(adata, ["FAKE_GENE_1", "FAKE_GENE_2"])
