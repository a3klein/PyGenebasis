"""
Visualization functions for pyGeneBasis.

All functions return (fig, ax) or (fig, axes) and do not call plt.show().
matplotlib only — no seaborn dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
import matplotlib.pyplot as plt


def plot_mapping_heatmap(
    mapping: pd.DataFrame,
    *,
    title: str | None = None,
):
    """Confusion matrix heatmap of cell type predictions vs ground truth.

    Rows = true cell type, columns = predicted cell type.
    Values are row-normalised (fraction of each true class predicted as each type).

    Parameters
    ----------
    mapping : pd.DataFrame
        Output of ``get_celltype_mapping``["mapping"].
        Columns: cell, celltype, mapped_celltype.
    title : str, optional

    Returns
    -------
    (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    ct_counts = pd.crosstab(mapping["celltype"], mapping["mapped_celltype"])
    all_types = sorted(set(ct_counts.index) | set(ct_counts.columns))
    ct_counts = ct_counts.reindex(index=all_types, columns=all_types, fill_value=0)

    row_sums = ct_counts.sum(axis=1).values[:, np.newaxis]
    ct_norm = ct_counts.values.astype(float) / np.where(row_sums == 0, 1, row_sums)

    n = len(all_types)
    cell_px = max(0.30, min(0.65, 18 / n))
    fig, ax = plt.subplots(figsize=(max(5, n * cell_px + 2), max(4, n * cell_px + 1)))
    im = ax.imshow(ct_norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Fraction")

    fs = max(5, min(9, 120 // n))
    ax.set_xticks(range(n))
    ax.set_xticklabels(all_types, rotation=90, fontsize=fs)
    ax.set_yticks(range(n))
    ax.set_yticklabels(all_types, fontsize=fs)
    ax.set_xlabel("Predicted cell type")
    ax.set_ylabel("True cell type")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_expression_heatmap(
    adata: AnnData,
    genes: list[str],
    *,
    celltype_key: str = "celltype",
    value_type: str = "mean",
    layer: str | None = None,
):
    """Heatmap of mean (or fraction non-zero) expression per cell type.

    Rows = cell types, columns = genes. Values are z-scored per gene column.

    Parameters
    ----------
    adata : AnnData
    genes : list[str]
    celltype_key : str
    value_type : {"mean", "frac"}
        "mean" — mean log-normalised expression per cell type (z-scored per gene).
        "frac" — fraction of cells with non-zero expression.
    layer : str, optional
        Layer to use instead of adata.X.

    Returns
    -------
    (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    genes_present = [g for g in genes if g in adata.var_names]
    src = adata[:, genes_present]
    X = src.layers[layer] if layer is not None else src.X
    if scipy.sparse.issparse(X):
        X = np.asarray(X.todense())

    celltypes = adata.obs[celltype_key].values.astype(str)
    unique_cts = sorted(np.unique(celltypes))

    matrix = np.zeros((len(unique_cts), len(genes_present)), dtype=np.float64)
    for i, ct in enumerate(unique_cts):
        mask = celltypes == ct
        if value_type == "frac":
            matrix[i] = (X[mask] > 0).mean(axis=0)
        else:
            matrix[i] = X[mask].mean(axis=0)

    # z-score per gene
    col_mean = matrix.mean(axis=0)
    col_std = matrix.std(axis=0)
    matrix_z = (matrix - col_mean) / np.where(col_std < 1e-12, 1.0, col_std)

    n_ct, n_g = matrix_z.shape
    fig, ax = plt.subplots(figsize=(max(5, n_g * 0.22 + 1.5), max(3, n_ct * 0.35 + 1)))
    im = ax.imshow(matrix_z, cmap="RdYlBu_r", aspect="auto", vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax, label="z-score")

    fs_x = max(5, min(8, 100 // max(n_g, 1)))
    fs_y = max(5, min(9, 100 // max(n_ct, 1)))
    ax.set_xticks(range(n_g))
    ax.set_xticklabels(genes_present, rotation=90, fontsize=fs_x)
    ax.set_yticks(range(n_ct))
    ax.set_yticklabels(unique_cts, fontsize=fs_y)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Cell type")
    fig.tight_layout()
    return fig, ax


def plot_coexpression(
    adata: AnnData,
    genes: list[str],
    *,
    title: str | None = None,
    layer: str | None = None,
):
    """Pairwise gene co-expression heatmap (Pearson correlation over cells).

    Parameters
    ----------
    adata : AnnData
    genes : list[str]
    title : str, optional
    layer : str, optional

    Returns
    -------
    (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    genes_present = [g for g in genes if g in adata.var_names]
    src = adata[:, genes_present]
    X = src.layers[layer] if layer is not None else src.X
    if scipy.sparse.issparse(X):
        X = np.asarray(X.todense())
    X = X.astype(np.float64)

    X_c = X - X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    X_n = X_c / std
    corr = (X_n.T @ X_n) / X.shape[0]
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    n = len(genes_present)
    size = max(4.0, n * 0.25)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    fs = max(5, min(8, 100 // max(n, 1)))
    ax.set_xticks(range(n))
    ax.set_xticklabels(genes_present, rotation=90, fontsize=fs)
    ax.set_yticks(range(n))
    ax.set_yticklabels(genes_present, fontsize=fs)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_redundancy_stat(redundancy_stat: pd.DataFrame):
    """Heatmap of per-gene, per-celltype redundancy (LOO mapping ratio).

    Genes sorted by mean ratio descending (most redundant = least important at top).
    Values ≈ 1 mean the gene is redundant for that cell type;
    values < 1 mean removing it hurts mapping accuracy.

    Parameters
    ----------
    redundancy_stat : pd.DataFrame
        Output of ``get_redundancy_stat``.
        Must contain columns: gene, celltype, frac_correctly_mapped_ratio.

    Returns
    -------
    (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    pivot = redundancy_stat.pivot_table(
        index="gene",
        columns="celltype",
        values="frac_correctly_mapped_ratio",
        aggfunc="mean",
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    n_genes, n_ct = pivot.shape
    fig, ax = plt.subplots(
        figsize=(max(4, n_ct * 0.38 + 1.5), max(3, n_genes * 0.28 + 1))
    )
    im = ax.imshow(pivot.values, vmin=0.6, vmax=1.1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Ratio (LOO / full panel)")

    fs_x = max(5, min(8, 100 // max(n_ct, 1)))
    fs_y = max(5, min(8, 100 // max(n_genes, 1)))
    ax.set_xticks(range(n_ct))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=fs_x)
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(pivot.index, fontsize=fs_y)
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Gene")
    fig.tight_layout()
    return fig, ax


def plot_umaps_w_counts(
    adata: AnnData,
    genes: list[str],
    *,
    umap_key: str = "X_umap",
    size: float = 0.25,
    ncol: int | None = None,
    layer: str | None = None,
):
    """Grid of UMAP plots coloured by expression of each selected gene.

    Parameters
    ----------
    adata : AnnData
        Must have UMAP coordinates in ``adata.obsm[umap_key]``.
    genes : list[str]
    umap_key : str
    size : float
        Point size.
    ncol : int, optional
        Number of columns in the grid. Defaults to min(5, n_genes).
    layer : str, optional

    Returns
    -------
    (fig, axes) : tuple[matplotlib.figure.Figure, numpy.ndarray of Axes]
        axes has shape (nrow, ncol).
    """
    genes_present = [g for g in genes if g in adata.var_names]
    if not genes_present:
        raise ValueError("No requested genes found in adata.var_names.")

    n = len(genes_present)
    if ncol is None:
        ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))

    umap_coords = adata.obsm[umap_key]
    src = adata[:, genes_present]
    X = src.layers[layer] if layer is not None else src.X
    if scipy.sparse.issparse(X):
        X = np.asarray(X.todense())

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.0, nrow * 2.8), squeeze=False)
    axes_flat = axes.flatten()

    for i, gene in enumerate(genes_present):
        ax = axes_flat[i]
        expr = np.asarray(X[:, i]).ravel()
        pos_vals = expr[expr > 0]
        vmax = float(np.percentile(pos_vals, 95)) if pos_vals.size > 0 else 1.0
        sc = ax.scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            c=expr, cmap="viridis", s=size,
            vmin=0.0, vmax=vmax, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        ax.set_title(gene, fontsize=9)
        ax.axis("off")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    return fig, axes
