"""
Orchestration: run several metrics over one panel in a single call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def evaluate_library(
    adata: AnnData,
    genes_selection: list[str],
    *,
    genes_all: list[str] | None = None,
    batch_key: str | None = None,
    celltype_key: str = "celltype",
    n_neighbors: int = 5,
    n_pcs_all: int = 50,
    library_size_type: str = "single",
    n_genes_step: int = 10,
    return_cell_score_stat: bool = True,
    return_gene_score_stat: bool = True,
    return_celltype_stat: bool = True,
    knn_method: str = "approx",
    batch_method: str = "per_batch",
    layer: str | None = None,
    random_state: int = 32,
    verbose: bool = True,
) -> dict:
    """Comprehensive evaluation of a gene panel.

    Orchestrates cell scores, gene scores, and cell type mapping for a
    given panel, optionally across a range of panel sizes.

    Parameters
    ----------
    adata : AnnData
    genes_selection : list[str]
        The full selected panel (in selection order).
    genes_all : list[str], optional
    batch_key : str, optional
    celltype_key : str
        ``adata.obs`` column with cell type labels.
    library_size_type : {"single", "series"}
        "single" evaluates only the full panel.
        "series" evaluates at every n_genes_step increment.
    n_genes_step : int
        Step size for "series" mode.
    return_cell_score_stat : bool
    return_gene_score_stat : bool
    return_celltype_stat : bool
    knn_method, batch_method, layer, random_state, verbose : see gene_search.

    Returns
    -------
    dict with keys (present only if the corresponding flag is True):
        "cell_score_stat"  : pd.DataFrame, columns cell, cell_score, n_genes
        "gene_score_stat"  : pd.DataFrame, columns gene, gene_score, n_genes
        "celltype_stat"    : pd.DataFrame, columns celltype,
                             frac_correctly_mapped, n_genes
    """
    from ._mapping import get_celltype_mapping
    from ._neighborhood import (
        get_neighs_all_stat,
        get_neighborhood_preservation_scores,
        get_gene_prediction_scores,
    )

    # Determine panel sizes to evaluate
    n_total = len(genes_selection)
    if library_size_type == "single":
        sizes = [n_total]
    elif library_size_type == "series":
        sizes = list(range(n_genes_step, n_total + 1, n_genes_step))
        if n_total not in sizes:
            sizes.append(n_total)
    else:
        raise ValueError(
            f"library_size_type must be 'single' or 'series', got {library_size_type!r}"
        )

    # Pre-compute true-graph stat once (shared across panel sizes)
    neighs_all_stat = get_neighs_all_stat(
        adata, genes_all=genes_all, batch_key=batch_key,
        n_neighbors=n_neighbors, n_pcs_all=n_pcs_all,
        knn_method=knn_method, batch_method=batch_method,
        layer=layer, random_state=random_state,
    )

    cell_score_rows: list[pd.DataFrame] = []
    gene_score_rows: list[pd.DataFrame] = []
    celltype_rows: list[pd.DataFrame] = []

    for n in sizes:
        panel = genes_selection[:n]
        if verbose:
            print(f"Evaluating panel size {n}...")

        if return_cell_score_stat:
            cs = get_neighborhood_preservation_scores(
                adata, panel, genes_all=genes_all,
                batch_key=batch_key, n_neighbors=n_neighbors,
                n_pcs_all=n_pcs_all, n_pcs_selection=None,
                knn_method=knn_method, batch_method=batch_method,
                neighs_all_stat=neighs_all_stat,
                layer=layer, random_state=random_state,
            )
            cs["n_genes"] = n
            cell_score_rows.append(cs)

        if return_gene_score_stat:
            gs = get_gene_prediction_scores(
                adata, panel, genes_all=genes_all,
                batch_key=batch_key, n_neighbors=n_neighbors,
                n_pcs_all=n_pcs_all, n_pcs_selection=None,
                knn_method=knn_method, batch_method=batch_method,
                layer=layer, random_state=random_state,
            )
            gs["n_genes"] = n
            gene_score_rows.append(gs)

        if return_celltype_stat:
            ct = get_celltype_mapping(
                adata, panel, celltype_key=celltype_key,
                batch_key=batch_key, n_neighbors=n_neighbors,
                n_pcs_selection=None,
                knn_method=knn_method, batch_method=batch_method,
                return_stat=True, layer=layer, random_state=random_state,
            )["stat"]
            ct["n_genes"] = n
            celltype_rows.append(ct)

    result: dict = {}
    if return_cell_score_stat:
        result["cell_score_stat"] = pd.concat(cell_score_rows, ignore_index=True)
    if return_gene_score_stat:
        result["gene_score_stat"] = pd.concat(gene_score_rows, ignore_index=True)
    if return_celltype_stat:
        result["celltype_stat"] = pd.concat(celltype_rows, ignore_index=True)

    return result


