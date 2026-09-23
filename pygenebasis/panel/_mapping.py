"""
Cell type classification and redundancy analysis.

get_celltype_mapping
    Assigns each cell a predicted cell type by majority vote (mode) from its k
    nearest neighbours in the selection graph.  Tie-breaking: choose the
    neighbour type with the smallest distance rank (matches geneBasisR).

get_redundancy_stat
    For each gene in the panel, temporarily removes it and measures the drop
    in cell type mapping accuracy per cell type.  Genes that cause a large
    drop when removed are non-redundant; genes that cause no drop are redundant.
    This function is embarrassingly parallel over genes and uses joblib.Parallel.

Both default to ``batch_method="harmony"``, deliberately diverging from
geneBasisR's MNN: our MNN is a from-scratch sklearn implementation that does not
scale (OOM-killed a 39k-cell Cla run at 133 GB, 2026-09-08).  Pass
``batch_method="mnn"`` for R parity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def get_celltype_mapping(
    adata: AnnData,
    genes_selection: list[str],
    *,
    celltype_key: str = "celltype",
    batch_key: str | None = None,
    n_neighbors: int = 5,
    n_pcs_selection: int | None = None,
    knn_method: str = "approx",
    batch_method: str = "harmony",
    return_stat: bool = True,
    layer: str | None = None,
    random_state: int = 32,
) -> dict:
    """Classify cells by majority vote from kNN neighbours.

    Matches geneBasisR::get_celltype_mapping.

    Parameters
    ----------
    adata : AnnData
        Must have ``celltype_key`` in ``adata.obs``.
    genes_selection : list[str]
        Gene panel for kNN graph construction.
    celltype_key : str
        Column in ``adata.obs`` with ground-truth cell type labels.
    batch_key : str, optional
    n_neighbors : int
    n_pcs_selection : int or None
        PCs for the selection graph. None skips PCA.
    knn_method : {"approx", "exact"}
    batch_method : {"harmony", "mnn"}
        Defaults to "harmony"; "mnn" matches geneBasisR but does not scale.
    return_stat : bool
        If True, also return per-cell-type accuracy (fraction correctly mapped).
    layer : str, optional
    random_state : int

    Returns
    -------
    dict with keys:
        "mapping" : pd.DataFrame, columns cell, celltype, mapped_celltype
        "stat"    : pd.DataFrame (only if return_stat=True),
                    columns celltype, frac_correctly_mapped
    """
    from ..knn._graph import build_knn_graph

    indices, distances = build_knn_graph(
        adata, genes_selection,
        batch_key=batch_key, knn_method=knn_method, batch_method=batch_method,
        n_neighbors=n_neighbors, n_pcs=n_pcs_selection, layer=layer,
        random_state=random_state,
    )

    true_labels = adata.obs[celltype_key].values.astype(str)
    cell_ids = adata.obs_names.tolist()

    # (n_cells, k) label matrix for each neighbour
    neighbor_labels = true_labels[indices]       # (n_cells, k)
    neighbor_distances = distances               # (n_cells, k)

    predicted = _mode_with_tiebreak(neighbor_labels, neighbor_distances)

    mapping_df = pd.DataFrame({
        "cell": cell_ids,
        "celltype": true_labels,
        "mapped_celltype": predicted,
    })

    result: dict = {"mapping": mapping_df}

    if return_stat:
        rows = []
        for ct in np.unique(true_labels):
            mask = true_labels == ct
            frac = (predicted[mask] == ct).mean()
            rows.append({"celltype": ct, "frac_correctly_mapped": float(frac)})
        result["stat"] = pd.DataFrame(rows)

    return result


def get_redundancy_stat(
    adata: AnnData,
    genes: list[str],
    *,
    genes_to_assess: list[str] | None = None,
    celltype_key: str = "celltype",
    batch_key: str | None = None,
    n_neighbors: int = 5,
    knn_method: str = "approx",
    batch_method: str = "harmony",
    n_jobs: int = -1,
    layer: str | None = None,
    random_state: int = 32,
) -> pd.DataFrame:
    """Assess redundancy of each gene by leave-one-out cell type mapping.

    For each gene g in genes_to_assess, runs get_celltype_mapping on
    (genes \\ {g}) and compares accuracy to the full panel.
    Parallelised over genes via joblib.Parallel.

    Parameters
    ----------
    adata : AnnData
    genes : list[str]
        Full selected gene panel.
    genes_to_assess : list[str], optional
        Subset of genes to assess. Defaults to all genes.
    celltype_key : str
    batch_key : str, optional
    n_neighbors : int
    knn_method, batch_method, layer, random_state : see get_celltype_mapping.
    n_jobs : int
        Number of parallel workers. -1 uses all available cores.

    Returns
    -------
    pd.DataFrame
        Columns: gene, celltype, frac_correctly_mapped (without gene),
        frac_correctly_mapped_all (with full panel),
        frac_correctly_mapped_ratio.
    """
    from joblib import Parallel, delayed

    if genes_to_assess is None:
        genes_to_assess = list(genes)

    # Accuracy with full panel (computed once)
    full_result = get_celltype_mapping(
        adata, genes,
        celltype_key=celltype_key, batch_key=batch_key,
        n_neighbors=n_neighbors, knn_method=knn_method, batch_method=batch_method,
        return_stat=True, layer=layer, random_state=random_state,
    )
    full_stat = full_result["stat"].set_index("celltype")["frac_correctly_mapped"]

    def _loo_one_gene(g: str) -> pd.DataFrame:
        genes_minus_g = [x for x in genes if x != g]
        result = get_celltype_mapping(
            adata, genes_minus_g,
            celltype_key=celltype_key, batch_key=batch_key,
            n_neighbors=n_neighbors, knn_method=knn_method, batch_method=batch_method,
            return_stat=True, layer=layer, random_state=random_state,
        )
        stat = result["stat"].copy()
        stat["gene"] = g
        stat["frac_correctly_mapped_all"] = stat["celltype"].map(full_stat)
        stat["frac_correctly_mapped_ratio"] = (
            stat["frac_correctly_mapped"] / stat["frac_correctly_mapped_all"]
        )
        return stat

    results = Parallel(n_jobs=n_jobs)(
        delayed(_loo_one_gene)(g) for g in genes_to_assess
    )

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _constrained_vote(
    indices: np.ndarray,
    distances: np.ndarray,
    true_labels: np.ndarray,
    parent_labels: np.ndarray,
) -> np.ndarray:
    """Hierarchy-constrained majority-vote cell type assignment.

    Like ``_mode_with_tiebreak`` but restricts eligible neighbours to those
    sharing the same parent-level label as the query cell.  Prevents
    cross-lineage contamination when predicting fine-grained cell types.

    Falls back to unconstrained majority vote for cells that have no valid
    neighbours after filtering (all k neighbours belong to a different parent).

    Parameters
    ----------
    indices : np.ndarray, shape (n_cells, k)
        kNN indices sorted by ascending distance.
    distances : np.ndarray, shape (n_cells, k)
        Distances to each neighbour (ascending).
    true_labels : np.ndarray, shape (n_cells,)
        Cell type labels at the level being predicted.
    parent_labels : np.ndarray, shape (n_cells,)
        Cell type labels at the parent (coarser) level.

    Returns
    -------
    predicted : np.ndarray, shape (n_cells,), dtype str
    """
    n_cells = indices.shape[0]
    predicted = np.empty(n_cells, dtype=object)

    for i in range(n_cells):
        neigh_idx = indices[i]
        valid = parent_labels[neigh_idx] == parent_labels[i]

        if not valid.any():
            predicted[i] = _mode_with_tiebreak(
                true_labels[neigh_idx].reshape(1, -1),
                distances[i : i + 1],
            )[0]
            continue

        valid_labels = true_labels[neigh_idx[valid]]
        unique, counts = np.unique(valid_labels, return_counts=True)
        max_count = counts.max()
        tied = unique[counts == max_count]

        if len(tied) == 1:
            predicted[i] = tied[0]
        else:
            best_rank = valid_labels.shape[0]
            best_type = tied[0]
            for ct in tied:
                first_rank = int(np.argmax(valid_labels == ct))
                if first_rank < best_rank:
                    best_rank = first_rank
                    best_type = ct
            predicted[i] = best_type

    return predicted.astype(str)


def _mode_with_tiebreak(
    neighbor_labels: np.ndarray,
    neighbor_distances: np.ndarray,
) -> np.ndarray:
    """Majority-vote cell type assignment with distance-rank tie-breaking.

    For each cell, counts votes for each cell type among its k neighbours.
    Ties are broken by preferring the cell type whose nearest representative
    neighbour has the smallest distance rank (i.e. among tied types, take
    the one whose closest neighbour appears earliest in the sorted neighbour
    list).  This matches geneBasisR's tie-breaking logic.

    Parameters
    ----------
    neighbor_labels : np.ndarray, shape (n_cells, k)
        Cell type label of each neighbour.
    neighbor_distances : np.ndarray, shape (n_cells, k)
        Distance to each neighbour (already sorted ascending by kNN).

    Returns
    -------
    predicted : np.ndarray, shape (n_cells,)
        Predicted cell type per cell.
    """
    n_cells, k = neighbor_labels.shape
    predicted = np.empty(n_cells, dtype=neighbor_labels.dtype)

    for i in range(n_cells):
        labels = neighbor_labels[i]          # (k,)
        unique_types, inverse, counts = np.unique(
            labels, return_inverse=True, return_counts=True
        )
        max_count = counts.max()
        tied = unique_types[counts == max_count]

        if len(tied) == 1:
            predicted[i] = tied[0]
        else:
            # Tie-breaking: for each tied type, find the rank of its first
            # occurrence in the distance-sorted neighbour list (already sorted).
            best_rank = k  # worst possible rank
            best_type = tied[0]
            for ct in tied:
                first_rank = int(np.argmax(labels == ct))  # first occurrence
                if first_rank < best_rank:
                    best_rank = first_rank
                    best_type = ct
            predicted[i] = best_type

    return predicted
