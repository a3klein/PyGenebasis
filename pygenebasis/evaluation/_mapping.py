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


def get_panel_celltype_accuracy(
    adata: AnnData,
    genes_selection: list[str],
    level_keys: list[str],
    *,
    constrained_method: str = "within_class",
    batch_key: str | list[str] | None = None,
    n_neighbors: int = 5,
    knn_method: str = "approx",
    batch_method: str = "harmony",
    n_pcs: int | None = None,
    n_cells_per_group: int | None = None,
    random_state: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute per-cell-type accuracy at every level of an annotation hierarchy.

    For each cell type at each level, two metrics are returned:

    ``constrained_accuracy``
        Fraction of cells correctly classified by majority vote.  For
        ``constrained_method="within_class"`` (default), a separate kNN graph
        is built within each parent class so that only within-lineage neighbours
        vote — preventing cross-lineage contamination at fine-grained levels.

    ``compounded_accuracy``
        Product of ``constrained_accuracy`` values along the hierarchy path from
        the coarsest level down to this cell type.  Represents the probability
        of a cell being correctly classified at every level of the hierarchy.

    At the coarsest level (no parent), ``constrained_accuracy == compounded_accuracy``.

    Parameters
    ----------
    adata : AnnData
        scRNA-seq reference.  Must have all ``level_keys`` columns in ``.obs``.
    genes_selection : list[str]
        Gene panel for kNN graph construction.
    level_keys : list[str]
        Annotation columns ordered coarsest → finest,
        e.g. ``["Class", "Subclass", "Group"]``.
    constrained_method : {"within_class", "global_filter", "none"}
        How to constrain the majority vote at levels below the coarsest:

        ``"within_class"`` *(default)*
            Build a separate kNN graph within each parent class subset.
            Every cell gets k valid voters from its own lineage.

        ``"global_filter"``
            Build one global kNN, restrict eligible voters at vote time to
            neighbours sharing the same parent-class label.  Matches the
            reliability module's constrained vote logic.

        ``"none"``
            Fully unconstrained majority vote at every level using the global
            kNN (equivalent to calling ``get_celltype_mapping`` per level).

    batch_key : str, list[str], or None
        Batch column(s).  If a list, a joint column is created.
    n_neighbors : int
    knn_method : {"approx", "exact"}
    batch_method : {"harmony", "mnn", "per_batch"}
        Default ``"harmony"`` (matches reliability module).  Ignored when
        ``batch_key`` is ``None`` (falls back to ``"per_batch"`` automatically).
    n_pcs : int or None
        PCA dimensionality for kNN.  ``None`` (default) skips PCA and builds
        the kNN in raw gene space — suitable for typical panel sizes.
    n_cells_per_group : int or None
        If set, stratified downsampling is applied before any kNN construction,
        keeping at most this many cells per finest-level group (``level_keys[-1]``).
        Groups with fewer cells are kept in full.  ``None`` (default) uses all
        cells.  Matches the subsampling behaviour of the reliability module.
    random_state : int
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Long-form, one row per ``(level, celltype)``.  Columns:
        ``level``, ``celltype``, ``constrained_accuracy``,
        ``compounded_accuracy``.

    Raises
    ------
    ValueError
        If any entry of ``level_keys`` is absent from ``adata.obs``,
        ``constrained_method`` is invalid, or the parent→child mapping is
        not 1-to-1 for any adjacent level pair.
    """
    import warnings
    from ..knn._graph import build_knn_graph
    from ..io._core import prepare_batch_key
    from ._mapping import _mode_with_tiebreak, _constrained_vote

    _VALID_METHODS = {"within_class", "global_filter", "none"}
    if constrained_method not in _VALID_METHODS:
        raise ValueError(
            f"constrained_method must be one of {sorted(_VALID_METHODS)}, "
            f"got {constrained_method!r}."
        )

    if len(level_keys) == 0:
        raise ValueError("level_keys must have at least one entry.")

    missing_keys = [k for k in level_keys if k not in adata.obs.columns]
    if missing_keys:
        raise ValueError(f"level_keys not found in adata.obs: {missing_keys}")

    # --- Validate 1-to-1 parent→child mapping; build lookup ---
    child_to_parent_maps: dict[int, dict[str, str]] = {}
    for li in range(1, len(level_keys)):
        parent_level = level_keys[li - 1]
        child_level  = level_keys[li]
        pairs = (
            adata.obs[[parent_level, child_level]]
            .astype(str)
            .drop_duplicates()
        )
        child_to_parents: dict[str, set] = {}
        for _, row in pairs.iterrows():
            child_to_parents.setdefault(row[child_level], set()).add(row[parent_level])

        non_unique = {c: p for c, p in child_to_parents.items() if len(p) > 1}
        if non_unique:
            details = "; ".join(
                f"'{c}' → {sorted(p)}" for c, p in non_unique.items()
            )
            raise ValueError(
                f"Non-1-to-1 parent→child mapping between '{parent_level}' and "
                f"'{child_level}': {details}. Each child cell type must belong "
                f"to exactly one parent."
            )
        child_to_parent_maps[li] = {
            child: next(iter(parents))
            for child, parents in child_to_parents.items()
        }

    # --- Resolve list batch_key → joint column ---
    adata, resolved_batch = prepare_batch_key(adata, batch_key)

    # --- Optional stratified downsampling ---
    if n_cells_per_group is not None:
        from ..reliability._reliability import subsample_adata
        if verbose:
            print(
                f"Subsampling to ≤{n_cells_per_group} cells per "
                f"'{level_keys[-1]}' group  ({adata.n_obs:,} → ...)..."
            )
        adata = subsample_adata(
            adata, level_keys,
            n_cells_per_group=n_cells_per_group,
            random_state=random_state,
        )
        if verbose:
            print(f"  {adata.n_obs:,} cells retained after subsampling")

    # --- Subset adata to genes present in adata ---
    genes_present = [g for g in genes_selection if g in adata.var_names]
    n_missing_genes = len(genes_selection) - len(genes_present)
    if n_missing_genes:
        warnings.warn(
            f"{n_missing_genes}/{len(genes_selection)} genes in genes_selection "
            f"not found in adata.var_names and will be skipped.",
            UserWarning,
            stacklevel=2,
        )
    adata_panel = adata[:, genes_present].copy()

    # When no batch_key, batch correction is a no-op regardless of batch_method
    effective_batch = batch_method if resolved_batch is not None else "per_batch"

    # --- Build global kNN (coarsest level + "none"/"global_filter" reuse it) ---
    if verbose:
        print("Building global kNN graph...")
    global_idx, global_dist = build_knn_graph(
        adata_panel, genes_present,
        batch_key=resolved_batch,
        knn_method=knn_method,
        batch_method=effective_batch,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        random_state=random_state,
    )

    def _per_ct_accuracy(
        true_labels: np.ndarray,
        predicted: np.ndarray,
    ) -> dict[str, float]:
        return {
            ct: float((predicted[true_labels == ct] == ct).mean())
            for ct in np.unique(true_labels)
        }

    # --- Per-level constrained accuracy ---
    level_constrained_acc: dict[str, dict[str, float]] = {}

    for li, level in enumerate(level_keys):
        if verbose:
            print(f"Computing accuracy at level '{level}'...")

        true_labels = adata_panel.obs[level].values.astype(str)

        if li == 0 or constrained_method == "none":
            predicted = _mode_with_tiebreak(true_labels[global_idx], global_dist)
            level_constrained_acc[level] = _per_ct_accuracy(true_labels, predicted)

        elif constrained_method == "global_filter":
            parent_labels = adata_panel.obs[level_keys[li - 1]].values.astype(str)
            predicted = _constrained_vote(global_idx, global_dist, true_labels, parent_labels)
            level_constrained_acc[level] = _per_ct_accuracy(true_labels, predicted)

        else:  # "within_class"
            parent_level  = level_keys[li - 1]
            parent_labels = adata_panel.obs[parent_level].values.astype(str)
            ct_acc: dict[str, float] = {}

            for parent_val in np.unique(parent_labels):
                parent_cell_idx = np.where(parent_labels == parent_val)[0]
                sub = adata_panel[parent_cell_idx]
                n_sub = sub.n_obs

                if n_sub <= 1:
                    warnings.warn(
                        f"Parent class '{parent_val}' at level '{parent_level}' "
                        f"has only {n_sub} cell(s) — skipping.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                k_eff = min(n_neighbors, n_sub - 1)
                if k_eff < n_neighbors:
                    warnings.warn(
                        f"Parent class '{parent_val}' at level '{parent_level}' "
                        f"has {n_sub} cells (fewer than n_neighbors+1={n_neighbors + 1}). "
                        f"Using k={k_eff}.",
                        UserWarning,
                        stacklevel=2,
                    )

                # Only apply batch correction when the subset has multiple batches
                if resolved_batch is not None and sub.obs[resolved_batch].nunique() > 1:
                    sub_batch_key    = resolved_batch
                    sub_batch_method = batch_method
                else:
                    sub_batch_key    = None
                    sub_batch_method = "per_batch"

                sub_idx, sub_dist = build_knn_graph(
                    sub, genes_present,
                    batch_key=sub_batch_key,
                    knn_method=knn_method,
                    batch_method=sub_batch_method,
                    n_neighbors=k_eff,
                    n_pcs=n_pcs,
                    random_state=random_state,
                )

                child_labels  = sub.obs[level].values.astype(str)
                sub_predicted = _mode_with_tiebreak(child_labels[sub_idx], sub_dist)
                ct_acc.update(_per_ct_accuracy(child_labels, sub_predicted))

            level_constrained_acc[level] = ct_acc

    # --- Compounded accuracy ---
    level_compounded_acc: dict[str, dict[str, float]] = {}
    for li, level in enumerate(level_keys):
        compounded: dict[str, float] = {}
        for ct, acc in level_constrained_acc[level].items():
            if li == 0:
                compounded[ct] = acc
            else:
                parent = child_to_parent_maps[li][ct]
                parent_compounded = level_compounded_acc[level_keys[li - 1]].get(parent, np.nan)
                compounded[ct] = acc * parent_compounded
        level_compounded_acc[level] = compounded

    # --- Assemble long-form DataFrame ---
    rows = []
    for level in level_keys:
        for ct in level_constrained_acc[level]:
            rows.append({
                "level":                level,
                "celltype":             ct,
                "constrained_accuracy": level_constrained_acc[level][ct],
                "compounded_accuracy":  level_compounded_acc[level].get(ct, np.nan),
            })

    return pd.DataFrame(rows)


