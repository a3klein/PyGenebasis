"""
Panel evaluation metrics.

Three complementary metrics assess how well a selected gene panel preserves
the transcriptional structure of the full dataset:

1. Cell neighbourhood preservation score (get_neighborhood_preservation_scores)
   Per-cell score in [0, 1] measuring how well a cell's k nearest neighbours
   in the full-transcriptome graph are recovered when using only the panel.
   Score = 1 means perfect recovery.  Primary metric used during selection.

2. Gene prediction score (get_gene_prediction_scores)
   Per-gene score measuring how well each gene's expression can be predicted
   from the panel's kNN neighbours.  Score = corr_panel / corr_true.

3. Cell type mapping accuracy (get_celltype_mapping is in mapping.py)

The orchestration function evaluate_library calls all three for a given panel,
optionally across a range of panel sizes.

Cell score formula (matches geneBasisR)
---------------------------------------
For each cell i:
    dist_true[i]      = median distance (in the selection embedding) to the k
                        true-graph neighbours of cell i
    dist_selection[i] = median distance (in the selection embedding) to the k
                        selection-graph neighbours of cell i
    mean_dist_all[i]  = mean pairwise distance from cell i to all other cells
                        (or a random 10% sample when option="approx")

    cell_score[i] = (mean_dist_all[i] - dist_selection[i])
                  / (mean_dist_all[i] - dist_true[i])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def get_neighborhood_preservation_scores(
    adata: AnnData,
    genes_selection: list[str],
    *,
    genes_all: list[str] | None = None,
    batch_key: str | None = None,
    n_neighbors: int = 5,
    n_pcs_all: int = 50,
    n_pcs_selection: int | None = None,
    knn_method: str = "approx",
    batch_method: str = "per_batch",
    option: str = "exact",
    neighs_all_stat: dict | None = None,
    layer: str | None = None,
    random_state: int = 32,
) -> pd.DataFrame:
    """Compute per-cell neighbourhood preservation scores.

    Parameters
    ----------
    adata : AnnData
    genes_selection : list[str]
        Selected gene panel.
    genes_all : list[str], optional
        Genes defining the true graph. Defaults to all genes in adata.
    batch_key : str, optional
    n_neighbors : int
        k. Default 5.
    n_pcs_all : int
        PCs for the true graph. Default 50.
    n_pcs_selection : int or None
        PCs for the selection graph. None skips PCA (default; use when
        panel is small).
    knn_method : {"approx", "exact"}
    batch_method : {"mnn", "harmony"}
    option : {"exact", "approx"}
        How to compute mean_dist_all. "exact" uses all cells; "approx"
        samples 10% (faster for large datasets).
    neighs_all_stat : dict, optional
        Pre-computed true-graph statistics from get_neighs_all_stat.
        Pass this to avoid recomputing the true graph on repeated calls.
    layer : str, optional
    random_state : int

    Returns
    -------
    pd.DataFrame
        Columns: ``cell`` (str), ``cell_score`` (float).
    """
    from ..knn._graph import build_knn_graph

    if genes_all is None:
        genes_all = adata.var_names.tolist()

    # --- True graph statistics (pre-computed or computed here) ---
    if neighs_all_stat is None:
        neighs_all_stat = get_neighs_all_stat(
            adata,
            genes_all=genes_all,
            batch_key=batch_key,
            n_neighbors=n_neighbors,
            n_pcs_all=n_pcs_all,
            option=option,
            knn_method=knn_method,
            batch_method=batch_method,
            layer=layer,
            random_state=random_state,
        )

    true_indices = neighs_all_stat["indices"]        # (n_cells, k)
    mean_dist_all = neighs_all_stat["mean_dist_all"]  # (n_cells,)
    # All distances are computed in the all-genes embedding (same space as
    # mean_dist_all) to keep the cell_score formula dimensionally consistent.
    all_embedding = neighs_all_stat["embedding"]      # (n_cells, n_pcs_all)

    # --- Selection graph ---
    sel_indices, _ = build_knn_graph(
        adata, genes_selection,
        batch_key=batch_key, knn_method=knn_method, batch_method=batch_method,
        n_neighbors=n_neighbors, n_pcs=n_pcs_selection, layer=layer,
        random_state=random_state,
    )

    # dist_true[i]:      median Euclidean distance (in all-genes embedding) to
    #                    true-graph neighbours of cell i
    # dist_selection[i]: median Euclidean distance (in all-genes embedding) to
    #                    selection-graph neighbours of cell i
    # All three quantities (dist_true, dist_selection, mean_dist_all) are in
    # the same all-genes embedding, so the cell_score formula is well-scaled.
    dist_true = _median_neighbour_dist(all_embedding, true_indices)
    dist_selection = _median_neighbour_dist(all_embedding, sel_indices)

    denom = mean_dist_all - dist_true
    # Avoid division by zero (can happen when all cells in a neighbourhood are identical)
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    cell_scores = (mean_dist_all - dist_selection) / denom

    return pd.DataFrame({
        "cell": adata.obs_names.tolist(),
        "cell_score": cell_scores,
    })


def get_gene_prediction_scores(
    adata: AnnData,
    genes_selection: list[str],
    *,
    genes_all: list[str] | None = None,
    batch_key: str | None = None,
    n_neighbors: int = 5,
    n_pcs_all: int = 50,
    n_pcs_selection: int | None = None,
    knn_method: str = "approx",
    batch_method: str = "per_batch",
    method: str = "spearman",
    layer: str | None = None,
    random_state: int = 32,
) -> pd.DataFrame:
    """Compute per-gene prediction scores.

    For each gene, measures how well its expression is predicted by the
    panel's kNN neighbourhood, normalised by how well the true graph predicts
    it.  Score ≈ 1 means the panel recovers gene expression as well as using
    all genes.

    Parameters
    ----------
    adata : AnnData
    genes_selection : list[str]
    genes_all : list[str], optional
    batch_key : str, optional
    n_neighbors : int
    n_pcs_all : int
    n_pcs_selection : int or None
    knn_method : {"approx", "exact"}
    batch_method : {"mnn", "harmony"}
    method : {"spearman", "pearson"}
        Correlation method.  Default "spearman" matches geneBasisR.
    layer : str, optional
    random_state : int

    Returns
    -------
    pd.DataFrame
        Columns: ``gene`` (str), ``corr`` (float), ``corr_all`` (float),
        ``gene_score`` (float = corr / corr_all).
    """
    from ..knn._graph import build_knn_graph, _extract_logcounts

    if genes_all is None:
        genes_all = adata.var_names.tolist()

    # True graph kNN
    true_indices, _ = build_knn_graph(
        adata, genes_all,
        batch_key=batch_key, knn_method=knn_method, batch_method=batch_method,
        n_neighbors=n_neighbors, n_pcs=n_pcs_all, layer=layer,
        random_state=random_state,
    )

    # Selection graph kNN
    sel_indices, _ = build_knn_graph(
        adata, genes_selection,
        batch_key=batch_key, knn_method=knn_method, batch_method=batch_method,
        n_neighbors=n_neighbors, n_pcs=n_pcs_selection, layer=layer,
        random_state=random_state,
    )

    # Logcounts for all genes
    X_all = _extract_logcounts(adata, genes_all, layer).astype(np.float64)

    corr_fn = _spearman_vec if method == "spearman" else _pearson_vec

    # Neighbour-averaged prediction from each graph
    corr_sel = corr_fn(X_all, sel_indices)
    corr_all = corr_fn(X_all, true_indices)

    gene_score = np.where(
        np.abs(corr_all) < 1e-12,
        np.nan,
        corr_sel / corr_all,
    )

    return pd.DataFrame({
        "gene": genes_all,
        "corr": corr_sel,
        "corr_all": corr_all,
        "gene_score": gene_score,
    })


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


def get_neighs_all_stat(
    adata: AnnData,
    *,
    genes_all: list[str] | None = None,
    batch_key: str | None = None,
    n_neighbors: int = 5,
    n_pcs_all: int = 50,
    option: str = "exact",
    knn_method: str = "approx",
    batch_method: str = "per_batch",
    layer: str | None = None,
    random_state: int = 32,
) -> dict:
    """Pre-compute true-graph neighbour statistics.

    Call this once and pass the result to ``get_neighborhood_preservation_scores``
    via ``neighs_all_stat`` to avoid recomputing the true graph on repeated
    calls (e.g. when evaluating multiple panel sizes).

    Returns
    -------
    dict with keys:
        "indices"       : np.ndarray (n_cells, k) — true-graph neighbour indices
        "embedding"     : np.ndarray (n_cells, n_pcs) — all-genes embedding
        "mean_dist_all" : np.ndarray (n_cells,)   — mean distance to all cells
                          in the all-genes embedding
    """
    from ..knn._graph import build_knn_graph, _build_embedding

    if genes_all is None:
        genes_all = adata.var_names.tolist()

    # Build the true-graph embedding once (used for both kNN and distances)
    all_embedding = _build_embedding(
        adata, genes_all,
        batch_key=batch_key, batch_method=batch_method,
        n_pcs=n_pcs_all, layer=layer, random_state=random_state,
    )

    true_indices, _ = build_knn_graph(
        adata, genes_all,
        batch_key=batch_key, knn_method=knn_method, batch_method=batch_method,
        n_neighbors=n_neighbors, n_pcs=n_pcs_all, layer=layer,
        random_state=random_state,
    )

    batch_labels_arr = adata.obs[batch_key].values if batch_key is not None else None
    per_batch_dists = (batch_method == "per_batch") and (batch_labels_arr is not None)
    mean_dist_all = _mean_dist_all(
        all_embedding, option=option, random_state=random_state,
        batch_labels=batch_labels_arr if per_batch_dists else None,
    )

    return {
        "indices": true_indices,
        "embedding": all_embedding,
        "mean_dist_all": mean_dist_all,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _median_neighbour_dist(
    embedding: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Compute the median Euclidean distance from each cell to its neighbours.

    Parameters
    ----------
    embedding : np.ndarray, shape (n_cells, n_features)
    indices   : np.ndarray, shape (n_cells, k)

    Returns
    -------
    median_dists : np.ndarray, shape (n_cells,)
    """
    # (n_cells, k, n_features)
    neighbour_embeddings = embedding[indices]
    # (n_cells, k)
    diffs = embedding[:, np.newaxis, :] - neighbour_embeddings
    dists = np.linalg.norm(diffs, axis=2)
    return np.median(dists, axis=1)


def _mean_dist_all(
    embedding: np.ndarray,
    option: str = "exact",
    random_state: int = 32,
    sample_frac: float = 0.10,
    batch_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Mean Euclidean distance from each cell to all (or sampled) other cells.

    Parameters
    ----------
    embedding : np.ndarray, shape (n_cells, n_features)
    option : {"exact", "approx"}
        "exact"  — pairwise distances to all cells (memory-intensive for large N)
        "approx" — sample 10% of cells as reference
    random_state : int
    sample_frac  : float
        Fraction of cells to sample in "approx" mode.
    batch_labels : np.ndarray, shape (n_cells,), optional
        If provided, distances are computed within each batch separately.
        Required when the embedding is from per-batch PCA (different PC bases
        across batches make cross-batch distances meaningless).

    Returns
    -------
    mean_dists : np.ndarray, shape (n_cells,)
    """
    if batch_labels is not None:
        # Compute per-batch and assemble into a single output vector.
        n_cells = embedding.shape[0]
        mean_dists = np.zeros(n_cells, dtype=np.float64)
        for batch in np.unique(batch_labels):
            mask = np.where(batch_labels == batch)[0]
            mean_dists[mask] = _mean_dist_all(
                embedding[mask], option=option,
                random_state=random_state, sample_frac=sample_frac,
            )
        return mean_dists

    n_cells = embedding.shape[0]

    if option == "approx":
        rng = np.random.default_rng(random_state)
        n_sample = max(1, int(n_cells * sample_frac))
        sample_idx = rng.choice(n_cells, size=n_sample, replace=False)
        ref = embedding[sample_idx]
    else:
        ref = embedding

    # Compute pairwise distances in chunks to limit memory usage
    chunk = 512
    mean_dists = np.zeros(n_cells, dtype=np.float64)
    for start in range(0, n_cells, chunk):
        end = min(start + chunk, n_cells)
        diff = embedding[start:end, np.newaxis, :] - ref[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)  # (chunk, n_ref)
        mean_dists[start:end] = dists.mean(axis=1)

    return mean_dists


def _spearman_vec(
    X: np.ndarray,
    knn_indices: np.ndarray,
) -> np.ndarray:
    """Vectorised Spearman correlation between each gene and its kNN prediction.

    For each gene g:
        predicted[i] = mean(X[neighbours_of_i, g])
        corr(g)      = spearman_r(X[:, g], predicted[:, g])

    Parameters
    ----------
    X           : np.ndarray, shape (n_cells, n_genes)
    knn_indices : np.ndarray, shape (n_cells, k)

    Returns
    -------
    corrs : np.ndarray, shape (n_genes,)
    """
    # Neighbour-averaged prediction for all genes at once
    predicted = X[knn_indices].mean(axis=1)  # (n_cells, n_genes)

    # Rank-transform each gene column, then compute Pearson (= Spearman)
    from scipy.stats import rankdata

    corrs = np.empty(X.shape[1], dtype=np.float64)
    n = X.shape[0]
    for g in range(X.shape[1]):
        r_x = rankdata(X[:, g])
        r_p = rankdata(predicted[:, g])
        # Pearson on ranks = Spearman
        mx = r_x - r_x.mean()
        mp = r_p - r_p.mean()
        denom = np.sqrt((mx ** 2).sum() * (mp ** 2).sum())
        corrs[g] = (mx * mp).sum() / denom if denom > 0 else 0.0

    return corrs


def _pearson_vec(
    X: np.ndarray,
    knn_indices: np.ndarray,
) -> np.ndarray:
    """Vectorised Pearson correlation between each gene and its kNN prediction."""
    predicted = X[knn_indices].mean(axis=1)  # (n_cells, n_genes)
    X_c = X - X.mean(axis=0)
    P_c = predicted - predicted.mean(axis=0)
    num = (X_c * P_c).sum(axis=0)
    denom = np.sqrt((X_c ** 2).sum(axis=0) * (P_c ** 2).sum(axis=0))
    return np.where(denom > 0, num / denom, 0.0)
