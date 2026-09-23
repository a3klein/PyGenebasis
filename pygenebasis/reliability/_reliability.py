"""
Gene reliability analysis via expression perturbation experiments.

Given a pre-computed ``results`` DataFrame that classifies each panel gene into
a technical reliability failure mode (from co-expression comparison between a
MERFISH spatial dataset and an scRNA-seq reference), this module runs perturbation
experiments on the scRNA-seq reference AnnData to quantify how each failure mode
impacts cell type mapping accuracy across levels of a nested annotation hierarchy.

Why on the scRNA-seq reference, not MERFISH
-------------------------------------------
Perturbations are run on the scRNA-seq AnnData because it has reliable, high-quality
cell type annotations at all hierarchy levels.  MERFISH annotations are not yet
available or are lower confidence.

Failure modes and perturbations
--------------------------------
The ``failure_mode`` column in ``results`` determines how each gene is perturbed:

  probe_failure
      Probe technically broken; cell type present but probe signal is noise.
      Perturbation: permute expression values across cells (independently per gene),
      destroying co-expression while preserving the marginal count distribution.
      Stochastic → run ``n_replicates`` seeded runs.

  composition_mismatch
      Cell type likely absent or rare in the MERFISH tissue section.
      Perturbation: zero out expression columns for affected genes.
      Deterministic → run once.

  idiosyncratic_noise
      Probe partially works; signal degraded with gene-specific noise.
      Perturbation: add Gaussian noise ~ N(0, σ_g) where σ_g is the empirical
      standard deviation of gene g across cells.
      Stochastic → run ``n_replicates`` seeded runs.

  reliable
      Not perturbed.  Only non-reliable failure modes produce delta values.

Baseline
--------
The baseline kNN graph is built from the **full unmodified gene panel** (all genes
in ``results.index``, none perturbed).  Delta CT accuracy is:

    delta = CT_acc_baseline[cell_type, level] - CT_acc_perturbed[cell_type, level]

Positive delta → failure mode genes contribute positively (perturbation hurts).
Negative delta → genes are actively misleading for that cell type.

Annotation hierarchy
--------------------
Pass ``level_keys`` as a list ordered coarsest → finest, e.g.:
    ["Class", "Subclass", "Group"]
The kNN graph is built once per perturbation condition and CT accuracy is evaluated
separately at each level, amortising the expensive kNN step.

.. note::
   **Upstream pipeline — co-expression reliability scoring:**
   The ``results`` DataFrame is produced by :mod:`pygenebasis.correlation`.
   See :func:`~pygenebasis.compute_gene_reliability` for the full pipeline
   (correlation computation, linear rescaling, PCA denoising, threshold
   fitting, and failure mode assignment).

Batch handling
--------------
``batch_key`` accepts a ``str`` or ``list[str]``.  If a list, a joint column is
created by concatenating with ``"_"``.  CT mapping uses ``harmony`` batch
correction; all other computations use ``per_batch``.

Subsampling
-----------
Stratified at the finest level in ``level_keys``, up to ``n_cells_per_group``
cells per group.  Subsampling happens once before the perturbation loop so all
conditions use the same subsampled AnnData.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse
import anndata as ad

from ..io._core import prepare_batch_key

_STOCHASTIC_MODES    = {"probe_failure", "idiosyncratic_noise"}
_DETERMINISTIC_MODES = {"composition_mismatch"}
_VALID_MODES         = _STOCHASTIC_MODES | _DETERMINISTIC_MODES
_VALID_VOTE          = {"unconstrained", "constrained", "both"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def subsample_adata(
    adata: ad.AnnData,
    level_keys: list[str],
    *,
    n_cells_per_group: int = 5000,
    random_state: int = 0,
) -> ad.AnnData:
    """Stratified downsample at the finest annotation level in ``level_keys``.

    Samples up to ``n_cells_per_group`` cells from each group defined by the
    last entry of ``level_keys`` (the finest level).  Groups with fewer cells
    are kept in full.

    Parameters
    ----------
    adata : AnnData
    level_keys : list[str]
        Annotation columns in ``adata.obs``, ordered coarsest → finest.
        Subsampling stratifies by the last (finest) key.
    n_cells_per_group : int
        Maximum cells per finest-level group.  Default 5000.
    random_state : int

    Returns
    -------
    AnnData
        A copy of ``adata`` containing the selected cells.

    Raises
    ------
    ValueError
        If any key in ``level_keys`` is not a column in ``adata.obs``.
    """
    missing = [k for k in level_keys if k not in adata.obs.columns]
    if missing:
        raise ValueError(f"level_keys not found in adata.obs: {missing}")

    finest_key = level_keys[-1]
    rng = np.random.default_rng(random_state)

    keep_idx = []
    for group, group_obs in adata.obs.groupby(finest_key, sort=True, observed=True):
        idx = group_obs.index
        if len(idx) <= n_cells_per_group:
            keep_idx.extend(adata.obs_names.get_indexer(idx))
        else:
            chosen = rng.choice(len(idx), size=n_cells_per_group, replace=False)
            chosen_names = idx[np.sort(chosen)]
            keep_idx.extend(adata.obs_names.get_indexer(chosen_names))

    return adata[keep_idx].copy()


def perturb_expression(
    adata: ad.AnnData,
    genes: list[str],
    failure_mode: str,
    *,
    random_state: int = 0,
) -> ad.AnnData:
    """Return a copy of ``adata`` with expression of ``genes`` perturbed.

    Does **not** modify ``adata`` in place.

    Perturbation per failure mode:

    ``probe_failure``
        Each gene's expression is independently permuted across cells.
        Destroys co-expression while preserving each gene's marginal distribution.

    ``composition_mismatch``
        Expression of affected genes is set to zero for all cells.
        Deterministic (ignores ``random_state``).

    ``idiosyncratic_noise``
        Gaussian noise N(0, σ_g) is added to each gene, where σ_g is the
        empirical standard deviation of gene g across cells in ``adata``.

    Parameters
    ----------
    adata : AnnData
        Input data (not modified).
    genes : list[str]
        Genes to perturb.  Genes not found in ``adata.var_names`` are silently
        skipped.
    failure_mode : str
        One of ``{"probe_failure", "composition_mismatch", "idiosyncratic_noise"}``.
    random_state : int
        Seed for stochastic perturbations.

    Returns
    -------
    AnnData
        Copy of ``adata`` with perturbed expression in ``.X``.

    Raises
    ------
    ValueError
        If ``failure_mode`` is not a recognised non-reliable mode.
    """
    if failure_mode not in _VALID_MODES:
        raise ValueError(
            f"failure_mode must be one of {sorted(_VALID_MODES)}, got {failure_mode!r}. "
            f"'reliable' genes are not perturbed."
        )

    # Work out which genes are actually present
    present = [g for g in genes if g in adata.var_names]
    if not present:
        return adata.copy()

    gene_idx = np.array([adata.var_names.get_loc(g) for g in present])

    # Extract expression as dense float64 array
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float64, copy=True)

    rng = np.random.default_rng(random_state)

    if failure_mode == "probe_failure":
        for idx in gene_idx:
            col = X[:, idx].copy()
            rng.shuffle(col)
            X[:, idx] = col

    elif failure_mode == "composition_mismatch":
        X[:, gene_idx] = 0.0

    elif failure_mode == "idiosyncratic_noise":
        for idx in gene_idx:
            sigma = X[:, idx].std()
            if sigma > 0:
                noise = rng.normal(0.0, sigma, size=X.shape[0])
                X[:, idx] = X[:, idx] + noise

    out = ad.AnnData(
        X=X,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        obsm={k: v.copy() for k, v in adata.obsm.items()},
    )
    return out


def run_perturbation_analysis(
    adata: ad.AnnData,
    results: pd.DataFrame,
    *,
    level_keys: list[str],
    batch_key: str | list[str] | None = None,
    n_neighbors: int = 5,
    knn_method: str = "approx",
    n_replicates: int = 5,
    n_cells_per_group: int = 5000,
    random_state: int = 0,
    vote: str = "both",
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Perturbation-based gene reliability analysis.

    For each non-reliable failure mode in ``results``, perturbs the expression
    of affected genes in ``adata``, rebuilds the kNN graph, and measures the
    change in cell type mapping accuracy (delta CT) at each annotation level.

    The gene panel is defined by ``results.index``.  Genes with
    ``failure_mode == "reliable"`` are never perturbed.

    Parameters
    ----------
    adata : AnnData
        scRNA-seq reference.  Must have ``level_keys`` columns in ``.obs``.
    results : pd.DataFrame
        Gene reliability DataFrame indexed by gene name.  Must have a
        ``failure_mode`` column.  The index defines the gene panel.
    level_keys : list[str]
        Annotation columns ordered coarsest → finest.
    batch_key : str, list[str], or None
        Batch column(s).  Harmony correction used for CT mapping.
    n_neighbors : int
    knn_method : {"approx", "exact"}
    n_replicates : int
        Seeded runs for stochastic modes.
    n_cells_per_group : int
    random_state : int
    vote : {"unconstrained", "constrained", "both"}
        Controls which majority-vote strategy is used for CT accuracy scoring.

        ``"unconstrained"``  (default behaviour pre-v2)
            All k neighbours vote regardless of their parent-level label.
            Identical to ``get_celltype_mapping``.

        ``"constrained"``
            When predicting a child level (e.g. ``Subclass``), only neighbours
            that share the same parent label (e.g. ``Class``) as the query cell
            are eligible to vote.  Prevents cross-lineage contamination.  For
            the coarsest level (no parent) this is identical to unconstrained.

        ``"both"``  (default)
            Both strategies are run from the same kNN build.  Unconstrained
            results occupy the standard output keys; constrained results are
            stored under the ``_constrained`` suffix keys.

        Unconstrained output keys are **always** present regardless of this
        setting.  Constrained keys are present only when
        ``vote in {"constrained", "both"}``.
    verbose : bool

    Returns
    -------
    dict with keys (unconstrained, always present):

    ``"delta_ct"``
        Long-form DataFrame: columns ``failure_mode, level, celltype, replicate,
        delta_ct``.  ``delta_ct = baseline_ct − perturbed_ct``.
        When both ``probe_failure`` and ``idiosyncratic_noise`` genes are present,
        a ``"combined_technical"`` entry is appended in which both perturbations
        are applied simultaneously to the same AnnData copy.
    ``"baseline_ct"``
        Per-level, per-celltype baseline accuracy: columns ``level, celltype,
        ct_accuracy``.
    ``"summary"``
        Macro-average delta per ``(failure_mode, level)``: columns
        ``failure_mode, level, mean_delta, std_delta, n_cell_types``.
    ``"delta_ct_removal"``
        Like ``"delta_ct"`` but for the removal condition (failure-mode genes
        excluded from the panel entirely, no replicate column): columns
        ``failure_mode, level, celltype, delta_ct``.
    ``"removal_benefit"``
        ``mean(delta_ct_perturbed) − delta_ct_removal``.  Positive → removing
        hurts less than keeping the broken genes.  Columns:
        ``failure_mode, level, celltype, removal_benefit``.

    Additional keys when ``vote in {"constrained", "both"}``
    (same schema as their unconstrained counterparts):

    ``"delta_ct_constrained"``
    ``"baseline_ct_constrained"``
    ``"summary_constrained"``
    ``"delta_ct_removal_constrained"``
    ``"removal_benefit_constrained"``

    Raises
    ------
    ValueError
        If ``vote`` is not one of ``{"unconstrained", "constrained", "both"}``.
    """
    from ..evaluation._mapping import get_celltype_mapping, _mode_with_tiebreak, _constrained_vote

    # --- Validate vote parameter ---
    if vote not in _VALID_VOTE:
        raise ValueError(
            f"vote must be one of {sorted(_VALID_VOTE)}, got {vote!r}."
        )
    do_constrained = vote in ("constrained", "both")

    # --- Validate level_keys ---
    missing = [k for k in level_keys if k not in adata.obs.columns]
    if missing:
        raise ValueError(f"level_keys not found in adata.obs: {missing}")

    # --- Subsample once ---
    adata_sub = subsample_adata(
        adata, level_keys, n_cells_per_group=n_cells_per_group, random_state=random_state,
    )

    # --- Resolve batch key ---
    adata_sub, resolved_batch = prepare_batch_key(adata_sub, batch_key)

    # --- Gene panel from results.index ---
    genes_panel = [g for g in results.index if g in adata_sub.var_names]

    # --- Subset adata to panel genes only ---
    adata_panel = adata_sub[:, genes_panel].copy()

    # -----------------------------------------------------------------------
    # Helper: CT accuracy at all levels for a given AnnData + gene list.
    #
    # Returns (unc_dict, con_dict | None) where each dict maps
    #   level → DataFrame(celltype, frac_correctly_mapped)
    #
    # ``genes_in`` controls which genes the kNN graph is built from.
    # ``adata_in`` controls the expression values (may be perturbed).
    # Separating the two lets removal conditions reuse the unperturbed
    # AnnData while changing only the gene list.
    # -----------------------------------------------------------------------
    def _ct_accuracy_all_levels(
        adata_in: ad.AnnData,
        genes_in: list[str],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame] | None]:
        from ..knn._graph import build_knn_graph

        indices, distances = build_knn_graph(
            adata_in,
            genes_in,
            batch_key=resolved_batch,
            knn_method=knn_method,
            batch_method="harmony" if resolved_batch else "per_batch",
            n_neighbors=n_neighbors,
            n_pcs=None,
            random_state=random_state,
        )

        unc_results: dict[str, pd.DataFrame] = {}
        con_results: dict[str, pd.DataFrame] | None = {} if do_constrained else None

        for level_idx, level in enumerate(level_keys):
            true_labels = adata_in.obs[level].values.astype(str)

            # --- Unconstrained ---
            predicted_unc = _mode_with_tiebreak(true_labels[indices], distances)
            rows_unc = []
            for ct in np.unique(true_labels):
                mask = true_labels == ct
                frac = float((predicted_unc[mask] == ct).mean())
                rows_unc.append({"celltype": ct, "frac_correctly_mapped": frac})
            unc_results[level] = pd.DataFrame(rows_unc)

            # --- Constrained ---
            if do_constrained:
                if level_idx == 0:
                    # Coarsest level has no parent → constrained == unconstrained
                    con_results[level] = unc_results[level].copy()
                else:
                    parent_level = level_keys[level_idx - 1]
                    parent_labels = adata_in.obs[parent_level].values.astype(str)
                    predicted_con = _constrained_vote(
                        indices, distances, true_labels, parent_labels,
                    )
                    rows_con = []
                    for ct in np.unique(true_labels):
                        mask = true_labels == ct
                        frac = float((predicted_con[mask] == ct).mean())
                        rows_con.append({"celltype": ct, "frac_correctly_mapped": frac})
                    con_results[level] = pd.DataFrame(rows_con)

        return unc_results, con_results

    # -----------------------------------------------------------------------
    # Baseline
    # -----------------------------------------------------------------------
    if verbose:
        print("Computing baseline CT accuracy on full unperturbed panel...")
    baseline_unc, baseline_con = _ct_accuracy_all_levels(adata_panel, genes_panel)

    def _acc_to_df(
        acc_dict: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        rows = []
        for level, df in acc_dict.items():
            for _, row in df.iterrows():
                rows.append({
                    "level":       level,
                    "celltype":    row["celltype"],
                    "ct_accuracy": row["frac_correctly_mapped"],
                })
        return pd.DataFrame(rows)

    baseline_ct     = _acc_to_df(baseline_unc)
    baseline_ct_con = _acc_to_df(baseline_con) if do_constrained else None

    # -----------------------------------------------------------------------
    # Perturbations
    # -----------------------------------------------------------------------
    failure_modes = [
        fm for fm in results["failure_mode"].unique()
        if fm != "reliable"
    ]

    # Pre-compute gene sets for the combined_technical condition.
    # Only produced when both probe_failure and idiosyncratic_noise genes are present.
    _TECH_MODES = {"probe_failure", "idiosyncratic_noise"}
    tech_gene_sets: dict[str, list[str]] = {}
    for _fm in failure_modes:
        if _fm in _TECH_MODES:
            _genes = [g for g in results.index[results["failure_mode"] == _fm] if g in genes_panel]
            if _genes:
                tech_gene_sets[_fm] = _genes

    delta_rows:     list[dict] = []
    delta_rows_con: list[dict] = [] if do_constrained else None  # type: ignore[assignment]

    for fm in failure_modes:
        fm_genes = results.index[results["failure_mode"] == fm].tolist()
        fm_genes = [g for g in fm_genes if g in genes_panel]

        if not fm_genes:
            continue

        n_runs = 1 if fm in _DETERMINISTIC_MODES else n_replicates

        if verbose:
            print(f"Perturbing '{fm}' ({len(fm_genes)} genes, {n_runs} run(s))...")

        for rep in range(n_runs):
            seed = random_state + rep
            adata_perturbed = perturb_expression(
                adata_panel, fm_genes, fm, random_state=seed,
            )
            pert_unc, pert_con = _ct_accuracy_all_levels(adata_perturbed, genes_panel)

            for level in level_keys:
                base_df = baseline_unc[level].set_index("celltype")
                pert_df = pert_unc[level].set_index("celltype")
                for ct in base_df.index:
                    base_val = base_df.loc[ct, "frac_correctly_mapped"]
                    pert_val = pert_df.loc[ct, "frac_correctly_mapped"] if ct in pert_df.index else np.nan
                    delta_rows.append({
                        "failure_mode": fm,
                        "level":        level,
                        "celltype":     ct,
                        "replicate":    rep,
                        "delta_ct":     float(base_val - pert_val),
                    })

                if do_constrained:
                    base_df_con = baseline_con[level].set_index("celltype")
                    pert_df_con = pert_con[level].set_index("celltype")
                    for ct in base_df_con.index:
                        base_val = base_df_con.loc[ct, "frac_correctly_mapped"]
                        pert_val = pert_df_con.loc[ct, "frac_correctly_mapped"] if ct in pert_df_con.index else np.nan
                        delta_rows_con.append({
                            "failure_mode": fm,
                            "level":        level,
                            "celltype":     ct,
                            "replicate":    rep,
                            "delta_ct":     float(base_val - pert_val),
                        })

    # -----------------------------------------------------------------------
    # Combined technical perturbation: probe_failure + idiosyncratic_noise
    # applied simultaneously to the same AnnData copy.
    # -----------------------------------------------------------------------
    if len(tech_gene_sets) >= 2:
        n_combined = sum(len(v) for v in tech_gene_sets.values())
        if verbose:
            print(
                f"Perturbing 'combined_technical' "
                f"({n_combined} genes across {len(tech_gene_sets)} modes, "
                f"{n_replicates} run(s))..."
            )
        for rep in range(n_replicates):
            seed = random_state + rep
            adata_combined = adata_panel
            for fm, fm_genes in tech_gene_sets.items():
                adata_combined = perturb_expression(adata_combined, fm_genes, fm, random_state=seed)
            pert_unc, pert_con = _ct_accuracy_all_levels(adata_combined, genes_panel)

            for level in level_keys:
                base_df = baseline_unc[level].set_index("celltype")
                pert_df = pert_unc[level].set_index("celltype")
                for ct in base_df.index:
                    base_val = base_df.loc[ct, "frac_correctly_mapped"]
                    pert_val = pert_df.loc[ct, "frac_correctly_mapped"] if ct in pert_df.index else np.nan
                    delta_rows.append({
                        "failure_mode": "combined_technical",
                        "level":        level,
                        "celltype":     ct,
                        "replicate":    rep,
                        "delta_ct":     float(base_val - pert_val),
                    })

                if do_constrained:
                    base_df_con = baseline_con[level].set_index("celltype")
                    pert_df_con = pert_con[level].set_index("celltype")
                    for ct in base_df_con.index:
                        base_val = base_df_con.loc[ct, "frac_correctly_mapped"]
                        pert_val = pert_df_con.loc[ct, "frac_correctly_mapped"] if ct in pert_df_con.index else np.nan
                        delta_rows_con.append({
                            "failure_mode": "combined_technical",
                            "level":        level,
                            "celltype":     ct,
                            "replicate":    rep,
                            "delta_ct":     float(base_val - pert_val),
                        })

    delta_ct = pd.DataFrame(delta_rows)
    delta_ct_con = (
        pd.DataFrame(delta_rows_con)
        if do_constrained
        else None
    )

    # -----------------------------------------------------------------------
    # Removal: build kNN with failure-mode genes excluded entirely.
    #
    # One deterministic run per failure mode (no expression modification).
    #   delta_ct_perturbed = baseline − perturbed_ct   (expression broken, genes present)
    #   delta_ct_removal   = baseline − removal_ct     (genes absent from panel)
    #   removal_benefit    = mean(delta_ct_perturbed) − delta_ct_removal
    #                      = removal_ct − mean(perturbed_ct)
    # Positive benefit → removing hurts less than keeping the broken genes.
    # Negative benefit → genes carry enough residual signal that removing
    #                    them hurts more than keeping them broken.
    # -----------------------------------------------------------------------
    removal_rows:     list[dict] = []
    removal_rows_con: list[dict] = [] if do_constrained else None  # type: ignore[assignment]

    for fm in failure_modes:
        fm_genes = results.index[results["failure_mode"] == fm].tolist()
        fm_genes = [g for g in fm_genes if g in genes_panel]

        if not fm_genes:
            continue

        remaining_genes = [g for g in genes_panel if g not in set(fm_genes)]
        if not remaining_genes:
            if verbose:
                print(f"  Skipping removal for '{fm}': would remove all panel genes.")
            continue

        if verbose:
            print(
                f"Running removal '{fm}' "
                f"({len(fm_genes)} genes removed, {len(remaining_genes)} remaining)..."
            )

        rem_unc, rem_con = _ct_accuracy_all_levels(adata_panel, remaining_genes)

        for level in level_keys:
            base_df = baseline_unc[level].set_index("celltype")
            rem_df  = rem_unc[level].set_index("celltype")
            for ct in base_df.index:
                base_val = base_df.loc[ct, "frac_correctly_mapped"]
                rem_val  = rem_df.loc[ct, "frac_correctly_mapped"] if ct in rem_df.index else np.nan
                removal_rows.append({
                    "failure_mode": fm,
                    "level":        level,
                    "celltype":     ct,
                    "delta_ct":     float(base_val - rem_val),
                })

            if do_constrained:
                base_df_con = baseline_con[level].set_index("celltype")
                rem_df_con  = rem_con[level].set_index("celltype")
                for ct in base_df_con.index:
                    base_val = base_df_con.loc[ct, "frac_correctly_mapped"]
                    rem_val  = rem_df_con.loc[ct, "frac_correctly_mapped"] if ct in rem_df_con.index else np.nan
                    removal_rows_con.append({
                        "failure_mode": fm,
                        "level":        level,
                        "celltype":     ct,
                        "delta_ct":     float(base_val - rem_val),
                    })

    # Combined technical removal: remove probe_failure + idiosyncratic_noise genes together.
    if len(tech_gene_sets) >= 2:
        all_tech_genes = {g for fm_genes in tech_gene_sets.values() for g in fm_genes}
        remaining_combined = [g for g in genes_panel if g not in all_tech_genes]
        if not remaining_combined:
            if verbose:
                print("  Skipping combined_technical removal: would remove all panel genes.")
        else:
            if verbose:
                print(
                    f"Running removal 'combined_technical' "
                    f"({len(all_tech_genes)} genes removed, {len(remaining_combined)} remaining)..."
                )
            rem_unc, rem_con = _ct_accuracy_all_levels(adata_panel, remaining_combined)

            for level in level_keys:
                base_df = baseline_unc[level].set_index("celltype")
                rem_df  = rem_unc[level].set_index("celltype")
                for ct in base_df.index:
                    base_val = base_df.loc[ct, "frac_correctly_mapped"]
                    rem_val  = rem_df.loc[ct, "frac_correctly_mapped"] if ct in rem_df.index else np.nan
                    removal_rows.append({
                        "failure_mode": "combined_technical",
                        "level":        level,
                        "celltype":     ct,
                        "delta_ct":     float(base_val - rem_val),
                    })

                if do_constrained:
                    base_df_con = baseline_con[level].set_index("celltype")
                    rem_df_con  = rem_con[level].set_index("celltype")
                    for ct in base_df_con.index:
                        base_val = base_df_con.loc[ct, "frac_correctly_mapped"]
                        rem_val  = rem_df_con.loc[ct, "frac_correctly_mapped"] if ct in rem_df_con.index else np.nan
                        removal_rows_con.append({
                            "failure_mode": "combined_technical",
                            "level":        level,
                            "celltype":     ct,
                            "delta_ct":     float(base_val - rem_val),
                        })

    _EMPTY_REMOVAL = pd.DataFrame(columns=["failure_mode", "level", "celltype", "delta_ct"])
    _EMPTY_BENEFIT = pd.DataFrame(columns=["failure_mode", "level", "celltype", "removal_benefit"])

    delta_ct_removal = pd.DataFrame(removal_rows) if removal_rows else _EMPTY_REMOVAL.copy()
    delta_ct_removal_con = (
        pd.DataFrame(removal_rows_con) if removal_rows_con else _EMPTY_REMOVAL.copy()
    ) if do_constrained else None

    def _removal_benefit(
        dct: pd.DataFrame,
        dcr: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute removal benefit from perturbed and removal delta DataFrames."""
        if dct.empty or dcr.empty:
            return _EMPTY_BENEFIT.copy()
        mean_pert = (
            dct
            .groupby(["failure_mode", "level", "celltype"])["delta_ct"]
            .mean()
            .reset_index()
            .rename(columns={"delta_ct": "_mean_delta_perturbed"})
        )
        rb = mean_pert.merge(
            dcr.rename(columns={"delta_ct": "_delta_removal"}),
            on=["failure_mode", "level", "celltype"],
            how="inner",
        )
        rb["removal_benefit"] = rb["_mean_delta_perturbed"] - rb["_delta_removal"]
        return rb[["failure_mode", "level", "celltype", "removal_benefit"]].reset_index(drop=True)

    removal_benefit     = _removal_benefit(delta_ct,     delta_ct_removal)
    removal_benefit_con = (
        _removal_benefit(delta_ct_con, delta_ct_removal_con)
        if do_constrained
        else None
    )

    # -----------------------------------------------------------------------
    # Summary: macro-average delta per (failure_mode, level)
    # -----------------------------------------------------------------------
    def _make_summary(dct: pd.DataFrame) -> pd.DataFrame:
        rows = []
        if not dct.empty:
            for (fm, level), grp in dct.groupby(["failure_mode", "level"]):
                # Average over replicates first, then over cell types
                per_ct_mean = grp.groupby(["celltype", "replicate"])["delta_ct"].mean()
                per_ct = per_ct_mean.groupby("celltype").mean()
                rows.append({
                    "failure_mode": fm,
                    "level":        level,
                    "mean_delta":   float(per_ct.mean()),
                    "std_delta":    float(per_ct.std(ddof=1)) if len(per_ct) > 1 else 0.0,
                    "n_cell_types": int(len(per_ct)),
                })
        return pd.DataFrame(rows)

    summary     = _make_summary(delta_ct)
    summary_con = _make_summary(delta_ct_con) if do_constrained else None

    # -----------------------------------------------------------------------
    # Assemble return dict
    # -----------------------------------------------------------------------
    result: dict[str, pd.DataFrame] = {
        "delta_ct":         delta_ct,
        "baseline_ct":      baseline_ct,
        "summary":          summary,
        "delta_ct_removal": delta_ct_removal,
        "removal_benefit":  removal_benefit,
    }
    if do_constrained:
        result.update({
            "delta_ct_constrained":         delta_ct_con,
            "baseline_ct_constrained":      baseline_ct_con,
            "summary_constrained":          summary_con,
            "delta_ct_removal_constrained": delta_ct_removal_con,
            "removal_benefit_constrained":  removal_benefit_con,
        })
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_delta_ct_heatmap(
    delta_ct: pd.DataFrame,
    *,
    level: str,
    baseline_ct: pd.DataFrame | None = None,
    delta_threshold: float = 0.05,
    figsize: tuple[float, float] | None = None,
) -> tuple:
    """Heatmap of delta CT accuracy per cell type and failure mode.

    Rows = cell types, columns = failure modes, colour = mean delta CT
    (averaged over replicates).  Diverging colourmap centred at 0.
    Cells where ``|delta| > delta_threshold`` are annotated with their value.

    If ``baseline_ct`` is provided, a side panel shows baseline CT accuracy
    per cell type for context.

    Parameters
    ----------
    delta_ct : pd.DataFrame
        Output of ``run_perturbation_analysis``[``"delta_ct"``].
        Will be filtered to ``level`` internally.
    level : str
        Annotation level to plot.
    baseline_ct : pd.DataFrame, optional
        Output of ``run_perturbation_analysis``[``"baseline_ct"``].
    delta_threshold : float
        Annotate cells where ``|delta| > delta_threshold``.
    figsize : (float, float), optional

    Returns
    -------
    (fig, ax) : tuple
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Filter to requested level and average over replicates
    df = delta_ct[delta_ct["level"] == level].copy()
    mean_delta = (
        df.groupby(["celltype", "failure_mode"])["delta_ct"]
        .mean()
        .unstack("failure_mode")
    )

    n_ct  = mean_delta.shape[0]
    n_fm  = mean_delta.shape[1]
    has_baseline = baseline_ct is not None

    n_cols = 2 if has_baseline else 1
    col_widths = [n_fm * 1.2, 0.8] if has_baseline else [n_fm * 1.2]

    if figsize is None:
        figsize = (sum(col_widths) + 1.5, max(3.0, n_ct * 0.4 + 1.5))

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=figsize,
        gridspec_kw={"width_ratios": col_widths} if has_baseline else None,
    )
    ax_main = axes[0] if has_baseline else axes

    # Symmetric colour scale
    vmax = max(abs(mean_delta.values[np.isfinite(mean_delta.values)]).max(), 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax_main.imshow(mean_delta.values, aspect="auto", cmap="RdBu_r", norm=norm)

    ax_main.set_xticks(range(n_fm))
    ax_main.set_xticklabels(mean_delta.columns, rotation=30, ha="right", fontsize=9)
    ax_main.set_yticks(range(n_ct))
    ax_main.set_yticklabels(mean_delta.index, fontsize=8)
    ax_main.set_title(f"Delta CT accuracy — {level}", fontsize=11)
    ax_main.set_xlabel("Failure mode")
    ax_main.set_ylabel("Cell type")

    # Annotate cells above threshold
    for i in range(n_ct):
        for j in range(n_fm):
            val = mean_delta.values[i, j]
            if np.isfinite(val) and abs(val) > delta_threshold:
                ax_main.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > 0.6 * vmax else "black",
                )

    plt.colorbar(im, ax=ax_main, label="Delta CT accuracy", fraction=0.046, pad=0.04)

    # Side panel: baseline accuracy
    if has_baseline:
        ax_side = axes[1]
        base = (
            baseline_ct[baseline_ct["level"] == level]
            .set_index("celltype")["ct_accuracy"]
            .reindex(mean_delta.index)
        )
        ax_side.imshow(
            base.values.reshape(-1, 1),
            aspect="auto", cmap="Greys", vmin=0, vmax=1,
        )
        ax_side.set_xticks([0])
        ax_side.set_xticklabels(["Baseline\naccuracy"], fontsize=8)
        ax_side.set_yticks(range(n_ct))
        ax_side.set_yticklabels([])
        for i, val in enumerate(base.values):
            if np.isfinite(val):
                ax_side.text(
                    0, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if val > 0.7 else "black",
                )

    fig.tight_layout()
    return fig, ax_main


def plot_perturbation_summary(
    summary: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
) -> tuple:
    """Bar chart of macro-average delta CT per failure mode, faceted by level.

    Error bars show standard deviation across cell types.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of ``run_perturbation_analysis``[``"summary"``].
    figsize : (float, float), optional

    Returns
    -------
    (fig, axes) : tuple
        ``axes`` is a 1-D array of Axes, one per annotation level.
    """
    import matplotlib.pyplot as plt

    levels = summary["level"].unique().tolist()
    n_levels = len(levels)

    if figsize is None:
        figsize = (4 * n_levels + 1, 4)

    fig, axes = plt.subplots(1, n_levels, figsize=figsize, sharey=False)
    if n_levels == 1:
        axes = np.array([axes])

    for ax, level in zip(axes, levels):
        sub = summary[summary["level"] == level].copy()
        x = np.arange(len(sub))
        colors = [
            "#d62728" if v > 0 else "#1f77b4"
            for v in sub["mean_delta"]
        ]
        ax.bar(x, sub["mean_delta"], yerr=sub["std_delta"], color=colors,
               capsize=4, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["failure_mode"], rotation=30, ha="right", fontsize=9)
        ax.set_title(level, fontsize=11)
        ax.set_ylabel("Mean delta CT accuracy")
        ax.set_xlabel("Failure mode")

    fig.suptitle("Perturbation impact by annotation level", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_removal_benefit(
    removal_benefit: pd.DataFrame,
    *,
    level: str,
    benefit_threshold: float = 0.02,
    figsize: tuple[float, float] | None = None,
) -> tuple:
    """Heatmap of removal benefit per cell type and failure mode.

    The removal benefit is defined as::

        removal_benefit = mean(delta_CT_perturbed) − delta_CT_removal
                        = CT_acc_removal − mean(CT_acc_perturbed)

    * **Positive** (red) — removing the genes hurts accuracy *less* than
      keeping them in their broken/modulated state.  There is a net benefit
      to physically removing these genes from the panel.
    * **Negative** (blue) — even in their broken state the genes still carry
      enough residual signal that removing them hurts accuracy *more*.  Keeping
      the broken genes is preferable to removing them.

    Rows = cell types, columns = failure modes, colour = removal benefit.
    Diverging colourmap centred at 0.  Cells where
    ``|benefit| > benefit_threshold`` are annotated with their value.

    Parameters
    ----------
    removal_benefit : pd.DataFrame
        Output of ``run_perturbation_analysis``[``"removal_benefit"``].
        Will be filtered to ``level`` internally.
    level : str
        Annotation level to plot.
    benefit_threshold : float
        Annotate cells where ``|benefit| > benefit_threshold``.  Default 0.02.
    figsize : (float, float), optional

    Returns
    -------
    (fig, ax) : tuple
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    df = removal_benefit[removal_benefit["level"] == level].copy()
    pivot = df.pivot(index="celltype", columns="failure_mode", values="removal_benefit")

    n_ct = pivot.shape[0]
    n_fm = pivot.shape[1]

    if figsize is None:
        figsize = (n_fm * 1.4 + 2.0, max(3.0, n_ct * 0.4 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)

    vals = pivot.values
    finite_vals = vals[np.isfinite(vals)]
    vmax = max(np.abs(finite_vals).max(), 1e-6) if len(finite_vals) else 1e-6
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(n_fm))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_ct))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"Removal benefit — {level}", fontsize=11)
    ax.set_xlabel("Failure mode")
    ax.set_ylabel("Cell type")

    for i in range(n_ct):
        for j in range(n_fm):
            val = vals[i, j]
            if np.isfinite(val) and abs(val) > benefit_threshold:
                ax.text(
                    j, i, f"{val:+.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > 0.6 * vmax else "black",
                )

    plt.colorbar(im, ax=ax, label="Removal benefit", fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax
