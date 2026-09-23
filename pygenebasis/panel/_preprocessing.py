"""
Gene pre-filtering before panel selection.

Removing uninformative genes before running gene_search is critical for
both speed and result quality:
  - Speed: the Minkowski distance computation scales with n_candidate_genes.
    Cutting from ~30k to ~3k genes gives a ~10× speedup per iteration.
  - Quality: low-variance genes add noise to the kNN graph without carrying
    useful signal.

Implementation matches geneBasisR::retain_informative_genes:
  - Selects highly variable genes (HVGs)
  - Optionally removes mitochondrial genes by prefix

HVG backends
------------
Two HVG selection methods are available via the ``flavor`` parameter:

``flavor="scran"`` (default, recommended):
    Uses scranpy.model_gene_variances (a Python reimplementation of
    scran::modelGeneVar) + scranpy.choose_highly_variable_genes.  Fits a
    LOWESS trend to the mean-variance relationship and retains genes whose
    residual variance exceeds the trend (biological CV > technical noise).
    This matches the algorithm used by geneBasisR and gives ~80%+ HVG
    overlap with R in practice.  Requires the ``scranpy`` package.

``flavor="seurat"`` / ``flavor="seurat_v3"``:
    Delegates to scanpy.pp.highly_variable_genes with the given flavor.
    "seurat" uses mean/dispersion bins on log-normalised data (~30-40%
    overlap with R); "seurat_v3" uses raw counts.  Available without extra
    dependencies.

``flavor="methylation"``:
    For mCH/mCG rate matrices, where dispersion must be normalised within
    **mean x coverage** bins rather than mean alone.  Requires per-gene mean
    coverage in ``adata.var`` (see ``coverage_key``).  Ported from ALLCools
    ``highly_variable_methylation_feature``; see
    ``highly_variable_methylation_features`` below.

Note on MT gene handling
-------------------------
When discard_mt=False, mitochondrial genes bypass the HVG filter and are always
retained.  This mirrors geneBasisR's behaviour: setting discard.mt=FALSE keeps
MT genes regardless of whether they are in the top HVGs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from anndata import AnnData

log = logging.getLogger(__name__)


def retain_informative_genes(
    adata: AnnData,
    *,
    n: int | None = None,
    var_thresh: float = 0.0,
    select_hvgs: bool = True,
    discard_mt: bool = True,
    flavor: str = "scran",
    coverage_key: str = "cov_mean",
    layer: str | None = None,
    inplace: bool = False,
) -> AnnData | None:
    """Filter genes to retain only informative (highly variable) ones.

    Replicates geneBasisR::retain_informative_genes.  Should be run before
    ``gene_search`` to reduce the candidate gene pool.

    Parameters
    ----------
    adata : AnnData
        Single-cell dataset.
    n : int, optional
        Number of top HVGs to keep.  If None, retains all genes whose
        residual variance exceeds the mean-variance trend (scran flavor)
        or uses scanpy's default mean/dispersion cutoffs (seurat flavors).
    var_thresh : float
        For seurat flavors: minimum normalised dispersion to retain a gene.
        Ignored for flavor="scran".
        Default 0.0 retains all HVGs selected by the method.
    select_hvgs : bool
        If True, apply HVG selection.  Set to False to only apply the MT
        gene filter.
    discard_mt : bool
        Remove mitochondrial genes (prefixes: "MT-", "mt-", "Mt-").
        Default True matches geneBasisR.
        When False, MT genes bypass the HVG filter and are always retained.
    flavor : str
        HVG selection backend:

        - ``"scran"`` (default): uses scranpy.model_gene_variances to fit a
          LOWESS mean-variance trend and selects genes with positive residual
          variance.  Closely matches geneBasisR (scran::modelGeneVar).
          Requires ``scranpy`` to be installed.
        - ``"seurat"`` / ``"seurat_v3"``: delegates to
          ``scanpy.pp.highly_variable_genes``.  "seurat" works on
          log-normalised data; "seurat_v3" requires raw counts.
        - ``"methylation"``: for mCH/mCG rate matrices; bins by mean and
          coverage.  Needs ``coverage_key`` in ``adata.var``.
    coverage_key : str
        For flavor="methylation": column in ``adata.var`` holding per-gene mean
        coverage.  Nothing computes this automatically — it comes from the MCDS
        via ``add_feature_cov_mean`` and must be carried over when the AnnData
        is built.
    layer : str, optional
        AnnData layer to use for HVG calculation.
    inplace : bool
        If True, filters ``adata`` in-place and returns None.
        If False (default), returns a filtered copy.

    Returns
    -------
    AnnData or None
        Filtered dataset (or None if inplace=True).
    """
    # Identify MT genes by prefix
    is_mt = np.array(
        adata.var_names.str.startswith("MT-")
        | adata.var_names.str.startswith("mt-")
        | adata.var_names.str.startswith("Mt-"),
        dtype=bool,
    )

    # Start with all genes kept
    keep_mask = np.ones(adata.n_vars, dtype=bool)

    if select_hvgs:
        if flavor == "scran":
            keep_mask &= _hvg_scran(adata, n=n, layer=layer)
        elif flavor == "methylation":
            hvf = highly_variable_methylation_features(
                adata, coverage_key=coverage_key, n_top_feature=n, layer=layer,
            )
            keep_mask &= hvf["feature_select"].to_numpy(dtype=bool)
        else:
            keep_mask &= _hvg_scanpy(adata, n=n, flavor=flavor,
                                     layer=layer, var_thresh=var_thresh)

    if discard_mt:
        keep_mask &= ~is_mt
    else:
        keep_mask |= is_mt

    if inplace:
        adata._inplace_subset_var(keep_mask)
        return None
    else:
        return adata[:, keep_mask].copy()


def highly_variable_methylation_features(
    adata: AnnData,
    *,
    coverage_key: str = "cov_mean",
    layer: str | None = None,
    n_top_feature: int | None = None,
    min_disp: float = 0.5,
    max_disp: float | None = None,
    min_mean: float = 0.0,
    max_mean: float = 5.0,
    bin_min_features: int = 5,
    mean_binsize: float = 0.05,
    cov_binsize: float = 100.0,
) -> pd.DataFrame:
    """Highly variable methylation features, normalised within mean x coverage bins.

    Port of ALLCools ``highly_variable_methylation_feature`` to AnnData.  RNA HVG
    methods bin by mean alone; methylation rates also depend on coverage, so a
    poorly covered gene looks variable for reasons that have nothing to do with
    biology.  Dispersion is therefore z-scored within a (mean, coverage) bin.

    Two upstream bugs are fixed here.  ALLCools crashes under pandas >= 2 (a
    ``reset_index`` column collision), and its ``n_top_feature`` is ignored — the
    cutoff is hardcoded at 5000, so asking for 3000 silently returns 5000.
    Everything else follows upstream exactly, including the bin-merging quirk
    noted below, so the selected features match ALLCools gene for gene.

    Parameters
    ----------
    adata : AnnData
        ``X`` (or ``layer``) holds the methylation **rate** — mCH/CH or mCG/CG —
        with cells in rows.
    coverage_key : str
        Column of ``adata.var`` holding per-gene mean coverage.  In ALLCools this
        is the ``{var_dim}_cov_mean`` coordinate produced by
        ``MCDS.add_feature_cov_mean``; nothing computes it automatically, so it
        must be attached when the AnnData is built.
    layer : str, optional
        Layer to use instead of ``X``.
    n_top_feature : int, optional
        Take this many features by normalised dispersion.  When given, every
        cutoff below is ignored.  When None (the ALLCools default), selection
        uses the cutoffs instead.
    min_disp, max_disp, min_mean, max_mean : float
        Cutoff-mode thresholds, applied only when ``n_top_feature`` is None.
    bin_min_features : int
        Bins with at most this many features are merged into the nearest bin.
    mean_binsize, cov_binsize : float
        Bin widths for mean and coverage.  ALLCools defaults.

    Returns
    -------
    pd.DataFrame
        One row per gene, indexed by ``adata.var_names``: ``mean``,
        ``dispersion``, ``cov``, ``mean_bin``, ``cov_bin``, ``dispersion_norm``
        and the boolean ``feature_select``.
    """
    import scipy.sparse

    if coverage_key not in adata.var.columns:
        raise KeyError(
            f"{coverage_key!r} not in adata.var. Methylation HVF needs per-gene mean "
            f"coverage; in ALLCools this is the '{{var_dim}}_cov_mean' coordinate from "
            f"MCDS.add_feature_cov_mean(), which must be carried onto the AnnData."
        )

    X = adata.layers[layer] if layer is not None else adata.X
    n_obs = X.shape[0]
    if n_obs < 2:
        raise ValueError("need at least 2 cells to estimate dispersion")

    # Mean and unbiased variance, matching ALLCools get_mean_dispersion.
    # ALLCools works on xarray, whose .mean(dim=...) skips NaN — and methylation
    # rate matrices carry NaN wherever a gene has no coverage in a cell, so the
    # means must be nan-aware or every gene comes out NaN.  The Bessel correction
    # below still uses the total cell count, not the per-gene non-NaN count,
    # because that is what upstream does.
    if scipy.sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    else:
        X = np.asarray(X)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(X, axis=0)
            mean_sq = np.nanmean(X * X, axis=0)
    abs_mean = np.abs(mean)
    var = (mean_sq - abs_mean ** 2) * (n_obs / (n_obs - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = np.log(var / np.where(abs_mean > 1e-12, abs_mean, 1e-12) + 1e-12)

    cov = np.asarray(adata.var[coverage_key], dtype=float)
    low_cov_portion = (cov < 10).sum() / cov.size
    if low_cov_portion > 0.2:
        log.warning(
            "%d%% of features have < 10 mean coverage; consider filtering by coverage "
            "first, or low-coverage features may be elevated by normalisation.",
            int(low_cov_portion * 100),
        )

    df = pd.DataFrame(
        {"mean": mean, "dispersion": dispersion, "cov": cov},
        index=adata.var_names.copy(),
    )
    df["mean_bin"] = (df["mean"] / mean_binsize).astype(int)
    df["cov_bin"] = (df["cov"] / cov_binsize).astype(int)

    # Bin counts.  ALLCools uses groupby.apply(...).reset_index() here, which
    # collides with the existing columns under pandas >= 2; size() is equivalent.
    bin_count = (
        df.groupby(["mean_bin", "cov_bin"]).size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    bin_more_than = bin_count[bin_count["count"] > bin_min_features]
    if bin_more_than.shape[0] == 0:
        raise ValueError(
            f"No bin has more than {bin_min_features} features, use a larger bin size."
        )

    # Merge sparse bins into the nearest well-populated bin (Manhattan distance).
    # Upstream has a dead `if count > 1` branch here whose assignment is always
    # overwritten by the unconditional one; replicated by simply always remapping.
    # A bin that is itself well-populated is at distance 0 from itself, so it maps
    # to itself and is unaffected.
    index_map = {}
    for mean_id, cov_id in zip(bin_count["mean_bin"], bin_count["cov_bin"]):
        manhattan = ((bin_more_than["mean_bin"] - mean_id).abs()
                     + (bin_more_than["cov_bin"] - cov_id).abs())
        closest = bin_more_than.loc[manhattan.sort_values().index[0]]
        index_map[(mean_id, cov_id)] = (int(closest["mean_bin"]), int(closest["cov_bin"]))

    remapped = [index_map[k] for k in zip(df["mean_bin"], df["cov_bin"])]
    df["mean_bin"] = [m for m, _ in remapped]
    df["cov_bin"] = [c for _, c in remapped]

    grouped = df.groupby(["mean_bin", "cov_bin"])["dispersion"]
    disp_mean_bin = grouped.mean()
    disp_std_bin = grouped.std(ddof=1)
    keys = list(zip(df["mean_bin"], df["cov_bin"]))
    df["dispersion_norm"] = (
        (df["dispersion"].values - disp_mean_bin.loc[keys].values)
        / disp_std_bin.loc[keys].values
    )

    if n_top_feature is not None:
        # upstream hardcodes 5000 here, ignoring the argument
        top = df.sort_values("dispersion_norm", ascending=False).index[:n_top_feature]
        feature_select = df.index.isin(top)
    else:
        dispersion_norm = df["dispersion_norm"].values.astype("float32")
        dispersion_norm[np.isnan(dispersion_norm)] = 0  # as Seurat does
        feature_select = np.logical_and.reduce((
            df["mean"].values > min_mean,
            df["mean"].values < max_mean,
            dispersion_norm > min_disp,
            dispersion_norm < (np.inf if max_disp is None else max_disp),
        ))
    df["feature_select"] = feature_select
    return df


# ---------------------------------------------------------------------------
# Private HVG backends
# ---------------------------------------------------------------------------

def _get_X(adata: AnnData, layer: str | None) -> np.ndarray:
    """Extract dense float64 expression matrix (cells × genes)."""
    import scipy.sparse
    X = adata.X if layer is None else adata.layers[layer]
    if scipy.sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def _hvg_scran(adata: AnnData, *, n: int | None, layer: str | None) -> np.ndarray:
    """Return a boolean mask of HVGs using scranpy (scran::modelGeneVar).

    scranpy.model_gene_variances expects rows=genes, cols=cells, so we
    transpose adata.X.  Genes with positive residual (bio variance > tech
    noise) are selected; if n is given, the top-n by residual are taken.
    """
    import scranpy

    X = _get_X(adata, layer)  # cells × genes
    result = scranpy.model_gene_variances(X.T)  # genes × cells
    residuals = np.array(result["statistics"]["residual"])

    if n is not None:
        chosen_idx = scranpy.choose_highly_variable_genes(residuals, top=n)
    else:
        # bound=0.0 mirrors scran::getTopHVGs(var.field="bio", var.threshold=0)
        chosen_idx = scranpy.choose_highly_variable_genes(
            residuals, top=len(residuals), bound=0.0
        )

    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[chosen_idx] = True
    return mask


def _hvg_scanpy(
    adata: AnnData,
    *,
    n: int | None,
    flavor: str,
    layer: str | None,
    var_thresh: float,
) -> np.ndarray:
    """Return a boolean mask of HVGs using scanpy."""
    import scanpy as sc

    adata_tmp = adata.copy()
    if n is not None:
        sc.pp.highly_variable_genes(
            adata_tmp, n_top_genes=min(n, adata.n_vars),
            flavor=flavor, layer=layer,
        )
    else:
        sc.pp.highly_variable_genes(adata_tmp, flavor=flavor, layer=layer)

    mask = adata_tmp.var["highly_variable"].values.copy()

    if var_thresh > 0.0 and "dispersions_norm" in adata_tmp.var.columns:
        mask &= adata_tmp.var["dispersions_norm"].values >= var_thresh

    return mask
