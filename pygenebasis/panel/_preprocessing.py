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

Note on MT gene handling
-------------------------
When discard_mt=False, mitochondrial genes bypass the HVG filter and are always
retained.  This mirrors geneBasisR's behaviour: setting discard.mt=FALSE keeps
MT genes regardless of whether they are in the top HVGs.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def retain_informative_genes(
    adata: AnnData,
    *,
    n: int | None = None,
    var_thresh: float = 0.0,
    select_hvgs: bool = True,
    discard_mt: bool = True,
    flavor: str = "scran",
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
