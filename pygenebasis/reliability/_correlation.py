"""
pygenebasis/correlation.py
==========================
Co-expression reliability scoring: computes per-gene failure-mode assignments
from paired MERFISH + scRNA-seq expression matrices.

Given a targeted gene panel deployed on a spatial platform (e.g. MERFISH), this
module scores each gene's technical reliability by comparing its co-expression
profile in the spatial dataset to the scRNA-seq reference.  Four failure modes
are assigned::

    reliable               passes all filters — safe to use in kNN analyses
    probe_failure          co-expression profile fundamentally mismatched
    composition_mismatch   detection rate far lower in spatial than scRNA-seq
    idiosyncratic_noise    high per-gene residual after structured PCA denoising

The resulting ``results`` DataFrame (one row per gene) is the required input to
:func:`~pygenebasis.run_perturbation_analysis`.

Typical usage::

    # Two-step (allows inspecting / caching the correlation matrices)
    corr_data = compute_corr_matrices(
        merfish_adata, ref_adata,
        merfish_cell_type_key="cell_type",  # stratified MERFISH subsampling
        merfish_n_cells_per_type=2000,
        ref_subsample_frac=0.5,             # random ref subsampling
    )
    results, thresholds = score_gene_reliability(corr_data)

    # One-step convenience wrapper
    results, thresholds = compute_gene_reliability(
        merfish_adata, ref_adata,
        merfish_cell_type_key="cell_type",
        merfish_n_cells_per_type=2000,
    )

    fig, axes = plot_metric_distributions(results, thresholds)
    fig, ax   = plot_failure_mode_scatter(results, thresholds)

Subsampling
-----------
All subsampling parameters are optional for both datasets.  When ``None``
(default), all cells are used.  Two strategies are available for each:

* **Stratified** (``*_cell_type_key`` + ``*_n_cells_per_type``): caps each cell
  type at N cells, guaranteeing that rare types contribute to the correlation
  estimate.  Recommended when annotations are available.
* **Random** (``*_subsample_frac``): draws a fixed fraction of cells uniformly
  at random.  Simple but may under-represent rare types.

Speed notes
-----------
Pearson correlation is the main computational bottleneck.  Both datasets use a
**sparse-aware** implementation that avoids converting to dense:

    Cov(i,j) = (X.T @ X)[i,j] / n  −  μᵢ μⱼ

For an (n_cells × n_genes) sparse matrix this keeps the inner product in
scipy sparse arithmetic — O(nnz × n_genes) rather than O(n_cells × n_genes²).
At typical scRNA-seq sparsity (~90 % zeros) this is ~10× faster than the dense
path.  MERFISH panels are also often sparse relative to full transcriptomes, so
both datasets benefit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy import stats
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from anndata import AnnData


__all__ = [
    "compute_corr_matrices",
    "score_gene_reliability",
    "compute_gene_reliability",
    "plot_metric_distributions",
    "plot_failure_mode_scatter",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAILURE_MODE_COLORS: dict[str, str] = {
    "reliable":             "#378ADD",
    "probe_failure":        "#E24B4A",
    "composition_mismatch": "#EF9F27",
    "idiosyncratic_noise":  "#888780",
}

# Plot legend order: reliable first, then failure modes by severity
_FAILURE_MODE_ORDER = [
    "reliable",
    "probe_failure",
    "composition_mismatch",
    "idiosyncratic_noise",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_expression(
    adata: AnnData,
    genes: list[str],
    layer: str | None,
):
    """Return (n_cells × n_genes) expression array for *genes*, preserving sparsity.

    Backed AnnData: only the panel columns are loaded into memory — feasible
    even for large objects when the panel is small (hundreds of genes).
    """
    idx = adata.var_names.get_indexer(genes)
    X = adata.layers[layer] if layer is not None else adata.X
    X = X[:, idx]
    if hasattr(X, "toarray"):
        return X           # scipy sparse — keep as-is
    if hasattr(X, "compute"):
        return X.compute()  # dask → trigger load
    return X               # dense ndarray


def _subsample(
    adata: AnnData,
    cell_type_key: str | None,
    n_cells_per_type: int | None,
    subsample_frac: float | None,
    random_state: int,
    label: str,
) -> AnnData:
    """Apply optional subsampling to *adata*.

    Priority: stratified (cell_type_key + n_cells_per_type) > random (subsample_frac).
    Returns *adata* unchanged when no subsampling parameters are provided.
    """
    if cell_type_key is not None and n_cells_per_type is not None:
        rng = np.random.default_rng(random_state)
        labels = adata.obs[cell_type_key].values
        keep: list[int] = []
        for ct in np.unique(labels):
            ct_idx = np.where(labels == ct)[0]
            if len(ct_idx) > n_cells_per_type:
                ct_idx = rng.choice(ct_idx, n_cells_per_type, replace=False)
            keep.extend(ct_idx.tolist())
        return adata[np.sort(keep)]

    if subsample_frac is not None:
        if not 0.0 < subsample_frac <= 1.0:
            raise ValueError(
                f"{label}_subsample_frac must be in (0, 1], got {subsample_frac}"
            )
        rng = np.random.default_rng(random_state)
        n_keep = max(1, int(round(adata.n_obs * subsample_frac)))
        keep = rng.choice(adata.n_obs, min(n_keep, adata.n_obs), replace=False)
        return adata[np.sort(keep)]

    return adata  # no subsampling


def _pearson_corr_from_matrix(X) -> np.ndarray:
    """Gene × gene Pearson correlation from an (n_cells × n_genes) array.

    Handles dense (ndarray) and sparse (scipy.sparse) inputs.

    For sparse *X* the covariance identity
        Cov(i,j) = (X.T @ X)[i,j] / n  −  μᵢ μⱼ
    is used so the inner product stays in sparse arithmetic — typically ~10×
    faster than converting to dense for scRNA-seq data (~90 % zeros).

    Constant (all-zero) genes: diagonal set to 1, off-diagonal to 0.
    """
    n = X.shape[0]
    if sps.issparse(X):
        mu = np.asarray(X.mean(axis=0)).ravel()          # (n_genes,)
        XTX = np.asarray((X.T @ X).todense())             # (n_genes, n_genes)
        cov = XTX / n - np.outer(mu, mu)
    else:
        X = np.asarray(X, dtype=float)
        mu = X.mean(axis=0)
        X_c = X - mu
        cov = (X_c.T @ X_c) / n

    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = np.outer(std, std)
    zero = denom == 0.0
    denom[zero] = 1.0
    corr = cov / denom
    corr[zero] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def _choose_n_pcs(explained_variance_ratio: np.ndarray, threshold: float) -> int:
    """Minimum k such that sum(explained_variance_ratio[:k]) >= threshold.

    Uses ``np.searchsorted`` on the cumulative sum, so O(log n_components).
    Returns ``len(explained_variance_ratio)`` when the threshold cannot be
    reached (e.g. threshold=1.0 with floating-point residuals).
    """
    cumvar = np.cumsum(explained_variance_ratio)
    idx = np.searchsorted(cumvar, threshold)  # leftmost idx where cumvar[idx] >= threshold
    return min(int(idx) + 1, len(explained_variance_ratio))


def _parallel_analysis(
    diff_c: np.ndarray,
    n_max: int,
    *,
    n_permutations: int = 100,
    percentile: float = 95.0,
    random_state: int = 0,
    n_jobs: int = -1,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Choose the number of PCs by comparing to a permutation null distribution.

    For each of ``n_permutations`` permutations, each column of ``diff_c`` is
    shuffled independently — this destroys inter-gene correlations while
    preserving each gene's marginal distribution, producing the correct null for
    "how large would this eigenvalue be if there were no structured platform
    differences?"

    The ``percentile``-th percentile of the resulting null eigenvalue
    distribution at each rank is used as the threshold.  A conservative choice
    (default 95th) is appropriate here because over-fitting spurious PCs absorbs
    idiosyncratic signal, making noisy genes harder to detect, whereas
    under-fitting leaves structured variation in the residual but the Gamma fit
    adapts to the higher baseline.

    Permutation PCAs are run in parallel via threads; numpy's SVD (LAPACK)
    releases the GIL so threading is efficient without the pickling overhead of
    a process pool.

    Parameters
    ----------
    diff_c
        Row-mean-centred difference matrix (n_genes × n_genes).
    n_max
        Maximum number of PCA components to evaluate.
    n_permutations
        Number of permuted matrices to generate.  100 gives stable 95th-
        percentile estimates; increase to 200+ for higher percentiles.
    percentile
        Null distribution quantile used as the threshold (default 95.0, i.e.
        α = 0.05 per PC).
    random_state
        Seed for reproducibility.
    n_jobs
        Parallel workers.  -1 uses all available cores.

    Returns
    -------
    k : int
        Number of leading PCs whose eigenvalue exceeds the null threshold.
        Minimum 1 (at least one PC is always kept).
    real_eigenvalues : ndarray, shape (n_max,)
        Eigenvalues of ``diff_c``.
    null_threshold : ndarray, shape (n_max,)
        ``percentile``-th percentile of null eigenvalues at each rank.
    """
    from joblib import Parallel, delayed

    pca_real = PCA(n_components=n_max, random_state=random_state)
    pca_real.fit(diff_c)
    real_eigenvalues = pca_real.explained_variance_   # (n_max,)

    # Independent RNG seeds — one per permutation for reproducibility
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**31, size=n_permutations)

    def _one_permutation(seed: int) -> np.ndarray:
        rng_p = np.random.default_rng(seed)
        perm = diff_c.copy()
        for col in range(perm.shape[1]):
            rng_p.shuffle(perm[:, col])
        return PCA(n_components=n_max, random_state=0).fit(perm).explained_variance_

    null_eigs = np.array(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_permutation)(seed) for seed in seeds
        )
    )  # (n_permutations, n_max)

    null_threshold = np.percentile(null_eigs, percentile, axis=0)  # (n_max,)

    # Sequential stopping rule: k = position of first real eigenvalue at or
    # below the null threshold.  At least 1 PC is always kept.
    exceeds = real_eigenvalues > null_threshold
    k = n_max if exceeds.all() else max(1, int(np.argmax(~exceeds)))

    return k, real_eigenvalues, null_threshold


def _linear_rescale(corr_m: np.ndarray, corr_r: np.ndarray) -> np.ndarray:
    """Linearly rescale MERFISH correlations to match the reference scale.

    Fits ``ref_vals ~ slope * merfish_vals + intercept`` on upper-triangle
    elements, then applies the inverse transform to all elements of *corr_m*.
    Result is clipped to [-1, 1].
    """
    idx = np.triu_indices(corr_m.shape[0], k=1)
    slope, intercept, *_ = stats.linregress(corr_m[idx], corr_r[idx])
    return np.clip((corr_m - intercept) / slope, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_corr_matrices(
    merfish_adata: AnnData,
    ref_adata: AnnData,
    *,
    genes: list[str] | None = None,
    # MERFISH subsampling
    merfish_cell_type_key: str | None = None,
    merfish_n_cells_per_type: int | None = None,
    merfish_subsample_frac: float | None = None,
    # Reference subsampling
    ref_cell_type_key: str | None = None,
    ref_n_cells_per_type: int | None = None,
    ref_subsample_frac: float | None = None,
    # Shared
    layer: str | None = None,
    random_state: int = 0,
) -> dict:
    """Compute paired gene × gene Pearson correlation matrices.

    All subsampling parameters are optional.  When none are provided for a
    dataset, all cells are used (relying on sparse-aware computation for
    efficiency).

    Parameters
    ----------
    merfish_adata
        Spatial (MERFISH) AnnData.  Raw counts or log-normalised expression
        (must match *ref_adata*).
    ref_adata
        scRNA-seq reference AnnData.  May be backed; only panel genes are
        loaded into memory.
    genes
        Gene panel to score.  Defaults to the intersection of both objects'
        ``var_names``.
    merfish_cell_type_key
        ``obs`` column in *merfish_adata* for **stratified** subsampling.
        Recommended when annotations are available.
    merfish_n_cells_per_type
        Maximum cells per type for stratified MERFISH subsampling.
        Required when *merfish_cell_type_key* is provided.
    merfish_subsample_frac
        Fraction of MERFISH cells to keep at random.  Ignored when
        *merfish_cell_type_key* is provided.  ``None`` → use all cells.
    ref_cell_type_key
        ``obs`` column in *ref_adata* for **stratified** subsampling.
    ref_n_cells_per_type
        Maximum cells per type for stratified reference subsampling.
        Required when *ref_cell_type_key* is provided.
    ref_subsample_frac
        Fraction of reference cells to keep at random.  Ignored when
        *ref_cell_type_key* is provided.  ``None`` → use all cells.
    layer
        Expression layer in both objects (``None`` → ``adata.X``).
    random_state
        Seed for all random subsampling.

    Returns
    -------
    dict with keys:

        genes                  list[str] — gene order for all matrices
        merfish_corr           ndarray (n_genes, n_genes) — raw correlations
        ref_corr               ndarray (n_genes, n_genes)
        merfish_corr_rescaled  ndarray (n_genes, n_genes) — MERFISH rescaled to ref scale
        merfish_detect_rate    ndarray (n_genes,)
        ref_detect_rate        ndarray (n_genes,)
        detection_log2_ratio   ndarray (n_genes,)
    """
    if genes is None:
        genes = list(merfish_adata.var_names.intersection(ref_adata.var_names))

    m_adata = _subsample(
        merfish_adata,
        merfish_cell_type_key, merfish_n_cells_per_type, merfish_subsample_frac,
        random_state, label="merfish",
    )
    r_adata = _subsample(
        ref_adata,
        ref_cell_type_key, ref_n_cells_per_type, ref_subsample_frac,
        random_state, label="ref",
    )

    X_m = _extract_expression(m_adata, genes, layer)
    X_r = _extract_expression(r_adata, genes, layer)

    # Detection rates (computed on the full / subsampled expression matrices)
    eps = 1e-6
    if sps.issparse(X_m):
        detect_m = np.asarray((X_m > 0).mean(axis=0)).ravel()
    else:
        detect_m = (np.asarray(X_m) > 0).mean(axis=0)

    if sps.issparse(X_r):
        detect_r = np.asarray((X_r > 0).mean(axis=0)).ravel()
    else:
        detect_r = (np.asarray(X_r) > 0).mean(axis=0)

    detection_log2_ratio = np.log2((detect_m + eps) / (detect_r + eps))

    # Correlation matrices (sparse-aware for both)
    corr_m = _pearson_corr_from_matrix(X_m)
    corr_r = _pearson_corr_from_matrix(X_r)
    corr_m_rescaled = _linear_rescale(corr_m, corr_r)

    return {
        "genes":                 genes,
        "merfish_corr":          corr_m,
        "ref_corr":              corr_r,
        "merfish_corr_rescaled": corr_m_rescaled,
        "merfish_detect_rate":   detect_m,
        "ref_detect_rate":       detect_r,
        "detection_log2_ratio":  detection_log2_ratio,
    }


def score_gene_reliability(
    corr_data: dict,
    *,
    pca_method: str = "parallel_analysis",
    n_permutations: int = 100,
    pa_percentile: float = 95.0,
    max_pcs: int = 50,
    variance_threshold: float = 0.90,
    alpha: float = 0.01,
    fidelity_n_sigma: float = 2.0,
    detection_trim: tuple[float, float] = (0.10, 0.90),
    residual_bulk_pct: float = 90.0,
    random_state: int = 0,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict]:
    """Score gene reliability from pre-computed correlation matrices.

    Takes the dict returned by :func:`compute_corr_matrices` and produces
    per-gene failure-mode assignments.

    Separating correlation computation from scoring is useful when you want to:

    * Inspect or cache the correlation matrices before scoring.
    * Re-run with different thresholds without re-computing correlations.

    Parameters
    ----------
    corr_data
        Dict returned by :func:`compute_corr_matrices`.
    pca_method : {"parallel_analysis", "scree"}
        How to choose the number of PCs for denoising the difference matrix
        (default ``"parallel_analysis"``).

        ``"parallel_analysis"`` — generates ``n_permutations`` column-wise
        permutations of the difference matrix and keeps PCs whose eigenvalue
        exceeds the ``pa_percentile``-th percentile of the null distribution.
        This is parameter-free and data-adaptive, and gives a principled
        answer to "how many structured modes exist beyond pure chance."

        ``"scree"`` — keeps the minimum k PCs that explain at least
        ``variance_threshold`` cumulative variance (the original approach).
        Faster but the threshold is arbitrary and dataset-dependent.
    n_permutations
        Number of column-wise permutations for parallel analysis (default 100).
        Increase to 200+ when using high percentiles (> 99).
    pa_percentile
        Null distribution quantile used as the eigenvalue threshold (default
        95.0).  Higher values are more conservative (fewer PCs kept).
    max_pcs
        Upper bound on the number of PCA components evaluated (default 50).
        Applies to both methods.  The actual number is
        ``min(n_genes − 1, max_pcs)``.
    variance_threshold
        Cumulative variance threshold for the ``"scree"`` method (default
        0.90).  Ignored when ``pca_method="parallel_analysis"``.
    alpha
        Tail probability for threshold fitting (default 0.01).  Applied as the
        upper tail for the Gamma fit (residual variance) and the lower tail for
        the normal fit (detection log2 ratio).
    fidelity_n_sigma
        Standard deviations below the trimmed mean for the
        ``profile_similarity`` threshold (default 2.0).
    detection_trim
        (lo, hi) quantile bounds for trimming before fitting the detection
        log2-ratio normal distribution (default (0.10, 0.90)).
    residual_bulk_pct
        Percentile cap for the Gamma fit on residual variance; excludes the
        right tail before fitting (default 90.0).
    random_state
        Seed for PCA and permutation generation.
    n_jobs
        Parallel workers for permutation PCAs (default -1 = all cores).
        Only used when ``pca_method="parallel_analysis"``.

    Returns
    -------
    results : pd.DataFrame
        One row per gene, indexed by gene name.  Columns:

            profile_similarity    Spearman ρ of co-expression profile vs reference
            residual_variance     per-gene variance after PCA denoising of diff matrix
            residual_z            z-scored residual_variance
            pc1_loading           gene loading on PC1 of the difference matrix
            detection_log2_ratio  log2(merfish_detect / ref_detect)
            flag_fidelity         profile_similarity < profile_threshold
            flag_residual         residual_variance > residual_threshold
            flag_detection        detection_log2_ratio < detection_threshold
            reliable              not (flag_fidelity | flag_residual)
            detection_rate        raw MERFISH detection frequency ∈ [0, 1]
            failure_mode          one of {reliable, probe_failure,
                                   composition_mismatch, idiosyncratic_noise}

    thresholds : dict
        Fitted thresholds and distribution parameters::

            residual_threshold     float
            profile_threshold      float
            detection_threshold    float
            n_pcs_used             int   — number of PCs used
            variance_explained     float — cumulative variance explained by those PCs
            pca_method             str   — "parallel_analysis" or "scree"
            _fit                   dict  — distribution parameters for plotting
            _pca                   dict  — eigenvalues and null threshold for plotting
    """
    genes = corr_data["genes"]
    corr_m = corr_data["merfish_corr_rescaled"]
    corr_r = corr_data["ref_corr"]
    detection_log2_ratio = corr_data["detection_log2_ratio"]
    detect_m = corr_data["merfish_detect_rate"]
    n = len(genes)

    # ── Difference matrix → PCA → residual variance ──────────────────────────
    diff = corr_m - corr_r
    np.fill_diagonal(diff, np.nan)
    row_means = np.nanmean(diff, axis=1, keepdims=True)
    diff_c = diff - row_means
    np.fill_diagonal(diff_c, 0.0)

    n_max = min(n - 1, max_pcs)

    if pca_method == "parallel_analysis":
        k, real_eigenvalues, null_threshold = _parallel_analysis(
            diff_c, n_max,
            n_permutations=n_permutations,
            percentile=pa_percentile,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        # Fit the full PCA to get components for reconstruction
        pca = PCA(n_components=n_max, random_state=random_state)
        pca.fit(diff_c)
    elif pca_method == "scree":
        pca = PCA(n_components=n_max, random_state=random_state)
        pca.fit(diff_c)
        k = _choose_n_pcs(pca.explained_variance_ratio_, variance_threshold)
        real_eigenvalues = pca.explained_variance_
        null_threshold = None
    else:
        raise ValueError(
            f"pca_method must be 'parallel_analysis' or 'scree', got {pca_method!r}"
        )

    # Reconstruct using first k components (truncated inverse transform)
    scores = pca.transform(diff_c)                                    # (n, n_max)
    diff_recon = scores[:, :k] @ pca.components_[:k] + pca.mean_     # (n, n)

    residual = diff_c - diff_recon
    residual_var = np.var(residual, axis=1)
    residual_z = (residual_var - residual_var.mean()) / residual_var.std()

    # ── Profile similarity (per-gene Spearman of co-expression rows) ───────
    profile_sim = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        rho, _ = spearmanr(corr_m[i, mask], corr_r[i, mask])
        profile_sim[i] = rho

    # ── Results DataFrame ──────────────────────────────────────────────────
    results = pd.DataFrame(
        {
            "profile_similarity":   profile_sim,
            "residual_variance":    residual_var,
            "residual_z":           residual_z,
            "pc1_loading":          pca.components_[0],
            "detection_log2_ratio": detection_log2_ratio,
        },
        index=pd.Index(genes, name="gene"),
    )

    # ── Threshold fitting ──────────────────────────────────────────────────
    # Residual variance → Gamma (fit on bulk, threshold at upper tail)
    rv_bulk_cut = np.percentile(residual_var, residual_bulk_pct)
    rv_bulk = residual_var[residual_var <= rv_bulk_cut]
    if rv_bulk.std() < 1e-12:
        # Degenerate case: nearly constant residuals (e.g. identical input matrices).
        # Nothing should be flagged — use the empirical upper-tail percentile as the threshold,
        # which equals (or is very close to) the maximum observed value.
        residual_threshold = float(np.percentile(residual_var, 100.0 * (1 - alpha)))
        gamma_shape = 1.0
        gamma_scale = max(float(residual_var.mean()), 1e-10)
    else:
        # Gamma with floc=0 requires strictly positive values; add a small floor.
        gamma_shape, _, gamma_scale = stats.gamma.fit(rv_bulk + 1e-10, floc=0)
        residual_threshold = float(
            stats.gamma.ppf(1 - alpha, gamma_shape, loc=0, scale=gamma_scale)
        )

    # Detection log2 ratio → normal (trim tails, left-tail threshold)
    dl = detection_log2_ratio
    lo_q = float(np.quantile(dl, detection_trim[0]))
    hi_q = float(np.quantile(dl, detection_trim[1]))
    dl_bulk = dl[(dl >= lo_q) & (dl <= hi_q)]
    mu_dl, sigma_dl = stats.norm.fit(dl_bulk)
    detection_threshold = float(stats.norm.ppf(alpha, loc=mu_dl, scale=sigma_dl))

    # Profile similarity → skew-normal fit on 5–95 % bulk.
    # The distribution is strongly left-skewed (reliable genes cluster near ρ ≈ 0.7;
    # probe failures form a long left tail), so a symmetric Normal under-estimates
    # the mode.  A skew-normal gives a faithful visual fit while keeping the
    # fidelity_n_sigma rule (threshold = mean − n_sigma × std of the fitted dist).
    ps = profile_sim
    ps_bulk = ps[(ps >= np.quantile(ps, 0.05)) & (ps <= np.quantile(ps, 0.95))]
    try:
        a_ps, loc_ps, scale_ps = stats.skewnorm.fit(ps_bulk)
    except stats.FitError:
        # Near-constant bulk (e.g. identical input matrices) — fall back to a
        # symmetric fit so nothing is flagged.
        a_ps = 0.0
        loc_ps, scale_ps = stats.norm.fit(ps_bulk)
        scale_ps = max(float(scale_ps), 1e-12)
    mu_ps    = float(stats.skewnorm.mean(a_ps, loc=loc_ps, scale=scale_ps))
    sigma_ps = float(stats.skewnorm.std(a_ps,  loc=loc_ps, scale=scale_ps))
    profile_threshold = float(mu_ps - fidelity_n_sigma * sigma_ps)

    thresholds = {
        "residual_threshold":  residual_threshold,
        "profile_threshold":   profile_threshold,
        "detection_threshold": detection_threshold,
        "n_pcs_used":          k,
        "variance_explained":  float(np.cumsum(pca.explained_variance_ratio_)[k - 1]),
        "pca_method":          pca_method,
        "_fit": {
            "gamma_shape":     float(gamma_shape),
            "gamma_scale":     float(gamma_scale),
            "profile_a":       float(a_ps),
            "profile_loc":     float(loc_ps),
            "profile_scale":   float(scale_ps),
            "profile_mu":      mu_ps,
            "profile_sigma":   sigma_ps,
            "detection_mu":    float(mu_dl),
            "detection_sigma": float(sigma_dl),
        },
        "_pca": {
            "real_eigenvalues":        real_eigenvalues,
            "null_threshold":          null_threshold,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        },
    }

    # ── Flags and failure mode assignment ─────────────────────────────────
    results["flag_fidelity"]  = results["profile_similarity"]   < profile_threshold
    results["flag_residual"]  = results["residual_variance"]    > residual_threshold
    results["flag_detection"] = results["detection_log2_ratio"] < detection_threshold
    results["reliable"]       = ~(results["flag_fidelity"] | results["flag_residual"])
    results["detection_rate"] = detect_m

    # Priority order: composition_mismatch > probe_failure > idiosyncratic_noise
    def _classify(row: pd.Series) -> str:
        if row["reliable"]:
            return "reliable"
        if row["flag_detection"]:
            return "composition_mismatch"
        if row["flag_fidelity"]:
            return "probe_failure"
        return "idiosyncratic_noise"

    results["failure_mode"] = results.apply(_classify, axis=1)

    return results, thresholds


def compute_gene_reliability(
    merfish_adata: AnnData,
    ref_adata: AnnData,
    *,
    genes: list[str] | None = None,
    merfish_cell_type_key: str | None = None,
    merfish_n_cells_per_type: int | None = None,
    merfish_subsample_frac: float | None = None,
    ref_cell_type_key: str | None = None,
    ref_n_cells_per_type: int | None = None,
    ref_subsample_frac: float | None = None,
    layer: str | None = None,
    pca_method: str = "parallel_analysis",
    n_permutations: int = 100,
    pa_percentile: float = 95.0,
    max_pcs: int = 50,
    variance_threshold: float = 0.90,
    alpha: float = 0.01,
    fidelity_n_sigma: float = 2.0,
    detection_trim: tuple[float, float] = (0.10, 0.90),
    residual_bulk_pct: float = 90.0,
    random_state: int = 0,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict]:
    """Full co-expression reliability scoring pipeline.

    Convenience wrapper combining :func:`compute_corr_matrices` and
    :func:`score_gene_reliability`.  Use the two-step form if you want to
    inspect or cache the correlation matrices before scoring.

    See :func:`compute_corr_matrices` and :func:`score_gene_reliability` for
    full parameter documentation.

    Returns
    -------
    results : pd.DataFrame
    thresholds : dict
    """
    corr_data = compute_corr_matrices(
        merfish_adata,
        ref_adata,
        genes=genes,
        merfish_cell_type_key=merfish_cell_type_key,
        merfish_n_cells_per_type=merfish_n_cells_per_type,
        merfish_subsample_frac=merfish_subsample_frac,
        ref_cell_type_key=ref_cell_type_key,
        ref_n_cells_per_type=ref_n_cells_per_type,
        ref_subsample_frac=ref_subsample_frac,
        layer=layer,
        random_state=random_state,
    )
    return score_gene_reliability(
        corr_data,
        pca_method=pca_method,
        n_permutations=n_permutations,
        pa_percentile=pa_percentile,
        max_pcs=max_pcs,
        variance_threshold=variance_threshold,
        alpha=alpha,
        fidelity_n_sigma=fidelity_n_sigma,
        detection_trim=detection_trim,
        residual_bulk_pct=residual_bulk_pct,
        random_state=random_state,
        n_jobs=n_jobs,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric_distributions(
    results: pd.DataFrame,
    thresholds: dict,
    *,
    bins: int = 50,
    figsize: tuple[float, float] = (15, 4),
) -> tuple[plt.Figure, np.ndarray]:
    """Three-panel histogram of reliability metrics with fitted distributions.

    Panels (left → right):

    1. **Residual variance** — Gamma fit + threshold line.
    2. **Profile similarity** — trimmed-normal fit + threshold line.
    3. **Detection log₂ ratio** — trimmed-normal fit + threshold line.

    Parameters
    ----------
    results
        DataFrame from :func:`score_gene_reliability` or
        :func:`compute_gene_reliability`.
    thresholds
        Dict returned alongside *results*.
    bins
        Number of histogram bins (default 50).
    figsize
        Figure width × height in inches (default (15, 4)).

    Returns
    -------
    fig, axes
        Figure and length-3 array of Axes.
    """
    fit = thresholds["_fit"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ── Panel 1: Residual variance (Gamma) ────────────────────────────────
    rv = results["residual_variance"].values
    ax = axes[0]
    ax.hist(rv, bins=bins, density=True, color="steelblue", alpha=0.7, label="observed")

    x_rv = np.linspace(0, rv.max(), 500)
    ax.plot(
        x_rv,
        stats.gamma.pdf(x_rv, fit["gamma_shape"], loc=0, scale=fit["gamma_scale"]),
        color="darkorange", lw=2, label="Gamma fit",
    )
    ax.axvline(
        thresholds["residual_threshold"], color="red", linestyle="--",
        label=f"threshold = {thresholds['residual_threshold']:.3f}",
    )
    ax.set_xlabel("Residual variance")
    ax.set_ylabel("Density")
    ax.set_title("Residual variance")
    ax.legend(fontsize=8)

    # ── Panel 2: Profile similarity (trimmed normal) ───────────────────────
    ps = results["profile_similarity"].values
    ax = axes[1]
    ax.hist(ps, bins=bins, density=True, color="steelblue", alpha=0.7, label="observed")

    x_ps = np.linspace(ps.min(), ps.max(), 500)
    if "profile_a" in fit:
        pdf_ps = stats.skewnorm.pdf(x_ps, fit["profile_a"], loc=fit["profile_loc"], scale=fit["profile_scale"])
        fit_label_ps = "Skew-normal fit"
    else:
        pdf_ps = stats.norm.pdf(x_ps, fit["profile_mu"], fit["profile_sigma"])
        fit_label_ps = "Normal fit (trimmed)"
    ax.plot(x_ps, pdf_ps, color="darkorange", lw=2, label=fit_label_ps)
    ax.axvline(
        thresholds["profile_threshold"], color="red", linestyle="--",
        label=f"threshold = {thresholds['profile_threshold']:.3f}",
    )
    ax.set_xlabel("Profile similarity (Spearman ρ)")
    ax.set_ylabel("Density")
    ax.set_title("Profile similarity")
    ax.legend(fontsize=8)

    # ── Panel 3: Detection log₂ ratio (trimmed normal) ────────────────────
    dl = results["detection_log2_ratio"].values
    ax = axes[2]
    ax.hist(dl, bins=bins, density=True, color="steelblue", alpha=0.7, label="observed")

    x_dl = np.linspace(dl.min(), dl.max(), 500)
    ax.plot(
        x_dl,
        stats.norm.pdf(x_dl, fit["detection_mu"], fit["detection_sigma"]),
        color="darkorange", lw=2, label="Normal fit (trimmed)",
    )
    ax.axvline(
        thresholds["detection_threshold"], color="red", linestyle="--",
        label=f"threshold = {thresholds['detection_threshold']:.3f}",
    )
    ax.set_xlabel("Detection log₂ ratio (MERFISH / scRNA-seq)")
    ax.set_ylabel("Density")
    ax.set_title("Detection log₂ ratio")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig, axes


def plot_failure_mode_scatter(
    results: pd.DataFrame,
    thresholds: dict,
    *,
    size_scale: float = 30.0,
    alpha: float = 0.7,
    annotate_flagged: bool = True,
    figsize: tuple[float, float] = (7, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """2-D scatter of gene reliability metrics coloured by failure mode.

    Axes:

    * **X** — residual variance (bias: how much co-expression deviates from
      structured PCA patterns).
    * **Y** — profile similarity (fidelity: Spearman ρ of co-expression
      profile vs reference).
    * **Dot size** — detection log₂ ratio, normalised to a positive range so
      all dots are visible.  Larger = more detected in MERFISH relative to
      scRNA-seq.
    * **Colour** — failure mode.

    Threshold lines for X and Y mark the classification boundaries.

    Parameters
    ----------
    results
        DataFrame from :func:`score_gene_reliability` or
        :func:`compute_gene_reliability`.
    thresholds
        Dict returned alongside *results*.
    size_scale
        Scaling factor for dot sizes.  Sizes are linearly normalised from the
        minimum to the maximum detection log₂ ratio and multiplied by this
        value (default 30).
    alpha
        Marker transparency (default 0.7).
    annotate_flagged
        Label non-reliable genes with their gene name (default True).
    figsize
        Figure width × height in inches (default (7, 6)).

    Returns
    -------
    fig, ax
        Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=figsize)

    dl = results["detection_log2_ratio"].values
    # Shift to strictly positive range so minimum becomes 0.5 × size_scale
    sizes = (dl - dl.min() + 0.5) * size_scale

    for mode in _FAILURE_MODE_ORDER:
        mask = results["failure_mode"] == mode
        if not mask.any():
            continue
        ax.scatter(
            results.loc[mask, "residual_variance"],
            results.loc[mask, "profile_similarity"],
            s=sizes[mask.values],
            c=_FAILURE_MODE_COLORS[mode],
            alpha=alpha,
            edgecolors="none",
            label=f"{mode} (n={mask.sum()})",
        )

    ax.axvline(
        thresholds["residual_threshold"],
        color="red", linestyle="--", linewidth=1, alpha=0.6,
        label="residual threshold",
    )
    ax.axhline(
        thresholds["profile_threshold"],
        color="darkorange", linestyle="--", linewidth=1, alpha=0.6,
        label="fidelity threshold",
    )

    if annotate_flagged:
        for gene, row in results[~results["reliable"]].iterrows():
            ax.annotate(
                gene,
                (row["residual_variance"], row["profile_similarity"]),
                fontsize=6, alpha=0.8, ha="left", va="bottom",
            )

    ax.set_xlabel("Residual variance (bias)")
    ax.set_ylabel("Profile similarity (fidelity)")
    ax.set_title("Gene reliability — failure modes")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig, ax
