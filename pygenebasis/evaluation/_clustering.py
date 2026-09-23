"""
Panel-only clustering: would unsupervised analysis rediscover the populations.

Every other metric compares the panel against labels it is given.  This one
asks the question you actually face with measured spatial data, where no labels
exist: cluster on the panel genes alone and see whether the reference
populations come back.

A panel can map at 0.98 by kNN vote and still cluster into something
unrecognisable, so this is not implied by the mapping accuracy.

**Re-embed and re-cluster your own data.** How to embed, correct and cluster is
a judgement call, and it is worth looking at the result before scoring it.
``cluster_on_panel`` is a reasonable starting point, not a recommendation; the
metrics take label vectors, so any clustering works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def cluster_on_panel(
    adata: AnnData,
    genes: list[str],
    *,
    batch_key: str | None = None,
    resolution: float = 1.0,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    layer: str | None = None,
    random_state: int = 32,
) -> pd.Series:
    """Cluster cells using only the panel genes.

    PCA on the panel genes, optional Harmony on ``batch_key``, then Leiden.
    This is a convenience: :func:`cluster_agreement` and
    :func:`agreement_crosstab` take label vectors, so any clustering will do.

    Requires ``igraph`` and ``leidenalg``, which the metric functions do not.

    ``resolution`` is load-bearing: ARI in particular moves with the number of
    clusters produced, so report it alongside any score.

    Returns
    -------
    pd.Series
        Cluster labels as strings, indexed by ``adata.obs_names``.
    """
    import scanpy as sc

    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "cluster_on_panel needs igraph and leidenalg. Either install them, "
            "or cluster however you like and pass the labels straight to "
            "cluster_agreement / agreement_crosstab."
        ) from exc

    present = [g for g in genes if g in adata.var_names]
    if not present:
        raise ValueError("none of the requested genes are in adata.var_names")

    sub = adata[:, present].copy()
    if layer is not None:
        sub.X = sub.layers[layer]
    sub.obs = sub.obs.copy()

    sc.pp.pca(sub, n_comps=min(n_pcs, len(present) - 1, sub.n_obs - 1),
              random_state=random_state)
    if batch_key is not None:
        # our own harmonypy wrapper, not scanpy's — scanpy's returns a
        # transposed Z_corr against the installed harmonypy
        from ..knn._backends.batch import get_batch_corrector
        corrector = get_batch_corrector("harmony", random_state=random_state)
        sub.obsm["X_pca_harmony"] = corrector.fit_transform(
            sub.obsm["X_pca"], sub.obs[batch_key].to_numpy()
        )
    rep = "X_pca_harmony" if batch_key is not None else "X_pca"
    sc.pp.neighbors(sub, use_rep=rep, n_neighbors=n_neighbors,
                    random_state=random_state)
    sc.tl.leiden(sub, resolution=resolution, key_added="panel_leiden",
                 random_state=random_state, flavor="igraph", n_iterations=2,
                 directed=False)
    return pd.Series(sub.obs["panel_leiden"].astype(str).to_numpy(),
                     index=adata.obs_names, name="panel_cluster")


def cluster_agreement(labels_true, labels_pred) -> dict:
    """Chance-corrected agreement between two partitions of the same cells.

    Both work on cell *pairs*, so neither needs the two partitions to share
    names or even to have the same number of groups.

    ARI counts pairs that are together-in-both or apart-in-both, corrected for
    the agreement expected at random.  AMI is the information-theoretic
    analogue.  They differ in how they treat splitting: breaking one true
    population into several clusters destroys many within-group pairs and hurts
    ARI badly, while each fragment still identifies its label, so AMI barely
    moves.  A large ARI-AMI gap therefore means over-splitting, and ARI on its
    own partly measures whether the clustering resolution happens to match the
    granularity of the reference labels.

    Returns
    -------
    dict
        ``ARI``, ``AMI``, ``n_true_levels``, ``n_pred_levels``.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    a = np.asarray(labels_true).astype(str)
    b = np.asarray(labels_pred).astype(str)
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.shape} vs {b.shape}")
    return {
        "ARI": float(adjusted_rand_score(a, b)),
        "AMI": float(adjusted_mutual_info_score(a, b)),
        "n_true_levels": int(len(set(a))),
        "n_pred_levels": int(len(set(b))),
    }


def agreement_crosstab(labels_true, labels_pred, *, order: bool = True) -> pd.DataFrame:
    """Row-normalised contingency table, optionally ordered to read diagonally.

    With ``order=True`` the rows and columns are permuted so that matched pairs
    sit on the diagonal: an optimal one-to-one assignment fixes the row order,
    then each remaining cluster is placed next to the row it most belongs to.
    Fragments of a split population therefore end up adjacent instead of
    scattered across the width.

    Returns
    -------
    pd.DataFrame
        Rows = true labels, columns = predicted clusters, rows summing to 1.
    """
    a = pd.Series(np.asarray(labels_true).astype(str), name="true")
    b = pd.Series(np.asarray(labels_pred).astype(str), name="pred")
    ct = pd.crosstab(a, b, normalize="index")
    if not order or ct.empty:
        return ct

    from scipy.optimize import linear_sum_assignment

    # best 1-1 matching on the row-normalised table
    rows, cols = linear_sum_assignment(-ct.to_numpy())
    matched_row = {int(r): int(c) for r, c in zip(rows, cols)}

    row_order = sorted(range(ct.shape[0]),
                       key=lambda r: (r not in matched_row, matched_row.get(r, 10**6)))
    ct = ct.iloc[row_order]

    # each column follows the row it most belongs to, strongest first
    dominant = ct.to_numpy().argmax(axis=0)
    strength = ct.to_numpy().max(axis=0)
    col_order = sorted(range(ct.shape[1]), key=lambda c: (dominant[c], -strength[c]))
    return ct.iloc[:, col_order]
