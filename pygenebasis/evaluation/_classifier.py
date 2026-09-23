"""
Panel vs full feature set, as a supervised classifier.

The kNN metrics ask whether same-type cells are each other's neighbours, and
the clustering metrics ask whether the populations separate by density.  This
asks a third thing: is the information still linearly decodable when you keep
only the panel.  A logistic regression is given the labels to train on, so it
is an upper bound on what a downstream analyst could extract — and it can
succeed on rare types where a kNN vote fails for want of same-type neighbours.

The gap between the two fits is the measured cost of restricting measurement to
the panel.  It is routinely near zero or negative: a curated panel can beat an
unfiltered transcriptome, because most genes are noise to a linear model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def _fit_one(X, y, feature_names, *, test_size, C, max_iter, random_state,
             keep_coefs):
    import scipy.sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)
    X = np.nan_to_num(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    # with_mean=False keeps sparse input viable and matches the eval suite
    scaler = StandardScaler(with_mean=False).fit(X_tr)
    clf = LogisticRegression(max_iter=max_iter, C=C).fit(
        scaler.transform(X_tr), y_tr)
    pred = clf.predict(scaler.transform(X_te))

    labels = sorted(set(y_te))
    summary = {
        "n_features": X.shape[1],
        "n_cells": X.shape[0],
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_F1": float(f1_score(y_te, pred, average="macro")),
        "weighted_F1": float(f1_score(y_te, pred, average="weighted")),
    }
    per_label = pd.DataFrame({
        "celltype": labels,
        "F1": f1_score(y_te, pred, average=None, labels=labels),
        "n_test": [int((y_te == lab).sum()) for lab in labels],
    })
    coefs = None
    if keep_coefs:
        imp = (np.abs(clf.coef_).max(axis=0) if clf.coef_.ndim > 1
               else np.abs(clf.coef_).ravel())
        coefs = pd.DataFrame({"gene": feature_names, "max_abs_coef": imp})
    confusion = pd.crosstab(pd.Series(y_te, name="celltype"),
                            pd.Series(pred, name="predicted"),
                            normalize="index")
    return summary, per_label, coefs, confusion


def classifier_gap(
    adata: AnnData,
    genes: list[str],
    label_key: str,
    *,
    genes_all: list[str] | None = None,
    n_cells: int = 25_000,
    test_size: float = 0.3,
    min_cells_per_label: int = 10,
    C: float = 1.0,
    max_iter: int = 200,
    layer: str | None = None,
    random_state: int = 32,
) -> dict:
    """Fit a cell type classifier on the panel and on the full feature set.

    Both fits use the same cells and the same split, so the difference is
    attributable to the features alone.

    Parameters
    ----------
    adata : AnnData
    genes : list[str]
        Panel genes.
    label_key : str
        Column of ``adata.obs`` with the labels to predict.
    genes_all : list[str], optional
        The comparison feature set.  Defaults to every gene in ``adata``.
    n_cells : int
        Cap on cells used.  Subsampling happens on indices, and only the
        subsample is ever densified.
    min_cells_per_label : int
        Labels rarer than this are dropped — a stratified split needs at least
        one cell of each label on both sides.
    C, max_iter : float, int
        Passed to ``LogisticRegression``.
    layer : str, optional
    random_state : int

    Returns
    -------
    dict
        ``summary`` (one row per feature set), ``per_label`` (F1 per cell type
        per feature set), ``gene_importance`` (panel fit only, max |coef|
        across labels), ``confusion`` (panel fit, row-normalised), and
        ``dropped_labels``.
    """
    import scipy.sparse

    if label_key not in adata.obs.columns:
        raise KeyError(f"{label_key!r} not in adata.obs")
    panel = [g for g in genes if g in adata.var_names]
    if not panel:
        raise ValueError("none of the requested genes are in adata.var_names")
    full = list(adata.var_names) if genes_all is None else \
        [g for g in genes_all if g in adata.var_names]

    y_all = adata.obs[label_key].astype(str).to_numpy()
    counts = pd.Series(y_all).value_counts()
    keep = set(counts[counts >= min_cells_per_label].index)
    dropped = sorted(set(counts.index) - keep)
    idx = np.where(np.isin(y_all, list(keep)))[0]
    if idx.size == 0:
        raise ValueError(
            f"no label has at least {min_cells_per_label} cells")
    if idx.size > n_cells:
        idx = np.random.RandomState(random_state).choice(
            idx, n_cells, replace=False)
    idx = np.sort(idx)
    y = y_all[idx]

    X_src = adata.layers[layer] if layer is not None else adata.X
    X_src = X_src if scipy.sparse.issparse(X_src) else \
        scipy.sparse.csr_matrix(X_src)
    X_src = X_src[idx]
    col = {g: i for i, g in enumerate(adata.var_names)}

    summaries, per_labels = [], []
    gene_importance = confusion = None
    for name, feats in (("panel", panel), ("full", full)):
        X = X_src[:, [col[g] for g in feats]]
        summary, per_label, coefs, conf = _fit_one(
            X, y, feats, test_size=test_size, C=C, max_iter=max_iter,
            random_state=random_state, keep_coefs=(name == "panel"))
        summary["feature_set"] = name
        summaries.append(summary)
        per_label["feature_set"] = name
        per_labels.append(per_label)
        if name == "panel":
            gene_importance = coefs.sort_values("max_abs_coef", ascending=False)
            confusion = conf

    summary_df = pd.DataFrame(summaries)[
        ["feature_set", "n_features", "n_cells", "accuracy",
         "macro_F1", "weighted_F1"]]
    return {
        "summary": summary_df,
        "per_label": pd.concat(per_labels, ignore_index=True),
        "gene_importance": gene_importance,
        "confusion": confusion,
        "dropped_labels": dropped,
    }
