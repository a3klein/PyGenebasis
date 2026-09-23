"""
Marker coverage: does each cell type actually have genes that mark it.

The structural metrics can all look healthy while a particular cell type has no
gene in the panel that distinguishes it.  This measures that directly, per
(gene, label) pair, from the data rather than from provenance annotations.

Detection is binarised at ``> 0``, which is invariant to normalisation, so it
does not matter whether ``X`` holds counts or log-normalised values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

PSEUDO = 1e-3


def marker_detection(
    adata: AnnData,
    genes: list[str],
    label_key: str,
    *,
    min_detect: float = 0.10,
    min_fold: float = 2.0,
    layer: str | None = None,
) -> pd.DataFrame:
    """Within-label and outside-label detection for every (gene, label) pair.

    A gene counts as a usable marker for a label when it is detected in at
    least ``min_detect`` of that label's cells and is at least ``min_fold``
    enriched over the rest.  Both thresholds are choices: moving them moves the
    usable-marker count, so they are carried in the returned frame's
    ``.attrs``.

    Parameters
    ----------
    adata : AnnData
    genes : list[str]
        Panel genes.  Genes absent from ``adata`` are skipped.
    label_key : str
        Column of ``adata.obs`` holding the cell type labels.
    min_detect, min_fold : float
        Usability thresholds.
    layer : str, optional
        Layer to binarise instead of ``X``.

    Returns
    -------
    pd.DataFrame
        One row per (gene, label): ``gene``, ``label``, ``n_cells``,
        ``detect_in``, ``detect_out``, ``fold``, ``usable``.
    """
    import scipy.sparse

    present = [g for g in genes if g in adata.var_names]
    if not present:
        raise ValueError("none of the requested genes are in adata.var_names")
    if label_key not in adata.obs.columns:
        raise KeyError(f"{label_key!r} not in adata.obs")

    X = adata[:, present].layers[layer] if layer is not None else adata[:, present].X
    X = X if scipy.sparse.issparse(X) else scipy.sparse.csr_matrix(X)
    X = (X > 0).astype(np.float32)

    labels = adata.obs[label_key].astype(str).to_numpy()
    total = np.asarray(X.sum(axis=0)).ravel()
    n_all = X.shape[0]

    rows = []
    for lab in sorted(set(labels)):
        m = labels == lab
        n_in = int(m.sum())
        s_in = np.asarray(X[m].sum(axis=0)).ravel()
        d_in = s_in / max(n_in, 1)
        d_out = (total - s_in) / max(n_all - n_in, 1)
        rows.append(pd.DataFrame({
            "gene": present,
            "label": lab,
            "n_cells": n_in,
            "detect_in": d_in,
            "detect_out": d_out,
            "fold": (d_in + PSEUDO) / (d_out + PSEUDO),
        }))

    out = pd.concat(rows, ignore_index=True)
    out["usable"] = (out["detect_in"] >= min_detect) & (out["fold"] >= min_fold)
    out.attrs["min_detect"] = min_detect
    out.attrs["min_fold"] = min_fold
    return out


def coverage_by_label(
    detection: pd.DataFrame,
    *,
    sources: dict[str, str] | None = None,
    source_sep: str = ";",
) -> pd.DataFrame:
    """Collapse a ``marker_detection`` table to one row per label.

    Parameters
    ----------
    detection : pd.DataFrame
        Output of :func:`marker_detection`.
    sources : dict, optional
        ``gene -> source`` for the usable markers, e.g. ``"literature"`` or
        ``"geneBasis;literature"``.  A gene with several sources is reported
        under the combined name rather than assigned to one of them, so the
        counts still sum to ``n_usable_markers``.
    source_sep : str
        Separator within a multi-source string.

    Returns
    -------
    pd.DataFrame
        Indexed by label: ``n_cells``, ``n_usable_markers``, ``best_fold``,
        ``best_marker``, and one ``n_from_<source>`` column per source when
        ``sources`` is given.
    """
    rows = []
    for lab, d in detection.groupby("label", sort=True):
        use = d[d["usable"]]
        best = use.nlargest(1, "fold") if len(use) else d.nlargest(1, "fold")
        row = {
            "label": lab,
            "n_cells": int(d["n_cells"].iloc[0]),
            "n_usable_markers": int(len(use)),
            "best_fold": float(best["fold"].iloc[0]) if len(best) else np.nan,
            "best_marker": str(best["gene"].iloc[0]) if len(best) else None,
        }
        if sources is not None:
            combos = [
                source_sep.join(sorted(
                    s.strip() for s in str(sources.get(g, "unknown")).split(source_sep)
                    if s.strip()
                )) or "unknown"
                for g in use["gene"]
            ]
            for combo, n in pd.Series(combos, dtype=object).value_counts().items():
                row[f"n_from_{combo}"] = int(n)
        rows.append(row)

    out = pd.DataFrame(rows).set_index("label")
    src_cols = [c for c in out.columns if c.startswith("n_from_")]
    if src_cols:
        out[src_cols] = out[src_cols].fillna(0).astype(int)
    out.attrs.update(detection.attrs)
    return out
