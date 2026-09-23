"""
Figures for panel evaluation.

Deliberately light on text — what each one means, and the caveats, live in
``docs/panel_evaluation.md``.  All return ``(fig, ax)`` and none call
``plt.show()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _annotate(ax, x, y, labels, *, fontsize=7):
    """Label points, nudging them apart when adjustText is available."""
    texts = [ax.text(xi, yi, s, fontsize=fontsize) for xi, yi, s in zip(x, y, labels)]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.4,
                                                  color="0.5"))
    except ImportError:
        pass
    return texts


def plot_preservation_violin(
    scores,
    labels,
    *,
    sort_by: str = "median",
    figsize: tuple[float, float] | None = None,
    max_points: int = 400,
    clip_quantiles: tuple[float, float] | None = (0.005, 0.995),
    random_state: int = 0,
):
    """Per-cell preservation, one violin per cell type, sorted by median.

    Parameters
    ----------
    scores : array-like
        Per-cell preservation score.
    labels : array-like
        Per-cell cell type, same order.
    sort_by : {"median", "mean", "name"}
    max_points : int
        Cap on the jittered points drawn per violin; the violin itself always
        uses every cell.
    clip_quantiles : tuple or None
        View limits, as quantiles of the pooled scores.  The score is a ratio
        and unbounded, so a handful of extreme cells otherwise compress every
        violin into a sliver.  Nothing is dropped — only the x-range is set.
        Pass None to show the full range.

    Returns
    -------
    (fig, ax)
    """
    s = pd.Series(np.asarray(scores, dtype=float),
                  index=np.asarray(labels).astype(str)).dropna()
    groups = {k: v.to_numpy() for k, v in s.groupby(level=0)}
    if sort_by == "name":
        order = sorted(groups)
    else:
        stat = np.median if sort_by == "median" else np.mean
        order = sorted(groups, key=lambda k: stat(groups[k]))

    n = len(order)
    fig, ax = plt.subplots(figsize=figsize or (8, max(2.5, 0.34 * n + 1.2)))
    rng = np.random.default_rng(random_state)

    parts = ax.violinplot([groups[k] for k in order], positions=range(n),
                          orientation="horizontal", showextrema=False, widths=0.9)
    for b in parts["bodies"]:
        b.set_facecolor("#c6dbef")
        b.set_edgecolor("none")
        b.set_alpha(0.9)

    for i, k in enumerate(order):
        v = groups[k]
        draw = v if len(v) <= max_points else rng.choice(v, max_points, replace=False)
        ax.scatter(draw, np.full(len(draw), i) + rng.uniform(-0.13, 0.13, len(draw)),
                   s=1.5, color="0.25", alpha=0.35, linewidths=0, rasterized=True)
        ax.scatter([np.mean(v)], [i], s=18, color="black", zorder=3)
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color="crimson", lw=1.6,
                zorder=3)

    ax.set_yticks(range(n))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("per-cell preservation score")
    ax.set_ylim(-0.8, n - 0.2)

    if clip_quantiles is not None:
        lo, hi = np.quantile(s.to_numpy(), clip_quantiles)
        # never clip away a group's own summary — a small, badly preserved type
        # sits outside the pooled quantiles and is exactly what must stay visible
        stats = [f(v) for v in groups.values() for f in (np.median, np.mean)]
        lo, hi = min(lo, min(stats)), max(hi, max(stats))
        pad = 0.04 * max(hi - lo, 1e-6)
        ax.set_xlim(lo - pad, hi + pad)

    x0, x1 = ax.get_xlim()
    for i, k in enumerate(order):
        ax.text(x1 + 0.012 * (x1 - x0), i, f"n={len(groups[k]):,}",
                fontsize=7, va="center", family="monospace")
    ax.scatter([], [], s=18, color="black", label="mean")
    ax.plot([], [], color="crimson", lw=1.6, label="median")
    ax.legend(frameon=False, fontsize=7, loc="upper left",
              bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    return fig, ax


def plot_weakness_map(
    per_label,
    *,
    x: str = "preservation_mean",
    y: str = "mapping_accuracy",
    colour: str = "n_usable_markers",
    annotate: str | None = "diagnosis",
    figsize: tuple[float, float] = (7.5, 6),
):
    """Cell types positioned by two metrics, coloured by a third.

    Only types whose ``diagnosis`` is not "ok" are labelled, so the healthy
    mass stays readable.  Pass ``annotate=None`` to label nothing, or a column
    name to label every type where that column is truthy.

    Returns
    -------
    (fig, ax)
    """
    d = per_label.dropna(subset=[c for c in (x, y) if c in per_label.columns])
    if x not in d or y not in d:
        raise KeyError(f"per_label needs both {x!r} and {y!r}")

    sizes = 20 + 320 * (d["n_cells"] / d["n_cells"].max()) if "n_cells" in d else 60
    c = d[colour] if colour in d else None

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(d[x], d[y], s=sizes, c=c, cmap="viridis",
                    edgecolor="0.3", linewidth=0.4, alpha=0.9)
    if c is not None:
        fig.colorbar(sc, ax=ax, shrink=0.75, label=colour.replace("_", " "))

    if annotate and annotate in d.columns:
        flag = d[annotate].astype(str).ne("ok") if annotate == "diagnosis" \
            else d[annotate].astype(bool)
        sub = d[flag]
        _annotate(ax, sub[x], sub[y], sub.index)

    for val, axis in ((per_label.attrs.get("pres_floor"), "v"),
                      (per_label.attrs.get("map_floor"), "h")):
        if val is None:
            continue
        (ax.axvline if axis == "v" else ax.axhline)(
            val, color="0.6", lw=0.8, ls="--", zorder=0)

    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, ax


def plot_agreement_crosstab(
    crosstab,
    agreement: dict | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "magma_r",
):
    """Reference labels against panel-only clusters, row-normalised.

    Pass the output of ``agreement_crosstab`` — already ordered so matched
    pairs sit on the diagonal — and optionally ``cluster_agreement``'s dict,
    whose ARI and AMI go in the title.

    Returns
    -------
    (fig, ax)
    """
    ct = crosstab
    nr, nc = ct.shape
    fig, ax = plt.subplots(figsize=figsize or (max(5, 0.28 * nc + 3),
                                               max(4, 0.26 * nr + 2)))
    im = ax.imshow(ct.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(nc))
    ax.set_xticklabels(ct.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(nr))
    ax.set_yticklabels(ct.index, fontsize=6)
    ax.set_xlabel("panel-only cluster")
    ax.set_ylabel("reference label")
    if agreement:
        ax.set_title(f"ARI {agreement['ARI']:.3f}   AMI {agreement['AMI']:.3f}"
                     f"   ({agreement['n_true_levels']} labels, "
                     f"{agreement['n_pred_levels']} clusters)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7, label="fraction of row")
    fig.tight_layout()
    return fig, ax


def plot_classifier_diagonal(
    per_label,
    *,
    reference: str = "full",
    panel: str = "panel",
    figsize: tuple[float, float] = (6, 6),
    label_delta: float = 0.05,
):
    """Per-cell-type F1 on the panel against the full feature set.

    Points above the diagonal are types the panel classifies better.  Only
    types differing by more than ``label_delta`` are named.

    Returns
    -------
    (fig, ax)
    """
    wide = per_label.pivot_table(index="celltype", columns="feature_set",
                                 values="F1").dropna()
    if panel not in wide or reference not in wide:
        raise KeyError(f"need both {panel!r} and {reference!r} in feature_set")
    n = per_label.groupby("celltype")["n_test"].max().reindex(wide.index)

    fig, ax = plt.subplots(figsize=figsize)
    lo = float(min(wide[reference].min(), wide[panel].min())) - 0.03
    ax.plot([lo, 1.01], [lo, 1.01], ls="--", color="0.6", lw=0.9, zorder=0)
    ax.scatter(wide[reference], wide[panel],
               s=20 + 180 * (n / n.max()) if n.notna().any() else 50,
               color="#2b6cb0", edgecolor="0.3", linewidth=0.4, alpha=0.85)

    gap = (wide[panel] - wide[reference]).abs()
    sub = wide[gap > label_delta]
    if len(sub):
        _annotate(ax, sub[reference], sub[panel], sub.index)

    ax.set_xlabel(f"F1, {reference}")
    ax.set_ylabel(f"F1, {panel}")
    ax.set_xlim(lo, 1.01)
    ax.set_ylim(lo, 1.01)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, ax


def plot_classifier_confusion(
    confusion,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "magma_r",
):
    """Row-normalised confusion matrix from the panel classifier fit.

    Returns
    -------
    (fig, ax)
    """
    n = len(confusion)
    fig, ax = plt.subplots(figsize=figsize or (max(5, 0.30 * n + 3),
                                               max(4, 0.28 * n + 2)))
    im = ax.imshow(confusion.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(confusion.shape[1]))
    ax.set_xticklabels(confusion.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(confusion.index, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true label")
    fig.colorbar(im, ax=ax, shrink=0.7, label="fraction of row")
    fig.tight_layout()
    return fig, ax


def plot_marker_count(
    coverage,
    *,
    marker_floor: int | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "tab20",
):
    """Usable markers per cell type, optionally split by gene source.

    Source columns (``n_from_*``, written by ``coverage_by_label`` when it is
    given a ``sources`` map) are drawn as a stacked bar.  A gene credited to
    several sources appears under its combined category, so the stack still
    sums to the marker count.

    Returns
    -------
    (fig, ax)
    """
    d = coverage.sort_values("n_usable_markers")
    src_cols = [c for c in d.columns if c.startswith("n_from_")]
    n = len(d)
    fig, ax = plt.subplots(figsize=figsize or (7, max(2.5, 0.30 * n + 1.2)))

    if src_cols:
        colours = plt.get_cmap(cmap)(np.linspace(0, 1, len(src_cols)))
        left = np.zeros(n)
        for col, colour in zip(src_cols, colours):
            v = d[col].to_numpy()
            ax.barh(range(n), v, left=left, color=colour,
                    label=col.replace("n_from_", ""))
            left += v
        ax.legend(frameon=False, fontsize=7, title="source", title_fontsize=7)
    else:
        ax.barh(range(n), d["n_usable_markers"], color="#4292c6")

    if marker_floor is None:
        marker_floor = coverage.attrs.get("marker_floor")
    if marker_floor:
        ax.axvline(marker_floor, color="crimson", lw=1.0, ls="--")

    ax.set_yticks(range(n))
    ax.set_yticklabels(d.index, fontsize=8)
    md = coverage.attrs.get("min_detect")
    mf = coverage.attrs.get("min_fold")
    ax.set_xlabel("usable markers"
                  + (f"  (detect >= {md}, fold >= {mf})" if md and mf else ""))
    fig.tight_layout()
    return fig, ax
