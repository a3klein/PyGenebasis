"""
The per-label summary: one row per cell type, joining every metric.

Each metric answers a different question and they disagree in informative ways
-- a type can map badly while carrying strong markers, which means the kNN vote
is failing on a small population rather than the panel lacking genes.  The
``diagnosis`` column makes that call explicitly, so a ranking is not mistaken
for a to-do list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRES_FLOOR = 0.90
MAP_FLOOR = 0.90
MARKER_FLOOR = 20
RARE_CELLS = 200


def _diagnose(row, *, pres_floor, map_floor, marker_floor, rare_cells) -> str:
    lean = row.get("n_usable_markers", np.inf) < marker_floor
    badmap = row.get("mapping_accuracy", 1.0) < map_floor
    badpres = row.get("preservation_mean", 1.0) < pres_floor
    if lean and (badmap or badpres):
        return "marker shortfall — add markers"
    if lean:
        return "few markers, metrics fine"
    if badmap and badpres:
        return "weak structure, markers present"
    if badmap:
        return ("rare-cell mapping noise" if row.get("n_cells", 0) < rare_cells
                else "confused with a neighbour")
    if badpres:
        return "diffuse neighbourhood, markers present"
    return "ok"


def per_label_summary(
    labels,
    *,
    preservation=None,
    mapping=None,
    mapping_stat=None,
    coverage=None,
    classifier=None,
    hierarchy=None,
    pres_floor: float = PRES_FLOOR,
    map_floor: float = MAP_FLOOR,
    marker_floor: int = MARKER_FLOOR,
    rare_cells: int = RARE_CELLS,
) -> pd.DataFrame:
    """Join the per-cell-type views of every metric into one table.

    Every input except ``labels`` is optional; whatever is supplied is used and
    the rest is skipped, so a partial report still works.

    Parameters
    ----------
    labels : array-like
        Per-cell cell type labels, in ``adata.obs_names`` order.
    preservation : array-like, optional
        Per-cell preservation score, same order as ``labels``.
    mapping : pd.DataFrame, optional
        ``get_celltype_mapping(...)["mapping"]`` — used for what each type is
        most often mistaken for.
    mapping_stat : pd.DataFrame, optional
        ``get_celltype_mapping(...)["stat"]``.
    coverage : pd.DataFrame, optional
        Output of :func:`coverage_by_label`.
    classifier : dict, optional
        Output of :func:`classifier_gap`.
    hierarchy : pd.DataFrame, optional
        Output of :func:`get_panel_celltype_accuracy`; the finest level is used.
    pres_floor, map_floor, marker_floor, rare_cells
        Thresholds behind ``diagnosis``.  They are choices, not facts, and are
        recorded in the returned frame's ``.attrs``.

    Returns
    -------
    pd.DataFrame
        Indexed by cell type, with a ``diagnosis`` column.
    """
    lab = pd.Series(np.asarray(labels).astype(str), name="label")
    out = lab.value_counts().rename("n_cells").to_frame()
    out.index.name = "label"

    if preservation is not None:
        p = pd.Series(np.asarray(preservation, dtype=float), index=lab.values)
        g = p.groupby(level=0)
        out["preservation_mean"] = g.mean()
        out["preservation_median"] = g.median()

    if mapping_stat is not None:
        s = mapping_stat.set_index("celltype")
        col = next(c for c in s.columns if "frac" in c or "accuracy" in c)
        out["mapping_accuracy"] = s[col]

    if mapping is not None:
        m = mapping[mapping["celltype"].astype(str)
                    != mapping["mapped_celltype"].astype(str)]
        if len(m):
            worst = (m.groupby(mapping["celltype"].astype(str))["mapped_celltype"]
                     .agg(lambda v: v.astype(str).value_counts().idxmax()))
            frac = (m.groupby(mapping["celltype"].astype(str))["mapped_celltype"]
                    .agg(lambda v: v.astype(str).value_counts().iloc[0]))
            out["most_confused_with"] = worst
            out["confusion_fraction"] = (frac / out["n_cells"]).round(4)

    if coverage is not None:
        for c in ("n_usable_markers", "best_fold", "best_marker"):
            if c in coverage.columns:
                out[c] = coverage[c]

    if classifier is not None:
        per = classifier["per_label"] if isinstance(classifier, dict) else classifier
        wide = per.pivot_table(index="celltype", columns="feature_set", values="F1")
        for fs in wide.columns:
            out[f"clf_F1_{fs}"] = wide[fs]

    if hierarchy is not None and len(hierarchy):
        finest = hierarchy["level"].iloc[-1] if "level" in hierarchy else None
        h = hierarchy[hierarchy["level"] == finest] if finest else hierarchy
        h = h.set_index("celltype")
        for c in ("constrained_accuracy", "compounded_accuracy"):
            if c in h.columns:
                out[c] = h[c]

    out["diagnosis"] = out.apply(
        _diagnose, axis=1, pres_floor=pres_floor, map_floor=map_floor,
        marker_floor=marker_floor, rare_cells=rare_cells)
    out.attrs.update(pres_floor=pres_floor, map_floor=map_floor,
                     marker_floor=marker_floor, rare_cells=rare_cells)
    sort_on = "mapping_accuracy" if "mapping_accuracy" in out else "n_cells"
    return out.sort_values(sort_on)


#: Metric names accepted by ``panel_report(skip=...)``.
REPORT_STEPS = ("preservation", "gene_scores", "mapping", "coverage",
                "clustering", "classifier")


def panel_report(
    adata,
    genes: list[str],
    label_key: str,
    *,
    genes_all: list[str] | None = None,
    batch_key: str | None = None,
    cluster_key: str | None = None,
    sources: dict[str, str] | None = None,
    skip: tuple[str, ...] = (),
    n_neighbors: int = 5,
    n_pcs_all: int = 50,
    min_detect: float = 0.10,
    min_fold: float = 2.0,
    resolution: float = 1.0,
    clf_n_cells: int = 25_000,
    layer: str | None = None,
    random_state: int = 32,
    verbose: bool = True,
) -> dict:
    """Run the evaluation suite over one panel and return the report tables.

    Convenience over calling each metric yourself; every step is skippable,
    since a full run on a large reference is not cheap.  ``clustering`` is the
    step most worth supplying yourself — pass ``cluster_key`` naming a column
    of ``adata.obs`` and it is used instead of clustering here.

    Parameters
    ----------
    adata : AnnData
    genes : list[str]
        The panel.
    label_key : str
        ``adata.obs`` column holding the reference cell type labels.
    genes_all : list[str], optional
        Reference feature set for preservation, gene scores and the classifier
        comparison.  Defaults to every gene in ``adata``.
    cluster_key : str, optional
        Precomputed panel-only clustering.  Strongly preferred over the
        built-in default — see ``cluster_on_panel``.
    sources : dict, optional
        ``gene -> source`` for the marker-count breakdown.
    skip : tuple of str
        Any of :data:`REPORT_STEPS`.
    verbose : bool
        Log each step as it starts.

    Returns
    -------
    dict
        The five report tables — ``summary``, ``per_label``, ``per_cell``,
        ``per_gene``, ``per_gene_label`` — plus ``crosstab``, ``agreement``,
        ``classifier_confusion`` and ``gene_importance`` for the figures.
        Skipped steps leave their entries as None.
    """
    import logging

    from ._classifier import classifier_gap
    from ._clustering import agreement_crosstab, cluster_agreement, cluster_on_panel
    from ._coverage import coverage_by_label, marker_detection
    from ._mapping import get_celltype_mapping
    from ._neighborhood import (get_gene_prediction_scores,
                                get_neighborhood_preservation_scores)

    log = logging.getLogger(__name__)
    bad = set(skip) - set(REPORT_STEPS)
    if bad:
        raise ValueError(f"unknown step(s) {sorted(bad)}; valid: {REPORT_STEPS}")

    def note(msg):
        if verbose:
            log.info(msg)
            print(f"[panel_report] {msg}", flush=True)

    panel = [g for g in genes if g in adata.var_names]
    if not panel:
        raise ValueError("none of the panel genes are in adata.var_names")
    if label_key not in adata.obs.columns:
        raise KeyError(f"{label_key!r} not in adata.obs")
    reference = list(adata.var_names) if genes_all is None else genes_all
    labels = adata.obs[label_key].astype(str)

    out = dict.fromkeys(
        ["summary", "per_label", "per_cell", "per_gene", "per_gene_label",
         "crosstab", "agreement", "classifier_confusion", "gene_importance"])
    head = {"n_panel_genes": len(panel), "n_cells": int(adata.n_obs),
            "n_labels": int(labels.nunique())}
    per_cell = pd.DataFrame({"label": labels.to_numpy()}, index=adata.obs_names)
    per_gene = pd.DataFrame(index=pd.Index(reference, name="gene"))
    per_gene["on_panel"] = per_gene.index.isin(panel)
    if sources:
        per_gene["source"] = [sources.get(g) for g in per_gene.index]

    cs = mapping = coverage = clf = None

    if "preservation" not in skip:
        note("preservation")
        cs = get_neighborhood_preservation_scores(
            adata, panel, genes_all=reference, batch_key=batch_key,
            n_neighbors=n_neighbors, n_pcs_all=n_pcs_all, n_pcs_selection=None,
            layer=layer, random_state=random_state)
        per_cell["preservation"] = cs["cell_score"].to_numpy()
        head["preservation_median"] = float(cs["cell_score"].median())

    if "gene_scores" not in skip:
        note("gene prediction scores")
        gs = get_gene_prediction_scores(
            adata, panel, genes_all=reference, batch_key=batch_key,
            n_neighbors=n_neighbors, n_pcs_all=n_pcs_all, n_pcs_selection=None,
            layer=layer, random_state=random_state).set_index("gene")
        per_gene = per_gene.join(gs[["corr", "corr_all", "gene_score"]])
        off = per_gene.loc[~per_gene["on_panel"], "gene_score"]
        head["gene_score_median_offpanel"] = float(off.median())

    if "mapping" not in skip:
        note("cell type mapping")
        ctm = get_celltype_mapping(
            adata, panel, celltype_key=label_key, batch_key=batch_key,
            n_neighbors=n_neighbors, n_pcs_selection=None, return_stat=True,
            layer=layer, random_state=random_state)
        mapping = ctm
        head["mapping_accuracy_overall"] = float(
            (ctm["mapping"]["mapped_celltype"] == ctm["mapping"]["celltype"]).mean())

    if "coverage" not in skip:
        note("marker coverage")
        detail = marker_detection(adata, panel, label_key, min_detect=min_detect,
                                  min_fold=min_fold, layer=layer)
        coverage = coverage_by_label(detail, sources=sources)
        out["per_gene_label"] = detail
        best = detail[detail["usable"]].groupby("gene")["fold"].agg(["size", "max"])
        per_gene["n_labels_marked"] = best["size"].reindex(per_gene.index).fillna(0).astype(int)
        per_gene["best_fold"] = best["max"].reindex(per_gene.index)

    if "clustering" not in skip:
        if cluster_key is not None:
            clusters = adata.obs[cluster_key].astype(str)
            note(f"clustering agreement (using obs[{cluster_key!r}])")
        else:
            note("panel-only clustering")
            clusters = cluster_on_panel(
                adata, panel, batch_key=batch_key, resolution=resolution,
                layer=layer, random_state=random_state)
        per_cell["panel_cluster"] = np.asarray(clusters).astype(str)
        out["agreement"] = cluster_agreement(labels, clusters)
        out["crosstab"] = agreement_crosstab(labels, clusters)
        head.update({k: out["agreement"][k] for k in ("ARI", "AMI")})
        head["n_panel_clusters"] = out["agreement"]["n_pred_levels"]

    if "classifier" not in skip:
        note("classifier, panel vs reference feature set")
        clf = classifier_gap(
            adata, panel, label_key, genes_all=reference, n_cells=clf_n_cells,
            layer=layer, random_state=random_state)
        s = clf["summary"].set_index("feature_set")["macro_F1"]
        head["clf_macroF1_panel"] = float(s["panel"])
        head["clf_macroF1_full"] = float(s["full"])
        head["clf_gap"] = float(s["panel"] - s["full"])
        out["classifier_confusion"] = clf["confusion"]
        out["gene_importance"] = clf["gene_importance"]

    note("per-label summary")
    out["per_label"] = per_label_summary(
        labels,
        preservation=per_cell["preservation"] if "preservation" in per_cell else None,
        mapping=mapping["mapping"] if mapping else None,
        mapping_stat=mapping["stat"] if mapping else None,
        coverage=coverage,
        classifier=clf,
    )
    head["n_labels_below_marker_floor"] = int(
        (out["per_label"].get("n_usable_markers",
                              pd.Series(dtype=float)) < MARKER_FLOOR).sum())

    out["summary"] = pd.DataFrame([head])
    out["per_cell"] = per_cell
    out["per_gene"] = per_gene.reset_index()
    return out
