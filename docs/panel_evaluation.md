# Evaluating a panel

How good is a panel, before you ever run it? This page covers the metrics, what each
one is actually measuring, and how to read the figures and tables.

This is evaluation against a reference taxonomy — a designed panel versus the scRNA-seq
it was built from. It is a different job from diagnosing a *measured* MERFISH dataset for
failing probes, which is [the reliability module](methods.md#reliability).

## Three machines, not one

The metrics look similar and are not. Four use no model at all; the rest use three
different ones, and they fail for different reasons. Reading them as one number is the
main way to misinterpret a panel.

| Metric | Machine | Uses labels? | What a failure means |
|---|---|---|---|
| preservation | none — distance geometry | no | panel space distorts neighbourhoods |
| gene score | none — correlation | no | panel space can't predict that gene |
| mapping, redundancy | **kNN majority vote**, k=5 | at vote time | same-type cells aren't each other's neighbours |
| ARI / AMI | **Leiden on the panel embedding** | **no** | panel space has the wrong density structure |
| classifier gap | **logistic regression**, 70/30 split | trains on them | the information isn't linearly decodable |

The practical consequence: **kNN is local and non-parametric, so it fails on rare types
that lack same-type neighbours even when the type is perfectly separable.** A cell type
with 58 cells can map at 0.69 while carrying a marker enriched 300-fold. A logistic
regression would likely classify it correctly. Neither number is wrong; they answer
different questions.

Note also that two of the figures are confusion matrices from **different machines** —
the mapping heatmap is a kNN vote, the classifier confusion is logistic regression.

---

## Neighbourhood preservation

Per cell, how well the panel keeps it among the right neighbours.

```
dist_true = median distance to the cell's REFERENCE-graph neighbours
dist_sel  = median distance to the cell's PANEL-graph neighbours
score     = (mean_dist_all - dist_sel) / (mean_dist_all - dist_true)
```

All three distances are measured **in the reference embedding**. The panel graph only
supplies *which* cells it calls neighbours; its own geometry never enters. So the
question is "the panel picked these cells — how far away are they really, compared to
the true neighbours?" A score of 1 means the panel recovers the reference neighbourhood
exactly.

**Caveat.** The denominator subtracts two nearly equal distances, so the score is badly
conditioned and unbounded in both directions. Compare medians, not individual cells, and
do not compare scores across different reference feature sets — changing the reference
changes the quantity.

```python
cs = pgb.get_neighborhood_preservation_scores(adata, panel, genes_all=hvg,
                                              batch_key="donor_id")
fig, ax = pgb.plot_preservation_violin(cs["cell_score"], adata.obs["celltype"])
```

*The violin*: one row per cell type, sorted by median, with mean (dot) and median (red
bar) marked and `n` on the right. The x-range is clipped to pooled quantiles because a
handful of extreme cells otherwise squeeze every violin flat — no cells are dropped, and
a group's own median is never clipped out of view.

### kNN overlap — the bounded alternative

The returned frame also carries `knn_overlap`: the fraction of its reference neighbours
the panel actually recovered, per cell. It is bounded [0, 1], means something plain, and
avoids the ill-conditioning above, so it is the better number to quote across runs. The
ratio score is kept because it is what geneBasisR computes.

### Scoring your own embeddings

Both metrics have array-level entry points that skip the AnnData entirely:

```python
scores  = pgb.preservation_from_embedding(ref_embedding, ref_indices, panel_indices)
overlap = pgb.knn_overlap(ref_indices, panel_indices)
```

This matters for methylation. An mC embedding is not a PCA over a gene list — it is
`balanced_pca` → `significant_pc_test` → scaled CH and CG components concatenated →
harmonypy — so `get_neighborhood_preservation_scores`, which builds its own embedding
from `(adata, genes)`, cannot produce it. Passing the finished arrays instead means the
methylation and RNA sides compute the same quantity on the same scale, rather than mC
falling back to raw overlap because the normalised score was unreachable.

Note the asymmetry in what each side needs. The reference side needs **coordinates**,
because the score measures distances to the panel's chosen neighbours and those pairs are
generally not edges of the reference graph. The selection side needs only **indices** —
the panel embedding's own geometry never enters. `knn_overlap` needs no coordinates at
all.

## Gene prediction score

Per gene, whether the panel's neighbourhoods can predict its expression:
`corr_sel / corr_all`, each a Spearman correlation between the gene and its
neighbour-average.

**Read the median over off-panel genes.** A gene on the panel helps define the graph it
is predicted from, so panel genes score high structurally. Scores above 1 are common and
mean two different things: a panel gene predicting itself, or a gene the reference graph
barely predicts at all, giving a near-zero denominator and a meaningless ratio.

## Cell-type mapping

Label each cell by majority vote among its k nearest neighbours in the panel graph and
compare to the truth. Returns per-cell predictions and per-type `frac_correctly_mapped`.

Usually the number to quote, because it needs no explanation of the embedding. Ties are
broken by smallest distance rank, matching geneBasisR.

`get_panel_celltype_accuracy` extends this to nested taxonomies:
`constrained_accuracy` is the accuracy *within the correct parent class*, and
`compounded_accuracy` is the product down the hierarchy — the probability of getting the
whole path right.

## Panel-only clustering — ARI and AMI

Every other metric compares the panel against labels it is handed. This one asks the
question you actually face in tissue, where no labels exist: **cluster on the panel genes
alone and see whether the populations come back**. A panel can map at 0.98 and still
cluster into something unrecognisable.

**Re-embed and re-cluster yourself, and look at the result before scoring it.** How to
embed, correct and cluster is a judgement call. `cluster_on_panel` is a reasonable
starting point, not a recommendation; the metrics take label vectors, so any clustering
works.

```python
clusters = pgb.cluster_on_panel(adata, panel, batch_key="donor_id", resolution=1.0)
agree = pgb.cluster_agreement(adata.obs["celltype"], clusters)
ct = pgb.agreement_crosstab(adata.obs["celltype"], clusters)
fig, ax = pgb.plot_agreement_crosstab(ct, agree)
```

Both scores work on **pairs of cells**, so neither needs the two partitions to share
names or to have the same number of groups. ARI counts pairs that are together-in-both
or apart-in-both, corrected for chance. AMI is the information-theoretic analogue.

**They differ on splitting, and that difference is the useful signal.** Breaking one true
population into three clusters destroys many within-group pairs, so ARI falls hard — but
each fragment still identifies its label, so AMI barely moves. Merging two populations
hurts both.

**ARI partly measures your resolution choice.** On one real panel, changing only the
reference partition's granularity moved ARI from 0.36 to 0.65 with the clustering
unchanged; on synthetic data with four perfectly recovered types, raising the resolution
until it over-split three-fold took ARI from 1.00 to 0.44 while AMI held at 0.71. Report
the resolution, treat AMI as the more robust headline, and read a large ARI–AMI gap as
over-splitting.

*The crosstab*: rows are reference labels, columns panel clusters, rows summing to 1. An
optimal one-to-one assignment orders the rows, then each remaining cluster is placed
beside the row it most belongs to, so a split population's fragments sit adjacent rather
than scattered. ARI and AMI are in the title.

## Classifier gap

Fit a multinomial logistic regression on the panel features and on the full feature set,
same cells and same split, and compare. This bounds what restricting measurement to the
panel actually costs.

```python
clf = pgb.classifier_gap(adata, panel, "celltype")
fig, ax = pgb.plot_classifier_diagonal(clf["per_label"])
fig, ax = pgb.plot_classifier_confusion(clf["confusion"])
```

The gap is routinely near zero or **negative** — a curated panel can beat an unfiltered
transcriptome, because most genes are noise to a linear model. On one real panel the
960-gene set scored macro-F1 0.944 against 0.918 for all 47,362 genes.

Labels with fewer than `min_cells_per_label` cells are dropped, since a stratified split
needs each label on both sides; they are returned in `dropped_labels`. Cells are
subsampled to `n_cells` and only the subsample is densified.

*The diagonal*: per-type F1, panel against full, point size = held-out cells. Above the
line means the panel does better. Only types differing by more than 0.05 are named.

## Marker coverage

Every structural metric can look healthy while a particular cell type has **no gene that
marks it**. This checks that directly, measured from the data rather than read off
provenance.

```
detect_in  = fraction of the label's cells with the gene detected
detect_out = the same outside the label
fold       = (detect_in + eps) / (detect_out + eps)
usable     = detect_in >= min_detect AND fold >= min_fold
```

Detection is binarised at `> 0`, which is invariant to normalisation, so counts and
log-normalised values give the same answer.

**Both thresholds are choices, not facts** — defaults are 0.10 and 2.0, and moving them
moves the usable-marker count. They travel in the returned frame's `.attrs` and are
printed on the figure.

```python
det = pgb.marker_detection(adata, panel, "celltype")
cov = pgb.coverage_by_label(det, sources=gene_to_source)   # sources optional
fig, ax = pgb.plot_marker_count(cov)
```

`sources` is an optional `gene -> source` map. A gene credited to several sources is
counted under its **combined** category — `"literature;geneBasis"` rather than assigned
to one — so the stacked bar still sums to the marker count.

## The per-label summary

One row per cell type, joining everything, plus a `diagnosis` naming which constraint
actually binds. This is the table to put in a supplement.

```python
tab = pgb.per_label_summary(adata.obs["celltype"], preservation=cs["cell_score"],
                            mapping=ctm["mapping"], mapping_stat=ctm["stat"],
                            coverage=cov, classifier=clf)
fig, ax = pgb.plot_weakness_map(tab)
```

Every input except the labels is optional; a partial report still works.

| diagnosis | meaning |
|---|---|
| `marker shortfall — add markers` | few markers **and** a failing metric — the real hole |
| `few markers, metrics fine` | thin but performing |
| `weak structure, markers present` | mapping and preservation both poor despite markers |
| `rare-cell mapping noise` | mapping poor, under `rare_cells` cells — adding genes won't help |
| `confused with a neighbour` | mapping poor on a large population |
| `diffuse neighbourhood, markers present` | preservation poor only |
| `ok` | |

**`marker_floor` scales with the panel.** The default of 20 suits a ~960-gene panel. An
80-gene panel over 11 cell types has roughly seven genes per type to give, so leaving the
default in place labels almost everything a shortfall. Set it to something the panel could
plausibly reach.

The distinction the diagnosis exists to make: **a type low on mapping but rich in markers
is usually kNN failing on a small population, not a marker gap.** Ranking by accuracy
alone reads as a to-do list and sends you adding genes that change nothing.

*The weakness map*: cell types positioned by preservation and mapping accuracy, sized by
cell count, coloured by usable markers, with the diagnosis thresholds drawn as dashed
lines. Only types not diagnosed `ok` are labelled. The colour carries the real
information — two types can both sit in the bad corner while one is dark (no markers, a
genuine coverage hole) and the other bright (plenty of markers, so something else is
wrong).

---

## Outputs

Five tables, one grain each. Every figure is a view of one of them, so you can always get
behind a plot.

| File | Grain | Contents |
|---|---|---|
| `panel_summary` | 1 row | ARI, AMI, classifier macro-F1 both ways, overall mapping, median preservation, median off-panel gene score |
| `per_label` | cell type | the main table — preservation, mapping, markers, classifier F1, diagnosis |
| `per_cell` | cell | preservation, label, panel cluster, embedding coordinates |
| `per_gene` | gene | gene score, `corr`, `corr_all`, `on_panel`, source |
| `per_gene_label` | gene × label | `detect_in`, `detect_out`, `fold`, `usable` |

## What is not here

**Gene redundancy** (`get_redundancy_stat`) is a leave-one-out drop in mapping accuracy
per gene. It is a *trimming* tool rather than a report metric, and it costs one full kNN
mapping per gene — hours on a 960-gene panel. Use `trim_panel`, which does the same thing
greedily and far more cheaply, or run it on a shortlist via `genes_to_assess`.
