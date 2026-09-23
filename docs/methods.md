# How it works

What the algorithm does and what the numbers it returns mean. For the API see the
[README](../README.md); for the commands see [cli.md](cli.md).

## The idea

A good panel is one where cells that were neighbours in the full transcriptome are still
neighbours when you only measure the panel. pyGeneBasis makes that concrete by comparing two
k-nearest-neighbour graphs:

- the **true graph**, built from all (informative) genes
- the **selection graph**, built from the panel only

Selection is greedy. Starting from nothing — or from `genes_base`, if you have genes you must
include — it adds one gene at a time, each time picking the candidate that most improves
agreement between the two graphs. Genes are returned in the order they were chosen, so the
`rank` column is meaningful: the first 100 rows of a 300-gene panel are a reasonable
100-gene panel.

## Choosing the next gene

At each step every remaining candidate is scored by how badly the current panel predicts it.
For a candidate gene `g`, using the current selection graph:

```
neighbor_avg[i] = mean over i's k neighbours of expression[neighbour, g]
score(g)        = ( sum_i |expression[i, g] - neighbor_avg[i]|^p )^(1/p)
```

with `p = 3` by default. A gene whose expression is already well predicted by averaging over
a cell's neighbours scores near zero — the panel already captures whatever structure that
gene carries, so adding it buys little. A gene with a high score is poorly reconstructed, so
it carries information the panel is missing. That gene is added, the selection graph is
rebuilt, and the process repeats.

This is why the greedy path matters and why the result is not simply "the top N most variable
genes": each choice is conditional on what has already been picked.

## Methylation panels

For mCH/mCG data the input is a **rate** matrix, not counts, and dispersion depends on
coverage as well as mean — a thinly covered gene looks variable for reasons that are not
biological. `retain_informative_genes(flavor="methylation")` therefore z-scores dispersion
within **mean x coverage** bins rather than mean alone, sparse bins being merged into their
nearest neighbour first.

It needs per-gene mean coverage in `adata.var` (`coverage_key`, default `cov_mean`). Nothing
computes this for you: in ALLCools it is the `{var_dim}_cov_mean` coordinate from
`MCDS.add_feature_cov_mean`, and it has to be carried across when the AnnData is built. The
function raises if the column is absent.

```python
adata = pgb.retain_informative_genes(
    adata, flavor="methylation", coverage_key="cov_mean", n=3000
)
```

`highly_variable_methylation_features` returns the full per-gene table — mean, dispersion,
coverage, bin assignments, `dispersion_norm`, `feature_select` — if you want to inspect the
fit. Passing `n` takes the top features by normalised dispersion; omitting it uses the
ALLCools cutoffs (`min_disp=0.5`, `min_mean=0`, `max_mean=5`).

This is a port of ALLCools `highly_variable_methylation_feature`, verified column-for-column
against it. Two upstream bugs are fixed: it crashes under pandas >= 2, and its
`n_top_feature` is ignored (hardcoded at 5000).

## Evaluation metrics

### Cell score — `get_neighborhood_preservation_scores`

Per cell, how well its neighbourhood survived the reduction to the panel. For cell `i`:

```
dist_true[i] = median distance, in the selection graph, to i's TRUE-graph neighbours
dist_sel[i]  = median distance, in the selection graph, to i's SELECTION-graph neighbours
cell_score[i] = (mean_dist_all[i] - dist_sel[i]) / (mean_dist_all[i] - dist_true[i])
```

`mean_dist_all[i]` is cell `i`'s mean distance to all other cells, which normalises for how
spread out that part of the embedding is. A score of 1 means the panel recovers exactly the
neighbours the full transcriptome would have given. Lower means the panel's neighbours are
further away than the true ones.

Note the denominator: it is the difference between two distances that are often close
together. This quantity is badly conditioned, which is why cell scores are sensitive to small
numerical differences (see the divergence note in the README).

### Gene score — `get_gene_prediction_scores`

The same question asked per gene rather than per cell: how well can a gene's expression be
predicted from neighbourhoods in the selection graph, relative to how well the true graph
predicts it. Low-scoring genes are ones the panel fails to represent.

### Cell type mapping — `get_celltype_mapping`

A direct, interpretable check: label each cell by majority vote among its k nearest
neighbours in the selection graph, and compare to its real label. Returns per-cell
predictions and a per-cell-type `frac_correctly_mapped`. This is usually the number to quote,
because it answers "would this panel let me identify my cell types" without needing any
explanation of the embedding.

Ties are broken by smallest distance rank, matching geneBasisR.

### Redundancy — `get_redundancy_stat`

For each gene, drop it, redo the cell type mapping, and see what happens:

```
frac_correctly_mapped      accuracy without the gene
frac_correctly_mapped_all  accuracy with the whole panel
ratio                      the first divided by the second
```

A gene whose removal costs nothing is redundant with the rest of the panel. Prefer the
absolute `frac_correctly_mapped` over the ratio when comparing across runs — the ratio
divides two nearly equal numbers and is noisy for the same reason cell scores are.

### Hierarchical accuracy — `get_panel_celltype_accuracy`

For taxonomies with nested levels (Class → Subclass → Group). Reports
`constrained_accuracy`, the accuracy within the correct parent class, and
`compounded_accuracy`, the product down the hierarchy, which is the probability of getting
the whole path right. The coarsest level has no parent, so its constrained accuracy is just
its accuracy.

## Trimming

`trim_panel` is the greedy procedure run backwards: repeatedly remove the gene whose removal
costs the least, by leave-one-out cell score. Use it to cut an over-sized panel to a target
size, with `genes_protect` for genes that must survive regardless — controls, housekeeping
genes, anything promised to a collaborator.

For speed it fits the PCA once per removal step and projects each leave-one-out candidate
into that fixed rotation instead of refitting. The approximation error is on the order of
`1/panel_size` per gene removed, which is negligible for panels of any realistic size.

## Batch handling

Batch structure is handled in one of two ways, and which is right depends on the question.

`per_batch` builds a separate PCA and kNN graph inside each batch and never links across
them. Neighbours are always same-batch. This is geneBasisR's behaviour and the default for
selection and evaluation: for judging a panel, you want to know it works within a sample, not
that batch correction can paper over it.

`harmony` and `mnn` correct a shared embedding so that neighbours can cross batches. This is
the default for the mapping functions, where the point is to classify a cell against the
whole reference. Prefer `harmony`; see the README on why `mnn` is kept but not recommended.

## Reliability

A separate question from panel design: given a panel you have already measured, which of its
genes are behaving badly?

`compute_gene_reliability` compares gene–gene co-expression in the measured data against the
same genes in an scRNA-seq reference. Genes whose co-expression neighbourhood does not match
the reference are flagged, and sorted into failure modes — `probe_failure`,
`composition_mismatch`, `idiosyncratic_noise`, or `reliable`.

`run_perturbation_analysis` then asks whether those failures matter. It simulates each
failure mode in the reference, re-runs cell type mapping, and reports the drop in accuracy
(`delta_ct`). It also reports `removal_benefit`: whether you would be better off dropping a
broken gene than keeping it. A gene can be unreliable and still harmless if nothing depended
on it.
