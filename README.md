# pyGeneBasis

Gene panel design from single-cell RNA-seq data — a Python reimplementation of
[geneBasisR](https://github.com/MarioniLab/geneBasisR) built for large references.

Given an annotated scRNA-seq reference, pyGeneBasis greedily selects a panel of genes that
best preserves the structure of the full transcriptome: at each step it adds the gene that
most reduces the discrepancy between a k-nearest-neighbour graph built on the selected panel
and one built on all genes. The result is a ranked gene list suitable for targeted spatial
assays such as MERFISH.

Beyond the R package it adds panel **trimming**, a **reliability** module for comparing a
measured spatial panel against its reference, approximate kNN and parallelised scoring for
references past 100k cells, and a command-line interface.

## Install

```bash
git clone git@github.com:a3klein/PyGenebasis.git
cd PyGenebasis
pip install -e .
```

Optional CLI polish (formatted help, rich logging):

```bash
pip install -e ".[cli]"
```

## Quickstart

Design a panel, evaluate it, and trim it. `adata.X` should hold log-normalised counts.

```python
import pygenebasis as pgb

adata = pgb.read_adata("reference.h5ad")

# 1. Restrict to informative genes before selection
adata = pgb.retain_informative_genes(adata, n=2000)

# 2. Select the panel — returns a DataFrame with columns rank, gene
panel = pgb.gene_search(adata, n_genes=300, batch_key="donor_id")
genes = panel["gene"].tolist()

# 3. How well does the panel preserve neighbourhood structure?
cell_scores = pgb.get_neighborhood_preservation_scores(
    adata, genes, batch_key="donor_id"
)
print(cell_scores["cell_score"].mean())

# 4. Can the panel recover cell type labels?
mapping = pgb.get_celltype_mapping(
    adata, genes, celltype_key="celltype", batch_key="donor_id"
)
print(mapping["stat"]["frac_correctly_mapped"].mean())

# 5. Drop the least informative genes to hit a target size
trimmed = pgb.trim_panel(adata, genes, n_remove=50, batch_key="donor_id")
final = trimmed["reduced_panel"]
```

`gene_search` is the expensive step and scales with reference size, panel size and gene
count. Start small to gauge runtime before committing to a full run.

## What's in the package

| Area | Functions |
|---|---|
| **I/O** | `read_adata`, `read_table`, `write_csv`, `write_tsv`, `write_json`, `prepare_batch_key` |
| **Preprocessing** | `retain_informative_genes`, `highly_variable_methylation_features` |
| **Graph** | `build_knn_graph` |
| **Selection** | `gene_search`, `trim_panel`, `calc_minkowski_distances` |
| **Evaluation** | `preservation_from_embedding`, `knn_overlap`, `get_neighborhood_preservation_scores`, `get_gene_prediction_scores`, `get_neighs_all_stat`, `evaluate_library`, `get_panel_celltype_accuracy` |
| **Mapping** | `get_celltype_mapping`, `get_redundancy_stat` |
| **Panel evaluation** | `cluster_on_panel`, `cluster_agreement`, `agreement_crosstab`, `classifier_gap`, `marker_detection`, `coverage_by_label` |
| **Reliability** | `compute_corr_matrices`, `score_gene_reliability`, `compute_gene_reliability`, `subsample_adata`, `perturb_expression`, `run_perturbation_analysis` |
| **Plotting** | `plot_mapping_heatmap`, `plot_expression_heatmap`, `plot_coexpression`, `plot_redundancy_stat`, `plot_umaps_w_counts` |
| **Evaluation figures** | `plot_preservation_violin`, `plot_weakness_map`, `plot_agreement_crosstab`, `plot_classifier_diagonal`, `plot_classifier_confusion`, `plot_marker_count` |

Every public function is importable from the top level: `from pygenebasis import gene_search`.

## Backend choices

Two flags appear on most functions and control the speed/fidelity trade-off.

`knn_method` selects the neighbour search. `"approx"` (default) uses PyNNDescent and is what
you want for real references; `"exact"` uses sklearn brute force, is far slower, and matches
geneBasisR most closely.

`batch_method` selects how batch structure is handled. `"per_batch"` builds a separate PCA
and kNN graph within each batch and never links across them — this is geneBasisR's behaviour
and the default for selection and evaluation. `"harmony"` and `"mnn"` instead correct a
shared embedding so neighbours can cross batches; these are the options for the mapping
functions, where cross-batch neighbours are the point.

## Differences from geneBasisR

**Cell type mapping defaults to harmony, not MNN.** `get_celltype_mapping` and
`get_redundancy_stat` default to `batch_method="harmony"`. geneBasisR uses `fastMNN`; our MNN
is a from-scratch sklearn implementation that does not scale and has OOM-killed runs at
~40k cells. Pass `batch_method="mnn"` if you need R parity.

**Cell scores do not reproduce R exactly.** The cell score formula divides two nearly equal
distances, and R's `irlba` and Python's sklearn PCA differ by a percent or two. That is
enough to flip the sign of the difference for many cells and invert their relative ranking.
Both implementations are correct; the quantity is just badly conditioned. The underlying
`dist_sel` values agree closely, so compare those rather than the scores if you are checking
against R.

**HVG selection uses scranpy**, a Python port of `scran::modelGeneVar`, rather than calling
scran itself. Overlap with R is high but not exact. `flavor="seurat"` and `"seurat_v3"`
delegate to scanpy instead.

**Methylation is supported**, which geneBasisR does not do: `flavor="methylation"` bins
dispersion by mean and coverage for mCH/mCG rate matrices. See
[docs/methods.md](docs/methods.md#methylation-panels).

Numerical tolerances used when comparing against R are defined in `tests/helpers.py`.

## Command line

The same workflows are available as commands:

```bash
pygenebasis panel search   --adata ref.h5ad --n-genes 300 --output panel.csv
pygenebasis panel trim     --adata ref.h5ad --panel panel.csv --n-remove 50 --output trimmed.csv
pygenebasis panel evaluate --adata ref.h5ad --panel panel.csv --celltype-key celltype --output eval.csv

pygenebasis reliability score-coexp --merfish m.h5ad --ref ref.h5ad --panel panel.csv --output scores.csv
pygenebasis reliability perturb --adata ref.h5ad --results scores.csv \
    --level-keys Class,Subclass,Group --out-dir perturb/
```

`panel evaluate` writes its gene and cell type tables beside `--output` with `_gene` and
`_celltype` suffixes. `reliability perturb` returns several tables, so it takes `--out-dir`
and writes one CSV per metric.

Any `--panel` CSV is read with the first column as the index, so it needs a leading index
column alongside `gene` — a file written with `to_csv(index=False)` will not be read
correctly. `pandas.DataFrame({"gene": genes}).to_csv(path)` produces the right shape.

Run `pygenebasis --help` or see [docs/cli.md](docs/cli.md) for all options.

## Documentation

- [docs/methods.md](docs/methods.md) — what the algorithm does and what each metric means
- [docs/panel_evaluation.md](docs/panel_evaluation.md) — evaluating a finished panel against a reference
- [docs/01_panel_design.ipynb](docs/01_panel_design.ipynb) — panel design, evaluation and trimming, worked through
- [docs/02_reliability.ipynb](docs/02_reliability.ipynb) — scoring a measured panel against its reference
- [docs/cli.md](docs/cli.md) — command line reference
- [docs/scripts/](docs/scripts/) — batch scripts and SLURM wrappers for long runs

## Tests

```bash
pytest tests -m "not mouse and not bg_sn"   # fast, no reference data needed
pytest tests                                 # includes R reference comparisons
```

The reference tests compare against outputs from geneBasisR and need reference data that is
not distributed with the repo; they skip automatically when it is absent.
