# Command line reference

`pygenebasis` wraps the main workflows as commands so they can be run from a batch script
without writing Python. Everything here is also available from the Python API; see the
[README](../README.md) for that.

```
pygenebasis
├── panel search      select a panel
├── panel trim        shrink an existing panel
├── panel evaluate    score a panel
├── reliability score-coexp   compare a measured panel against its reference
└── reliability perturb       measure the cost of unreliable genes
```

Run `pygenebasis <group> <command> --help` for the authoritative option list.

## Input files

`--adata`, `--merfish` and `--ref` take `.h5ad` files with log-normalised counts in `.X`, or
in the layer named by `--layer`.

`--panel` and `--results` take CSVs read with the **first column as the index**. A panel file
therefore needs a leading index column next to `gene`:

```python
pd.DataFrame({"gene": genes}).to_csv("panel.csv")      # correct
pd.DataFrame({"gene": genes}).to_csv("panel.csv", index=False)   # will not read back
```

`--log-dir` is accepted by every command. Logging detects Slurm and writes to a file there
when running batch, or to the terminal when interactive.

---

## panel search

Greedy panel selection. Filters to informative genes, runs `gene_search`, and writes a CSV
with `rank` and `gene` columns in selection order.

```bash
pygenebasis panel search --adata ref.h5ad --n-genes 300 --output panel.csv \
    --batch-key donor_id
```

| Option | Default | |
|---|---|---|
| `--adata` | required | reference `.h5ad` |
| `--n-genes` | required | target panel size |
| `--output` | required | output CSV |
| `--genes-base` | | comma-separated seed genes to start from |
| `--genes-discard` | | comma-separated genes to exclude |
| `--batch-key` | | obs column identifying batches |
| `--knn-method` | `approx` | `approx` or `exact` |
| `--batch-method` | `per_batch` | `per_batch`, `mnn` or `harmony` |
| `--n-neighbors` | `5` | k for the kNN graphs |
| `--n-pcs` | `50` | PCA components |
| `--layer` | | layer holding logcounts |

This is the expensive command; cost grows with reference size, gene count and panel size.

> `panel search` always applies `retain_informative_genes` with default settings before
> selecting, and does not currently expose a way to tune or skip that filter. Use the Python
> API if you need control over HVG selection.

## panel trim

Remove the least informative genes from an existing panel, by leave-one-out scoring.

```bash
pygenebasis panel trim --adata ref.h5ad --panel panel.csv --n-remove 50 \
    --output trimmed.csv --genes-protect GFAP,MBP
```

| Option | Default | |
|---|---|---|
| `--adata` `--panel` `--n-remove` `--output` | required | |
| `--genes-protect` | | comma-separated genes that can never be removed |
| `--batch-key` | | |
| `--knn-method` | `approx` | |
| `--batch-method` | `per_batch` | |
| `--n-neighbors` | `5` | |
| `--n-jobs` | `-1` | workers over leave-one-out candidates |
| `--layer` | | |

## panel evaluate

Score a panel with `evaluate_library`. Writes cell scores to `--output` and the gene and
cell type tables beside it with `_gene` and `_celltype` suffixes, so
`--output eval.csv` also produces `eval_gene.csv` and `eval_celltype.csv`.

```bash
pygenebasis panel evaluate --adata ref.h5ad --panel panel.csv \
    --celltype-key celltype --output eval.csv
```

| Option | Default | |
|---|---|---|
| `--adata` `--panel` `--output` | required | |
| `--celltype-key` | | obs column with labels; enables the mapping accuracy table |
| `--batch-key` | | |
| `--knn-method` | `approx` | |
| `--batch-method` | `per_batch` | |
| `--n-neighbors` | `5` | |
| `--layer` | | |

---

## reliability score-coexp

Compare gene–gene co-expression in a measured spatial dataset against the same genes in an
scRNA-seq reference, and classify each gene's failure mode. Writes one row per panel gene
with its metrics and a `failure_mode` column.

```bash
pygenebasis reliability score-coexp --merfish measured.h5ad --ref ref.h5ad \
    --panel panel.csv --output scores.csv \
    --ref-cell-type-key Subclass --ref-n-cells-per-type 500
```

| Option | Default | |
|---|---|---|
| `--merfish` `--ref` `--panel` `--output` | required | |
| `--merfish-cell-type-key` | | obs column to stratify MERFISH subsampling by |
| `--merfish-n-cells-per-type` | | cells kept per MERFISH type |
| `--ref-cell-type-key` | | obs column to stratify reference subsampling by |
| `--ref-n-cells-per-type` | | cells kept per reference type |
| `--pca-method` | `parallel_analysis` | or `scree` |
| `--n-permutations` | `100` | permutations for parallel analysis |
| `--pa-percentile` | `95.0` | null percentile |
| `--n-jobs` | `-1` | |
| `--layer` | | applies to both inputs |
| `--random-state` | `0` | |

Subsampling is optional and independent for the two datasets; with no keys given all cells
are used.

## reliability perturb

Take the failure modes from `score-coexp`, simulate each one, and measure the resulting drop
in cell type mapping accuracy. Returns several tables, so this command writes a **directory**
rather than a single file — one CSV per metric, including `delta_ct`, `baseline_ct`,
`summary`, `removal_benefit` and their `_constrained` counterparts.

```bash
pygenebasis reliability perturb --adata ref.h5ad --results scores.csv \
    --level-keys Class,Subclass,Group --out-dir perturb/
```

| Option | Default | |
|---|---|---|
| `--adata` `--results` `--out-dir` | required | |
| `--level-keys` | required | comma-separated obs columns, **coarsest first** |
| `--n-replicates` | `5` | perturbation replicates per failure mode |
| `--n-cells-per-group` | `5000` | cells subsampled per group at the finest level |
| `--vote` | `both` | `unconstrained`, `constrained` or `both` |
| `--batch-key` | | |
| `--knn-method` | `approx` | |
| `--n-neighbors` | `5` | |
| `--random-state` | `0` | |

`--level-keys` is a hierarchy, not a list of alternatives: each level must nest inside the
one before it. With `constrained` voting, a cell's neighbours are restricted to those sharing
its parent-level label, which separates within-lineage confusion from cross-lineage
confusion. The coarsest level has no parent, so constrained and unconstrained agree there.

---

## Longer runs

For jobs that need a scheduler, `docs/scripts/` holds argparse equivalents and SLURM
wrappers that run the same pipelines with more reporting:

- `run_panel_design.py` / `slurm_panel_design.sh` — HVG selection through trimming
- `run_reliability.py` / `slurm_reliability.sh` — the reliability pipeline
