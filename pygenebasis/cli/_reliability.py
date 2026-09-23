"""
Reliability subcommands: score-coexp, perturb.
"""
from __future__ import annotations

import logging

import click

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pygenebasis reliability score-coexp
# ---------------------------------------------------------------------------

@click.command("score-coexp")
@click.option("--merfish", "merfish_path", required=True, type=click.Path(exists=True),
              help="Path to MERFISH .h5ad file.")
@click.option("--ref", "ref_path", required=True, type=click.Path(exists=True),
              help="Path to scRNA-seq reference .h5ad file.")
@click.option("--panel", "panel_path", required=True, type=click.Path(exists=True),
              help="CSV with a 'gene' column listing panel genes.")
@click.option("--output", required=True, type=click.Path(),
              help="Output CSV path for reliability scores.")
@click.option("--merfish-cell-type-key", default=None, type=str,
              help="obs column in the MERFISH data to stratify subsampling by.")
@click.option("--merfish-n-cells-per-type", default=None, type=int,
              help="Cells to keep per MERFISH cell type (requires the key above).")
@click.option("--ref-cell-type-key", default=None, type=str,
              help="obs column in the reference to stratify subsampling by.")
@click.option("--ref-n-cells-per-type", default=None, type=int,
              help="Cells to keep per reference cell type (requires the key above).")
@click.option("--pca-method", default="parallel_analysis", show_default=True,
              type=click.Choice(["parallel_analysis", "scree"]),
              help="Method for selecting number of PCs.")
@click.option("--n-permutations", default=100, show_default=True, type=int,
              help="Number of permutations for parallel analysis.")
@click.option("--pa-percentile", default=95.0, show_default=True, type=float,
              help="Null percentile for parallel analysis.")
@click.option("--n-jobs", default=-1, show_default=True, type=int,
              help="Parallel workers (-1 = all cores).")
@click.option("--layer", default=None, type=str,
              help="Layer with logcounts in both AnnDatas (default: .X).")
@click.option("--random-state", default=0, show_default=True, type=int)
@click.option("--log-dir", default=None, type=click.Path())
def reliability_score_coexp(
    merfish_path, ref_path, panel_path, output,
    merfish_cell_type_key, merfish_n_cells_per_type,
    ref_cell_type_key, ref_n_cells_per_type,
    pca_method, n_permutations, pa_percentile, n_jobs,
    layer, random_state, log_dir,
):
    """Score gene co-expression reliability using MERFISH vs scRNA-seq comparison.

    Computes per-gene reliability metrics (fidelity, idiosyncratic_noise,
    detection_noise, residual_structure) and writes scores to OUTPUT.
    """
    from ._utils import configure_logging
    configure_logging(log_dir=log_dir)

    from ..io import read_adata, read_table, write_csv
    from ..reliability import compute_gene_reliability

    log.info("Loading MERFISH: %s", merfish_path)
    adata_merfish = read_adata(merfish_path)
    log.info("Loading reference: %s", ref_path)
    adata_ref = read_adata(ref_path)

    panel_df = read_table(panel_path)
    panel = list(panel_df["gene"])
    log.info("Panel size: %d genes", len(panel))

    log.info("Scoring gene reliability (pca_method=%s)", pca_method)
    scores, _ = compute_gene_reliability(
        adata_merfish, adata_ref,
        genes=panel,
        merfish_cell_type_key=merfish_cell_type_key,
        merfish_n_cells_per_type=merfish_n_cells_per_type,
        ref_cell_type_key=ref_cell_type_key,
        ref_n_cells_per_type=ref_n_cells_per_type,
        layer=layer,
        pca_method=pca_method,
        n_permutations=n_permutations,
        pa_percentile=pa_percentile,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    write_csv(scores, output)
    log.info("Reliability scores written to %s", output)


# ---------------------------------------------------------------------------
# pygenebasis reliability perturb
# ---------------------------------------------------------------------------

@click.command("perturb")
@click.option("--adata", "adata_path", required=True, type=click.Path(exists=True),
              help="Path to scRNA-seq reference .h5ad file.")
@click.option("--results", "results_path", required=True, type=click.Path(exists=True),
              help="CSV with reliability scores (output of score-coexp).")
@click.option("--out-dir", required=True, type=click.Path(),
              help="Directory for the per-metric CSVs this command writes.")
@click.option("--level-keys", required=True, type=str,
              help="Comma-separated obs columns, coarsest first (e.g. Class,Subclass,Group).")
@click.option("--n-replicates", default=5, show_default=True, type=int,
              help="Perturbation replicates per failure mode.")
@click.option("--n-cells-per-group", default=5000, show_default=True, type=int,
              help="Cells subsampled per group at the finest level.")
@click.option("--vote", default="both", show_default=True,
              type=click.Choice(["unconstrained", "constrained", "both"]),
              help="Voting strategy for cell type assignment.")
@click.option("--batch-key", default=None, type=str)
@click.option("--knn-method", default="approx", show_default=True,
              type=click.Choice(["approx", "exact"]))
@click.option("--n-neighbors", default=5, show_default=True, type=int)
@click.option("--random-state", default=0, show_default=True, type=int)
@click.option("--log-dir", default=None, type=click.Path())
def reliability_perturb(
    adata_path, results_path, out_dir,
    level_keys, n_replicates, n_cells_per_group, vote,
    batch_key, knn_method, n_neighbors, random_state, log_dir,
):
    """Run expression perturbation analysis on the scRNA-seq reference.

    For each reliability failure mode (from score-coexp), perturbs gene
    expression and measures impact on cell type mapping accuracy.
    """
    from ._utils import configure_logging
    configure_logging(log_dir=log_dir)

    from pathlib import Path

    from ..io import read_adata, read_table, write_csv
    from ..reliability import run_perturbation_analysis

    log.info("Loading %s", adata_path)
    adata = read_adata(adata_path)
    results_df = read_table(results_path)
    levels = [k.strip() for k in level_keys.split(",") if k.strip()]

    log.info("Running perturbation analysis (%d replicates, levels=%s)",
             n_replicates, levels)
    perturb_results = run_perturbation_analysis(
        adata, results_df,
        level_keys=levels,
        batch_key=batch_key,
        knn_method=knn_method,
        n_neighbors=n_neighbors,
        n_replicates=n_replicates,
        n_cells_per_group=n_cells_per_group,
        vote=vote,
        random_state=random_state,
    )

    # One CSV per returned metric.
    out = Path(out_dir)
    for key, df in perturb_results.items():
        write_csv(df, out / f"{key}.csv")
    log.info("Wrote %d tables to %s", len(perturb_results), out)


# ---------------------------------------------------------------------------
# pygenebasis reliability group
# ---------------------------------------------------------------------------

@click.group("reliability")
def reliability_group():
    """Gene reliability analysis commands."""


reliability_group.add_command(reliability_score_coexp, name="score-coexp")
reliability_group.add_command(reliability_perturb, name="perturb")
