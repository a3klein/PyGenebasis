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
@click.option("--batch-key-merfish", default=None, type=str,
              help="obs column for MERFISH batches.")
@click.option("--batch-key-ref", default=None, type=str,
              help="obs column for reference batches.")
@click.option("--pca-method", default="parallel_analysis", show_default=True,
              type=click.Choice(["parallel_analysis", "scree"]),
              help="Method for selecting number of PCs.")
@click.option("--n-permutations", default=100, show_default=True, type=int,
              help="Number of permutations for parallel analysis.")
@click.option("--n-jobs", default=-1, show_default=True, type=int,
              help="Parallel workers (-1 = all cores).")
@click.option("--layer-merfish", default=None, type=str,
              help="Layer in MERFISH adata with logcounts.")
@click.option("--layer-ref", default=None, type=str,
              help="Layer in reference adata with logcounts.")
@click.option("--log-dir", default=None, type=click.Path())
def reliability_score_coexp(
    merfish_path, ref_path, panel_path, output,
    batch_key_merfish, batch_key_ref,
    pca_method, n_permutations, n_jobs,
    layer_merfish, layer_ref, log_dir,
):
    """Score gene co-expression reliability using MERFISH vs scRNA-seq comparison.

    Computes per-gene reliability metrics (fidelity, idiosyncratic_noise,
    detection_noise, residual_structure) and writes scores to OUTPUT.
    """
    from ._utils import configure_logging
    configure_logging(log_dir=log_dir)

    from ..io import read_adata, read_table, write_csv
    from ..reliability import compute_corr_matrices, compute_gene_reliability

    log.info("Loading MERFISH: %s", merfish_path)
    adata_merfish = read_adata(merfish_path)
    log.info("Loading reference: %s", ref_path)
    adata_ref = read_adata(ref_path)

    panel_df = read_table(panel_path)
    panel = list(panel_df["gene"])
    log.info("Panel size: %d genes", len(panel))

    log.info("Computing co-expression matrices")
    corr_data = compute_corr_matrices(
        adata_merfish, adata_ref, panel,
        batch_key_merfish=batch_key_merfish,
        batch_key_ref=batch_key_ref,
        layer_merfish=layer_merfish,
        layer_ref=layer_ref,
    )

    log.info("Scoring gene reliability (pca_method=%s)", pca_method)
    scores, _ = compute_gene_reliability(
        corr_data,
        pca_method=pca_method,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
    )

    write_csv(scores, output)
    log.info("Reliability scores written to %s", output)


# ---------------------------------------------------------------------------
# pygenebasis reliability perturb
# ---------------------------------------------------------------------------

@click.command("perturb")
@click.option("--adata", "adata_path", required=True, type=click.Path(exists=True),
              help="Path to scRNA-seq reference .h5ad file.")
@click.option("--panel", "panel_path", required=True, type=click.Path(exists=True),
              help="CSV with a 'gene' column listing panel genes.")
@click.option("--results", "results_path", required=True, type=click.Path(exists=True),
              help="CSV with reliability scores (output of score-coexp).")
@click.option("--output", required=True, type=click.Path(),
              help="Output CSV path for perturbation analysis results.")
@click.option("--celltype-key", required=True, type=str,
              help="obs column with cell type labels.")
@click.option("--n-perturbations", default=10, show_default=True, type=int,
              help="Number of perturbation replicates.")
@click.option("--vote", default="both", show_default=True,
              type=click.Choice(["unconstrained", "constrained", "both"]),
              help="Voting strategy for cell type assignment.")
@click.option("--batch-key", default=None, type=str)
@click.option("--knn-method", default="approx", show_default=True,
              type=click.Choice(["approx", "exact"]))
@click.option("--batch-method", default="mnn", show_default=True,
              type=click.Choice(["per_batch", "mnn", "harmony"]))
@click.option("--n-jobs", default=-1, show_default=True, type=int)
@click.option("--log-dir", default=None, type=click.Path())
def reliability_perturb(
    adata_path, panel_path, results_path, output,
    celltype_key, n_perturbations, vote,
    batch_key, knn_method, batch_method, n_jobs, log_dir,
):
    """Run expression perturbation analysis on the scRNA-seq reference.

    For each reliability failure mode (from score-coexp), perturbs gene
    expression and measures impact on cell type mapping accuracy.
    """
    from ._utils import configure_logging
    configure_logging(log_dir=log_dir)

    from ..io import read_adata, read_table, write_csv
    from ..reliability import run_perturbation_analysis

    log.info("Loading %s", adata_path)
    adata = read_adata(adata_path)
    panel_df = read_table(panel_path)
    panel = list(panel_df["gene"])
    results_df = read_table(results_path)

    log.info("Running perturbation analysis (%d replicates)", n_perturbations)
    perturb_results = run_perturbation_analysis(
        adata, panel, results_df,
        celltype_key=celltype_key,
        n_perturbations=n_perturbations,
        vote=vote,
        batch_key=batch_key,
        knn_method=knn_method,
        batch_method=batch_method,
        n_jobs=n_jobs,
    )

    write_csv(perturb_results, output)
    log.info("Perturbation results written to %s", output)


# ---------------------------------------------------------------------------
# pygenebasis reliability group
# ---------------------------------------------------------------------------

@click.group("reliability")
def reliability_group():
    """Gene reliability analysis commands."""


reliability_group.add_command(reliability_score_coexp, name="score-coexp")
reliability_group.add_command(reliability_perturb, name="perturb")
