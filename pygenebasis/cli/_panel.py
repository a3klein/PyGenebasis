"""
Panel subcommands: search, trim, evaluate.
"""
from __future__ import annotations

import logging

import click

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pygenebasis panel search
# ---------------------------------------------------------------------------

@click.command("search")
@click.option("--adata", "adata_path", required=True, type=click.Path(exists=True),
              help="Path to .h5ad file (logcounts in .X).")
@click.option("--n-genes", required=True, type=int,
              help="Target panel size (number of genes to select).")
@click.option("--output", required=True, type=click.Path(),
              help="Output CSV path for the selected gene list.")
@click.option("--genes-base", default=None, type=str,
              help="Comma-separated list of seed genes to start from.")
@click.option("--genes-discard", default=None, type=str,
              help="Comma-separated genes to exclude from selection.")
@click.option("--batch-key", default=None, type=str,
              help="obs column identifying batches (None = no batch handling).")
@click.option("--knn-method", default="approx", show_default=True,
              type=click.Choice(["approx", "exact"]),
              help="kNN backend.")
@click.option("--batch-method", default="per_batch", show_default=True,
              type=click.Choice(["per_batch", "mnn", "harmony"]),
              help="Batch correction strategy.")
@click.option("--n-neighbors", default=5, show_default=True, type=int,
              help="Number of nearest neighbours k.")
@click.option("--n-pcs", default=50, show_default=True, type=int,
              help="Number of PCA components for graph construction.")
@click.option("--hvg-flavor", default="scran", show_default=True,
              type=click.Choice(["scran", "seurat", "seurat_v3", "methylation"]),
              help="HVG backend. Use 'methylation' for mCH/mCG rates.")
@click.option("--hvg-n", default=None, type=int,
              help="Keep this many HVGs before selection (default: the flavor's own cutoff).")
@click.option("--coverage-key", default="cov_mean", show_default=True,
              help="For --hvg-flavor methylation: adata.var column with per-gene mean coverage.")
@click.option("--layer", default=None, type=str,
              help="AnnData layer with logcounts (default: .X).")
@click.option("--log-dir", default=None, type=click.Path(),
              help="Directory for Slurm log files.")
def panel_search(
    adata_path, n_genes, output,
    genes_base, genes_discard,
    batch_key, knn_method, batch_method,
    n_neighbors, n_pcs, hvg_flavor, hvg_n, coverage_key, layer, log_dir,
):
    """Greedily select a gene panel of size N_GENES.

    Reads ADATA_PATH, runs gene_search, and writes the ordered gene list to
    OUTPUT (one gene per row, with iteration order).
    """
    from ._utils import configure_logging
    configure_logging(log_dir=log_dir)

    from ..io import read_adata, write_csv
    from ..panel import retain_informative_genes, gene_search

    log.info("Loading %s", adata_path)
    adata = read_adata(adata_path)

    log.info("Filtering to informative genes (flavor=%s)", hvg_flavor)
    adata = retain_informative_genes(
        adata, n=hvg_n, flavor=hvg_flavor, coverage_key=coverage_key, layer=layer,
    )

    gb = [g.strip() for g in genes_base.split(",")] if genes_base else None
    gd = [g.strip() for g in genes_discard.split(",")] if genes_discard else None

    log.info("Running gene_search (n_genes=%d)", n_genes)
    panel = gene_search(
        adata, n_genes,
        genes_base=gb,
        genes_discard=gd,
        batch_key=batch_key,
        knn_method=knn_method,
        batch_method=batch_method,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        layer=layer,
    )

    # gene_search returns a DataFrame with columns rank, gene
    write_csv(panel, output)
    log.info("Panel written to %s (%d genes)", output, len(panel))


# ---------------------------------------------------------------------------
# pygenebasis panel trim
# ---------------------------------------------------------------------------

@click.command("trim")
@click.option("--adata", "adata_path", required=True, type=click.Path(exists=True),
              help="Path to .h5ad file.")
@click.option("--panel", "panel_path", required=True, type=click.Path(exists=True),
              help="CSV with a 'gene' column (output of panel search or similar).")
@click.option("--n-remove", required=True, type=int,
              help="Number of genes to remove from the panel.")
@click.option("--output", required=True, type=click.Path(),
              help="Output CSV path for the trimmed gene list.")
@click.option("--genes-protect", default=None, type=str,
              help="Comma-separated genes that cannot be removed.")
@click.option("--batch-key", default=None, type=str)
@click.option("--knn-method", default="approx", show_default=True,
              type=click.Choice(["approx", "exact"]))
@click.option("--batch-method", default="per_batch", show_default=True,
              type=click.Choice(["per_batch", "mnn", "harmony"]))
@click.option("--n-neighbors", default=5, show_default=True, type=int)
@click.option("--n-jobs", default=-1, show_default=True, type=int,
              help="Parallel workers over leave-one-out candidates (-1 = all cores).")
@click.option("--layer", default=None, type=str)
@click.option("--log-dir", default=None, type=click.Path())
def panel_trim(
    adata_path, panel_path, n_remove, output,
    genes_protect, batch_key, knn_method, batch_method,
    n_neighbors, n_jobs, layer, log_dir,
):
    """Remove N_REMOVE genes from a panel, keeping the most informative ones.

    Reads ADATA_PATH and PANEL_PATH, runs trim_panel, writes the trimmed gene
    list to OUTPUT.
    """
    from ._utils import configure_logging
    configure_logging(log_dir=log_dir)

    from ..io import read_adata, read_table, write_csv
    from ..panel import trim_panel
    import pandas as pd

    log.info("Loading %s", adata_path)
    adata = read_adata(adata_path)
    panel_df = read_table(panel_path)
    panel = list(panel_df["gene"])

    gp = [g.strip() for g in genes_protect.split(",")] if genes_protect else None

    log.info("Trimming panel from %d → %d genes", len(panel), len(panel) - n_remove)
    result = trim_panel(
        adata, panel,
        n_remove=n_remove,
        genes_protect=gp,
        batch_key=batch_key,
        knn_method=knn_method,
        batch_method=batch_method,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
        layer=layer,
    )

    df = pd.DataFrame({"gene": result["reduced_panel"]})
    write_csv(df, output)
    log.info("Trimmed panel written to %s (%d genes)", output, len(df))


# ---------------------------------------------------------------------------
# pygenebasis panel evaluate
# ---------------------------------------------------------------------------

@click.command("evaluate")
@click.option("--adata", "adata_path", required=True, type=click.Path(exists=True),
              help="Path to .h5ad file.")
@click.option("--panel", "panel_path", required=True, type=click.Path(exists=True),
              help="CSV with a 'gene' column.")
@click.option("--output", required=True, type=click.Path(),
              help="Output CSV path for evaluation metrics.")
@click.option("--celltype-key", default=None, type=str,
              help="obs column with cell type labels (enables mapping accuracy).")
@click.option("--batch-key", default=None, type=str)
@click.option("--knn-method", default="approx", show_default=True,
              type=click.Choice(["approx", "exact"]))
@click.option("--batch-method", default="per_batch", show_default=True,
              type=click.Choice(["per_batch", "mnn", "harmony"]))
@click.option("--n-neighbors", default=5, show_default=True, type=int)
@click.option("--layer", default=None, type=str)
@click.option("--log-dir", default=None, type=click.Path())
def panel_evaluate(
    adata_path, panel_path, output, celltype_key,
    batch_key, knn_method, batch_method, n_neighbors, layer, log_dir,
):
    """Evaluate neighbourhood preservation for a gene panel.

    Computes per-cell neighbourhood preservation scores and optionally cell
    type mapping accuracy when --celltype-key is provided.
    """
    from ._utils import configure_logging
    configure_logging(log_dir=log_dir)

    from ..io import read_adata, read_table, write_csv
    from ..panel import evaluate_library

    log.info("Loading %s", adata_path)
    adata = read_adata(adata_path)
    panel_df = read_table(panel_path)
    panel = list(panel_df["gene"])

    log.info("Evaluating panel of %d genes", len(panel))
    results = evaluate_library(
        adata, panel,
        celltype_key=celltype_key,
        batch_key=batch_key,
        knn_method=knn_method,
        batch_method=batch_method,
        n_neighbors=n_neighbors,
        layer=layer,
    )

    # evaluate_library returns one frame per metric; --output names the cell
    # scores, the others are written beside it.
    from pathlib import Path
    out = Path(output)
    write_csv(results["cell_score_stat"], out)
    log.info("Cell scores written to %s", out)
    for key, suffix in (("gene_score_stat", "_gene"), ("celltype_stat", "_celltype")):
        if key in results:
            sibling = out.with_name(f"{out.stem}{suffix}{out.suffix}")
            write_csv(results[key], sibling)
            log.info("%s written to %s", key, sibling)


# ---------------------------------------------------------------------------
# pygenebasis panel group
# ---------------------------------------------------------------------------

@click.group("panel")
def panel_group():
    """Gene panel design commands."""


panel_group.add_command(panel_search)
panel_group.add_command(panel_trim)
panel_group.add_command(panel_evaluate)
