#!/usr/bin/env python3
"""
pyGeneBasis gene reliability analysis script.

Runs the full reliability pipeline:
  1. Load MERFISH and scRNA-seq reference AnnData
  2. Compute co-expression matrices (gene × gene Pearson correlations)
  3. Score gene reliability: classify each gene into a failure mode
  4. Run perturbation analysis: quantify impact on cell type mapping accuracy
  5. Save all outputs and plots to --output-dir

Failure modes
-------------
  reliable              Co-expression consistent between MERFISH and reference
  probe_failure         Gene poorly detected in MERFISH (vs. reference)
  idiosyncratic_noise   Structured residuals not aligned with PCA noise
  composition_mismatch  Co-expression difference driven by cell type composition

Usage
-----
  python run_reliability.py \\
      --merfish    merfish.h5ad \\
      --ref        reference.h5ad \\
      --panel      gene_panel.csv \\
      --output-dir results/reliability/ \\
      --level-keys Class Subclass Group \\
      --batch-key  donor_id

SLURM: see docs/scripts/slurm_reliability.sh
CLI:   pygenebasis reliability score-coexp / perturb
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_reliability.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    req = p.add_argument_group("required")
    req.add_argument("--merfish",    required=True, help="Path to MERFISH .h5ad.")
    req.add_argument("--ref",        required=True, help="Path to scRNA-seq reference .h5ad.")
    req.add_argument("--panel",      required=True,
                     help="CSV with a 'gene' column listing panel genes.")
    req.add_argument("--output-dir", required=True, help="Output directory.")
    req.add_argument("--level-keys", required=True, nargs="+",
                     help="Annotation columns ordered coarsest → finest "
                          "(e.g. Class Subclass Group).")

    g = p.add_argument_group("data columns")
    g.add_argument("--batch-key",        default=None,
                   help="obs column for batch correction in perturbation analysis.")
    g.add_argument("--merfish-obs-file", default=None,
                   help="Optional .tsv annotation file to merge into MERFISH obs.")
    g.add_argument("--ref-obs-file",     default=None,
                   help="Optional .tsv annotation file to merge into reference obs.")
    g.add_argument("--layer",            default=None,
                   help="Expression layer in both objects (default: .X).")

    g = p.add_argument_group("co-expression subsampling")
    g.add_argument("--merfish-celltype-key",       default=None,
                   help="obs column for stratified MERFISH subsampling.")
    g.add_argument("--merfish-n-cells-per-type",   default=None, type=int,
                   help="Max MERFISH cells per type (requires --merfish-celltype-key).")
    g.add_argument("--ref-celltype-key",           default=None,
                   help="obs column for stratified reference subsampling.")
    g.add_argument("--ref-n-cells-per-type",       default=None, type=int,
                   help="Max reference cells per type (requires --ref-celltype-key).")

    g = p.add_argument_group("reliability scoring")
    g.add_argument("--pca-method",     default="parallel_analysis",
                   choices=["parallel_analysis", "scree"],
                   help="PC selection method (default: parallel_analysis).")
    g.add_argument("--n-permutations", default=100, type=int,
                   help="Permutations for parallel analysis (default: 100).")
    g.add_argument("--n-jobs",         default=-1, type=int,
                   help="Parallel workers (-1 = all cores).")

    g = p.add_argument_group("perturbation analysis")
    g.add_argument("--n-replicates",      default=5, type=int,
                   help="Perturbation replicates per failure mode (default: 5).")
    g.add_argument("--n-cells-per-group", default=5000, type=int,
                   help="Cells sampled per group in perturbation analysis (default: 5000).")
    g.add_argument("--vote",              default="both",
                   choices=["unconstrained", "constrained", "both"],
                   help="Majority-vote strategy (default: both).")
    g.add_argument("--n-neighbors",       default=5, type=int)
    g.add_argument("--skip-perturbation", action="store_true",
                   help="Only run co-expression scoring; skip perturbation analysis.")

    g = p.add_argument_group("output")
    g.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    import pygenebasis as pgb

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    log.info("Loading MERFISH: %s", args.merfish)
    adata_merfish = pgb.read_adata(
        args.merfish,
        obs_file=args.merfish_obs_file,
    )
    log.info("  %d cells × %d genes", adata_merfish.n_obs, adata_merfish.n_vars)

    log.info("Loading reference: %s", args.ref)
    adata_ref = pgb.read_adata(
        args.ref,
        obs_file=args.ref_obs_file,
    )
    log.info("  %d cells × %d genes", adata_ref.n_obs, adata_ref.n_vars)

    panel_df    = pgb.read_table(args.panel)
    panel_genes = list(panel_df["gene"])
    in_both = [
        g for g in panel_genes
        if g in adata_merfish.var_names and g in adata_ref.var_names
    ]
    log.info("Panel: %d genes (%d in both datasets)", len(panel_genes), len(in_both))
    if len(in_both) < len(panel_genes):
        missing = set(panel_genes) - set(in_both)
        log.warning("  Genes not in both datasets (excluded): %s", sorted(missing))

    # ------------------------------------------------------------------
    # 2. Co-expression matrices
    # ------------------------------------------------------------------
    log.info("Computing co-expression matrices...")
    t1 = time.time()
    corr_data = pgb.compute_corr_matrices(
        adata_merfish, adata_ref,
        genes=in_both,
        merfish_cell_type_key=args.merfish_celltype_key,
        merfish_n_cells_per_type=args.merfish_n_cells_per_type,
        ref_cell_type_key=args.ref_celltype_key,
        ref_n_cells_per_type=args.ref_n_cells_per_type,
        layer=args.layer,
    )
    log.info("  Done in %.1fs  (matrix shape: %s)", time.time() - t1,
             corr_data["merfish_corr"].shape)

    # Correlation scatter plot
    import numpy as np
    off_diag = np.triu(np.ones_like(corr_data["ref_corr"], dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(corr_data["ref_corr"][off_diag],
               corr_data["merfish_corr"][off_diag],
               s=0.3, alpha=0.3, rasterized=True)
    lim = max(abs(corr_data["ref_corr"][off_diag]).max(),
              abs(corr_data["merfish_corr"][off_diag]).max()) * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axline((0, 0), slope=1, color="red", lw=0.8, ls="--")
    ax.set_xlabel("Reference co-expression (Pearson r)")
    ax.set_ylabel("MERFISH co-expression (Pearson r)")
    ax.set_title("Co-expression agreement (gene pairs)")
    fig.tight_layout()
    fig.savefig(out / "coexpression_scatter.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Reliability scoring
    # ------------------------------------------------------------------
    log.info("Scoring gene reliability (pca_method=%s)...", args.pca_method)
    t1 = time.time()
    results, thresholds = pgb.score_gene_reliability(
        corr_data,
        pca_method=args.pca_method,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
    )
    log.info("  Done in %.1fs", time.time() - t1)

    log.info("Failure mode breakdown:")
    for fm, cnt in results["failure_mode"].value_counts().items():
        log.info("  %-30s %d", fm, cnt)

    pgb.write_csv(results, out / "reliability_scores.csv")

    # Scree / parallel analysis plot
    if "_pca" in thresholds:
        _plot_pc_selection(thresholds, out)

    # Metric distribution and scatter plots
    try:
        fig, axes = pgb.plot_metric_distributions(results, thresholds)
        fig.savefig(out / "metric_distributions.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        log.warning("plot_metric_distributions failed: %s", e)

    try:
        fig, ax = pgb.plot_failure_mode_scatter(results)
        fig.savefig(out / "failure_mode_scatter.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        log.warning("plot_failure_mode_scatter failed: %s", e)

    if args.skip_perturbation:
        log.info("--skip-perturbation set; done.")
        return

    # ------------------------------------------------------------------
    # 4. Perturbation analysis
    # ------------------------------------------------------------------
    # Subset reference to panel genes
    keep = [g for g in results.index if g in adata_ref.var_names]
    adata_panel = adata_ref[:, keep].copy()

    log.info("Perturbation analysis (%d replicates, vote=%s)...",
             args.n_replicates, args.vote)
    t1 = time.time()
    perturb = pgb.run_perturbation_analysis(
        adata_panel, results,
        level_keys=args.level_keys,
        batch_key=args.batch_key,
        n_neighbors=args.n_neighbors,
        n_replicates=args.n_replicates,
        n_cells_per_group=args.n_cells_per_group,
        vote=args.vote,
        verbose=True,
    )
    log.info("  Done in %.1fs", time.time() - t1)

    for key, df in perturb.items():
        pgb.write_csv(df, out / f"perturb_{key}.csv")

    log.info("\nSummary (mean delta CT):")
    log.info(perturb["summary"].to_string(index=False))

    # Perturbation plots
    _make_perturbation_plots(out, perturb, args.level_keys, args.vote)

    log.info("Done. Total time: %.1fs", time.time() - t0)
    log.info("Outputs in: %s", out)


def _plot_pc_selection(thresholds: dict, out: Path) -> None:
    import numpy as np
    pca_info    = thresholds["_pca"]
    real_eig    = pca_info["real_eigenvalues"]
    null_thresh = pca_info["null_threshold"]
    evr         = pca_info["explained_variance_ratio"]
    n_pcs_used  = thresholds["n_pcs"]
    ranks       = np.arange(1, len(real_eig) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.bar(ranks, real_eig, color="steelblue", alpha=0.7, label="Real eigenvalue")
    ax1.plot(ranks, null_thresh, color="red", lw=1.5,
             label="95th-pct null (parallel analysis)")
    ax1.axvline(n_pcs_used + 0.5, color="black", lw=1.2, ls="--",
                label=f"k = {n_pcs_used} PCs chosen")
    cum_var = np.cumsum(evr) * 100
    ax2.plot(ranks, cum_var, color="orange", lw=1.5, ls=":", label="Cumulative variance (%)")
    ax2.set_ylabel("Cumulative variance (%)", color="orange")
    ax2.tick_params(axis="y", colors="orange")
    ax1.set_xlabel("PC rank")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title(f"PC selection — {n_pcs_used} PCs chosen")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pc_selection.png", dpi=150)
    plt.close(fig)


def _make_perturbation_plots(out, perturb, level_keys, vote):
    import pygenebasis as pgb

    try:
        fig, axes = pgb.plot_perturbation_summary(
            perturb["summary"],
            title="Mean drop in cell type mapping accuracy by failure mode",
        )
        fig.savefig(out / "perturbation_summary.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        log.warning("plot_perturbation_summary failed: %s", e)

    for level in level_keys:
        try:
            fig, ax = pgb.plot_delta_ct_heatmap(
                perturb["delta_ct"], level=level,
                title=f"Delta CT — {level}",
            )
            fig.savefig(out / f"delta_ct_{level}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            log.warning("plot_delta_ct_heatmap(%s) failed: %s", level, e)

        try:
            fig, ax = pgb.plot_removal_benefit(
                perturb["removal_benefit"], level=level,
                title=f"Removal benefit — {level}",
            )
            fig.savefig(out / f"removal_benefit_{level}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            log.warning("plot_removal_benefit(%s) failed: %s", level, e)

    if vote in ("both", "constrained") and "summary_constrained" in perturb:
        try:
            fig, axes = pgb.plot_perturbation_summary(
                perturb["summary_constrained"],
                title="Mean delta CT (constrained vote)",
            )
            fig.savefig(out / "perturbation_summary_constrained.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            log.warning("plot_perturbation_summary (constrained) failed: %s", e)


if __name__ == "__main__":
    main()
