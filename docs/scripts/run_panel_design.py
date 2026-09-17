#!/usr/bin/env python3
"""
pyGeneBasis panel design, trimming, and evaluation script.

Runs the full gene panel design workflow:
  1. Load AnnData (log-normalised counts expected in .X or --layer)
  2. HVG selection (scranpy, matching geneBasisR)
  3. gene_search: greedy panel selection
  4. Evaluation: cell scores, gene scores, cell type mapping
  5. trim_panel: remove the most redundant genes
  6. Save all outputs to --output-dir

Usage
-----
  python run_panel_design.py \\
      --adata      data.h5ad \\
      --n-genes    100 \\
      --batch-key  donor_id \\
      --celltype-key celltype \\
      --output-dir results/panel/

SLURM: see docs/scripts/slurm_panel_design.sh
CLI:   pygenebasis panel search / trim / evaluate
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_panel_design.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    req = p.add_argument_group("required")
    req.add_argument("--adata",      required=True, help="Path to .h5ad file.")
    req.add_argument("--n-genes",    required=True, type=int, help="Target panel size.")
    req.add_argument("--output-dir", required=True, help="Output directory (created if needed).")

    g = p.add_argument_group("data columns")
    g.add_argument("--batch-key",    default=None, help="obs column for batch correction.")
    g.add_argument("--celltype-key", default="celltype", help="obs column with cell type labels.")
    g.add_argument("--layer",        default=None, help="AnnData layer with logcounts (default: .X).")

    g = p.add_argument_group("gene selection")
    g.add_argument("--genes-base",    default=None,
                   help="CSV file with seed genes (column 'gene'), or comma-separated list.")
    g.add_argument("--n-remove",      default=0, type=int,
                   help="Genes to remove via trim_panel after selection (default: 0 = no trimming).")
    g.add_argument("--genes-protect", default=None,
                   help="Comma-separated genes that cannot be removed during trimming.")
    g.add_argument("--n-hvgs",        default=None, type=int,
                   help="Top N HVGs to retain (default: all genes with positive residual variance).")
    g.add_argument("--discard-prefix", default=None, nargs="+",
                   metavar="PREFIX",
                   help="Exclude genes with these name prefixes (e.g. MT- RPL RPS).")

    g = p.add_argument_group("algorithm")
    g.add_argument("--knn-method",   default="approx", choices=["approx", "exact"])
    g.add_argument("--batch-method", default="per_batch",
                   choices=["per_batch", "mnn", "harmony"],
                   help="Batch correction for gene_search/evaluation (default: per_batch).")
    g.add_argument("--n-neighbors",  default=5, type=int)
    g.add_argument("--n-pcs",        default=50, type=int)
    g.add_argument("--random-state", default=32, type=int)
    g.add_argument("--n-jobs",       default=-1, type=int,
                   help="Parallel workers for redundancy stat (-1 = all cores).")

    g = p.add_argument_group("output")
    g.add_argument("--skip-evaluation", action="store_true",
                   help="Skip evaluation (cell/gene scores, mapping). Saves time for large datasets.")
    g.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"],
                   help="Logging verbosity.")
    return p


def _parse_gene_list(spec: str | None, adata, label: str) -> list[str] | None:
    if spec is None:
        return None
    # Comma-separated inline list
    if "," in spec and not Path(spec).exists():
        return [g.strip() for g in spec.split(",") if g.strip()]
    # File path
    path = Path(spec)
    if path.exists():
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            col = "gene" if "gene" in df.columns else df.columns[0]
            genes = df[col].dropna().astype(str).tolist()
        else:
            genes = [
                l.strip() for l in path.read_text().splitlines()
                if l.strip() and not l.startswith("#")
            ]
        log.info("%s: %d genes from %s", label, len(genes), path)
        return genes
    # adata.var column
    if spec in adata.var.columns:
        mask = adata.var[spec].astype(bool)
        genes = adata.var_names[mask].tolist()
        log.info("%s: %d genes from adata.var['%s']", label, len(genes), spec)
        return genes
    raise ValueError(f"--{label}: '{spec}' is not a file, comma list, or adata.var column.")


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
    from pygenebasis.panel._evaluation import get_neighs_all_stat

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    log.info("Loading %s", args.adata)
    adata = pgb.read_adata(args.adata)
    log.info("  %d cells × %d genes", adata.n_obs, adata.n_vars)

    has_celltype = (
        not args.skip_evaluation
        and args.celltype_key in adata.obs.columns
    )
    if not has_celltype and not args.skip_evaluation:
        log.warning("--celltype-key '%s' not in adata.obs; skipping mapping evaluation",
                    args.celltype_key)

    # ------------------------------------------------------------------
    # 2. HVG selection
    # ------------------------------------------------------------------
    log.info("Running HVG selection (scranpy)...")
    adata_hvg = pgb.retain_informative_genes(
        adata, n=args.n_hvgs, flavor="scran", discard_mt=True, layer=args.layer,
    )
    hvg_genes = adata_hvg.var_names.tolist()
    log.info("  %d → %d HVGs", adata.n_vars, len(hvg_genes))
    pgb.write_csv(pd.DataFrame({"gene": hvg_genes}), out / "hvg_genes.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Gene search
    # ------------------------------------------------------------------
    genes_base = _parse_gene_list(args.genes_base, adata, "genes-base")
    genes_base = [g for g in (genes_base or []) if g in adata_hvg.var_names]

    log.info("Running gene_search (n_genes=%d)...", args.n_genes)
    t1 = time.time()
    panel_df = pgb.gene_search(
        adata_hvg, args.n_genes,
        genes_base=genes_base or None,
        genes_discard_prefix=args.discard_prefix,
        batch_key=args.batch_key,
        knn_method=args.knn_method,
        batch_method=args.batch_method,
        n_neighbors=args.n_neighbors,
        n_pcs=args.n_pcs,
        layer=args.layer,
        random_state=args.random_state,
        verbose=True,
    )
    log.info("  Selected %d genes in %.1fs", len(panel_df), time.time() - t1)
    pgb.write_csv(panel_df, out / "gene_panel.csv")
    panel_genes = panel_df["gene"].tolist()

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    if not args.skip_evaluation:
        log.info("Pre-computing true graph statistics...")
        neighs_all_stat = get_neighs_all_stat(
            adata_hvg, genes_all=hvg_genes,
            batch_key=args.batch_key,
            n_neighbors=args.n_neighbors,
            n_pcs_all=args.n_pcs,
            knn_method=args.knn_method,
            batch_method=args.batch_method,
            option="approx",
            layer=args.layer,
            random_state=args.random_state,
        )

        log.info("Cell neighbourhood preservation scores...")
        cell_scores = pgb.get_neighborhood_preservation_scores(
            adata_hvg,
            genes_selection=panel_genes,
            genes_all=hvg_genes,
            batch_key=args.batch_key,
            n_neighbors=args.n_neighbors,
            n_pcs_all=args.n_pcs,
            n_pcs_selection=None,
            knn_method=args.knn_method,
            batch_method=args.batch_method,
            neighs_all_stat=neighs_all_stat,
            layer=args.layer,
            random_state=args.random_state,
        )
        if has_celltype:
            cell_scores[args.celltype_key] = adata_hvg.obs[args.celltype_key].values
        log.info("  median=%.3f  mean=%.3f",
                 cell_scores["cell_score"].median(), cell_scores["cell_score"].mean())
        pgb.write_csv(cell_scores, out / "cell_scores.csv")

        log.info("Gene prediction scores...")
        gene_scores = pgb.get_gene_prediction_scores(
            adata_hvg,
            genes_selection=panel_genes,
            genes_all=hvg_genes,
            batch_key=args.batch_key,
            n_neighbors=args.n_neighbors,
            n_pcs_all=args.n_pcs,
            n_pcs_selection=None,
            knn_method=args.knn_method,
            batch_method=args.batch_method,
            layer=args.layer,
            random_state=args.random_state,
        )
        log.info("  median=%.3f  mean=%.3f",
                 gene_scores["gene_score"].median(), gene_scores["gene_score"].mean())
        pgb.write_csv(gene_scores, out / "gene_scores.csv")

        if has_celltype:
            log.info("Cell type mapping (batch_method=mnn)...")
            ct_result = pgb.get_celltype_mapping(
                adata_hvg,
                genes_selection=panel_genes,
                celltype_key=args.celltype_key,
                batch_key=args.batch_key,
                n_neighbors=args.n_neighbors,
                n_pcs_selection=None,
                knn_method=args.knn_method,
                batch_method="harmony",
                return_stat=True,
                layer=args.layer,
                random_state=args.random_state,
            )
            mapping_df = ct_result["mapping"]
            ct_stat    = ct_result["stat"]
            overall_acc = (
                mapping_df["mapped_celltype"] == mapping_df[args.celltype_key]
            ).mean()
            log.info("  overall accuracy=%.3f", overall_acc)
            pgb.write_csv(mapping_df, out / "celltype_mapping.csv")
            pgb.write_csv(ct_stat,    out / "celltype_mapping_stat.csv")

        # Plots
        _make_evaluation_plots(
            out, panel_genes, cell_scores, gene_scores,
            ct_result if has_celltype else None,
            args.celltype_key,
        )

    # ------------------------------------------------------------------
    # 5. Trimming
    # ------------------------------------------------------------------
    if args.n_remove > 0:
        gp = [g.strip() for g in args.genes_protect.split(",")
              if g.strip()] if args.genes_protect else None

        log.info("Trimming %d genes from panel of %d...", args.n_remove, len(panel_genes))
        t1 = time.time()
        trim_result = pgb.trim_panel(
            adata_hvg, panel_genes,
            n_remove=args.n_remove,
            genes_all=hvg_genes,
            genes_protect=gp,
            batch_key=args.batch_key,
            n_neighbors=args.n_neighbors,
            n_pcs_all=args.n_pcs,
            n_pcs_selection=50,
            knn_method=args.knn_method,
            batch_method=args.batch_method,
            layer=args.layer,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            verbose=True,
        )
        log.info("  Trimmed to %d genes in %.1fs",
                 len(trim_result["reduced_panel"]), time.time() - t1)

        trimmed_df = pd.DataFrame({"gene": trim_result["reduced_panel"]})
        pgb.write_csv(trimmed_df, out / "gene_panel_trimmed.csv", index=False)

        pgb.write_json({
            "removal_order":    trim_result["removal_order"],
            "score_trajectory": trim_result["score_trajectory"],
        }, out / "trim_trajectory.json")

        # Score trajectory plot
        traj = trim_result["score_trajectory"]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(len(traj)), traj, marker="o", ms=4, color="steelblue")
        ax.set_xlabel("Genes removed")
        ax.set_ylabel("Mean cell score")
        ax.set_title("Score trajectory during trimming")
        fig.tight_layout()
        fig.savefig(out / "trim_trajectory.png", dpi=150)
        plt.close(fig)

    log.info("Done. Total time: %.1fs", time.time() - t0)
    log.info("Outputs in: %s", out)


def _make_evaluation_plots(out, panel_genes, cell_scores, gene_scores, ct_result, ct_key):
    # Cell score histogram
    fig, ax = plt.subplots(figsize=(6, 3))
    scores = cell_scores["cell_score"].dropna()
    ax.hist(scores, bins=80, color="steelblue", edgecolor="none")
    ax.axvline(1, color="red", lw=1.2, ls="--")
    ax.set_xlabel("Cell neighbourhood preservation score")
    ax.set_ylabel("# cells")
    ax.set_title(f"{len(panel_genes)}-gene panel  (median={scores.median():.3f})")
    fig.tight_layout()
    fig.savefig(out / "cell_scores_hist.png", dpi=150)
    plt.close(fig)

    # Gene score (bottom 40)
    gs_sorted = gene_scores.dropna(subset=["gene_score"]).sort_values("gene_score")
    n_show = min(40, len(gs_sorted))
    fig, ax = plt.subplots(figsize=(5, max(3, n_show * 0.22)))
    ax.barh(gs_sorted["gene"].iloc[:n_show], gs_sorted["gene_score"].iloc[:n_show],
            color="salmon")
    ax.axvline(1, color="red", lw=1.2, ls="--")
    ax.set_xlabel("Gene prediction score")
    ax.set_title(f"Bottom {n_show} gene scores")
    fig.tight_layout()
    fig.savefig(out / "gene_scores_bottom.png", dpi=150)
    plt.close(fig)

    if ct_result is not None:
        import pygenebasis as pgb
        mapping_df = ct_result["mapping"]
        ct_stat    = ct_result["stat"]
        overall_acc = (mapping_df["mapped_celltype"] == mapping_df[ct_key]).mean()

        fig, ax = pgb.plot_mapping_heatmap(
            mapping_df, title=f"Cell type mapping — {len(panel_genes)}-gene panel"
        )
        fig.savefig(out / "mapping_heatmap.png", dpi=150)
        plt.close(fig)

        ct_sorted = ct_stat.sort_values("frac_correctly_mapped")
        fig, ax = plt.subplots(figsize=(5, max(3, len(ct_sorted) * 0.3)))
        ax.barh(ct_sorted["celltype"], ct_sorted["frac_correctly_mapped"],
                color="mediumseagreen")
        ax.axvline(1, color="red", lw=1.2, ls="--")
        ax.set_xlabel("Fraction correctly mapped")
        ax.set_title(f"Celltype accuracy  (overall={overall_acc:.3f})")
        ax.set_xlim(0, 1.05)
        fig.tight_layout()
        fig.savefig(out / "celltype_accuracy.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
