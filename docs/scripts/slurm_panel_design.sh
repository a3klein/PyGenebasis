#!/usr/bin/env bash
# =============================================================================
# slurm_panel_design.sh — Gene panel design on Slurm
#
# Runs the full panel design workflow (HVG filtering, gene_search, evaluation,
# and optional trimming) as a Slurm batch job.
#
# Usage
# -----
#   sbatch slurm_panel_design.sh
#
# Or override variables at submission time:
#   sbatch --export=ADATA=/data/ref.h5ad,N_GENES=150 slurm_panel_design.sh
#
# Alternatively, run the CLI commands directly in an interactive session:
#   srun --ntasks=1 --cpus-per-task=16 --mem=64G --pty bash
#   cd /path/to/repo/gene_panel
#   pixi run -e py-genebasis pygenebasis panel search --help
# =============================================================================

#SBATCH --job-name=genebasis_panel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/panel_%j.out
#SBATCH --error=logs/panel_%j.err

set -euo pipefail

# ── Edit these variables ─────────────────────────────────────────────────────
ADATA="${ADATA:-/path/to/reference.h5ad}"
OUTPUT_DIR="${OUTPUT_DIR:-results/panel_$(date +%Y%m%d_%H%M%S)}"
N_GENES="${N_GENES:-100}"
BATCH_KEY="${BATCH_KEY:-donor_id}"          # set to "" for single-batch
CELLTYPE_KEY="${CELLTYPE_KEY:-celltype}"
LAYER="${LAYER:-}"                          # set to e.g. "lognorm" if needed
N_REMOVE="${N_REMOVE:-0}"                   # genes to trim after selection

# Algorithm
KNN_METHOD="${KNN_METHOD:-approx}"
BATCH_METHOD="${BATCH_METHOD:-per_batch}"
N_NEIGHBORS="${N_NEIGHBORS:-5}"
N_PCS="${N_PCS:-50}"
N_JOBS="${N_JOBS:--1}"                      # -1 = all cores

# Optional: comma-separated seed genes or path to CSV
GENES_BASE="${GENES_BASE:-}"
GENES_PROTECT="${GENES_PROTECT:-}"         # only used when N_REMOVE > 0
# ─────────────────────────────────────────────────────────────────────────────

# Path to the gene_panel directory (adjust if running from a different location)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

echo "============================================================"
echo "pyGeneBasis panel design"
echo "  ADATA:       $ADATA"
echo "  N_GENES:     $N_GENES"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
echo "  BATCH_KEY:   $BATCH_KEY"
echo "  JOB_ID:      $SLURM_JOB_ID"
echo "============================================================"

mkdir -p logs "$OUTPUT_DIR"

# Activate environment
cd "$REPO_DIR"

# ── Option A: Python script ───────────────────────────────────────────────────
SCRIPT="software/geneBasis/pyGeneBasis/docs/scripts/run_panel_design.py"

ARGS=(
    --adata       "$ADATA"
    --n-genes     "$N_GENES"
    --output-dir  "$OUTPUT_DIR"
    --celltype-key "$CELLTYPE_KEY"
    --knn-method  "$KNN_METHOD"
    --batch-method "$BATCH_METHOD"
    --n-neighbors  "$N_NEIGHBORS"
    --n-pcs        "$N_PCS"
    --n-jobs       "$N_JOBS"
)

[[ -n "$BATCH_KEY"     ]] && ARGS+=(--batch-key     "$BATCH_KEY")
[[ -n "$LAYER"         ]] && ARGS+=(--layer          "$LAYER")
[[ -n "$GENES_BASE"    ]] && ARGS+=(--genes-base     "$GENES_BASE")
[[ "$N_REMOVE" -gt 0   ]] && ARGS+=(--n-remove       "$N_REMOVE")
[[ -n "$GENES_PROTECT" ]] && ARGS+=(--genes-protect  "$GENES_PROTECT")

pixi run -e py-genebasis python "$SCRIPT" "${ARGS[@]}"

# ── Option B: CLI (uncomment to use instead of the script above) ──────────────
# pixi run -e py-genebasis pygenebasis panel search \
#     --adata      "$ADATA" \
#     --n-genes    "$N_GENES" \
#     --batch-key  "$BATCH_KEY" \
#     --output     "$OUTPUT_DIR/gene_panel.csv"
#
# pixi run -e py-genebasis pygenebasis panel evaluate \
#     --adata       "$ADATA" \
#     --panel       "$OUTPUT_DIR/gene_panel.csv" \
#     --celltype-key "$CELLTYPE_KEY" \
#     --batch-key   "$BATCH_KEY" \
#     --output      "$OUTPUT_DIR/evaluation.csv"
#
# # Trim (optional)
# if [[ "$N_REMOVE" -gt 0 ]]; then
#     pixi run -e py-genebasis pygenebasis panel trim \
#         --adata    "$ADATA" \
#         --panel    "$OUTPUT_DIR/gene_panel.csv" \
#         --n-remove "$N_REMOVE" \
#         --batch-key "$BATCH_KEY" \
#         --output   "$OUTPUT_DIR/gene_panel_trimmed.csv"
# fi

echo "Done. Outputs in: $OUTPUT_DIR"
