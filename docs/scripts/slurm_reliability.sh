#!/usr/bin/env bash
# =============================================================================
# slurm_reliability.sh — Gene reliability analysis on Slurm
#
# Runs co-expression scoring (MERFISH vs scRNA-seq) and perturbation analysis
# as a Slurm batch job.
#
# Usage
# -----
#   sbatch slurm_reliability.sh
#
# Or override at submission:
#   sbatch --export=MERFISH=/data/merfish.h5ad,REF=/data/ref.h5ad slurm_reliability.sh
#
# Alternatively, use the CLI in an interactive session:
#   srun --ntasks=1 --cpus-per-task=16 --mem=64G --pty bash
#   pixi run -e py-genebasis pygenebasis reliability score-coexp --help
# =============================================================================

#SBATCH --job-name=genebasis_reliability
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/reliability_%j.out
#SBATCH --error=logs/reliability_%j.err

set -euo pipefail

# ── Edit these variables ─────────────────────────────────────────────────────
MERFISH="${MERFISH:-/path/to/merfish.h5ad}"
REF="${REF:-/path/to/reference.h5ad}"
PANEL="${PANEL:-results/panel/gene_panel.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-results/reliability_$(date +%Y%m%d_%H%M%S)}"

# Annotation columns ordered coarsest → finest
# Pass as a space-separated string: "Class Subclass Group"
LEVEL_KEYS="${LEVEL_KEYS:-Class Subclass Group}"

BATCH_KEY="${BATCH_KEY:-donor_id}"          # set to "" for single-batch
LAYER="${LAYER:-}"                          # set to e.g. "lognorm" if needed

# Subsampling (optional — recommended for large datasets)
MERFISH_CELLTYPE_KEY="${MERFISH_CELLTYPE_KEY:-}"
MERFISH_N_CELLS_PER_TYPE="${MERFISH_N_CELLS_PER_TYPE:-}"   # e.g. 200
REF_CELLTYPE_KEY="${REF_CELLTYPE_KEY:-}"
REF_N_CELLS_PER_TYPE="${REF_N_CELLS_PER_TYPE:-}"           # e.g. 200

# Reliability scoring
PCA_METHOD="${PCA_METHOD:-parallel_analysis}"
N_PERMUTATIONS="${N_PERMUTATIONS:-100}"
N_JOBS="${N_JOBS:--1}"

# Perturbation analysis
N_REPLICATES="${N_REPLICATES:-5}"
N_CELLS_PER_GROUP="${N_CELLS_PER_GROUP:-5000}"
VOTE="${VOTE:-both}"
N_NEIGHBORS="${N_NEIGHBORS:-5}"
SKIP_PERTURBATION="${SKIP_PERTURBATION:-false}"
# ─────────────────────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

echo "============================================================"
echo "pyGeneBasis reliability analysis"
echo "  MERFISH:      $MERFISH"
echo "  REF:          $REF"
echo "  PANEL:        $PANEL"
echo "  OUTPUT_DIR:   $OUTPUT_DIR"
echo "  LEVEL_KEYS:   $LEVEL_KEYS"
echo "  JOB_ID:       $SLURM_JOB_ID"
echo "============================================================"

mkdir -p logs "$OUTPUT_DIR"

cd "$REPO_DIR"

SCRIPT="software/geneBasis/pyGeneBasis/docs/scripts/run_reliability.py"

# Build argument array
ARGS=(
    --merfish      "$MERFISH"
    --ref          "$REF"
    --panel        "$PANEL"
    --output-dir   "$OUTPUT_DIR"
    --level-keys   $LEVEL_KEYS        # intentional word-split
    --pca-method   "$PCA_METHOD"
    --n-permutations "$N_PERMUTATIONS"
    --n-replicates   "$N_REPLICATES"
    --n-cells-per-group "$N_CELLS_PER_GROUP"
    --vote           "$VOTE"
    --n-neighbors    "$N_NEIGHBORS"
    --n-jobs         "$N_JOBS"
)

[[ -n "$BATCH_KEY"              ]] && ARGS+=(--batch-key                "$BATCH_KEY")
[[ -n "$LAYER"                  ]] && ARGS+=(--layer                    "$LAYER")
[[ -n "$MERFISH_CELLTYPE_KEY"   ]] && ARGS+=(--merfish-celltype-key     "$MERFISH_CELLTYPE_KEY")
[[ -n "$MERFISH_N_CELLS_PER_TYPE" ]] && ARGS+=(--merfish-n-cells-per-type "$MERFISH_N_CELLS_PER_TYPE")
[[ -n "$REF_CELLTYPE_KEY"       ]] && ARGS+=(--ref-celltype-key         "$REF_CELLTYPE_KEY")
[[ -n "$REF_N_CELLS_PER_TYPE"   ]] && ARGS+=(--ref-n-cells-per-type     "$REF_N_CELLS_PER_TYPE")
[[ "$SKIP_PERTURBATION" == "true" ]] && ARGS+=(--skip-perturbation)

# ── Option A: Python script ───────────────────────────────────────────────────
pixi run -e py-genebasis python "$SCRIPT" "${ARGS[@]}"

# ── Option B: CLI (uncomment to use instead) ─────────────────────────────────
# pixi run -e py-genebasis pygenebasis reliability score-coexp \
#     --merfish  "$MERFISH" \
#     --ref      "$REF" \
#     --panel    "$PANEL" \
#     --output   "$OUTPUT_DIR/reliability_scores.csv" \
#     --pca-method "$PCA_METHOD"
#
# pixi run -e py-genebasis pygenebasis reliability perturb \
#     --adata    "$REF" \
#     --panel    "$PANEL" \
#     --results  "$OUTPUT_DIR/reliability_scores.csv" \
#     --celltype-key "${LEVEL_KEYS##* }" \
#     --output   "$OUTPUT_DIR/perturbation_results.csv" \
#     --vote     "$VOTE"

echo "Done. Outputs in: $OUTPUT_DIR"
