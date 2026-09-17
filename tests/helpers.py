"""
Shared constants, tolerances, and helper functions for pyGeneBasis tests.

Imported directly by test modules (unlike conftest.py, which pytest manages).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Tolerance constants (from DEVELOPMENT.md)
# ---------------------------------------------------------------------------

SPEARMAN_TOL        = 0.95   # min Spearman ρ between Python and R score vectors
# Cell score MAE is not used directly; the test uses Spearman instead.
# Kept for backward compat in case any test still imports it.
MAE_TOL             = 0.01   # max mean absolute error for cell scores
GENE_OVERLAP_EXACT  = 0.90   # min gene panel overlap: exact kNN vs R
GENE_OVERLAP_APPROX = 0.85   # min gene panel overlap: approx kNN vs R (reference tests)
# On the synthetic small_adata, many genes are equally informative within each
# cell type, so the greedy algorithm diverges early between exact and approx kNN.
# This lower threshold is used only for the contract test; real-data reference
# tests still enforce GENE_OVERLAP_APPROX = 0.85.
GENE_OVERLAP_APPROX_CONTRACT = 0.40
KNN_OVERLAP_TOL          = 0.90   # min per-row neighbour overlap: approx vs exact (contract)
# Per-batch PCA differs numerically between R (irlba) and Python (sklearn),
# leading to ~5% kNN disagreement even with the correct algorithm.
KNN_OVERLAP_TOL_REFERENCE = 0.80   # min kNN overlap for Python vs R reference tests
CT_AGREEMENT_TOL    = 0.90   # min cell-level celltype mapping agreement vs R

# Jaccard similarity of the full directed edge set: approx vs exact kNN.
# Stricter than per-row overlap because it counts every (cell, neighbour) pair.
KNN_JACCARD_TOL = 0.85

# gene_search_adata has 25 truly informative genes (5 markers × 5 cell types)
# out of 200 total.  We require that at least this fraction of the n_genes
# selected by gene_search come from the informative set.
GENE_SEARCH_INFORMATIVE_TOL = 0.60

# retain_informative_genes on gene_search_adata should keep mostly the 25
# informative genes.  We require that at least this fraction of retained genes
# are in the truly informative set.
RETAIN_INFORMATIVE_PURITY_TOL = 0.60

# ---------------------------------------------------------------------------
# Shared parameters — must match generate_r_reference.R
# ---------------------------------------------------------------------------

N_NEIGHBORS  = 5
N_PCS_ALL    = 50
P_MINKOWSKI  = 3.0
N_GENES      = 50
RANDOM_STATE = 32

SEED_GENES_MOUSE = ["Hba-x", "Acta2", "Ttr", "Crabp1", "Hoxaas3"]
SEED_GENES_BG_SN = ["RELN", "TH", "MOBP", "APOE", "AIF1", "SNAP25", "SLC17A6", "PDGFRA"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def overlap_fraction(a: list | np.ndarray, b: list | np.ndarray) -> float:
    """Fraction of elements in a that appear in b."""
    return len(set(a) & set(b)) / len(set(a))


def knn_overlap(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    """Mean per-row fraction of neighbours in common between two index matrices."""
    fracs = [
        len(set(row_a) & set(row_b)) / len(row_a)
        for row_a, row_b in zip(idx_a, idx_b)
    ]
    return float(np.mean(fracs))


def knn_jaccard(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    """Jaccard similarity of the full directed edge sets of two kNN index matrices.

    Each (cell_i, neighbour_j) pair is an edge.  Unlike knn_overlap (which
    averages per-row recall), this is a single global score and is more
    sensitive to cases where one graph has edges the other entirely misses.
    """
    edges_a = {(i, int(j)) for i, row in enumerate(idx_a) for j in row}
    edges_b = {(i, int(j)) for i, row in enumerate(idx_b) for j in row}
    union = edges_a | edges_b
    return len(edges_a & edges_b) / len(union) if union else 1.0


def weighted_overlap_fraction(genes_a: list, genes_b: list) -> float:
    """Position-weighted overlap fraction.

    Genes ranked earlier in *genes_a* contribute more to the score.  Weight
    for rank i (0-indexed) is 1/(i+1), normalised so the maximum score is 1.

    This is more meaningful than unweighted overlap when the greedy selection
    becomes noisy at late ranks: early selections (where the score landscape
    is steep) are weighted more heavily.
    """
    n = len(genes_a)
    weights = [1.0 / (i + 1) for i in range(n)]
    total = sum(weights)
    genes_b_set = set(genes_b)
    score = sum(w for g, w in zip(genes_a, weights) if g in genes_b_set)
    return score / total if total > 0 else 0.0


def spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D arrays."""
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)
