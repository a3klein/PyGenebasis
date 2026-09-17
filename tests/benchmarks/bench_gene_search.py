"""
Runtime benchmarks for gene_search across kNN methods and dataset sizes.

Run with:
    pixi run -e py-genebasis python tests/benchmarks/bench_gene_search.py

Compares:
  - R geneBasisR (baseline, read from a pre-timed CSV if available)
  - pyGeneBasis exact kNN   (sklearn)
  - pyGeneBasis approx kNN  (PyNNDescent)

Results are printed to stdout and saved to benchmarks/results.csv.

The benchmark generates synthetic datasets at increasing cell counts
(1k, 5k, 10k, 50k, 100k) with a fixed gene count of 3000 and selects
a 50-gene panel seeded with 5 genes.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse

RESULTS_PATH = Path(__file__).parent / "results.csv"

N_GENES_PANEL = 50
N_GENES_TOTAL = 3_000
N_NEIGHBORS   = 5
N_PCS         = 50
SEED_GENES    = [f"gene_{i}" for i in range(5)]
CELL_COUNTS   = [1_000, 5_000, 10_000, 50_000, 100_000]


def make_synthetic_adata(n_cells: int, n_genes: int, seed: int = 42) -> ad.AnnData:
    """Generate a random logcounts AnnData for benchmarking."""
    rng = np.random.default_rng(seed)
    X = rng.negative_binomial(n=5, p=0.5, size=(n_cells, n_genes)).astype(np.float32)
    X = np.log1p(X)
    obs = pd.DataFrame(
        {"batch": rng.choice(["b1", "b2", "b3"], size=n_cells)},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return ad.AnnData(X=scipy.sparse.csr_matrix(X), obs=obs, var=var)


def time_gene_search(adata, knn_method: str) -> float | None:
    """Run gene_search and return wall-clock seconds, or None on error."""
    try:
        from pygenebasis import gene_search
        t0 = time.perf_counter()
        gene_search(
            adata,
            n_genes=N_GENES_PANEL,
            genes_base=SEED_GENES,
            batch_key="batch",
            batch_method="mnn",
            n_neighbors=N_NEIGHBORS,
            n_pcs=N_PCS,
            knn_method=knn_method,
            random_state=32,
            verbose=False,
        )
        return time.perf_counter() - t0
    except NotImplementedError:
        print(f"    [SKIP] gene_search not yet implemented (knn_method={knn_method!r})")
        return None
    except Exception:
        traceback.print_exc()
        return None


def main():
    rows = []

    print(f"\n{'Cells':>8}  {'Exact (s)':>12}  {'Approx (s)':>12}  {'Speedup':>10}")
    print("-" * 50)

    for n_cells in CELL_COUNTS:
        adata = make_synthetic_adata(n_cells, N_GENES_TOTAL)
        print(f"\n{n_cells:>8,}  ", end="", flush=True)

        t_exact  = time_gene_search(adata, "exact")
        t_approx = time_gene_search(adata, "approx")

        exact_str  = f"{t_exact:.2f}s"  if t_exact  is not None else "N/A"
        approx_str = f"{t_approx:.2f}s" if t_approx is not None else "N/A"
        speedup    = (
            f"{t_exact / t_approx:.1f}×"
            if (t_exact is not None and t_approx is not None and t_approx > 0)
            else "N/A"
        )

        print(f"{exact_str:>12}  {approx_str:>12}  {speedup:>10}")

        rows.append({
            "n_cells":       n_cells,
            "n_genes":       N_GENES_TOTAL,
            "n_genes_panel": N_GENES_PANEL,
            "knn_exact_s":   t_exact,
            "knn_approx_s":  t_approx,
        })

    results = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
