"""
pyGeneBasis — Fast Python reimplementation of geneBasisR.

Public API
----------
I/O
    read_adata                  Read h5ad into memory (backed mode + gene subset + obs merge).
    read_table                  Read a CSV or TSV file into a DataFrame.
    write_csv                   Write a DataFrame to CSV (creates parent dirs).
    write_tsv                   Write a DataFrame to TSV (creates parent dirs).
    write_json                  Write a dict or list to JSON (creates parent dirs).
    prepare_batch_key           Resolve a batch key; creates a joint column from list[str].

Preprocessing
    retain_informative_genes    Filter genes to HVGs before selection.
    highly_variable_methylation_features
                                mCH/mCG HVF, binned by mean x coverage.

Graph construction
    build_knn_graph             Build a kNN graph from a gene subset.

Distances
    calc_minkowski_distances    Score candidate genes by reconstruction error.

Selection
    gene_search                 Greedy iterative gene panel selection (main entry point).
    trim_panel                  Greedy iterative panel trimming (inverse of gene_search).

Evaluation
    get_neighborhood_preservation_scores
    get_gene_prediction_scores
    get_neighs_all_stat
    evaluate_library
    get_panel_celltype_accuracy Hierarchical CT accuracy (constrained + compounded).

Mapping
    get_celltype_mapping
    get_redundancy_stat

Co-expression reliability scoring
    compute_corr_matrices
    score_gene_reliability
    compute_gene_reliability
    plot_metric_distributions
    plot_failure_mode_scatter

Visualization
    plot_mapping_heatmap
    plot_expression_heatmap
    plot_coexpression
    plot_redundancy_stat
    plot_umaps_w_counts

Backend control
    Users control performance vs accuracy via two flags accepted by most functions:

    knn_method : {"approx", "exact"}
        "approx" → PyNNDescent (fast, approximate — default production backend)
        "exact"  → sklearn brute-force (slower, matches geneBasisR most closely)

    batch_method : {"mnn", "harmony"}
        "mnn"     → MNN via scanpy (default — matches geneBasisR)
        "harmony" → harmonypy directly (not scanpy's wrapper)

    GPU support (cuML) is deferred until CPU paths are validated against R.
"""

from .io import read_adata, read_table, write_csv, write_tsv, write_json, prepare_batch_key
from .knn import build_knn_graph
from .panel import (
    retain_informative_genes,
    highly_variable_methylation_features,
    calc_minkowski_distances,
    gene_search,
    trim_panel,
    get_neighborhood_preservation_scores,
    get_gene_prediction_scores,
    get_neighs_all_stat,
    evaluate_library,
    get_panel_celltype_accuracy,
    get_celltype_mapping,
    get_redundancy_stat,
)
from .pl import (
    plot_mapping_heatmap,
    plot_expression_heatmap,
    plot_coexpression,
    plot_redundancy_stat,
    plot_umaps_w_counts,
)
from .reliability import (
    subsample_adata,
    perturb_expression,
    run_perturbation_analysis,
    plot_delta_ct_heatmap,
    plot_perturbation_summary,
    plot_removal_benefit,
    compute_corr_matrices,
    score_gene_reliability,
    compute_gene_reliability,
    plot_metric_distributions,
    plot_failure_mode_scatter,
)

__all__ = [
    "read_adata",
    "read_table",
    "write_csv",
    "write_tsv",
    "write_json",
    "prepare_batch_key",
    "retain_informative_genes",
    "highly_variable_methylation_features",
    "build_knn_graph",
    "calc_minkowski_distances",
    "gene_search",
    "trim_panel",
    "get_neighborhood_preservation_scores",
    "get_gene_prediction_scores",
    "get_neighs_all_stat",
    "evaluate_library",
    "get_panel_celltype_accuracy",
    "get_celltype_mapping",
    "get_redundancy_stat",
    "plot_mapping_heatmap",
    "plot_expression_heatmap",
    "plot_coexpression",
    "plot_redundancy_stat",
    "plot_umaps_w_counts",
    "subsample_adata",
    "perturb_expression",
    "run_perturbation_analysis",
    "plot_delta_ct_heatmap",
    "plot_perturbation_summary",
    "plot_removal_benefit",
    "compute_corr_matrices",
    "score_gene_reliability",
    "compute_gene_reliability",
    "plot_metric_distributions",
    "plot_failure_mode_scatter",
]
