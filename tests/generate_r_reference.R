# generate_r_reference.R
#
# Generates reference outputs from geneBasisR for both datasets.
# These files are used by pyGeneBasis pytest tests to verify numerical correctness.
#
# Run from the gene_panel project root:
#   pixi run -e r-genebasis Rscript software/geneBasis/pyGeneBasis/tests/generate_r_reference.R
#
# Output directories:
#   software/geneBasis/pyGeneBasis/tests/reference_data/mouse_embryo/
#   software/geneBasis/pyGeneBasis/tests/reference_data/bg_sn/

library(geneBasisR)
library(SingleCellExperiment)
library(Matrix)
library(BiocNeighbors)
library(irlba)

# =============================================================================
# Paths
# =============================================================================

ROOT       <- getwd()   # must be run from gene_panel project root
REF_DIR    <- file.path(ROOT, "software/geneBasis/pyGeneBasis/tests/reference_data")
BG_MTX_DIR <- file.path(ROOT, "software/geneBasis/genebasis_testing/test_data/BG_SN_only_mtx_sub")

# =============================================================================
# Shared parameters — must match what pyGeneBasis uses in tests
# =============================================================================

N_NEIGH      <- 5      # k for kNN graphs
NPC_ALL      <- 50     # PCs for true graph
P_MINKOWSKI  <- 3      # Minkowski distance order
N_GENES      <- 50     # panel size to select

SEED_GENES_MOUSE <- c("Hba-x", "Acta2", "Ttr", "Crabp1", "Hoxaas3")

SEED_GENES_BG_SN <- c("RELN", "TH", "MOBP", "APOE", "AIF1", "SNAP25", "SLC17A6", "PDGFRA")

# =============================================================================
# Helper: export a kNN graph with both integer indices and distances
# =============================================================================
# Uses geneBasisR:::.get_mapping internally (same function and seed state as
# get_neighborhood_preservation_scores) so the exported indices are identical
# to those used in R's cell score computation.
# .get_mapping returns a cell-name matrix; we convert to 0-indexed integers
# and recompute distances from the same embedding for Python.

.export_knn <- function(sce, genes, batch = NULL, n_neigh = N_NEIGH, npc = NPC_ALL, out_dir, prefix) {
  cell_order <- colnames(sce)

  if (!is.null(batch)) {
    meta      <- as.data.frame(colData(sce))
    batch_ids <- factor(meta[[batch]])
    all_indices   <- matrix(NA_integer_, nrow = ncol(sce), ncol = n_neigh,
                            dimnames = list(cell_order, NULL))
    all_distances <- matrix(NA_real_,    nrow = ncol(sce), ncol = n_neigh,
                            dimnames = list(cell_order, NULL))
    for (b in levels(batch_ids)) {
      idx     <- which(batch_ids == b)
      sce_b   <- sce[, idx]
      sub     <- .export_knn_single(sce_b, genes, n_neigh, npc)
      all_indices[colnames(sce_b),   ] <- sub$indices
      all_distances[colnames(sce_b), ] <- sub$distances
    }
  } else {
    sub           <- .export_knn_single(sce, genes, n_neigh, npc)
    all_indices   <- sub$indices
    all_distances <- sub$distances
    rownames(all_indices)   <- cell_order
    rownames(all_distances) <- cell_order
  }

  write.csv(all_indices,   file.path(out_dir, paste0(prefix, "_knn_indices.csv")))
  write.csv(all_distances, file.path(out_dir, paste0(prefix, "_knn_distances.csv")))
  invisible(list(indices = all_indices, distances = all_distances))
}

.export_knn_single <- function(sce, genes, n_neigh, npc) {
  # Use geneBasisR's internal .get_mapping (same function called by
  # get_neighborhood_preservation_scores) so the neighbour assignments are
  # identical to those used in cell score computation.
  name_mat <- geneBasisR:::.get_mapping(sce, genes = genes, batch = NULL,
                                        n.neigh = n_neigh, nPC = npc)
  cells    <- colnames(sce)
  name_to_idx <- setNames(seq_along(cells) - 1L, cells)  # 0-indexed local

  # Convert cell-name matrix → 0-indexed integer matrix
  indices <- apply(name_mat, c(1, 2), function(nm) name_to_idx[[nm]])

  # Recompute distances from the same embedding for Python's use
  set.seed(32)
  counts <- t(as.matrix(logcounts(sce[genes, ])))
  if (!is.null(npc)) {
    n <- min(npc, nrow(counts) - 1, ncol(counts) - 1)
    if (n > 2) {
      pcs    <- suppressWarnings(prcomp_irlba(counts, n = n))
      counts <- pcs$x
    }
  }
  rownames(counts) <- cells
  distances <- t(sapply(seq_along(cells), function(i) {
    neigh_names <- name_mat[i, ]
    sqrt(rowSums((counts[rep(i, n_neigh), , drop = FALSE] -
                    counts[neigh_names,  , drop = FALSE])^2))
  }))

  list(indices = indices, distances = distances)
}

# =============================================================================
# Helper: run and save all outputs for one dataset
# =============================================================================

run_and_save <- function(sce, out_dir, batch_col, celltype_col, seed_genes, n_genes = N_GENES) {
  dir.create(file.path(out_dir, "r_outputs"), showWarnings = FALSE, recursive = TRUE)
  rout <- file.path(out_dir, "r_outputs")

  # ---- 1. Raw data export (for Python to load) -------------------------------
  cat("  Exporting raw data...\n")
  writeMM(logcounts(sce), file.path(out_dir, "counts.mtx"))
  write.csv(as.data.frame(colData(sce)),
            file.path(out_dir, "obs.csv"))
  write.csv(data.frame(gene = rownames(sce)),
            file.path(out_dir, "var.csv"), row.names = FALSE)

  # ---- 2. retain_informative_genes -------------------------------------------
  cat("  retain_informative_genes...\n")
  sce_hvg   <- retain_informative_genes(sce)
  genes_hvg <- rownames(sce_hvg)
  write.csv(data.frame(gene = genes_hvg),
            file.path(rout, "hvg_genes.csv"), row.names = FALSE)

  # Use HVG-filtered SCE for all downstream steps (matches intended workflow)
  genes_all <- rownames(sce)   # full set — used as reference for evaluation
  sce       <- sce_hvg

  # ---- 3. True kNN graph (all HVG genes) -------------------------------------
  cat("  Building true kNN graph...\n")
  .export_knn(sce, rownames(sce), batch = batch_col, npc = NPC_ALL,
              out_dir = rout, prefix = "true_graph")

  # ---- 4. calc_Minkowski_distances on the true graph -------------------------
  cat("  calc_Minkowski_distances (true graph)...\n")
  minkowski_true <- calc_Minkowski_distances(
    sce,
    genes         = rownames(sce),
    batch         = batch_col,
    n.neigh       = N_NEIGH,
    nPC           = NPC_ALL,
    genes.predict = rownames(sce),
    p.minkowski   = P_MINKOWSKI
  )
  write.csv(minkowski_true, file.path(rout, "minkowski_true_graph.csv"), row.names = FALSE)

  # ---- 5. gene_search --------------------------------------------------------
  cat("  gene_search (n =", n_genes, ")...\n")
  genes_stat <- gene_search(
    sce,
    genes_base           = seed_genes,
    n_genes_total        = n_genes,
    batch                = batch_col,
    n.neigh              = N_NEIGH,
    nPC.all              = NPC_ALL,
    p.minkowski          = P_MINKOWSKI,
    genes.discard_prefix = c("MT-", "RPL", "RPS"),
    verbose              = TRUE
  )
  write.csv(genes_stat, file.path(rout, "gene_search_results.csv"), row.names = FALSE)
  genes_selected <- genes_stat$gene

  # ---- 6. Selection kNN graph (for comparison with pyGeneBasis) --------------
  cat("  Building selection kNN graph...\n")
  .export_knn(sce, genes_selected, batch = batch_col, npc = NULL,
              out_dir = rout, prefix = "selection_graph")

  # ---- 7. calc_Minkowski_distances on the selection graph --------------------
  cat("  calc_Minkowski_distances (selection graph)...\n")
  minkowski_sel <- calc_Minkowski_distances(
    sce,
    genes         = genes_selected,
    batch         = batch_col,
    n.neigh       = N_NEIGH,
    nPC           = NULL,
    genes.predict = rownames(sce),
    p.minkowski   = P_MINKOWSKI
  )
  write.csv(minkowski_sel, file.path(rout, "minkowski_selection_graph.csv"), row.names = FALSE)

  # ---- 8. get_neighborhood_preservation_scores -------------------------------
  cat("  get_neighborhood_preservation_scores...\n")
  cell_scores <- get_neighborhood_preservation_scores(
    sce,
    genes.all       = rownames(sce),
    genes.selection = genes_selected,
    batch           = batch_col,
    n.neigh         = N_NEIGH,
    nPC.all         = NPC_ALL,
    nPC.selection   = NULL,
    option          = "exact"
  )
  write.csv(cell_scores, file.path(rout, "cell_scores.csv"), row.names = FALSE)

  # ---- 8b. Cell score intermediates (for Python debugging) -------------------
  # Exports mean_dist, dist_true, dist_sel per cell — the three quantities
  # that go into the cell score formula — so Python can compare them directly.
  # Also exports the HVG PCA coordinates used for distance measurement.
  cat("  Exporting cell score intermediates...\n")
  meta        <- as.data.frame(colData(sce))
  batchFactor <- factor(meta[[batch_col]])

  # Pre-compute neighs.all_stat (same call as inside get_neighborhood_preservation_scores)
  neighs.all_stat_all <- get_neighs_all_stat(
    sce, genes.all = rownames(sce), batch = batch_col,
    n.neigh = N_NEIGH, nPC.all = NPC_ALL, option = "exact"
  )

  intermediates <- lapply(levels(batchFactor), function(b) {
    idx          <- which(batchFactor == b)
    current.sce  <- sce[, idx]
    stat         <- neighs.all_stat_all[[b]]

    counts       <- stat$counts        # HVG PCA for this batch
    neighs.all   <- stat$neighs.all    # cell-name matrix, true kNN
    mean_dist    <- stat$mean_dist     # per-cell mean distance

    # Selection kNN — identical call to what get_neighborhood_preservation_scores uses
    neighs.compare <- geneBasisR:::.get_mapping(
      current.sce, genes = genes_selected, batch = NULL,
      n.neigh = N_NEIGH, nPC = NULL
    )

    # Sort everything alphabetically (matches internal R ordering)
    counts         <- counts[order(rownames(counts)), ]
    neighs.all     <- neighs.all[order(rownames(neighs.all)), ]
    neighs.compare <- neighs.compare[order(rownames(neighs.compare)), ]
    mean_dist      <- mean_dist[order(names(mean_dist))]

    rows <- lapply(seq_len(nrow(counts)), function(i) {
      dt <- median(Rfast::dista(t(counts[i, ]), counts[neighs.all[i, ],    ]))
      ds <- median(Rfast::dista(t(counts[i, ]), counts[neighs.compare[i, ],]))
      md <- mean_dist[i]
      data.frame(
        cell       = rownames(counts)[i],
        mean_dist  = md,
        dist_true  = dt,
        dist_sel   = ds,
        cell_score = (md - ds) / (md - dt)
      )
    })
    do.call(rbind, rows)
  })
  intermediates <- do.call(rbind, intermediates)
  write.csv(intermediates, file.path(rout, "cell_score_intermediates.csv"), row.names = FALSE)

  # Also export the raw HVG PCA coordinates per batch so Python can compare
  # its own embedding values cell-by-cell.
  cat("  Exporting HVG PCA coordinates...\n")
  for (b in levels(batchFactor)) {
    pca_mat <- neighs.all_stat_all[[b]]$counts
    write.csv(as.data.frame(pca_mat),
              file.path(rout, paste0("hvg_pca_batch_", b, ".csv")))
  }

  # ---- 9. get_gene_prediction_scores -----------------------------------------
  cat("  get_gene_prediction_scores...\n")
  gene_scores <- get_gene_prediction_scores(
    sce,
    genes.selection = genes_selected,
    genes.all       = rownames(sce),
    batch           = batch_col,
    n.neigh         = N_NEIGH,
    nPC.all         = NPC_ALL,
    nPC.selection   = NULL,
    method          = "spearman"
  )
  write.csv(gene_scores, file.path(rout, "gene_scores.csv"), row.names = FALSE)

  # ---- 10. get_celltype_mapping ----------------------------------------------
  cat("  get_celltype_mapping...\n")
  ct_mapping <- get_celltype_mapping(
    sce,
    genes.selection = genes_selected,
    celltype.id     = celltype_col,
    batch           = batch_col,
    n.neigh         = N_NEIGH,
    return.stat     = TRUE
  )
  write.csv(ct_mapping$mapping, file.path(rout, "celltype_mapping.csv"), row.names = FALSE)
  write.csv(ct_mapping$stat,    file.path(rout, "celltype_stat.csv"),    row.names = FALSE)

  # ---- 11. get_redundancy_stat -----------------------------------------------
  cat("  get_redundancy_stat...\n")
  redundancy <- get_redundancy_stat(
    sce,
    genes           = genes_selected,
    genes_to_assess = genes_selected,
    batch           = batch_col,
    celltype.id     = celltype_col
  )
  write.csv(redundancy, file.path(rout, "redundancy_stat.csv"), row.names = FALSE)

  cat("  Done. Outputs written to:", rout, "\n\n")
}

# =============================================================================
# Dataset 1: Mouse embryo E8.5
# =============================================================================

cat("=== Dataset 1: Mouse Embryo E8.5 ===\n")
data("sce_mouseEmbryo", package = "geneBasisR")

run_and_save(
  sce          = sce_mouseEmbryo,
  out_dir      = file.path(REF_DIR, "mouse_embryo"),
  batch_col    = "sample",
  celltype_col = "celltype",
  seed_genes   = SEED_GENES_MOUSE,
  n_genes      = N_GENES
)

# =============================================================================
# Dataset 2: BG Substantia Nigra
# =============================================================================

cat("=== Dataset 2: BG Substantia Nigra ===\n")

counts   <- readMM(file.path(BG_MTX_DIR, "matrix.mtx"))
genes    <- readLines(file.path(BG_MTX_DIR, "genes.tsv"))
barcodes <- readLines(file.path(BG_MTX_DIR, "barcodes.tsv"))
metadata <- read.csv(file.path(BG_MTX_DIR, "obs_metadata.csv"), row.names = 1)

rownames(counts) <- genes
colnames(counts) <- barcodes

sce_bg_sn <- SingleCellExperiment(
  assays  = list(logcounts = counts),
  colData = metadata
)

run_and_save(
  sce          = sce_bg_sn,
  out_dir      = file.path(REF_DIR, "bg_sn"),
  batch_col    = "donor_id",
  celltype_col = "Group",
  seed_genes   = SEED_GENES_BG_SN,
  n_genes      = N_GENES
)

cat("All reference outputs generated.\n")
