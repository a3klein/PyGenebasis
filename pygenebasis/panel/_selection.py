"""
Greedy iterative gene panel selection.

This module implements the core geneBasis algorithm: at each iteration,
add the candidate gene that most improves neighbourhood preservation.

Algorithm (matches geneBasisR::gene_search)
-------------------------------------------
1. Build the "true graph" kNN using all HVG genes (or their top n_pcs PCs).
2. Pre-compute true-graph Minkowski distances for every gene (one-time cost).
3. If genes_base is empty, bootstrap the first gene by repeating step 4
   five times with different random seeds and taking the modal result.
4. Greedy loop until n_genes genes are selected:
   a. Build "selection graph" from the current panel (no PCA; see note below).
   b. Compute Minkowski distances for all remaining candidates given the
      selection graph (vectorised — see distances.py).
   c. Score each candidate: score = sel_dist - true_dist.
   d. Add the candidate with the highest score; remove from candidate pool.
5. Return the ordered list of selected genes.

PCA note for the selection graph
---------------------------------
The true graph is built with n_pcs PCA components (default 50), matching
geneBasisR's nPC.all.  The selection graph is always built in raw gene space
(n_pcs=None), matching geneBasisR's per-iteration graph which uses the
selected genes directly as features.

Bottleneck and speedup
----------------------
Step 4b is the bottleneck in geneBasisR: it loops over candidates in R.
Here it is a single NumPy broadcast over all candidates simultaneously
(see calc_minkowski_distances in distances.py), giving an O(n_candidates)
speedup.  The selection-graph kNN rebuild in 4a is unavoidable but is
much faster with PyNNDescent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def gene_search(
    adata: AnnData,
    n_genes: int,
    *,
    genes_base: list[str] | None = None,
    genes_discard: list[str] | None = None,
    genes_discard_prefix: list[str] | None = None,
    batch_key: str | None = None,
    knn_method: str = "approx",
    batch_method: str = "per_batch",
    n_neighbors: int = 5,
    n_pcs: int = 50,
    p_minkowski: float = 3.0,
    layer: str | None = None,
    random_state: int = 32,
    verbose: bool = True,
) -> pd.DataFrame:
    """Select an informative gene panel using greedy neighbourhood preservation.

    Replicates geneBasisR::gene_search with vectorised NumPy internals and
    a choice of exact or approximate kNN backends.

    Parameters
    ----------
    adata : AnnData
        Single-cell dataset with logcounts in ``adata.X`` or ``layer``.
    n_genes : int
        Total number of genes to select.
    genes_base : list[str], optional
        Genes to force-include as seeds.  They occupy the first len(genes_base)
        ranks.  Providing at least ~5 genes is recommended for large datasets
        to avoid PCA failures in early iterations.
    genes_discard : list[str], optional
        Genes to exclude from selection.
    genes_discard_prefix : list[str], optional
        Exclude any gene whose name starts with one of these prefixes
        (e.g. ["MT-", "RPL", "RPS"]).
    batch_key : str, optional
        ``adata.obs`` column identifying batches.
    knn_method : {"approx", "exact"}
        kNN backend (see graph.py).
    batch_method : {"mnn", "harmony"}
        Batch correction method (see graph.py).
    n_neighbors : int
        k for kNN graphs.  Default 5 matches geneBasisR.
    n_pcs : int
        PCs for the true graph.  Default 50 matches geneBasisR.
    p_minkowski : float
        Minkowski order.  Default 3 matches geneBasisR.
    layer : str, optional
        AnnData layer holding logcounts.
    random_state : int
        Random seed.  Default 32 matches geneBasisR's set.seed(32).
    verbose : bool
        Print progress after each gene is added.

    Returns
    -------
    pd.DataFrame
        Columns: ``rank`` (1-indexed int), ``gene`` (str).
        Row order matches selection order (genes_base first, then greedy).
    """
    from ..knn._graph import build_knn_graph, _extract_logcounts
    from ._distances import calc_minkowski_distances

    all_genes = adata.var_names.tolist()
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}

    genes_base = [g for g in (genes_base or []) if g in gene_to_idx]

    candidates = _filter_candidates(
        all_genes, genes_base, genes_discard, genes_discard_prefix
    )

    # Dense logcounts for all genes (used for Minkowski scoring)
    X_all = _extract_logcounts(adata, all_genes, layer).astype(np.float64)

    base_indices = [gene_to_idx[g] for g in genes_base]
    cand_indices = [gene_to_idx[g] for g in candidates]

    # Step 1: build true graph from all genes.
    # Always use exact kNN for the true graph: it is computed once and serves
    # as the reference for scoring.  Using approx for the true graph would
    # make true_dists differ between knn_method="exact" and knn_method="approx"
    # runs, causing unnecessary divergence in the greedy selection.
    # The speed benefit of approx is in the selection graph (rebuilt every iteration).
    true_knn_idx, _ = build_knn_graph(
        adata, all_genes,
        batch_key=batch_key, knn_method="exact", batch_method=batch_method,
        n_neighbors=n_neighbors, n_pcs=n_pcs, layer=layer, random_state=random_state,
    )

    # Step 2: pre-compute true-graph Minkowski distances for ALL genes (one-time)
    true_dists = calc_minkowski_distances(X_all, true_knn_idx, p=p_minkowski)

    # Step 3: initialise selected and remaining lists
    selected_indices = list(base_indices)
    remaining = list(cand_indices)

    # Bootstrap first gene if no seeds provided
    if len(selected_indices) == 0:
        first = _bootstrap_first_gene(
            X_all, true_dists, remaining,
            knn_method=knn_method, n_neighbors=n_neighbors, p=p_minkowski,
            random_state=random_state,
        )
        selected_indices.append(first)
        remaining.remove(first)
        if verbose:
            print(f"  [1/{n_genes}] Bootstrap: {all_genes[first]}")

    # Step 4: greedy loop
    while len(selected_indices) < n_genes:
        selected_genes = [all_genes[i] for i in selected_indices]

        # Build selection graph in raw gene space (n_pcs=None)
        sel_knn_idx, _ = build_knn_graph(
            adata, selected_genes,
            batch_key=batch_key, knn_method=knn_method, batch_method=batch_method,
            n_neighbors=n_neighbors, n_pcs=None, layer=layer, random_state=random_state,
        )

        # Score remaining candidates: sel_dist - true_dist
        cand_X = X_all[:, remaining]
        sel_dists = calc_minkowski_distances(cand_X, sel_knn_idx, p=p_minkowski)
        true_dists_cands = true_dists[remaining]
        scores = sel_dists - true_dists_cands

        best_local = int(np.argmax(scores))
        best_gene_idx = remaining[best_local]
        selected_indices.append(best_gene_idx)
        remaining.pop(best_local)

        if verbose:
            n_sel = len(selected_indices)
            print(f"  [{n_sel}/{n_genes}] Added: {all_genes[best_gene_idx]}")

    selected_genes = [all_genes[i] for i in selected_indices]
    return pd.DataFrame({
        "rank": range(1, len(selected_genes) + 1),
        "gene": selected_genes,
    })


def trim_panel(
    adata: AnnData,
    genes_panel: list[str],
    n_remove: int,
    *,
    genes_all: list[str] | None = None,
    genes_protect: list[str] | None = None,
    batch_key: str | None = None,
    n_neighbors: int = 5,
    n_pcs_all: int = 50,
    n_pcs_selection: int = 50,
    knn_method: str = "approx",
    batch_method: str = "per_batch",
    option: str = "approx",
    layer: str | None = None,
    random_state: int = 32,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict:
    """Greedy panel trimming by neighbourhood preservation.

    Iteratively removes the most redundant gene from ``genes_panel``.  At each
    step every remaining gene is evaluated by leave-one-out: the panel without
    that gene is scored by mean cell neighbourhood preservation against the
    true (HVG) graph, and the gene whose removal causes the smallest drop is
    removed first.  This is the inverse of ``gene_search``.

    Parameters
    ----------
    adata : AnnData
        Single-cell dataset with logcounts in ``adata.X`` or ``layer``.
    genes_panel : list[str]
        Starting gene panel to trim.
    n_remove : int
        Number of genes to remove.
    genes_all : list[str], optional
        Reference gene set defining the true kNN graph.  Defaults to all genes
        in ``adata``.  Pass your HVG list here if available.
    genes_protect : list[str], optional
        Genes that must not be removed from the panel.  Protected genes still
        participate in scoring at every step — they are present in the panel
        when all LOO scores are computed — but they are never selected as the
        gene to remove.  Useful for anchoring known marker genes or controls.
    batch_key : str, optional
        ``adata.obs`` column identifying batches.
    n_neighbors : int
        k for kNN graphs.  Default 5.
    n_pcs_all : int
        PCs for the true graph embedding.  Default 50.
    n_pcs_selection : int
        PCs for the selection graph at each LOO step.  Default 50.  Set to
        ``None`` to use raw gene space (faster but less accurate for large
        panels).
    knn_method : {"approx", "exact"}
        kNN backend.  "approx" (PyNNDescent) is strongly recommended for
        datasets > 10k cells given the number of kNN builds required.
    batch_method : {"per_batch"}
        Batch handling strategy.
    option : {"approx", "exact"}
        How to compute ``mean_dist_all`` in the true graph.  "approx" samples
        10% of cells and is recommended for datasets > 50k cells.
    layer : str, optional
    random_state : int
    n_jobs : int
        Parallel workers for LOO evaluation within each step.  -1 uses all
        available cores.
    verbose : bool
        Print progress after each removal.

    Returns
    -------
    dict with keys:
        ``removal_order`` : list[str]
            Genes in removal order, most redundant first.
        ``reduced_panel`` : list[str]
            Remaining genes after ``n_remove`` removals.
        ``score_trajectory`` : list[float]
            Mean cell score at each stage: entry 0 is the full panel, entry i
            is the mean score after removing the i-th gene.
    """
    from joblib import Parallel, delayed
    from sklearn.decomposition import PCA
    from ..evaluation._neighborhood import get_neighs_all_stat, _median_neighbour_dist
    from ..knn._graph import _extract_logcounts, build_knn_graph
    from ..knn._backends import get_knn_backend

    # Validate genes_protect
    protected: set[str] = set()
    if genes_protect is not None:
        panel_set = set(genes_panel)
        not_in_panel = [g for g in genes_protect if g not in panel_set]
        if not_in_panel:
            raise ValueError(
                f"genes_protect contains genes not in genes_panel: {not_in_panel}"
            )
        protected = set(genes_protect)

    n_removable = len(genes_panel) - len(protected)
    if n_remove >= len(genes_panel):
        raise ValueError(
            f"n_remove ({n_remove}) must be less than the panel size "
            f"({len(genes_panel)})."
        )
    if n_remove > n_removable:
        raise ValueError(
            f"n_remove ({n_remove}) exceeds the number of unprotected genes "
            f"({n_removable} = {len(genes_panel)} total − {len(protected)} protected)."
        )

    # Precompute true graph — done once, reused every iteration
    if verbose:
        print("Precomputing true graph...")
    neighs_all_stat = get_neighs_all_stat(
        adata,
        genes_all=genes_all,
        batch_key=batch_key,
        n_neighbors=n_neighbors,
        n_pcs_all=n_pcs_all,
        option=option,
        knn_method=knn_method,
        batch_method=batch_method,
        layer=layer,
        random_state=random_state,
    )
    all_embedding = neighs_all_stat["embedding"]   # (n_cells, n_pcs_all)
    mean_dist_all = neighs_all_stat["mean_dist_all"]
    dist_true     = _median_neighbour_dist(all_embedding, neighs_all_stat["indices"])
    denom         = mean_dist_all - dist_true
    denom         = np.where(np.abs(denom) < 1e-12, np.nan, denom)

    n_cells = len(adata)
    use_pca = n_pcs_selection is not None and n_pcs_selection > 0
    batch_labels = adata.obs[batch_key].values if batch_key is not None else None

    # -----------------------------------------------------------------------
    # Per-step helpers
    # -----------------------------------------------------------------------

    def _build_batch_data(panel: list[str]) -> list[tuple]:
        """Extract logcounts and fit PCA (if requested) once for the current panel.

        Returns a list of (global_mask, X_batch, pca_or_None) tuples — one per
        batch (or one tuple covering all cells when batch_key is None).
        """
        if batch_labels is not None and batch_method == "per_batch":
            batches = np.unique(batch_labels)
        else:
            batches = [None]

        result = []
        for b in batches:
            global_mask = (
                np.where(batch_labels == b)[0] if b is not None
                else np.arange(n_cells)
            )
            X = _extract_logcounts(adata[global_mask], panel, layer).astype(np.float64)
            pca_obj = None
            if use_pca and X.shape[1] > n_pcs_selection:
                n_pcs_actual = min(n_pcs_selection, X.shape[0] - 1, X.shape[1] - 1)
                pca_obj = PCA(n_components=n_pcs_actual, random_state=random_state)
                pca_obj.fit(X)
            result.append((global_mask, X, pca_obj))
        return result

    def _loo_score(remove_col: int, batch_data: list[tuple]) -> float:
        """Score the panel with column ``remove_col`` removed.

        Uses precomputed per-batch PCA components — LOO projection is a fast
        matrix multiply, no SVD.  All P LOO evaluations within a step share
        the same PCA rotation (computed on the full current panel), which is
        an excellent approximation for panels with many genes.
        """
        n_genes = batch_data[0][1].shape[1]
        keep_cols = [i for i in range(n_genes) if i != remove_col]

        all_knn_indices = np.full((n_cells, n_neighbors), -1, dtype=np.int32)
        knn = get_knn_backend(knn_method, random_state=random_state)

        for global_mask, X_batch, pca_obj in batch_data:
            X_loo = X_batch[:, keep_cols]
            if pca_obj is not None:
                # Reproject without gene remove_col using stored rotation.
                # components_ is (n_pcs, n_genes); mean_ is (n_genes,).
                mean_loo = pca_obj.mean_[keep_cols]
                comp_loo = pca_obj.components_[:, keep_cols]
                emb = (X_loo - mean_loo) @ comp_loo.T  # (n_batch, n_pcs)
            else:
                emb = X_loo  # raw gene space

            local_idx, _ = knn.fit_query(emb.astype(np.float32), n_neighbors)
            all_knn_indices[global_mask] = global_mask[local_idx]

        dist_sel = _median_neighbour_dist(all_embedding, all_knn_indices)
        return float(np.nanmean((mean_dist_all - dist_sel) / denom))

    def _full_score(panel: list[str]) -> float:
        """Mean score for the given panel (used for initial and per-step scoring)."""
        sel_indices, _ = build_knn_graph(
            adata, panel,
            batch_key=batch_key, knn_method=knn_method, batch_method=batch_method,
            n_neighbors=n_neighbors, n_pcs=n_pcs_selection, layer=layer,
            random_state=random_state,
        )
        dist_sel = _median_neighbour_dist(all_embedding, sel_indices)
        return float(np.nanmean((mean_dist_all - dist_sel) / denom))

    current_panel = list(genes_panel)
    removal_order: list[str] = []
    score_trajectory: list[float] = [_full_score(current_panel)]

    if verbose:
        print(f"  Full panel ({len(current_panel)} genes): "
              f"mean score = {score_trajectory[0]:.4f}")

    for step in range(n_remove):
        # Fit PCA once on the current panel (one SVD per step instead of P SVDs)
        batch_data = _build_batch_data(current_panel)

        # Parallel LOO: only kNN per gene (no PCA)
        loo_scores: list[float] = Parallel(n_jobs=n_jobs)(
            delayed(_loo_score)(col, batch_data)
            for col in range(len(current_panel))
        )

        # Most redundant = highest score after removal (smallest drop).
        # Protected genes are excluded from selection by setting their LOO
        # scores to -inf so they can never be argmax.
        if protected:
            masked = [
                s if current_panel[i] not in protected else -np.inf
                for i, s in enumerate(loo_scores)
            ]
        else:
            masked = loo_scores
        best_idx = int(np.argmax(masked))
        removed  = current_panel[best_idx]

        removal_order.append(removed)
        current_panel.pop(best_idx)
        score_trajectory.append(loo_scores[best_idx])

        if verbose:
            drop = score_trajectory[-2] - score_trajectory[-1]
            print(f"  [{step + 1}/{n_remove}] Removed: {removed:20s}  "
                  f"score = {loo_scores[best_idx]:.4f}  "
                  f"(drop = {drop:+.4f})")

    return {
        "removal_order": removal_order,
        "reduced_panel": current_panel,
        "score_trajectory": score_trajectory,
    }


def _bootstrap_first_gene(
    logcounts: np.ndarray,
    true_dists: np.ndarray,
    candidates: list[int],
    n_tries: int = 5,
    random_state: int = 32,
    knn_method: str = "approx",
    n_neighbors: int = 5,
    p: float = 3.0,
) -> int:
    """Bootstrap selection of the first gene when no seeds are provided.

    For each of n_tries trials, picks a random candidate as a provisional
    single-gene panel, builds a 1-gene selection graph, scores all candidates,
    and records the best pick.  Returns the gene chosen most frequently
    (modal result).  Matches geneBasisR's bootstrap behaviour.

    Parameters
    ----------
    logcounts : np.ndarray, shape (n_cells, n_all_genes)
    true_dists : np.ndarray, shape (n_all_genes,)
    candidates : list[int]
        Column indices into logcounts of candidate genes.
    n_tries : int
    random_state : int
    knn_method, n_neighbors, p : passed to kNN backend

    Returns
    -------
    gene_idx : int
        Column index (into logcounts) of the selected first gene.
    """
    from ._distances import calc_minkowski_distances
    from ..knn._backends import get_knn_backend

    rng = np.random.default_rng(random_state)
    chosen = []

    for t in range(n_tries):
        seed_col = int(rng.choice(candidates))
        seed_X = logcounts[:, [seed_col]]  # (n_cells, 1)

        knn = get_knn_backend(knn_method, random_state=int(random_state) + t)
        sel_indices, _ = knn.fit_query(seed_X.astype(np.float32), n_neighbors)

        cand_X = logcounts[:, candidates]
        sel_dists = calc_minkowski_distances(cand_X, sel_indices, p=p)
        true_dists_cands = true_dists[candidates]
        scores = sel_dists - true_dists_cands

        best_local = int(np.argmax(scores))
        chosen.append(candidates[best_local])

    # Modal result
    unique, counts = np.unique(chosen, return_counts=True)
    return int(unique[np.argmax(counts)])


def _filter_candidates(
    all_genes: list[str],
    genes_base: list[str],
    genes_discard: list[str] | None,
    genes_discard_prefix: list[str] | None,
) -> list[str]:
    """Return candidate genes excluding seeds, discards, and prefix matches."""
    exclude = set(genes_base)
    if genes_discard:
        exclude.update(genes_discard)
    if genes_discard_prefix:
        for g in all_genes:
            if any(g.startswith(pfx) for pfx in genes_discard_prefix):
                exclude.add(g)
    return [g for g in all_genes if g not in exclude]
