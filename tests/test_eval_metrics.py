"""
Tests for the panel-evaluation metrics: coverage, clustering agreement,
classifier gap, and the per-label summary.

Each fixture plants a known answer — markers per type, a known partition, a
known separable feature set — so the assertions are about correctness rather
than just "it ran".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scanpy as sc

from pygenebasis import (
    agreement_crosstab,
    classifier_gap,
    cluster_agreement,
    coverage_by_label,
    marker_detection,
)
from pygenebasis.evaluation._report import per_label_summary

N_MARKER = 8
TYPES = ("A", "B", "C", "D")


@pytest.fixture(scope="module")
def marked_adata():
    """4 types, N_MARKER planted markers each, the rest noise."""
    rng = np.random.default_rng(0)
    n_per, n_genes = 70, 200
    labels = np.repeat(TYPES, n_per)
    X = rng.poisson(0.3, size=(len(labels), n_genes)).astype(np.float32)
    for t in range(len(TYPES)):
        rows = np.where(labels == TYPES[t])[0]
        for col in range(t, t + N_MARKER * len(TYPES), len(TYPES)):
            X[np.ix_(rows, [col])] = rng.poisson(8.0, size=(n_per, 1))
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"ct": labels, "donor": rng.choice(["d1", "d2"], len(labels))},
                         index=[f"c{i}" for i in range(len(labels))]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


@pytest.fixture(scope="module")
def panel(marked_adata):
    # the first 32 genes are exactly the planted markers
    return list(marked_adata.var_names[:N_MARKER * len(TYPES)])


class TestMarkerDetection:

    def test_recovers_the_planted_markers(self, marked_adata, panel):
        det = marker_detection(marked_adata, panel, "ct")
        assert len(det) == len(panel) * len(TYPES)
        per_type = det[det["usable"]].groupby("label").size()
        assert (per_type == N_MARKER).all(), per_type.to_dict()

    def test_fold_is_enrichment_over_the_rest(self, marked_adata, panel):
        det = marker_detection(marked_adata, panel, "ct").set_index(["gene", "label"])
        # g0 was planted in type A only
        assert det.loc[("g0", "A"), "fold"] > 2
        assert det.loc[("g0", "B"), "fold"] < 1

    def test_thresholds_are_recorded_and_bite(self, marked_adata, panel):
        strict = marker_detection(marked_adata, panel, "ct", min_fold=1e6)
        assert strict.attrs["min_fold"] == 1e6
        assert not strict["usable"].any()

    def test_missing_label_key_raises(self, marked_adata, panel):
        with pytest.raises(KeyError):
            marker_detection(marked_adata, panel, "not_a_column")

    def test_coverage_by_label_counts_and_sources(self, marked_adata, panel):
        det = marker_detection(marked_adata, panel, "ct")
        src = {g: ("lit;geneBasis" if i % 4 == 0 else "geneBasis")
               for i, g in enumerate(panel)}
        cov = coverage_by_label(det, sources=src)
        assert (cov["n_usable_markers"] == N_MARKER).all()
        # multi-source genes get their own combined column, so counts still sum
        src_cols = [c for c in cov.columns if c.startswith("n_from_")]
        assert cov[src_cols].sum(axis=1).equals(cov["n_usable_markers"])


class TestClusterAgreement:

    def test_identical_partitions_score_one(self):
        truth = np.repeat(list("ABCD"), 40)
        renamed = np.repeat(list("wxyz"), 40)          # names are irrelevant
        a = cluster_agreement(truth, renamed)
        assert a["ARI"] == pytest.approx(1.0)
        assert a["AMI"] == pytest.approx(1.0)

    def test_random_partition_scores_near_zero(self):
        truth = np.repeat(list("ABCD"), 60)
        rand = np.random.default_rng(0).permutation(truth)
        a = cluster_agreement(truth, rand)
        assert abs(a["ARI"]) < 0.05 and abs(a["AMI"]) < 0.05

    def test_splitting_hurts_ari_more_than_ami(self):
        """The load-bearing difference between the two."""
        truth = np.repeat(list("ABCD"), 60)
        split = np.array([f"{t}{i % 2}" for i, t in enumerate(truth)])
        a = cluster_agreement(truth, split)
        assert a["n_pred_levels"] == 8
        assert a["ARI"] < a["AMI"] - 0.1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            cluster_agreement(np.arange(10), np.arange(9))

    def test_crosstab_ordering_puts_fragments_together(self):
        truth = np.repeat(list("ABCD"), 60)
        split = np.array([f"{t}{i % 2}" for i, t in enumerate(truth)])
        ct = agreement_crosstab(truth, split)
        # each row's two non-zero columns must be adjacent
        for _, row in ct.iterrows():
            nz = np.flatnonzero(row.to_numpy() > 0)
            assert nz.max() - nz.min() == len(nz) - 1
        assert np.allclose(ct.sum(axis=1), 1.0)

    def test_unordered_keeps_alphabetical(self):
        truth = np.repeat(list("AB"), 20)
        pred = np.repeat(list("yx"), 20)
        ct = agreement_crosstab(truth, pred, order=False)
        assert list(ct.columns) == ["x", "y"]


class TestClassifierGap:

    def test_panel_beats_noise_padded_full_set(self, marked_adata, panel):
        """The informative genes are exactly the panel; the rest is noise."""
        res = classifier_gap(marked_adata, panel, "ct", random_state=0)
        s = res["summary"].set_index("feature_set")
        assert s.loc["panel", "n_features"] == len(panel)
        assert s.loc["full", "n_features"] == marked_adata.n_vars
        assert s.loc["panel", "macro_F1"] >= s.loc["full", "macro_F1"]

    def test_returns_every_piece(self, marked_adata, panel):
        res = classifier_gap(marked_adata, panel, "ct", random_state=0)
        assert set(res) == {"summary", "per_label", "gene_importance",
                            "confusion", "dropped_labels"}
        assert set(res["per_label"]["feature_set"]) == {"panel", "full"}
        assert list(res["gene_importance"]["gene"]) != []
        np.testing.assert_allclose(res["confusion"].sum(axis=1), 1.0)

    def test_rare_labels_are_dropped_not_crashed_on(self, marked_adata, panel):
        a = marked_adata.copy()
        ct = a.obs["ct"].astype(str).to_numpy()
        ct[:3] = "TOOFEW"
        a.obs["ct"] = ct
        res = classifier_gap(a, panel, "ct", min_cells_per_label=10, random_state=0)
        assert res["dropped_labels"] == ["TOOFEW"]
        assert "TOOFEW" not in set(res["per_label"]["celltype"])

    def test_subsample_cap_respected(self, marked_adata, panel):
        res = classifier_gap(marked_adata, panel, "ct", n_cells=100, random_state=0)
        assert res["summary"]["n_cells"].max() == 100


class TestPerLabelSummary:

    def test_labels_alone_is_enough(self, marked_adata):
        tab = per_label_summary(marked_adata.obs["ct"])
        assert set(tab.index) == set(TYPES)
        assert (tab["n_cells"] == 70).all()
        assert "diagnosis" in tab.columns

    def test_joins_what_it_is_given(self, marked_adata, panel):
        det = marker_detection(marked_adata, panel, "ct")
        tab = per_label_summary(
            marked_adata.obs["ct"],
            preservation=np.full(marked_adata.n_obs, 0.95),
            coverage=coverage_by_label(det),
        )
        assert tab["preservation_mean"].eq(0.95).all()
        assert (tab["n_usable_markers"] == N_MARKER).all()

    def test_diagnosis_flags_a_thin_badly_preserved_type(self, marked_adata):
        pres = np.where(marked_adata.obs["ct"].to_numpy() == "A", 0.2, 0.99)
        cov = pd.DataFrame({"n_usable_markers": [1, 99, 99, 99]},
                           index=pd.Index(list(TYPES), name="label"))
        tab = per_label_summary(marked_adata.obs["ct"], preservation=pres,
                                coverage=cov)
        assert tab.loc["A", "diagnosis"] == "marker shortfall — add markers"
        assert (tab.drop(index="A")["diagnosis"] == "ok").all()

    def test_thresholds_are_parameters(self, marked_adata):
        cov = pd.DataFrame({"n_usable_markers": [5, 5, 5, 5]},
                           index=pd.Index(list(TYPES), name="label"))
        lax = per_label_summary(marked_adata.obs["ct"], coverage=cov,
                                marker_floor=1)
        assert (lax["diagnosis"] == "ok").all()
        assert lax.attrs["marker_floor"] == 1


class TestPanelReport:
    """The orchestrator. Steps are skipped where they only cost time."""

    FAST = ("clustering", "classifier", "gene_scores")

    def test_returns_every_table(self, marked_adata, panel):
        from pygenebasis import panel_report
        rep = panel_report(marked_adata, panel, "ct", skip=self.FAST, verbose=False)
        assert set(rep) == {"summary", "per_label", "per_cell", "per_gene",
                            "per_gene_label", "crosstab", "agreement",
                            "classifier_confusion", "gene_importance"}
        assert len(rep["summary"]) == 1
        assert len(rep["per_label"]) == len(TYPES)
        assert len(rep["per_cell"]) == marked_adata.n_obs
        assert "diagnosis" in rep["per_label"].columns

    def test_skipped_steps_are_none(self, marked_adata, panel):
        from pygenebasis import panel_report
        rep = panel_report(marked_adata, panel, "ct", skip=self.FAST, verbose=False)
        assert rep["agreement"] is None
        assert rep["crosstab"] is None
        assert rep["classifier_confusion"] is None
        # and their summary columns are absent rather than NaN
        assert "ARI" not in rep["summary"].columns
        assert "clf_macroF1_panel" not in rep["summary"].columns

    def test_unknown_step_rejected(self, marked_adata, panel):
        from pygenebasis import panel_report
        with pytest.raises(ValueError, match="unknown step"):
            panel_report(marked_adata, panel, "ct", skip=("nope",), verbose=False)

    def test_precomputed_clustering_is_used(self, marked_adata, panel):
        """cluster_key avoids re-clustering, which is the expensive step."""
        from pygenebasis import panel_report
        a = marked_adata.copy()
        a.obs["mine"] = a.obs["ct"].astype(str)          # a perfect clustering
        rep = panel_report(a, panel, "ct", cluster_key="mine",
                           skip=("classifier", "gene_scores"), verbose=False)
        assert rep["agreement"]["ARI"] == pytest.approx(1.0)
        assert rep["summary"]["ARI"].iloc[0] == pytest.approx(1.0)

    def test_missing_label_key_raises(self, marked_adata, panel):
        from pygenebasis import panel_report
        with pytest.raises(KeyError):
            panel_report(marked_adata, panel, "nope", skip=self.FAST, verbose=False)

    def test_no_panel_genes_present_raises(self, marked_adata):
        from pygenebasis import panel_report
        with pytest.raises(ValueError, match="none of the panel genes"):
            panel_report(marked_adata, ["nosuchgene"], "ct", skip=self.FAST,
                         verbose=False)

    def test_per_gene_marks_panel_membership(self, marked_adata, panel):
        from pygenebasis import panel_report
        rep = panel_report(marked_adata, panel, "ct", skip=self.FAST, verbose=False)
        pg = rep["per_gene"].set_index("gene")
        assert pg["on_panel"].sum() == len(panel)
        assert pg.loc[panel[0], "n_labels_marked"] >= 1
