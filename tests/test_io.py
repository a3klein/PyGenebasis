"""
Tests for pygenebasis.io.

All tests are self-contained — no reference data files required.
AnnData fixtures are written to temporary directories by pytest.

Fixture overview
----------------
base_adata (module scope)
    Synthetic AnnData: 50 cells × 20 genes, two obs columns (celltype, batch).
    AnnData is held in memory; ``h5ad_path`` saves it to disk with .raw set.

h5ad_path (module scope)
    Path to the saved h5ad.  Has .raw populated so drop_raw behaviour can be
    tested.

annotation_csv / annotation_tsv (module scope)
    40-row annotation files covering cell_000 … cell_039.  The last 10 cells
    (cell_040 … cell_049) are deliberately unmatched so inner vs. left join
    behaviour can be exercised.

annotation_conflict_csv (module scope)
    40-row annotation file that shares the 'celltype' column with base_adata.obs,
    used to test the overwrite-with-warning behaviour.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import anndata as ad

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pygenebasis.io import read_adata, read_table, write_csv, write_tsv, write_json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_OBS   = 50
N_VARS  = 20
N_ANNOT = 40   # annotation file covers only the first 40 cells

CELL_IDS = [f"cell_{i:03d}" for i in range(N_OBS)]
GENE_IDS = [f"gene_{i:02d}" for i in range(N_VARS)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_adata() -> ad.AnnData:
    """Synthetic AnnData (in memory only — not saved to disk)."""
    rng = np.random.default_rng(0)
    X = scipy.sparse.csr_matrix(
        rng.poisson(1.0, (N_OBS, N_VARS)).astype(np.float32)
    )
    obs = pd.DataFrame(
        {
            "celltype": np.repeat(["TypeA", "TypeB"], N_OBS // 2),
            "batch":    np.tile(["batch1", "batch2"], N_OBS // 2),
        },
        index=CELL_IDS,
    )
    var = pd.DataFrame(index=GENE_IDS)
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture(scope="module")
def h5ad_path(tmp_path_factory, base_adata):
    """Save base_adata to a temporary h5ad file with .raw populated."""
    adata_copy = base_adata.copy()
    adata_copy.raw = adata_copy          # populate .raw so drop_raw can be tested
    path = tmp_path_factory.mktemp("h5ad") / "test.h5ad"
    adata_copy.write_h5ad(path)
    return path


@pytest.fixture(scope="module")
def annotation_csv(tmp_path_factory):
    """40-row CSV annotation (covers first 40 of 50 cells)."""
    annot = pd.DataFrame(
        {
            "label": np.repeat(["LabelA", "LabelB"], N_ANNOT // 2),
            "score": np.arange(N_ANNOT, dtype=float),
        },
        index=CELL_IDS[:N_ANNOT],
    )
    path = tmp_path_factory.mktemp("annot_csv") / "annotation.csv"
    annot.to_csv(path)
    return path, annot


@pytest.fixture(scope="module")
def annotation_tsv(tmp_path_factory):
    """40-row TSV annotation (same cell IDs as annotation_csv)."""
    annot = pd.DataFrame(
        {"label": np.repeat(["LabelA", "LabelB"], N_ANNOT // 2)},
        index=CELL_IDS[:N_ANNOT],
    )
    path = tmp_path_factory.mktemp("annot_tsv") / "annotation.tsv"
    annot.to_csv(path, sep="\t")
    return path, annot


@pytest.fixture(scope="module")
def annotation_conflict_csv(tmp_path_factory):
    """40-row CSV that shares the 'celltype' column with base_adata.obs."""
    annot = pd.DataFrame(
        {"celltype": ["NEW"] * N_ANNOT},
        index=CELL_IDS[:N_ANNOT],
    )
    path = tmp_path_factory.mktemp("annot_conflict") / "conflict.csv"
    annot.to_csv(path)
    return path, annot


# ---------------------------------------------------------------------------
# TestReadAdata
# ---------------------------------------------------------------------------

class TestReadAdata:

    # --- basic loading ---

    def test_shape_matches_source(self, h5ad_path):
        adata = read_adata(h5ad_path)
        assert adata.n_obs == N_OBS
        assert adata.n_vars == N_VARS

    def test_result_is_in_memory(self, h5ad_path):
        adata = read_adata(h5ad_path)
        assert not adata.isbacked

    def test_obs_columns_preserved(self, h5ad_path):
        adata = read_adata(h5ad_path)
        assert "celltype" in adata.obs.columns
        assert "batch" in adata.obs.columns

    # --- drop_raw ---

    def test_drop_raw_true_removes_raw(self, h5ad_path):
        adata = read_adata(h5ad_path, drop_raw=True)
        assert adata.raw is None

    def test_drop_raw_false_keeps_raw(self, h5ad_path):
        adata = read_adata(h5ad_path, drop_raw=False)
        assert adata.raw is not None

    # --- gene subsetting ---

    def test_gene_subset_correct_count(self, h5ad_path):
        adata = read_adata(h5ad_path, genes=GENE_IDS[:5])
        assert adata.n_vars == 5

    def test_gene_subset_correct_names(self, h5ad_path):
        requested = GENE_IDS[:5]
        adata = read_adata(h5ad_path, genes=requested)
        assert list(adata.var_names) == requested

    def test_gene_subset_preserves_request_order(self, h5ad_path):
        # Request genes out of the original order
        requested = [GENE_IDS[9], GENE_IDS[2], GENE_IDS[14]]
        adata = read_adata(h5ad_path, genes=requested)
        assert list(adata.var_names) == requested

    def test_gene_subset_missing_warns(self, h5ad_path):
        requested = GENE_IDS[:3] + ["not_a_gene", "also_missing"]
        with pytest.warns(UserWarning, match="not found in the h5ad file"):
            adata = read_adata(h5ad_path, genes=requested)
        assert adata.n_vars == 3

    def test_gene_subset_all_missing_raises(self, h5ad_path):
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError, match="None of the.*requested genes"):
                read_adata(h5ad_path, genes=["fake_gene_1", "fake_gene_2"])

    def test_gene_subset_empty_list_raises(self, h5ad_path):
        with pytest.raises(ValueError, match="genes list is empty"):
            read_adata(h5ad_path, genes=[])

    # --- obs_file: inner join (default) ---

    def test_inner_join_retains_matched_cells_only(self, h5ad_path, annotation_csv):
        path, annot = annotation_csv
        adata = read_adata(h5ad_path, obs_file=path, obs_join="inner")
        assert adata.n_obs == N_ANNOT

    def test_inner_join_cell_ids_are_subset_of_annotation(self, h5ad_path, annotation_csv):
        path, annot = annotation_csv
        adata = read_adata(h5ad_path, obs_file=path, obs_join="inner")
        assert set(adata.obs_names).issubset(set(annot.index))

    def test_inner_join_annotation_columns_present(self, h5ad_path, annotation_csv):
        path, annot = annotation_csv
        adata = read_adata(h5ad_path, obs_file=path, obs_join="inner")
        for col in annot.columns:
            assert col in adata.obs.columns

    def test_inner_join_tsv_file_works(self, h5ad_path, annotation_tsv):
        path, annot = annotation_tsv
        adata = read_adata(h5ad_path, obs_file=path, obs_join="inner")
        assert adata.n_obs == N_ANNOT
        assert "label" in adata.obs.columns

    # --- obs_file: left join ---

    def test_left_join_keeps_all_cells(self, h5ad_path, annotation_csv):
        path, _ = annotation_csv
        adata = read_adata(h5ad_path, obs_file=path, obs_join="left")
        assert adata.n_obs == N_OBS

    def test_left_join_unmatched_cells_have_nan(self, h5ad_path, annotation_csv):
        path, _ = annotation_csv
        adata = read_adata(h5ad_path, obs_file=path, obs_join="left")
        unmatched_labels = adata.obs.loc[CELL_IDS[N_ANNOT:], "label"]
        assert unmatched_labels.isna().all()

    def test_left_join_matched_cells_have_annotation(self, h5ad_path, annotation_csv):
        path, annot = annotation_csv
        adata = read_adata(h5ad_path, obs_file=path, obs_join="left")
        matched_labels = adata.obs.loc[CELL_IDS[:N_ANNOT], "label"]
        assert not matched_labels.isna().any()

    # --- column conflict ---

    def test_column_conflict_emits_warning(self, h5ad_path, annotation_conflict_csv):
        path, _ = annotation_conflict_csv
        with pytest.warns(UserWarning, match="overwritten"):
            read_adata(h5ad_path, obs_file=path, obs_join="inner")

    def test_column_conflict_annotation_wins(self, h5ad_path, annotation_conflict_csv):
        path, _ = annotation_conflict_csv
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata = read_adata(h5ad_path, obs_file=path, obs_join="inner")
        # annotation file has "NEW" for all cells; original was "TypeA"/"TypeB"
        assert (adata.obs["celltype"] == "NEW").all()

    # --- error cases ---

    def test_no_cell_match_raises(self, h5ad_path, tmp_path_factory):
        annot = pd.DataFrame({"x": [1]}, index=["completely_wrong_id"])
        path = tmp_path_factory.mktemp("nomatch") / "nm.csv"
        annot.to_csv(path)
        with pytest.raises(ValueError, match="No cells.*matched"):
            read_adata(h5ad_path, obs_file=path)

    def test_invalid_obs_join_raises(self, h5ad_path):
        with pytest.raises(ValueError, match="obs_join"):
            read_adata(h5ad_path, obs_join="outer")

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            read_adata("/nonexistent/path/to/file.h5ad")


# ---------------------------------------------------------------------------
# TestReadTable
# ---------------------------------------------------------------------------

class TestReadTable:

    def test_reads_csv(self, tmp_path_factory):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["r0", "r1"])
        path = tmp_path_factory.mktemp("rt") / "data.csv"
        df.to_csv(path)
        result = read_table(path)
        pd.testing.assert_frame_equal(result, df)

    def test_reads_tsv(self, tmp_path_factory):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["r0", "r1"])
        path = tmp_path_factory.mktemp("rt") / "data.tsv"
        df.to_csv(path, sep="\t")
        result = read_table(path)
        pd.testing.assert_frame_equal(result, df)

    def test_default_index_col_zero(self, tmp_path_factory):
        df = pd.DataFrame({"val": [10, 20]}, index=["x", "y"])
        path = tmp_path_factory.mktemp("rt") / "idx.csv"
        df.to_csv(path)
        result = read_table(path)
        assert list(result.index) == ["x", "y"]

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            read_table("/nonexistent/path.csv")


# ---------------------------------------------------------------------------
# TestWriteCsv
# ---------------------------------------------------------------------------

class TestWriteCsv:

    def _df(self) -> pd.DataFrame:
        return pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}, index=["a", "b"])

    def test_roundtrip(self, tmp_path_factory):
        df = self._df()
        path = tmp_path_factory.mktemp("wc") / "out.csv"
        write_csv(df, path)
        result = pd.read_csv(path, index_col=0)
        pd.testing.assert_frame_equal(result, df)

    def test_creates_nested_parent_dirs(self, tmp_path_factory):
        df = self._df()
        path = tmp_path_factory.mktemp("wc") / "nested" / "deep" / "out.csv"
        write_csv(df, path)
        assert path.exists()

    def test_index_false_omits_index(self, tmp_path_factory):
        df = self._df()
        path = tmp_path_factory.mktemp("wc") / "noidx.csv"
        write_csv(df, path, index=False)
        result = pd.read_csv(path)
        assert list(result.columns) == ["x", "y"]


# ---------------------------------------------------------------------------
# TestWriteTsv
# ---------------------------------------------------------------------------

class TestWriteTsv:

    def _df(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2]}, index=["r0", "r1"])

    def test_roundtrip(self, tmp_path_factory):
        df = self._df()
        path = tmp_path_factory.mktemp("wt") / "out.tsv"
        write_tsv(df, path)
        result = pd.read_csv(path, sep="\t", index_col=0)
        pd.testing.assert_frame_equal(result, df)

    def test_file_uses_tab_separator(self, tmp_path_factory):
        df = self._df()
        path = tmp_path_factory.mktemp("wt") / "out.tsv"
        write_tsv(df, path)
        assert "\t" in path.read_text()

    def test_creates_nested_parent_dirs(self, tmp_path_factory):
        df = self._df()
        path = tmp_path_factory.mktemp("wt") / "sub" / "out.tsv"
        write_tsv(df, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# TestWriteJson
# ---------------------------------------------------------------------------

class TestWriteJson:

    def test_roundtrip_dict(self, tmp_path_factory):
        obj = {"key": "value", "n": 42}
        path = tmp_path_factory.mktemp("wj") / "out.json"
        write_json(obj, path)
        assert json.loads(path.read_text()) == obj

    def test_roundtrip_list(self, tmp_path_factory):
        obj = [1, 2, "hello", True]
        path = tmp_path_factory.mktemp("wj") / "out.json"
        write_json(obj, path)
        assert json.loads(path.read_text()) == obj

    def test_indent_applied(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("wj") / "indented.json"
        write_json({"a": 1}, path, indent=4)
        assert "    " in path.read_text()

    def test_creates_nested_parent_dirs(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("wj") / "nested" / "out.json"
        write_json({"x": 1}, path)
        assert path.exists()
