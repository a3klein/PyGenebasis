"""
Smoke tests for the pygenebasis CLI.

Each command is invoked end to end on tiny synthetic data and checked for a
clean exit plus the files it claims to write.  These are deliberately shallow —
they catch call-signature drift between the CLI and the library, which is what
previously left every command broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scanpy as sc
from click.testing import CliRunner

from pygenebasis.cli._main import cli

pytestmark = pytest.mark.slow

GENES = [f"g{i}" for i in range(20)]


def _make_adata(seed: int, n_per: int = 40, n_genes: int = 120, n_marker: int = 10):
    types = ("A", "B", "C")
    rng = np.random.default_rng(seed)
    labels = np.repeat(types, n_per)
    n = n_per * len(types)
    X = rng.poisson(0.5, size=(n, n_genes)).astype(np.float32)
    for t in range(len(types)):
        rows = np.where(labels == types[t])[0]
        for col in range(t, t + n_marker * len(types), len(types)):
            X[np.ix_(rows, [col])] = rng.poisson(6.0, size=(n_per, 1)).astype(np.float32)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"celltype": labels, "donor_id": rng.choice(["d1", "d2"], n)},
            index=[f"c{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


@pytest.fixture(scope="module")
def cli_inputs(tmp_path_factory):
    """Reference, MERFISH, panel and reliability-results files on disk."""
    d = tmp_path_factory.mktemp("cli")
    _make_adata(seed=0).write_h5ad(d / "ref.h5ad")
    _make_adata(seed=1).write_h5ad(d / "merfish.h5ad")

    # read_table uses index_col=0, so both CSVs need a leading index column
    pd.DataFrame({"gene": GENES}).to_csv(d / "panel.csv")
    modes = (["reliable"] * 11 + ["probe_failure"] * 3
             + ["composition_mismatch"] * 3 + ["idiosyncratic_noise"] * 3)
    pd.DataFrame({"failure_mode": modes},
                 index=pd.Index(GENES, name="gene")).to_csv(d / "coexp.csv")
    return d


def _run(args):
    result = CliRunner().invoke(cli, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


class TestPanelCommands:

    def test_search(self, cli_inputs, tmp_path):
        out = tmp_path / "sel.csv"
        _run(["panel", "search", "--adata", str(cli_inputs / "ref.h5ad"),
              "--n-genes", "10", "--output", str(out)])
        df = pd.read_csv(out, index_col=0)
        assert "gene" in df.columns
        assert len(df) == 10

    def test_trim(self, cli_inputs, tmp_path):
        out = tmp_path / "trim.csv"
        _run(["panel", "trim", "--adata", str(cli_inputs / "ref.h5ad"),
              "--panel", str(cli_inputs / "panel.csv"),
              "--n-remove", "3", "--n-jobs", "1", "--output", str(out)])
        df = pd.read_csv(out, index_col=0)
        assert len(df) == len(GENES) - 3

    def test_evaluate_writes_sibling_tables(self, cli_inputs, tmp_path):
        out = tmp_path / "eval.csv"
        _run(["panel", "evaluate", "--adata", str(cli_inputs / "ref.h5ad"),
              "--panel", str(cli_inputs / "panel.csv"),
              "--celltype-key", "celltype", "--output", str(out)])
        assert out.exists()
        assert (tmp_path / "eval_gene.csv").exists()
        assert (tmp_path / "eval_celltype.csv").exists()


class TestReliabilityCommands:

    def test_score_coexp(self, cli_inputs, tmp_path):
        out = tmp_path / "scores.csv"
        _run(["reliability", "score-coexp",
              "--merfish", str(cli_inputs / "merfish.h5ad"),
              "--ref", str(cli_inputs / "ref.h5ad"),
              "--panel", str(cli_inputs / "panel.csv"),
              "--output", str(out)])
        df = pd.read_csv(out, index_col=0)
        assert "failure_mode" in df.columns
        assert len(df) == len(GENES)

    def test_perturb(self, cli_inputs, tmp_path):
        out_dir = tmp_path / "perturb"
        _run(["reliability", "perturb", "--adata", str(cli_inputs / "ref.h5ad"),
              "--results", str(cli_inputs / "coexp.csv"),
              "--level-keys", "celltype",
              "--n-replicates", "2", "--n-cells-per-group", "30",
              "--out-dir", str(out_dir)])
        written = {p.name for p in out_dir.glob("*.csv")}
        assert {"delta_ct.csv", "baseline_ct.csv", "summary.csv"} <= written
