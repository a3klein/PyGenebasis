"""
File I/O for pyGeneBasis.

All file reading and writing in the package should go through this module so
that behaviour is consistent: backed-mode loading, gene subsetting, annotation
merging, and safe directory creation are all handled here.

Functions
---------
read_adata
    Read an h5ad file into memory, optionally subsetting to a gene list and
    merging an external cell-level annotation table into ``adata.obs``.
read_table
    Read a CSV or TSV file into a DataFrame.  Separator is inferred from the
    file extension; ``index_col=0`` by default.
write_csv
    Write a DataFrame to a CSV file (creates parent directories as needed).
write_tsv
    Write a DataFrame to a TSV file (creates parent directories as needed).
write_json
    Write a dict or list to a JSON file (creates parent directories as needed).
prepare_batch_key
    Resolve a batch key, creating a joint column from a list of keys if needed.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import anndata as ad
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sep_from_path(path: Path) -> str | None:
    """Return the CSV separator inferred from the file extension.

    Returns ``None`` for unrecognised extensions, signalling that ``read_table``
    should fall back to the pandas CSV sniffer.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix in (".tsv", ".txt"):
        return "\t"
    return None


def _merge_obs_file(
    adata: ad.AnnData,
    obs_file: str | os.PathLike,
    *,
    obs_join: str,
) -> ad.AnnData:
    """Merge an external annotation table into ``adata.obs``.

    Called by :func:`read_adata` after the AnnData is in memory.
    """
    annot = read_table(obs_file)

    # Report how many cells have a match
    n_total = adata.n_obs
    n_matched = adata.obs_names.isin(annot.index).sum()
    print(
        f"[read_adata] annotation file: {n_matched}/{n_total} cells matched"
        + (f" ({n_total - n_matched} unmatched)" if n_total - n_matched else "")
    )

    if n_matched == 0:
        raise ValueError(
            "No cells in adata.obs_names matched the annotation file index. "
            "Verify that cell IDs use the same format in both sources."
        )

    # Column conflicts — annotation file takes precedence; warn the user
    conflict = sorted(set(adata.obs.columns) & set(annot.columns))
    if conflict:
        warnings.warn(
            f"[read_adata] The following adata.obs columns will be overwritten "
            f"by the annotation file: {conflict}",
            UserWarning,
            stacklevel=3,
        )
        adata = adata.copy()  # avoid mutating the caller's object
        adata.obs = adata.obs.drop(columns=conflict)

    # Merge
    obs_merged = adata.obs.merge(
        annot, left_index=True, right_index=True, how=obs_join
    )

    if obs_join == "inner":
        adata = adata[obs_merged.index, :].copy()
        print(f"[read_adata] after inner join: {adata.n_obs} cells retained")

    adata.obs = obs_merged
    return adata


# ---------------------------------------------------------------------------
# AnnData utilities
# ---------------------------------------------------------------------------

def prepare_batch_key(
    adata: ad.AnnData,
    batch_key: str | list[str] | None,
) -> tuple[ad.AnnData, str | None]:
    """Resolve a batch key, creating a joint column if necessary.

    If ``batch_key`` is a list of column names, a new column is added to
    ``adata.obs`` (on a copy) by concatenating values with ``"_"``.
    If ``batch_key`` is a string or ``None``, returns ``(adata, batch_key)``
    unchanged (no copy made).

    Parameters
    ----------
    adata : AnnData
    batch_key : str, list[str], or None

    Returns
    -------
    (adata, resolved_batch_key) : tuple
        ``adata`` may be a copy with a new joint column added.
        ``resolved_batch_key`` is the string column name to use downstream,
        or ``None`` if no batch key was provided.

    Raises
    ------
    ValueError
        If any element of ``batch_key`` (when a list) is not in ``adata.obs``.
    """
    if batch_key is None:
        return adata, None

    if isinstance(batch_key, str):
        return adata, batch_key

    missing = [k for k in batch_key if k not in adata.obs.columns]
    if missing:
        raise ValueError(f"batch_key columns not found in adata.obs: {missing}")

    joint_col = "__batch__" + "_".join(batch_key)
    adata = adata.copy()
    adata.obs[joint_col] = adata.obs[batch_key[0]].astype(str)
    for k in batch_key[1:]:
        adata.obs[joint_col] = adata.obs[joint_col] + "_" + adata.obs[k].astype(str)

    return adata, joint_col


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_adata(
    path: str | os.PathLike,
    *,
    genes: list[str] | None = None,
    obs_file: str | os.PathLike | None = None,
    obs_join: str = "inner",
    drop_raw: bool = True,
) -> ad.AnnData:
    """Read an h5ad file into memory.

    The file is always opened in backed mode first so that gene subsetting is
    cheap even for very large files.  Only after subsetting is the data pulled
    into memory.

    Parameters
    ----------
    path
        Path to the ``.h5ad`` file.
    genes
        List of gene names to subset to.  Only genes present in the file are
        retained; missing genes are reported with a warning and skipped.  If
        ``None``, all genes are loaded.
    obs_file
        Optional path to a ``.csv`` or ``.tsv`` annotation file whose row
        index contains cell IDs matching ``adata.obs_names``.  Columns are
        merged into ``adata.obs``; any column already present in ``adata.obs``
        is overwritten (a ``UserWarning`` is emitted).
    obs_join : {"inner", "left"}
        How to handle cells that have no matching row in the annotation file.

        * ``"inner"`` *(default)* — keep only cells that appear in both
          ``adata.obs_names`` and the annotation file index.  The number of
          cells retained is printed.
        * ``"left"`` — keep all cells; unmatched cells receive ``NaN`` in the
          annotation columns.

        Ignored when ``obs_file`` is ``None``.
    drop_raw
        Delete ``adata.raw`` after loading to free memory.  Defaults to
        ``True``.

    Returns
    -------
    AnnData
        Fully in-memory AnnData, ready for analysis.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``obs_join`` is not ``"inner"`` or ``"left"``, if ``genes`` is an
        empty list, if none of the requested genes are found, or if no cells
        in ``adata.obs_names`` match the annotation file index.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"h5ad file not found: {path}")

    if obs_join not in ("inner", "left"):
        raise ValueError(f"obs_join must be 'inner' or 'left', got {obs_join!r}")

    if genes is not None and len(genes) == 0:
        raise ValueError("genes list is empty.")

    # 1. Open backed — cheap regardless of file size
    adata = ad.read_h5ad(path, backed="r")
    print(f"[read_adata] opened backed: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # 2. Subset genes
    if genes is not None:
        var_set = set(adata.var_names)
        common = [g for g in genes if g in var_set]
        n_missing = len(genes) - len(common)
        if n_missing:
            warnings.warn(
                f"[read_adata] {n_missing}/{len(genes)} requested genes not found "
                f"in the h5ad file and will be skipped.",
                UserWarning,
                stacklevel=2,
            )
        if len(common) == 0:
            raise ValueError(
                f"None of the {len(genes)} requested genes were found in the adata "
                f"({adata.n_vars:,} genes available)."
            )
        print(f"[read_adata] subsetting to {len(common)}/{len(genes)} requested genes")
        adata = adata[:, common]

    # 3. Pull into memory
    adata = adata.to_memory()

    # 4. Drop raw (large files often carry a full-count copy here)
    if drop_raw and adata.raw is not None:
        del adata.raw

    # 5. Merge external annotation
    if obs_file is not None:
        adata = _merge_obs_file(adata, obs_file, obs_join=obs_join)

    return adata


def read_table(
    path: str | os.PathLike,
    *,
    index_col: int | str = 0,
) -> pd.DataFrame:
    """Read a CSV or TSV file into a DataFrame.

    The separator is inferred from the file extension:

    * ``.csv`` → ``,``
    * ``.tsv`` / ``.txt`` → ``\\t``
    * other → detected automatically by the pandas CSV sniffer

    Parameters
    ----------
    path
        Path to the file.
    index_col
        Column to use as the row index.  Defaults to ``0`` (first column).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Table file not found: {path}")

    sep = _sep_from_path(path)
    if sep is None:
        return pd.read_csv(path, index_col=index_col, sep=None, engine="python")
    return pd.read_csv(path, index_col=index_col, sep=sep)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_csv(
    df: pd.DataFrame,
    path: str | os.PathLike,
    *,
    index: bool = True,
) -> None:
    """Write a DataFrame to a CSV file.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    df
        DataFrame to write.
    path
        Destination path.  Conventionally ends with ``.csv``.
    index
        Whether to write the row index.  Defaults to ``True``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def write_tsv(
    df: pd.DataFrame,
    path: str | os.PathLike,
    *,
    index: bool = True,
) -> None:
    """Write a DataFrame to a TSV file.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    df
        DataFrame to write.
    path
        Destination path.  Conventionally ends with ``.tsv``.
    index
        Whether to write the row index.  Defaults to ``True``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index)


def write_json(
    obj: dict | list,
    path: str | os.PathLike,
    *,
    indent: int = 2,
) -> None:
    """Write a dict or list to a JSON file.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    obj
        Object to serialise.  Must be JSON-serialisable.
    path
        Destination path.  Conventionally ends with ``.json``.
    indent
        Number of spaces per indentation level.  Defaults to ``2``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent)
