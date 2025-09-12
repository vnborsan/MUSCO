"""
save_load.py

Centralised paths and helpers for saving/loading suffix arrays, bit vectors,
and descriptor dictionaries. Ensures consistent locations for raw data,
processed artefacts, and exports.

Environment override
--------------------
Set EDUFILT_DATA_DIR to redirect data directories, e.g.:

    export EDUFILT_DATA_DIR=/path/to/external/disk/educationalfilters-data
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union, Any
import os
import pickle
import pysdsl

# -----------------------------
# Project paths (single source)
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("EDUFILT_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXPORTS_DIR = ROOT / "exports"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _as_child_path(base: Path, name_or_path: Union[str, Path]) -> Path:
    p = Path(name_or_path)
    return p if p.is_absolute() or p.parts[:1] == (".",) else base / p


# -----------------------------------------
# SAVE / LOAD: SuffixArray + BitVector (SA/BV)
# -----------------------------------------
def save_sa_bv(
    s_a: pysdsl.SuffixArrayBitcompressed,
    b_v: pysdsl.BitVector,
    s_a_filename: Union[str, Path],
    b_v_filename: Union[str, Path],
    *,
    into: Path = PROCESSED_DIR,
) -> bool:
    """
    Save a suffix array and bit vector.

    If bare filenames are given, files are saved under ``data/processed``.
    You may also provide full paths.

    Parameters
    ----------
    s_a : pysdsl.SuffixArrayBitcompressed
        The suffix array object.
    b_v : pysdsl.BitVector
        The bit vector object.
    s_a_filename : str or Path
        Filename (or path) for suffix array.
    b_v_filename : str or Path
        Filename (or path) for bit vector.
    into : Path, optional
        Base directory (defaults to processed/).

    Returns
    -------
    bool
        True if both saves succeed.

    Examples
    --------
    >>> from educationalfilters.save_load import save_sa_bv
    >>> save_sa_bv(sa, bv, "bv_slp_mel.sa", "bv_slp_mel.bv")
    True
    """
    sa_path = _as_child_path(into, s_a_filename)
    bv_path = _as_child_path(into, b_v_filename)
    _ensure_parent(sa_path)
    _ensure_parent(bv_path)
    ok_sa = s_a.store_to_file(str(sa_path))
    ok_bv = b_v.store_to_file(str(bv_path))
    return bool(ok_sa and ok_bv)


def load_sa_bv(
    s_a_filename: Union[str, Path],
    b_v_filename: Union[str, Path],
    *,
    from_dir: Path = PROCESSED_DIR,
) -> Tuple[pysdsl.SuffixArrayBitcompressed, pysdsl.BitVector]:
    """
    Load a suffix array and bit vector.

    If bare filenames are given, they are resolved under ``data/processed``.

    Parameters
    ----------
    s_a_filename : str or Path
        Filename (or path) for suffix array.
    b_v_filename : str or Path
        Filename (or path) for bit vector.
    from_dir : Path, optional
        Base directory (defaults to processed/).

    Returns
    -------
    (SuffixArrayBitcompressed, BitVector)

    Examples
    --------
    >>> from educationalfilters.save_load import load_sa_bv
    >>> sa, bv = load_sa_bv("bv_slp_mel.sa", "bv_slp_mel.bv")
    """
    sa_path = _as_child_path(from_dir, s_a_filename)
    bv_path = _as_child_path(from_dir, b_v_filename)
    s_a = pysdsl.SuffixArrayBitcompressed.load_from_file(str(sa_path))
    b_v = pysdsl.BitVector.load_from_file(str(bv_path))
    return s_a, b_v


# ------------------------------
# SAVE / LOAD: descriptors (pkl)
# ------------------------------
def save_pickle(
    descriptor_dict: Any,
    descriptor_filename: Union[str, Path],
    *,
    into: Path = PROCESSED_DIR,
) -> Path:
    """
    Save a Python object (e.g. descriptor dictionary) as a pickle.

    By default, files go into ``data/processed``.

    Parameters
    ----------
    descriptor_dict : Any
        The object to save.
    descriptor_filename : str or Path
        Filename (or path).
    into : Path, optional
        Base directory (defaults to processed/).

    Returns
    -------
    Path
        The path written.

    Examples
    --------
    >>> from educationalfilters.save_load import save_pickle
    >>> save_pickle({"x": 42}, "descriptor_dict.pickle")
    Dictionary saved to .../data/processed/descriptor_dict.pickle
    """
    out_path = _as_child_path(into, descriptor_filename)
    _ensure_parent(out_path)
    with open(out_path, "wb") as f:
        pickle.dump(descriptor_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Dictionary saved to {out_path}")
    return out_path


def load_pickle(
    descriptor_filepath: Union[str, Path],
    *,
    from_dir: Path = PROCESSED_DIR,
) -> Any:
    """
    Load a pickle.

    If a bare filename is given, it is resolved under ``data/processed``.

    Parameters
    ----------
    descriptor_filepath : str or Path
        Filename (or path).
    from_dir : Path, optional
        Base directory (defaults to processed/).

    Returns
    -------
    Any
        The loaded object.

    Examples
    --------
    >>> from educationalfilters.save_load import load_pickle
    >>> desc = load_pickle("descriptor_dict.pickle")
    """
    in_path = _as_child_path(from_dir, descriptor_filepath)
    with open(in_path, "rb") as f:
        return pickle.load(f)


# ---------------
# Export helpers
# ---------------
def export_path(*parts: str) -> Path:
    """
    Return a path under exports/, creating parent folders as required.

    Parameters
    ----------
    parts : str
        Subdirectory/filename parts joined to form the export path.

    Returns
    -------
    Path
        The resulting path.

    Examples
    --------
    >>> from educationalfilters.save_load import export_path
    >>> out = export_path("figures", "overview.png")
    >>> print(out)
    .../exports/figures/overview.png
    """
    target = EXPORTS_DIR.joinpath(*parts)
    _ensure_parent(target)
    return target