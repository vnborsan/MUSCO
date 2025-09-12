"""
prepare_df_patch.py

Data patching/normalization helpers applied BEFORE heavy processing.
Use these to make raw CSV/JSON inputs consistent.

Typical fixes:
- Trim/normalize string columns (IDs, time_signature, ABC strings)
- Sanitize time signatures like '2/4, 2/4' → '2/4'
- Coerce pause_count to integer, fill missing with 0
- Ensure required columns exist with safe defaults
"""

from __future__ import annotations
from typing import Iterable
import pandas as pd


REQUIRED_COLUMNS = [
    "metadata_filename",
    "corpus",
    "melodic_string_relative",
    "rhythm_string_abc",
    "time_signature",
    "ambitus_min",
    "ambitus_max",
    # optional/derived
    "pause_count",
]


def _strip_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _normalize_time_signature(ts: str) -> str:
    """
    Keep a single meter if duplicates given separated by commas.
    '2/4, 2/4' → '2/4'
    """
    if ts is None or pd.isna(ts):
        return ts
    parts = [p.strip() for p in str(ts).split(",") if p.strip()]
    return parts[0] if parts else ts


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all REQUIRED_COLUMNS exist; create safe defaults where missing.
    """
    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            if col in ("pause_count",):
                out[col] = 0
            else:
                out[col] = pd.NA
    return out


def patch_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a set of safe, idempotent cleanups to raw data.
    """
    out = ensure_columns(df)

    # ID + corpus
    out["metadata_filename"] = _strip_series(out["metadata_filename"])
    out["corpus"] = _strip_series(out["corpus"])

    # Strings
    if "melodic_string_relative" in out.columns:
        out["melodic_string_relative"] = out["melodic_string_relative"].astype(str).str.strip()

    if "rhythm_string_abc" in out.columns:
        out["rhythm_string_abc"] = out["rhythm_string_abc"].astype(str).str.strip()

    # Time signature
    if "time_signature" in out.columns:
        out["time_signature"] = (
            out["time_signature"]
            .astype(str)
            .str.strip()
            .apply(_normalize_time_signature)
        )

    # Ambitus strings (keep as strings; MIDI added later)
    if "ambitus_min" in out.columns:
        out["ambitus_min"] = out["ambitus_min"].astype(str).str.strip()
    if "ambitus_max" in out.columns:
        out["ambitus_max"] = out["ambitus_max"].astype(str).str.strip()

    # Pause count
    if "pause_count" in out.columns:
        out["pause_count"] = pd.to_numeric(out["pause_count"], errors="coerce").fillna(0).astype(int)

    # Drop obvious duplicates by (metadata_filename, corpus) while keeping first
    out = out.drop_duplicates(subset=["metadata_filename", "corpus"], keep="first")

    return out


def coalesce_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate frames with consistent columns and patch again to be safe.
    """
    frames = [ensure_columns(f) for f in frames]
    out = pd.concat(frames, ignore_index=True)
    return patch_basic(out)