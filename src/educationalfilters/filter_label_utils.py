"""
filter_label_utils.py
---------------------
Utilities to normalise and validate filter labels across the project.

- Maps:
    range_check:    PRE -> VRF1, PRE_PLUS -> VRF2
    interval_jumps: PRE -> IF1,  PRE_PLUS -> IF2
    rhythm_check:   passthrough (RF1..RF4 or X), with guard on unexpected values

- Adds cumulative flags compatible with "cumulative RF" style plots:
    VRF2_cum: True if VRF1 or VRF2
    IF2_cum:  True if IF1 or IF2

These helpers are non-destructive: they *add* columns alongside originals.
"""

from __future__ import annotations

from typing import Dict, Any
import pandas as pd

# Mappings for clearer labels
VRF_MAP: Dict[str, str] = {"PRE": "VRF1", "PRE_PLUS": "VRF2"}
IF_MAP: Dict[str, str] = {"PRE": "IF1", "PRE_PLUS": "IF2"}

# Allowed RF label values (also accept None before fillna)
RF_ALLOWED = {"RF1", "RF2", "RF3", "RF4", "X", None}


def standardise_filter_labels(
    df: pd.DataFrame,
    range_col: str = "range_check",
    interval_col: str = "interval_jumps_check",
    rhythm_col: str = "rhythm_check",
    in_place: bool = False,
) -> pd.DataFrame:
    """
    Add normalised labels alongside the original filter columns.

    Parameters
    ----------
    df : DataFrame
        Source dataframe (should already include the original *_check columns).
    range_col : str
        Column name for range filter ('range_check').
    interval_col : str
        Column name for interval jumps filter ('interval_jumps_check').
    rhythm_col : str
        Column name for rhythm filter ('rhythm_check').
    in_place : bool
        If True, mutate df; else return a copy.

    Returns
    -------
    DataFrame
        With added columns:
          - VRF_label (VRF1/VRF2/X)
          - IF_label  (IF1/IF2/X)
          - RF_label  (RF1..RF4/X, guarded)
          - VRF_exact (alias of VRF_label)
          - IF_exact  (alias of IF_label)
          - VRF2_cum (bool): VRF1 or VRF2
          - IF2_cum  (bool): IF1 or IF2
    """
    target = df if in_place else df.copy()

    if range_col in target.columns:
        target["VRF_label"] = target[range_col].map(VRF_MAP).fillna("X")
        target["VRF_exact"] = target["VRF_label"]
        target["VRF2_cum"] = target["VRF_label"].isin(["VRF1", "VRF2"])

    if interval_col in target.columns:
        target["IF_label"] = target[interval_col].map(IF_MAP).fillna("X")
        target["IF_exact"] = target["IF_label"]
        target["IF2_cum"] = target["IF_label"].isin(["IF1", "IF2"])

    if rhythm_col in target.columns:
        # Guard against unexpected values (e.g., typos like 'RF5')
        vals = target[rhythm_col]
        # Mark anything not in RF_ALLOWED as 'X'
        safe_vals = vals.where(vals.isin(RF_ALLOWED), other="X")
        target["RF_label"] = safe_vals.fillna("X")

    return target


def validate_filters(
    df: pd.DataFrame,
    range_col: str = "range_check",
    interval_col: str = "interval_jumps_check",
    rhythm_col: str = "rhythm_check",
) -> Dict[str, Any]:
    """
    Produce a compact report of unique values and counts in the filter columns.
    Adds a warning for unexpected RF values (outside RF_ALLOWED).

    Returns
    -------
    dict
        {
          'VRF (range_check)': { 'counts': {...}, 'uniques': [...] },
          'IF (interval_jumps_check)': {...},
          'RF (rhythm_check)': {..., 'warning': '...'}  # only if invalid RF values found
        }
    """
    report: Dict[str, Any] = {}

    for col, name in [
        (range_col, "VRF (range_check)"),
        (interval_col, "IF (interval_jumps_check)"),
        (rhythm_col, "RF (rhythm_check)"),
    ]:
        if col in df.columns:
            vals = df[col].astype("string").fillna("<NA>")
            counts = vals.value_counts(dropna=False).to_dict()
            uniques = sorted(set(vals.unique()))
            entry = {"counts": counts, "uniques": uniques}

            # Add an RF warning if we see unexpected values (ignore <NA>)
            if name.startswith("RF"):
                invalid_rf = [v for v in uniques if v not in {str(x) for x in RF_ALLOWED} and v != "<NA>"]
                if invalid_rf:
                    entry["warning"] = f"Unexpected RF values: {invalid_rf} (expected one of {sorted(RF_ALLOWED)})"

            report[name] = entry
        else:
            report[name] = {"error": f'Column "{col}" not found in DataFrame.'}

    return report