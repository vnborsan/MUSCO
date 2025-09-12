"""
rfilters.py – Rhythm Filters (RF1–RF4).
"""

from __future__ import annotations
import pandas as pd

ALLOWED_BINARY_METERS = {"2/4", "4/4", "2/2"}
SIMPLE_RHYTHM  = {0.5, 1.0}
COMPLEX_RHYTHM = {0.5, 1.0, 2.0}
_RHYTHM_VALUES_MAP = {'a': 0.5, 'd': 1.0, 'e': 2.0}

def _check_ts(ts: str, allow_all=False) -> bool:
    if pd.isna(ts):
        return False
    sigs = set(str(ts).split(', '))
    return (len(sigs) == 1) if allow_all else (len(sigs) == 1 and sigs.issubset(ALLOWED_BINARY_METERS))

def compute_rhythm_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Compute RF label per row => 'RF1'...'RF4' or 'X', with counts.

    Returns
    -------
    df_out, criteria_counts
    """
    new_df = df.copy()
    new_df['rhythm_check'] = 'X'
    counts = {"RF1": 0, "RF2": 0, "RF3": 0, "RF4": 0, "X": 0}

    for i, row in new_df.iterrows():
        ts_binary = _check_ts(row.get("time_signature"))
        ts_single = _check_ts(row.get("time_signature"), allow_all=True)
        rhythm_string = str(row.get("rhythm_string_abc", ""))

        values = [_RHYTHM_VALUES_MAP.get(ch) for ch in rhythm_string]
        values = [v for v in values if v is not None]
        if not values:
            counts["X"] += 1
            continue

        total = len(values)
        simple_pc  = sum(1 for v in values if v in SIMPLE_RHYTHM)  / total
        complex_pc = sum(1 for v in values if v in COMPLEX_RHYTHM) / total
        pause_cnt  = int(row.get("pause_count", 0) or 0)

        if ts_binary and simple_pc  >= 0.90 and pause_cnt == 0:
            new_df.at[i, "rhythm_check"] = "RF1"; counts["RF1"] += 1
        elif ts_binary and complex_pc >= 0.70 and pause_cnt == 0:
            new_df.at[i, "rhythm_check"] = "RF2"; counts["RF2"] += 1
        elif ts_binary and complex_pc >= 0.70 and 0 <= pause_cnt <= 2:
            new_df.at[i, "rhythm_check"] = "RF3"; counts["RF3"] += 1
        elif ts_single and complex_pc >= 0.70 and 0 <= pause_cnt <= 4:
            new_df.at[i, "rhythm_check"] = "RF4"; counts["RF4"] += 1
        else:
            counts["X"] += 1

    return new_df, counts

# Backward compatibility: older code may still call filter_songs(...)
def filter_songs(df: pd.DataFrame, rhythm_mapping: dict, label: str):
    """
    Alias to compute_rhythm_labels(). rhythm_mapping and label are ignored
    by the current implementation; kept only to preserve the old signature.
    """
    return compute_rhythm_labels(df)