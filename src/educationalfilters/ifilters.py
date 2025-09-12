"""
ifilters.py – Interval Filters (IF1/IF2) and helpers.
"""

from __future__ import annotations
import re
import pandas as pd

# Precompiled patterns (fast)
_PAT_FULL = re.compile(r'^(-?(0|1|2|3|4|5|7))(\s+-?(0|1|2|3|4|5|7))*$')
_PAT_11   = re.compile(r'.*\b1\s+1\b.*')
_PAT_m11  = re.compile(r'.*\b-1\s+-1\b.*')

def _parse_moves_str(melodic_string: str):
    s = str(melodic_string).strip()
    if not s:
        return []
    return list(map(int, s.split()))

def if1_allows(melodic_string: str) -> bool:
    """IF1: {0, ±2, ±3, ±4}; ±1 only if isolated (no adjacent ±1)."""
    try:
        moves = _parse_moves_str(melodic_string)
    except Exception:
        return False
    allowed_non_m2 = {0, 2, -2, 3, -3, 4, -4}
    n = len(moves)
    for i, m in enumerate(moves):
        if m in allowed_non_m2:
            continue
        if abs(m) == 1:
            left = (i > 0 and abs(moves[i-1]) == 1)
            right = (i < n - 1 and abs(moves[i+1]) == 1)
            if left or right:
                return False
            continue
        return False
    return True

def if2_allows(melodic_string: str) -> bool:
    """
    IF2: steps in {0, ±1, ±2, ±3, ±4, ±5, ±7}; 
         reject ±6 and intervals >±7; 
         reject same-direction adjacent semitones (1 1 or -1 -1).
    """
    try:
        moves = list(map(int, str(melodic_string).split()))
    except Exception:
        return False
    allowed_abs = {0, 1, 2, 3, 4, 5, 7}
    if any(abs(m) not in allowed_abs for m in moves):
        return False
    for i in range(1, len(moves)):
        if moves[i] == 1 and moves[i-1] == 1:
            return False
        if moves[i] == -1 and moves[i-1] == -1:
            return False
    return True

def recompute_if_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IF_label and interval_jumps_check for each row from melodic_string_relative.
    """
    out = df.copy()
    IF_label = []
    interval_jumps_check = []
    IF1_reason = []
    IF2_reason = []

    for mstr in out.get('melodic_string_relative', pd.Series(index=out.index, dtype=object)):
        p1 = if1_allows(mstr)
        p2 = if2_allows(mstr)
        IF1_reason.append("OK" if p1 else "fails IF1")
        IF2_reason.append("OK" if p2 else "fails IF2")
        if p1:
            IF_label.append('IF1'); interval_jumps_check.append('PRE')
        elif p2:
            IF_label.append('IF2'); interval_jumps_check.append('PRE_PLUS')
        else:
            IF_label.append('X');   interval_jumps_check.append('X')

    out['IF_label'] = IF_label
    out['interval_jumps_check'] = interval_jumps_check
    out['IF1_reason'] = IF1_reason
    out['IF2_reason'] = IF2_reason
    return out