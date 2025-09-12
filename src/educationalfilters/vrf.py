"""
vrf.py – Vocal Range Filters (VRF1/VRF2)
"""

from __future__ import annotations
import pandas as pd
from music21 import pitch as _m21_pitch

def _to_midi(s: str) -> int:
    return _m21_pitch.Pitch(s).midi

def ensure_ambitus_midi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'ambitus_min_midi' not in out.columns and 'ambitus_min' in out.columns:
        out['ambitus_min_midi'] = out['ambitus_min'].apply(lambda v: _m21_pitch.Pitch(v).midi if pd.notna(v) else None)
    if 'ambitus_max_midi' not in out.columns and 'ambitus_max' in out.columns:
        out['ambitus_max_midi'] = out['ambitus_max'].apply(lambda v: _m21_pitch.Pitch(v).midi if pd.notna(v) else None)
    if 'ambitus_span' not in out.columns:
        out['ambitus_span'] = out['ambitus_max_midi'] - out['ambitus_min_midi']
    return out

def filter_by_range(
    df: pd.DataFrame,
    pre_plus_min_pitch: str='A3',  # VRF2 lower
    pre_plus_max_pitch: str='C5',  # VRF2 upper
    pre_min_pitch: str='C4',       # VRF1 lower
    pre_max_pitch: str='A4'        # VRF1 upper
):
    """
    VRF2: A3..C5 AND span ≤ 12
    VRF1: C4..A4 AND span ≤ 12
    Returns (pre_plus_songs_df, pre_songs_df).
    """
    out = ensure_ambitus_midi(df)
    p2m = _m21_pitch.Pitch

    pre_plus = out[
        (out['ambitus_min_midi'] >= p2m(pre_plus_min_pitch).midi) &
        (out['ambitus_max_midi'] <= p2m(pre_plus_max_pitch).midi) &
        (out['ambitus_span'] <= 12)
    ]

    pre = out[
        (out['ambitus_min_midi'] >= p2m(pre_min_pitch).midi) &
        (out['ambitus_max_midi'] <= p2m(pre_max_pitch).midi) &
        (out['ambitus_span'] <= 12)
    ]
    return pre_plus, pre