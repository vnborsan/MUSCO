"""
filter_df – unified wrapper to apply VRF, IF, RF in one call (pipeline-compatible).

This wrapper preserves your previous logic:
  - VRF (range) via vrf.filter_by_range -> maps directly to VRF1 / VRF2 / X
  - IF  (intervals) via ifilters.recompute_if_labels -> direct IF1 / IF2 / X
  - RF  (rhythm) via rfilters.compute_rhythm_labels -> RF1..RF4 / X

It writes:
  - VRF_label ∈ {VRF1, VRF2, X}
  - IF_label  ∈ {IF1,  IF2,  X}
  - RF_label  ∈ {RF1, RF2, RF3, RF4, X}
  - rhythm_check (same as RF_label; kept for compatibility)
  - VRF_BOTH (bool): VRF1 or VRF2
  - IF_BOTH  (bool): IF1  or IF2
  - RF_BOTH  (bool): RF1..RF4

"""

from __future__ import annotations
import numpy as np
import pandas as pd

from . import vrf, ifilters, rfilters
from . import filter_label_utils as flu


def preschool_filter(
    df: pd.DataFrame,
    *,
    pre_plus_min_pitch: str = "A3",
    pre_plus_max_pitch: str = "C5",
    pre_min_pitch: str = "C4",
    pre_max_pitch: str = "A4",
    rhythm_mapping: dict | None = None,  # accepted for signature compatibility
) -> pd.DataFrame:
    """
    Apply VRF (range), IF (intervals), RF (rhythm) and return a copy of df
    with direct labels and *_BOTH flags. Preserves your prior filter logic.
    """
    out = df.copy()

    # ---- VRF (range) -> VRF_label (VRF1 / VRF2 / X) -------------------------
    pre_plus_songs, pre_songs = vrf.filter_by_range(
        out,
        pre_plus_min_pitch=pre_plus_min_pitch,
        pre_plus_max_pitch=pre_plus_max_pitch,
        pre_min_pitch=pre_min_pitch,
        pre_max_pitch=pre_max_pitch,
    )
    pre_ids      = set(pre_songs.get('metadata_filename', pd.Series([], dtype=object)))
    pre_plus_ids = set(pre_plus_songs.get('metadata_filename', pd.Series([], dtype=object)))

    def _vrf_label(mid: str) -> str:
        if mid in pre_ids:
            return 'VRF1'
        if mid in pre_plus_ids:
            return 'VRF2'
        return 'X'

    out['VRF_label'] = out['metadata_filename'].astype(str).map(_vrf_label)

    # ---- IF (intervals) -> IF_label (IF1 / IF2 / X), independent of VRF -----
    # This calls your existing function that computes IF1/IF2/X (no VRF gating).
    out = ifilters.recompute_if_labels(out)

    # ---- RF (rhythm) -> RF_label (RF1..RF4 / X) ------------------------------
    # rfilters uses rhythm_string_abc + time_signature (+ pause_count).
    rf_df, _ = rfilters.compute_rhythm_labels(out)
    rf_df['RF_label'] = rf_df['rhythm_check'].fillna('X')

    # ---- Convenience flags you requested -------------------------------------
    rf_df['VRF_BOTH'] = rf_df['VRF_label'].isin(['VRF1', 'VRF2'])
    rf_df['IF_BOTH']  = rf_df['IF_label'].isin(['IF1', 'IF2'])
    rf_df['RF_BOTH']  = rf_df['RF_label'].isin(['RF1', 'RF2', 'RF3', 'RF4'])

    # Optional: tidy dtypes (safe)
    for col in ('VRF_label', 'IF_label', 'RF_label'):
        if col in rf_df.columns:
            rf_df[col] = rf_df[col].astype('category')

    cols_to_drop = [c for c in ["interval_jumps_check", "rhythm_check"] if c in rf_df.columns]
    rf_df = rf_df.drop(columns=cols_to_drop)

    return rf_df

