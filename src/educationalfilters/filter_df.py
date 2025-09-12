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

import pandas as pd

def prepare_all_filters(new_df):
    """
    Legacy wide summary (kept for compatibility with your original notebook).
    Computes percentages per corpus at the song level using:
      - range_check -> VRF1/VRF2 mapping
      - interval_jumps_check -> IF1/IF2 mapping
      - rhythm_check -> RF1..RF4 (already assigned elsewhere)
    """
    corpora = {
        "Ciciban": new_df[new_df['corpus'] == 'Ciciban'],
        "SLP":     new_df[new_df['corpus'] == 'SLP']
    }
    
    all_results = {}
    for label, df in corpora.items():
        # Use song-level unit (avoid multi-row drift)
        if 'metadata_filename' in df.columns:
            base = df.drop_duplicates('metadata_filename')
        else:
            base = df.copy()

        total = len(base) if len(base) > 0 else 1  # safe denom

        vrf = base.get('range_check', pd.Series(index=base.index, dtype=object)) \
                  .map({'PRE': 'VRF1', 'PRE_PLUS': 'VRF2'}).fillna('X')
        iff = base.get('interval_jumps_check', pd.Series(index=base.index, dtype=object)) \
                  .map({'PRE': 'IF1', 'PRE_PLUS': 'IF2'}).fillna('X')
        rf  = base.get('rhythm_check', pd.Series(index=base.index, dtype=object)).fillna('X')

        # Cumulative flags to match RF style
        vrf2_cum = vrf.isin(['VRF1','VRF2'])
        if2_cum  = iff.isin(['IF1','IF2'])

        # RF cumulative counts
        rf1 = (rf == 'RF1').sum()
        rf2 = (rf.isin(['RF1','RF2'])).sum()
        rf3 = (rf.isin(['RF1','RF2','RF3'])).sum()
        rf4 = (rf.isin(['RF1','RF2','RF3','RF4'])).sum()

        results = {
            "VRF1": (vrf == 'VRF1').mean() * 100,
            "VRF2": vrf.isin(['VRF1','VRF2']).mean() * 100,  # cumulative
            "IF1":  (iff == 'IF1').mean() * 100,
            "IF2":  iff.isin(['IF1','IF2']).mean() * 100,    # cumulative
            "VRF1 + IF1": (((vrf == 'VRF1') & (iff == 'IF1')).mean() * 100),
            "VRF2+IF2":   (((vrf == 'VRF2') & (iff == 'IF2')).mean() * 100),
            "ANY (VRF+IF)": ((vrf != 'X') & (iff != 'X')).mean() * 100,

            # RF cumulative (as before)
            "RF1": rf1 / total * 100,
            "RF2": rf2 / total * 100,
            "RF3": rf3 / total * 100,
            "RF4": rf4 / total * 100,

            # Mixed combos
            "VRF2+IF2+RF3": (((vrf == 'VRF2') & (iff == 'IF2') & rf.isin(['RF1','RF2','RF3'])).mean() * 100),
            "VRF2+IF2+RF4": (((vrf == 'VRF2') & (iff == 'IF2') & rf.isin(['RF1','RF2','RF3','RF4'])).mean() * 100),

            # Any valid melodic / full
            "ANY (VRF+IF+RF)": ((vrf != 'X') & (iff != 'X') & rf.isin(['RF1','RF2','RF3','RF4'])).mean() * 100,

            # Extra cumulative coverage (if you were inspecting these)
            "VRF2_cummulative": (vrf2_cum.mean() * 100),
            "IF2_cumnulative":  (if2_cum.mean() * 100),
        }
        all_results[label] = results
    
    return pd.DataFrame(all_results)


def prepare_all_filters_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean summary with fixed row order, using VRF_label / IF_label / RF_label.
    """
    def song_base(dfc):
        return dfc.drop_duplicates('metadata_filename') if 'metadata_filename' in dfc.columns else dfc.copy()

    corpora = {
        "Ciciban": df[df['corpus'] == 'Ciciban'].copy(),
        "SLP":     df[df['corpus'] == 'SLP'].copy(),
    }

    out = {}

    for corpus_name, dfc in corpora.items():
        base = song_base(dfc)

        # Pull labels (default to 'X' if missing)
        vrf = base.get('VRF_label', pd.Series(index=base.index, dtype=object)).fillna('X')
        iff = base.get('IF_label',  pd.Series(index=base.index, dtype=object)).fillna('X')
        rf  = base.get('RF_label',  pd.Series(index=base.index, dtype=object)).fillna('X')

        # Booleans
        vrf1      = (vrf == 'VRF1')
        vrf2      = (vrf == 'VRF2')
        vrf2_cum  = vrf.isin(['VRF1','VRF2'])

        if1       = (iff == 'IF1')
        if2       = (iff == 'IF2')
        if2_cum   = iff.isin(['IF1','IF2'])

        rf1       = (rf == 'RF1')
        rf2_cum   = rf.isin(['RF1','RF2'])
        rf3_cum   = rf.isin(['RF1','RF2','RF3'])
        rf4_cum   = rf.isin(['RF1','RF2','RF3','RF4'])

        p = lambda s: float(s.mean()) * 100.0 if len(base) else 0.0

        results = {
            # Panel A (Melody)
            "VRF1":             p(vrf1),
            "IF1":              p(if1),
            "VRF2":             p(vrf2_cum),                 # cumulative
            "IF2":              p(if2_cum),                  # cumulative
            "VRF1 + IF1":       p(vrf1 & if1),               # strict
            "VRF2+IF2":         p(vrf2 & if2),               # strict
            "ANY (VRF+IF)":     p(vrf2_cum & if2_cum),       # cumulative

            # Panel B (Rhythm + Mixed)
            "RF1":              p(rf1),
            "RF2":              p(rf2_cum),
            "RF3":              p(rf3_cum),
            "RF4":              p(rf4_cum),
            "VRF2+IF2+RF3":     p(vrf2 & if2 & rf3_cum),     # strict VRF/IF + cum RF
            "VRF2+IF2+RF4":     p(vrf2 & if2 & rf4_cum),     # strict VRF/IF + cum RF
            "ANY (VRF+IF+RF)":  p(vrf2_cum & if2_cum & rf4_cum),
        }

        out[corpus_name] = results

    row_order_top = [
        "VRF1", "IF1", "VRF2", "IF2",
        "VRF1 + IF1", "VRF2+IF2", "ANY (VRF+IF)"
    ]
    row_order_bottom = [
        "RF1", "RF2", "RF3", "RF4",
        "VRF2+IF2+RF3", "VRF2+IF2+RF4", "ANY (VRF+IF+RF)"
    ]
    full_order = row_order_top + row_order_bottom

    df_summary = pd.DataFrame(out).reindex(full_order)
    return df_summary