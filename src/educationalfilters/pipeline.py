from . import vrf, ifilters, rfilters, filter_label_utils
import numpy as np
import pandas as pd

from educationalfilters import vrf, ifilters, rfilters, filter_label_utils
import numpy as np
import pandas as pd

def apply_all_filters(
    df: pd.DataFrame,
    rhythm_mapping: dict,
    pre_plus_min_pitch: str,
    pre_plus_max_pitch: str,
    pre_min_pitch: str,
    pre_max_pitch: str,
) -> pd.DataFrame:
    """
    Replacement for the old `filter_df.preschool_filter`.
    Applies:
      - VRF (range)
      - IF (interval jumps)
      - RF (rhythm)
    Produces:
      - range_check, interval_jumps_check, rhythm_check
      - VRF_label/IF_label/RF_label + cumulative flags via `standardise_filter_labels`
    """
    # --- VRF (range) ---
    pre_plus_songs, pre_songs = vrf.filter_by_range(
        df,
        pre_plus_min_pitch=pre_plus_min_pitch,
        pre_plus_max_pitch=pre_plus_max_pitch,
        pre_min_pitch=pre_min_pitch,
        pre_max_pitch=pre_max_pitch,
    )

    # --- IF (intervals), with VRF gating sets ---
    match_pre_plus, match_pre, df_with_scores = ifilters.filter_by_interval_jumps(
        df, pre_plus_songs, pre_songs
    )

    # Build one output DF with consistent columns
    out = df_with_scores.copy()

    # Map range to PRE / PRE_PLUS / X
    out['range_check'] = np.where(
        out['metadata_filename'].isin(pre_songs['metadata_filename']), 'PRE',
        np.where(out['metadata_filename'].isin(pre_plus_songs['metadata_filename']), 'PRE_PLUS', 'X')
    )

    # Map IF hits to PRE / PRE_PLUS / X
    out['interval_jumps_check'] = np.where(
        out['metadata_filename'].isin(match_pre), 'PRE',
        np.where(out['metadata_filename'].isin(match_pre_plus), 'PRE_PLUS', 'X')
    )

    # Optional: recompute IF labels independently (ensures consistency for all rows)
    out = ifilters.recompute_if_labels(out)

    # --- RF (rhythm) ---
    # Label each song with its corpus if not present
    if 'corpus' not in out.columns:
        out['corpus'] = 'UNKNOWN'
    label = str(out['corpus'].iloc[0]) if len(out) else 'UNKNOWN'

    rf_df, _rf_counts = rfilters.filter_songs(out, rhythm_mapping, label)

    # --- Normalise labels to VRF/IF/RF short names + cumulative flags ---
    rf_df = filter_label_utils.standardise_filter_labels(
        rf_df,
        range_col='range_check',
        interval_col='interval_jumps_check',
        rhythm_col='rhythm_check',
        in_place=False
    )

    return rf_df