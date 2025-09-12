import pandas as pd
from music21 import pitch
import numpy as np
import matplotlib.pyplot as plt

def prepare_all_filters(new_df):
    import numpy as np
    import pandas as pd

    corpora = {
        "Ciciban": new_df[new_df['corpus'] == 'Ciciban'],
        "SLP": new_df[new_df['corpus'] == 'SLP']
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
            "VRF2": vrf.isin(['VRF1','VRF2']).mean() * 100,  # cumulative (VRF1 ∪ VRF2),
            "IF1":  (iff == 'IF1').mean() * 100,
            "IF2": iff.isin(['IF1','IF2']).mean() * 100,
            "VRF1+IF1": (((vrf == 'VRF1') & (iff == 'IF1')).mean() * 100),
            "VRF2+IF2": (((vrf == 'VRF2') & (iff == 'IF2')).mean() * 100),
            "ANY (VRF+IF)": ((vrf != 'X') & (iff != 'X')).mean() * 100,

            # RF cumulative (as before)
            "RF1": rf1 / total * 100,
            "RF2": rf2 / total * 100,
            "RF3": rf3 / total * 100,
            "RF4": rf4 / total * 100,

            # Mixed combos (as in your original intent)
            "VRF2+IF2+RF3": (((vrf == 'VRF2') & (iff == 'IF2') & rf.isin(['RF1','RF2','RF3'])).mean() * 100),
            "VRF2+IF2+RF4": (((vrf == 'VRF2') & (iff == 'IF2') & rf.isin(['RF1','RF2','RF3','RF4'])).mean() * 100),

            # Any valid melodic / full
            "ANY (VRF+IF)": ((vrf != 'X') & (iff != 'X')).mean() * 100,
            "ANY (VRF+IF+RF)": ((vrf != 'X') & (iff != 'X') & rf.isin(['RF1','RF2','RF3','RF4'])).mean() * 100,

            # NEW: cumulative coverage for fair comparison with RF
            "VRF2_cummulative": (vrf2_cum.mean() * 100),
            "IF2_cumnulative":  (if2_cum.mean() * 100),
        }
        all_results[label] = results
    
    return pd.DataFrame(all_results)

#*** CLEAN PREPARE***

def prepare_all_filters_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentages at the song level (drop duplicate metadata_filename per corpus).
    Uses ONLY VRF_label, IF_label, RF_label (and corpus, metadata_filename).
    Definitions:
      - VRF1: VRF_label == 'VRF1'
      - VRF2 (cum): VRF_label in {'VRF1','VRF2'}
      - IF1: IF_label == 'IF1'
      - IF2 (cum): IF_label in {'IF1','IF2'}
      - RFk (cum): RF_label in {'RF1',..., 'RFk'}

      Combos:
      - VRF1 + IF1: (VRF_label == 'VRF1') & (IF_label == 'IF1')       [strict]
      - VRF2+IF2:   (VRF_label == 'VRF2') & (IF_label == 'IF2')       [strict]
      - ANY (VRF+IF): (VRF ∈ {VRF1,VRF2}) & (IF ∈ {IF1,IF2})         [cumulative]
      - VRF2+IF2+RF3: (VRF=='VRF2') & (IF=='IF2') & (RF ∈ {RF1..RF3}) [strict VRF/IF, cum RF]
      - VRF2+IF2+RF4: (VRF=='VRF2') & (IF=='IF2') & (RF ∈ {RF1..RF4}) [strict VRF/IF, cum RF]
      - ANY (VRF+IF+RF): (VRF ∈{VRF1,VRF2}) & (IF ∈{IF1,IF2}) & (RF ∈{RF1..RF4}) [cumulative]
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
            # ---- Panel A (Melody) ----
            "VRF1":               p(vrf1),
            "IF1":                p(if1),
            "VRF2":               p(vrf2_cum),     # cumulative single
            "IF2":                p(if2_cum),      # cumulative single
            "VRF1 + IF1":         p(vrf1 & if1),   # strict
            "VRF2+IF2":           p(vrf2 & if2),   # strict (differs from ANY)
            "ANY (VRF+IF)":       p(vrf2_cum & if2_cum),  # cumulative intersection

            # ---- Panel B (Rhythm + Mixed) ----
            "RF1":                p(rf1),
            "RF2":                p(rf2_cum),      # cumulative
            "RF3":                p(rf3_cum),      # cumulative
            "RF4":                p(rf4_cum),      # cumulative
            "VRF2+IF2+RF3":       p(vrf2 & if2 & rf3_cum),  # strict VRF/IF + cum RF
            "VRF2+IF2+RF4":       p(vrf2 & if2 & rf4_cum),  # strict VRF/IF + cum RF
            "ANY (VRF+IF+RF)":    p(vrf2_cum & if2_cum & rf4_cum),  # all cumulative
        }

        out[corpus_name] = results

    # Fixed row order to match your figure captions
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

# === IF HELPERS (module-level) ===============================================
ALLOWED_IF2_ABS = {0, 1, 2, 3, 4, 5, 7}  # tritone (±6) excluded; >±7 excluded

def _parse_moves_str(melodic_string: str):
    s = str(melodic_string).strip()
    if not s:
        return []
    return list(map(int, s.split()))

def if1_allows(melodic_string: str) -> bool:
    """
    IF1: allowed {0, ±2, ±3, ±4}; ±1 only if isolated (no adjacent ±1).
    """
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
            left_is_m2  = (i > 0 and abs(moves[i-1]) == 1)
            right_is_m2 = (i < n - 1 and abs(moves[i+1]) == 1)
            if left_is_m2 or right_is_m2:
                return False
            continue
        return False
    return True

def if2_allows(melodic_string: str) -> bool:
    """
    IF2: every step in {0, ±1, ±2, ±3, ±4, ±5, ±7};
         reject tritone (±6) and any interval >±7;
         reject chromatic runs in the SAME direction (1 1 or -1 -1).
    """
    try:
        moves = list(map(int, str(melodic_string).split()))
    except Exception:
        return False

    allowed_abs = {0, 1, 2, 3, 4, 5, 7}
    if any(abs(m) not in allowed_abs for m in moves):
        return False

    # only disallow same-direction semitone sequences
    for i in range(1, len(moves)):
        if moves[i] == 1 and moves[i-1] == 1:
            return False
        if moves[i] == -1 and moves[i-1] == -1:
            return False

    return True


# *********
import re
pat_full = re.compile(r'^(-?(0|1|2|3|4|5|7))(\s+-?(0|1|2|3|4|5|7))*$')
pat_11   = re.compile(r'.*1\s+1.*')
pat_m11  = re.compile(r'.*-1\s+-1.*')

def es_like_if2(s: str) -> bool:
    s = str(s).strip()
    if not pat_full.match(s): 
        return False
    if pat_11.match(s) or pat_m11.match(s):
        return False
    return True

# *********
# ============================================================================#

def filter_by_range(
    df,
    pre_plus_min_pitch='A3',  # VRF2 lower
    pre_plus_max_pitch='C5',  # VRF2 upper
    pre_min_pitch='C4',       # VRF1 lower
    pre_max_pitch='A4'        # VRF1 upper
):
    """
    VRF2: A3..C5 AND span ≤ 12
    VRF1: C4..A4 (or D4..A4) AND span ≤ 12
    """
    from music21 import pitch

    def p2m(x): return pitch.Pitch(x).midi

    df = df.copy()
    if 'ambitus_min_midi' not in df.columns:
        df['ambitus_min_midi'] = df['ambitus_min'].apply(lambda v: pitch.Pitch(v).midi)
    if 'ambitus_max_midi' not in df.columns:
        df['ambitus_max_midi'] = df['ambitus_max'].apply(lambda v: pitch.Pitch(v).midi)
    df['ambitus_span'] = df['ambitus_max_midi'] - df['ambitus_min_midi']

    pre_plus_songs = df[
        (df['ambitus_min_midi'] >= p2m(pre_plus_min_pitch)) &
        (df['ambitus_max_midi'] <= p2m(pre_plus_max_pitch)) &
        (df['ambitus_span'] <= 12)
    ]

    pre_songs = df[
        (df['ambitus_min_midi'] >= p2m(pre_min_pitch)) &
        (df['ambitus_max_midi'] <= p2m(pre_max_pitch)) &
        (df['ambitus_span'] <= 12)
    ]

    return pre_plus_songs, pre_songs


def _parse_moves(melodic_string):
    import pandas as pd
    if pd.isna(melodic_string):
        return [], "empty melody"
    s = str(melodic_string).strip()
    if not s:
        return [], "empty melody"
    moves = []
    for tok in s.split():
        try:
            moves.append(int(tok))
        except Exception:
            return [], f"invalid token '{tok}'"
    return moves, ""

def _if2_pass_fail(moves):
    for i, m in enumerate(moves):
        if abs(m) > 7: 
            return False, f"interval >7 at idx {i} (value {m})"
    return True, "OK"

def _if1_pass_fail(moves):
    # IF1: allowed {0, ±2, ±3, ±4}; minor second (±1) allowed only in isolation
    allowed_non_m2 = {0, 2, -2, 3, -3, 4, -4}
    n = len(moves)
    for i, m in enumerate(moves):
        if m in allowed_non_m2:
            continue
        if m in (1, -1):
            left_is_m2  = (i > 0 and abs(moves[i-1]) == 1)
            right_is_m2 = (i < n-1 and abs(moves[i+1]) == 1)
            if left_is_m2 or right_is_m2:
                return False, f"consecutive m2 around idx {i}"
            continue
        return False, f"disallowed interval {m} at idx {i}"
    return True, "OK"

def recompute_if_labels(df):
    """
    Compute IF on EVERY entry, independent of VRF gating.
    Writes:
      - IF_label: 'IF1' / 'IF2' / 'X'
      - interval_jumps_check: 'PRE' (IF1) / 'PRE_PLUS' (IF2) / 'X'
      - IF1_reason, IF2_reason: short diagnostics
    """
    out = df.copy()
    IF_label = []
    interval_jumps_check = []
    IF1_reason = []
    IF2_reason = []

    for mstr in out['melodic_string_relative']:
        p1 = if1_allows(mstr)
        p2 = if2_allows(mstr)

        IF1_reason.append("OK" if p1 else "fails IF1")
        IF2_reason.append("OK" if p2 else "fails IF2")

        if p1:
            IF_label.append('IF1')
            interval_jumps_check.append('PRE')
        elif p2:
            IF_label.append('IF2')
            interval_jumps_check.append('PRE_PLUS')
        else:
            IF_label.append('X')
            interval_jumps_check.append('X')

    out['IF_label'] = IF_label
    out['interval_jumps_check'] = interval_jumps_check
    out['IF1_reason'] = IF1_reason
    out['IF2_reason'] = IF2_reason
    return out


def filter_by_interval_jumps(df, pre_plus_songs, pre_songs):
    """
    Computes IF matches with optional VRF gating, and adds interval_score.

    Returns
    -------
    matching_pre_plus_filenames : list[str]  # IF2 matches (gated by VRF2 set)
    matching_pre_filenames      : list[str]  # IF1 matches (gated by VRF1 set)
    df_with_scores              : DataFrame  # copy of df with 'interval_score'
    """
    import pandas as pd

    df_with_scores = df.copy()

    # Safety: required columns
    for col in ('metadata_filename', 'melodic_string_relative'):
        if col not in df_with_scores.columns:
            raise KeyError(f"filter_by_interval_jumps: missing column '{col}'")

    # --- interval_score helper (adjacent repeated unisons / total intervals)
    def calculate_interval_score(melodic_string: str) -> float:
        try:
            moves = list(map(int, str(melodic_string).split()))
        except Exception:
            return 0.0
        n = len(moves)
        if n == 0:
            return 0.0
        repeats = sum(1 for i in range(1, n) if moves[i] == 0 and moves[i-1] == 0)
        return repeats / n

    df_with_scores['interval_score'] = df_with_scores['melodic_string_relative'].apply(
        calculate_interval_score
    )

    # ---- VRF gating sets (names)
    pre_names       = set(pre_songs.get('metadata_filename', pd.Series([], dtype=object)))
    pre_plus_names  = set(pre_plus_songs.get('metadata_filename', pd.Series([], dtype=object)))

    # ---- IF rules (call the module-level helpers you added earlier)
    # IF1: {0, ±2, ±3, ±4} + isolated ±1 only (no adjacent ±1)
    mask_if1 = (
        df_with_scores['metadata_filename'].isin(pre_names) &
        df_with_scores['melodic_string_relative'].apply(if1_allows)
    )
    # IF2: {0, ±1, ±2, ±3, ±4, ±5, ±7} (±6 excluded) and no adjacent ±1
    mask_if2 = (
        df_with_scores['metadata_filename'].isin(pre_plus_names) &
        df_with_scores['melodic_string_relative'].apply(if2_allows)
    )

    matching_pre_filenames      = df_with_scores.loc[mask_if1, 'metadata_filename'].tolist()
    matching_pre_plus_filenames = df_with_scores.loc[mask_if2, 'metadata_filename'].tolist()

    return matching_pre_plus_filenames, matching_pre_filenames, df_with_scores

def get_pre_and_pre_plus_titles(ciciban_df):
    """
    Gets titles based on their 'range_check' and 'interval_jumps_check' values.

    Parameters:
    - ciciban_df (DataFrame): The DataFrame with 'range_check' and 'interval_jumps_check' columns.

    Returns:
    - pre_only_titles (list): List of titles where 'interval_jumps_check' is only 'PRE'.
    - pre_or_pre_plus_titles (list): List of titles where 'interval_jumps_check' includes either 'PRE' or 'PRE_PLUS'.
    """
    
    # Step 1: Filter rows where 'range_check' is 'PRE'
    pre_range_df = ciciban_df[ciciban_df['range_check'] == 'PRE']
    
    # Step 2: Group by 'metadata_filename' and collect all unique 'interval_jumps_check' values for each title
    grouped = pre_range_df.groupby('metadata_filename')['interval_jumps_check'].unique()
    
    # List 1: Titles where 'interval_jumps_check' is only 'PRE'
    pre_only_titles = grouped[grouped.apply(lambda x: set(x) == {'PRE'})].index.tolist()
    
    # List 2: Titles where 'interval_jumps_check' includes either 'PRE' or 'PRE_PLUS'
    pre_or_pre_plus_titles = grouped[grouped.apply(lambda x: {'PRE'}.issubset(x) or {'PRE_PLUS'}.issubset(x))].index.tolist()
    
    return pre_only_titles, pre_or_pre_plus_titles

def preschool_filter(df, pre_plus_min_pitch='A3', pre_plus_max_pitch='C5', pre_min_pitch='C4', pre_max_pitch='A4'):
    import numpy as np

    # 1) Range filtering (as before)
    pre_plus_songs, pre_songs = filter_by_range(
        df,
        pre_plus_min_pitch=pre_plus_min_pitch,
        pre_plus_max_pitch=pre_plus_max_pitch,
        pre_min_pitch=pre_min_pitch,
        pre_max_pitch=pre_max_pitch
    )
    
    # 2) Interval filtering (as before)
    matching_pre_plus_filenames, matching_pre_filenames, df_with_scores = filter_by_interval_jumps(
        df, pre_plus_songs, pre_songs
    )

    # ---- Work on a single output DataFrame ----
    df_out = df_with_scores.copy()

    # 3) Assign range_check (PRE / PRE_PLUS / X) consistently
    df_out['range_check'] = np.where(
        df_out['metadata_filename'].isin(pre_songs['metadata_filename']), 'PRE',
        np.where(df_out['metadata_filename'].isin(pre_plus_songs['metadata_filename']), 'PRE_PLUS', 'X')
    )

    # 4) Assign interval_jumps_check consistently (legacy mapping from matches)
    df_out['interval_jumps_check'] = np.where(
        df_out['metadata_filename'].isin(matching_pre_filenames), 'PRE',
        np.where(df_out['metadata_filename'].isin(matching_pre_plus_filenames), 'PRE_PLUS', 'X')
    )

    # 5) Add short labels (VRF / IF) from current checks
    df_out['VRF_label'] = df_out['range_check'].map({'PRE': 'VRF1', 'PRE_PLUS': 'VRF2'}).fillna('X')
    df_out['IF_label']  = df_out['interval_jumps_check'].map({'PRE': 'IF1', 'PRE_PLUS': 'IF2'}).fillna('X')
    df_out['VRF2_cum']  = df_out['VRF_label'].isin(['VRF1', 'VRF2'])
    df_out['IF2_cum']   = df_out['IF_label'].isin(['IF1', 'IF2'])

    # 6) NOW recompute IF on EVERY entry (independent run wins; ensures SLP gets populated)
    df_out = recompute_if_labels(df_out)
    df_out['IF2_cum'] = df_out['IF_label'].isin(['IF1','IF2'])

    # 7) Song-level counts & percentages using VRF/IF short names
    total_rows = len(df_out)
    vrf1_count = (df_out['VRF_label'] == 'VRF1').sum()
    vrf2_count = (df_out['VRF_label'] == 'VRF2').sum()
    if1_count  = (df_out['IF_label'] == 'IF1').sum()
    if2_count  = (df_out['IF_label'] == 'IF2').sum()
    both_vrf1_if1 = ((df_out['VRF_label'] == 'VRF1') & (df_out['IF_label'] == 'IF1')).sum()

    print(f"VRF1 count: {vrf1_count} ({(vrf1_count/total_rows)*100:.2f}%)")
    print(f"VRF2 count: {vrf2_count} ({(vrf2_count/total_rows)*100:.2f}%)")
    print(f"IF1 count: {if1_count} ({(if1_count/total_rows)*100:.2f}%)")
    print(f"IF2 count: {if2_count} ({(if2_count/total_rows)*100:.2f}%)")
    print(f"VRF1 & IF1 (both): {both_vrf1_if1} ({(both_vrf1_if1/total_rows)*100:.2f}%)")

    # 8) Optional: VRF/IF title lists (VRF/IF naming only)
    # If you want to keep the "PRE titles" idea but with new labels, you can switch to a VRF/IF version later.

    return df_out


##### RHYTHM ######

import pandas as pd
from music21 import meter


# Allowed binary meters
ALLOWED_BINARY_METERS = {"2/4", "4/4", "2/2"}
SIMPLE_RHYTHM = {0.5, 1.0}
COMPLEX_RHYTHM = {0.5, 1.0, 2.0}

def check_time_signature_validity(time_signature, allow_all=False):
    """
    Checks whether a song has a valid time signature.
    - If allow_all=True, any single time signature is allowed.
    """
    if pd.isna(time_signature):
        return False  # If empty, it's invalid

    time_sigs = set(time_signature.split(', '))  # Convert to a set

    if allow_all:
        return len(time_sigs) == 1  # Any single time signature
    else:
        return len(time_sigs) == 1 and time_sigs.issubset(ALLOWED_BINARY_METERS)  # Strict binary check

from music21 import pitch as _m21_pitch

def filter_songs(df, rhythm_mapping, label):
    """
    Build the per-song frame and compute RF labels.
    Safe against missing columns: will create ambitus_*_midi, interval_score,
    range_check, interval_jumps_check if absent.
    """
    import pandas as pd

    base = df.copy()

    # --- ensure MIDI ambitus cols exist --------------------------------------
    if 'ambitus_min_midi' not in base.columns:
        base['ambitus_min_midi'] = base['ambitus_min'].apply(
            lambda v: _m21_pitch.Pitch(v).midi if pd.notna(v) else None
        )
    if 'ambitus_max_midi' not in base.columns:
        base['ambitus_max_midi'] = base['ambitus_max'].apply(
            lambda v: _m21_pitch.Pitch(v).midi if pd.notna(v) else None
        )

    # --- ensure other optional fields exist -----------------------------------
    if 'interval_score' not in base.columns:
        base['interval_score'] = 0.0
    if 'range_check' not in base.columns:
        base['range_check'] = 'X'
    if 'interval_jumps_check' not in base.columns:
        base['interval_jumps_check'] = 'X'
    if 'pause_count' not in base.columns:
        # fall back to 0 if you don’t have pauses computed yet
        base['pause_count'] = 0
    if 'has_pauses' not in base.columns:
        base['has_pauses'] = base['pause_count'].fillna(0).astype(int).gt(0)

    # --- keep the columns we need (after creating the missing ones) -----------
    cols = [
        'metadata_filename', 'corpus',
        'time_signature', 'ambitus_min', 'ambitus_max',
        'ambitus_semitones', 'ambitus_interval',
        'melodic_string_abc', 'melodic_string_relative',
        'rhythm_string_abc',
        'has_pauses', 'pause_count',
        'ambitus_min_midi', 'ambitus_max_midi', 'interval_score',
        'range_check', 'interval_jumps_check'
    ]
    # intersect with existing (should all exist now, but keep safe):
    cols = [c for c in cols if c in base.columns]
    new_df = base[cols].copy()
    new_df["corpus"] = label
    new_df['rhythm_check'] = None  # will be filled below

    # --- RF classification (unchanged logic except now safe to run) -----------
    ALLOWED_BINARY_METERS = {"2/4", "4/4", "2/2"}
    SIMPLE_RHYTHM  = {0.5, 1.0}
    COMPLEX_RHYTHM = {0.5, 1.0, 2.0}

    def check_ts(ts: str, allow_all=False) -> bool:
        if pd.isna(ts):
            return False
        sigs = set(str(ts).split(', '))
        return (len(sigs) == 1) if allow_all else (len(sigs) == 1 and sigs.issubset(ALLOWED_BINARY_METERS))

    rhythm_values_map = {'a': 0.5, 'd': 1.0, 'e': 2.0}

    criteria_counts = {"RF1": 0, "RF2": 0, "RF3": 0, "RF4": 0, "X": 0}

    for i, row in new_df.iterrows():
        ts_binary = check_ts(row.get("time_signature"))
        ts_single = check_ts(row.get("time_signature"), allow_all=True)
        rhythm_string = str(row.get("rhythm_string_abc", ""))

        values = [rhythm_values_map.get(ch) for ch in rhythm_string]
        values = [v for v in values if v is not None]
        if not values:
            new_df.at[i, "rhythm_check"] = "X"
            criteria_counts["X"] += 1
            continue

        total = len(values)
        simple_pc  = sum(1 for v in values if v in SIMPLE_RHYTHM)  / total
        complex_pc = sum(1 for v in values if v in COMPLEX_RHYTHM) / total
        pause_cnt  = int(row.get("pause_count", 0))

        if ts_binary and simple_pc  >= 0.90 and pause_cnt == 0:  # your chosen RF1 threshold
            new_df.at[i, "rhythm_check"] = "RF1"; criteria_counts["RF1"] += 1
        elif ts_binary and complex_pc >= 0.70 and pause_cnt == 0:
            new_df.at[i, "rhythm_check"] = "RF2"; criteria_counts["RF2"] += 1
        elif ts_binary and complex_pc >= 0.70 and 0 <= pause_cnt <= 2:
            new_df.at[i, "rhythm_check"] = "RF3"; criteria_counts["RF3"] += 1
        elif ts_single and complex_pc >= 0.70 and 0 <= pause_cnt <= 4:
            new_df.at[i, "rhythm_check"] = "RF4"; criteria_counts["RF4"] += 1
        else:
            new_df.at[i, "rhythm_check"] = "X";   criteria_counts["X"]  += 1

    return new_df, criteria_counts


