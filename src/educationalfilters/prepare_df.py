# --- Helpers restored from notebook-era logic --------------------------------
import os, json
import pandas as pd
from fractions import Fraction
from music21 import pitch as _m21_pitch, converter as _m21_converter, note as _m21_note
from typing import List, Dict, Optional, Tuple, Union
from . import save_load  # centralised paths for pickles
from pathlib import Path
from pandas.errors import ParserError
import ast, re

def convert_fraction_to_numeric_duration(rhythm_string: Union[str, List[str]]) -> List[float]:
    """
    Convert tokens like '1/2 1 2 3/4' (or list of tokens) to floats [0.5, 1.0, 2.0, 0.75].
    Silently skips tokens that cannot be parsed.
    """
    if isinstance(rhythm_string, list):
        tokens = rhythm_string
    else:
        tokens = str(rhythm_string).replace(",", " ").split()
    out = []
    for tok in tokens:
        try:
            if "/" in tok:
                out.append(float(Fraction(tok)))
            else:
                out.append(float(tok))
        except Exception:
            # ignore non-numeric tokens (ties, dots, etc.)
            continue
    return out

def midi_to_note_name(midi_val: Union[int, str, float, None]) -> Optional[str]:
    """60 -> 'C4'; returns None if midi is missing/invalid."""
    try:
        m = int(float(midi_val))
        return _m21_pitch.Pitch(m).nameWithOctave
    except Exception:
        return None

def convert_midi_to_notes(midi_string_abs: str) -> List[str]:
    """
    Input like '60 62 64' -> ['C4','D4','E4'].
    """
    toks = str(midi_string_abs).split()
    out = []
    for t in toks:
        try:
            out.append(_m21_pitch.Pitch(int(float(t))).nameWithOctave)
        except Exception:
            continue
    return out

def get_interval_name(semitones: Union[int, float, str, None]) -> Optional[str]:
    """
    Bucketize ambitus in semitones into labels you used in analysis.
    Adjust if you had a more detailed scheme.
    """
    try:
        s = int(float(semitones))
    except Exception:
        return None
    if s <= 8:
        return "WITHIN EIGHTH"
    if s <= 9:
        return "NINTH"
    if s <= 12:
        return "WITHIN OCTAVE"
    return "ABOVE EIGHTH"

# Default rhythm mapping you used in RF & indexing
_DEFAULT_RHYTHM_MAP: Dict[float, str] = {0.5: "a", 1.0: "d", 2.0: "e"}

def _ensure_single_time_signature(ts) -> Optional[str]:
    """Normalize time signature to a single string like '2/4'."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    if isinstance(ts, (list, tuple)):
        return str(ts[0]) if ts else None
    if isinstance(ts, dict):
        num, den = ts.get("num"), ts.get("den")
        return f"{num}/{den}" if num and den else None
    return str(ts)

# --- 1) CICIBAN JSONs -> DataFrame ------------------------------------------
def convert_jsons_to_df(folder_path: str) -> pd.DataFrame:
    """
    Build a Ciciban DataFrame from per-song JSON files.
    Keeps rhythm as numeric in 'rhythm_string'; letters added later in df_upgrade().
    """
    all_rows = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue
        fp = os.path.join(folder_path, filename)
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        melodic_abs = data["contour"]["melodic_contour_string_absolute"]
        rhythm_numeric = convert_fraction_to_numeric_duration(data["rhythm"]["rhythm_string"])

        rhythm_len = len(rhythm_numeric)
        melody_len = len(str(melodic_abs).split())
        pause_count = max(0, rhythm_len - melody_len)
        has_pauses = bool(pause_count > 0)

        stem = os.path.splitext(filename)[0]
        row = {
            "metadata_filename": stem,
            "corpus": "Ciciban",

            "metadata_composer": data.get("metadata", {}).get("composer"),
            "metadata_lyricist": data.get("metadata", {}).get("lyricist"),
            "metadata_title":   data.get("metadata", {}).get("title"),

            "key": data.get("key", {}).get("most_certain_key"),
            "time_signature": _ensure_single_time_signature(data.get("time_signature")),

            "ambitus_min": midi_to_note_name(data.get("ambitus", {}).get("min_note")),
            "ambitus_max": midi_to_note_name(data.get("ambitus", {}).get("max_note")),
            "ambitus_semitones": data.get("ambitus", {}).get("ambitus_semitones"),
            "ambitus_interval":  get_interval_name(data.get("ambitus", {}).get("ambitus_semitones")),

            "number_of_measures": data.get("duration", {}).get("measures"),

            "melodic_string":           data["contour"]["melodic_contour_string"],
            "melodic_string_absolute":  melodic_abs,
            "melodic_string_abc":       " ".join(convert_midi_to_notes(melodic_abs)),
            "melodic_string_relative":  data["contour"]["melodic_contour_string_relative"],

            "rhythm_string": rhythm_numeric,  # numeric durations here

            "rhythmic_measures": data["rhythm"]["measure_starts"],
            "melodic_measures":  data["contour"]["measure_starts"],

            "has_pauses": has_pauses,
            "pause_count": int(pause_count),
        }
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return df

# --- 2) Upgrade Ciciban DF: add letters + save mapping ----------------------
def df_upgrade(df: pd.DataFrame, save_file_path: str = "rhythm_mapping.pickle"
               ) -> Tuple[pd.DataFrame, Dict[float, str]]:
    """
    Convert df['rhythm_string'] (list of floats) to letters in df['rhythm_string_abc'],
    save mapping to data/processed/<save_file_path>, and return (df, mapping).
    """
    # Build mapping from durations present (fallback to defaults if unseen)
    durations = set()
    for seq in df.get("rhythm_string", []):
        if isinstance(seq, list):
            durations.update(seq)
    durations = {float(x) for x in durations if x is not None}

    # Reuse default letters for common values; extend with extra letters if needed
    mapping = dict(_DEFAULT_RHYTHM_MAP)  # start with defaults
    extra_letters = [chr(c) for c in range(ord("f"), ord("z")+1)]
    i = 0
    for d in sorted(durations):
        if d not in mapping:
            mapping[d] = extra_letters[i] if i < len(extra_letters) else f"x{len(mapping)}"
            i += 1

    # Apply mapping to create rhythm_string_abc
    def _to_letters(seq):
        if not isinstance(seq, list):
            return ""
        return "".join(mapping.get(float(x), "?") for x in seq)

    df = df.copy()
    df["rhythm_string_abc"] = df["rhythm_string"].apply(_to_letters)

    save_load.save_pickle(descriptor_dict=mapping, descriptor_filename=save_file_path)
   
    return df, mapping


# ------- HELPER -----------
def _coerce_rhythm_sequence(x):
    """
    Ensure rhythm_string is a list[float].
    Accepts:
      - list[float] already
      - stringified list: "[0.5, 1, 2]"
      - whitespace/comma-separated text: "0.5 1 2" or "0.5,1,2"
    """
    if isinstance(x, list):
        return [float(v) for v in x if v is not None]
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    # try Python literal (e.g., "[0.5, 1, 2]")
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [float(v) for v in val]
    except Exception:
        pass
    # fallback: split by whitespace/commas
    toks = re.split(r"[,\s]+", s)
    out = []
    for t in toks:
        if not t:
            continue
        try:
            out.append(float(t))
        except Exception:
            continue
    return out

def _melody_token_count_row(row: pd.Series) -> int:
    """Count melodic tokens; prefer absolute MIDI if present."""
    for col in ("melodic_string_absolute", "melodic_string", "melodic_string_abc"):
        if col in row and pd.notna(row[col]):
            return len(str(row[col]).split())
    return 0

def prepare_slp(filepath, mxl_folder, rhythm_mapping_path):
    """
    Loads and prepares the SLP dataset, adding:
    - **Melodic data (from CSV)**
    - **Rhythmic data (from rhythm_string IF present, else from MusicXML)**
    - **Time signatures normalized**
    - **Pauses detection and count**
    - **Rhythms converted to ABC notation**
    """
    # ✅ Load SLP metadata (keep your delimiter=';' as in your original)
    file_path = os.path.join(os.getcwd(), filepath)
    slp_df = pd.read_csv(file_path, delimiter=';')

    # --- Your existing tweak for quoted multi-TS rows ---
    if 'time_signature' in slp_df.columns:
        slp_df['time_signature'] = slp_df['time_signature'].apply(
            lambda x: f'"{x}"' if isinstance(x, str) and ',' in x else x
        )

    # ✅ Rename columns for consistency (as in your file)
    slp_df.rename(
        columns={
            'min_pitch': 'ambitus_min',
            'max_pitch': 'ambitus_max',
            'metadata_title': 'metadata_filename'
        },
        inplace=True
    )
    if 'ambitus_semitones' in slp_df.columns:
        slp_df['ambitus_interval'] = slp_df['ambitus_semitones'].apply(get_interval_name)

    # ✅ Ensure required columns exist
    if 'melodic_string_abc' not in slp_df.columns:
        slp_df['melodic_string_abc'] = ""
    if 'rhythm_string_abc' not in slp_df.columns:
        slp_df['rhythm_string_abc'] = ""
    if 'corpus' not in slp_df.columns:
        slp_df['corpus'] = 'SLP'
    if 'metadata_filename' not in slp_df.columns:
        slp_df['metadata_filename'] = slp_df.get('metadata_title', slp_df.index.astype(str))

    # ✅ Load rhythm mapping for ABC conversion (your original approach)
    rhythm_mapping = pickle.load(open(rhythm_mapping_path, 'rb'))

    # ---------- NEW: FAST-PATH USING CSV rhythm_string (no MXL needed) ----------
    def _coerce_rhythm_sequence(x):
        """
        Ensure rhythm_string is list[float] even if stored as text like:
        "[0.5, 1, 2]" or "0.5 1 2" or "0.5,1,2".
        """
        if isinstance(x, list):
            return [float(v) for v in x if v is not None]
        if pd.isna(x):
            return []
        s = str(x).strip()
        if not s:
            return []
        # attempt Python literal list
        if s.startswith('[') and s.endswith(']'):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list):
                    return [float(v) for v in val]
            except Exception:
                pass
        # fallback: split by commas/whitespace
        toks = re.split(r"[,\s]+", s)
        out = []
        for t in toks:
            if not t:
                continue
            try:
                out.append(float(t))
            except Exception:
                continue
        return out

    # If CSV has rhythm_string, coerce and use it
    csv_rhythm_present = 'rhythm_string' in slp_df.columns
    if csv_rhythm_present:
        slp_df['rhythm_string'] = slp_df['rhythm_string'].apply(_coerce_rhythm_sequence)
    else:
        slp_df['rhythm_string'] = [[] for _ in range(len(slp_df))]

    # Map durations -> letters using your saved mapping
    def _to_letters(seq):
        if not isinstance(seq, list):
            return ""
        return "".join(rhythm_mapping.get(float(x), "?") for x in seq)

    slp_df['rhythm_string_abc'] = slp_df['rhythm_string'].apply(_to_letters)

    # Compute pause_count and has_pauses (same logic as Ciciban)
    def _melody_token_count_row(row):
        for col in ('melodic_string_absolute', 'melodic_string', 'melodic_string_abc'):
            if col in row and pd.notna(row[col]):
                return len(str(row[col]).split())
        return 0

    slp_df['pause_count'] = slp_df.apply(
        lambda row: max(0, len(row['rhythm_string']) - _melody_token_count_row(row)),
        axis=1
    ).astype('Int64')
    slp_df['has_pauses'] = slp_df['pause_count'].fillna(0).astype(int).gt(0)

    # Normalize time_signature to 'N/D'
    if 'time_signature' in slp_df.columns:
        slp_df['time_signature'] = slp_df['time_signature'].astype(str).str.extract(r'(\d+/\d+)', expand=False)

    # Fill ambitus_min/max if the table had min/max pitch names
    if 'ambitus_min' in slp_df.columns and 'min_pitch' in slp_df.columns:
        slp_df['ambitus_min'] = slp_df['ambitus_min'].fillna(slp_df['min_pitch'])
    if 'ambitus_max' in slp_df.columns and 'max_pitch' in slp_df.columns:
        slp_df['ambitus_max'] = slp_df['ambitus_max'].fillna(slp_df['max_pitch'])

    # ---------- FALLBACK: MusicXML path (run only if rhythms still empty) ----------
    # If ALL rows have empty rhythm (length==0) and MXL folder exists, run your original MXL extraction
    try:
        need_mxl = (slp_df['rhythm_string'].apply(len) == 0).all() and os.path.isdir(mxl_folder)
    except Exception:
        need_mxl = False

    if need_mxl:
        # (keep your original MXL logic here exactly as you had it)
        mxl_files = {
            f.replace('.mxl', '').replace('--', '.').replace('-', '.'): os.path.join(mxl_folder, f)
            for f in os.listdir(mxl_folder) if f.endswith('.mxl')
        }

        for idx, row in slp_df.iterrows():
            meta_filename = str(row['metadata_filename']).replace(' ', '')
            matched_mxl = mxl_files.get(meta_filename)

            if matched_mxl:
                # Your original helper: extract_rhythm_and_time_from_mxl
                rhythm_sequence, time_signatures = extract_rhythm_and_time_from_mxl(matched_mxl)

                slp_df.at[idx, 'rhythm_string'] = rhythm_sequence or []
                slp_df.at[idx, 'time_signature'] = ', '.join(time_signatures) if time_signatures else row.get('time_signature')

                # ABC letters
                rhythm_ABC = ''.join(rhythm_mapping.get(float(num), '?') for num in rhythm_sequence)
                slp_df.at[idx, 'rhythm_string_abc'] = rhythm_ABC

                # pauses
                melody_length = _melody_token_count_row(row)
                pause_count = max(len(rhythm_sequence) - melody_length, 0)
                slp_df.at[idx, 'pause_count'] = pause_count
                slp_df.at[idx, 'has_pauses'] = bool(pause_count > 0)
            else:
                print(f"❌ No match for {meta_filename} (no MXL file)")

    slp_df['corpus'] = 'SLP'
    return slp_df