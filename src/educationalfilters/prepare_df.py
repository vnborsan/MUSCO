import os
import json
import re
import ast
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import pandas as pd
from pandas.errors import ParserError

import music21
from music21 import pitch as _m21_pitch, converter as _m21_converter, note as _m21_note

from . import save_load  # centralised paths for pickles

# ---------------------------------------------

# --- Canonical rhythm mapping (shared across corpora) ------------------------
_BASE_MAP = {0.5: "a", 1.0: "d", 2.0: "e"}          # your original base
_EXTRA_LETTERS = [chr(c) for c in range(ord("f"), ord("z")+1)]  # f..z

def _norm_dur(x, ndigits: int = 3):
    """Normalize to avoid float drift (e.g., 0.4999999 → 0.5)."""
    try:
        return round(float(x), ndigits)
    except Exception:
        return None

from typing import Iterable

def _collect_durations_from_column(col: Iterable) -> set[float]:
    """Collect unique float durations from a column that may contain lists."""
    out = set()
    for seq in col:
        if isinstance(seq, list):
            for v in seq:
                try:
                    out.add(float(v))
                except Exception:
                    continue
    return out

def get_or_extend_canonical_map(observed_durations, map_path: str | Path) -> dict[float, str]:
    """
    Load the canonical rhythm map from pickle; extend it with any unseen durations.
    Saves back only if extended. Returns the mapping dict.
    """
    mapping = save_load.load_pickle(map_path)  # existing map (dict[float->str])
    # Letters pool for new durations
    used_letters = set(mapping.values())
    next_letters = [chr(c) for c in range(ord("h"), ord("z") + 1)]
    # Make sure we don't reuse letters
    next_letters = [ch for ch in next_letters if ch not in used_letters]

    changed = False
    for d in sorted({float(x) for x in observed_durations}):
        if d not in mapping:
            mapping[d] = next_letters.pop(0) if next_letters else f"x{len(mapping)}"
            changed = True

    if changed:
        # ✅ Correct argument order (or use keywords)
        save_load.save_pickle(mapping, map_path)
        # or: save_load.save_pickle(descriptor_dict=mapping, descriptor_filename=map_path)
    return mapping

def convert_to_abc(seq: list[float], mapping: dict[float, str]) -> str:
    """Map a list of durations to a compact ABC-letter string."""
    if not isinstance(seq, list):
        return ""
    return "".join(mapping.get(float(x), "?") for x in seq)

def durations_from_score(path: Path):
    """
    Parse a MusicXML score with music21 and return:
      - a list of quarterLength floats for all notes/rests
      - a list of unique time signature strings
    """
    try:
        s = _m21_converter.parse(str(path))
        durations = []
        for e in s.recurse().notesAndRests:
            ql = getattr(e, "quarterLength", None)
            if ql is not None:
                durations.append(float(ql))
        # collect time signatures
        ts = {ts.ratioString for ts in s.recurse().getElementsByClass("TimeSignature")}
        return durations, list(ts)
    except Exception:
        return [], []
    
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

def interval_label(semitones):
    """
    Map an integer semitone span to interval names up to OCTAVE.
    >12 collapses to 'ABOVE OCTAVE'.
    """
    try:
        s = int(round(float(semitones)))
    except Exception:
        return None

    names = {
         0: "UNISON",
         1: "SECOND",
         2: "SECOND",
         3: "THIRD",
         4: "THIRD",
         5: "PERFECT FOURTH",
         6: "TRITONE",
         7: "PERFECT FIFTH",
         8: "SIXTH",
         9: "SIXTH",
        10: "SEVENTH",
        11: "SEVENTH",
        12: "OCTAVE",
    }
    if s in names:
        return names[s]
    return "ABOVE OCTAVE"

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
    Enforces NO-PAUSES-IN-SEQUENCE:
      - rhythm_string = notes-only (truncate to melody length)
      - pause_count = (full rhythm events) - (melody tokens), never < 0
    """
    all_rows = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue
        fp = os.path.join(folder_path, filename)
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --- melody & rhythm (raw) ---
        melodic_abs = data["contour"]["melodic_contour_string_absolute"]
        rhythm_numeric_full = convert_fraction_to_numeric_duration(
            data["rhythm"]["rhythm_string"]
        )

        # --- enforce notes-only rhythm; pauses recorded separately ---
        melody_len = len(str(melodic_abs).split())
        pause_count = max(0, len(rhythm_numeric_full) - melody_len)
        rhythm_numeric_notes_only = rhythm_numeric_full[:melody_len]
        has_pauses = bool(pause_count > 0)

        stem = os.path.splitext(filename)[0]
        row = {
            "metadata_filename": stem,
            "corpus": "Ciciban",

            "metadata_composer": data.get("metadata", {}).get("composer"),
            "metadata_lyricist": data.get("metadata", {}).get("lyricist"),
            "metadata_title":   data.get("metadata", {}).get("title"),

            "key": data.get("key", {}).get("most_certain_key"),
            "time_signature": (lambda xs: ", ".join(dict.fromkeys(xs)) if xs else None)(
            re.findall(r"\d+/\d+", str(data.get("time_signature") or ""))),
            "ambitus_min": midi_to_note_name(data.get("ambitus", {}).get("min_note")),
            "ambitus_max": midi_to_note_name(data.get("ambitus", {}).get("max_note")),
            "ambitus_semitones": data.get("ambitus", {}).get("ambitus_semitones"),
            "ambitus_interval":  interval_label(data.get("ambitus", {}).get("ambitus_semitones")),

            "number_of_measures": data.get("duration", {}).get("measures"),

            "melodic_string":           data["contour"]["melodic_contour_string"],
            "melodic_string_absolute":  melodic_abs,
            "melodic_string_abc":       " ".join(convert_midi_to_notes(melodic_abs)),
            "melodic_string_relative":  data["contour"]["melodic_contour_string_relative"],

            # notes-only; pauses counted separately
            "rhythm_string": rhythm_numeric_notes_only,

            "rhythmic_measures": data["rhythm"]["measure_starts"],
            "melodic_measures":  data["contour"]["measure_starts"],

            "has_pauses": has_pauses,
            "pause_count": int(pause_count),
        }
        all_rows.append(row)

    return pd.DataFrame(all_rows)

# --- 2) Upgrade Ciciban DF: add letters + save mapping ----------------------
def df_upgrade(df: pd.DataFrame, save_file_path: str = "rhythm_mapping.pickle"):
    """
    Convert df['rhythm_string'] (notes-only, list[float]) to ABC using a single
    canonical mapping shared across corpora. Extends the mapping deterministically
    if new durations appear, then saves it to data/processed/<save_file_path>.
    Returns (df, mapping).
    """
    df = df.copy()
    observed = _collect_durations_from_column(df.get("rhythm_string", []))
    mapping = get_or_extend_canonical_map(observed, save_file_path)
    df["rhythm_string_abc"] = df["rhythm_string"].apply(lambda s: convert_to_abc(s, mapping))
    return df, mapping

# --- 2) Upgrade SLP DF: add rhythm and convert it to letters + save mapping ----------------------

def convert_rhythm_to_ABC(query, rhythm_mapping):
    """
    Converts rhythm to ABC notation.
    If the query is already a string (like 'accb'), return it as is.
    If it's a list of durations (like [0.5, 1.0]), map using the rhythm_mapping.
    """
    if isinstance(query, str):
        return query  # Already in ABC form

    if isinstance(query, list):
        try:
            # ensure float keys
            rm = {float(k): v for k, v in rhythm_mapping.items()}
            return ''.join(rm.get(float(num), '?') for num in query)
        except Exception as e:
            print("Error during rhythm conversion:", e)
            return "?" * len(query)  # fallback

    raise ValueError("Unrecognized input format for rhythm query: must be str or list of floats.")


def extract_rhythm_and_time_from_mxl(mxl_file):
    """
    Extract note durations (no rests) + all time signatures + pause count from a MusicXML (.mxl).
    Returns: (rhythm_sequence, time_signatures, pause_count)
    """
    try:
        score = _m21_converter.parse(mxl_file)
        rhythm_sequence = []
        time_signatures = set()
        pause_count = 0

        for part in score.parts:
            for element in part.flat:
                if isinstance(element, music21.meter.TimeSignature):
                    time_signatures.add(str(element.ratioString))
                elif isinstance(element, music21.note.Note):
                    rhythm_sequence.append(element.quarterLength)  # notes only
                elif isinstance(element, music21.note.Rest):
                    pause_count += 1  # rests counted separately

        return rhythm_sequence, list(time_signatures), pause_count

    except Exception as e:
        print(f"⚠️ Error parsing {mxl_file}: {e}")
        return [], [], 0


# --- main SLP prep (replace your current prepare_slp with this) --------------

def prepare_slp(filepath, mxl_folder, rhythm_mapping_path):
    """
    Loads and prepares the SLP dataset, adding:
    - **Melodic data (from CSV)**
    - **Rhythmic data (from MusicXML)**
    - **Time signatures (all found in score)**
    - **Pauses detection and count (vs melodic_string_nice tokens)**
    - **Rhythms converted to ABC notation**
    """
    # ✅ Load SLP metadata (your original delimiter)
    file_path = os.path.join(os.getcwd(), filepath)
    slp_df = pd.read_csv(file_path, delimiter=';')

    # ✅ Your tweak for quoted multi-TS rows
    if 'time_signature' in slp_df.columns:
        slp_df['time_signature'] = slp_df['time_signature'].apply(
            lambda x: f'"{x}"' if isinstance(x, str) and ',' in x else x
        )

    # ✅ Rename columns for consistency
    slp_df.rename(
        columns={
            'min_pitch': 'ambitus_min',
            'max_pitch': 'ambitus_max',
            'metadata_title': 'metadata_filename'
        },
        inplace=True
    )
    if 'ambitus_semitones' in slp_df.columns:
        slp_df['ambitus_interval'] = slp_df['ambitus_semitones'].apply(interval_label)

        # Ensure a unique, simple index
    slp_df = slp_df.reset_index(drop=True)

    # Pre-create target columns (avoid shared list objects!)
    if 'rhythm_string' not in slp_df.columns:
        slp_df['rhythm_string'] = pd.Series([[] for _ in range(len(slp_df))], dtype=object)
    if 'rhythm_string_abc' not in slp_df.columns:
        slp_df['rhythm_string_abc'] = ""
    if 'rhythm_length' not in slp_df.columns:
        slp_df['rhythm_length'] = 0
    if 'pause_count' not in slp_df.columns:
        slp_df['pause_count'] = 0
    if 'has_pauses' not in slp_df.columns:
        slp_df['has_pauses'] = "NO"

    # ✅ Ensure required columns exist
    if 'melodic_string_abc' not in slp_df.columns:
        slp_df['melodic_string_abc'] = ""
    if 'rhythm_string_abc' not in slp_df.columns:
        slp_df['rhythm_string_abc'] = ""
    if 'corpus' not in slp_df.columns:
        slp_df['corpus'] = 'SLP'
    if 'metadata_filename' not in slp_df.columns:
        slp_df['metadata_filename'] = slp_df.get('metadata_title', slp_df.index.astype(str))

    # ✅ Load rhythm mapping (via project helper for correct paths)
    rhythm_mapping = save_load.load_pickle(rhythm_mapping_path)

    # ✅ Prepare MXL file-matching dictionary
    mxl_files = {
        f.replace('.mxl', '').replace('--', '.').replace('-', '.'): os.path.join(mxl_folder, f)
        for f in os.listdir(mxl_folder) if f.endswith('.mxl')
    }

    for i, row in slp_df.iterrows():
        meta_filename = str(row['metadata_filename']).replace(' ', '')
        matched_mxl = mxl_files.get(meta_filename)

        if not matched_mxl:
            print(f"❌ No match for {meta_filename}")
            continue

        print(f"✅ Updating: {meta_filename} → {matched_mxl}")

        # Extract note durations (no rests), all TS, and real rest count
        rhythm_sequence, time_signatures, pause_count = extract_rhythm_and_time_from_mxl(matched_mxl)

        if rhythm_sequence is not None and time_signatures is not None:
            print(f"🎵 Extracted Rhythm ({meta_filename}): {rhythm_sequence}")
            print(f"🕒 Extracted Time Signatures: {time_signatures}")

            # Canonical mapping enforcement
            obs_slp   = _collect_durations_from_column([rhythm_sequence])
            canon_map = get_or_extend_canonical_map(obs_slp, rhythm_mapping_path)
            rhythm_ABC = convert_to_abc(rhythm_sequence, canon_map)

            print(f"🎼 Converted Rhythm ABC: {rhythm_ABC}")

            # Positional assignments (avoid .loc with possibly non-unique labels)
            ts_list = time_signatures if time_signatures else re.findall(r'\d+/\d+', str(row.get('time_signature', '')))
            slp_df.at[i, 'time_signature'] = ', '.join(dict.fromkeys(ts_list)) if ts_list else None
            slp_df.at[i, 'rhythm_string']      = rhythm_sequence
            slp_df.at[i, 'rhythm_string_abc']  = rhythm_ABC
            slp_df.at[i, 'rhythm_length']      = len(rhythm_sequence)
            slp_df.at[i, 'pause_count']        = int(pause_count)
            slp_df.at[i, 'has_pauses']         = "YES" if pause_count > 0 else "NO"
        else:
            print(f"⚠️ Warning: No rhythm data found for {meta_filename}")
    # else:
    #     print(f"❌ No match for {meta_filename}")

    return slp_df