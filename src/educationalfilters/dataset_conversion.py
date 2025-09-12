"""
dataset_conversion.py

Convert annotated datasets to:
- a global melodic string and a global rhythmic string
- suffix arrays + bit vectors (pysdsl) for fast pattern search
- a descriptor dictionary: {(category, value) -> [IDs]}
- reference indices to map suffix-array hits back to (ID, phrase index)

All files are saved under data/processed/ by default via educationalfilters.save_load.
You can override the data root with ENV var: EDUFILT_DATA_DIR=/path/to/data-root

Examples
--------
>>> import pandas as pd
>>> from educationalfilters import dataset_conversion as dc
>>> df = pd.DataFrame({
...     "metadata_filename": ["song1", "song2"],
...     "melodic_intervals": ["A5 B5 A5", "C5 D5 E5"],
...     "rhythm_string_abc": ["adad", "aede"],
...     "corpus": ["Ciciban", "SLP"],
...     "range_check": ["PRE", "PRE_PLUS"],
... })
>>> desc, mel_ref, rhy_ref, sa_m, bv_m, sa_r, bv_r = dc.dataset_conversion(
...     df, selected_categories=["corpus", "range_check"]
... )
Dictionary saved to .../data/processed/descriptor_dict.pickle
Successfully converted 2 sequences
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Any, Sequence, Union
import traceback
import pandas as pd
import pysdsl

from . import save_load  # centralised paths + I/O


# -----------------------------
# Descriptor dictionary
# -----------------------------
def create_descriptor_dict(df: pd.DataFrame, selected_categories: List[str]) -> Dict[Tuple[str, str], List[str]]:
    """
    Initialise an empty descriptor dictionary keyed by (category, element).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the categories.
    selected_categories : list of str
        Columns to extract unique elements from.

    Returns
    -------
    dict
        Keys = (category, element), values = empty lists.

    Examples
    --------
    >>> df = pd.DataFrame({"corpus": ["Ciciban", "SLP"]})
    >>> create_descriptor_dict(df, ["corpus"])
    {("corpus", "Ciciban"): [], ("corpus", "SLP"): []}
    """
    descriptor_list: List[Tuple[str, str]] = []
    for category in selected_categories:
        unique_elements = set(df[category])
        for element in unique_elements:
            descriptor_list.append((category, str(element)))
    return {key: [] for key in descriptor_list}


# -----------------------------
# String preparation and encoding
# -----------------------------
def prepare_string(string: str) -> str:
    """
    Remove digits, accidentals (#, b) and spaces from a melodic string,
    and append a terminating '$'.

    Parameters
    ----------
    string : str
        Input melodic string (e.g. "A5 B5 A5").

    Returns
    -------
    str
        Cleaned string (e.g. "ABA$").

    Examples
    --------
    >>> prepare_string("A5B5A5")
    'ABA$'
    """
    phrase = string.replace(" ", "")
    # remove digits and accidentals
    for ele in "0123456789#b":
        phrase = phrase.replace(ele, "")
    return phrase + "$"


def encode_string(full_phrase_string: str):
    """
    Encode a string into suffix array and bit vector.

    Parameters
    ----------
    full_phrase_string : str
        Input melodic or rhythmic string with '$' as sequence delimiter.

    Returns
    -------
    (SuffixArrayBitcompressed, BitVector)

    Examples
    --------
    >>> sa, bv = encode_string("ABABA$")
    >>> isinstance(bv, pysdsl.BitVector)
    True
    """
    s_a = pysdsl.SuffixArrayBitcompressed(full_phrase_string)
    bit_vector = [0 if ele != "$" else 1 for ele in full_phrase_string]
    b_v = pysdsl.BitVector(bit_vector)
    return s_a, b_v


# -----------------------------
# Dataset conversion
# -----------------------------
def dataset_conversion(
    df: pd.DataFrame,
    selected_categories: List[str],
    descriptor_filename: Union[str, "save_load.PathLike"] = "descriptor_dict.pickle",
    sa_bv_filenames: Sequence[Union[str, "save_load.PathLike"]] = ("melody.sa", "melody.bv", "rhythm.sa", "rhythm.bv"),
):
    """
    Convert dataset into descriptors, references, and SA/BV encodings.

    Steps
    -----
    1) Initialise descriptor dictionary
    2) Iterate rows, fill descriptors + references
    3) Build global melodic and rhythmic strings
    4) Encode into SA/BV
    5) Save descriptor dict and SA/BV via save_load (→ data/processed/)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns:
        - "metadata_filename" (unique ID per song/row)
        - "melodic_intervals" (melodic letters like 'ABA' or 'A5 B5 A5')
        - "rhythm_string_abc" (ABC rhythm letters, e.g., 'adad')
        plus any selected category columns (e.g., 'corpus', 'range_check', ...)
    selected_categories : list of str
        Columns used to build descriptor dictionary.
    descriptor_filename : str or Path-like, optional
        Filename for saved descriptor dict (default "descriptor_dict.pickle").
        Resolved under data/processed/ by save_load.
    sa_bv_filenames : sequence of 4 str/Path-like, optional
        Filenames for melody SA/BV and rhythm SA/BV:
        (mel_sa, mel_bv, rhy_sa, rhy_bv). Saved under data/processed/.

    Returns
    -------
    (descriptor_dict, melody_ref, rhythm_ref, sa_mel, bv_mel, sa_rhythm, bv_rhythm)

    Examples
    --------
    >>> # Minimal real-field example
    >>> desc, mel_ref, rhy_ref, sa_m, bv_m, sa_r, bv_r = dataset_conversion(
    ...     df, selected_categories=["corpus", "range_check"]
    ... )
    """
    melody_ref: List[List[Any]] = []
    rhythm_ref: List[List[Any]] = []
    full_phrase_string = ""
    full_rhythmic_string = ""
    melodic_number = 0
    rhythmic_number = 0
    descriptor_dict = create_descriptor_dict(df, selected_categories)

    for idx, row in df.iterrows():
        try:
            phrase = str(row["melodic_intervals"])
            rhythm = str(row["rhythm_string_abc"])
            ID = row["metadata_filename"]

            # fill descriptor dict
            for s in selected_categories:
                descriptor_dict[(s, str(row[s]))].append(ID)

            # references
            melody_ref.append([ID, melodic_number])
            rhythm_ref.append([ID, rhythmic_number])
            melodic_number += 1
            rhythmic_number += 1

            # strings
            phrase_string = prepare_string(phrase)   # remove digits/#/b + add '$'
            rhythmic_string = rhythm.strip() + "$"   # ABC rhythm + '$'

            # concat global sequences
            full_phrase_string += phrase_string
            full_rhythmic_string += rhythmic_string

        except Exception as e:
            print(f"Error in row {idx} (ID: {row.get('metadata_filename','<no ID>')}): {e}")
            traceback.print_exc()

    # Encode
    s_a_mel, b_v_mel = encode_string(full_phrase_string)
    s_a_rhythm, b_v_rhythm = encode_string(full_rhythmic_string)

    # Save SA/BV using centralised paths (→ data/processed/)
    if sa_bv_filenames is not None:
        mel_sa, mel_bv, rhy_sa, rhy_bv = sa_bv_filenames
        save_load.save_sa_bv(s_a_mel, b_v_mel, mel_sa, mel_bv)
        save_load.save_sa_bv(s_a_rhythm, b_v_rhythm, rhy_sa, rhy_bv)

    # Save descriptor dictionary (→ data/processed/)
    save_load.save_pickle(descriptor_dict, descriptor_filename)

    print(f"Successfully converted {melodic_number} sequences")

    return descriptor_dict, melody_ref, rhythm_ref, s_a_mel, b_v_mel, s_a_rhythm, b_v_rhythm


# -----------------------------
# Helpers

def prepare_melodic_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'melodic_string_relative' to a compact letter-coded 'melodic_intervals'.
    '0'->'a', '1'->'b', ... (up to 120 for safety).
    """
    number_to_letter = {str(i): chr(97 + i) for i in range(120)}

    def translate_intervals(relative_string):
        if pd.isna(relative_string):
            return None
        clean = str(relative_string).replace('-', '').replace(' ', '')
        return ''.join(number_to_letter.get(ch, ch) for ch in clean)

    if 'melodic_string_relative' in df.columns:
        df = df.copy()
        df['melodic_intervals'] = df['melodic_string_relative'].apply(translate_intervals)
    else:
        print("Column 'melodic_string_relative' not found in DataFrame.")
    return df


def process_and_merge_dfs(dfs, labels, rhythm_mapping, filter_function, keep_cols=None):
    """
    Apply a filter function per-DF, tag with 'corpus' label, and concatenate.
    Compatible with the old API: filter_function(df, rhythm_mapping, label) -> (df_out, counts)

    Parameters
    ----------
    dfs : list[pd.DataFrame]
    labels : list[str]
    rhythm_mapping : dict
    filter_function : callable
        (df, rhythm_mapping, label) -> (df_out, counts)
    keep_cols : list[str] or None
        If provided, the merged frame is reindexed to these columns (missing filled with NA).

    Returns
    -------
    merged_df : pd.DataFrame
    all_counts : dict[label -> counts]
    """
    combined_rows = []
    all_counts = {}

    for df, label in zip(dfs, labels):
        out_df, counts = filter_function(df, rhythm_mapping, label)
        if 'corpus' not in out_df.columns:
            out_df = out_df.copy()
            out_df['corpus'] = label
        combined_rows.append(out_df)
        all_counts[label] = counts

    merged_df = pd.concat(combined_rows, ignore_index=True)

    if keep_cols is not None:
        merged_df = merged_df.reindex(columns=keep_cols)

    return merged_df, all_counts