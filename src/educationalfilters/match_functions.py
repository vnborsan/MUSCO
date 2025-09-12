"""
match_functions.py

Pattern matching utilities for melody and rhythm against prebuilt
suffix arrays (pysdsl) plus descriptor gating.

Highlights
----------
- Uses package-relative imports (from . import save_load)
- Centralized path handling via save_load.load_pickle (data/processed by default)
- `melody_match` returns [id, phrase_bar, position] (bugfix)
- Clear semantics: None = filter not requested; [] = requested but no matches
"""

from __future__ import annotations

from typing import List, Tuple, Sequence, Optional, Dict, Any, Union, Set
from pathlib import Path
import pysdsl

from . import save_load  # resolves pickles under data/processed via your helpers


# ---------------------------
# Rhythm helpers
# ---------------------------
def convert_rhythm_to_ABC(query: Union[str, Sequence[float]], rhythm_mapping: Dict[float, str]) -> str:
    """
    Convert a rhythm query to ABC compact notation.

    Parameters
    ----------
    query : str | list[float]
        - If str (e.g., 'adad'), return as-is.
        - If list[float] (e.g., [0.5, 1.0]), map via rhythm_mapping.
    rhythm_mapping : dict[float,str]
        Mapping from duration -> ABC letter (e.g., {0.5:'a', 1.0:'d', 2.0:'e'}).

    Returns
    -------
    str

    Examples
    --------
    >>> convert_rhythm_to_ABC([0.5, 1.0], {0.5:'a', 1.0:'d'})
    'ad'
    """
    if isinstance(query, str):
        return query
    if isinstance(query, (list, tuple)):
        rm = {float(k): v for k, v in rhythm_mapping.items()}
        return "".join(rm.get(float(x), "?") for x in query)
    raise ValueError("Rhythm query must be either a string or a list/tuple of durations.")


def convert_rhythm_melodic_results(data: List[Any]) -> List[Any]:
    """
    Hook to post-process match triplets [id, phrase_bar, pos].
    Currently returns input unchanged (reserved for future tweaks).
    """
    return data


# ---------------------------
# Melody matching
# ---------------------------
def melody_match(
    melody: Optional[str],
    melody_ref: List[List[Any]],
    s_a: pysdsl.SuffixArrayBitcompressed,
    b_v: pysdsl.BitVector,
) -> Optional[List[List[Any]]]:
    """
    Match a melodic sequence against the suffix array.

    Parameters
    ----------
    melody : str | None
        Query (e.g., 'ABA'). If None/empty, returns None (meaning "no melodic filter").
    melody_ref : list[[id, phrase_bar]]
        Reference list whose index corresponds to phrase delimiter rank.
    s_a : pysdsl.SuffixArrayBitcompressed
    b_v : pysdsl.BitVector

    Returns
    -------
    list[[id, phrase_bar, position]] | None
        None → no melodic filtering requested.
        []   → requested melody but no matches found.
    """
    if not melody:
        return None  # no melodic filter requested

    matches: List[List[Any]] = []
    match_positions = s_a.locate(melody)
    if not match_positions:
        return []  # explicit "no matches"

    rnk = b_v.init_rank_1()
    slct = b_v.init_select_1()

    for pos in match_positions:
        phrase_idx = rnk.rank(pos)
        if phrase_idx >= len(melody_ref):  # guard
            continue
        b_id, ph_bar = melody_ref[phrase_idx][0], melody_ref[phrase_idx][1]
        melody_pos = pos if phrase_idx == 0 else pos - slct.select(phrase_idx)
        matches.append(convert_rhythm_melodic_results([b_id, ph_bar, melody_pos]))

    return matches


def filter_melody_results(
    melody_matches: List[List[Any]],
    phrase_nr: Optional[int],
    position_nr: Optional[int],
) -> List[List[Any]]:
    """
    Filter melody match triplets by phrase number (heuristic: last digit of ID) and/or position.

    Parameters
    ----------
    melody_matches : list[[id, phrase_bar, position]]
    phrase_nr : int | None
    position_nr : int | None
    """
    if melody_matches is None:
        return []

    positional_matches = []
    phrase_matches = []

    for m in melody_matches:
        melody_pos = m[2] if len(m) > 2 else None
        phrase = int(m[0][-1]) if m and isinstance(m[0], str) and m[0][-1].isdigit() else None

        if position_nr is not None:
            if melody_pos == position_nr:
                positional_matches.append(m)
        else:
            positional_matches = melody_matches

        if phrase_nr is not None and phrase is not None:
            if phrase == phrase_nr:
                phrase_matches.append(m)
        else:
            phrase_matches = melody_matches

    set1 = set(tuple(item) for item in positional_matches)
    set2 = set(tuple(item) for item in phrase_matches)
    return [list(t) for t in (set1 & set2)]


# ---------------------------
# Rhythm matching
# ---------------------------
def rhythm_match(
    rhythm: Optional[Union[str, Sequence[float]]],
    rhythm_ref: List[List[Any]],
    s_a_rhythm: pysdsl.SuffixArrayBitcompressed,
    b_v_rhythm: pysdsl.BitVector,
    rhythm_mapping: Dict[float, str],
) -> Optional[List[List[Any]]]:
    """
    Match a rhythmic pattern against the suffix array.

    Parameters
    ----------
    rhythm : str | list[float] | None
        ABC compact string (e.g., 'adad') OR list of durations (e.g., [0.5, 1.0]).
        None/empty ⇒ no rhythm filter requested.
    rhythm_ref : list[[id, phrase_bar]]
    s_a_rhythm : pysdsl.SuffixArrayBitcompressed
    b_v_rhythm : pysdsl.BitVector
    rhythm_mapping : dict[float,str]

    Returns
    -------
    list[[id, phrase_bar, position]] | None
        None → no rhythm filtering requested.
        []   → requested rhythm but no matches found.
    """
    if rhythm in (None, ""):
        return None

    matches: List[List[Any]] = []
    rhythm_ABC = convert_rhythm_to_ABC(rhythm, rhythm_mapping)

    match_positions = s_a_rhythm.locate(rhythm_ABC)
    if not match_positions:
        return []

    rnk = b_v_rhythm.init_rank_1()
    slct = b_v_rhythm.init_select_1()

    for pos in match_positions:
        phrase_idx = rnk.rank(pos)
        if phrase_idx >= len(rhythm_ref):
            continue
        b_id, ph_bar = rhythm_ref[phrase_idx][0], rhythm_ref[phrase_idx][1]
        rhy_pos = pos if phrase_idx == 0 else pos - slct.select(phrase_idx)
        matches.append(convert_rhythm_melodic_results([b_id, ph_bar, rhy_pos]))

    return matches


# ---------------------------
# Descriptor gating + finders
# ---------------------------
def _descriptor_ids_from_query(
    target_keys: Sequence[Tuple[str, Union[str, Sequence[str]]]],
    descriptor_pickle_path: Union[str, Path],
) -> Set[str]:
    """
    Evaluate a descriptor query (OR within key, AND across keys) and return matching IDs.
    Uses save_load.load_pickle so relative filenames resolve under data/processed.
    """
    descriptor_dict = save_load.load_pickle(descriptor_pickle_path)

    # Seed with first term
    first_key, first_val = target_keys[0]
    if isinstance(first_val, (list, tuple, set)):
        current = set().union(*(set(descriptor_dict.get((first_key, v), [])) for v in first_val))
    else:
        current = set(descriptor_dict.get((first_key, first_val), []))

    # AND across keys, OR within a key
    for key, value in target_keys[1:]:
        if isinstance(value, (list, tuple, set)):
            ids_for_key = set().union(*(set(descriptor_dict.get((key, v), [])) for v in value))
        else:
            ids_for_key = set(descriptor_dict.get((key, value), []))
        current &= ids_for_key
        if not current:
            break

    return current


def descriptor_match(
    target_keys: Sequence[Tuple[str, Union[str, Sequence[str]]]],
    descriptor_pickle_path: Union[str, Path],
    melodies: Optional[List[List[Any]]],
    rhythms: Optional[List[List[Any]]],
) -> Set[str]:
    """
    Descriptor query optionally intersected with melody/rhythm matches.

    Returns
    -------
    set[str]
        Final set of matching IDs.

    Notes
    -----
    - If melody or rhythm was REQUESTED but returned [], we return an empty set.
    """
    if melodies == []:
        print("⚠️ Melody was attempted but no matches found — returning 0 results.")
        return set()
    if rhythms == []:
        print("⚠️ Rhythm was attempted but no matches found — returning 0 results.")
        return set()

    result_ids = _descriptor_ids_from_query(target_keys, descriptor_pickle_path)

    if melodies is not None:
        result_ids &= {m[0] for m in melodies}
    if rhythms is not None:
        result_ids &= {r[0] for r in rhythms}

    return result_ids


def find_all(
    melody: Optional[str],
    rhythm: Optional[Union[str, Sequence[float]]],
    target_keys: Sequence[Tuple[str, Union[str, Sequence[str]]]],
    descriptor_pickle_path: Union[str, Path],
    melody_ref: List[List[Any]],
    rhythm_ref: List[List[Any]],
    sa_mel: pysdsl.SuffixArrayBitcompressed,
    bv_mel: pysdsl.BitVector,
    sa_rhythm: pysdsl.SuffixArrayBitcompressed,
    bv_rhythm: pysdsl.BitVector,
    rhythm_mapping: Dict[float, str],
) -> Set[str]:
    """
    Full query: (optional) melody + (optional) rhythm + descriptors.

    Returns
    -------
    set[str]
        IDs that satisfy all requested constraints.
    """
    melodies = melody_match(melody, melody_ref, sa_mel, bv_mel) if melody else None
    rhythms  = rhythm_match(rhythm, rhythm_ref, sa_rhythm, bv_rhythm, rhythm_mapping) if rhythm else None

    print(f"Melody matches: {melodies if melodies is not None else 'None (not requested)'}")
    print(f"Rhythm matches: {rhythms  if rhythms  is not None else 'None (not requested)'}")

    match_ids = descriptor_match(target_keys, descriptor_pickle_path, melodies, rhythms)

    if match_ids:
        preview = sorted(match_ids)
        print(f"Matched IDs ({len(match_ids)}): {preview[:10]}{'...' if len(preview) > 10 else ''}")
    else:
        print("No matches were found.")

    return match_ids


# ---------------------------
# “Melody + descriptors only” convenience
# ---------------------------
def descriptor_match_new(
    target_keys: Sequence[Tuple[str, Union[str, Sequence[str]]]],
    descriptor_pickle_path: Union[str, Path],
    melodies: Optional[List[List[Any]]],
    rhythms: Optional[List[List[Any]]] = None,
):
    """
    Return matched MELODY entries (not just IDs), filtered by descriptor query.
    If melodies is None (no melodic filter), returns sorted IDs.
    """
    result_ids = _descriptor_ids_from_query(target_keys, descriptor_pickle_path)
    if melodies is None:
        return sorted(result_ids)
    return [entry for entry in melodies if entry[0] in result_ids]


def find_all_new(
    melody: str,
    target_keys: Sequence[Tuple[str, Union[str, Sequence[str]]]],
    descriptor_pickle_path: Union[str, Path],
    melody_ref: List[List[Any]],
    sa_mel: pysdsl.SuffixArrayBitcompressed,
    bv_mel: pysdsl.BitVector,
):
    """
    Convenience: match a melodic query and then gate by descriptors, returning melody entries.
    """
    melodies = melody_match(melody, melody_ref, sa_mel, bv_mel) or []
    match_list = descriptor_match_new(target_keys, descriptor_pickle_path, melodies, None)

    if match_list:
        print(f"Matched {len(match_list)} entries.")
    else:
        print("No matches were found.")
    return match_list