"""
evaluation_descriptors.py

Descriptor matching and evaluation against corpora (Ciciban vs SLP).

Functions
---------
- descriptor_match : run a query against a descriptor_dict pickle
- evaluate_descriptors_only : evaluate queries across two corpora

Examples
--------
>>> from educationalfilters import evaluation_descriptors as ed
>>> q = [("corpus", "Ciciban"), ("range_check", "PRE")]
>>> ids = ed.descriptor_match(q, "descriptor_dict_ciciban.pickle")
>>> isinstance(ids, set)
True
"""

from __future__ import annotations
from typing import List, Tuple, Union, Sequence, Set, Dict
import pandas as pd

from . import save_load

# Types
QueryTerm = Tuple[str, Union[str, Sequence[str]]]
Query = List[QueryTerm]


def _normalize_value(v: Union[str, Sequence[str]]) -> List[str]:
    """Normalize a query value into a list of strings."""
    if isinstance(v, str):
        return [v]
    return list(v)


def descriptor_match(query: Query, descriptor_pickle_path: Union[str, 'Path']) -> Set[str]:
    """
    Match a descriptor query against a precomputed descriptor dictionary.

    Parameters
    ----------
    query : list of (category, value or [values])
        Values are ORed within a category; categories are ANDed.
    descriptor_pickle_path : str or Path
        Path or filename to the descriptor_dict pickle (resolved via save_load).

    Returns
    -------
    set of str
        Set of matching song IDs.

    Examples
    --------
    >>> q = [("corpus", "Ciciban"), ("range_check", ["PRE", "PRE_PLUS"])]
    >>> ids = descriptor_match(q, "descriptor_dict_ciciban.pickle")
    >>> isinstance(ids, set)
    True
    """
    descriptor_dict: Dict[Tuple[str, str], Sequence[str]] = save_load.load_pickle(descriptor_pickle_path)

    current: Union[Set[str], None] = None
    for category, value in query:
        values = _normalize_value(value)
        cat_ids: Set[str] = set()
        for v in values:
            cat_ids |= set(descriptor_dict.get((category, str(v)), []))

        if current is None:
            current = cat_ids
        else:
            current &= cat_ids

        if not current:
            return set()

    return current or set()


def evaluate_descriptors_only(
    QUERIES: List[Query],
    ciciban_list: Sequence,
    slp_list: Sequence,
) -> pd.DataFrame:
    """
    Evaluate queries against Ciciban and SLP descriptor dictionaries.

    Assumes:
    - ciciban_list[0] = path to Ciciban pickle
    - ciciban_list[-1] = number of Ciciban items
    - slp_list[0] = path to SLP pickle
    - slp_list[-1] = number of SLP items

    Parameters
    ----------
    QUERIES : list of Query
        List of descriptor queries.
    ciciban_list : sequence
        [pickle_path, ..., N] where N = number of Ciciban items.
    slp_list : sequence
        [pickle_path, ..., N] where N = number of SLP items.

    Returns
    -------
    pandas.DataFrame
        One row per query with TP/FP/FN/TN and precision/recall/F1.

    Examples
    --------
    >>> queries = [[("corpus", "Ciciban"), ("range_check", "PRE")]]
    >>> ciciban = ["descriptor_dict_ciciban.pickle", 123]
    >>> slp = ["descriptor_dict_slp.pickle", 402]
    >>> df = evaluate_descriptors_only(queries, ciciban, slp)
    >>> list(df.columns)[:3]
    ['Query Index', 'Query', 'TP']
    """
    results = []

    ciciban_pickle = ciciban_list[0]
    slp_pickle = slp_list[0]
    len_ciciban = ciciban_list[-1]
    len_slp = slp_list[-1]

    for q_index, query in enumerate(QUERIES):
        retrieved_ciciban = descriptor_match(query, ciciban_pickle)
        retrieved_slp = descriptor_match(query, slp_pickle)

        TP = len(retrieved_ciciban)
        FN = len_ciciban - TP
        FP = len(retrieved_slp)
        TN = len_slp - FP

        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        results.append({
            "Query Index": q_index,
            "Query": query,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1 Score": round(f1, 3),
        })

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define some realistic queries based on your descriptors
    target_queries = [
        # Ciciban corpus, within pre-approved range/jumps
        [
            ("corpus", "Ciciban"),
            ("range_check", ["PRE", "PRE_PLUS"]),
            ("interval_jumps_check", ["PRE", "PRE_PLUS"]),
        ],
        # SLP corpus, strict PRE check
        [
            ("corpus", "SLP"),
            ("range_check", "PRE"),
            ("interval_jumps_check", "PRE"),
        ],
        # Songs with limited ambitus
        [
            ("ambitus_interval", ["FIFTH", "SIXTH"]),
        ],
        # Ciciban songs that exceeded recommended ambitus
        [
            ("corpus", "Ciciban"),
            ("ambitus_interval", "ABOVE EIGHTH"),
        ],
        # Both corpora, major mode
        [
            ("corpus", ["Ciciban", "SLP"]),
            ("mode", "major"),
        ],
        # Folk (SLP) songs that passed both filters and limited ambitus
        [
            ("corpus", "SLP"),
            ("range_check", "PRE"),
            ("interval_jumps_check", "PRE"),
            ("ambitus_interval", ["FIFTH", "SIXTH"]),
        ],
    ]

    # Example lists (replace with your actual pickles + counts)
    ciciban_list = ["data/processed/descriptor_dict_ciciban.pickle", 123]
    slp_list = ["data/processed/descriptor_dict_slp.pickle", 402]

    print("Running evaluation on example queries...")
    df = evaluate_descriptors_only(target_queries, ciciban_list, slp_list)
    print(df)