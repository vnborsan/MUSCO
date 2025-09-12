"""
Educational Filters package.

Lazy-load submodules to avoid circular imports and heavy deps at import time.
"""

from __future__ import annotations
import importlib

# Public submodules you want to expose (strings only; no eager imports here)
__all__ = [
    "dataset_conversion",
    "evaluation_descriptors",
    "filter_df",           # keep only if you actually have this file
    "filter_label_utils",
    "match_functions",
    "prepare_df",
    "prepare_df_patch",
    "save_load",
    "vrf",
    "ifilters",
    "rfilters",
    "summarize",
]

__version__ = "0.1.0"

def __getattr__(name):  # PEP 562: lazy attribute access
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")