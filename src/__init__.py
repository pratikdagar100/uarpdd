"""Rainfall early-warning pipeline.

Importing this package sizes the numeric stack's thread pools for the host
(see `runtime`). It has to happen on import rather than in each entry point:
OpenMP fixes its thread count when its runtime loads, which `src.train` would
trigger via scikit-learn before any caller got the chance.

An entry point that imports a third-party numeric library *before* this
package -- app.py imports Streamlit first -- must still call
`runtime.configure_runtime()` itself, ahead of that import.
"""
from .runtime import configure_runtime as _configure_runtime

_configure_runtime()
