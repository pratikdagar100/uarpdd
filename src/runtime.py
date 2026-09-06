"""Thread-pool sizing for the machine the app is running on.

Every heavy step in the pipeline is thread-pooled, and the libraries' shared
default -- "use every logical CPU" -- is the wrong call on Apple Silicon,
where the efficiency cores run at a fraction of the performance cores'
throughput. Measured on an M2 (4 performance + 4 efficiency cores) over this
project's 254k x 23 training block:

    threads               RandomForest   HistGradientBoosting
    OMP=8, n_jobs=-1           13.2 s                  3.7 s
    OMP=4, n_jobs=4            19.5 s                  2.7 s
    OMP=4, n_jobs=8            13.8 s                  2.8 s

The two pools want opposite things. HistGradientBoosting is OpenMP code with
a barrier at every split, so it runs at the speed of its slowest thread and
is ~25% faster confined to the performance cores. A random forest fits
independent trees with no barrier between them, so the efficiency cores add
real throughput and cutting the pool to four workers costs ~45%.

So: pin the OpenMP/BLAS pools to the performance cores, and let joblib keep
every logical CPU. On every other platform the library defaults are already
reasonable and this module leaves them alone.

configure_runtime() must run BEFORE numpy, scikit-learn or streamlit are
imported -- OpenMP reads OMP_NUM_THREADS once, when its runtime loads.
"""
from __future__ import annotations

import os
import platform
import subprocess

# The thread-count variables the numeric stack reads at import time.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_state: dict | None = None


def _sysctl_int(key: str) -> int | None:
    try:
        proc = subprocess.run(["sysctl", "-n", key], capture_output=True,
                              text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def performance_cores() -> int | None:
    """Logical CPUs in Apple Silicon's performance cluster, else None.

    perflevel0 is the fast cluster on every Apple Silicon part shipped so
    far; the key is absent on Intel Macs and on non-Darwin hosts.
    """
    if not is_apple_silicon():
        return None
    n = _sysctl_int("hw.perflevel0.logicalcpu")
    return n if n and n > 0 else None


def configure_runtime() -> dict:
    """Size the thread pools for this host and return what was decided.

    Idempotent, and never overrides a thread count the user set themselves.
    """
    global _state
    if _state is not None:
        return _state

    logical = os.cpu_count() or 1
    perf = performance_cores()
    barrier_threads = perf or logical

    preset = {var: os.environ[var] for var in THREAD_VARS if var in os.environ}
    for var in THREAD_VARS:
        os.environ.setdefault(var, str(barrier_threads))

    _state = {
        "platform": f"{platform.system()} {platform.machine()}",
        "logical_cpus": logical,
        "performance_cores": perf,
        "omp_threads": int(os.environ["OMP_NUM_THREADS"]),
        "joblib_workers": logical,
        "preset_by_user": preset,
    }
    return _state


def joblib_workers() -> int:
    """Worker count for joblib-parallel estimators.

    These run independent tasks with no barrier between them, so unlike the
    OpenMP pool they should span every logical CPU.
    """
    return (_state or configure_runtime())["joblib_workers"]


def describe() -> dict:
    """The active configuration, for display and debugging."""
    return dict(_state or configure_runtime())
