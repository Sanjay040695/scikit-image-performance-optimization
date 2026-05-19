"""
Shared benchmark utilities. Used by baseline_colab.ipynb,
candidate_colab.ipynb and tests.ipynb to ensure identical timing
methodology across all three notebooks.

Methodology (per the take-home brief):
- High-precision timer: time.perf_counter()
- Warmup runs before measurement (default 2) to amortize JIT / cache
  / page-fault costs.
- Multiple measured runs (default 7) so we can report a central tendency
  AND a spread.
- Central tendency: MEDIAN (more robust to a single noisy run than mean).
- Spread: IQR (interquartile range) — robust, doesn't blow up on outliers.
"""

from __future__ import annotations

import json
import statistics
import time
from typing import Callable, Iterable, Sequence


def bench(
    fn: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    n_warmup: int = 2,
    n_measured: int = 7,
    label: str = "",
    verbose: bool = True,
) -> dict:
    """Run `fn(*args, **kwargs)` n_warmup + n_measured times, return stats.

    Returns a dict suitable for direct inclusion in reward.json:
        {"median": ..., "iqr": ..., "n_warmup": ..., "n_measured": ...,
         "runs": [...]}
    """
    if kwargs is None:
        kwargs = {}

    # Warmup — discard results.
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    # Measure.
    runs = []
    for _ in range(n_measured):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        runs.append(t1 - t0)

    runs_sorted = sorted(runs)
    median = statistics.median(runs_sorted)
    # IQR via the 25th / 75th percentile (linear interpolation if needed).
    q1 = _percentile(runs_sorted, 25.0)
    q3 = _percentile(runs_sorted, 75.0)
    iqr = q3 - q1

    if verbose:
        runs_str = ", ".join(f"{r:.4f}" for r in runs)
        print(f"[{label}] median={median:.4f}s  IQR={iqr:.4f}s  "
              f"min={min(runs):.4f}s  max={max(runs):.4f}s")
        print(f"    runs: [{runs_str}]")

    return {
        "median": float(median),
        "iqr": float(iqr),
        "n_warmup": int(n_warmup),
        "n_measured": int(n_measured),
        "runs": [float(r) for r in runs],
    }


def _percentile(sorted_data: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (matches numpy default behaviour)."""
    if not sorted_data:
        raise ValueError("empty data")
    if len(sorted_data) == 1:
        return float(sorted_data[0])
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return float(sorted_data[f])
    return float(sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f))


def write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def collect_environment() -> dict:
    """Collect environment info for the reward.json `environment` block."""
    import os
    import platform
    import sys

    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except FileNotFoundError:
        pass

    ram_gb = 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    ram_gb = round(kb / (1024.0 * 1024.0), 1)
                    break
    except FileNotFoundError:
        pass

    is_colab = os.path.exists("/content") or "COLAB_GPU" in os.environ
    runtime = "CPU" if is_colab else f"local ({platform.system()})"

    return {
        "cpu_model": cpu_model,
        "ram_gb": ram_gb,
        "python_version": sys.version.split()[0],
        "colab_runtime": runtime,
    }


def normalize_table(table: dict) -> dict:
    """Sort the columns of a regionprops_table dict so column ordering
    differences between baseline and candidate don't break np.allclose
    comparisons."""
    import numpy as np
    out = {}
    for k in sorted(table.keys()):
        out[k] = np.asarray(table[k])
    return out


def compare_tables(
    a: dict,
    b: dict,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> tuple[bool, dict]:
    """Compare two regionprops_table-style dicts column-by-column.

    Returns (ok, details).
    """
    import numpy as np

    details: dict = {"columns": {}, "missing_in_a": [], "missing_in_b": []}
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    details["missing_in_a"] = sorted(b_keys - a_keys)
    details["missing_in_b"] = sorted(a_keys - b_keys)
    ok = not details["missing_in_a"] and not details["missing_in_b"]

    for k in sorted(a_keys & b_keys):
        av = np.asarray(a[k])
        bv = np.asarray(b[k])
        col = {"shape_a": list(av.shape), "shape_b": list(bv.shape)}
        if av.shape != bv.shape:
            col["match"] = False
            col["reason"] = "shape mismatch"
            ok = False
            details["columns"][k] = col
            continue

        # Integer columns: must match exactly.
        if np.issubdtype(av.dtype, np.integer) and np.issubdtype(bv.dtype, np.integer):
            equal = bool(np.array_equal(av, bv))
            col["match"] = equal
            if not equal:
                col["reason"] = "integer values differ"
                col["max_abs_diff"] = int(np.abs(av - bv).max())
                ok = False
        else:
            close = bool(np.allclose(av, bv, rtol=rtol, atol=atol,
                                     equal_nan=True))
            col["match"] = close
            diff = np.abs(av.astype(np.float64) - bv.astype(np.float64))
            col["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
            if not close:
                col["reason"] = "float values differ beyond tolerance"
                ok = False
        details["columns"][k] = col

    return ok, details
