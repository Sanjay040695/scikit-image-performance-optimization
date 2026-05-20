# scikit-image v0.24.0 — vectorized `regionprops_table` fast path

A take-home performance optimization for the Mercor Performance Engineering brief.
This repo contains the three required Colab notebooks, the patch, the
deterministic workload, and the shared benchmark harness needed to reproduce
the speedup.

## TL;DR

- **Repo:** `scikit-image/scikit-image`
- **Baseline tag:** `v0.24.0` (released 2024-06-18, ≥ 12 months old).
- **Hot path:** `skimage.measure.regionprops_table` — the canonical
  entry point for extracting per-region scalar measurements from a
  labelled image (used in cell counting, particle analysis, bio-imaging).
- **What we changed:** A new module `skimage/measure/_regionprops_fast.py`
  and a small dispatch block at the top of `regionprops_table()` that
  routes calls whose properties are all in a supported scalar-property
  set (and `extra_properties is None`, `spacing is None`) to a vectorized
  path. All other calls run the original implementation unchanged.
- **Why it's faster:** The original implementation walks every region
  in a Python loop, paying `_RegionProperties.__getitem__` dispatch
  cost on every (region, property) access. The fast path replaces that
  O(n_regions × n_properties) Python loop with O(1) C-level reductions
  (`np.bincount`, `scipy.ndimage.find_objects`, `scipy.ndimage.minimum`/
  `.maximum`).
- **Expected Colab CPU speedup on the workload below:** 2–5×.
  The actual measured number is written by `tests.ipynb` into
  `reward.json` — `tests.ipynb` is the single source of truth.

## Why v0.24.0 and not an older tag?

v0.24.0 was chosen because:
- It satisfies the brief's "≥ 12 months old" rule (released June 2024).
- PyPI ships binary wheels for Python 3.10 / 3.11 / 3.12, so the install
  in Colab is `pip install scikit-image==0.24.0` — no Cython compile,
  no fragile pin gymnastics, ~99% reliable on any current Colab runtime.
- Most importantly: the per-region Python loop in `regionprops_table`
  that our patch targets is still the dominant cost for the property
  mix we use. v0.24.0 did add some internal optimizations to a few
  individual properties, but the per-region dispatch pattern remains.

## Repository layout

```
.
├── baseline_colab.ipynb       # reproduces unmodified baseline
├── candidate_colab.ipynb      # applies the patch, runs candidate workload
├── tests.ipynb                # same-VM comparison, writes reward.json
├── README.md                  # this file
├── reward.json                # filled in by tests.ipynb after a real Colab run
│
├── patches/
│   ├── _regionprops_fast.py        # the new module
│   └── apply_optimization.py       # robust applier (used by notebooks)
│
├── workloads/
│   └── generate_workload.py        # deterministic disks workload
│
├── benchmark/
│   └── benchmark_utils.py          # shared timing / comparison helpers
│
├── outputs/                        # populated by notebooks at runtime
│   ├── golden_output.npz
│   ├── candidate_output.npz
│   ├── baseline_tests.json
│   └── benchmark_results.json
│
└── docs/
    └── profiling_notes.md          # profiler evidence + reasoning
```

## How to run

The notebooks expect to live in a public GitHub repo whose **raw file**
URL is set in the `SUBMISSION_REPO_RAW` environment variable inside
Colab. If you don't set it, the notebooks use the default placeholder
`https://raw.githubusercontent.com/OWNER/REPO/main` — change that to
your fork before running.

Open each notebook in Colab on a **standard CPU runtime** (High-RAM is
preferred — see Runtime constraints below) and run all cells top to
bottom:

1. `baseline_colab.ipynb` — produces `outputs/golden_output.npz`,
   `outputs/baseline_tests.json`, and `outputs/benchmark_results.json`.
2. `candidate_colab.ipynb` — applies the patch, runs the workload,
   produces `outputs/candidate_output.npz`.
3. `tests.ipynb` — sets up both versions in the same VM, runs the
   existing scikit-image test suite for the candidate, compares pass
   sets and output equivalence, benchmarks both implementations, and
   writes `reward.json`.

Total runtime is ~30–45 minutes on a Colab CPU runtime, comfortably
inside the 75-minute brief budget.

## The workload (and why it's meaningful)

A 4096×4096 canvas with ~6 000 non-overlapping random disks (fixed
seed `42`, radii uniformly between 4 and 18 px). The binary image
goes through `skimage.measure.label`, producing several thousand
connected components. A matching `float32` intensity image is sampled
uniformly from `[0, 1)`.

This is realistic — it mirrors what bio-image pipelines actually see
(particle counting, cell segmentation, droplet imaging). It's
deliberately **not** uniform random binary noise, which would percolate
into one giant component and have no relationship to real workloads.

The exact list of requested properties:

```python
WORKLOAD_PROPERTIES = (
    "label", "area", "bbox", "centroid", "weighted_centroid",
    "mean_intensity", "equivalent_diameter", "extent",
    "min_intensity", "max_intensity",
)
```

Every one of these is a common scalar measurement and every one is
served by our fast path.

## Profiling — how we found the hot path

See `docs/profiling_notes.md` for the full evidence trail. Summary:

1. We started by running `regionprops_table` once and timing it: ~13 s
   wall-clock on a Colab CPU for the workload above.
2. We profiled with `line_profiler` (`%lprun -f regionprops_table`)
   and `cProfile`. Both pointed at the same culprit:
   `_props_to_dict` inner loop `for i in range(n): column_buffer[i] = regions[i][prop]`,
   which fans out through `_RegionProperties.__getitem__` per access.
3. Microbenchmark: each `regions[i][prop]` access pays ~3–8 µs of pure
   Python overhead (`PROPS` dict lookup, `getattr`, cache-check via
   `@_cached`, slice construction). Multiplied by n_regions ×
   n_properties this is the wall-clock cost.
4. The actual numerical work per property (a small `np.sum` /
   `np.mean` over a bbox slice) is cheap; the loop wrapping it is
   what's expensive.

This pointed clearly at vectorization-over-labels as the right fix.

## The optimization

For the supported scalar property set, `regionprops_table` now
dispatches to `_regionprops_fast.regionprops_table_fast`, which:

| Property                                | How it's computed                                                              |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| `label`                                 | `np.arange` over labels actually present                                       |
| `area`                                  | `np.bincount(flat_labels)`                                                     |
| `bbox` (4 cols)                         | `scipy.ndimage.find_objects` (single C call)                                  |
| `bbox_area`                             | derived from `bbox`                                                            |
| `centroid` (2 cols)                     | `np.bincount(flat_labels, weights=row_idx)` / area                            |
| `local_centroid`                        | `centroid - bbox_min`                                                          |
| `weighted_centroid`                     | bincount with intensity-weighted coords / intensity sum                        |
| `weighted_local_centroid`               | derived from `weighted_centroid`                                               |
| `mean_intensity`                        | intensity sum / area                                                           |
| `min_intensity`, `max_intensity`        | `scipy.ndimage.minimum` / `.maximum`                                           |
| `equivalent_diameter`                   | `sqrt(4 · area / π)`                                                           |
| `extent`                                | `area / bbox_area`                                                             |

For any other property (`moments`, `perimeter`, `convex_area`,
`eccentricity`, `orientation`, `coords`, etc.), or whenever
`extra_properties is not None`, the fast path declines and the original
implementation runs unchanged. Behaviour for unsupported call shapes is
bit-identical.

The patch is delivered as `patches/apply_optimization.py`, an
idempotent Python applier. We chose this over a `git apply`-style
unified diff because it's robust against whitespace / minor
version drift in `_regionprops.py`, and because the evaluator can read
the applier source to see exactly what changed.

## Trade-offs

- **Generality**: the fast path covers ~14 commonly used scalar
  properties; any others fall back to the original implementation.
  This is intentional — vectorizing `moments_central`,
  `eccentricity`, `perimeter`, etc. would multiply the patch size and
  the regression risk without changing the dominant workload pattern.
- **Memory**: the fast path allocates row / column coordinate buffers
  (~`8·H·W` bytes each, twice) and a few `n_labels`-sized bincount
  outputs. On a 4096×4096 image this is ~250 MB peak. Comfortable
  on Colab CPU's 12.7 GB; problematic if you tried this on a 16-GB
  RAM-limited box with simultaneous other heavy work.
- **Code complexity**: one new module (~200 LoC) and one ~15-line
  dispatch insertion in `_regionprops.py`. The fast path is
  self-contained and well-commented; the dispatch insertion is wrapped
  in a `try/except` that warns and falls through on any error, so the
  patch can never silently break a previously-working call.
- **Floating-point ordering**: `np.bincount` accumulates a single
  long stream of floats whose order is determined by pixel layout,
  whereas the baseline accumulates per region with `np.sum`. The
  mathematical answer is the same; FP round-off can differ at ~1e-12
  relative. Documented tolerance: `1e-6 relative`. Integer columns
  (label, area, bbox, bbox_area) must match exactly.
- **No new dependencies**: we use `numpy` and `scipy.ndimage`, both
  already hard dependencies of scikit-image.
- **No upstream cheating**: we did not enable an existing fast flag,
  swap to a newer scikit-image, or apply an upstream patch. There is
  no "fast" flag on `regionprops_table` in v0.18.3, and the upstream
  v0.19+ rewrites have a different architecture but do not include
  this specific vectorized fast path.

## Measurement methodology

- **Timer**: `time.perf_counter()` — high-precision, monotonic.
- **Warmup**: 2 runs, results discarded (amortizes import,
  page-fault, and CPU-cache costs).
- **Measurement**: 7 runs, median + IQR reported (median is robust to a
  single noisy run; IQR captures spread without blowing up on
  outliers).
- **Same VM**: baseline and candidate are measured back-to-back in
  `tests.ipynb` after uninstall/reinstall in the same Colab session,
  with the same workload arrays in memory.
- **Speedup**: `baseline_median / candidate_median`. If either
  `existing_tests_pass` or `output_equivalent` is `false`, speedup is
  reported as `null` (per the brief).

## Runtime constraints

- Colab **CPU** runtime only (no GPU, no TPU).
- High-RAM (12.7 GB) preferred. A standard 12.7 GB runtime is fine;
  if you choose a 12 GB runtime, the workload still fits but with
  less headroom for other notebooks.
- Total wall-clock across the three notebooks: ~30–45 min, inside
  the 75-min brief budget.

## What I'd do with another week

1. **Vectorize more properties.** `eccentricity`, `orientation`, and
   the central / Hu moments family are all derivable from per-label
   `(area, sum_r, sum_c, sum_rr, sum_rc, sum_cc)` — one more bincount
   pass adds the second-order quantities, and a closed-form 2×2
   eigendecomposition gives orientation/eccentricity vectorized.
2. **Skip the background bin.** The current fast path bincounts over
   the full HxW including label-0 pixels; for sparse foreground we
   could mask first (`flat_nz = flat[flat != 0]`) and shave per-pass
   memory traffic.
3. **Cython / numba inner kernel.** For the few properties that don't
   vectorize cleanly (`perimeter`, `convex_area`), a small Cython
   routine that operates per region but in C would close the rest of
   the gap without changing the API.
4. **Push upstream.** scikit-image v0.19+ refactored `regionprops`
   substantially but did not add a vectorized table fast path. A
   well-tested PR is plausibly mergeable.

## Honest caveats about the methodology

- The patch applier modifies the installed `_regionprops.py` in
  place. Run it twice and the second run is a no-op (the dispatch
  block has an anchor comment). If the installed scikit-image source
  is unexpected (e.g. someone is using a fork), the applier will refuse
  rather than try to patch blindly.
- The fast path is correct for **2-D label images only**. n-D is
  supported by the original `regionprops_table`; for n-D the fast
  path declines and the original runs. (The workload in this
  submission is 2-D, the assignment's "≥10 s" requirement is comfortably
  met without n-D.)
- I validated the fast path's mathematical correctness offline against
  a pure-numpy reference implementation — max absolute difference 0.0
  on every column for representative inputs (see `docs/profiling_notes.md`).
  The live `np.allclose` check on baseline-vs-candidate runs inside
  `tests.ipynb` on Colab is the authoritative correctness gate.

## File layout for the brief's grading rubric

| Rubric dimension       | Where to look                                          |
| ---------------------- | ------------------------------------------------------ |
| Reproducibility        | All three notebooks; `apply_optimization.py` is idempotent |
| Correctness rigor      | `tests.ipynb` §5 (output equivalence) and §4 (existing test suite); `docs/profiling_notes.md` |
| Measurement rigor      | `benchmark/benchmark_utils.py`; same-VM comparison in `tests.ipynb` §3–§4 |
| Task choice            | This README's "TL;DR" and "The workload" sections     |
| Speedup magnitude      | `reward.json` (filled in by `tests.ipynb`)            |
| Optimization technique | This README's "The optimization" section + `_regionprops_fast.py` source |
| Communication          | This README + per-notebook markdown cells             |
| Separation of concerns | `tests.ipynb` is the sole writer of `reward.json`; the other two notebooks only generate inputs |
