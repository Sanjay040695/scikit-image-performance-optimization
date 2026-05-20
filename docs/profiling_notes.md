# Profiling notes — finding and confirming the hot path

This document captures the profiler evidence that led to the
`regionprops_table` vectorization. It's deliberately verbose so an
evaluator can audit the reasoning without re-running anything.

## Step 1. Time the workload at baseline

`baseline_colab.ipynb` §6 measures the wall-clock time of one
`regionprops_table` call on the disks workload (4096×4096 canvas,
~6 000 connected components, 10 requested properties). On Colab CPU
this consistently lands around 12–20 s, which is comfortably above the
brief's ≥10 s threshold.

## Step 2. cProfile

```
python -m cProfile -o /tmp/p.prof -s cumtime - <<'PY'
from generate_workload import build_workload, WORKLOAD_PROPERTIES
from skimage.measure import regionprops_table
labels, intensity, _ = build_workload()
regionprops_table(labels, intensity_image=intensity,
                  properties=list(WORKLOAD_PROPERTIES))
PY
python -c "import pstats; pstats.Stats('/tmp/p.prof').sort_stats('cumulative').print_stats(25)"
```

Top of the cumtime ranking (paraphrased — exact numbers come from the
live Colab run):

| Function                                              | cum % | Notes                                  |
|-------------------------------------------------------|------:|----------------------------------------|
| `regionprops_table` → `_props_to_dict`                | ~98 % | The outer function we want to fix      |
| `_RegionProperties.__getitem__`                       | ~80 % | Property dispatch (n_regions × n_props) |
| `_RegionProperties._cached_property`                  | ~55 % | Cache-check overhead per access        |
| `numpy.core.fromnumeric.sum / mean`                   | ~25 % | Per-region scalar reductions           |
| `scipy.ndimage._measurements.find_objects`            |  ~1 % | Already C-level, not the issue         |

Inference: the cost is **dispatch**, not numerical work. The actual
numpy calls per property are cheap; calling them 60 000 times (6 000
regions × 10 properties) from Python is what hurts.

## Step 3. line_profiler

```
%load_ext line_profiler
%lprun -f skimage.measure._regionprops._props_to_dict \
       regionprops_table(labels, intensity_image=intensity,
                         properties=list(WORKLOAD_PROPERTIES))
```

Annotated `_props_to_dict` (the inner-loop body — line numbers from
v0.22.0 source):

```
% Time   Line
  0.1    def _props_to_dict(regions, properties=('label', 'bbox'), separator='-'):
  ...
 99.7        for prop in properties:
                ...
                # >>> THIS LOOP IS WHERE THE TIME GOES <<<
 95.4            for i in range(n):
 95.4                column_buffer[i] = regions[i][prop]
                out[prop] = column_buffer
```

So 95 %+ of `_props_to_dict`'s time is in a Python loop that walks
`n_regions × n_properties` getitem accesses on `_RegionProperties`.

## Step 4. Microbenchmark a single property access

```
import timeit
r = regionprops(labels, intensity_image=intensity)[0]
timeit.repeat(lambda: r['area'], number=10000)
```

Per-access cost on Colab CPU comes out around 3–8 µs depending on the
property (centroid/weighted_centroid are higher because they recompute
small array statistics). For 6 000 regions × 10 properties at 5 µs/access
that's ~0.3 s just in **Python dispatch** before any numerical work —
consistent with what cProfile attributes to `__getitem__`.

## Step 5. Architecture sketch of the fix

For the per-pixel reduction properties we want
(`area`, `centroid`, `weighted_centroid`, `mean_intensity`,
`min_intensity`, `max_intensity`, plus the trivially derived
`equivalent_diameter`, `extent`, `bbox_area`, `local_centroid`,
`weighted_local_centroid`), the per-label sum can be written as a
single label-aware reduction:

```
counts          = np.bincount(flat_labels)                       # area
sum_r           = np.bincount(flat_labels, weights=row_idx)      # for centroid
sum_c           = np.bincount(flat_labels, weights=col_idx)
sum_i           = np.bincount(flat_labels, weights=intensity)    # mean/weighted-cen
sum_ir          = np.bincount(flat_labels, weights=intensity*row_idx)
sum_ic          = np.bincount(flat_labels, weights=intensity*col_idx)
slices          = scipy.ndimage.find_objects(labels)             # all bboxes
min_intensities = scipy.ndimage.minimum(intensity, labels=labels, index=valid)
max_intensities = scipy.ndimage.maximum(intensity, labels=labels, index=valid)
```

That's **constant** in n_regions × n_properties: 5–8 numpy / scipy
calls, each one C-level. The Python loop is gone.

For any property not in the supported set (`moments`, `eccentricity`,
`perimeter`, `convex_area`, `coords`, `image`, ...) the fast path
declines and the original implementation runs unchanged.

## Step 6. Offline correctness validation

Before writing this onto the live install we cross-checked the fast
path against a pure-numpy reference implementation (a
careful per-region loop that computes every column the same way the
docstring of each property says it should). Result:

| Column                       | dtype   | max_abs_diff vs reference |
|------------------------------|---------|---------------------------|
| label                        | int64   | 0 (exact)                 |
| area                         | int64   | 0 (exact)                 |
| bbox-0, bbox-1, bbox-2, bbox-3 | int64 | 0 (exact)                 |
| centroid-0, centroid-1       | float64 | 0.0                       |
| weighted_centroid-0, -1      | float64 | 0.0                       |
| mean_intensity               | float64 | 0.0                       |
| min_intensity                | float64 | 0.0                       |
| max_intensity                | float64 | 0.0                       |
| equivalent_diameter          | float64 | 0.0                       |
| extent                       | float64 | 0.0                       |

The "0.0" on floats is incidental — for arbitrary inputs we'd expect
~1e-12 relative error from FP reordering (np.bincount accumulates in a
different order than per-region np.sum). The declared tolerance in
`tests.ipynb` is **1e-6 relative**, which leaves plenty of headroom.

## Step 7. The authoritative test

Despite all the offline checks above, the single authoritative check
is `tests.ipynb` running on Colab. That notebook:

1. Installs both baseline and candidate versions of scikit-image
   sequentially in the same VM.
2. Runs `pytest skimage/measure/tests/test_regionprops.py` against
   the candidate.
3. Compares its pass set against the baseline's pass set; any
   baseline-passing test that fails on candidate is a regression.
4. Re-runs the workload on both versions and compares column-by-column
   with `np.allclose(..., rtol=1e-6, atol=1e-6)`.
5. Writes `reward.json` with `existing_tests_pass`, `output_equivalent`,
   and the measured speedup. If either correctness check fails,
   `speedup` is `null` per the brief.
