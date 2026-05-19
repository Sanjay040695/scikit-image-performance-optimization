"""
Vectorized fast path for skimage.measure.regionprops_table.

Why this exists
---------------
In scikit-image v0.18.3, `regionprops_table` works by:
    1) calling `regionprops()` to build a list of `_RegionProperties` objects
       (one per connected region);
    2) for each requested property name `p`, looping over every region and
       calling `region[p]` to populate the output column.

For workloads with thousands of small regions and a handful of common
scalar properties (`area`, `bbox`, `centroid`, `mean_intensity`, ...), the
inner `for i in range(n_regions): out[p][i] = regions[i][p]` loop is
overwhelmingly dominated by Python-side overhead: attribute lookup,
cache check, slice construction, and many tiny `np.sum` / `np.mean`
calls. None of that scales — and none of it is needed when the
property can be expressed as a single bulk reduction over the whole
label image.

This module replaces that inner loop, for the supported property set,
with a small number of label-aware bulk reductions:

    * `np.bincount(flat_labels, weights=...)`  -> area, sums for centroid,
                                                  intensity sums for weighted
                                                  centroid, etc.
    * `scipy.ndimage.find_objects`             -> bounding boxes for every
                                                  label in one C call.
    * `scipy.ndimage.minimum` / `.maximum`     -> per-label intensity extrema.

The number of Python-level operations becomes O(n_properties) instead
of O(n_regions * n_properties), and each operation is a single
C-level reduction over a contiguous numpy buffer.

Properties this module can serve fast
-------------------------------------
Without intensity image:
    label, area, bbox, bbox_area, centroid, local_centroid,
    equivalent_diameter, extent

Additionally with intensity image:
    weighted_centroid, weighted_local_centroid,
    mean_intensity, min_intensity, max_intensity

Any property outside this set, or any non-None `extra_properties`, falls
back to the original scikit-image implementation. We never silently
change behaviour for unsupported cases.

Correctness contract
--------------------
Output dict has the same keys and the same column dtypes as the
original `regionprops_table` would have produced for the same arguments.
Floating-point columns may differ by <=1e-6 relative due to FP-add
reordering (bincount accumulates in a different order than the
per-region np.sum); this is the tolerance documented in tests.ipynb.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Property sets supported by this fast path.
# ---------------------------------------------------------------------------

_FAST_PROPS_BINARY = frozenset({
    "label", "area", "bbox", "bbox_area",
    "centroid", "local_centroid",
    "equivalent_diameter", "extent",
})

_FAST_PROPS_INTENSITY = frozenset({
    "weighted_centroid", "weighted_local_centroid",
    "mean_intensity", "min_intensity", "max_intensity",
})

_FAST_PROPS_ALL = _FAST_PROPS_BINARY | _FAST_PROPS_INTENSITY


def can_use_fast_path(properties, intensity_image, extra_properties) -> bool:
    """Decide whether the fast vectorized path supports this call."""
    if extra_properties is not None:
        return False
    props_set = set(properties)
    if not props_set.issubset(_FAST_PROPS_ALL):
        return False
    if intensity_image is None and (props_set & _FAST_PROPS_INTENSITY):
        return False
    return True


# ---------------------------------------------------------------------------
# Column dtypes (mirrors COL_DTYPES in skimage's own _regionprops.py)
# ---------------------------------------------------------------------------

_INT_PROPS = {"label", "area", "bbox", "bbox_area"}


def _int_column(values: np.ndarray) -> np.ndarray:
    """Return an integer-typed column (matches v0.18.3 behaviour)."""
    return np.asarray(values, dtype=np.int64)


def _float_column(values: np.ndarray) -> np.ndarray:
    """Return a float64-typed column (matches v0.18.3 behaviour)."""
    return np.asarray(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# The fast path.
# ---------------------------------------------------------------------------

def regionprops_table_fast(
    label_image: np.ndarray,
    intensity_image,
    properties,
    separator: str = "-",
) -> dict:
    """Vectorized regionprops_table for the supported property set.

    Notes
    -----
    Only supports 2-D label images. (The slow path supports n-D, but the
    common-case workload that motivates this optimization is 2-D, and
    handling n-D correctly here would add complexity without buying
    much.)
    """
    if label_image.ndim != 2:
        raise NotImplementedError(
            "_regionprops_fast supports 2-D label images only "
            "(got ndim=%d)." % label_image.ndim
        )

    # SciPy is a hard dependency of scikit-image, so this import is safe.
    from scipy import ndimage as ndi

    labels = np.ascontiguousarray(label_image)
    if labels.dtype.kind not in ("i", "u"):
        labels = labels.astype(np.int64, copy=False)

    H, W = labels.shape
    n_labels = int(labels.max())

    # find_objects gives one slice-tuple per label index in [1..n_labels],
    # or None where that label is absent. This is identical to what
    # regionprops() uses internally, and identical to what skimage's
    # _RegionProperties walks over.
    slices = ndi.find_objects(labels, max_label=n_labels)
    present_idx = [i for i, s in enumerate(slices) if s is not None]
    n_regions = len(present_idx)
    label_values = np.array(
        [i + 1 for i in present_idx], dtype=np.int64
    )

    # ----- bbox (min_row, min_col, max_row, max_col), max-exclusive -----
    bbox = np.empty((n_regions, 4), dtype=np.int64)
    for j, i in enumerate(present_idx):
        sr, sc = slices[i]
        bbox[j, 0] = sr.start
        bbox[j, 1] = sc.start
        bbox[j, 2] = sr.stop
        bbox[j, 3] = sc.stop
    # Tight Python loop, but only over n_regions and only doing four
    # integer extractions per region. This is ~1000x cheaper than the
    # original per-region property dispatch.

    # ----- bulk reductions over the full label image -----
    flat = labels.ravel()

    # area via bincount.
    counts_full = np.bincount(flat, minlength=n_labels + 1)
    area = counts_full[label_values]  # int64

    # centroid: row/col sums per label, divided by area.
    # We build the broadcasted row / col index buffers as float64 once.
    row_idx = np.repeat(
        np.arange(H, dtype=np.float64), W
    )
    col_idx = np.tile(
        np.arange(W, dtype=np.float64), H
    )
    sum_r_full = np.bincount(flat, weights=row_idx, minlength=n_labels + 1)
    sum_c_full = np.bincount(flat, weights=col_idx, minlength=n_labels + 1)
    centroid_r = sum_r_full[label_values] / area
    centroid_c = sum_c_full[label_values] / area

    # intensity-derived quantities (only if needed).
    have_intensity = intensity_image is not None
    if have_intensity:
        intens = np.ascontiguousarray(intensity_image).astype(
            np.float64, copy=False
        ).ravel()
        sum_i_full = np.bincount(flat, weights=intens, minlength=n_labels + 1)
        sum_ir_full = np.bincount(
            flat, weights=intens * row_idx, minlength=n_labels + 1
        )
        sum_ic_full = np.bincount(
            flat, weights=intens * col_idx, minlength=n_labels + 1
        )
        sum_i = sum_i_full[label_values]
        mean_int = sum_i / area
        wcen_r = sum_ir_full[label_values] / sum_i
        wcen_c = sum_ic_full[label_values] / sum_i
        # min / max via scipy.ndimage (one C call each).
        # `index` controls which labels we want — pass our valid labels.
        min_int = np.asarray(
            ndi.minimum(intensity_image, labels=labels, index=label_values),
            dtype=np.float64,
        )
        max_int = np.asarray(
            ndi.maximum(intensity_image, labels=labels, index=label_values),
            dtype=np.float64,
        )

    # bbox_area, equivalent_diameter, extent.
    bbox_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    equivalent_diameter = np.sqrt(4.0 * area / np.pi)
    extent = area / bbox_area

    # ----- assemble the output dict in the same column-layout as the
    # original _props_to_dict.
    out: dict = {}
    for prop in properties:
        if prop == "label":
            out["label"] = _int_column(label_values)
        elif prop == "area":
            out["area"] = _int_column(area)
        elif prop == "bbox":
            for k in range(4):
                out[f"bbox{separator}{k}"] = _int_column(bbox[:, k])
        elif prop == "bbox_area":
            out["bbox_area"] = _int_column(bbox_area)
        elif prop == "centroid":
            out[f"centroid{separator}0"] = _float_column(centroid_r)
            out[f"centroid{separator}1"] = _float_column(centroid_c)
        elif prop == "local_centroid":
            out[f"local_centroid{separator}0"] = _float_column(
                centroid_r - bbox[:, 0].astype(np.float64)
            )
            out[f"local_centroid{separator}1"] = _float_column(
                centroid_c - bbox[:, 1].astype(np.float64)
            )
        elif prop == "equivalent_diameter":
            out["equivalent_diameter"] = _float_column(equivalent_diameter)
        elif prop == "extent":
            out["extent"] = _float_column(extent)
        elif prop == "weighted_centroid":
            out[f"weighted_centroid{separator}0"] = _float_column(wcen_r)
            out[f"weighted_centroid{separator}1"] = _float_column(wcen_c)
        elif prop == "weighted_local_centroid":
            out[f"weighted_local_centroid{separator}0"] = _float_column(
                wcen_r - bbox[:, 0].astype(np.float64)
            )
            out[f"weighted_local_centroid{separator}1"] = _float_column(
                wcen_c - bbox[:, 1].astype(np.float64)
            )
        elif prop == "mean_intensity":
            out["mean_intensity"] = _float_column(mean_int)
        elif prop == "min_intensity":
            out["min_intensity"] = _float_column(min_int)
        elif prop == "max_intensity":
            out["max_intensity"] = _float_column(max_int)
        else:
            # Defensive: should never get here because can_use_fast_path
            # already filtered.
            raise RuntimeError(
                f"_regionprops_fast: property {prop!r} not supported"
            )
    return out
