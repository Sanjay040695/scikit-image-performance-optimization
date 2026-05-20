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
#
# scikit-image renamed many properties in v0.20 (Mar 2023):
#   bbox_area               -> area_bbox
#   convex_area             -> area_convex
#   filled_area             -> area_filled
#   equivalent_diameter     -> equivalent_diameter_area
#   weighted_centroid       -> centroid_weighted
#   weighted_local_centroid -> centroid_weighted_local
#   local_centroid          -> centroid_local
#   mean_intensity          -> intensity_mean
#   min_intensity           -> intensity_min
#   max_intensity           -> intensity_max
#
# We accept BOTH names for forward/backward compatibility; the output
# column name matches whatever the caller asked for.
# ---------------------------------------------------------------------------

_NAME_ALIASES = {
    # old name -> new name
    "bbox_area":                "area_bbox",
    "equivalent_diameter":      "equivalent_diameter_area",
    "weighted_centroid":        "centroid_weighted",
    "weighted_local_centroid":  "centroid_weighted_local",
    "local_centroid":           "centroid_local",
    "mean_intensity":           "intensity_mean",
    "min_intensity":            "intensity_min",
    "max_intensity":            "intensity_max",
}
_NAME_ALIASES.update({v: k for k, v in _NAME_ALIASES.items()})   # bidirectional

_FAST_PROPS_BINARY_CANONICAL = frozenset({
    "label", "area", "bbox", "area_bbox",
    "centroid", "centroid_local",
    "equivalent_diameter_area", "extent",
})

_FAST_PROPS_INTENSITY_CANONICAL = frozenset({
    "centroid_weighted", "centroid_weighted_local",
    "intensity_mean", "intensity_min", "intensity_max",
})


def _canonical(prop: str) -> str:
    """Return the canonical (v0.22+) name for a property."""
    if prop in _FAST_PROPS_BINARY_CANONICAL or prop in _FAST_PROPS_INTENSITY_CANONICAL:
        return prop
    return _NAME_ALIASES.get(prop, prop)


def can_use_fast_path(properties, intensity_image, extra_properties,
                      spacing=None) -> bool:
    """Decide whether the fast vectorized path supports this call.

    Falls back to the original implementation if:
      - any extra_properties were passed,
      - any spacing was passed (centroids would be in physical units),
      - any requested property isn't in the supported set,
      - intensity properties are requested without an intensity image.
    """
    if extra_properties is not None:
        return False
    if spacing is not None:
        return False
    canon = {_canonical(p) for p in properties}
    if not canon.issubset(
        _FAST_PROPS_BINARY_CANONICAL | _FAST_PROPS_INTENSITY_CANONICAL
    ):
        return False
    if intensity_image is None and (canon & _FAST_PROPS_INTENSITY_CANONICAL):
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

    # ----- assemble the output dict using the EXACT property names the
    # caller asked for (so a caller using v0.18 names sees v0.18 columns
    # and a caller using v0.22 names sees v0.22 columns).
    out: dict = {}
    for prop in properties:
        canon = _canonical(prop)
        if canon == "label":
            out[prop] = _int_column(label_values)
        elif canon == "area":
            out[prop] = _int_column(area)
        elif canon == "bbox":
            for k in range(4):
                out[f"{prop}{separator}{k}"] = _int_column(bbox[:, k])
        elif canon == "area_bbox":
            out[prop] = _int_column(bbox_area)
        elif canon == "centroid":
            out[f"{prop}{separator}0"] = _float_column(centroid_r)
            out[f"{prop}{separator}1"] = _float_column(centroid_c)
        elif canon == "centroid_local":
            out[f"{prop}{separator}0"] = _float_column(
                centroid_r - bbox[:, 0].astype(np.float64)
            )
            out[f"{prop}{separator}1"] = _float_column(
                centroid_c - bbox[:, 1].astype(np.float64)
            )
        elif canon == "equivalent_diameter_area":
            out[prop] = _float_column(equivalent_diameter)
        elif canon == "extent":
            out[prop] = _float_column(extent)
        elif canon == "centroid_weighted":
            out[f"{prop}{separator}0"] = _float_column(wcen_r)
            out[f"{prop}{separator}1"] = _float_column(wcen_c)
        elif canon == "centroid_weighted_local":
            out[f"{prop}{separator}0"] = _float_column(
                wcen_r - bbox[:, 0].astype(np.float64)
            )
            out[f"{prop}{separator}1"] = _float_column(
                wcen_c - bbox[:, 1].astype(np.float64)
            )
        elif canon == "intensity_mean":
            out[prop] = _float_column(mean_int)
        elif canon == "intensity_min":
            out[prop] = _float_column(min_int)
        elif canon == "intensity_max":
            out[prop] = _float_column(max_int)
        else:
            raise RuntimeError(
                f"_regionprops_fast: property {prop!r} not supported"
            )
    return out
