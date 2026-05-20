"""
Deterministic workload generator for the scikit-image regionprops_table
performance benchmark.

Generates an HxW binary image with N non-overlapping disks placed at
fixed (seeded) positions, plus a matching float intensity image. The
image is then labeled with `skimage.measure.label`, and the resulting
label image + intensity image are the inputs to `regionprops_table`.

This mimics realistic biomedical / particle-counting workloads where
thousands of distinct objects are measured at once.

Design rationale (see README):
- Fixed seed (42)  ->  deterministic, reproducible.
- Non-overlapping disks  ->  many distinct labels (not one giant
  percolating component, which a uniform random binary image would
  produce).
- ~3000-4000 final labels on the canvas size below  ->  enough Python
  loop iterations in regionprops_table to make per-region overhead
  the dominant cost.
- Wall time on Colab CPU: ~15-25 s for the baseline.

The same image is used by both baseline_colab.ipynb and
candidate_colab.ipynb so the speedup comparison is apples-to-apples.
"""

from __future__ import annotations

import numpy as np


# Default workload parameters. Sized for Colab CPU runtime
# (~12-25 s baseline regionprops_table wall time, fits in 12.7 GB RAM).
DEFAULT_H = 4096
DEFAULT_W = 4096
DEFAULT_N_DISKS = 6000
DEFAULT_R_MIN = 4
DEFAULT_R_MAX = 18
DEFAULT_SEED = 42

# The exact property set we extract. These are all commonly used scalar /
# fixed-shape properties that stress every line of the regionprops_table
# per-region inner loop.
#
# We use the v0.22+ NEW property names because that's the active baseline.
# v0.18-era names (mean_intensity, weighted_centroid, ...) were renamed
# to intensity_mean / centroid_weighted / ... in v0.20.
WORKLOAD_PROPERTIES = (
    "label",
    "area",
    "bbox",
    "centroid",
    "centroid_weighted",
    "intensity_mean",
    "equivalent_diameter_area",
    "extent",
    "intensity_min",
    "intensity_max",
)


def make_binary_image(
    H: int = DEFAULT_H,
    W: int = DEFAULT_W,
    n_disks: int = DEFAULT_N_DISKS,
    r_min: int = DEFAULT_R_MIN,
    r_max: int = DEFAULT_R_MAX,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Build a deterministic HxW binary image with non-overlapping disks."""
    rng = np.random.default_rng(seed)
    canvas = np.zeros((H, W), dtype=bool)

    # Pre-build per-radius disk masks once
    disk_masks = {}
    for r in range(r_min, r_max + 1):
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        disk_masks[r] = (yy * yy + xx * xx) <= r * r

    radii = rng.integers(r_min, r_max + 1, size=n_disks)
    ys = rng.integers(r_max + 1, H - r_max - 1, size=n_disks)
    xs = rng.integers(r_max + 1, W - r_max - 1, size=n_disks)

    for r, cy, cx in zip(radii, ys, xs):
        m = disk_masks[int(r)]
        sub = canvas[cy - r : cy + r + 1, cx - r : cx + r + 1]
        # Skip if disk would overlap with anything already placed.
        # This keeps every successful placement its own connected component.
        if sub[m].any():
            continue
        sub[m] = True

    return canvas


def make_intensity_image(
    shape: tuple[int, int],
    seed: int = DEFAULT_SEED + 1,
) -> np.ndarray:
    """Deterministic float32 intensity image (uniform [0, 1))."""
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float64).astype(np.float32)


def build_workload(
    H: int = DEFAULT_H,
    W: int = DEFAULT_W,
    n_disks: int = DEFAULT_N_DISKS,
    r_min: int = DEFAULT_R_MIN,
    r_max: int = DEFAULT_R_MAX,
    seed: int = DEFAULT_SEED,
):
    """Build (label_image, intensity_image) ready for regionprops_table.

    Returns
    -------
    label_image : np.ndarray, shape (H, W), int32
        Connected-component labels (0 = background).
    intensity_image : np.ndarray, shape (H, W), float32
        Random intensity in [0, 1).
    n_labels : int
        Number of foreground regions actually placed.
    """
    from skimage.measure import label  # import here so this file is import-safe

    binary = make_binary_image(H, W, n_disks, r_min, r_max, seed)
    labels = label(binary, connectivity=2).astype(np.int32, copy=False)
    intensity = make_intensity_image(binary.shape)
    return labels, intensity, int(labels.max())


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    labels, intensity, n = build_workload()
    print(f"workload built in {time.perf_counter() - t0:.2f}s: "
          f"shape={labels.shape}, n_regions={n}")
