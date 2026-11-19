"""
utils.py
--------
Project-wide conventions: axis order, units, sentinel values, and the
canonical review-queue schema.

This module is the dependency root — it imports nothing from the project,
so every other module may import from it freely.

Conventions
~~~~~~~~~~~
Axis order
    Arrays are ALWAYS stored and processed as ``(z, y, x)`` (NumPy / mrcfile
    default).  Tuples of per-axis quantities carry their order in the name:
    ``spacing_nm_xyz`` is ``(x, y, z)``; ``scale_zyx`` is ``(z, y, x)``.
    Convert only when reporting coordinates to the user.

Units
    Every physical quantity carries its unit in the name:
    ``_nm`` (nanometre), ``_angstrom`` (Ångström), ``_px`` / ``_vox`` (voxels).
    File headers are read in Ångström; everything downstream works in nm.

Sentinels
    ``LABEL_BACKGROUND``     0  in instance / cell / semantic volumes
    ``CELL_ID_UNASSIGNED``   0  in ``inst_to_cell`` maps and queue CSVs
    ``FEATURE_NOT_COMPUTED`` -1 for feature fields that were skipped

Key exports
~~~~~~~~~~~
ensure_zyx()            - contiguous (z, y, x) view of an array
angstrom_to_nm(), nm_to_angstrom() - unit conversion
REVIEW_QUEUE_COLUMNS    - canonical review-queue column order
STRUCT_26, STRUCT_XY    - canonical connectivity structuring elements
"""

from typing import Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

LABEL_BACKGROUND: int = 0
CELL_ID_UNASSIGNED: int = 0
FEATURE_NOT_COMPUTED: float = -1.0

# ---------------------------------------------------------------------------
# Connectivity structuring elements
# ---------------------------------------------------------------------------
# Built once here and imported everywhere else, so every connected-component
# call in the project uses the exact same connectivity.

STRUCT_26 = np.ones((3, 3, 3), dtype=bool)  # full 26-connectivity (3D)
STRUCT_XY = np.ones((1, 3, 3), dtype=bool)  # 8-connectivity within one Z slice

# ---------------------------------------------------------------------------
# Canonical review-queue schema
# ---------------------------------------------------------------------------
# One row per instance.  Every producer (segmentation pipelines, the napari
# session, the CLI) writes these columns in this order so that any CSV can be
# round-tripped through any stage of the workflow.

REVIEW_QUEUE_COLUMNS = [
    "instance_id",       # int   > 0, unique per instance in the volume
    "cell_id",           # int   0 == CELL_ID_UNASSIGNED
    "score",             # int   review priority; > 0 means "flagged"
    "reasons",           # str   ';'-joined reason labels, '' when unflagged
    "volume_vox",        # int   voxel count
    "max_radius_nm",     # float max EDT radius; -1 == not computed
    "n_strong_peaks",    # int   strong EDT peaks; -1 == not computed
    "erosion_splits_xy", # int   0/1, component falls apart under XY erosion
    "centroid_z",        # float voxel index, global volume coordinates
    "centroid_y",
    "centroid_x",
]


# ---------------------------------------------------------------------------
# Axis order
# ---------------------------------------------------------------------------

def ensure_zyx(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` as a C-contiguous array, assumed to already be (z, y, x).

    This does **not** reorder axes — readers are responsible for delivering
    (z, y, x).  It only guarantees contiguity (no copy if already contiguous)
    so that downstream ``scipy.ndimage`` calls take the fast path.
    """
    return np.ascontiguousarray(arr)


def spacing_xyz_to_scale_zyx(
    spacing_nm_xyz: Sequence[float],
) -> Tuple[float, float, float]:
    """Convert a ``(x, y, z)`` spacing tuple to a napari ``(z, y, x)`` scale."""
    sx, sy, sz = spacing_nm_xyz
    return (float(sz), float(sy), float(sx))


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def angstrom_to_nm(value_angstrom: float) -> float:
    """Convert Ångström to nanometre (1 nm == 10 Å)."""
    return float(value_angstrom) / 10.0


def nm_to_angstrom(value_nm: float) -> float:
    """Convert nanometre to Ångström (1 nm == 10 Å)."""
    return float(value_nm) * 10.0
