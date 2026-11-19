"""
heuristics.py
-------------
Heuristic parameters and per-instance feature computation used to flag
suspicious segmentation instances for human review.

Key exports
~~~~~~~~~~~
HeuristicParams      - dataclass controlling all tunable thresholds
mad()                - median absolute deviation helper
ellipsoid_footprint()- anisotropy-aware peak-detection footprint
compute_basic_features()   - cheap features (no EDT)
compute_dt_and_peaks()     - spacing-aware EDT + strong-peak count
score_instance()     - convert features → (score, reasons) for review queue

Feature dict
~~~~~~~~~~~~
Feature keys use the SAME names as the review-queue columns in
``utils.REVIEW_QUEUE_COLUMNS`` so rows can be built by lookup rather than by
renaming: ``volume_vox``, ``erosion_splits_xy``, ``n_strong_peaks``,
``max_radius_nm``.  Fields that were deliberately skipped hold
``utils.FEATURE_NOT_COMPUTED`` (-1) rather than 0, so "not computed" is never
confused with a genuine zero.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

from utils import FEATURE_NOT_COMPUTED, STRUCT_26, STRUCT_XY


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class HeuristicParams:
    """Tunable thresholds for instance building and review flagging.

    Every field name matches its CLI flag in ``napari_session.py``
    (``--min_sep_nm`` → ``min_sep_nm``) and every default here matches the
    CLI default, so ``HeuristicParams()`` and a bare CLI run behave alike.
    """

    # Geometry.  spacing is (x, y, z) — see the axis-order convention in utils.
    spacing_nm_xyz: Tuple[float, float, float] = (5.0, 5.0, 20.0)
    min_sep_nm: float = 120.0        # min peak separation for peak detection
    strong_peak_frac: float = 0.70   # peaks >= frac * max(dist) are "strong"
    erosion_xy_iters: int = 1        # erosion iterations, XY plane only
    size_outlier_k_mad: float = 3.0  # outlier threshold: median + k * MAD
    min_instance_voxels: int = 100   # drop connected components below this
    verbose: bool = True

    # Watershed-first refinement (optional; off unless watershed_first=True)
    watershed_first: bool = False
    ws_min_peaks: int = 2            # minimum strong peaks to trigger watershed
    ws_max_seeds: int = 4            # cap on seeds per component
    ws_min_voxels: int = 10000       # only watershed components >= this size
    ws_strong_peak_frac: float = 0.75

    # Fast pre-split for huge touching clusters (optional; 0 disables)
    pre_split_xy_erosion_iters: int = 0          # 1–2 typical when enabled
    pre_split_apply_over_voxels: int = 1000000   # only components >= this size


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def mad(x: np.ndarray) -> float:
    """Median absolute deviation (with a small epsilon to avoid zero)."""
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-9)


# ---------------------------------------------------------------------------
# Peak-detection footprint
# ---------------------------------------------------------------------------

def ellipsoid_footprint(
    min_sep_nm: float,
    spacing_nm_xyz: Tuple[float, float, float],
) -> np.ndarray:
    """Build an anisotropy-aware ellipsoidal footprint in (z, y, x) index space
    for use with ``skimage.feature.peak_local_max``.

    The radius in each dimension is chosen so that the footprint covers
    approximately ``min_sep_nm / 2`` physical nanometres.
    """
    sx, sy, sz = spacing_nm_xyz  # x, y, z
    rx = max(1, int(round((min_sep_nm / 2) / sx)))
    ry = max(1, int(round((min_sep_nm / 2) / sy)))
    rz = max(1, int(round((min_sep_nm / 2) / sz)))

    z = np.arange(-rz, rz + 1)[:, None, None]
    y = np.arange(-ry, ry + 1)[None, :, None]
    x = np.arange(-rx, rx + 1)[None, None, :]

    val = (
        (x * sx) ** 2 / (min_sep_nm / 2) ** 2
        + (y * sy) ** 2 / (min_sep_nm / 2) ** 2
        + (z * sz) ** 2 / (min_sep_nm / 2) ** 2
    )
    return (val <= 1.0).astype(bool)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_basic_features(
    inst_mask_zyx: np.ndarray,
    params: HeuristicParams,
) -> Dict:
    """Cheap features that avoid EDT / peak finding.

    Returns a dict keyed by ``utils.REVIEW_QUEUE_COLUMNS`` names:
    ``volume_vox``, ``erosion_splits_xy``, ``n_strong_peaks``,
    ``max_radius_nm``.  The last two are left at ``FEATURE_NOT_COMPUTED``
    and are filled in by ``compute_dt_and_peaks`` only when needed.
    """
    vol = int(inst_mask_zyx.sum())

    eroded = ndi.binary_erosion(
        inst_mask_zyx, structure=STRUCT_XY, iterations=params.erosion_xy_iters
    )
    _, n_e = ndi.label(eroded, structure=STRUCT_26)
    erosion_splits = bool(n_e >= 2)

    return {
        "volume_vox": vol,
        "erosion_splits_xy": erosion_splits,
        "n_strong_peaks": FEATURE_NOT_COMPUTED,   # filled by compute_dt_and_peaks
        "max_radius_nm": FEATURE_NOT_COMPUTED,    # filled by compute_dt_and_peaks
    }


def compute_dt_and_peaks(
    inst_mask_zyx: np.ndarray,
    params: HeuristicParams,
    strong_frac: float,
) -> Tuple[np.ndarray, float, int]:
    """Compute a spacing-aware EDT and count strong peaks.

    Returns ``(dist_nm, max_radius_nm, n_strong_peaks)``.
    """
    sx, sy, sz = params.spacing_nm_xyz
    dist_nm = ndi.distance_transform_edt(inst_mask_zyx, sampling=(sz, sy, sx))
    max_r = float(dist_nm.max())
    if max_r <= 0:
        return dist_nm, 0.0, 0

    fp = ellipsoid_footprint(params.min_sep_nm, params.spacing_nm_xyz)
    coords = peak_local_max(
        dist_nm,
        footprint=fp,
        labels=inst_mask_zyx.astype(np.uint8),
        exclude_border=False,
    )
    if coords.size == 0:
        return dist_nm, max_r, 0

    vals = dist_nm[coords[:, 0], coords[:, 1], coords[:, 2]]
    n_strong = int(np.sum(vals >= strong_frac * max_r))
    return dist_nm, max_r, n_strong


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_instance(feat: Dict, size_outlier: bool) -> Tuple[int, List[str]]:
    """Conservative, merge-first scoring.

    Rules:
    - ``erosion_splits_xy``    → +3  (thin contact / touching clump)
    - ``n_strong_peaks >= 2``  → +2 per extra peak (multi-lobed)
    - ``size_outlier``         → +2  (unusually large; needs review)

    A ``n_strong_peaks`` of ``FEATURE_NOT_COMPUTED`` (-1) never triggers the
    multi-peak rule, so skipping the EDT can only lower a score, never raise it.

    Returns ``(score, reasons)`` where *reasons* is a list of string labels.
    """
    score = 0
    reasons: List[str] = []

    if feat["erosion_splits_xy"]:
        score += 3
        reasons.append("erosion_splits_xy")

    if feat["n_strong_peaks"] >= 2:
        score += 2 * (feat["n_strong_peaks"] - 1)
        reasons.append(f"multi_peaks({feat['n_strong_peaks']})")

    if size_outlier:
        score += 2
        reasons.append("size_outlier")

    return score, reasons
