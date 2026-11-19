"""
segmentation.py
---------------
Algorithms for splitting, labeling, and creating instance segmentations.

Key exports
~~~~~~~~~~~
Vocabulary (see ``utils`` for the project-wide conventions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
semantic mask   binary volume, nonzero == foreground object
cell labels     integer volume, one id per parent cell
instance        one object, one integer id in the output volume
part            a piece of a connected component after a split
review queue    one row per instance, columns per ``REVIEW_QUEUE_COLUMNS``

Both public pipelines below run the same three stages — filter connected
components by size, optionally split them, then emit one queue row per part —
and delegate each stage to the same private helper, so the per-cell and global
entry points cannot drift apart.

Key exports
~~~~~~~~~~~
watershed_from_dist()              - watershed on a pre-computed EDT
watershed_split_mask()             - end-to-end watershed split of a binary mask
pre_split_xy_erosion_parts()       - fast XY-erosion-based pre-split
make_instances_from_cell_labels()  - per-cell instance labeling pipeline
make_instances_from_semantic_mask()- global instance labeling pipeline
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from heuristics import (
    HeuristicParams,
    ellipsoid_footprint,
    mad,
    compute_basic_features,
    compute_dt_and_peaks,
    score_instance,
)
from utils import CELL_ID_UNASSIGNED, REVIEW_QUEUE_COLUMNS, STRUCT_26, STRUCT_XY, ensure_zyx


# ---------------------------------------------------------------------------
# Watershed helpers
# ---------------------------------------------------------------------------

def watershed_from_dist(
    mask: np.ndarray,
    dist_nm: np.ndarray,
    params: HeuristicParams,
) -> np.ndarray:
    """Return a label map (0..K) from watershed on ``-dist_nm`` using strong-peak markers.

    Assumes ``dist_nm`` is the spacing-aware EDT for ``mask``.
    Seeds are capped at ``params.ws_max_seeds``.
    """
    max_r = float(dist_nm.max())
    if max_r <= 0:
        return np.zeros(mask.shape, dtype=np.int32)

    fp = ellipsoid_footprint(params.min_sep_nm, params.spacing_nm_xyz)
    coords = peak_local_max(
        dist_nm,
        footprint=fp,
        labels=mask.astype(np.uint8),
        exclude_border=False,
    )
    if coords.size == 0:
        return np.zeros(mask.shape, dtype=np.int32)

    vals = dist_nm[coords[:, 0], coords[:, 1], coords[:, 2]]
    thr = float(params.ws_strong_peak_frac * max_r)
    keep = vals >= thr
    coords, vals = coords[keep], vals[keep]

    if coords.shape[0] < 2:
        return np.zeros(mask.shape, dtype=np.int32)

    order = np.argsort(vals)[::-1]
    if coords.shape[0] > params.ws_max_seeds:
        order = order[: params.ws_max_seeds]
    coords = coords[order]

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (z, y, x) in enumerate(coords, start=1):
        markers[int(z), int(y), int(x)] = i

    markers = ndi.grey_dilation(markers, size=(3, 3, 3))
    ws = watershed(-dist_nm, markers=markers, mask=mask)
    return ws.astype(np.int32)


def watershed_split_mask(
    mask_zyx: np.ndarray,
    params: HeuristicParams,
) -> np.ndarray:
    """End-to-end watershed split of a binary mask.

    Computes a spacing-aware EDT internally, then calls ``watershed_from_dist``.
    Returns a label map with 0 background and 1..K parts.
    """
    mask = mask_zyx.astype(bool) if mask_zyx.dtype != bool else mask_zyx
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.int32)

    sx, sy, sz = params.spacing_nm_xyz
    dist_nm = ndi.distance_transform_edt(mask, sampling=(sz, sy, sx))
    return watershed_from_dist(mask, dist_nm, params)


def pre_split_xy_erosion_parts(mask_zyx: np.ndarray, iters: int) -> np.ndarray:
    """Partition ``mask_zyx`` into parts using XY-erosion seeds.

    A fast alternative to watershed that avoids EDT entirely.  Each connected
    component of the eroded mask becomes a seed; voxels are then assigned to
    the nearest seed by Euclidean distance.

    Returns an int32 label map with 0 background and 1..K parts, or an
    all-zero array if ``iters <= 0`` or the erosion removes everything.
    """
    if iters <= 0:
        return np.zeros(mask_zyx.shape, dtype=np.int32)

    mask = mask_zyx.astype(bool, copy=False)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.int32)

    seeds = ndi.binary_erosion(mask, structure=STRUCT_XY, iterations=int(iters))
    if not np.any(seeds):
        return np.zeros(mask.shape, dtype=np.int32)

    seeds_lbl, n = ndi.label(seeds, structure=STRUCT_26)
    if n < 2:
        return np.zeros(mask.shape, dtype=np.int32)

    # Assign every voxel in `mask` to the nearest seed
    _, inds = ndi.distance_transform_edt(seeds_lbl == 0, return_indices=True)
    nearest_lbl = seeds_lbl[inds[0], inds[1], inds[2]].astype(np.int32, copy=False)
    return np.where(mask, nearest_lbl, 0).astype(np.int32, copy=False)


# ---------------------------------------------------------------------------
# Shared pipeline stages
# ---------------------------------------------------------------------------
# Both public pipelines call these, so per-cell and global (semantic) runs
# produce identical splits, identical scores and identical queue columns.

def _select_components(
    cc: np.ndarray,
    n_cc: int,
    cc_bboxes: List,
    params: HeuristicParams,
    tag: str,
) -> Tuple[List[int], List[int], float]:
    """Keep components with at least ``min_instance_voxels`` voxels.

    Returns ``(cc_ids, volumes_vox, size_outlier_threshold_vox)``.  The
    threshold is ``median + k * MAD`` over the KEPT components only, so it is
    computed per cell in the per-cell pipeline and globally in the semantic one.
    """
    vols, cc_ids = [], []
    for k in range(1, n_cc + 1):
        slc_cc = cc_bboxes[k - 1] if (k - 1) < len(cc_bboxes) else None
        if slc_cc is None:
            continue
        n_vox = int(np.count_nonzero(cc[slc_cc] == k))
        if n_vox >= params.min_instance_voxels:
            vols.append(n_vox)
            cc_ids.append(k)

    if params.verbose:
        print(
            f"{tag} kept_components={len(cc_ids)} "
            f"dropped_small={n_cc - len(cc_ids)} (min={params.min_instance_voxels})"
        )

    if not cc_ids:
        return cc_ids, vols, float("inf")

    vols_arr = np.array(vols, dtype=float)
    v_med = float(np.median(vols_arr))
    v_mad = mad(vols_arr)
    outlier_thr = v_med + params.size_outlier_k_mad * v_mad
    if params.verbose:
        print(
            f"{tag} size stats: median={v_med:.1f} "
            f"MAD={v_mad:.1f} outlier_thr={outlier_thr:.1f}"
        )
    return cc_ids, vols, outlier_thr


def _split_component(
    inst_crop: np.ndarray,
    n_vox: int,
    is_outlier: bool,
    params: HeuristicParams,
    tag: str,
) -> Tuple[bool, List[Tuple[np.ndarray, bool]]]:
    """Decide whether one connected component should be split, and split it.

    Two strategies, tried in this order:

    1. **pre-split** — XY erosion, for components at or above
       ``pre_split_apply_over_voxels`` (cheap, no EDT).
    2. **watershed-first** — EDT + strong-peak markers, for components at or
       above ``ws_min_voxels`` that look multi-lobed or are size outliers.

    Returns ``(was_split, parts)`` where *parts* is a list of
    ``(part_mask, part_is_size_outlier)``.  When nothing is split, *parts* is
    the single original component carrying its own outlier flag.
    """
    unsplit: List[Tuple[np.ndarray, bool]] = [(inst_crop, is_outlier)]

    pre_parts = None
    if params.pre_split_xy_erosion_iters > 0 and n_vox >= params.pre_split_apply_over_voxels:
        pre_parts = pre_split_xy_erosion_parts(inst_crop, params.pre_split_xy_erosion_iters)
        if int(pre_parts.max()) < 2:
            pre_parts = None
        elif params.verbose:
            print(f"{tag} vox={n_vox} -> pre_split parts={int(pre_parts.max())}")

    parts_label = None
    if pre_parts is not None:
        parts_label = pre_parts
    elif params.watershed_first and n_vox >= params.ws_min_voxels:
        feat0 = compute_basic_features(inst_crop, params)
        dist0, maxr0, nstrong0 = compute_dt_and_peaks(
            inst_crop, params, strong_frac=params.strong_peak_frac
        )
        feat0.update(max_radius_nm=float(maxr0), n_strong_peaks=int(nstrong0))
        trigger_ws = (
            (feat0["n_strong_peaks"] >= params.ws_min_peaks and feat0["erosion_splits_xy"])
            or is_outlier
        )
        if trigger_ws:
            ws_parts = watershed_from_dist(inst_crop.astype(bool), dist0, params)
            if int(ws_parts.max()) >= 2:
                parts_label = ws_parts
                if params.verbose:
                    print(f"{tag} vox={n_vox} -> watershed parts={int(ws_parts.max())}")

    if parts_label is None:
        return False, unsplit

    # Parts of a split component carry no size-outlier flag of their own; the
    # parent's outlier status is recorded separately by ``_make_queue_row``.
    parts = [(parts_label == pid, False)
             for pid in sorted(set(parts_label.flat) - {0})]
    return True, parts


def _make_queue_row(
    inst_id: int,
    cell_id: int,
    part_mask: np.ndarray,
    offset_zyx: Tuple[int, int, int],
    params: HeuristicParams,
    part_outlier: bool,
    parent_outlier: bool,
) -> Dict:
    """Compute features + score for one part and return one review-queue row.

    The EDT is only computed when the part is already suspicious (a size
    outlier, or it falls apart under XY erosion); otherwise the EDT fields stay
    at ``FEATURE_NOT_COMPUTED``.  Centroids are returned in GLOBAL voxel indices
    by adding ``offset_zyx``.
    """
    feat = compute_basic_features(part_mask, params)
    if part_outlier or feat["erosion_splits_xy"]:
        _, maxr, nstrong = compute_dt_and_peaks(
            part_mask, params, strong_frac=params.strong_peak_frac
        )
        feat.update(max_radius_nm=float(maxr), n_strong_peaks=int(nstrong))

    score, reasons = score_instance(feat, size_outlier=part_outlier)
    if parent_outlier and not part_outlier:
        score += 1
        reasons.append("parent_cc_size_outlier")

    cz0, cy0, cx0 = offset_zyx
    cz, cy, cx = ndi.center_of_mass(part_mask)
    return {
        "instance_id": int(inst_id),
        "cell_id": int(cell_id),
        "score": int(score),
        "reasons": ";".join(reasons) if reasons else "",
        "volume_vox": int(feat["volume_vox"]),
        "max_radius_nm": float(feat["max_radius_nm"]),
        "n_strong_peaks": int(feat["n_strong_peaks"]),
        "erosion_splits_xy": int(feat["erosion_splits_xy"]),
        "centroid_z": float(cz + cz0),
        "centroid_y": float(cy + cy0),
        "centroid_x": float(cx + cx0),
    }


def _finalize_queue(rows: List[Dict]) -> pd.DataFrame:
    """Build the review queue: canonical columns, sorted by descending score."""
    queue_df = pd.DataFrame(rows, columns=REVIEW_QUEUE_COLUMNS)
    if len(queue_df):
        queue_df = queue_df.sort_values(
            ["score", "volume_vox"], ascending=[False, False]
        ).reset_index(drop=True)
    return queue_df


# ---------------------------------------------------------------------------
# Instance creation pipelines
# ---------------------------------------------------------------------------

def make_instances_from_cell_labels(
    cell_lbl_zyx: np.ndarray,
    params: HeuristicParams,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Build instance labels from a per-cell label volume.

    Connected components are found INSIDE each cell, so an instance can never
    span two cells and the size-outlier threshold is computed per cell.  Each
    instance records its parent ``cell_id`` in the review queue.

    Parameters
    ----------
    cell_lbl_zyx:
        Integer label volume, ``(z, y, x)``, where each nonzero value is one
        cell's objects.
    params:
        Heuristic settings controlling splitting and flagging behaviour.

    Returns
    -------
    inst_zyx:
        int32 instance-labeled volume, one unique id per object,
        ``LABEL_BACKGROUND`` (0) elsewhere.
    queue_df:
        Review queue with ``REVIEW_QUEUE_COLUMNS``, sorted by descending score.
    """
    cell_lbl_zyx = ensure_zyx(cell_lbl_zyx)
    inst_zyx = np.zeros_like(cell_lbl_zyx, dtype=np.int32)
    cell_bboxes = ndi.find_objects(cell_lbl_zyx)
    t0 = time.time()

    if params.verbose:
        nz = int(np.count_nonzero(cell_lbl_zyx))
        print(
            f"[start] Building instances from cell labels. "
            f"volume shape (z,y,x)={cell_lbl_zyx.shape} nonzero_voxels={nz}"
        )

    rows: List[Dict] = []
    next_id = 1

    cell_ids = [int(x) for x in np.unique(cell_lbl_zyx) if x != 0]
    if params.verbose:
        print(f"[info] Found {len(cell_ids)} cell label ids (excluding 0).")

    for cid in cell_ids:
        slc_cell = cell_bboxes[cid - 1] if (cid - 1) < len(cell_bboxes) else None
        if slc_cell is None:
            continue

        sub_mask = cell_lbl_zyx[slc_cell] == cid
        cell_vox = int(sub_mask.sum())
        if cell_vox == 0:
            continue
        if params.verbose:
            print(
                f"[cell {cid}] voxels={cell_vox} bbox={slc_cell} "
                f"... labeling connected components"
            )

        cc, n_cc = ndi.label(sub_mask, structure=STRUCT_26)
        if n_cc == 0:
            continue
        if params.verbose:
            print(f"[cell {cid}] connected components found (pre-filter) = {n_cc}")

        cc_bboxes = ndi.find_objects(cc)

        # Stage 1 — size filter + per-cell outlier threshold
        cc_ids, vols, outlier_thr = _select_components(
            cc, n_cc, cc_bboxes, params, tag=f"[cell {cid}]"
        )
        if not cc_ids:
            continue

        for k, n_vox in zip(cc_ids, vols):
            slc_cc = cc_bboxes[k - 1] if (k - 1) < len(cc_bboxes) else None
            if slc_cc is None:
                continue
            inst_crop = cc[slc_cc] == k
            is_outlier = n_vox > outlier_thr

            # Stage 2 — optional split
            was_split, parts = _split_component(
                inst_crop, n_vox, is_outlier, params, tag=f"[cell {cid}] cc {k}"
            )

            # The component bbox is relative to the cell bbox; compose the two
            # so writes and centroids land in GLOBAL volume coordinates.
            offset_zyx = (
                slc_cell[0].start + slc_cc[0].start,
                slc_cell[1].start + slc_cc[1].start,
                slc_cell[2].start + slc_cc[2].start,
            )
            gz = slice(offset_zyx[0], slc_cell[0].start + slc_cc[0].stop)
            gy = slice(offset_zyx[1], slc_cell[1].start + slc_cc[1].stop)
            gx = slice(offset_zyx[2], slc_cell[2].start + slc_cc[2].stop)

            # Stage 3 — emit one instance + one queue row per part
            sub_out = inst_zyx[gz, gy, gx]
            for part_mask, part_outlier in parts:
                inst_id = next_id
                next_id += 1
                sub_out[part_mask] = inst_id

                if params.verbose and inst_id % 250 == 0:
                    print(
                        f"[progress] created instance_id={inst_id} "
                        f"(cell {cid}) elapsed={time.time()-t0:.1f}s"
                    )

                rows.append(_make_queue_row(
                    inst_id=inst_id,
                    cell_id=cid,
                    part_mask=part_mask,
                    offset_zyx=offset_zyx,
                    params=params,
                    part_outlier=part_outlier,
                    parent_outlier=was_split and is_outlier,
                ))

            inst_zyx[gz, gy, gx] = sub_out

    queue_df = _finalize_queue(rows)

    if params.verbose:
        flagged = int((queue_df["score"] > 0).sum()) if len(queue_df) else 0
        print(
            f"[done] Built {int(inst_zyx.max())} instances from cell labels. "
            f"flagged(score>0)={flagged}. elapsed={time.time()-t0:.1f}s"
        )

    return inst_zyx, queue_df


def make_instances_from_semantic_mask(
    semantic_mask_zyx: np.ndarray,
    params: HeuristicParams,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Build instance labels from a binary semantic segmentation mask.

    The same three stages as ``make_instances_from_cell_labels``, but connected
    components are found GLOBALLY (no cell boundaries), so the size-outlier
    threshold is computed over the whole volume and every row is written with
    ``cell_id = CELL_ID_UNASSIGNED`` (0).  Assign parent cells afterwards in the
    napari session, or up front with ``--remap_from_cell``.

    Parameters
    ----------
    semantic_mask_zyx:
        Binary volume, ``(z, y, x)``; any nonzero value counts as foreground.
    params:
        Heuristic settings controlling splitting and flagging behaviour.

    Returns
    -------
    inst_zyx:
        int32 instance-labeled volume, one unique id per object,
        ``LABEL_BACKGROUND`` (0) elsewhere.
    queue_df:
        Review queue with ``REVIEW_QUEUE_COLUMNS``, sorted by descending score.
    """
    semantic_mask_zyx = ensure_zyx(semantic_mask_zyx)
    mask_zyx = semantic_mask_zyx != 0
    t0 = time.time()

    if params.verbose:
        nz = int(np.count_nonzero(mask_zyx))
        print(
            f"[start] Building instances from semantic mask. "
            f"volume shape (z,y,x)={mask_zyx.shape} nonzero_voxels={nz}"
        )

    cc, n_cc = ndi.label(mask_zyx, structure=STRUCT_26)
    if params.verbose:
        print(f"[info] Global connected components found (pre-filter) = {n_cc}")

    inst_zyx = np.zeros_like(cc, dtype=np.int32)
    cc_bboxes = ndi.find_objects(cc)

    rows: List[Dict] = []
    next_id = 1

    # Stage 1 — size filter + global outlier threshold
    cc_ids, vols, outlier_thr = _select_components(
        cc, n_cc, cc_bboxes, params, tag="[info]"
    )
    if not cc_ids:
        return inst_zyx, _finalize_queue([])

    for k, n_vox in zip(cc_ids, vols):
        slc_cc = cc_bboxes[k - 1] if (k - 1) < len(cc_bboxes) else None
        if slc_cc is None:
            continue
        inst_crop = cc[slc_cc] == k
        is_outlier = n_vox > outlier_thr

        # Stage 2 — optional split
        was_split, parts = _split_component(
            inst_crop, n_vox, is_outlier, params, tag=f"[global cc {k}]"
        )

        offset_zyx = (slc_cc[0].start, slc_cc[1].start, slc_cc[2].start)

        # Stage 3 — emit one instance + one queue row per part
        sub_out = inst_zyx[slc_cc]
        for part_mask, part_outlier in parts:
            inst_id = next_id
            next_id += 1
            sub_out[part_mask] = inst_id

            if params.verbose and inst_id % 250 == 0:
                print(
                    f"[progress] created instance_id={inst_id} "
                    f"(global cc {k}) elapsed={time.time()-t0:.1f}s"
                )

            rows.append(_make_queue_row(
                inst_id=inst_id,
                cell_id=CELL_ID_UNASSIGNED,
                part_mask=part_mask,
                offset_zyx=offset_zyx,
                params=params,
                part_outlier=part_outlier,
                parent_outlier=was_split and is_outlier,
            ))

        inst_zyx[slc_cc] = sub_out

    queue_df = _finalize_queue(rows)

    if params.verbose:
        flagged = int((queue_df["score"] > 0).sum()) if len(queue_df) else 0
        print(
            f"[done] Built {int(inst_zyx.max())} instances from semantic mask. "
            f"flagged(score>0)={flagged}. elapsed={time.time()-t0:.1f}s"
        )

    return inst_zyx, queue_df
