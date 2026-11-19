"""
napari_session.py
-----------------
Entry point for the instance-segmentation and proofreading
workflow: build instances from a mask, flag suspicious ones, review and fix
them by hand in napari, then write the corrected volume back out.

Pipeline
~~~~~~~~
    read volume ──▶ build instances ──▶ review queue ──▶ napari UI ──▶ write MRC
    (io_utils)      (segmentation)      (heuristics)     (this file)   (io_utils)

Conventions
~~~~~~~~~~~
All axis-order, unit and sentinel conventions come from ``utils``; every CLI
flag shares its name and default with the matching ``HeuristicParams`` field,
so ``--min_sep_nm 120`` and ``HeuristicParams(min_sep_nm=120)`` are the same
thing.  Run with ``--help`` for the full flag list.

Key exports
~~~~~~~~~~~
launch_napari_review()  - open the proofreading UI, return edited labels + map
main()                  - CLI entry point
"""

import argparse
import os
import time
from dataclasses import replace as _replace_params
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import watershed

# Project modules (each lives in its own file alongside this one)
from utils import (
    CELL_ID_UNASSIGNED,
    LABEL_BACKGROUND,
    REVIEW_QUEUE_COLUMNS,
    STRUCT_26,
    ensure_zyx,
    spacing_xyz_to_scale_zyx,
)
from heuristics import HeuristicParams
from segmentation import (
    make_instances_from_cell_labels,
    make_instances_from_semantic_mask,
)
from io_utils import (
    read_volume,
    write_mrc,
    load_inst_to_cell_override_from_csv,
)

# Optional UI imports (napari, magicgui, qtpy) are loaded lazily inside
# `launch_napari_review()` so the CLI works in a headless environment.


def launch_napari_review(
    inst_zyx: Optional[np.ndarray] = None,
    queue_df: Optional[pd.DataFrame] = None,
    out_path: Optional[str] = None,
    raw_zyx: Optional[np.ndarray] = None,
    raw_clim_pcts: Tuple[float, float] = (1.0, 99.0),
    raw_gamma: float = 1.0,
    cell_lbl_zyx: Optional[np.ndarray] = None,
    auto_assign_overlap: bool = False,
    inst_to_cell_override: Optional[Dict[int, int]] = None,
    spacing_nm_xyz: Tuple[float, float, float] = (5.0, 5.0, 20.0),
    params: Optional[HeuristicParams] = None,
    queue_csv_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[int, int], str]:
    """Open the napari proofreading UI and block until the window is closed.

    Parameters
    ----------
    inst_zyx:
        int32 instance labels, ``(z, y, x)``.  Edited in place by the UI.
        Pass ``None`` (along with ``out_path=None``) to have napari open
        FIRST, empty, and then show a single dialog asking for the label
        volume, the raw vEM volume (required), and a save directory — read
        below under "Interactive loading".
    queue_df:
        Review queue with ``REVIEW_QUEUE_COLUMNS``.  Rows with ``score > 0``
        become the "flagged" navigation list in the Navigation tab.  Ignored (and
        rebuilt) when ``inst_zyx`` is ``None``.
    out_path:
        Destination MRC path. Sets the default file name and folder of the
        Save MRC button, and where the screenshots folder goes.  Ignored (and resolved
        from the interactive dialog's save directory) when ``inst_zyx`` is
        ``None``.
    raw_zyx, raw_clim_pcts, raw_gamma:
        Optional grayscale vEM volume and its display settings.
    cell_lbl_zyx:
        Optional per-cell label volume, shown as the ``cell_ids`` layer and
        used for overlap-based parent assignment.
    auto_assign_overlap:
        Assign a parent cell to every unmapped instance by majority overlap
        before the window opens.
    inst_to_cell_override:
        ``{instance_id: cell_id}`` that REPLACES any mapping carried in
        ``queue_df``.
    spacing_nm_xyz:
        ``(x, y, z)`` voxel spacing in nm.  Drives the 3D aspect ratio and the
        EDT/watershed used by the in-UI split tool.
    params:
        ``HeuristicParams`` used ONLY for interactive loading (see below) —
        specifically the segmentation settings applied when the user checks
        "compute automatic watershed first". Defaults to
        ``HeuristicParams(spacing_nm_xyz=spacing_nm_xyz)`` if not given.
    queue_csv_path:
        Existing review-queue CSV to reload when interactively loading a
        precomputed label volume (mirrors ``--queue_csv``); ignored otherwise.

    Interactive loading
    --------------------
    When ``inst_zyx is None``, napari opens with no layers and a single
    dialog immediately asks for:

    - **Label** (optional) with a "Compute automatic watershed first"
      checkbox — checked: the file is a semantic mask that still needs
      segmenting, run globally with watershed-first refinement; unchecked:
      the file is already a precomputed instance-label volume, loaded as-is.
      Left blank: start from an empty label volume shaped like the raw
      volume.
    - **Cell map CSV** (optional) — an ``instance_id -> cell_id`` CSV, as
      written to ``<out>_cellmap.csv``. Replaces any mapping carried in the
      review queue, and takes precedence over ``--cell_id_csv``.
    - **Raw vEM** (required — the dialog will not accept without it).
    - **Save directory** (required) — the output MRC name is derived from
      the label/raw file name.

    Returns
    -------
    ``(edited_inst_zyx, inst_to_cell, out_path)`` — the final label volume,
    the final ``{instance_id: cell_id}`` mapping, and the output path (the
    one passed in, or the one resolved from the interactive dialog).  The
    caller is responsible for writing the volume to disk.
    """
    try:
        import napari
        from magicgui import magicgui
        from magicgui.widgets import Slider
    except Exception as e:
        raise RuntimeError(
            "Napari could not be imported. If you're on Python 3.14, napari is currently incompatible. "
            "Please use a Python 3.10–3.12 environment for --napari.\n\nOriginal error: "
            + str(e)
        )

    viewer = napari.Viewer(ndisplay=2)

    if inst_zyx is None:
        # Nothing was given yet -- napari is already open (as requested), so
        # prompt for the label/raw/save-dir now instead of before it opened.
        loaded = _run_interactive_napari_load(
            viewer,
            params or HeuristicParams(spacing_nm_xyz=spacing_nm_xyz),
            queue_csv_path=queue_csv_path,
        )
        if loaded is None:
            viewer.close()
            raise SystemExit("[napari] Load dialog canceled; nothing to review.")
        inst_zyx, queue_df, raw_zyx, out_path, dialog_override = loaded
        # A cell map chosen in the dialog wins over one passed via --cell_id_csv.
        if dialog_override:
            inst_to_cell_override = dialog_override

    # --- Arrow key slice navigation (Z axis) ---
    # The only keyboard binding kept: it steps the Z slider one voxel at a time
    # and triggers no edits, so a stray keypress cannot change the labels.
    @viewer.bind_key('Right')
    def next_slice(viewer):
        try:
            sz = float(_cur_scale[0][0])
            zi = int(round(float(viewer.dims.point[0]) / sz))
            zi = min(zi + 1, int(lab.data.shape[0]) - 1)
            viewer.dims.set_point(0, float(zi) * sz)
        except Exception:
            pass

    @viewer.bind_key('Left')
    def prev_slice(viewer):
        try:
            sz = float(_cur_scale[0][0])
            zi = int(round(float(viewer.dims.point[0]) / sz))
            zi = max(zi - 1, 0)
            viewer.dims.set_point(0, float(zi) * sz)
        except Exception:
            pass

    # Physical voxel spacing is still needed for EDT / watershed computations,
    # even though the viewer itself uses voxel-index coordinates in 2D.
    # It arrives from --spacing_nm; nothing here is hardcoded.
    spacing_nm_xyz = tuple(float(v) for v in spacing_nm_xyz)  # (x, y, z) in nm
    _sz_nm, _sy_nm, _sx_nm = spacing_xyz_to_scale_zyx(spacing_nm_xyz)

    # 2D: z scale = 1 so the dims slider steps one voxel at a time.
    # 3D: z scale = the true z spacing, for a correct physical aspect ratio.
    _scale_2d = (1.0, _sy_nm, _sx_nm)
    _scale_3d = (_sz_nm, _sy_nm, _sx_nm)
    # Mutable container so arrow keys, coord helpers and the smoothing widget
    # always resolve the CURRENTLY active scale.
    _cur_scale = [_scale_2d]

    def _get_scale_zyx() -> Tuple[float, float, float]:
        """Active (z, y, x) layer scale; changes when toggling 2D/3D."""
        return tuple(_cur_scale[0])

    scale_zyx = _scale_2d  # initial scale for layer construction

    def _world_to_zyx(pt_world):
        sz, sy, sx = _cur_scale[0]
        return (float(pt_world[0]) / sz, float(pt_world[1]) / sy, float(pt_world[2]) / sx)

    def _zyx_to_world(pt_zyx):
        sz, sy, sx = _cur_scale[0]
        return (float(pt_zyx[0]) * sz, float(pt_zyx[1]) * sy, float(pt_zyx[2]) * sx)

    # Disable scale bar because viewer now uses voxel-index coordinates.
    try:
        viewer.scale_bar.visible = False
    except Exception:
        pass

    raw_layer = None
    raw_range = {"min": None, "max": None}
    if raw_zyx is not None:
        raw_zyx = np.asarray(raw_zyx)

        # Use the full data range so we do not pre-clip the displayed histogram
        # with percentile-based contrast limits.
        raw_min = float(np.min(raw_zyx))
        raw_max = float(np.max(raw_zyx))
        raw_range["min"] = raw_min
        raw_range["max"] = raw_max

        raw_layer = viewer.add_image(
            raw_zyx,
            name="raw_vem",
            colormap="gray",
            contrast_limits=(raw_min, raw_max),
            scale=scale_zyx,
        )
        try:
            raw_layer.gamma = float(raw_gamma)
        except Exception:
            pass

        # Start hidden for huge volumes; toggle with 'r'
        if raw_zyx.size > 200_000_000:
            raw_layer.visible = False

        print(f"[raw] Added raw_vem layer. full_range=({raw_min:.3g}, {raw_max:.3g}) gamma={raw_gamma} visible={raw_layer.visible}")
    
    lab = viewer.add_labels(inst_zyx, name="instances", scale=scale_zyx)

    # Optional per-cell label volume for fast parent assignment
    cell_layer = None
    if cell_lbl_zyx is not None:
        try:
            cell_lbl_zyx = np.asarray(cell_lbl_zyx)
            cell_layer = viewer.add_labels(cell_lbl_zyx, name="cell_ids", opacity=0.35, scale=scale_zyx)
            cell_layer.visible = False  # toggle with 't'
            print("[cell] Added cell_ids layer (hidden by default). Toggle it in the Cell tab.")
        except Exception as e:
            print(f"[warn] Could not add cell_ids layer: {e}")
            cell_layer = None

    split_seeds = viewer.add_points(name="split_seeds", ndim=3, size=8)
    # Points used to select multiple labels to merge (place points inside labels)
    merge_points = viewer.add_points(name="merge_points", ndim=3, size=10)
    # Points used to assign parent cell id: point0 inside the instance, point1 inside cell (in cell_ids)
    assign_points = viewer.add_points(name="assign_points", ndim=3, size=10)


    flagged = queue_df[queue_df["score"] > 0].copy() if queue_df is not None and len(queue_df) else pd.DataFrame()
    flagged_ids = flagged["instance_id"].tolist() if len(flagged) else []
    idx = {"i": 0}

    # Keyed by instance_id, NOT by position: flagged_ids shrinks as instances are
    # merged or deleted, so a positional lookup into `flagged` would print the
    # details of a different instance than the one actually jumped to.
    flagged_info: Dict[int, Tuple[int, int, str]] = {}
    if len(flagged):
        for r in flagged[["instance_id", "cell_id", "score", "reasons"]].itertuples(index=False):
            try:
                cid = int(r.cell_id) if pd.notna(r.cell_id) else CELL_ID_UNASSIGNED
            except Exception:
                cid = CELL_ID_UNASSIGNED
            try:
                sc = int(r.score) if pd.notna(r.score) else 0
            except Exception:
                sc = 0
            reasons = "" if not pd.notna(r.reasons) else str(r.reasons)
            flagged_info[int(r.instance_id)] = (cid, sc, reasons)

    def _clamp_flagged_idx(i: int) -> int:
        """Keep the cursor inside flagged_ids, which shrinks during editing."""
        return max(0, min(int(i), len(flagged_ids) - 1))

    def _print_flagged(inst_id: int) -> None:
        cid, score, reasons = flagged_info.get(int(inst_id), (CELL_ID_UNASSIGNED, 0, ""))
        print(
            f"[{idx['i'] + 1}/{len(flagged_ids)}] instance={int(inst_id)} "
            f"cell={cid} score={score} reasons={reasons}"
        )

    # --- cell ↔ instance mappings ---
    inst_to_cell: Dict[int, int] = {}
    cell_to_insts: Dict[int, List[int]] = {}
    if queue_df is not None and len(queue_df):
        for r in queue_df[["instance_id", "cell_id"]].itertuples(index=False):
            iid = int(r.instance_id)
            cid = int(r.cell_id)
            inst_to_cell[iid] = cid
            cell_to_insts.setdefault(cid, []).append(iid)
    
    if inst_to_cell_override:
        print(f"[map] Applying override mappings ({len(inst_to_cell_override)}) from CSV")
        inst_to_cell.clear()
        cell_to_insts.clear()

        for iid, cid in inst_to_cell_override.items():
            inst_to_cell[int(iid)] = int(cid)
            cell_to_insts.setdefault(int(cid), []).append(int(iid))

        for cid in list(cell_to_insts.keys()):
            cell_to_insts[cid] = sorted(set(cell_to_insts[cid]))
    

    # --- New cell-id generator (for cells that don't exist in cell_ids yet) ---
    # We never have to edit the cell_ids volume; we just need a unique integer id to assign.
    max_cell_from_mapping = max([int(v) for v in inst_to_cell.values() if int(v) > 0], default=0)
    max_cell_from_layer = 0
    if cell_layer is not None:
        try:
            # Sample-based max to avoid full scan if huge; fall back to full max if needed
            arr = np.asarray(cell_layer.data)
            if arr.size > 5_000_000:
                rng = np.random.default_rng(0)
                sample = arr.ravel()[rng.integers(0, arr.size, size=5_000_000)]
                max_cell_from_layer = int(sample.max())
                # ensure correctness if sample suggests close to max
                if max_cell_from_layer < int(arr.max()):
                    max_cell_from_layer = int(arr.max())
            else:
                max_cell_from_layer = int(arr.max())
        except Exception:
            max_cell_from_layer = 0

    _next_cell = {"cid": int(max(max_cell_from_mapping, max_cell_from_layer)) + 1}
    _last_new_cell = {"cid": None}  # last created/used "new" cell_id for quick reuse

    def _new_cell_id() -> int:
        cid = int(_next_cell["cid"])
        _next_cell["cid"] = cid + 1
        _last_new_cell["cid"] = int(cid)
        print(f"[cell_new] Created new cell_id={cid}")
        return cid

    def _reuse_last_new_cell_id() -> int:
        """Return the most recently created/used new cell_id. If none exists, create one."""
        if _last_new_cell["cid"] is None:
            return _new_cell_id()
        return int(_last_new_cell["cid"])

    ui_state = {"cell_filter_enabled": False, "current_cell": None}
    original_color = getattr(lab, "color", None)

    # --- Performance caches ---
    print("[ui] Precomputing instance bounding boxes (one-time)...")
    t_bbox = time.time()
    label_bboxes = ndi.find_objects(np.asarray(lab.data).astype(np.int32, copy=False))
    print(f"[ui] Bounding boxes ready for labels 1..{int(np.max(lab.data))}. elapsed={time.time()-t_bbox:.1f}s")

    centroid_cache: Dict[int, Tuple[float, float, float]] = {}
    if queue_df is not None and len(queue_df) and all(c in queue_df.columns for c in ["instance_id", "centroid_z", "centroid_y", "centroid_x"]):
        for r in queue_df[["instance_id", "centroid_z", "centroid_y", "centroid_x"]].itertuples(index=False):
            try:
                iid = int(r.instance_id)
                cz = float(r.centroid_z)
                cy = float(r.centroid_y)
                cx = float(r.centroid_x)
                if np.isfinite(cz) and np.isfinite(cy) and np.isfinite(cx):
                    centroid_cache[iid] = (cz, cy, cx)
            except Exception:
                # Ignore any malformed centroid rows
                pass

    # Track instance ids from BOTH the queue (if present) and the actual volume.
    # The volume is the source of truth because you may create new labels during Napari review.
    all_instance_ids: set[int] = set()

    queue_ids: set[int] = set()
    if queue_df is not None and len(queue_df) and "instance_id" in queue_df.columns:
        try:
            queue_ids = set(int(x) for x in queue_df["instance_id"].to_numpy())
        except Exception:
            queue_ids = set(int(x) for x in queue_df["instance_id"].tolist())

    volume_ids: set[int] = set(int(x) for x in np.unique(lab.data) if int(x) != 0)

    all_instance_ids = set(queue_ids) | set(volume_ids)
    print(f"[ui] instance ids: volume={len(volume_ids)} queue={len(queue_ids)} union={len(all_instance_ids)}")

    volume_instance_ids = set(volume_ids)

    # --- Unmapped instances (final label volume driven) ---
    def _instance_voxels(iid: int) -> int:
        """Voxel count for a label id using cached bboxes; falls back to full scan only if needed."""
        iid = int(iid)
        if iid <= 0:
            return 0

        # Try cached bbox first
        slc = None
        if 0 < iid <= len(label_bboxes):
            slc = label_bboxes[iid - 1]

        data = np.asarray(lab.data)

        # If bbox missing, compute it (slow path, but only for this iid)
        if slc is None:
            mask = (data == iid)
            if not np.any(mask):
                if 0 < iid <= len(label_bboxes):
                    label_bboxes[iid - 1] = None
                return 0
            zz, yy, xx = np.where(mask)
            slc = (
                slice(int(zz.min()), int(zz.max()) + 1),
                slice(int(yy.min()), int(yy.max()) + 1),
                slice(int(xx.min()), int(xx.max()) + 1),
            )
            while len(label_bboxes) < iid:
                label_bboxes.append(None)
            label_bboxes[iid - 1] = slc

        sub = data[slc]
        return int(np.count_nonzero(sub == iid))

    # Define unmapped: present in volume and missing/0 cell_id
    unmapped_ids: List[int] = []
    tmp = []
    for iid in sorted(volume_instance_ids):
        cid = inst_to_cell.get(int(iid), None)
        if cid is None or int(cid) == 0:
            if _instance_voxels(iid) > 0:
                tmp.append(int(iid))
    unmapped_ids = tmp

    unmapped_idx = {"i": -1}
    print(f"[ui] Unmapped instances (cell_id missing/0): {len(unmapped_ids)}")

    # --- Overlap-based auto-assign ---
    def _overlap_parent_cell(iid: int) -> Tuple[int, int, int]:
        """Return (best_cell_id, best_count, inst_vox) for instance iid by majority overlap with cell_layer.

        Uses cached bboxes (label_bboxes) and computes bbox on-demand if missing.
        Avoids calling helper functions defined later in this scope.
        """
        iid = int(iid)
        if cell_layer is None or iid <= 0:
            return 0, 0, 0

        # Get bbox from cache
        slc = None
        if 0 < iid <= len(label_bboxes):
            slc = label_bboxes[iid - 1]

        inst_data = np.asarray(lab.data)

        # If bbox missing, compute it (slow path, but only for this iid)
        if slc is None:
            mask = (inst_data == iid)
            if not np.any(mask):
                if 0 < iid <= len(label_bboxes):
                    label_bboxes[iid - 1] = None
                return 0, 0, 0
            zz, yy, xx = np.where(mask)
            slc = (
                slice(int(zz.min()), int(zz.max()) + 1),
                slice(int(yy.min()), int(yy.max()) + 1),
                slice(int(xx.min()), int(xx.max()) + 1),
            )
            while len(label_bboxes) < iid:
                label_bboxes.append(None)
            label_bboxes[iid - 1] = slc

        inst_sub = inst_data[slc]
        m = (inst_sub == iid)
        inst_vox = int(np.count_nonzero(m))
        if inst_vox == 0:
            return 0, 0, 0

        cell_sub = np.asarray(cell_layer.data)[slc]
        vals = cell_sub[m].astype(np.int64, copy=False)
        if vals.size == 0:
            return 0, 0, inst_vox

        bc = np.bincount(vals)
        if bc.size <= 1:
            return 0, 0, inst_vox

        bc[0] = 0
        cid = int(bc.argmax())
        cnt = int(bc[cid]) if cid > 0 else 0
        return cid, cnt, inst_vox


    def _assign_instance_to_cell(iid: int, cid: int, reason: str = "overlap",
                                 cnt: Optional[int] = None, inst_vox: Optional[int] = None):
        iid = int(iid)
        cid = int(cid)
        if iid <= 0 or cid <= 0:
            return False

        prev = inst_to_cell.get(iid, None)
        inst_to_cell[iid] = cid
        cell_to_insts.setdefault(cid, []).append(iid)
        cell_to_insts[cid] = sorted(set(cell_to_insts[cid]))

        if prev is not None and int(prev) != cid:
            try:
                cell_to_insts[int(prev)] = [x for x in cell_to_insts.get(int(prev), []) if int(x) != iid]
            except Exception:
                pass

        try:
            if iid in unmapped_ids:
                unmapped_ids.remove(iid)
        except Exception:
            pass

        if cnt is not None and inst_vox is not None and inst_vox > 0:
            frac = float(cnt) / float(inst_vox)
            print(f"[{reason}] instance={iid} -> cell={cid} vox_in_cell={int(cnt)} frac={frac:.3f} prev={prev}")
        else:
            print(f"[{reason}] instance={iid} -> cell={cid} prev={prev}")
        return True

    def _rebuild_unmapped_ids():
        tmp2 = []
        for iid in sorted(volume_instance_ids):
            cid = inst_to_cell.get(int(iid), None)
            if cid is None or int(cid) == 0:
                if _instance_voxels(iid) > 0:
                    tmp2.append(int(iid))
        return tmp2
    
    # If a mapping CSV override was provided, ensure unmapped list reflects it now that helpers exist.
    if inst_to_cell_override:
        unmapped_ids[:] = _rebuild_unmapped_ids()
        unmapped_idx["i"] = -1
        print(f"[map] Unmapped instances after CSV load: {len(unmapped_ids)}")

    if auto_assign_overlap and cell_layer is not None and unmapped_ids:
        print(f"[auto_assign] Starting overlap assignment for {len(unmapped_ids)} unmapped instances...")
        t_auto = time.time()
        assigned = 0
        skipped = 0
        total = len(unmapped_ids)

        # iterate over a snapshot list because we remove from unmapped_ids during assignment
        for i, iid in enumerate(list(unmapped_ids), start=1):
            cid, cnt, inst_vox = _overlap_parent_cell(iid)
            if cid > 0 and cnt > 0:
                _assign_instance_to_cell(iid, cid, reason="auto_assign", cnt=cnt, inst_vox=inst_vox)
                assigned += 1
            else:
                skipped += 1

            if i % 25 == 0 or i == total:
                print(f"[auto_assign] {i}/{total} assigned={assigned} skipped={skipped} elapsed={time.time()-t_auto:.1f}s")

        # Ensure unmapped list reflects any changes
        unmapped_ids[:] = _rebuild_unmapped_ids()
        unmapped_idx["i"] = -1
        print(f"[auto_assign] Done. Remaining unmapped: {len(unmapped_ids)}")

    @magicgui(call_button="Auto-assign selected by overlap")
    def auto_assign_selected():
        if cell_layer is None:
            print("[auto_assign] No cell_ids layer loaded. Re-run with --cell.")
            return
        iid = int(lab.selected_label)
        if iid == 0:
            print("[auto_assign] Select a nonzero instance first.")
            return
        cid, cnt, inst_vox = _overlap_parent_cell(iid)
        if cid <= 0 or cnt <= 0:
            print(f"[auto_assign] No overlapping nonzero cell_id found for instance {iid}.")
            return
        _assign_instance_to_cell(iid, cid, reason="auto_assign_sel", cnt=cnt, inst_vox=inst_vox)
        update_visibility_from_selected()


    @magicgui(
        call_button="Assign selected instance -> new cell_id",
        reuse_last={"label": "Reuse last new cell_id"},
    )
    def assign_selected_to_new_cell(reuse_last: bool = False):
        """Give the selected instance a cell_id not present in cell_ids.

        Ticked: reuse the cell_id most recently created here, so several
        instances can be put in the same new cell one after another.
        """
        iid = int(lab.selected_label)
        if iid == 0:
            print("[cell_new] Select a nonzero instance first.")
            return
        if reuse_last:
            cid = _reuse_last_new_cell_id()
            reason = "assign_reuse_new_cell"
        else:
            cid = _new_cell_id()
            reason = "assign_new_cell"
        # Use existing assignment helper so mappings + unmapped list stay consistent
        _assign_instance_to_cell(iid, cid, reason=reason)
        update_visibility_from_selected()

    # --- Undo stack for script-driven edits (merge/delete/split/point-merge) ---
    # Note: Napari's native Cmd/Ctrl+Z will still undo paint/fill done with built-in tools.
    # This undo stack covers operations that directly replace lab.data.
    undo_stack: List[Dict] = []
    UNDO_MAX = 25

    def _snapshot_state(slc_zyx, op: str, meta: Optional[Dict] = None):
        """Store a cropped backup of lab.data plus small side-structures."""
        if slc_zyx is None:
            # As a fallback (rare), snapshot the whole array (can be large!)
            slc_zyx = (slice(0, lab.data.shape[0]), slice(0, lab.data.shape[1]), slice(0, lab.data.shape[2]))
            print("[undo] WARNING: snapshotting full volume; this may use lots of RAM.")

        # backup the data crop
        data = np.asarray(lab.data)
        backup = data[slc_zyx].copy()

        # backup small state so undo is consistent
        state = {
            "op": op,
            "slc": slc_zyx,
            "backup": backup,
            "inst_to_cell": inst_to_cell.copy(),
            "cell_to_insts": {k: list(v) for k, v in cell_to_insts.items()},
            "flagged_ids": list(flagged_ids),
            "all_instance_ids": set(all_instance_ids),
            "label_bboxes": list(label_bboxes),
            "ui_state": dict(ui_state),
            "meta": meta or {},
        }
        undo_stack.append(state)
        if len(undo_stack) > UNDO_MAX:
            undo_stack.pop(0)
        print(f"[undo] snapshot saved ({op}). depth={len(undo_stack)}/{UNDO_MAX}")

    def _restore_state(state: Dict):
        data = np.asarray(lab.data)
        slc = state["slc"]
        data[slc] = state["backup"]
        lab.data = data

        # restore side structures
        inst_to_cell.clear(); inst_to_cell.update(state["inst_to_cell"])
        cell_to_insts.clear(); cell_to_insts.update({k: list(v) for k, v in state["cell_to_insts"].items()})

        flagged_ids.clear(); flagged_ids.extend([int(x) for x in state["flagged_ids"]])

        all_instance_ids.clear(); all_instance_ids.update(set(state["all_instance_ids"]))

        # restore bbox cache
        label_bboxes.clear(); label_bboxes.extend(list(state["label_bboxes"]))

        # restore ui state
        ui_state.clear(); ui_state.update(dict(state["ui_state"]))

    def undo_last():
        if not undo_stack:
            print("[undo] Nothing to undo.")
            return
        state = undo_stack.pop()
        _restore_state(state)
        print(f"[undo] Undid: {state.get('op','?')} (remaining depth={len(undo_stack)})")
        # keep visibility consistent
        update_visibility_from_selected()

    def jump_to_instance(inst_id: int):
        """Jump to an instance robustly.

        Guards against NaNs coming from centroid_cache (e.g., missing/invalid centroids)
        and clamps points to valid data ranges so napari dims never receives NaN.
        """
        inst_id = int(inst_id)
        if inst_id == 0:
            return

        def _fallback_center_from_bbox() -> Optional[Tuple[float, float, float]]:
            slc = None
            if 0 < inst_id <= len(label_bboxes):
                slc = label_bboxes[inst_id - 1]
            if slc is None:
                # Try to recompute bbox if cache is stale
                try:
                    _ensure_bbox_cached(inst_id)
                except Exception:
                    pass
                if 0 < inst_id <= len(label_bboxes):
                    slc = label_bboxes[inst_id - 1]
            if slc is None:
                return None
            cz0 = 0.5 * (float(slc[0].start) + float(slc[0].stop))
            cy0 = 0.5 * (float(slc[1].start) + float(slc[1].stop))
            cx0 = 0.5 * (float(slc[2].start) + float(slc[2].stop))
            return cz0, cy0, cx0

        # Prefer centroid cache, but validate it.
        cz = cy = cx = None
        if inst_id in centroid_cache:
            cz0, cy0, cx0 = centroid_cache[inst_id]
            arr = np.array([cz0, cy0, cx0], dtype=float)
            if np.all(np.isfinite(arr)):
                cz, cy, cx = float(cz0), float(cy0), float(cx0)
            else:
                print(f"[jump] centroid for instance {inst_id} is invalid (NaN/inf). Falling back to bbox center.")

        if cz is None or cy is None or cx is None:
            fb = _fallback_center_from_bbox()
            if fb is None:
                print(f"[jump] instance {inst_id} not found (maybe merged away).")
                return
            cz, cy, cx = fb

        # Clamp to valid ranges and ensure finite
        zmax, ymax, xmax = lab.data.shape
        cz = float(np.clip(cz, 0.0, float(zmax - 1)))
        cy = float(np.clip(cy, 0.0, float(ymax - 1)))
        cx = float(np.clip(cx, 0.0, float(xmax - 1)))
        if not (np.isfinite(cz) and np.isfinite(cy) and np.isfinite(cx)):
            print(f"[jump] computed jump point is invalid for instance {inst_id}: (cz,cy,cx)=({cz},{cy},{cx})")
            return

        # Napari can behave oddly if given floats here (especially after switching 2D/3D).
        # Use integer voxel indices for stepping, and set camera center depending on ndisplay.
        iz = int(round(cz))
        iy = int(round(cy))
        ix = int(round(cx))

        wz, wy, wx = _zyx_to_world((iz, iy, ix))

        viewer.dims.set_point(0, wz)
        viewer.dims.set_point(1, wy)
        viewer.dims.set_point(2, wx)

        try:
            if int(getattr(viewer.dims, "ndisplay", 2)) == 2:
                viewer.camera.center = (wy, wx)
            else:
                viewer.camera.center = (wz, wy, wx)
        except Exception:
            pass

        lab.selected_label = int(inst_id)


    def _bbox_for_label(iid: int):
        iid = int(iid)
        if iid <= 0:
            return None
        if 0 < iid <= len(label_bboxes):
            return label_bboxes[iid - 1]
        return None

    def _ensure_bbox_cached(iid: int):
        """If bbox is missing/stale, recompute bbox for iid by scanning only the current view window is hard;
        fallback: scan full volume for this iid (slow) but only when necessary."""
        iid = int(iid)
        if iid <= 0:
            return
        if 0 < iid <= len(label_bboxes) and label_bboxes[iid - 1] is not None:
            return
        data = np.asarray(lab.data)
        mask = (data == iid)
        if not np.any(mask):
            if 0 < iid <= len(label_bboxes):
                label_bboxes[iid - 1] = None
            return
        zz, yy, xx = np.where(mask)
        slc = (slice(int(zz.min()), int(zz.max()) + 1), slice(int(yy.min()), int(yy.max()) + 1), slice(int(xx.min()), int(xx.max()) + 1))
        while len(label_bboxes) < iid:
            label_bboxes.append(None)
        label_bboxes[iid - 1] = slc


    def _label_id_at_zyx(pt_zyx: np.ndarray) -> int:
        zz, yy, xx = _world_to_zyx(pt_zyx)
        z, y, x = (int(round(zz)), int(round(yy)), int(round(xx)))
        data = np.asarray(lab.data)
        if z < 0 or y < 0 or x < 0 or z >= data.shape[0] or y >= data.shape[1] or x >= data.shape[2]:
            return 0
        return int(data[z, y, x])

    def _cell_id_at_zyx(pt_zyx: np.ndarray) -> int:
        if cell_layer is None:
            return 0
        zz, yy, xx = _world_to_zyx(pt_zyx)
        z, y, x = (int(round(zz)), int(round(yy)), int(round(xx)))
        data = np.asarray(cell_layer.data)
        if z < 0 or y < 0 or x < 0 or z >= data.shape[0] or y >= data.shape[1] or x >= data.shape[2]:
            return 0
        return int(data[z, y, x])

    def _label_ids_from_points(points_layer) -> List[int]:
        pts = np.asarray(points_layer.data)
        if pts.size == 0:
            return []
        ids: List[int] = []
        for p in pts:
            ids.append(_label_id_at_zyx(p))
        # unique, nonzero, stable order
        out: List[int] = []
        seen = set()
        for iid in ids:
            if iid != 0 and iid not in seen:
                out.append(int(iid))
                seen.add(int(iid))
        return out

    def merge_many_into(dst_id: int, src_ids: List[int]):
        """Merge each src_id into dst_id using bbox-cropped updates (fast)."""
        dst_id = int(dst_id)
        for src_id in [int(s) for s in src_ids if int(s) != dst_id]:
            # Reuse existing merge() logic by calling it; ensures mappings/flags/visibility update.
            merge(src_id=src_id, dst_id=dst_id)

    def apply_cell_visibility(cell_id: int):
        if not ui_state["cell_filter_enabled"] or cell_id is None:
            return

        keep = set(cell_to_insts.get(int(cell_id), []))
        hide = [iid for iid in all_instance_ids if iid not in keep]

        color_map = {0: (0, 0, 0, 0)}
        for hid in hide:
            color_map[int(hid)] = (0, 0, 0, 0)

        try:
            lab.color = color_map
        except Exception as e:
            print(f"[warn] Could not apply per-cell visibility filter: {e}")
            return

        ui_state["current_cell"] = int(cell_id)
        print(f"[view] Showing only cell {int(cell_id)} (kept {len(keep)} instances; hid {len(hide)}).")

    def clear_cell_visibility():
        try:
            lab.color = None
        except Exception:
            try:
                lab.color = original_color
            except Exception:
                pass
        ui_state["current_cell"] = None
        print("[view] Showing all cells (cell filter disabled).")

    def update_visibility_from_selected():
        if not ui_state["cell_filter_enabled"]:
            return
        iid = int(lab.selected_label)
        if iid == 0:
            return
        cid = inst_to_cell.get(iid, None)
        if cid is None:
            return
        apply_cell_visibility(cid)

    def split_selected_with_seeds():
        target_id = int(lab.selected_label)
        if target_id == 0:
            print("[split] Select a nonzero label first.")
            return
        pts = np.asarray(split_seeds.data)
        if pts.shape[0] < 2:
            print("[split] Add at least 2 seed points inside the object (split_seeds layer).")
            return

        data = np.asarray(lab.data)
        mask = (data == target_id)
        if not np.any(mask):
            print(f"[split] Label {target_id} not found.")
            return

        zz, yy, xx = np.where(mask)
        z0, z1 = int(zz.min()), int(zz.max()) + 1
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        x0, x1 = int(xx.min()), int(xx.max()) + 1
        # Undo snapshot (only the affected crop)
        _snapshot_state((slice(z0, z1), slice(y0, y1), slice(x0, x1)), op="split", meta={"target_id": int(target_id)})

        crop_mask = mask[z0:z1, y0:y1, x0:x1]

        markers = np.zeros(crop_mask.shape, dtype=np.int32)
        m_id = 1
        for p in pts:
            zz, yy, xx = _world_to_zyx(p)
            z, y, x = int(round(zz)) - z0, int(round(yy)) - y0, int(round(xx)) - x0
            if 0 <= z < markers.shape[0] and 0 <= y < markers.shape[1] and 0 <= x < markers.shape[2]:
                if crop_mask[z, y, x]:
                    markers[z, y, x] = m_id
                    m_id += 1

        if markers.max() < 2:
            print("[split] Seeds must land inside the selected label. Add 2+ seeds inside the object.")
            return

        markers = ndi.grey_dilation(markers, size=(3, 3, 3))

        sx, sy, sz = spacing_nm_xyz
        dist_nm = ndi.distance_transform_edt(crop_mask, sampling=(sz, sy, sx))
        labels_ws = watershed(-dist_nm, markers=markers, mask=crop_mask)

        unique_parts = [pid for pid in np.unique(labels_ws) if pid != 0]
        if len(unique_parts) < 2:
            print("[split] Watershed did not produce multiple parts; try moving/adding seeds.")
            return

        current_max = int(data.max())
        parent_cell = inst_to_cell.get(target_id, None)

        sub = data[z0:z1, y0:y1, x0:x1].copy()
        for i, pid in enumerate(unique_parts):
            part = (labels_ws == pid)
            if i == 0:
                sub[part] = target_id
            else:
                current_max += 1
                sub[part] = current_max
                all_instance_ids.add(int(current_max))

                while len(label_bboxes) < int(current_max):
                    label_bboxes.append(None)
                label_bboxes[int(current_max) - 1] = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

                if parent_cell is not None:
                    inst_to_cell[int(current_max)] = int(parent_cell)
                    cell_to_insts.setdefault(int(parent_cell), []).append(int(current_max))

        data[z0:z1, y0:y1, x0:x1] = sub
        lab.data = data

        # Approx bbox for target in the crop region
        if 0 < int(target_id) <= len(label_bboxes):
            label_bboxes[int(target_id) - 1] = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

        split_seeds.data = np.empty((0, 3), dtype=float)
        print(f"[split] Split label {target_id} into {len(unique_parts)} parts.")
        if parent_cell is not None:
            cell_to_insts[int(parent_cell)] = sorted(set(cell_to_insts[int(parent_cell)]))
        update_visibility_from_selected()

    @magicgui(call_button="Toggle cell filter")
    def toggle_cell_filter():
        """Show only the instances belonging to the selected label's cell."""
        ui_state["cell_filter_enabled"] = not ui_state["cell_filter_enabled"]
        if ui_state["cell_filter_enabled"]:
            update_visibility_from_selected()
        else:
            clear_cell_visibility()

    @magicgui(call_button="Next unmapped")
    def next_unmapped():
        if not unmapped_ids:
            print("No unmapped instances.")
            return
        unmapped_idx["i"] = min(unmapped_idx["i"] + 1, len(unmapped_ids) - 1)
        iid = int(unmapped_ids[unmapped_idx["i"]])
        jump_to_instance(iid)
        cid = inst_to_cell.get(iid, None)
        vox = _instance_voxels(iid)
        print(f"[unmapped {unmapped_idx['i']+1}/{len(unmapped_ids)}] instance={iid} cell={cid} voxels={vox}")

    @magicgui(call_button="Prev unmapped")
    def prev_unmapped():
        if not unmapped_ids:
            return
        unmapped_idx["i"] = max(unmapped_idx["i"] - 1, 0)
        iid = int(unmapped_ids[unmapped_idx["i"]])
        jump_to_instance(iid)
        cid = inst_to_cell.get(iid, None)
        vox = _instance_voxels(iid)
        print(f"[unmapped {unmapped_idx['i']+1}/{len(unmapped_ids)}] instance={iid} cell={cid} voxels={vox}")

    @magicgui(call_button="Next flagged")
    def next_flagged():
        if not flagged_ids:
            print("No flagged instances (score>0).")
            return
        idx["i"] = _clamp_flagged_idx(idx["i"] + 1)
        iid = int(flagged_ids[idx["i"]])
        jump_to_instance(iid)
        _print_flagged(iid)

    @magicgui(call_button="Prev flagged")
    def prev_flagged():
        if not flagged_ids:
            return
        idx["i"] = _clamp_flagged_idx(idx["i"] - 1)
        iid = int(flagged_ids[idx["i"]])
        jump_to_instance(iid)
        _print_flagged(iid)

    @magicgui(call_button="Jump to selected label")
    def jump_selected():
        jump_to_instance(lab.selected_label)

    def merge(src_id: int = 1, dst_id: int = 2):
        src_id = int(src_id)
        dst_id = int(dst_id)
        if src_id == dst_id:
            return

        data = np.asarray(lab.data)

        slc = None
        if 0 < src_id <= len(label_bboxes):
            slc = label_bboxes[src_id - 1]
        # Undo snapshot: only the region we will edit (src bbox crop or whole volume fallback)
        _snapshot_state(slc, op="merge", meta={"src": int(src_id), "dst": int(dst_id)})

        if slc is None:
            data[data == src_id] = dst_id
        else:
            sub = data[slc].copy()
            m = (sub == src_id)
            if np.any(m):
                sub[m] = dst_id
                data[slc] = sub

        lab.data = data
        lab.selected_label = int(dst_id)
        print(f"[merge] {src_id} -> {dst_id}")

        all_instance_ids.discard(src_id)
        all_instance_ids.add(dst_id)
        if 0 < src_id <= len(label_bboxes):
            label_bboxes[src_id - 1] = None

        # Existing mapping update logic (same as your original)
        src_c = inst_to_cell.get(int(src_id), None)
        dst_c = inst_to_cell.get(int(dst_id), None)
        if dst_c is None and src_c is not None:
            inst_to_cell[int(dst_id)] = int(src_c)
            dst_c = int(src_c)
            cell_to_insts.setdefault(dst_c, []).append(int(dst_id))
        if src_c is not None:
            inst_to_cell.pop(int(src_id), None)
            try:
                cell_to_insts[int(src_c)] = [i for i in cell_to_insts.get(int(src_c), []) if int(i) != int(src_id)]
            except Exception:
                pass
        if src_id in flagged_ids:
            try:
                flagged_ids.remove(int(src_id))
            except ValueError:
                pass
        update_visibility_from_selected()


    def _merge_all_at_points():
        ids = _label_ids_from_points(merge_points)
        if len(ids) < 2:
            print("[merge_pts] Place 2+ points inside labels in the 'merge_points' layer.")
            return

        sel = int(lab.selected_label)
        if sel != 0 and sel in ids:
            dst = sel
        else:
            dst = ids[0]

        srcs = [i for i in ids if int(i) != int(dst)]
        print(f"[merge_pts] dst={dst} srcs={srcs}")
        merge_many_into(dst, srcs)

        # Clear points after merge
        try:
            merge_points.data = np.empty((0, 3), dtype=float)
        except Exception:
            pass
        lab.selected_label = int(dst)
        print("[merge_pts] Done.")

    @magicgui(call_button="Delete selected label")
    def delete_selected():
        target_id = int(lab.selected_label)
        if target_id == 0:
            print("[delete] Select a nonzero label first.")
            return

        data = np.asarray(lab.data)

        slc = None
        if 0 < target_id <= len(label_bboxes):
            slc = label_bboxes[target_id - 1]

        # Undo snapshot: only the region we will edit (target bbox crop or whole volume fallback)
        _snapshot_state(slc, op="delete", meta={"target_id": int(target_id)})

        if slc is None:
            n = int(np.count_nonzero(data == target_id))
            data[data == target_id] = 0
        else:
            sub = data[slc].copy()
            m = (sub == target_id)
            n = int(np.count_nonzero(m))
            if n:
                sub[m] = 0
                data[slc] = sub

        # Ensure napari layer is refreshed after in-place edits
        lab.data = data
        try:
            lab.refresh()
        except Exception:
            pass

        # Update mappings
        cid = inst_to_cell.pop(target_id, None)
        if cid is not None:
            try:
                cell_to_insts[cid] = [i for i in cell_to_insts.get(cid, []) if int(i) != target_id]
            except Exception:
                pass

        # Remove from flagged list if present
        if target_id in flagged_ids:
            try:
                flagged_ids.remove(target_id)
            except ValueError:
                pass

        all_instance_ids.discard(int(target_id))
        if 0 < target_id <= len(label_bboxes):
            label_bboxes[target_id - 1] = None

        lab.selected_label = 0
        if n == 0:
            print(f"[delete] Label {target_id} had 0 voxels in its bbox (maybe already deleted or bbox stale).")
        print(f"[delete] Deleted label {target_id} (cleared {n} voxels).")
        # Re-apply filter if enabled
        update_visibility_from_selected()


    @magicgui(call_button="Undo last")
    def undo_widget():
        undo_last()




    @magicgui(call_button="Fill holes in slices")
    def fill_holes_slices():
        iid = int(lab.selected_label)
        if iid == 0:
            print("[fill] Select a nonzero label first.")
            return
        slc = _bbox_for_label(iid)
        if slc is None:
            _ensure_bbox_cached(iid)
            slc = _bbox_for_label(iid)
        if slc is None:
            print(f"[fill] No bbox for label {iid}.")
            return

        # Undo snapshot on this crop
        _snapshot_state(slc, op="fill_holes_slices", meta={"label": iid})

        data = np.asarray(lab.data)
        sub = data[slc].copy()
        changed = 0
        changed_slices = 0
        for zi in range(sub.shape[0]):
            m = (sub[zi] == iid)
            if not np.any(m):
                continue

            # Pad by 1 pixel so holes touching the bbox border are still treated as enclosed holes.
            mp = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=False)
            filledp = binary_fill_holes(mp)
            filled = filledp[1:-1, 1:-1]

            add = filled & (~m)
            if np.any(add):
                sub[zi][add] = iid
                changed += int(np.count_nonzero(add))
                changed_slices += 1

        data[slc] = sub
        lab.data = data
        try:
            lab.refresh()
        except Exception:
            pass

        # bbox unchanged, but ensure it remains cached
        label_bboxes[iid - 1] = slc
        print(f"[fill] Filled holes for label {iid}. added_voxels={changed} slices_changed={changed_slices}")
        update_visibility_from_selected()


    @magicgui(call_button="Create new label")
    def create_new_label():
        data = np.asarray(lab.data)
        new_id = int(data.max()) + 1

        # Expand caches
        while len(label_bboxes) < new_id:
            label_bboxes.append(None)
        all_instance_ids.add(new_id)

        # Assign to current cell if possible
        cur = int(lab.selected_label)
        parent_cell = None
        if cur != 0:
            parent_cell = inst_to_cell.get(cur, None)
        if parent_cell is not None:
            inst_to_cell[new_id] = int(parent_cell)
            cell_to_insts.setdefault(int(parent_cell), []).append(int(new_id))
            cell_to_insts[int(parent_cell)] = sorted(set(cell_to_insts[int(parent_cell)]))

        lab.selected_label = int(new_id)
        print(f"[new] Created new label id={new_id} cell={parent_cell}")
        update_visibility_from_selected()


    @magicgui(call_button="Assign parent cell from points")
    def assign_parent_cell():
        if cell_layer is None:
            print("[assign] No cell_ids layer loaded. Re-run with --cell to enable fast assignment.")
            return

        pts = np.asarray(assign_points.data)
        if pts.shape[0] >= 2:
            iid = _label_id_at_zyx(pts[0])
            cid = _cell_id_at_zyx(pts[1])
        elif pts.shape[0] == 1:
            iid = int(lab.selected_label)
            cid = _cell_id_at_zyx(pts[0])
        else:
            print("[assign] Add points in 'assign_points': point0 inside the instance, point1 inside the cell (or select the instance and place 1 point in the cell).")
            return

        if int(iid) == 0:
            print("[assign] Could not determine instance id (iid=0).")
            return
        if int(cid) == 0:
            print("[assign] Could not determine cell id (cid=0). Make sure point is inside a nonzero cell label.")
            return

        prev = inst_to_cell.get(int(iid), None)
        inst_to_cell[int(iid)] = int(cid)
        cell_to_insts.setdefault(int(cid), []).append(int(iid))
        cell_to_insts[int(cid)] = sorted(set(cell_to_insts[int(cid)]))

        # If it belonged to a previous cell, remove from that list
        if prev is not None and int(prev) != int(cid):
            try:
                cell_to_insts[int(prev)] = [x for x in cell_to_insts.get(int(prev), []) if int(x) != int(iid)]
            except Exception:
                pass

        # Update unmapped list if present
        try:
            if int(iid) in unmapped_ids:
                unmapped_ids.remove(int(iid))
        except Exception:
            pass

        print(f"[assign] instance={int(iid)} -> cell={int(cid)} (prev={prev})")

        # Clear points after assignment
        try:
            assign_points.data = np.empty((0, 3), dtype=float)
        except Exception:
            pass

        lab.selected_label = int(iid)
        update_visibility_from_selected()


    @magicgui(call_button="Toggle cell_ids layer")
    def toggle_cell_ids_layer():
        if cell_layer is None:
            print("[cell] No cell_ids layer loaded.")
            return
        cell_layer.visible = not bool(cell_layer.visible)
        print(f"[cell] cell_ids visible={cell_layer.visible}")


    _default_screenshot_dir = Path(os.path.dirname(out_path) or ".") / "napari_screenshots"

    @magicgui(
        call_button="Screenshot PNG + SVG 600 dpi",
        filename={"label": "File name (no extension)"},
        save_dir={"widget_type": "FileEdit", "mode": "d", "label": "Save folder"},
    )
    def export_screenshot_600dpi(
        filename: str = "",
        save_dir: Path = _default_screenshot_dir,
    ):
        screenshot_dir = str(save_dir)
        os.makedirs(screenshot_dir, exist_ok=True)

        name = filename.strip() or f"napari_screenshot_{time.strftime('%Y%m%d_%H%M%S')}"
        base = os.path.join(screenshot_dir, name)
        png_path = base + ".png"
        svg_path = base + ".svg"

        # Always full resolution (no supersampling option — was unused beyond 1x).
        try:
            img = viewer.screenshot(canvas_only=True, scale=1)
        except TypeError:
            img = viewer.screenshot(scale=1)

        img = np.asarray(img)
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            print(f"[screenshot] Unexpected screenshot shape: {img.shape}")
            return

        try:
            from PIL import Image
            import base64
            import io

            mode = "RGBA" if img.shape[2] == 4 else "RGB"
            pil_img = Image.fromarray(img.astype(np.uint8), mode=mode)

            # Save PNG with 600 dpi metadata
            pil_img.save(png_path, dpi=(600, 600))

            # Make an SVG that embeds the same PNG at 600 dpi physical sizing
            width_px, height_px = pil_img.size
            width_in = width_px / 600.0
            height_in = height_px / 600.0
            width_pt = width_in * 72.0
            height_pt = height_in * 72.0

            png_buffer = io.BytesIO()
            pil_img.save(png_buffer, format="PNG", dpi=(600, 600))
            encoded = base64.b64encode(png_buffer.getvalue()).decode("ascii")

            svg_text = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    width="{width_in:.6f}in"
    height="{height_in:.6f}in"
    viewBox="0 0 {width_pt:.6f} {height_pt:.6f}">
<image width="{width_pt:.6f}" height="{height_pt:.6f}" x="0" y="0"
        preserveAspectRatio="none"
        xlink:href="data:image/png;base64,{encoded}" />
</svg>
'''
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg_text)

            print(f"[screenshot] Saved PNG 600 dpi: {png_path}")
            print(f"[screenshot] Saved SVG with embedded 600 dpi PNG: {svg_path}")

        except Exception as e:
            print(f"[screenshot] Failed to save screenshot: {e}")

    @magicgui(
        call_button="Apply raw intensity window",
        low={"widget_type": Slider, "min": 0, "max": 1, "step": 1},
        high={"widget_type": Slider, "min": 0, "max": 1, "step": 1},
    )
    def set_raw_intensity_window(low: int = 0, high: int = 1):
        """Adjust the displayed raw intensity window for the raw_vem layer."""
        if raw_layer is None:
            print("[raw_window] No raw_vem layer loaded.")
            return
        try:
            lo = int(low)
            hi = int(high)
            if hi < lo:
                lo, hi = hi, lo
            raw_layer.contrast_limits = (lo, hi)
            print(f"[raw_window] contrast_limits set to ({lo:.3g}, {hi:.3g})")
        except Exception as e:
            print(f"[warn] Could not set raw intensity window: {e}")

    @magicgui(call_button="Reset raw intensity window")
    def reset_raw_intensity_window():
        """Reset raw_vem contrast limits to the full raw data range."""
        if raw_layer is None or raw_range["min"] is None or raw_range["max"] is None:
            print("[raw_window] No raw_vem layer loaded.")
            return
        raw_layer.contrast_limits = (float(raw_range["min"]), float(raw_range["max"]))
        try:
            set_raw_intensity_window.low.value = float(raw_range["min"])
            set_raw_intensity_window.high.value = float(raw_range["max"])
        except Exception:
            pass
        print(f"[raw_window] reset to full range ({float(raw_range['min']):.3g}, {float(raw_range['max']):.3g})")
    
    if raw_layer is not None and raw_range["min"] is not None and raw_range["max"] is not None:
        try:
            rmin = int(raw_range["min"])
            rmax = int(raw_range["max"])

            set_raw_intensity_window.low.min = rmin
            set_raw_intensity_window.low.max = rmax
            set_raw_intensity_window.low.step = 1
            set_raw_intensity_window.low.value = rmin

            set_raw_intensity_window.high.min = rmin
            set_raw_intensity_window.high.max = rmax
            set_raw_intensity_window.high.step = 1
            set_raw_intensity_window.high.value = rmax
        except Exception:
            pass

    def _merge_cc_at_seed():
        pts = np.asarray(merge_points.data)
        if pts.shape[0] < 1:
            print("[merge_cc] Place 1 point inside the SOURCE label in 'merge_points'.")
            return

        seed = pts[0]
        seed_vox = _world_to_zyx(seed)
        src_id = _label_id_at_zyx(seed)
        if src_id == 0:
            print("[merge_cc] Seed landed on background (0).")
            return

        dst_sel = int(lab.selected_label)
        if dst_sel != 0 and dst_sel != src_id:
            dst_id = dst_sel
        else:
            # If no selected destination, use the first non-src id found under remaining points
            ids = _label_ids_from_points(merge_points)
            dst_id = None
            for i in ids:
                if int(i) != int(src_id):
                    dst_id = int(i)
                    break
            if dst_id is None:
                print("[merge_cc] Select a destination label (different from src) OR add a second point inside the DEST label.")
                return

        if int(dst_id) == int(src_id):
            print("[merge_cc] src and dst are the same.")
            return

        slc = _bbox_for_label(src_id)
        if slc is None:
            _ensure_bbox_cached(src_id)
            slc = _bbox_for_label(src_id)
        if slc is None:
            print(f"[merge_cc] No bbox for src label {src_id}.")
            return

        # Undo snapshot for the region we'll edit
        _snapshot_state(slc, op="merge_cc", meta={"src": int(src_id), "dst": int(dst_id)})

        data = np.asarray(lab.data)
        sub = data[slc].copy()
        src_mask = (sub == int(src_id))
        if not np.any(src_mask):
            print(f"[merge_cc] src label {src_id} not found in its bbox (bbox stale?).")
            return

        # Connected components within src mask
        cc, ncc = ndi.label(src_mask, structure=STRUCT_26)
        if ncc < 1:
            print("[merge_cc] No CCs found.")
            return

        # Seed in crop coordinates
        z0, y0, x0 = slc[0].start, slc[1].start, slc[2].start
        sz, sy, sx = int(round(float(seed_vox[0])) - z0), int(round(float(seed_vox[1])) - y0), int(round(float(seed_vox[2])) - x0)
        if sz < 0 or sy < 0 or sx < 0 or sz >= cc.shape[0] or sy >= cc.shape[1] or sx >= cc.shape[2]:
            print("[merge_cc] Seed outside src bbox.")
            return
        comp_id = int(cc[sz, sy, sx])
        if comp_id == 0:
            print("[merge_cc] Seed not inside the src label component.")
            return

        comp_mask = (cc == comp_id)
        moved = int(np.count_nonzero(comp_mask))
        sub[comp_mask] = int(dst_id)
        data[slc] = sub
        lab.data = data
        try:
            lab.refresh()
        except Exception:
            pass

        # Update mappings/sets: dst keeps its cell; src may become empty
        all_instance_ids.add(int(dst_id))
        all_instance_ids.discard(int(src_id))  # may add back if still exists

        # Recompute whether src still exists within its bbox
        remains = int(np.count_nonzero(sub == int(src_id)))
        if remains > 0:
            all_instance_ids.add(int(src_id))
            label_bboxes[int(src_id) - 1] = slc
        else:
            if 0 < int(src_id) <= len(label_bboxes):
                label_bboxes[int(src_id) - 1] = None
            src_c = inst_to_cell.pop(int(src_id), None)
            if src_c is not None:
                try:
                    cell_to_insts[int(src_c)] = [i for i in cell_to_insts.get(int(src_c), []) if int(i) != int(src_id)]
                except Exception:
                    pass
            if int(src_id) in flagged_ids:
                try:
                    flagged_ids.remove(int(src_id))
                except ValueError:
                    pass

        lab.selected_label = int(dst_id)
        print(f"[merge_cc] Merged CC from src={src_id} into dst={dst_id}. moved_voxels={moved} src_remains={remains}")
        update_visibility_from_selected()

        # Clear merge points
        try:
            merge_points.data = np.empty((0, 3), dtype=float)
        except Exception:
            pass

    @magicgui(
        call_button="Merge labels at points",
        piece_only={"label": "Only the connected piece under point 1"},
    )
    def merge_labels(piece_only: bool = False):
        """Merge the labels under the points in 'merge_points'.

        Unticked: every label under a point is merged into one (the selected
        label if it is among them, otherwise the label under point 1).
        Ticked: only the connected piece of the label under point 1 moves,
        into the selected label or the label under point 2.
        """
        if piece_only:
            _merge_cc_at_seed()
        else:
            _merge_all_at_points()

    @magicgui(call_button="Split selected label using seeds")
    def split_selected():
        split_selected_with_seeds()

    # Tracks whether an MRC has actually been written during this session, so
    # the close-confirmation dialog (see below, near napari.run()) knows
    # whether to warn. Only a successful "Save MRC" click sets this — the
    # automatic write-on-close in main() is a fallback, not a guarantee.
    save_state = {"mrc_saved": False, "last_path": None}

    _default_out_dir = Path(os.path.dirname(out_path) or ".")
    _default_out_name = os.path.splitext(os.path.basename(out_path))[0]

    @magicgui(
        call_button="Save MRC",
        filename={"label": "File name (no extension)"},
        save_dir={"widget_type": "FileEdit", "mode": "d", "label": "Save folder"},
    )
    def save_mrc(
        filename: str = _default_out_name,
        save_dir: Path = _default_out_dir,
    ):
        """Write the current instance labels as an MRC file right now.

        Unlike the automatic save that happens after this window closes, this
        writes immediately to a folder you choose and reports whether it
        worked — use it before closing if you don't trust the destination
        given to ``--output`` (e.g. a network drive that sometimes drops).
        """
        name = filename.strip() or _default_out_name
        mrc_path = os.path.join(str(save_dir), name + ".mrc")
        try:
            os.makedirs(str(save_dir), exist_ok=True)
            write_mrc(mrc_path, np.ascontiguousarray(lab.data), voxel_size_nm_xyz=spacing_nm_xyz)
            save_state["mrc_saved"] = True
            save_state["last_path"] = mrc_path
            print(f"[save] MRC written: {mrc_path}")
        except Exception as e:
            print(f"[save] FAILED to write MRC to {mrc_path}: {e}")
            return
        map_path = os.path.join(str(save_dir), name + "_cellmap.csv")
        try:
            pd.DataFrame({"instance_id": list(inst_to_cell.keys()),
                          "cell_id": list(inst_to_cell.values())}).to_csv(map_path, index=False)
            print(f"[save] Cell map written: {map_path}")
        except Exception as e:
            print(f"[save] FAILED to write cell map to {map_path}: {e}")

    from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget

    # Create tab container
    tabs = QTabWidget()

    def make_tab(widgets):
        container = QWidget()
        layout = QVBoxLayout()
        for w in widgets:
            layout.addWidget(w.native)
        container.setLayout(layout)
        return container

    # --- Define tabs ---
    navigation_tab = make_tab([
        next_flagged,
        prev_flagged,
        next_unmapped,
        prev_unmapped,
        jump_selected,
    ])

    editing_tab = make_tab([
        merge_labels,
        split_selected,
        delete_selected,
        undo_widget,
        fill_holes_slices,
        create_new_label,
    ])

    cell_tab = make_tab([
        assign_parent_cell,
        auto_assign_selected,
        assign_selected_to_new_cell,
        toggle_cell_ids_layer,
        toggle_cell_filter,
    ])


    # ---------------------------------------------------------------------------
    # ChimeraX-style Laplacian mesh smoothing (see smoothing.py)
    # ---------------------------------------------------------------------------
    from smoothing import make_smooth_widget
    # Pass the getter, not a snapshot, so smoothed surfaces stay registered
    # with the labels layer after a 2D/3D toggle changes the scale.
    smooth_labels_widget = make_smooth_widget(viewer, lab, _get_scale_zyx)

    view_tab_widgets = [smooth_labels_widget]
    if raw_layer is not None:
        view_tab_widgets.extend([set_raw_intensity_window, reset_raw_intensity_window])
    view_tab_widgets.append(export_screenshot_600dpi)
    view_tab = make_tab(view_tab_widgets)

    save_tab = make_tab([
        save_mrc,
    ])

    # Add tabs
    tabs.addTab(navigation_tab, "Navigation")
    tabs.addTab(editing_tab, "Editing")
    tabs.addTab(cell_tab, "Cell")
    tabs.addTab(view_tab, "View")
    tabs.addTab(save_tab, "Save")

    # Add tab widget once
    viewer.window.add_dock_widget(tabs, area="right")

    if flagged_ids:
        idx["i"] = 0
        jump_to_instance(int(flagged_ids[0]))
        _print_flagged(int(flagged_ids[0]))
    else:
        print("No flagged instances (score>0). You can still browse/merge manually.")

    try:
        lab.events.selected_label.connect(lambda event: update_visibility_from_selected())
    except Exception:
        pass

    # --- Switch layer scale between 2D (voxel-index z) and 3D (physical z) ---
    def _on_ndisplay_change(event=None):
        nd = int(getattr(viewer.dims, "ndisplay", 2))
        new_scale = _scale_3d if nd == 3 else _scale_2d
        if new_scale == _cur_scale[0]:
            return
        # Convert current world-z to voxel index before changing scale
        try:
            old_sz = float(_cur_scale[0][0])
            zi = int(round(float(viewer.dims.point[0]) / old_sz))
            zi = int(np.clip(zi, 0, int(lab.data.shape[0]) - 1))
        except Exception:
            zi = 0
        _cur_scale[0] = new_scale
        for layer in viewer.layers:
            try:
                layer.scale = new_scale
            except Exception:
                pass
        try:
            viewer.dims.set_point(0, float(zi) * float(new_scale[0]))
        except Exception:
            pass
        print(f"[view] {'3D' if nd == 3 else '2D'} scale_z={f'{_scale_3d[0]:g} nm' if nd == 3 else '1 (voxel index)'}")

    try:
        viewer.dims.events.ndisplay.connect(_on_ndisplay_change)
    except Exception:
        pass

    update_visibility_from_selected()

    print(
        "\nNapari controls (all actions are buttons in the docked tabs on the right):\n"
        "  Navigation : next/prev flagged instance, next/prev unmapped instance, jump to selected label\n"
        "  Editing    : merge (at points / connected component at a seed), split with seeds, delete,\n"
        "               undo last action, fill holes per slice, create a new label\n"
        "  Cell       : assign parent cell (from assign_points, or by overlap), assign to a new cell_id\n"
        "               (or reuse the last new one), toggle the cell_ids layer, toggle the per-cell filter\n"
        "  View       : Laplacian surface smoothing, raw intensity window, 600 dpi PNG + SVG screenshot\n"
        "  Save       : 'Save MRC' writes the MRC + cell map CSV now, to a folder you choose\n"
        "\n"
        "  Left/Right arrow keys: step one Z slice at a time (the only keyboard shortcut).\n"
        "  Layer visibility and the 2D/3D toggle are napari's own controls.\n"
        "  Bottom slider: uses voxel Z index directly (0..Z-1)\n"
        "  Merge: use the paint brush, OR place points in 'merge_points' and click 'Merge labels at points'\n"
        "  Split: select a label, add 2+ points in the 'split_seeds' layer, then click 'Split selected label using seeds'\n"
        "  Final MRC is ALSO written automatically after you close Napari, but that alone is not guaranteed --\n"
        "  you'll be asked to confirm on close if you have not used 'Save MRC' yet this session.\n"
        "  Hover ANY layer: bottom-left overlay shows 'Label ID: xxx; cell ID: yyy' for the instance under the cursor.\n"
    )

    # --- Label ID / cell ID hover overlay ------------------------------------
    # Always reads the instance id out of the `instances` volume and its cell
    # from the live inst_to_cell mapping, so the readout is the same whichever
    # layer you happen to be hovering.
    viewer.text_overlay.visible = True
    viewer.text_overlay.font_size = 10

    def _overlay_text_for(position) -> str:
        """'Label ID: x; cell ID: y' for the instance under `position`, or ''."""
        try:
            iid = int(_label_id_at_zyx(position))
        except Exception:
            return ""
        if iid <= 0:
            return ""
        return f"Label ID: {iid}; cell ID: {int(inst_to_cell.get(iid, CELL_ID_UNASSIGNED))}"

    # --- overlay colour: dark text on bright data, light text on dark data ---
    # Keyed by (slice, contrast limits) so the brightness estimate is computed
    # once per view rather than on every mouse move.
    _overlay_color_cache: Dict[Tuple[int, float, float], str] = {}

    def _theme_overlay_color() -> str:
        """Colour for the plain canvas background, from the napari theme."""
        try:
            return "black" if "light" in str(viewer.theme).lower() else "white"
        except Exception:
            return "white"

    def _overlay_color() -> str:
        """Pick a colour that contrasts with whatever is behind the readout."""
        # With no visible raw layer, or in 3D, the canvas shows the theme
        # background rather than image data, so let the theme decide.
        if raw_layer is None or not raw_layer.visible:
            return _theme_overlay_color()
        try:
            if int(getattr(viewer.dims, "ndisplay", 2)) != 2:
                return _theme_overlay_color()

            data = np.asarray(raw_layer.data)
            zi = int(round(float(viewer.dims.point[0]) / float(_cur_scale[0][0])))
            zi = int(np.clip(zi, 0, data.shape[0] - 1))
            lo, hi = (float(v) for v in raw_layer.contrast_limits)
            if not (hi > lo):
                return _theme_overlay_color()

            key = (zi, lo, hi)
            cached = _overlay_color_cache.get(key)
            if cached is not None:
                return cached

            # Coarse subsample: a full slice is far too much work per move.
            sample = data[zi][::8, ::8]
            level = (float(np.median(sample)) - lo) / (hi - lo)
            color = "black" if level > 0.5 else "white"

            if len(_overlay_color_cache) > 512:   # bound the cache
                _overlay_color_cache.clear()
            _overlay_color_cache[key] = color
            return color
        except Exception:
            return _theme_overlay_color()

    def _on_cursor_move(event=None):
        viewer.text_overlay.text = _overlay_text_for(viewer.cursor.position)
        viewer.text_overlay.color = _overlay_color()

    # viewer.cursor.position changes on every mouse move over the canvas, no
    # matter which layer is selected. Per-layer mouse_move_callbacks only fire
    # for the ACTIVE layer, so the readout used to vanish the moment you
    # selected a different layer -- hence the viewer-level hook here.
    viewer.text_overlay.color = _overlay_color()   # before the first mouse move

    _overlay_hooked = False
    try:
        viewer.cursor.events.position.connect(_on_cursor_move)
        _overlay_hooked = True
        print("[ui] Label/cell overlay follows the cursor across all layers.")
    except Exception as e:
        print(f"[warn] Could not hook the cursor overlay ({e}); using per-layer callbacks instead.")

    if not _overlay_hooked:
        # Fallback for napari versions without an evented cursor: attach to
        # every layer, so at least the active one always reports.
        def _layer_overlay_cb(target_layer, event):
            viewer.text_overlay.text = _overlay_text_for(event.position)
            viewer.text_overlay.color = _overlay_color()

        def _attach_label_cell_overlay(target_layer):
            try:
                if _layer_overlay_cb not in target_layer.mouse_move_callbacks:
                    target_layer.mouse_move_callbacks.append(_layer_overlay_cb)
            except Exception as e:
                # Never silent: a layer the overlay skipped is exactly the
                # layer the readout will look broken on.
                print(f"[warn] overlay not attached to layer '{getattr(target_layer, 'name', '?')}': {e}")

        for _existing_layer in list(viewer.layers):
            _attach_label_cell_overlay(_existing_layer)

        @viewer.layers.events.inserted.connect
        def _on_layer_inserted_attach_overlay(event):
            _attach_label_cell_overlay(event.value)

    # --- Warn on close if no MRC has actually been saved this session ---
    # The automatic write-on-close in main() can silently fail (e.g. a broken
    # or unmounted output directory); this catches that before the window
    # goes away, rather than after.
    try:
        qt_main_window = viewer.window._qt_window
    except Exception:
        qt_main_window = None

    if qt_main_window is not None:
        from qtpy.QtWidgets import QMessageBox

        _orig_close_event = qt_main_window.closeEvent

        def _guarded_close_event(event, _orig=_orig_close_event):
            if not save_state["mrc_saved"]:
                resp = QMessageBox.warning(
                    qt_main_window,
                    "MRC not saved",
                    "You have not used 'Save MRC' in this session.\n\n"
                    "Closing now relies entirely on the automatic save to:\n"
                    f"{out_path}\n\n"
                    "If that location is unavailable (e.g. a disconnected drive), "
                    "your edits will be lost.\n\n"
                    "Close anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if resp != QMessageBox.Yes:
                    event.ignore()
                    return
            _orig(event)

        qt_main_window.closeEvent = _guarded_close_event
    else:
        print("[warn] Could not attach close-confirmation dialog (Qt main window not found).")

    napari.run()

    # Return edited labels + mapping so it can be saved by CLI
    return np.asarray(lab.data), dict(inst_to_cell), out_path

# ----------------------------
# Interactive loading for --napari: napari opens first (empty), THEN a single
# dialog asks for the label volume, the raw vEM volume, and a save directory.
# ----------------------------
def _resolve_interactive_load(
    label_path: str,
    use_watershed: bool,
    raw_path: str,
    save_dir: str,
    params: HeuristicParams,
    queue_csv_path: Optional[str] = None,
    cellmap_csv_path: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, str, Optional[Dict[int, int]]]:
    """Read the files chosen in the load dialog and build instances/queue.

    Pure data step (no Qt) so it can be exercised without a running dialog.
    ``label_path`` may be empty (start from an empty label volume). Exactly
    mirrors the equivalent CLI-driven branches in ``main()``.

    Returns ``(inst_zyx, queue_df, raw_zyx, out_path, inst_to_cell_override)``;
    the last is ``None`` unless a cell map CSV was chosen.
    """
    print(f"[step] Reading raw volume: {raw_path}")
    raw_raw, raw_vox, raw_dtype = read_volume(raw_path)
    print(f"[io] Read raw volume: shape={raw_raw.shape} dtype={raw_dtype} header_voxel_size(angstrom)={raw_vox}")
    raw_zyx = ensure_zyx(raw_raw)

    default_stem = os.path.splitext(os.path.basename(label_path or raw_path))[0] + "_instances"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, default_stem + ".mrc")
    queue_path = os.path.splitext(out_path)[0] + "_review_queue.csv"

    if label_path and use_watershed:
        print(f"[step] Reading label volume for segmentation: {label_path}")
        lbl_raw, lbl_vox, lbl_dtype = read_volume(label_path)
        lbl_zyx = ensure_zyx(lbl_raw)
        seg_params = _replace_params(params, watershed_first=True)
        print("[mode] interactive load: semantic mask -> global connected components, watershed_first=True")
        inst_zyx, queue_df = make_instances_from_semantic_mask(lbl_zyx, seg_params)
        queue_df.to_csv(queue_path, index=False)
        print(f"[done] Review queue written: {queue_path}")
        auto_path = os.path.splitext(out_path)[0] + "_AUTO.mrc"
        write_mrc(auto_path, inst_zyx, voxel_size_nm_xyz=seg_params.spacing_nm_xyz)
        print(f"[io] Saved AUTO instances before napari: {auto_path}")
    elif label_path:
        print(f"[step] Loading precomputed instance labels: {label_path}")
        inst_raw, inst_vox, inst_dtype = read_volume(label_path)
        inst_zyx = ensure_zyx(inst_raw)
        if queue_csv_path and os.path.exists(queue_csv_path):
            queue_df = pd.read_csv(queue_csv_path)
            print(f"[io] Loaded review queue: {queue_csv_path} rows={len(queue_df)}")
        else:
            queue_df = pd.DataFrame(columns=REVIEW_QUEUE_COLUMNS)
            print("[io] No review queue CSV given/found; starting an empty review queue.")
    else:
        print("[mode] no label provided -> starting from an empty label volume matching raw shape")
        inst_zyx = np.full_like(raw_zyx, LABEL_BACKGROUND, dtype=np.int32)
        queue_df = pd.DataFrame(columns=REVIEW_QUEUE_COLUMNS)

    # An instance -> cell_id map REPLACES whatever mapping the queue carries,
    # matching --cell_id_csv on the command line.
    inst_to_cell_override = None
    if cellmap_csv_path:
        print(f"[io] Loading instance→cell mapping from: {cellmap_csv_path}")
        inst_to_cell_override = load_inst_to_cell_override_from_csv(cellmap_csv_path)

    return inst_zyx, queue_df, raw_zyx, out_path, inst_to_cell_override


def _run_interactive_napari_load(
    viewer,
    params: HeuristicParams,
    queue_csv_path: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, pd.DataFrame, np.ndarray, str, Optional[Dict[int, int]]]]:
    """Show the single load dialog on top of the already-open ``viewer``.

    Returns ``(inst_zyx, queue_df, raw_zyx, out_path, inst_to_cell_override)``,
    or ``None`` if the user canceled.
    """
    from qtpy.QtWidgets import (
        QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
        QFormLayout, QHBoxLayout, QLineEdit, QMessageBox, QPushButton,
        QVBoxLayout, QWidget,
    )

    _ = QApplication.instance() or QApplication([])
    vol_filter = "Volumes (*.mrc *.rec *.tif *.tiff *.nii *.nii.gz *.npy);;All files (*)"
    csv_filter = "CSV files (*.csv);;All files (*)"

    dialog = QDialog()
    dialog.setWindowTitle("Load files for review")
    layout = QVBoxLayout(dialog)
    form = QFormLayout()

    def _path_row(browse_fn):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: browse_fn(edit))
        h.addWidget(edit)
        h.addWidget(btn)
        return row, edit

    def _browse_open(edit, title, file_filter=None):
        path, _sel = QFileDialog.getOpenFileName(dialog, title, "", file_filter or vol_filter)
        if path:
            edit.setText(path)

    def _browse_dir(edit, title):
        path = QFileDialog.getExistingDirectory(dialog, title)
        if path:
            edit.setText(path)

    label_row, label_edit = _path_row(lambda e: _browse_open(e, "Select label volume"))
    watershed_checkbox = QCheckBox("Compute automatic watershed first")
    label_container = QWidget()
    label_v = QVBoxLayout(label_container)
    label_v.setContentsMargins(0, 0, 0, 0)
    label_v.addWidget(label_row)
    label_v.addWidget(watershed_checkbox)
    form.addRow("Label (optional):", label_container)

    cellmap_row, cellmap_edit = _path_row(
        lambda e: _browse_open(e, "Select cell map CSV (instance_id -> cell_id)", csv_filter)
    )
    form.addRow("Cell map CSV (optional):", cellmap_row)

    raw_row, raw_edit = _path_row(lambda e: _browse_open(e, "Select raw vEM volume"))
    form.addRow("Raw vEM (required):", raw_row)

    dir_row, dir_edit = _path_row(lambda e: _browse_dir(e, "Select save directory"))
    form.addRow("Save directory (required):", dir_row)

    layout.addLayout(form)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    layout.addWidget(buttons)

    def _try_accept():
        if not raw_edit.text().strip():
            QMessageBox.warning(dialog, "Raw vEM required", "Raw vEM is required — please choose a file.")
            return
        if not dir_edit.text().strip():
            QMessageBox.warning(dialog, "Save directory required", "Please choose a save directory.")
            return
        dialog.accept()

    buttons.accepted.connect(_try_accept)
    buttons.rejected.connect(dialog.reject)

    if dialog.exec_() != QDialog.Accepted:
        return None

    return _resolve_interactive_load(
        label_path=label_edit.text().strip(),
        use_watershed=watershed_checkbox.isChecked(),
        raw_path=raw_edit.text().strip(),
        save_dir=dir_edit.text().strip(),
        params=params,
        queue_csv_path=queue_csv_path,
        cellmap_csv_path=cellmap_edit.text().strip() or None,
    )


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto_assign_overlap", action="store_true", help="If --cell is provided, auto-assign parent cell_id for unmapped instances by majority overlap before starting Napari UI.")
    ap.add_argument("--cell", default=None, help="Optional per-cell label volume (.mrc/.tif/.tiff/.nii/.nii.gz, each cell is one integer id). Used for fast parent assignment in Napari.")
    ap.add_argument("--remap_from_cell", action="store_true", help="Recompute instance->cell mapping for ALL instances by majority overlap with --cell before starting Napari. Useful if existing cell_id mappings are wrong.")
    ap.add_argument("--input", default=None, help="Optional input volume (.mrc/.tif/.tiff/.nii/.nii.gz). Can be a cell-label volume, semantic mask (with --semantic), or omitted entirely to start with an empty labels volume.")
    ap.add_argument("--semantic", action="store_true", help="Interpret --input as a semantic mask (0 background, nonzero = foreground) and build instances globally using connected components first, then conservative split heuristics.")
    ap.add_argument("--load_instances", default=None, help="Path to a precomputed instance-label volume (.mrc/.tif/.tiff/.nii/.nii.gz, e.g., *_AUTO.mrc). If provided, skips instance computation and loads this file instead.")
    ap.add_argument("--output", default=None, help="Output MRC path for instance labels. Required unless --napari prompts for it interactively.")
    ap.add_argument("--queue_csv", default=None, help="CSV path for review queue (default: output stem + _review_queue.csv).")
    ap.add_argument("--napari", action="store_true", help="Launch Napari proofreading UI on flagged instances.")
    ap.add_argument("--spacing_nm", nargs=3, type=float, default=[5.0, 5.0, 20.0], help="Voxel spacing in nm: x y z")
    ap.add_argument("--min_sep_nm", type=float, default=120.0, help="Peak min separation (nm) for multi-lobe suspicion.")
    ap.add_argument("--strong_peak_frac", type=float, default=0.70, help="Peak strength threshold as fraction of max radius.")
    ap.add_argument("--erosion_xy_iters", type=int, default=1, help="XY-only erosion iterations for thin-contact suspicion.")
    ap.add_argument("--size_outlier_k_mad", type=float, default=3.0, help="Size outlier threshold: median + k*MAD within cell.")
    ap.add_argument("--min_instance_voxels", type=int, default=100, help="Drop CCs smaller than this.")
    ap.add_argument("--watershed_first", action="store_true", help="Watershed-first refinement: split suspicious/large CCs into instances before UI review.")
    ap.add_argument("--ws_min_peaks", type=int, default=2, help="Minimum strong peaks to attempt watershed split.")
    ap.add_argument("--ws_max_seeds", type=int, default=4, help="Maximum watershed seeds per component.")
    ap.add_argument("--ws_min_voxels", type=int, default=10000, help="Only attempt watershed on components with at least this many voxels.")
    ap.add_argument("--ws_strong_peak_frac", type=float, default=0.75, help="Seed peaks >= frac * max distance for watershed.")
    ap.add_argument("--raw", default=None, help="Optional raw vEM volume (.mrc/.tif/.tiff/.nii/.nii.gz) to display as grayscale in Napari (bypasses napari file readers).")
    ap.add_argument("--raw_clim_pcts", nargs=2, type=float, default=[1.0, 99.0], help="Raw contrast percentiles, e.g. 1 99")
    ap.add_argument("--raw_gamma", type=float, default=1.0, help="Gamma for raw_vem display")
    ap.add_argument("--pre_split_xy_erosion_iters", "--pre_split_xy_erosion", dest="pre_split_xy_erosion_iters", type=int, default=0, help="Fast pre-split for huge touching clusters: XY-only erosion iterations (0 disables; try 1). Name matches HeuristicParams.pre_split_xy_erosion_iters.")
    ap.add_argument("--pre_split_apply_over_voxels", type=int, default=1000000, help="Only pre-split components with at least this many voxels.")
    # Instance -> cell mapping override.  One concept, one flag:
    # --load_cellmap_csv is kept as a deprecated alias of --cell_id_csv.
    ap.add_argument("--cell_id_csv", "--load_cellmap_csv", dest="cell_id_csv", default=None, help="CSV mapping instance_id -> cell_id. REPLACES any mapping carried in the review queue.")
    ap.add_argument("--cell_id_csv_instance_col", default="instance_id", help="Instance-id column name in --cell_id_csv.")
    ap.add_argument("--cell_id_csv_cell_col", default="cell_id", help="Cell-id column name in --cell_id_csv.")

    args = ap.parse_args()

    has_label = args.input is not None or args.load_instances is not None
    fully_specified = has_label and args.raw is not None and args.output is not None
    # --remap_from_cell needs a fully-resolved inst_zyx before napari even
    # launches, so it's incompatible with interactive loading -- that advanced
    # flag always requires full CLI specification.
    need_interactive = bool(args.napari) and not fully_specified and not args.remap_from_cell

    if not args.output and not need_interactive:
        ap.error(
            "--output is required (pass it directly, or use --napari alone / "
            "with --input, --raw, or --output left out, to get the interactive "
            "load dialog after napari opens)."
        )

    print("[run] napari_session.py")
    print(f"[run] input={args.input}")
    print(f"[run] output={args.output}")
    print(f"[run] napari={args.napari}")
    print(f"[run] spacing_nm(x,y,z)={args.spacing_nm}")
    print(f"[run] min_sep_nm={args.min_sep_nm} strong_peak_frac={args.strong_peak_frac} erosion_xy_iters={args.erosion_xy_iters} size_outlier_k_mad={args.size_outlier_k_mad} min_instance_voxels={args.min_instance_voxels}")
    print(f"[run] watershed_first={args.watershed_first} ws_min_peaks={args.ws_min_peaks} ws_max_seeds={args.ws_max_seeds} ws_min_voxels={args.ws_min_voxels} ws_strong_peak_frac={args.ws_strong_peak_frac}")
    print(f"[run] pre_split_xy_erosion_iters={args.pre_split_xy_erosion_iters} pre_split_apply_over_voxels={args.pre_split_apply_over_voxels}")
    print(f"[run] load_instances={args.load_instances}")
    print(f"[run] raw={args.raw}")
    print(f"[run] raw_clim_pcts={args.raw_clim_pcts} raw_gamma={args.raw_gamma}")
    print(f"[run] cell={args.cell}")
    print(f"[run] remap_from_cell={args.remap_from_cell}")
    print(f"[run] auto_assign_overlap={args.auto_assign_overlap}")

    # Parameters used for feature computation (only needed when computing instances)
    params = HeuristicParams(
        spacing_nm_xyz=(float(args.spacing_nm[0]), float(args.spacing_nm[1]), float(args.spacing_nm[2])),
        min_sep_nm=float(args.min_sep_nm),
        strong_peak_frac=float(args.strong_peak_frac),
        erosion_xy_iters=int(args.erosion_xy_iters),
        size_outlier_k_mad=float(args.size_outlier_k_mad),
        min_instance_voxels=int(args.min_instance_voxels),
        verbose=True,
        watershed_first=bool(args.watershed_first),
        ws_min_peaks=int(args.ws_min_peaks),
        ws_max_seeds=int(args.ws_max_seeds),
        ws_min_voxels=int(args.ws_min_voxels),
        ws_strong_peak_frac=float(args.ws_strong_peak_frac),
        pre_split_xy_erosion_iters=int(args.pre_split_xy_erosion_iters),
        pre_split_apply_over_voxels=int(args.pre_split_apply_over_voxels),
    )

    if need_interactive:
        print("[run] --input/--raw/--output not all given: napari will open and prompt for them interactively.")
        out_path = None
        queue_path = None
        raw_zyx = None
        inst_zyx = None
        queue_df = None
    else:
        out_path = args.output
        queue_path = args.queue_csv or (os.path.splitext(out_path)[0] + "_review_queue.csv")

        raw_zyx = None
        if args.raw:
            print(f"[step] Reading raw volume: {args.raw}")
            raw_raw, raw_vox, raw_dtype = read_volume(args.raw)
            print(f"[io] Read raw volume: shape={raw_raw.shape} dtype={raw_dtype} header_voxel_size(angstrom)={raw_vox}")
            raw_zyx = ensure_zyx(raw_raw)

        if args.load_instances:
            # Load precomputed instance labels and an existing queue CSV.
            inst_path = args.load_instances
            print(f"[step] Loading precomputed instances from: {inst_path}")
            inst_raw, inst_vox, inst_dtype = read_volume(inst_path)
            print(f"[io] Read instances MRC: shape={inst_raw.shape} dtype={inst_dtype} header_voxel_size(angstrom)={inst_vox}")
            inst_zyx = ensure_zyx(inst_raw)

            if args.queue_csv is None or not os.path.exists(args.queue_csv):
                print(f"[warn] Queue CSV not provided or not found at {args.queue_csv}. Continuing without review queue.")
                queue_df = pd.DataFrame()
                print("[io] Created empty review queue DataFrame.")
            else:
                queue_df = pd.read_csv(args.queue_csv)
                print(f"[io] Loaded review queue: {queue_path} rows={len(queue_df)}")

        elif args.input:
            in_path = args.input
            print(f"[step] Reading input per-cell labels: {in_path}")
            lbl_raw, mrc_vox, dtype = read_volume(in_path)
            print(f"[io] Read volume: shape={lbl_raw.shape} dtype={dtype} header_voxel_size(angstrom)={mrc_vox}")
            if any(np.isnan(x) for x in mrc_vox):
                print("[warn] Could not read voxel size from file header; proceeding with --spacing_nm.")

            # Assume MRC data comes as (z,y,x). This is the standard for mrcfile.
            lbl_zyx = ensure_zyx(lbl_raw)

            t0 = time.time()
            print("[step] Creating instance labels + review queue...")
            if args.semantic:
                if args.input is None:
                    raise ValueError("--semantic requires --input")
                if params.verbose:
                    print("[mode] semantic=True -> global connected components + conservative watershed")
                inst_zyx, queue_df = make_instances_from_semantic_mask(lbl_zyx, params)
            else:
                if params.verbose:
                    print("[mode] semantic=False -> per-cell connected components")
                inst_zyx, queue_df = make_instances_from_cell_labels(lbl_zyx, params)

            print(f"[timing] Instance creation finished in {time.time()-t0:.1f}s")
            print(f"[done] Instances created. Max instance id = {int(inst_zyx.max())}")

            queue_df.to_csv(queue_path, index=False)
            print(f"[done] Review queue written: {queue_path}")
            print(f"[info] Flagged (score>0): {int((queue_df['score']>0).sum())} / {len(queue_df)}")

            # Always save an auto result before launching napari (so long compute isn't lost)
            auto_path = os.path.splitext(out_path)[0] + "_AUTO.mrc"
            print(f"[io] Saving AUTO instances before napari: {auto_path}")
            write_mrc(auto_path, inst_zyx, voxel_size_nm_xyz=params.spacing_nm_xyz)

        else:
            if raw_zyx is None:
                raise ValueError("When neither --input nor --load_instances is provided, you must provide --raw so the script knows what volume shape to use.")
            if params.verbose:
                print("[mode] no input labels provided -> starting from empty labels volume matching --raw shape")
            inst_zyx = np.full_like(raw_zyx, LABEL_BACKGROUND, dtype=np.int32)
            queue_df = pd.DataFrame(columns=REVIEW_QUEUE_COLUMNS)

    cell_lbl_zyx = None
    if args.cell:
        print(f"[step] Reading cell-id volume: {args.cell}")
        cell_raw, cell_vox, cell_dtype = read_volume(args.cell)
        print(f"[io] Read cell-id volume: shape={cell_raw.shape} dtype={cell_dtype} header_voxel_size(angstrom)={cell_vox}")
        cell_lbl_zyx = ensure_zyx(cell_raw)

    # ----------------------------
    # Optional: remap ALL instance->cell ids using overlap with cell_lbl_zyx
    # (cell_lbl_zyx should be the per-cell object group label volume)
    # ----------------------------
    if args.remap_from_cell:
        if cell_lbl_zyx is None:
            raise ValueError("--remap_from_cell requires --cell")
        if cell_lbl_zyx.shape != inst_zyx.shape:
            raise ValueError(f"--cell shape {cell_lbl_zyx.shape} does not match instances shape {inst_zyx.shape}")

        print("[remap] Recomputing instance->cell mapping by majority overlap with --cell (ALL instances)...")
        t_remap = time.time()

        # Ensure queue has a row for every instance id in the current volume.
        vol_ids = [int(x) for x in np.unique(inst_zyx) if int(x) != 0]
        if queue_df is None or not len(queue_df):
            queue_df = pd.DataFrame({"instance_id": vol_ids})
        else:
            # Add missing ids
            q_ids = set(int(x) for x in queue_df["instance_id"].astype(int).tolist())
            missing = [iid for iid in vol_ids if int(iid) not in q_ids]
            if missing:
                queue_df = pd.concat([queue_df, pd.DataFrame({"instance_id": missing})], ignore_index=True)

        if "cell_id" not in queue_df.columns:
            queue_df["cell_id"] = 0
        queue_df["instance_id"] = queue_df["instance_id"].astype(int)
        queue_df["cell_id"] = queue_df["cell_id"].fillna(0).astype(int)

        # Precompute bounding boxes for speed
        print("[remap] Precomputing bounding boxes (one-time)...")
        label_bboxes = ndi.find_objects(inst_zyx.astype(np.int32, copy=False))

        assigned = 0
        unassigned = 0
        mapping: Dict[int, int] = {}

        inst_data = inst_zyx.astype(np.int32, copy=False)
        cell_data = cell_lbl_zyx.astype(np.int32, copy=False)

        for iid in vol_ids:
            iid = int(iid)
            slc = None
            if 0 < iid <= len(label_bboxes):
                slc = label_bboxes[iid - 1]
            if slc is None:
                # No voxels or bbox not found
                mapping[iid] = 0
                unassigned += 1
                continue

            inst_sub = inst_data[slc]
            m = (inst_sub == iid)
            inst_vox = int(np.count_nonzero(m))
            if inst_vox == 0:
                mapping[iid] = 0
                unassigned += 1
                continue

            cell_sub = cell_data[slc]
            vals = cell_sub[m]
            if vals.size == 0:
                mapping[iid] = 0
                unassigned += 1
                continue

            bc = np.bincount(vals.astype(np.int64, copy=False))
            if bc.size <= 1:
                mapping[iid] = 0
                unassigned += 1
                continue

            bc[0] = 0
            cid = int(bc.argmax())
            cnt = int(bc[cid]) if cid > 0 else 0
            if cid > 0 and cnt > 0:
                mapping[iid] = cid
                assigned += 1
            else:
                mapping[iid] = 0
                unassigned += 1

        # Apply to queue_df
        queue_df["cell_id"] = queue_df["instance_id"].map(lambda x: int(mapping.get(int(x), 0))).fillna(0).astype(int)

        # Print sanity
        uniq_cells = sorted(set(int(x) for x in queue_df["cell_id"].tolist() if int(x) != 0))
        print(f"[remap] Done. assigned={assigned} unassigned={unassigned} unique_nonzero_cell_ids={len(uniq_cells)}")
        if len(uniq_cells) <= 50:
            print(f"[remap] cell_ids={uniq_cells}")
        print(f"[remap] elapsed={time.time()-t_remap:.1f}s")

    # ----------------------------
    # Optional: instance -> cell mapping override CSV.
    # Applied inside launch_napari_review(), so it wins over any mapping
    # carried in the review queue.
    # ----------------------------
    inst_to_cell_override = None

    if args.cell_id_csv is not None:
        print(f"[io] Loading instance→cell mapping from: {args.cell_id_csv}")
        inst_to_cell_override = load_inst_to_cell_override_from_csv(
            csv_path=args.cell_id_csv,
            instance_col=args.cell_id_csv_instance_col,
            cell_col=args.cell_id_csv_cell_col,
        )

    inst_to_cell_after_ui = None
    if args.napari:
        edited, inst_to_cell_after_ui, resolved_out_path = launch_napari_review(
            inst_zyx,
            queue_df,
            out_path,
            raw_zyx=raw_zyx,
            raw_clim_pcts=(float(args.raw_clim_pcts[0]), float(args.raw_clim_pcts[1])),
            raw_gamma=float(args.raw_gamma),
            cell_lbl_zyx=cell_lbl_zyx,
            auto_assign_overlap=bool(args.auto_assign_overlap),
            inst_to_cell_override=inst_to_cell_override,
            spacing_nm_xyz=params.spacing_nm_xyz,
            params=params,
            queue_csv_path=args.queue_csv,
        )
        inst_zyx = edited
        out_path = resolved_out_path  # may differ from args.output when chosen interactively
        if queue_path is None:
            queue_path = args.queue_csv or (os.path.splitext(out_path)[0] + "_review_queue.csv")
    # ----------------------------
    # Save mapping (instance_id -> cell_id) after Napari edits
    # ----------------------------
    if inst_to_cell_after_ui is not None:
        # Build a mapping dataframe for instances actually present in the final volume.
        final_ids = [int(x) for x in np.unique(inst_zyx) if int(x) != 0]
        rows = []
        unmapped = 0
        for iid in sorted(final_ids):
            cid = inst_to_cell_after_ui.get(int(iid), CELL_ID_UNASSIGNED)
            cid = int(cid) if cid is not None else CELL_ID_UNASSIGNED
            if cid == CELL_ID_UNASSIGNED:
                unmapped += 1
            rows.append({"instance_id": int(iid), "cell_id": int(cid)})

        map_df = pd.DataFrame(rows)
        map_path = os.path.splitext(out_path)[0] + "_cellmap.csv"
        map_df.to_csv(map_path, index=False)
        print(f"[map] Wrote FINAL instance->cell mapping: {map_path} rows={len(map_df)} unmapped={unmapped}")

        # Update (or extend) the queue CSV so it reflects the edited mappings.
        try:
            q = queue_df.copy() if (queue_df is not None and len(queue_df)) else pd.DataFrame()
            if len(q) and "instance_id" in q.columns:
                q["instance_id"] = q["instance_id"].astype(int)
                # map existing rows
                q["cell_id"] = q["instance_id"].map(lambda x: int(inst_to_cell_after_ui.get(int(x), 0))).astype(int)
                existing = set(int(x) for x in q["instance_id"].tolist())
            else:
                existing = set()

            # add new instances created during UI that were not in the original queue
            new_ids = [iid for iid in map_df["instance_id"].tolist() if int(iid) not in existing]
            if new_ids:
                add = map_df[map_df["instance_id"].isin(new_ids)].copy()
                # minimal columns for new rows
                add["score"] = 0
                add["reasons"] = ""
                add["volume_vox"] = 0
                add["max_radius_nm"] = -1.0
                add["n_strong_peaks"] = 0
                add["erosion_splits_xy"] = 0
                add["centroid_z"] = np.nan
                add["centroid_y"] = np.nan
                add["centroid_x"] = np.nan
                q = pd.concat([q, add], ignore_index=True, sort=False)

            # Write updated queue
            q.to_csv(queue_path, index=False)
            print(f"[queue] Updated queue CSV with mappings: {queue_path} rows={len(q)}")
        except Exception as e:
            print(f"[warn] Could not update queue CSV with mappings: {e}")

    print("[io] Writing output instance MRC...")
    write_mrc(out_path, inst_zyx, voxel_size_nm_xyz=params.spacing_nm_xyz)
    print(f"[done] Output instance MRC written: {out_path}")


if __name__ == "__main__":
    main()
    