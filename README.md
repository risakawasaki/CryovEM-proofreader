# Cryo-vEM proofreader

Semi-automated **instance segmentation and human proofreading** for cryogenic volume EM (FIB-SEM / cryo-vEM) data.

The tool takes a binary semantic mask (or a per-cell label volume), turns it into *instances* — one integer id per object — flags the ones most likely to be wrong, and opens a [napari](https://napari.org) session where you fix them by hand. The corrected volume is written back out as MRC alongside a CSV recording which object belongs to which cell.

It is built for the case where automatic segmentation is good but not good enough: the failure mode is almost always **two touching objects merged into one label**, and the fix is almost always a human looking at it for three seconds.

> Developed and tested on mitochondria in retinal photoreceptors. Nothing in
> the splitting or scoring assumes them — the heuristics are generic
> blob-separation, so any densely packed, roughly convex structure should
> behave the same way.

---

## Contents

- [Why this exists](#why-this-exists)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [The review queue](#the-review-queue)
- [The napari session](#the-napari-session)
- [Command-line reference](#command-line-reference)
- [Module map](#module-map)
- [Project conventions](#project-conventions)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Why this exists

A semantic segmentation model tells you *where* the objects are. It does not tell you *how many* there are. Converting a mask to instances with connected components merges every pair of objects that touch at a single voxel — and in densely packed tissue, such as mitochondria in photoreceptor inner segments, that is a lot of them.

Fully automatic instance separation over-splits as often as it under-splits, and both errors are invisible unless someone looks. So this workflow does three things:

1. **Splits conservatively.** Merge errors are cheaper to fix by hand than split errors, so the automatic stage under-splits on purpose.
2. **Ranks what to look at.** Every instance gets a score from cheap geometric heuristics. You review the top of the list, not the whole volume.
3. **Makes the fix fast.** Merge, split, delete, fill and assign are all one click in napari, with undo.

---

## Installation

Python **3.10–3.12**. (napari does not yet support 3.13+; the non-UI parts of the pipeline run fine on any modern Python.)

```bash
git clone https://github.com/<you>/CryovEM-proofreader.git
cd CryovEM-proofreader

conda create -n vem python=3.11
conda activate vem

pip install -r requirements.txt
```

Everything except the napari UI runs without the UI packages. `tifffile` and
`nibabel` are only needed for TIFF and NIfTI input respectively.

`requirements.txt` installs **PySide6** as the Qt backend. Any qtpy backend
works — PyQt5, PyQt6 and PySide2 are all fine — but PySide6 and PySide2 are
LGPL, whereas PyQt is GPL or a paid commercial licence, which is worth knowing
if you redistribute this alongside one.

The modules sit flat in one directory and import each other by name, so run the CLI from inside the repo (or put the directory on `PYTHONPATH`).

---

## Quick start

**Interactive** — let napari open first and choose the files there:

```bash
python napari_session.py --napari
```

napari opens empty, then one dialog asks for the label volume, the raw vEM
volume and a save directory. See [Loading files interactively](#loading-files-interactively).

**Semantic mask in, proofread instances out:**

```bash
python napari_session.py \
    --input  semantic_mask.mrc \
    --semantic \
    --raw    dataset.mrc \
    --output instances.mrc \
    --spacing_nm 5 5 20 \
    --napari
```

**Per-cell mask** (each cell's objects already carry that cell's id) — drop `--semantic`:

```bash
python napari_session.py \
    --input  labels_by_cell.mrc \
    --raw    dataset.mrc \
    --output instances.mrc \
    --spacing_nm 5 5 20 \
    --napari
```

**Resume a session** without recomputing instances:

```bash
python napari_session.py \
    --load_instances instances_AUTO.mrc \
    --queue_csv      instances_review_queue.csv \
    --raw            dataset.mrc \
    --output         instances_v2.mrc \
    --napari
```

Outputs, all derived from `--output` — or from the save directory you pick in the load dialog:

| File | Contents |
|---|---|
| `<out>.mrc` | final instance labels, written after napari closes |
| `<out>_AUTO.mrc` | automatic instances, written *before* the UI opens so a long compute is never lost |
| `<out>_review_queue.csv` | one row per instance, scores and features |
| `<out>_cellmap.csv` | final `instance_id → cell_id` mapping |

---

## How it works

```
  input volume            instances                  review                 output
 ┌────────────┐   ┌───────────────────────┐   ┌────────────────┐   ┌───────────────────┐
 │ semantic   │──▶│ connected components  │──▶│ score + rank   │──▶│ napari: merge /   │
 │ mask       │   │ size filter           │   │ (heuristics)   │   │ split / assign    │
 │  ─ or ─    │   │ optional split        │   │                │   │                   │
 │ cell labels│   │ (pre-split, watershed)│   │ review_queue   │   │ ──▶ instances.mrc │
 └────────────┘   └───────────────────────┘   └────────────────┘   └───────────────────┘
   io_utils            segmentation              heuristics            napari_session
```

Both entry points — per-cell and semantic — run the **same three stages** through the same helpers, so they cannot drift apart:

**Stage 1 — select components.** Label connected components (26-connectivity), drop anything below `--min_instance_voxels`, then compute a size-outlier threshold as `median + k · MAD` over the components that survived. In per-cell mode this happens *inside each cell*, so a cell full of small objects isn't judged against a cell full of large ones.

**Stage 2 — split (optional, off by default).** Two strategies, tried in order:

- **Pre-split** (`--pre_split_xy_erosion_iters`): erode in XY only, treat each surviving piece as a seed, assign every voxel to its nearest seed. No distance transform, so it is cheap enough for components of millions of voxels. Erosion is XY-only because anisotropic vEM data has far coarser Z sampling — eroding in Z would destroy thin structures.
- **Watershed-first** (`--watershed_first`): compute a spacing-aware Euclidean distance transform, find peaks at least `--min_sep_nm` apart, keep those above `--ws_strong_peak_frac` of the maximum radius, cap them at `--ws_max_seeds`, and watershed. Only triggered for components that are multi-lobed *and* fall apart under erosion, or that are size outliers.

**Stage 3 — score and emit.** Each resulting part becomes one instance and one review-queue row. The distance transform is only computed for parts that already look suspicious, which is why most rows carry `-1` in the EDT columns.

### Scoring

| Signal | Points | Meaning |
|---|---|---|
| `erosion_splits_xy` | +3 | one XY erosion breaks it in two — a thin contact, probably two objects |
| `n_strong_peaks ≥ 2` | +2 per extra peak | multi-lobed in the distance transform |
| `size_outlier` | +2 | volume above `median + k · MAD` |
| `parent_cc_size_outlier` | +1 | this part came from splitting an oversized component |

Anything with `score > 0` is **flagged** and enters the **Next / Prev flagged** navigation list in the UI. Higher score means look sooner; the queue is sorted by score, then by volume.

---

## The review queue

A CSV with one row per instance, written before the UI opens and updated after it closes. Columns are fixed and identical everywhere they appear (`utils.REVIEW_QUEUE_COLUMNS`), so a queue CSV can be fed back into any stage of the workflow.

| Column | Type | Notes |
|---|---|---|
| `instance_id` | int | `> 0`, unique in the volume |
| `cell_id` | int | `0` = unassigned |
| `score` | int | `> 0` = flagged |
| `reasons` | str | `;`-joined labels, empty when unflagged |
| `volume_vox` | int | voxel count |
| `max_radius_nm` | float | max EDT radius; **`-1` = not computed** |
| `n_strong_peaks` | int | strong EDT peaks; **`-1` = not computed** |
| `erosion_splits_xy` | int | 0 or 1 |
| `centroid_z/y/x` | float | voxel index, global volume coordinates |

`-1` and `0` mean different things: `-1` means the feature was deliberately skipped for speed, `0` means it was computed and came out zero. Filter on `>= 0` before doing statistics on these columns.

---

## The napari session

### Loading files interactively

If you pass `--napari` without all of `--input`/`--load_instances`, `--raw` and
`--output`, napari opens **first**, empty, and a single dialog asks for:

| Field | Notes |
|---|---|
| **Label** | Optional. Leave blank to start from an empty label volume shaped like the raw volume. |
| ☐ **Compute automatic watershed first** | Ticked: the label is a semantic mask that still needs segmenting — instances are built with global connected components plus watershed-first refinement. Unticked: the label is already an instance volume (e.g. a previous `*_AUTO.mrc`) and is loaded as-is. |
| **Cell map CSV** | Optional. An `instance_id → cell_id` CSV, such as the `<out>_cellmap.csv` written by an earlier session. Replaces any mapping carried in the review queue, and takes precedence over `--cell_id_csv`. |
| **Raw vEM** | **Required.** The dialog will not accept without it. |
| **Save directory** | **Required.** The output MRC name is derived from the label (or raw) file name. |

Ticking the checkbox runs the segmentation before the layers appear, and writes
`<out>_AUTO.mrc` and `<out>_review_queue.csv` on the way, exactly as the
equivalent command line would.

Give all of `--input`/`--load_instances`, `--raw` and `--output` on the command
line and no dialog appears, so scripted runs are unaffected. `--remap_from_cell`
needs a volume in hand before napari opens, so it always requires the full set.

### Layers

Six layers open:

| Layer | What it is |
|---|---|
| `instances` | the editable instance labels — this is the one you paint |
| `raw_vem` | grayscale EM, hidden automatically for volumes over 200 M voxels |
| `cell_ids` | optional per-cell labels, hidden by default |
| `merge_points`, `split_seeds`, `assign_points` | point layers that drive the merge / split / assign tools |

### Controls

Every action is a button in the docked tabs on the right. The only keyboard
shortcut is <kbd>←</kbd> / <kbd>→</kbd>, which steps one Z slice at a time —
it triggers no edits, so a stray keypress cannot change your labels. Layer
visibility and the 2D/3D toggle are napari's own controls.

**Navigation**

| Button | Action |
|---|---|
| Next / Prev flagged | step through instances the heuristics scored `> 0`, worst first |
| Next / Prev unmapped | step through instances with no `cell_id` |
| Jump to selected label | centre the view on whatever label is selected |

**Editing**

| Button | Action |
|---|---|
| Merge labels at points | merge all labels under points in `merge_points`. Tick **Only the connected piece under point 1** to move just that piece of the label under point 1 into the selected label (or the label under point 2) |
| Split selected label using seeds | split using ≥ 2 points in `split_seeds` |
| Delete selected label | set it to background |
| Undo last | undo the last scripted action |
| Fill holes in slices | fill holes, per Z slice, in the selected label |
| Create new label | create a new label id |

**Cell**

| Button | Action |
|---|---|
| Assign parent cell from points | point 0 in the object, point 1 in the cell |
| Auto-assign selected by overlap | majority overlap with `cell_ids` |
| Assign selected instance -> new cell_id | for cells missing from `cell_ids`. Tick **Reuse last new cell_id** to put several instances into the same new cell |
| Toggle cell_ids layer | show/hide the `cell_ids` layer |
| Toggle cell filter | show only instances belonging to the selected label's cell |

> Hovering **any** layer shows `Label ID: xxx; cell ID: yyy` for the instance
> under the cursor, in the overlay at the corner of the canvas. The text
> switches between black and white to stay legible against whatever is behind
> it — judged from the displayed brightness of the current `raw_vem` slice, or
> from the napari theme when no raw layer is visible.

### Dock panels

**Navigation · Editing · Cell · View · Save** — the buttons listed above, plus:

- **View → Laplacian smoothing**: ChimeraX-style mesh smoothing. Extracts a surface with marching cubes and moves each vertex a `factor` of the way toward the average of its edge-connected neighbours, `iterations` times. The result is added as a **new Surface layer** — the label voxels are never modified, so editing tools keep working and you can toggle between the blocky truth and the smooth rendering.
- **View → raw intensity window**: contrast limits on the grayscale layer.
- **View → Screenshot PNG + SVG 600 dpi**: publication-resolution export; choose the file name and folder before exporting.
- **Save**: `Save MRC` writes the MRC and its `_cellmap.csv` now, to a folder you choose. An MRC is also written automatically after you close the window, and you are warned on close if you never saved one.

> **Z scale changes between 2D and 3D.** In 2D the Z scale is 1 so the slider steps one voxel at a time; in 3D it becomes the real Z spacing so the volume has the correct aspect ratio. This is handled automatically, including for smoothed surfaces.

---

## Command-line reference

Every flag shares its name and default with the matching `HeuristicParams` field, so `--min_sep_nm 120` and `HeuristicParams(min_sep_nm=120)` are the same thing.

### Input / output

| Flag | Default | Description |
|---|---|---|
| `--input` | — | Input volume. A per-cell label volume, or a semantic mask with `--semantic`. Omit to start from empty labels (then `--raw` is required). |
| `--semantic` | off | Treat `--input` as a binary mask; build instances globally instead of per cell. |
| `--load_instances` | — | Load precomputed instances (e.g. `*_AUTO.mrc`) and skip the compute. |
| `--output` | — | Output MRC path; every sidecar filename derives from it. Required **unless** you use `--napari`, which asks for a save directory when this is missing. |
| `--queue_csv` | `<out>_review_queue.csv` | Review queue path. |
| `--raw` | — | Grayscale vEM volume to display. |
| `--cell` | — | Per-cell label volume for parent assignment. |
| `--napari` | off | Open the proofreading UI. Without it, the pipeline runs headless. If the input/raw/output flags are not all given, napari opens first and asks for them. |

Formats: `.mrc` / `.rec`, `.tif` / `.tiff` (file or directory of slices), `.nii` / `.nii.gz`, `.npy`.

### Geometry

| Flag | Default | Description |
|---|---|---|
| `--spacing_nm X Y Z` | `5 5 20` | Voxel spacing in nm. **Always set this** — file headers are not trusted. |
| `--min_sep_nm` | `120.0` | Minimum peak separation for multi-lobe detection. |
| `--strong_peak_frac` | `0.70` | A peak counts as strong at this fraction of the max radius. |
| `--erosion_xy_iters` | `1` | XY erosion iterations for thin-contact detection. |
| `--size_outlier_k_mad` | `3.0` | Outlier threshold, `median + k · MAD`. |
| `--min_instance_voxels` | `100` | Drop components smaller than this. |

### Splitting

| Flag | Default | Description |
|---|---|---|
| `--watershed_first` | off | Enable EDT watershed refinement. |
| `--ws_min_peaks` | `2` | Minimum strong peaks to attempt a watershed. |
| `--ws_max_seeds` | `4` | Cap on seeds per component. |
| `--ws_min_voxels` | `10000` | Only watershed components at least this large. |
| `--ws_strong_peak_frac` | `0.75` | Seed peaks at this fraction of the max distance. |
| `--pre_split_xy_erosion_iters` | `0` | XY erosion pre-split; `0` disables, `1`–`2` typical. |
| `--pre_split_apply_over_voxels` | `1000000` | Only pre-split components at least this large. |

`--pre_split_xy_erosion` is a deprecated alias of `--pre_split_xy_erosion_iters`.

### Cell assignment

| Flag | Default | Description |
|---|---|---|
| `--auto_assign_overlap` | off | Assign unmapped instances by majority overlap before the UI opens. |
| `--remap_from_cell` | off | Recompute the mapping for **all** instances. Requires `--cell`. Use when existing mappings are wrong. |
| `--cell_id_csv` | — | CSV mapping `instance_id → cell_id`; replaces any mapping in the queue. |
| `--cell_id_csv_instance_col` | `instance_id` | Column names, if yours differ. |
| `--cell_id_csv_cell_col` | `cell_id` | |

`--load_cellmap_csv` is a deprecated alias of `--cell_id_csv`.

### Display

| Flag | Default | Description |
|---|---|---|
| `--raw_clim_pcts LO HI` | `1 99` | Contrast percentiles. |
| `--raw_gamma` | `1.0` | Display gamma. |

---

## Module map

| File | Responsibility | Depends on |
|---|---|---|
| `utils.py` | conventions, sentinels, queue schema, axis and unit helpers | — |
| `heuristics.py` | `HeuristicParams`, per-instance features, scoring | `utils` |
| `segmentation.py` | watershed, pre-split, the two instance pipelines | `utils`, `heuristics` |
| `io_utils.py` | readers, MRC writer, CSV loader | `utils` |
| `smoothing.py` | ChimeraX-style Laplacian mesh smoothing + its widget | — |
| `napari_session.py` | the napari UI and the CLI entry point | all of the above |

`utils.py` is the dependency root — it imports nothing from the project, so anything may import from it. Dependencies flow in one direction only, and only `napari_session.py` and `smoothing.py` touch napari.

---

## Project conventions

These hold across every module. Breaking one is a bug.

**Axis order.** Arrays are always `(z, y, x)`. Per-axis tuples carry their order in the name: `spacing_nm_xyz` is `(x, y, z)`, `scale_zyx` is `(z, y, x)`. Readers deliver `(z, y, x)`; nothing downstream transposes.

**Units.** Every physical quantity names its unit: `_nm`, `_angstrom`, `_px`, `_vox`. File headers are read in Ångström and reported as `voxel_size_angstrom_xyz`; everything else in the project works in nanometres. The MRC writer takes `voxel_size_nm_xyz` and converts on the way out, so Ångström never leaks past the I/O boundary.

**Sentinels.**

| Constant | Value | Meaning |
|---|---|---|
| `LABEL_BACKGROUND` | `0` | background in any label volume |
| `CELL_ID_UNASSIGNED` | `0` | no parent cell yet |
| `FEATURE_NOT_COMPUTED` | `-1` | feature deliberately skipped |

**Vocabulary.** *semantic mask* = binary volume · *cell labels* = one id per cell · *instance* = one object · *part* = a piece of a component after a split · *review queue* = one row per instance.

**Reader contract.** Every `read_*` returns `(data_zyx, voxel_size_angstrom_xyz, dtype)`. `(nan, nan, nan)` means the format carries no spacing — fall back to `--spacing_nm`.

**Feature keys match queue columns.** `compute_basic_features()` returns a dict keyed by the same names as the CSV columns, so rows are built by lookup, never by renaming.

**Log tags.** Every line is prefixed: `[run]` config echo · `[step]` stage start · `[io]` file read/write · `[mode]` branch taken · `[info]` counts and stats · `[progress]` long loops · `[timing]` elapsed · `[done]` stage finished · `[warn]` recoverable problem · `[ui]` `[map]` `[remap]` `[save]` `[view]` `[smooth]` UI-side. `grep '^\[warn\]'` to audit a long run.

**CLI names match parameter names.** `--min_sep_nm` ↔ `HeuristicParams.min_sep_nm`, with the same default on both sides.

---

## Troubleshooting

**napari won't import.** You are probably on Python 3.13+. Use 3.10–3.12. Everything except `--napari` works regardless.

**Everything is one giant label.** Your objects touch. Enable `--pre_split_xy_erosion_iters 1` for very large clusters, or `--watershed_first` for multi-lobed ones. Then fix the rest with **Split selected label using seeds** in the UI.

**My edits vanished when I closed the window.** The automatic write-on-close can fail if its destination has gone away — an unmounted share, a deleted folder. Use **Save MRC** in the Save tab during the session: it writes immediately, to a folder you choose, and reports success or failure in the terminal while you can still do something about it. If you have not used it, closing the window asks you to confirm first.

**Too many instances / over-split.** Raise `--ws_strong_peak_frac` toward `0.85`, raise `--min_sep_nm`, or turn splitting off entirely and merge by hand with **Merge labels at points** — merging is faster than splitting.

**Nothing is flagged.** With few components the MAD-based outlier test has little to work with, and `--watershed_first` resolves the suspicious cases before they can be scored. That is expected. You can still browse every instance manually.

**Wrong aspect ratio in 3D.** `--spacing_nm` is wrong. It is `x y z`, and for anisotropic FIB-SEM the Z value is usually the largest.

**Cell ids are all zero after `--semantic`.** Expected — the global pipeline has no cell information. Supply `--cell` with `--remap_from_cell`, or assign in the UI with the buttons in the **Cell** tab.

**The UI is sluggish.** Untick `raw_vem` in napari's layer list to hide it. Volumes over 200 M voxels start with it hidden already.

**Long compute lost.** It isn't — `<out>_AUTO.mrc` is written before napari opens. Reload it with `--load_instances`.

---

## License

BSD 3-Clause — see [LICENSE](LICENSE). The whole scientific Python stack this
builds on (napari, NumPy, SciPy, scikit-image, pandas, mrcfile) is BSD-3 too,
so the terms are the ones people in this field already expect.

Copyright is held by The Board of Trustees of the Leland Stanford Junior
University, the standard form for Stanford-owned work.

---

## Citation

If this tool contributes to published work, please cite the accompanying manuscript and link back to this repository.
