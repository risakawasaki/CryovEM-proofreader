"""
io_utils.py
-----------
File I/O and data-preparation helpers.

Reader contract
~~~~~~~~~~~~~~~
Every ``read_*`` function returns the same triple::

    (data_zyx, voxel_size_angstrom_xyz, dtype)

* ``data_zyx``                 — array in ``(z, y, x)`` order, always
* ``voxel_size_angstrom_xyz``  — ``(x, y, z)`` in ÅNGSTRÖM as stored in the
  file header, or ``(nan, nan, nan)`` when the format carries no spacing.
  NaN means "fall back to ``--spacing_nm``".

Writers take NANOMETRE spacing (``voxel_size_nm_xyz``) and convert on the way
out, so nm is the only unit that appears anywhere else in the project.

Key exports
~~~~~~~~~~~
read_volume()                           - dispatch by extension / path type
read_mrc()                              - read an MRC / REC file
read_tiff()                             - read a TIFF stack or directory
read_nifti()                            - read a NIfTI file
write_mrc()                             - write an int label volume as MRC
load_inst_to_cell_override_from_csv()   - load instance→cell mapping from CSV
"""

import os
from typing import Tuple

import mrcfile
import numpy as np
import pandas as pd

from utils import nm_to_angstrom


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_volume(path: str):
    """Dispatch to the appropriate reader based on file extension or path type.

    Supported formats:
    - directory               — all .tif/.tiff files inside, sorted, stacked along Z
    - ``.tif`` / ``.tiff``    — TIFF stack via tifffile
    - ``.nii`` / ``.nii.gz``  — NIfTI via nibabel
    - ``.npy``                — NumPy array
    - anything else           — attempted as MRC via mrcfile (``.mrc``, ``.rec``)

    Returns the standard reader triple
    ``(data_zyx, voxel_size_angstrom_xyz, dtype)`` — see the module docstring.
    Only MRC and NIfTI carry a usable spacing; the others return NaN.
    """
    if os.path.isdir(path):
        return read_tiff(path)
    p = path.lower()
    if p.endswith(".nii.gz") or p.endswith(".nii"):
        return read_nifti(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return read_tiff(path)
    if ext == ".npy":
        data = np.load(path)
        return data, (np.nan, np.nan, np.nan), data.dtype
    return read_mrc(path)


def read_tiff(path: str) -> Tuple[np.ndarray, Tuple[float, float, float], np.dtype]:
    """Read a TIFF stack; returns ``(data_zyx, voxel_size_angstrom_xyz, dtype)``.

    ``path`` may be:
    - A single multi-page TIFF file (pages = Z slices).
    - A directory containing ``.tif``/``.tiff`` files; they are sorted
      lexicographically and stacked along Z in that order.

    Voxel size is always ``(nan, nan, nan)`` — use ``--spacing_nm`` on the CLI.
    """
    try:
        import tifffile
    except ImportError as e:
        raise ImportError(
            "tifffile is required to read TIFF files. Install it with: pip install tifffile"
        ) from e

    if os.path.isdir(path):
        tiff_files = sorted(
            f for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in (".tif", ".tiff")
            and not f.startswith(".")  # exclude macOS AppleDouble ._* resource forks
        )
        if not tiff_files:
            raise ValueError(f"No .tif/.tiff files found in directory: {path}")
        full_paths = [os.path.join(path, f) for f in tiff_files]
        print(f"[io] Reading {len(full_paths)} TIFF slices from directory: {path}")
        data = tifffile.imread(full_paths)
    else:
        data = tifffile.imread(path)

    return data, (np.nan, np.nan, np.nan), data.dtype


def read_mrc(path: str) -> Tuple[np.ndarray, Tuple[float, float, float], np.dtype]:
    """Read an MRC file; returns ``(data_zyx, voxel_size_angstrom_xyz, dtype)``.

    ``voxel_size_angstrom_xyz`` is an ``(x, y, z)`` tuple in Ångström, or
    ``(nan, nan, nan)`` if the header does not contain valid voxel sizes.
    """
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.array(mrc.data)
        try:
            voxel_size = (
                float(mrc.voxel_size.x),
                float(mrc.voxel_size.y),
                float(mrc.voxel_size.z),
            )
        except Exception:
            voxel_size = (np.nan, np.nan, np.nan)
        dtype = data.dtype
    return data, voxel_size, dtype


def read_nifti(path: str) -> Tuple[np.ndarray, Tuple[float, float, float], np.dtype]:
    """Read a NIfTI file; returns ``(data_zyx, voxel_size_angstrom_xyz, dtype)``.

    NIfTI data is loaded in x,y,z order and transposed to z,y,x to match the
    rest of the pipeline. Voxel sizes are read from the header zooms (assumed mm)
    and converted to Ångström (×10,000,000). If zooms are unavailable,
    ``(nan, nan, nan)`` is returned — use ``--spacing_nm`` on the CLI to set spacing.
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError(
            "nibabel is required to read NIfTI files. Install it with: pip install nibabel"
        ) from e

    img = nib.load(path)
    # Use dataobj (not get_fdata) to preserve the stored integer dtype.
    # get_fdata() always upcasts to float64, which napari Labels rejects.
    data_xyz = np.asarray(img.dataobj)
    # NIfTI default axis order is x,y,z; transpose to z,y,x
    data_zyx = np.transpose(data_xyz, (2, 1, 0))
    data_zyx = np.ascontiguousarray(data_zyx)

    try:
        zooms = img.header.get_zooms()[:3]  # (sx, sy, sz) in mm
        voxel_size = (
            float(zooms[0]) * 1e7,
            float(zooms[1]) * 1e7,
            float(zooms[2]) * 1e7,
        )
        if not all(np.isfinite(v) and v > 0 for v in voxel_size):
            voxel_size = (np.nan, np.nan, np.nan)
    except Exception:
        voxel_size = (np.nan, np.nan, np.nan)

    return data_zyx, voxel_size, data_zyx.dtype


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_mrc(
    path: str,
    data_zyx: np.ndarray,
    voxel_size_nm_xyz: Tuple[float, float, float],
) -> None:
    """Write an integer label volume as an MRC file.

    Converts to uint16 / int16 when the label range fits; falls back to
    float32 with a warning for very large label IDs.

    ``voxel_size_nm_xyz`` is ``(x, y, z)`` in NANOMETRES — the project-wide
    unit — and is converted to Ångström for the MRC header on the way out.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = np.ascontiguousarray(data_zyx)
    dmin = int(data.min()) if data.size else 0
    dmax = int(data.max()) if data.size else 0

    if dmin >= 0 and dmax <= 65535:
        out, out_dtype = data.astype(np.uint16, copy=False), "uint16"
    elif dmin >= -32768 and dmax <= 32767:
        out, out_dtype = data.astype(np.int16, copy=False), "int16"
    else:
        out, out_dtype = data.astype(np.float32), "float32"
        print(f"[warn] Label range [{dmin},{dmax}] exceeds 16-bit; writing float32 MRC.")

    print(
        f"[io] write_mrc -> {path}  shape={out.shape}  dtype={out_dtype}  "
        f"label_range=[{dmin},{dmax}]  voxel_size_nm(x,y,z)={voxel_size_nm_xyz}"
    )
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(out)
        sx_nm, sy_nm, sz_nm = voxel_size_nm_xyz
        mrc.voxel_size = (
            nm_to_angstrom(sx_nm),
            nm_to_angstrom(sy_nm),
            nm_to_angstrom(sz_nm),
        )
        mrc.update_header_from_data()
        mrc.update_header_stats()


# ---------------------------------------------------------------------------
# CSV override loader
# ---------------------------------------------------------------------------

def load_inst_to_cell_override_from_csv(
    csv_path: str,
    instance_col: str = "instance_id",
    cell_col: str = "cell_id",
) -> dict:
    """Load an instance → cell_id mapping from a CSV file.

    Column defaults match ``utils.REVIEW_QUEUE_COLUMNS``, so a review-queue CSV
    can be fed straight back in.

    Rows with a missing instance_id or cell_id are skipped; a cell_id of
    ``CELL_ID_UNASSIGNED`` (0) is loaded as-is and means "no parent cell yet".

    Returns a ``{instance_id: cell_id}`` dict.
    """
    df = pd.read_csv(csv_path)

    if instance_col not in df.columns:
        raise ValueError(f"Missing column '{instance_col}'. Found: {list(df.columns)}")
    if cell_col not in df.columns:
        raise ValueError(f"Missing column '{cell_col}'. Found: {list(df.columns)}")

    # Vectorized instead of df.iterrows(): row-by-row Python iteration does not
    # scale to large review-queue CSVs (tens of thousands of instances).
    valid = df.dropna(subset=[instance_col, cell_col])
    iids = valid[instance_col].astype(int).to_numpy()
    cids = valid[cell_col].astype(int).to_numpy()

    # dict(zip(...)) preserves row order, so a duplicate instance_id resolves
    # to its LAST row — same behavior as the previous row-by-row assignment.
    mapping = dict(zip(iids.tolist(), cids.tolist()))

    print(f"[csv_map] loaded {len(mapping)} instance→cell_id mappings")
    return mapping
