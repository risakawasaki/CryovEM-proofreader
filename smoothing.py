"""
smoothing.py
------------
ChimeraX-style Laplacian mesh smoothing for napari Labels layers.

The algorithm matches ChimeraX's ``smooth`` option exactly:
  each vertex is moved ``smoothing_factor`` of the way toward the average
  position of the vertices connected to it by triangle edges; this is
  repeated ``smoothing_iterations`` times.

Key exports
~~~~~~~~~~~
laplacian_smooth(vertices, faces, iterations, factor)
    The bare algorithm, independent of napari.
make_smooth_widget(viewer, lab, scale_zyx)
    Factory function.  Call once after the napari viewer is set up; returns
    a magicgui widget ready to be added to a tab or dock widget.

Coordinate convention
~~~~~~~~~~~~~~~~~~~~~
``scale_zyx`` is the ``(z, y, x)`` scale of the Labels layer, matching the
project axis-order convention.  The napari session switches this scale when
toggling 2D/3D, so ``make_smooth_widget`` also accepts a zero-argument
CALLABLE returning the current scale; the surface is then always built in the
same world coordinates as the layer it came from.
"""

from typing import Any, Callable, Sequence, Tuple, Union

import numpy as np

ScaleLike = Union[Sequence[float], Callable[[], Sequence[float]]]


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int,
    factor: float,
) -> np.ndarray:
    """Apply Laplacian smoothing to a triangle mesh.

    For each iteration every vertex moves ``factor`` of the way toward the
    average position of its edge-connected neighbours (the ChimeraX algorithm).

    Parameters
    ----------
    vertices : (N, 3) float array
        Vertex positions (any physical units).
    faces : (M, 3) int array
        Triangle indices into ``vertices``.
    iterations : int
        Number of smoothing passes (ChimeraX ``smoothingIterations``).
    factor : float
        Step size in [0, 1] (ChimeraX ``smoothingFactor``).

    Returns
    -------
    (N, 3) float64 array of smoothed vertex positions.
    """
    import scipy.sparse as sp

    n = len(vertices)
    if n == 0 or len(faces) == 0:
        return vertices.copy()

    # Build a symmetric adjacency matrix from the three directed edges of
    # every triangle (i→j, j→k, k→i plus their reverses).
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    adj = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n),
    )
    deg = np.array(adj.sum(axis=1), dtype=np.float64).ravel()
    deg[deg == 0] = 1.0  # isolated vertices stay put

    verts = vertices.astype(np.float64)
    for _ in range(int(iterations)):
        avg = adj.dot(verts) / deg[:, np.newaxis]
        verts += factor * (avg - verts)
    return verts


# ---------------------------------------------------------------------------
# napari widget factory
# ---------------------------------------------------------------------------

def make_smooth_widget(
    viewer: Any,
    lab: Any,
    scale_zyx: ScaleLike,
) -> Any:
    """Create and return a magicgui Laplacian-smoothing widget.

    Call this inside ``launch_napari_review`` after ``viewer`` and ``lab``
    have been created::

        from smoothing import make_smooth_widget
        smooth_labels_widget = make_smooth_widget(viewer, lab, scale_zyx)
        # then add to a tab or dock widget as usual

    Parameters
    ----------
    viewer    : napari.Viewer
    lab       : napari Labels layer  (the ``instances`` layer)
    scale_zyx : ``(sz, sy, sx)`` layer scale, OR a zero-argument callable
                returning it.  Pass the callable when the session changes
                scale between 2D and 3D so surfaces stay registered with the
                labels.
    """
    from magicgui import magicgui

    def _current_scale() -> Tuple[float, float, float]:
        s = scale_zyx() if callable(scale_zyx) else scale_zyx
        return (float(s[0]), float(s[1]), float(s[2]))

    @magicgui(
        call_button="Apply Laplacian Smoothing",
        smoothing_factor={
            "widget_type": "FloatSlider",
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "label": "Factor (0–1)",
        },
        smoothing_iterations={
            "widget_type": "SpinBox",
            "min": 1,
            "max": 100,
            "step": 1,
            "label": "Iterations",
        },
    )
    def smooth_labels_widget(
        smoothing_factor: float = 1.0,
        smoothing_iterations: int = 10,
    ) -> None:
        """ChimeraX-style Laplacian surface smoothing for the selected label.

        A mesh is extracted with marching cubes for the currently selected
        label, then every vertex is moved ``smoothing_factor`` of the way
        toward the average of its edge-connected neighbours, repeated
        ``smoothing_iterations`` times.

        The result is added as a napari Surface layer so the original label
        voxels are never changed — editing tools remain fully functional.
        """
        from skimage.measure import marching_cubes

        iid = int(lab.selected_label)
        if iid == 0:
            print("[smooth] Select a nonzero label first.")
            return

        binary = (np.asarray(lab.data) == iid).astype(np.float32)
        if binary.sum() < 4:
            print(f"[smooth] Label {iid} has fewer than 4 voxels; cannot build mesh.")
            return

        print(
            f"[smooth] Laplacian smoothing — label {iid}, "
            f"factor={smoothing_factor:.1f}, iterations={smoothing_iterations}"
        )

        try:
            verts, faces, _, _ = marching_cubes(binary, level=0.5)
        except Exception as exc:
            print(f"[smooth] marching_cubes failed for label {iid}: {exc}")
            return

        # Scale into the SAME world coordinates as the Labels layer, resolved
        # at click time so a 2D→3D toggle cannot leave the surface misaligned.
        verts = verts * np.array(_current_scale())

        # Apply the ChimeraX Laplacian algorithm
        verts = laplacian_smooth(
            verts, faces,
            iterations=smoothing_iterations,
            factor=float(smoothing_factor),
        )

        values = np.full(len(verts), float(iid), dtype=np.float32)
        layer_name = f"smooth_{iid}_f{smoothing_factor:.1f}_i{smoothing_iterations}"
        surface_layer = viewer.add_surface(
            (verts, faces, values),
            name=layer_name,
        )

        # Match color to the original Labels layer
        try:
            rgba = np.zeros((len(verts), 4), dtype=np.float32)
            rgba[:] = lab.get_color(iid)
            surface_layer.vertex_colors = rgba
        except Exception:
            try:
                surface_layer.colormap = lab.colormap
            except Exception as e:
                print(f"[smooth] Could not copy colors ({e}); default colors applied.")

        print(f"[smooth] Surface layer '{layer_name}' added for label {iid}.")

    return smooth_labels_widget
