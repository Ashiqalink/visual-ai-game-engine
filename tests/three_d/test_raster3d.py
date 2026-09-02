"""
The compiled rasteriser against the Python one it replaces.

Two levels, because a pixel comparison alone cannot say *why* two frames
differ. `test_face_parity` compares the cull/shade/sort stage face by face,
where the two paths are meant to agree exactly - same faces, same screen
corners, same shaded colour, same draw order. `test_pixel_parity` then compares
the frames, where they are only meant to agree closely: the fill rules are
different (cv2's polygon fill against the scanline fill in raster3d.cpp), so
the disagreement is a fringe of boundary pixels and the test bounds it rather
than forbidding it.

The guard tests are the ones that can fail loudly if the routing breaks: a
renderer that quietly drew nothing, or drew into a copy of the caller's frame,
would pass a parity test that only compared two blank images.
"""

import numpy as np
import pytest

from visual_ai.material import Material
from visual_ai.three_d.camera import Camera3D
from visual_ai.three_d.mesh import Mesh3D
from visual_ai.three_d.renderer import CPP_RASTERISER_AVAILABLE, Renderer3D
from visual_ai.three_d.transform import Transform3D
from visual_ai.three_d import voxel

pytestmark = pytest.mark.skipif(not CPP_RASTERISER_AVAILABLE,
                                reason="engine_core built without the rasteriser")

WIDTH, HEIGHT = 320, 240


def _scene():
    """A scene with both mesh kinds, two materials and overlapping depth."""
    grid = voxel.ellipsoid((15, 15, 15), (7.0, 7.0, 7.0), (7.0, 5.0, 7.0))
    lump = voxel.voxel_mesh(grid, 14.0)
    return [
        (Mesh3D.create_cube(size=150.0),
         Transform3D(x=-95.0, y=10.0, z=-30.0, rx=25.0, ry=40.0, rz=10.0),
         Material(base_color=(0.9, 0.4, 0.2, 1.0))),
        (lump,
         Transform3D(x=35.0, y=-15.0, z=20.0, rx=-15.0, ry=70.0, rz=5.0,
                     sx=1.2, sy=0.8, sz=1.1),
         Material(base_color=(0.3, 0.7, 1.0, 1.0))),
        (Mesh3D.create_icosahedron(radius=75.0),
         Transform3D(x=0.0, y=95.0, z=60.0, ry=15.0),
         Material(base_color=(0.5, 0.9, 0.5, 1.0))),
    ]


def _renderer():
    return Renderer3D(camera=Camera3D(screen_width=WIDTH, screen_height=HEIGHT),
                      light_angle_deg=35.0)


def _python_faces(renderer, items, wireframe=False):
    """The Python path's faces as one dict keyed on (item, face number)."""
    from visual_ai.three_d.renderer import _FaceBatch

    batch = _FaceBatch.concat([
        renderer._collect_faces(mesh, transform, material, wireframe, item=index)
        for index, (mesh, transform, material) in enumerate(items)])
    order = np.lexsort((batch.face_no, batch.item, -batch.depth))
    faces = {}
    for i in range(len(batch)):
        faces[(int(batch.item[i]), int(batch.face_no[i]))] = (
            float(batch.depth[i]), np.asarray(batch.pts[i], dtype=np.int64),
            tuple(int(c) for c in batch.colors[i]), float(batch.opacity[i]),
            bool(batch.wireframe[i]))
    sequence = [(int(batch.item[i]), int(batch.face_no[i])) for i in order.tolist()]
    return faces, sequence


def _cpp_faces(renderer, items, wireframe=False):
    """The same, from collect_faces3d."""
    import engine_core

    out = engine_core.collect_faces3d(*renderer._scene_arrays(items, wireframe))
    offsets = out["pts_off"]
    faces = {}
    for i in range(len(out["depth"])):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        faces[(int(out["item"][i]), int(out["face_no"][i]))] = (
            float(out["depth"][i]),
            out["pts"][2 * lo:2 * hi].reshape(-1, 2).astype(np.int64),
            tuple(int(c) for c in out["color"][3 * i:3 * i + 3]),
            float(out["opacity"][i]), bool(out["wireframe"][i]))
    sequence = [(int(out["item"][i]), int(out["face_no"][i]))
                for i in out["order"].tolist()]
    return faces, sequence


@pytest.mark.parametrize("wireframe", [False, True])
def test_face_parity(wireframe):
    """Same survivors, same corners, same colour, same order."""
    renderer, items = _renderer(), _scene()
    py_faces, py_order = _python_faces(renderer, items, wireframe)
    cpp_faces, cpp_order = _cpp_faces(renderer, items, wireframe)

    assert len(py_faces) > 200, "scene culled to nothing; the test proves nothing"
    assert set(py_faces) == set(cpp_faces)

    for key, (depth, pts, color, opacity, wf) in py_faces.items():
        c_depth, c_pts, c_color, c_opacity, c_wf = cpp_faces[key]
        assert c_depth == pytest.approx(depth, rel=1e-12, abs=1e-9), key
        assert np.array_equal(c_pts, pts), key
        assert c_color == color, key
        assert c_opacity == pytest.approx(opacity), key
        assert c_wf == wf, key

    assert cpp_order == py_order


def test_face_parity_can_fail():
    """
    The comparison above is only a check if a real difference breaks it.

    Shading is the part with the most arithmetic in it, so move the light: the
    faces and their corners must stay identical and the colours must not.
    """
    renderer, items = _renderer(), _scene()
    py_faces, _ = _python_faces(renderer, items)
    renderer.set_light_angle(125.0)
    cpp_faces, _ = _cpp_faces(renderer, items)

    assert set(py_faces) == set(cpp_faces)
    differing = sum(1 for key in py_faces if py_faces[key][2] != cpp_faces[key][2])
    assert differing > len(py_faces) // 10


def _face_edges(frame: np.ndarray) -> np.ndarray:
    """
    Where one face meets another, or the background, widened by a pixel.

    A morphological gradient rather than the face list: this is asked of a
    finished frame, so what counts as an edge is where the painted colour
    actually changes - which includes the seam between two faces of one lump
    as well as the silhouette.
    """
    import cv2

    kernel = np.ones((3, 3), np.uint8)
    grey = frame.max(axis=2)
    gradient = cv2.dilate(grey, kernel).astype(np.int16) - cv2.erode(grey, kernel)
    return cv2.dilate((gradient > 0).astype(np.uint8), kernel) > 0


def _render(use_cpp, items, wireframe=False, camera=None):
    renderer = _renderer()
    renderer.use_cpp = use_cpp
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    renderer.render_scene(frame, items, wireframe=wireframe)
    return frame


def _parity_failures(cpp: np.ndarray, py: np.ndarray) -> list[str]:
    """
    What stops ``cpp`` and ``py`` being the same frame bar a boundary fringe.

    Bounded rather than forbidden: cv2.fillPoly and the scanline fill in
    raster3d.cpp round the edge of a polygon differently, so a face's outline
    can move by a pixel. What must not happen is a face going missing or
    landing somewhere else.

    A misplaced face is a patch that is (a) away from any edge of the Python
    frame, (b) strongly coloured, and (c) connected. All three conditions
    matter: translucency spreads a boundary error inward, because every later
    face blends onto whatever the earlier one left, and that leaves a large
    connected off-edge patch which is nonetheless only a rounding step deep.

    Returned as a list rather than asserted, so `test_pixel_parity_can_fail`
    can put a frame in that must not pass.
    """
    import cv2

    failures = []
    delta = np.abs(cpp.astype(np.int16) - py.astype(np.int16)).max(axis=2)
    if (delta > 0).mean() >= 0.08:
        failures.append(f"{(delta > 0).mean():.3%} of pixels differ")
    if delta.mean() >= 3.0:
        failures.append(f"mean difference {delta.mean():.2f} of 255")

    strong = ((delta > 32) & ~_face_edges(py)).astype(np.uint8)
    if strong.sum() >= 0.001 * strong.size:
        failures.append(f"{strong.sum()} strong pixels differ away from any edge")
    count, _, stats, _ = cv2.connectedComponentsWithStats(strong, 8)
    if count > 1:
        largest = int(stats[1:, cv2.CC_STAT_AREA].max())
        if largest > 32:
            failures.append(f"a {largest}-pixel patch differs away from any edge")
    return failures


@pytest.mark.parametrize("materials", ["opaque", "translucent"])
def test_pixel_parity(materials):
    """The two fills land on the same pixels bar a boundary fringe."""
    items = _scene()
    if materials == "translucent":
        items = [(mesh, transform,
                  Material(base_color=material.base_color, opacity=0.55))
                 for mesh, transform, material in items]

    cpp = _render(True, items)
    py = _render(False, items)

    assert cpp.any() and py.any(), "nothing was drawn"
    covered = (py.max(axis=2) > 0) | (cpp.max(axis=2) > 0)
    assert covered.mean() > 0.15, "the scene barely covers the frame"
    assert _parity_failures(cpp, py) == []


@pytest.mark.parametrize("wrong,expected", [
    (Transform3D(x=-65.0, y=10.0, z=-30.0, rx=25.0, ry=40.0, rz=10.0),
     "an item shifted 30 units"),
    (Transform3D(x=-95.0, y=10.0, z=-30.0, rx=25.0, ry=45.0, rz=10.0),
     "an item turned 5 degrees"),
])
def test_pixel_parity_can_fail(wrong, expected):
    """
    A rasteriser that put a face in the wrong place has to trip the check.

    Without this the bound above is only a claim: a fringe-sized tolerance
    that nothing can exceed would pass two blank frames just as happily. The
    two mutations are the smallest ones tried that a face-level comparison
    would also catch - so if this stops failing, the pixel bound has gone
    slack, not the renderer right.
    """
    items = _scene()
    mesh, _, material = items[0]
    py = _render(False, items)
    cpp = _render(True, [(mesh, wrong, material)] + items[1:])
    assert _parity_failures(cpp, py), f"{expected} went unnoticed"


def test_wireframe_draws_lines_not_fills():
    """Wireframe must be sparse - a filled frame would also 'not be blank'."""
    items = _scene()
    wire = _render(True, items, wireframe=True)
    solid = _render(True, items)
    assert wire.any()
    assert (wire.max(axis=2) > 0).sum() < 0.35 * (solid.max(axis=2) > 0).sum()


def test_frame_is_written_in_place():
    """The caller's buffer is the one painted, not a copy of it."""
    renderer = _renderer()
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    returned = renderer.render_scene(frame, _scene())
    assert returned is frame
    assert frame.any()


def test_non_contiguous_frame_falls_back_and_still_draws():
    """
    A view the extension cannot write into must go down the Python path.

    Painting a non-contiguous frame through a forced copy would lose the frame
    silently, so the routing refuses it. The pixels still have to appear.
    """
    canvas = np.zeros((HEIGHT, WIDTH * 2, 3), dtype=np.uint8)
    view = canvas[:, :WIDTH]
    assert not view.flags["C_CONTIGUOUS"]
    _renderer().render_scene(view, _scene())
    assert view.any()
    assert not canvas[:, WIDTH:].any()


def test_empty_and_degenerate_scenes():
    """No meshes, no faces, and a mesh entirely behind the camera."""
    renderer = _renderer()
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    renderer.render_scene(frame, [])
    assert not frame.any()

    renderer.render_scene(frame, [(Mesh3D(vertices=np.zeros((0, 3)), faces=[]),
                                   Transform3D(), Material())])
    assert not frame.any()

    behind = Transform3D(z=renderer.camera.position[2] + 500.0)
    renderer.render_scene(frame, [(Mesh3D.create_cube(size=150.0), behind,
                                   Material())])
    assert not frame.any()


def test_face_arrays_match_faces():
    """The CSR the binding is handed is the mesh's own face list."""
    mesh = voxel.voxel_mesh(
        voxel.ellipsoid((7, 7, 7), (3.0, 3.0, 3.0), (3.0, 3.0, 3.0)), 9.0)
    indices, offsets = mesh.face_arrays()
    assert offsets[0] == 0 and offsets[-1] == len(indices)
    assert len(offsets) == len(mesh.faces) + 1
    for i, face in enumerate(mesh.faces):
        assert list(indices[offsets[i]:offsets[i + 1]]) == list(face)
