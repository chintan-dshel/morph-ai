"""
meshing.py — FEM mesh generation for ICEM / Fluent / Nastran / Abaqus handoff.

Workflow
--------
  1. Laplacian smooth the marching-cubes STL surface (trimesh)
  2. Feed into Gmsh → classify surfaces → create volume → tet mesh
  3. Stress-guided sizing: refine elements near high strain-energy zones
  4. Tag FIXED / LOAD / FREE surface groups + SOLID volume group
  5. Export .msh v2.2 (Fluent/ICEM), .bdf (Nastran), .inp (Abaqus)

The resulting mesh is a valid starting point for ICEM CFD hex-blocking,
HyperMesh refinement, or direct Abaqus/CalculiX analysis.
"""

import io
import math
import os
import tempfile

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Face-plane lookup  (axis index, coordinate value — None = full dimension)
# ─────────────────────────────────────────────────────────────────────────────
def _face_plane(face_label: str, box_w: float, box_h: float, box_d: float):
    """Return (axis_index, coordinate_value) for a named face label."""
    return {
        "Left (X=0)":   (0, 0.0),
        "Right (X=W)":  (0, box_w),
        "Bottom (Y=0)": (1, 0.0),
        "Top (Y=H)":    (1, box_h),
        "Front (Z=0)":  (2, 0.0),
        "Back (Z=D)":   (2, box_d),
    }.get(face_label, (0, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
#  Surface smoothing
# ─────────────────────────────────────────────────────────────────────────────
def _heal_stl(stl_bytes: bytes) -> bytes:
    """Remove degenerate/duplicate triangles and merge close vertices.

    This is always run before Gmsh to prevent "overlapping facets" errors.
    Marching-cubes output commonly has duplicate faces at thin-wall regions.
    """
    try:
        import trimesh
        import trimesh.repair

        mesh = trimesh.load(
            io.BytesIO(stl_bytes), file_type="stl", force="mesh"
        )
        if not isinstance(mesh, trimesh.Trimesh):
            return stl_bytes

        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.merge_vertices()
        trimesh.repair.fix_normals(mesh)

        buf = io.BytesIO()
        mesh.export(buf, file_type="stl")
        return buf.getvalue()
    except Exception:
        return stl_bytes


def smooth_surface(stl_bytes: bytes, iterations: int = 3) -> bytes:
    """Apply Laplacian smoothing to reduce marching-cubes stairstepping.

    Returns smoothed STL bytes, or the original bytes if trimesh is unavailable
    or if the mesh cannot be loaded as a single Trimesh object.
    """
    try:
        import trimesh
        import trimesh.smoothing
        import trimesh.repair

        mesh = trimesh.load(
            io.BytesIO(stl_bytes), file_type="stl", force="mesh"
        )
        if not isinstance(mesh, trimesh.Trimesh):
            return stl_bytes

        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.merge_vertices()
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fill_holes(mesh)
        trimesh.smoothing.filter_laplacian(mesh, iterations=iterations)
        mesh.remove_degenerate_faces()

        buf = io.BytesIO()
        mesh.export(buf, file_type="stl")
        return buf.getvalue()

    except Exception:
        return stl_bytes


# ─────────────────────────────────────────────────────────────────────────────
#  Main mesh generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_fem_mesh(
    stl_bytes: bytes,
    box_w: float,
    box_h: float,
    box_d: float,
    fixed_face: str,
    load_face: str,
    stress_field=None,
    global_size: float = 5.0,
    min_size: float = 1.0,
    max_size: float = 15.0,
    smooth: bool = True,
    refine_stress: bool = True,
) -> dict:
    """Generate a quality tet FEM mesh from an optimized topology STL.

    Parameters
    ----------
    stl_bytes    : Binary STL of the optimized surface (from to_stl_bytes).
    box_w/h/d    : Bounding box dimensions in mm — used for face tagging.
    fixed_face   : Label string, e.g. "Left (X=0)".
    load_face    : Label string, e.g. "Right (X=W)".
    stress_field : (nely, nelx) float array from simp_core — optional.
                   Used to place finer elements at high-strain zones.
    global_size  : Nominal element edge length (mm).
    min_size     : Minimum element edge length (mm).
    max_size     : Maximum element edge length (mm).
    smooth       : Apply Laplacian smoothing before meshing.
    refine_stress: Place refinement attractors at high-strain voxels.

    Returns
    -------
    dict with keys:
        msh_bytes   bytes   Gmsh .msh v2.2  (Fluent / ICEM / OpenFOAM)
        bdf_bytes   bytes   Nastran BDF     (HyperMesh / Patran)
        inp_bytes   bytes   Abaqus INP      (Abaqus / CalculiX)
        n_nodes     int
        n_elements  int     4-node tet elements
        fixed_tris  int     surface triangles tagged FIXED
        load_tris   int     surface triangles tagged LOAD
        free_tris   int     surface triangles tagged FREE

    Raises
    ------
    ImportError   if gmsh is not installed.
    ValueError    if mesh generation fails (non-watertight surface, etc.).
    """
    try:
        import gmsh
    except ImportError:
        raise ImportError(
            "gmsh is not installed. Run:  pip install gmsh\n"
            "gmsh is ~50 MB but provides ICEM/Fluent-compatible tet meshing."
        )

    # ── Always heal the STL first (removes duplicate/degenerate triangles) ──
    # Marching-cubes output at thin-wall regions often has exact duplicate
    # faces that cause Gmsh "overlapping facets" errors during surface meshing.
    stl_bytes = _heal_stl(stl_bytes)

    # ── Optional Laplacian smoothing ───────────────────────────
    if smooth:
        stl_bytes = smooth_surface(stl_bytes, iterations=3)

    # ── Temp files (Gmsh needs file paths, not in-memory streams) ─
    tmp = tempfile.mkdtemp()
    paths = {k: os.path.join(tmp, f"mesh.{k}") for k in ("stl", "msh", "bdf", "inp")}
    with open(paths["stl"], "wb") as fh:
        fh.write(stl_bytes)

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)   # suppress console spam
        gmsh.model.add("topology_opt")

        # ── Load STL and classify into CAD surfaces ────────────
        gmsh.merge(paths["stl"])

        # Remove any duplicate nodes that survived STL export/import
        gmsh.model.mesh.removeDuplicateNodes()

        # classifySurfaces detects sharp edges (40° dihedral threshold)
        # and groups triangles into distinct parametric surfaces
        angle = math.pi * 40 / 180
        gmsh.model.mesh.classifySurfaces(angle, True, True, math.pi)
        gmsh.model.mesh.createGeometry()

        surfs = gmsh.model.getEntities(2)
        if not surfs:
            raise ValueError(
                "No surfaces found after classification. "
                "The STL may not be watertight — try lowering the iso-level or "
                "running geometry repair in trimesh."
            )

        # ── Deduplicate surfaces with identical bounding boxes ─
        # classifySurfaces occasionally creates two overlapping patches at the
        # same location (thin walls, interior holes).  Keeping duplicates causes
        # "Invalid boundary mesh" and aborts tet meshing.
        seen_bbs: dict = {}
        unique_surfs = []
        for dim, tag in surfs:
            bb  = gmsh.model.getBoundingBox(dim, tag)
            key = tuple(round(v, 1) for v in bb)
            if key not in seen_bbs:
                seen_bbs[key] = tag
                unique_surfs.append((dim, tag))
        surfs = unique_surfs

        # ── Create closed volume ───────────────────────────────
        surf_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfs])
        vol       = gmsh.model.geo.addVolume([surf_loop])
        gmsh.model.geo.synchronize()

        # ── Mesh sizing ────────────────────────────────────────
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
        # Automatically refine at geometric curvature (corners & fillets)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumCurvePoints", 5)
        # Delaunay (Algorithm3D=1) is more tolerant of imperfect surface meshes
        # than the default Frontal-Delaunay, reducing "No tetrahedra" failures.
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)

        # Optional: stress-guided refinement field
        if refine_stress and stress_field is not None:
            _apply_stress_field(
                stress_field, box_w, box_h, box_d,
                min_size, max_size, global_size,
            )

        # ── Generate 3-D tet mesh ──────────────────────────────
        gmsh.model.mesh.generate(3)

        # Quality optimisation (Netgen smoother — may not be available in all
        # gmsh builds; failure here is non-fatal)
        try:
            gmsh.model.mesh.optimize("Netgen")
        except Exception:
            pass

        # ── Mesh statistics ────────────────────────────────────
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)

        def _count_tets():
            # getElementsByType returns (elementTags, nodeTags) — 2 values
            try:
                tags, _ = gmsh.model.mesh.getElementsByType(4)
                return len(tags)
            except Exception:
                pass
            # Fallback: query the volume entity directly
            try:
                _, etags, _ = gmsh.model.mesh.getElements(3, vol)
                return sum(len(t) for t in etags) if etags else 0
            except Exception:
                return 0

        n_elements = _count_tets()

        # If Delaunay still produced 0 tets (overlapping surface facets), try
        # once more with a coarser mesh which merges thin/degenerate triangles
        if n_elements == 0:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin",  min_size * 2)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax",  max_size * 2)
            gmsh.model.mesh.generate(3)
            n_elements = _count_tets()

        if n_elements == 0:
            raise ValueError(
                "Volume meshing produced 0 tet elements. The topology surface has "
                "overlapping facets (thin walls). Try increasing SIMP Penalty to "
                "4.0+ and Max Iterations to 80+ in Advanced Settings for a sharper, "
                "cleaner geometry before re-running mesh generation."
            )

        # ── Tag BC surface groups (element-level, not entity-level) ──
        # classifySurfaces merges end-cap faces with larger curved surfaces,
        # making entity-centroid detection unreliable.  Instead, fetch every
        # surface triangle's centroid from node coordinates and classify directly.
        fixed_tris, load_tris, free_tris = _tag_bc_elements(
            surfs, vol,
            fixed_face, load_face,
            box_w, box_h, box_d,
        )

        # ── Export ─────────────────────────────────────────────
        # v2.2 for maximum tool compatibility
        # (ICEM, Fluent, OpenFOAM all read .msh v2 natively)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(paths["msh"])
        gmsh.write(paths["bdf"])
        gmsh.write(paths["inp"])

        result = {
            "n_nodes":    n_nodes,
            "n_elements": n_elements,
            "fixed_tris": fixed_tris,
            "load_tris":  load_tris,
            "free_tris":  free_tris,
        }
        for fmt in ("msh", "bdf", "inp"):
            try:
                with open(paths[fmt], "rb") as fh:
                    result[f"{fmt}_bytes"] = fh.read()
            except Exception:
                result[f"{fmt}_bytes"] = b""

        return result

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Mesh generation failed: {exc}") from exc

    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
        for p in paths.values():
            try:
                os.remove(p)
            except Exception:
                pass
        try:
            os.rmdir(tmp)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _tag_bc_elements(surfs, vol_tag, fixed_face, load_face, box_w, box_h, box_d):
    """Tag FIXED / LOAD / FREE boundary groups at the *element* level.

    classifySurfaces can merge end-cap faces with larger curved surfaces, making
    entity-centroid detection unreliable.  This function:
      1. Fetches every surface triangle's centroid from actual node coordinates.
      2. Classifies each triangle as FIXED, LOAD, or FREE by proximity to the
         bounding-box face planes.
      3. Creates discrete Gmsh surface entities holding each group's triangles.
      4. Registers them as named physical groups.

    Returns (n_fixed_tris, n_load_tris, n_free_tris).
    """
    import gmsh

    # ── Build node-coord lookup ───────────────────────────────
    node_tags_all, coords_flat, _ = gmsh.model.mesh.getNodes(-1, -1)
    nc = coords_flat.reshape(-1, 3)
    node_xyz: dict = {int(t): nc[i] for i, t in enumerate(node_tags_all)}

    # ── Face plane params ─────────────────────────────────────
    dims = [box_w, box_h, box_d]
    fai, fav = _face_plane(fixed_face, box_w, box_h, box_d)
    lai, lav = _face_plane(load_face,  box_w, box_h, box_d)
    tol_f = dims[fai] * 0.15   # 15% — generous to catch slightly-off caps
    tol_l = dims[lai] * 0.15

    # ── Collect all surface triangles ─────────────────────────
    all_surf_entities = [tag for _, tag in surfs]
    fixed_etags, fixed_enodes = [], []
    load_etags,  load_enodes  = [], []
    free_etags,  free_enodes  = [], []

    for stag in all_surf_entities:
        try:
            etypes, etag_lists, enode_lists = gmsh.model.mesh.getElements(2, stag)
        except Exception:
            continue
        if not etypes:
            continue
        tri_tags  = etag_lists[0]
        tri_nodes = enode_lists[0].reshape(-1, 3)

        for etag, nodes in zip(tri_tags, tri_nodes):
            coords = np.array([node_xyz[int(n)] for n in nodes])
            cx = coords.mean(axis=0)
            if abs(cx[fai] - fav) < tol_f:
                fixed_etags.append(etag); fixed_enodes.extend(nodes)
            elif abs(cx[lai] - lav) < tol_l:
                load_etags.append(etag);  load_enodes.extend(nodes)
            else:
                free_etags.append(etag);  free_enodes.extend(nodes)

    # ── Create discrete surfaces and physical groups ──────────
    def _add_tri_group(etags, enodes, name):
        if not etags:
            return
        ds = gmsh.model.addDiscreteEntity(2)
        # addElements(dim, tag, types, elementTags, nodeTags)
        # Each arg is a list-of-lists — one sub-list per element type.
        gmsh.model.mesh.addElements(
            2, ds,
            [2],                        # element type 2 = 3-node triangle
            [list(int(x) for x in etags)],
            [list(int(x) for x in enodes)],
        )
        gmsh.model.addPhysicalGroup(2, [ds], name=name)

    _add_tri_group(fixed_etags, fixed_enodes, "FIXED")
    _add_tri_group(load_etags,  load_enodes,  "LOAD")
    _add_tri_group(free_etags,  free_enodes,  "FREE")
    gmsh.model.addPhysicalGroup(3, [vol_tag], name="SOLID")

    return len(fixed_etags), len(load_etags), len(free_etags)


def _apply_stress_field(stress_field, box_w, box_h, box_d,
                        min_size, max_size, global_size):
    """Add a Gmsh Distance+Threshold background field from high-strain voxels.

    Elements near the top-25% strain-energy voxels get smaller than global_size;
    elements far from them get up to max_size.
    """
    import gmsh

    nely, nelx = stress_field.shape
    threshold = float(np.percentile(stress_field, 75))

    pts = []
    for iy in range(nely):
        for ix in range(nelx):
            if stress_field[iy, ix] > threshold:
                px = (ix + 0.5) * box_w / nelx
                py = (iy + 0.5) * box_h / nely
                pz = box_d / 2            # mid-plane (2-D stress → 3-D midplane)
                pt = gmsh.model.geo.addPoint(px, py, pz, 0)
                pts.append(pt)

    if not pts:
        return

    gmsh.model.geo.synchronize()

    f = gmsh.model.mesh.field

    dist_id = f.add("Distance")
    f.setNumbers(dist_id, "PointsList", pts)

    thr_id = f.add("Threshold")
    f.setNumber(thr_id, "InField",  dist_id)
    f.setNumber(thr_id, "SizeMin",  min_size)
    f.setNumber(thr_id, "SizeMax",  max_size)
    f.setNumber(thr_id, "DistMin",  global_size * 0.5)
    f.setNumber(thr_id, "DistMax",  global_size * 3.0)
    f.setAsBackgroundMesh(thr_id)
