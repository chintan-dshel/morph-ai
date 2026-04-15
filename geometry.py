import io
import struct
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
from materials import INFILL_PATTERNS


def generate_infill(xPhys, box_w, box_h, box_d, pattern_name,
                    period_mm=12.0, void_threshold=0.15, solid_threshold=0.75,
                    fine_nx=60, fine_ny=36, fine_nz=20):
    nely, nelx = xPhys.shape
    y_c = ((np.arange(nely) + 0.5) * box_h / nely)[::-1]
    x_c = (np.arange(nelx) + 0.5) * box_w / nelx
    interp = RegularGridInterpolator(
        (y_c[::-1], x_c), xPhys[::-1, :],
        method='linear', bounds_error=False, fill_value=0.0
    )
    xs = np.linspace(0, box_w, fine_nx)
    ys = np.linspace(0, box_h, fine_ny)
    zs = np.linspace(0, box_d, fine_nz)
    X3, Y3, Z3 = np.meshgrid(xs, ys, zs, indexing='ij')
    dens = interp(np.stack([Y3.ravel(), X3.ravel()], axis=-1)).reshape(fine_nx, fine_ny, fine_nz)

    fn = INFILL_PATTERNS[pattern_name]["fn"]
    tpms = fn(X3, Y3, Z3, period_mm)
    tpms_norm = tpms / (np.abs(tpms).max() + 1e-9)

    sf = np.zeros_like(dens)
    vm = dens < void_threshold
    sm = dens > solid_threshold
    im = (~vm) & (~sm)
    sf[im] = tpms_norm[im] - (1.0 - 2.0 * dens[im])
    sf[sm] = 2.0
    sf[vm] = -2.0

    if 0. <= sf.min() or 0. >= sf.max():
        return None, 0, 0.0, sf

    try:
        v, f, _, _ = measure.marching_cubes(
            sf, level=0.0,
            spacing=(box_w / fine_nx, box_h / fine_ny, box_d / fine_nz)
        )
    except Exception:
        return None, 0, 0.0, sf

    stl_bytes = _triangles_to_stl(v, f)
    return stl_bytes, len(f), float((sf > 0).sum() / sf.size), sf


def to_stl_bytes(vol_or_xphys, box_w, box_h, box_d, iso, is_xphys=True):
    if is_xphys:
        nely, nelx = vol_or_xphys.shape
        n_depth = 8
        # Build 3-D volume: axis-0=nely(Y), axis-1=nelx(X), axis-2=depth
        vol = np.zeros((nely, nelx, n_depth))
        for d in range(n_depth):
            vol[:, :, d] = vol_or_xphys

        # Pad with a zero-shell on all sides so that marching-cubes produces a
        # *closed* watertight surface even where the part touches the bounding box.
        # Without padding, those faces are open edges → Gmsh cannot tet-mesh them.
        vol_pad = np.pad(vol, 1, mode='constant', constant_values=0.0)

        # Spacing along (axis-0=Y, axis-1=X, axis-2=Z)
        dy, dx, dz = box_h / nely, box_w / nelx, box_d / n_depth

        if iso <= vol_pad.min() or iso >= vol_pad.max():
            return None, 0

        try:
            verts, faces, _, _ = measure.marching_cubes(
                vol_pad, level=iso, spacing=(dy, dx, dz)
            )
        except Exception:
            return None, 0

        # Subtract the 1-voxel padding offset so the part starts at (0,0,0)
        verts[:, 0] -= dy
        verts[:, 1] -= dx
        verts[:, 2] -= dz
        np.clip(verts[:, 0], 0, box_h, out=verts[:, 0])
        np.clip(verts[:, 1], 0, box_w, out=verts[:, 1])
        np.clip(verts[:, 2], 0, box_d, out=verts[:, 2])

        # marching_cubes returns (Y, X, Z). Swap to (X, Y, Z).
        # Swapping axes 0↔1 is a reflection (det=-1) → flip winding to preserve
        # outward normals so Gmsh classifySurfaces works correctly.
        verts = verts[:, [1, 0, 2]]
        faces = faces[:, [0, 2, 1]]
    else:
        vol = vol_or_xphys
        spacing = (box_w / vol.shape[0], box_h / vol.shape[1], box_d / vol.shape[2])

        if iso <= vol.min() or iso >= vol.max():
            return None, 0

        try:
            verts, faces, _, _ = measure.marching_cubes(vol, level=iso, spacing=spacing)
        except Exception:
            return None, 0

    return _triangles_to_stl(verts, faces), len(faces)


def _triangles_to_stl(verts, faces):
    buf = io.BytesIO()
    buf.write(b'\x00' * 80)
    buf.write(struct.pack('<I', len(faces)))
    for tri in faces:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        n = np.cross(v1 - v0, v2 - v0)
        nl = np.linalg.norm(n)
        n = (n / nl if nl > 0 else np.zeros(3)).astype(float)
        buf.write(struct.pack('<fff', *n))
        for vi in tri:
            buf.write(struct.pack('<fff', *verts[vi].astype(float)))
        buf.write(struct.pack('<H', 0))
    return buf.getvalue()


def voxelize_mesh(stl_bytes: bytes, nelx: int, nely: int, nelz: int) -> np.ndarray:
    """Voxelize an uploaded STL mesh onto a (nely, nelx, nelz) boolean grid.

    Returns a bool array where True = inside the design space.
    Falls back to all-True (full box) if trimesh is unavailable or mesh is not watertight.
    """
    try:
        import trimesh
        import trimesh.util

        mesh = trimesh.load(
            trimesh.util.wrap_as_stream(stl_bytes),
            file_type='stl',
            force='mesh'
        )
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Uploaded file did not produce a single mesh.")

        pitch = max(mesh.extents) / max(nelx, nely, nelz)
        vox = mesh.voxelized(pitch=pitch).fill()
        raw = vox.matrix  # (nx, ny, nz) bool, may differ from target resolution

        # Resample to target resolution using nearest-neighbour
        from scipy.ndimage import zoom
        scale = (nely / raw.shape[0], nelx / raw.shape[1], nelz / raw.shape[2])
        resampled = zoom(raw.astype(float), scale, order=0) > 0.5
        # Ensure exact shape
        mask = np.zeros((nely, nelx, nelz), dtype=bool)
        sy = min(resampled.shape[0], nely)
        sx = min(resampled.shape[1], nelx)
        sz = min(resampled.shape[2], nelz)
        mask[:sy, :sx, :sz] = resampled[:sy, :sx, :sz]
        return mask

    except Exception:
        # Graceful fallback: treat entire bounding box as design space
        return np.ones((nely, nelx, nelz), dtype=bool)
