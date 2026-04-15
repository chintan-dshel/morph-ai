import numpy as np
import plotly.graph_objects as go
from skimage import measure
from scipy.sparse import csr_matrix


# ─────────────────────────────────────────────────────────────────────────────
#  Mesh post-processing
# ─────────────────────────────────────────────────────────────────────────────

def laplacian_smooth(verts: np.ndarray, faces: np.ndarray,
                     n_iter: int = 5, lam: float = 0.5, mu: float = -0.53
                     ) -> np.ndarray:
    """Taubin Laplacian smoothing — removes staircase artefacts without shrinkage.

    The Taubin scheme alternates a positive-lambda shrink step and a
    negative-mu expansion step.  The two steps approximately cancel volume loss
    while still smoothing high-frequency bumps from marching cubes.

    Parameters
    ----------
    verts   : (V, 3) float array — vertex positions
    faces   : (F, 3) int array  — triangle indices
    n_iter  : number of iteration *pairs* (each pair = 1 shrink + 1 expand)
    lam     : shrink factor  (0 < lam < 1, typically 0.5)
    mu      : expand factor  (mu < −lam, typically −0.53)
    """
    n = len(verts)
    if n == 0 or len(faces) == 0:
        return verts

    # Build symmetric adjacency matrix in one vectorised pass
    i_idx = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                             faces[:, 1], faces[:, 2], faces[:, 0]])
    j_idx = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                             faces[:, 0], faces[:, 1], faces[:, 2]])
    adj = csr_matrix((np.ones(len(i_idx)), (i_idx, j_idx)), shape=(n, n))

    # Row-normalise: each row sums to 1  → multiply = weighted average of neighbours
    deg = np.asarray(adj.sum(axis=1)).flatten()
    deg_inv = 1.0 / np.maximum(deg, 1.0)

    v = verts.astype(np.float64)
    for step in range(n_iter * 2):
        factor = lam if step % 2 == 0 else mu
        neighbour_avg = adj.dot(v) * deg_inv[:, np.newaxis]
        v = v + factor * (neighbour_avg - v)

    return v.astype(np.float32)


def mesh_traces(vol_or_xphys, box_w, box_h, box_d, iso, color, name="Mesh",
                is_xphys=True, stress_field=None, colormode='density',
                smooth=True, smooth_iter=5):
    """Return Plotly traces for the optimized 3D mesh.

    Parameters
    ----------
    stress_field : ndarray (nely, nelx) or None
        Strain energy density per element (from simp_core).
        Only used when colormode='stress'.
    colormode : 'density' | 'stress'
        'density' — flat material color (original behavior).
        'stress'  — rainbow Jet colormap: blue=low, red=high stress.
    """
    if is_xphys:
        if vol_or_xphys.ndim == 2:
            # 2-D SIMP result — extrude uniformly along Z for visualisation
            nely, nelx = vol_or_xphys.shape
            vol = np.broadcast_to(
                vol_or_xphys[:, :, np.newaxis], (nely, nelx, 8)
            ).copy()
            spacing = (box_h / nely, box_w / nelx, box_d / 8)
        else:
            # True 3-D SIMP result — use directly
            nely, nelx, nelz_act = vol_or_xphys.shape
            vol = vol_or_xphys
            spacing = (box_h / nely, box_w / nelx, box_d / nelz_act)
        swap_xy = True
    else:
        vol = vol_or_xphys
        spacing = (box_w / vol.shape[0], box_h / vol.shape[1], box_d / vol.shape[2])
        swap_xy = False

    if iso <= vol.min() or iso >= vol.max():
        return []

    try:
        v, f, _, _ = measure.marching_cubes(vol, level=iso, spacing=spacing)
    except Exception:
        return []

    # Smooth the marching-cubes mesh to remove staircase artefacts
    if smooth and smooth_iter > 0:
        v = laplacian_smooth(v, f, n_iter=smooth_iter)

    if swap_xy:
        x, y = v[:, 1], v[:, 0]
    else:
        x, y = v[:, 0], v[:, 1]

    if colormode == 'stress' and stress_field is not None and is_xphys:
        # Map stress field to vertex intensities via nearest-neighbour voxel lookup
        if stress_field.ndim == 2:
            nely_s, nelx_s = stress_field.shape
            vx = np.clip(np.round(v[:, 1] / box_w * nelx_s).astype(int), 0, nelx_s - 1)
            vy = np.clip(np.round(v[:, 0] / box_h * nely_s).astype(int), 0, nely_s - 1)
            intensity = stress_field[vy, vx]
        else:
            nely_s, nelx_s, nelz_s = stress_field.shape
            vx = np.clip(np.round(v[:, 1] / box_w * nelx_s).astype(int), 0, nelx_s - 1)
            vy = np.clip(np.round(v[:, 0] / box_h * nely_s).astype(int), 0, nely_s - 1)
            vz = np.clip(np.round(v[:, 2] / box_d * nelz_s).astype(int), 0, nelz_s - 1)
            intensity = stress_field[vy, vx, vz]
        s_min, s_max = intensity.min(), intensity.max()
        intensity = (intensity - s_min) / (s_max - s_min + 1e-12)

        return [go.Mesh3d(
            x=x, y=y, z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            intensity=intensity,
            colorscale='Jet',
            showscale=True,
            colorbar=dict(
                title=dict(text="Von Mises (MPa)", font=dict(color="white", size=11)),
                tickfont=dict(color="white"),
                x=1.02, thickness=12,
            ),
            opacity=0.92,
            flatshading=False,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.4, roughness=0.3, fresnel=0.3),
            lightposition=dict(x=box_w * 2, y=box_h * 2, z=box_d * 3),
            name=name,
            showlegend=True,
            hovertemplate="x=%{x:.1f} y=%{y:.1f} z=%{z:.1f}<extra></extra>",
        )]

    # Default: flat material color
    return [go.Mesh3d(
        x=x, y=y, z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color,
        opacity=0.88,
        flatshading=False,
        lighting=dict(ambient=0.4, diffuse=0.7, specular=0.3, roughness=0.4, fresnel=0.2),
        lightposition=dict(x=box_w * 2, y=box_h * 2, z=box_d * 3),
        name=name,
        showlegend=True,
        hovertemplate="x=%{x:.1f} y=%{y:.1f} z=%{z:.1f}<extra></extra>",
    )]


def density_isosurface_traces(xPhys, box_w, box_h, box_d, n_depth=8):
    """OptiStruct-style density field rendering.

    Uses Plotly Isosurface to show multiple nested density shells coloured
    blue (void) → yellow (transition) → red (solid), exactly like HyperView /
    OptiStruct results.  Returns a list with one Isosurface trace.
    """
    nely, nelx = xPhys.shape

    # Element-centre coordinates on a regular grid
    xs = (np.arange(nelx) + 0.5) * box_w / nelx
    ys = (np.arange(nely) + 0.5) * box_h / nely
    zs = (np.arange(n_depth) + 0.5) * box_d / n_depth

    # Build (nelx × nely × n_depth) arrays; xPhys[iy, ix] → V[ix, iy, iz]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    V = np.broadcast_to(
        xPhys.T[:, :, np.newaxis], (nelx, nely, n_depth)
    ).copy()

    return [go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=V.flatten(),
        isomin=0.20,
        isomax=0.95,
        surface_count=5,
        colorscale=[
            [0.00, '#1a3a8a'],
            [0.25, '#3a6fcc'],
            [0.50, '#f0c020'],
            [0.75, '#e06010'],
            [1.00, '#bb1111'],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Density", font=dict(color="white", size=11)),
            tickfont=dict(color="white", size=10),
            tickvals=[0.20, 0.50, 0.80, 0.95],
            ticktext=["Void", "50%", "80%", "Solid"],
            x=1.02, thickness=12, len=0.7,
        ),
        opacity=0.88,
        caps=dict(x_show=False, y_show=False, z_show=False),
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.3),
        name="Density field",
        showlegend=True,
        hovertemplate="x=%{x:.1f} y=%{y:.1f} z=%{z:.1f}<br>ρ=%{value:.2f}<extra></extra>",
    )]


def parse_msh_surface(msh_bytes: bytes):
    """Parse Gmsh .msh v2.2 bytes.

    Returns
    -------
    nodes     : dict {node_id: (x, y, z)}
    triangles : dict {group_name: list of (n1, n2, n3)}
    """
    lines = msh_bytes.decode("utf-8", errors="replace").splitlines()
    nodes, phys_names, triangles = {}, {}, {}
    i = 0
    while i < len(lines):
        tag = lines[i].strip()

        if tag == "$PhysicalNames":
            i += 1
            for _ in range(int(lines[i].strip())):
                i += 1
                parts = lines[i].strip().split(None, 2)
                phys_names[int(parts[1])] = parts[2].strip('"')

        elif tag == "$Nodes":
            i += 1
            for _ in range(int(lines[i].strip())):
                i += 1
                p = lines[i].strip().split()
                nodes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))

        elif tag == "$Elements":
            i += 1
            for _ in range(int(lines[i].strip())):
                i += 1
                p = lines[i].strip().split()
                if int(p[1]) != 2:          # only triangles (type 2)
                    continue
                n_tags  = int(p[2])
                phys_id = int(p[3]) if n_tags >= 1 else 0
                name    = phys_names.get(phys_id, "FREE")
                ns      = [int(p[3 + n_tags + k]) for k in range(3)]
                triangles.setdefault(name, []).append(tuple(ns))
        i += 1
    return nodes, triangles


def fem_surface_traces(msh_bytes: bytes, render_mode: str = "solid_edges"):
    """ICEM-style FEM surface visualisation.

    Parameters
    ----------
    msh_bytes   : Gmsh .msh v2.2 bytes (from generate_fem_mesh).
    render_mode : One of:
        'solid_edges'  — flat-shaded solid faces + element edge wireframe (ICEM default)
        'solid'        — flat-shaded solid faces only (faster for large meshes)
        'wireframe'    — element edges only, transparent faces
        'quality'      — faces coloured by triangle aspect-ratio (1.0 = equilateral)

    Returns list of Plotly traces ready for go.Figure.
    """
    nodes, triangles = parse_msh_surface(msh_bytes)
    if not nodes:
        return []

    # ── ICEM-standard BC-group palette ────────────────────────
    # Blue = support/fixed, Red = load/force, Grey-steel = free
    GROUP_STYLE = {
        "FIXED": dict(face_color="#0d47a1", edge_color="rgba(130,190,255,0.55)",
                      opacity=0.93, label="Fixed (support)"),
        "LOAD":  dict(face_color="#b71c1c", edge_color="rgba(255,130,130,0.55)",
                      opacity=0.93, label="Load (force)"),
        "FREE":  dict(face_color="#263238", edge_color="rgba(170,210,230,0.35)",
                      opacity=0.78, label="Free surface"),
    }
    default_style = dict(face_color="#37474f", edge_color="rgba(200,230,255,0.3)",
                         opacity=0.75, label="Surface")

    traces = []
    edge_xs, edge_ys, edge_zs = [], [], []

    draw_faces = render_mode in ("solid_edges", "solid", "quality")
    draw_edges = render_mode in ("solid_edges", "wireframe")

    for gname, tris in triangles.items():
        sty = GROUP_STYLE.get(gname, default_style)

        # Compact local node array for this group
        nids  = sorted({n for t in tris for n in t})
        loc   = {nid: k for k, nid in enumerate(nids)}
        xs_g  = np.array([nodes[n][0] for n in nids], dtype=float)
        ys_g  = np.array([nodes[n][1] for n in nids], dtype=float)
        zs_g  = np.array([nodes[n][2] for n in nids], dtype=float)
        ii_g  = [loc[t[0]] for t in tris]
        jj_g  = [loc[t[1]] for t in tris]
        kk_g  = [loc[t[2]] for t in tris]

        # ── Solid face trace ──────────────────────────────────
        if draw_faces:
            if render_mode == "quality":
                # Per-element aspect ratio: longest / shortest edge.
                # 1.0 = perfect equilateral, higher = worse quality.
                vert_int   = np.zeros(len(nids))
                vert_cnt   = np.zeros(len(nids))
                for t in tris:
                    pts = np.array([nodes[n] for n in t])
                    el  = [np.linalg.norm(pts[1]-pts[0]),
                           np.linalg.norm(pts[2]-pts[1]),
                           np.linalg.norm(pts[0]-pts[2])]
                    ratio = max(el) / (min(el) + 1e-9)
                    for n in t:
                        vert_int[loc[n]] += ratio
                        vert_cnt[loc[n]] += 1
                vert_cnt = np.maximum(vert_cnt, 1)
                vert_int /= vert_cnt

                traces.append(go.Mesh3d(
                    x=xs_g, y=ys_g, z=zs_g,
                    i=ii_g, j=jj_g, k=kk_g,
                    intensity=vert_int,
                    colorscale=[
                        [0.00, "#1b5e20"],   # 1× — excellent (dark green)
                        [0.25, "#66bb6a"],   # 1.5 — good
                        [0.55, "#fdd835"],   # 2×  — acceptable
                        [0.80, "#ef6c00"],   # 3×  — coarse
                        [1.00, "#b71c1c"],   # 5+× — poor
                    ],
                    cmin=1.0, cmax=5.0,
                    showscale=(gname == list(triangles.keys())[0]),  # once
                    colorbar=dict(
                        title=dict(text="Aspect ratio",
                                   font=dict(color="white", size=11)),
                        tickfont=dict(color="white", size=10),
                        tickvals=[1, 2, 3, 5],
                        ticktext=["1 (ideal)", "2", "3", "5+"],
                        x=1.02, thickness=12, len=0.65,
                    ),
                    flatshading=True,
                    lighting=dict(ambient=0.55, diffuse=0.9,
                                  specular=0.1, roughness=0.6),
                    lightposition=dict(x=1e4, y=2e4, z=3e4),
                    name=f"{sty['label']} — quality",
                    showlegend=True,
                    hovertemplate=(
                        f"{sty['label']}<br>"
                        "x=%{x:.2f}  y=%{y:.2f}  z=%{z:.2f}<br>"
                        "AR=%{intensity:.2f}<extra></extra>"
                    ),
                ))
            else:
                traces.append(go.Mesh3d(
                    x=xs_g, y=ys_g, z=zs_g,
                    i=ii_g, j=jj_g, k=kk_g,
                    color=sty["face_color"],
                    opacity=sty["opacity"],
                    flatshading=True,
                    lighting=dict(ambient=0.45, diffuse=0.9,
                                  specular=0.35, roughness=0.35, fresnel=0.05),
                    lightposition=dict(x=1e4, y=2e4, z=3e4),
                    name=sty["label"],
                    showlegend=True,
                    hovertemplate=(
                        f"{sty['label']}<br>"
                        "x=%{x:.2f}  y=%{y:.2f}  z=%{z:.2f}<extra></extra>"
                    ),
                ))

        # ── Collect edge segments (deduplicated per group) ─────
        if draw_edges:
            seen_edges: set = set()
            for t in tris:
                for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    key = (min(a, b), max(a, b))
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    xa, ya, za = nodes[a]
                    xb, yb, zb = nodes[b]
                    edge_xs += [xa, xb, None]
                    edge_ys += [ya, yb, None]
                    edge_zs += [za, zb, None]

    # ── Single merged edge trace (one draw call — much faster) ─
    if draw_edges and edge_xs:
        e_alpha = 0.70 if render_mode == "wireframe" else 0.40
        traces.append(go.Scatter3d(
            x=edge_xs, y=edge_ys, z=edge_zs,
            mode="lines",
            line=dict(color=f"rgba(200,225,255,{e_alpha})", width=1),
            name="Element edges",
            showlegend=True,
            hoverinfo="skip",
        ))

    return traces


def fem_quality_stats(msh_bytes: bytes) -> dict:
    """Compute per-group triangle quality statistics.

    Returns dict with keys: 'FIXED', 'LOAD', 'FREE', each a dict with
    n_tris, mean_ar, max_ar, pct_good (AR < 2), pct_poor (AR > 4).
    """
    nodes, triangles = parse_msh_surface(msh_bytes)
    stats = {}
    for gname, tris in triangles.items():
        ars = []
        for t in tris:
            pts = np.array([nodes[n] for n in t])
            el  = [np.linalg.norm(pts[1]-pts[0]),
                   np.linalg.norm(pts[2]-pts[1]),
                   np.linalg.norm(pts[0]-pts[2])]
            ars.append(max(el) / (min(el) + 1e-9))
        ar = np.array(ars)
        stats[gname] = {
            "n_tris":   len(ars),
            "mean_ar":  float(np.mean(ar)),
            "max_ar":   float(np.max(ar)),
            "pct_good": float((ar < 2.0).sum() / len(ar) * 100),
            "pct_poor": float((ar > 4.0).sum() / len(ar) * 100),
        }
    return stats


def sculpt_solid_frames(xPhys, box_w, box_h, box_d, color, n_frames=16, upsample=3):
    """Smooth sculpt animation via marching cubes on an upsampled density field.

    Each frame extracts the iso-surface at a progressively higher density
    threshold — the block smoothly erodes from a full solid down to the
    structural skeleton, like watching clay being carved.

    Parameters
    ----------
    upsample  : zoom factor before MC (3 gives ~3× finer apparent resolution)
    n_frames  : total animation frames (16 is smooth without being slow to build)

    Returns
    -------
    frames     : list[go.Frame]
    thresholds : np.ndarray
    """
    from scipy.ndimage import zoom as _zoom

    # ── Build 3D density volume ───────────────────────────────────────────────
    if xPhys.ndim == 2:
        nely, nelx = xPhys.shape
        vox_size = box_w / nelx
        nelz = max(4, int(round(box_d / vox_size)))
        xp = np.repeat(xPhys[:, :, np.newaxis], nelz, axis=2)
    else:
        nely, nelx, nelz = xPhys.shape
        xp = xPhys

    # ── Upsample for finer mesh ───────────────────────────────────────────────
    if upsample > 1:
        xp = _zoom(xp.astype(np.float32), upsample, order=1)

    ny, nx, nz = xp.shape
    dx = box_w / nx
    dy = box_h / ny
    dz = box_d / nz
    spacing = (dy, dx, dz)  # (Y, X, Z)

    d_max = float(xp.max())
    lo = 0.04
    hi = min(d_max - 0.01, 0.82)
    thresholds = np.linspace(lo, hi, n_frames)

    _lighting  = dict(ambient=0.45, diffuse=0.85, specular=0.55,
                      roughness=0.25, fresnel=0.4)
    _light_pos = dict(x=box_w * 1.5, y=box_h * 2.5, z=box_d * 3.0)

    frames = []
    for thr in thresholds:
        if xp.max() <= thr:
            frames.append(go.Frame(
                data=[go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[],
                                showlegend=False)],
                name=f"{thr:.3f}",
            ))
            continue

        try:
            verts, faces, _, _ = measure.marching_cubes(
                xp, level=thr, spacing=spacing, allow_degenerate=False
            )
        except (ValueError, RuntimeError):
            frames.append(go.Frame(
                data=[go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[],
                                showlegend=False)],
                name=f"{thr:.3f}",
            ))
            continue

        if len(verts) == 0:
            frames.append(go.Frame(
                data=[go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[],
                                showlegend=False)],
                name=f"{thr:.3f}",
            ))
            continue

        # verts: (V, 3) in physical units, marching_cubes returns (Y, X, Z)
        px = verts[:, 1].astype(np.float32)
        py = verts[:, 0].astype(np.float32)
        pz = verts[:, 2].astype(np.float32)

        frames.append(go.Frame(
            data=[go.Mesh3d(
                x=px, y=py, z=pz,
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color,
                opacity=1.0,
                flatshading=False,   # smooth shading = organic sculpted appearance
                lighting=_lighting,
                lightposition=_light_pos,
                showlegend=False,
                hoverinfo="skip",
            )],
            name=f"{thr:.3f}",
        ))

    return frames, thresholds


def solid_box_mesh(w, h, d, color, opacity=1.0):
    """A fully solid opaque box Mesh3d — used as the pre-optimisation reference."""
    x = [0, w, w, 0, 0, w, w, 0]
    y = [0, 0, h, h, 0, 0, h, h]
    z = [0, 0, 0, 0, d, d, d, d]
    ii = [0, 0,  3, 3,  0, 0,  4, 4,  0, 0,  1, 1]
    jj = [1, 5,  2, 6,  1, 2,  5, 6,  4, 7,  2, 6]
    kk = [5, 4,  6, 7,  2, 3,  6, 7,  7, 3,  6, 5]
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=ii, j=jj, k=kk,
        color=color,
        opacity=opacity,
        flatshading=False,
        lighting=dict(ambient=0.45, diffuse=0.85, specular=0.45,
                      roughness=0.25, fresnel=0.35),
        lightposition=dict(x=w * 1.5, y=h * 2.5, z=d * 3.0),
        showlegend=False,
        hoverinfo="skip",
        name="Design space",
    )


def before_after_traces(box_w, box_h, box_d):
    """Return a semi-transparent filled bounding box (original design space)."""
    corners = [
        [0, 0, 0], [box_w, 0, 0], [box_w, box_h, 0], [0, box_h, 0],
        [0, 0, box_d], [box_w, 0, box_d], [box_w, box_h, box_d], [0, box_h, box_d],
    ]
    x = [c[0] for c in corners]
    y = [c[1] for c in corners]
    z = [c[2] for c in corners]
    # 12 triangles for a box (2 per face × 6 faces)
    i = [0, 0, 1, 1, 2, 2, 4, 4, 0, 0, 3, 3]
    j = [1, 2, 2, 5, 3, 6, 5, 6, 4, 7, 7, 6]
    k = [2, 3, 5, 6, 6, 7, 6, 7, 7, 3, 6, 2]
    return [go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='#888888',
        opacity=0.06,
        flatshading=True,
        name='Original space',
        showlegend=True,
        hoverinfo='skip',
    )]


def wireframe(w, h, d, alpha=0.3):
    vx = [0, w, w, 0, 0, w, w, 0]
    vy = [0, 0, h, h, 0, 0, h, h]
    vz = [0, 0, 0, 0, d, d, d, d]
    tr = []
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ]
    for e in edges:
        tr.append(go.Scatter3d(
            x=[vx[e[0]], vx[e[1]]],
            y=[vy[e[0]], vy[e[1]]],
            z=[vz[e[0]], vz[e[1]]],
            mode="lines",
            line=dict(color=f"rgba(100,160,255,{alpha})", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
    return tr


def design_space_box(w, h, d, fill_opacity=0.07, edge_alpha=0.55):
    """Solid semi-transparent design-space box: filled faces + bright edges.

    Replaces bare wireframe() in the main 3-D view to give a proper
    sense of the volume the optimizer is working inside.

    Corners (index):
      0(0,0,0)  1(W,0,0)  2(W,H,0)  3(0,H,0)
      4(0,0,D)  5(W,0,D)  6(W,H,D)  7(0,H,D)
    """
    x = [0, w, w, 0, 0, w, w, 0]
    y = [0, 0, h, h, 0, 0, h, h]
    z = [0, 0, 0, 0, d, d, d, d]

    # 12 triangles covering all 6 faces
    # Bottom Y=0:  0,1,5 / 0,5,4
    # Top    Y=H:  3,2,6 / 3,6,7
    # Front  Z=0:  0,1,2 / 0,2,3
    # Back   Z=D:  4,5,6 / 4,6,7
    # Left   X=0:  0,4,7 / 0,7,3
    # Right  X=W:  1,2,6 / 1,6,5
    ii = [0, 0,  3, 3,  0, 0,  4, 4,  0, 0,  1, 1]
    jj = [1, 5,  2, 6,  1, 2,  5, 6,  4, 7,  2, 6]
    kk = [5, 4,  6, 7,  2, 3,  6, 7,  7, 3,  6, 5]

    traces = [
        go.Mesh3d(
            x=x, y=y, z=z,
            i=ii, j=jj, k=kk,
            color="#1a3a5c",
            opacity=fill_opacity,
            flatshading=False,
            showlegend=False,
            hoverinfo="skip",
            lighting=dict(ambient=0.9, diffuse=0.3),
            name="Design space",
        )
    ]

    # Edges on top — slightly brighter than the old wireframe
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ]
    for e in edges:
        traces.append(go.Scatter3d(
            x=[x[e[0]], x[e[1]]],
            y=[y[e[0]], y[e[1]]],
            z=[z[e[0]], z[e[1]]],
            mode="lines",
            line=dict(color=f"rgba(120,180,255,{edge_alpha})", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
    return traces


def element_mesh_traces(xPhys, box_w, box_h, box_d, threshold=0.3, upsample=2):
    """FEA-style tet-surface element mesh via marching cubes.

    Upsamples the density field, extracts the iso-surface at `threshold`,
    and renders it with flat per-triangle shading + edge wireframe — visually
    identical to how OptiStruct/Inspire display tet mesh surfaces.

    Parameters
    ----------
    upsample  : integer zoom factor applied before MC (2 = 2× finer mesh)
    threshold : iso-level for the surface; elements below are void
    """
    from scipy.ndimage import zoom as _zoom, map_coordinates as _mapc

    # ── Build 3D density volume ───────────────────────────────────────────────
    if xPhys.ndim == 2:
        nely, nelx = xPhys.shape
        vox_size = box_w / nelx
        nelz = max(4, int(round(box_d / vox_size)))
        xp = np.repeat(xPhys[:, :, np.newaxis], nelz, axis=2)   # (nely,nelx,nelz)
    else:
        nely, nelx, nelz = xPhys.shape
        xp = xPhys

    # ── Upsample ─────────────────────────────────────────────────────────────
    if upsample > 1:
        xp = _zoom(xp.astype(np.float32), upsample, order=1)

    ny, nx, nz = xp.shape
    dx = box_w / nx
    dy = box_h / ny
    dz = box_d / nz
    spacing = (dy, dx, dz)   # skimage uses (row,col,depth) = (Y,X,Z)

    if xp.max() <= threshold:
        return []

    # ── Marching cubes at density threshold ──────────────────────────────────
    try:
        verts, faces, _, _ = measure.marching_cubes(
            xp, level=threshold, spacing=spacing, allow_degenerate=False
        )
    except (ValueError, RuntimeError):
        return []

    if len(verts) == 0 or len(faces) == 0:
        return []

    # ── Sample density at each vertex (for coloring) ─────────────────────────
    # verts[:,0]=Y, verts[:,1]=X, verts[:,2]=Z in physical units → grid indices
    vy = np.clip(verts[:, 0] / dy, 0, ny - 1)
    vx = np.clip(verts[:, 1] / dx, 0, nx - 1)
    vz = np.clip(verts[:, 2] / dz, 0, nz - 1)
    vertex_dens = _mapc(xp.astype(np.float64), [vy, vx, vz], order=1, mode='nearest')
    vertex_dens = np.clip(vertex_dens, 0.0, 1.0).astype(np.float32)

    px = verts[:, 1].astype(np.float32)   # X
    py = verts[:, 0].astype(np.float32)   # Y
    pz = verts[:, 2].astype(np.float32)   # Z

    solid = go.Mesh3d(
        x=px, y=py, z=pz,
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=vertex_dens,
        colorscale=[
            [0.0,  "#0d1b2e"],
            [0.3,  "#1565c0"],
            [0.6,  "#42a5f5"],
            [0.85, "#4CAF50"],
            [1.0,  "#ffffff"],
        ],
        cmin=0.0, cmax=1.0,
        showscale=True,
        colorbar=dict(
            title=dict(text="Density", font=dict(color="white", size=11)),
            tickfont=dict(color="white", size=10),
            x=1.02, thickness=10, len=0.6,
            tickvals=[0, 0.5, 1],
            ticktext=["Void", "0.5", "Solid"],
        ),
        flatshading=True,          # flat per-triangle shading = classic FEA mesh look
        opacity=0.95,
        lighting=dict(ambient=0.45, diffuse=0.85, specular=0.3, roughness=0.35),
        lightposition=dict(x=box_w * 2, y=box_h * 2, z=box_d * 3),
        name="Element mesh",
        showlegend=True,
        hoverinfo="skip",
    )

    # ── Triangle edge wireframe ───────────────────────────────────────────────
    # Build edge list (each triangle contributes 3 edges; use None separator)
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    nans = np.full(len(faces), np.nan, dtype=np.float32)

    ex = np.stack([px[f0], px[f1], nans, px[f1], px[f2], nans, px[f2], px[f0], nans], axis=1).ravel()
    ey = np.stack([py[f0], py[f1], nans, py[f1], py[f2], nans, py[f2], py[f0], nans], axis=1).ravel()
    ez = np.stack([pz[f0], pz[f1], nans, pz[f1], pz[f2], nans, pz[f2], pz[f0], nans], axis=1).ravel()

    # Thin edges — every other triangle's edges to avoid overdraw on fine mesh
    stride = max(1, len(faces) // 800)
    sub = np.arange(0, len(faces), stride)
    ex2 = np.stack([px[f0[sub]], px[f1[sub]], nans[sub],
                    px[f1[sub]], px[f2[sub]], nans[sub],
                    px[f2[sub]], px[f0[sub]], nans[sub]], axis=1).ravel()
    ey2 = np.stack([py[f0[sub]], py[f1[sub]], nans[sub],
                    py[f1[sub]], py[f2[sub]], nans[sub],
                    py[f2[sub]], py[f0[sub]], nans[sub]], axis=1).ravel()
    ez2 = np.stack([pz[f0[sub]], pz[f1[sub]], nans[sub],
                    pz[f1[sub]], pz[f2[sub]], nans[sub],
                    pz[f2[sub]], pz[f0[sub]], nans[sub]], axis=1).ravel()

    edges = go.Scatter3d(
        x=ex2, y=ey2, z=ez2,
        mode="lines",
        line=dict(color="rgba(160, 200, 255, 0.22)", width=1),
        showlegend=False,
        hoverinfo="skip",
        name="Tet edges",
    )

    return [solid, edges]


def face_trace(face, w, h, d, color, opacity, name):
    fv = {
        "Left (X=0)":   ([0,0,0,0], [0,0,h,h], [0,d,d,0]),
        "Right (X=W)":  ([w,w,w,w], [0,0,h,h], [0,d,d,0]),
        "Bottom (Y=0)": ([0,w,w,0], [0,0,0,0], [0,0,d,d]),
        "Top (Y=H)":    ([0,w,w,0], [h,h,h,h], [0,0,d,d]),
        "Front (Z=0)":  ([0,w,w,0], [0,0,h,h], [0,0,0,0]),
        "Back (Z=D)":   ([0,w,w,0], [0,0,h,h], [d,d,d,d]),
    }
    x, y, z = fv.get(face, fv["Left (X=0)"])
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=color, opacity=opacity,
        name=name, showlegend=True,
    )


def face_center(face, w, h, d):
    return {
        "Left (X=0)":   (0,   h/2, d/2),
        "Right (X=W)":  (w,   h/2, d/2),
        "Bottom (Y=0)": (w/2, 0,   d/2),
        "Top (Y=H)":    (w/2, h,   d/2),
        "Front (Z=0)":  (w/2, h/2, 0),
        "Back (Z=D)":   (w/2, h/2, d),
    }.get(face, (w/2, h/2, d/2))


def arrow_traces(tx, ty, tz, direction, length=15, force_n=None):
    """Return (line_trace, cone_trace, annotation_trace) for a force arrow.

    direction : str  e.g. "-Y", "+X"
              | tuple/list (fx, fy, fz)  arbitrary vector — will be normalised
    """
    if isinstance(direction, (tuple, list, np.ndarray)):
        fx, fy, fz = float(direction[0]), float(direction[1]), float(direction[2])
        mag = (fx**2 + fy**2 + fz**2) ** 0.5
        if mag > 1e-9:
            dx, dy, dz = fx / mag, fy / mag, fz / mag
        else:
            dx, dy, dz = 0.0, -1.0, 0.0
    else:
        dirs = {
            "-X": (-1,0,0), "+X": (1,0,0),
            "-Y": (0,-1,0), "+Y": (0,1,0),
            "-Z": (0,0,-1), "+Z": (0,0,1),
        }
        dx, dy, dz = dirs.get(direction, (0,-1,0))

    line = go.Scatter3d(
        x=[tx - dx*length, tx],
        y=[ty - dy*length, ty],
        z=[tz - dz*length, tz],
        mode="lines",
        line=dict(color="red", width=5),
        showlegend=False,
        hoverinfo="skip",
    )
    cone = go.Cone(
        x=[tx], y=[ty], z=[tz],
        u=[dx*8], v=[dy*8], w=[dz*8],
        colorscale=[[0, "red"], [1, "red"]],
        showscale=False,
        sizemode="absolute",
        sizeref=7,
        showlegend=False,
    )

    label = f"{force_n:.0f}N" if force_n is not None else direction
    annot = go.Scatter3d(
        x=[tx - dx * (length + 8)],
        y=[ty - dy * (length + 8)],
        z=[tz - dz * (length + 8)],
        mode="text",
        text=[label],
        textfont=dict(color="red", size=13),
        showlegend=False,
        hoverinfo="skip",
    )
    return line, cone, annot


def fixed_label_trace(tx, ty, tz):
    """Return a 'FIXED' text annotation at the fixed face center."""
    return go.Scatter3d(
        x=[tx], y=[ty], z=[tz],
        mode="text",
        text=["FIXED"],
        textfont=dict(color="royalblue", size=13),
        showlegend=False,
        hoverinfo="skip",
    )


def scene3d(w, h, d, height=500):
    return dict(
        scene=dict(
            xaxis=dict(title="X (mm)", backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="#1e1e1e", zerolinecolor="#333"),
            yaxis=dict(title="Y (mm)", backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="#1e1e1e", zerolinecolor="#333"),
            zaxis=dict(title="Z (mm)", backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="#1e1e1e", zerolinecolor="#333"),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=0.9, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white", size=11),
        ),
    )


def rgba(h, a=0.3):
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return f"rgba({r},{g},{b},{a})"


def norm_v(v, vals):
    mn, mx = min(vals), max(vals)
    return (v - mn) / (mx - mn) if mx != mn else 0.5
