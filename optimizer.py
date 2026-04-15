import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


def get_Ke(nu):
    k = np.array([
        1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
        -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8
    ])
    return np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
    ])


def build_filter(nelx, nely, rmin):
    iH, jH, sH = [], [], []
    for i1 in range(nelx):
        for j1 in range(nely):
            row = j1 + i1 * nely
            for i2 in range(max(i1 - int(rmin), 0), min(i1 + int(rmin) + 1, nelx)):
                for j2 in range(max(j1 - int(rmin), 0), min(j1 + int(rmin) + 1, nely)):
                    fac = rmin - np.sqrt((i1 - i2)**2 + (j1 - j2)**2)
                    if fac > 0:
                        iH.append(row)
                        jH.append(j2 + i2 * nely)
                        sH.append(fac)
    H = csc_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely))
    return H, np.asarray(H.sum(axis=1)).flatten()


def simp_core(nelx, nely, volfrac, nu, penal, H, Hs, max_iter, load_cases,
              fixed_dofs=None, yield_callback=None, design_mask=None):
    """Run SIMP topology optimization.

    Parameters
    ----------
    load_cases : list of (ndarray(ndof), float)
        Each entry is (force_vector, weight).  The force_vector is a full
        ndof-length array with applied nodal forces.
    fixed_dofs : ndarray or None
        DOF indices to constrain.  Defaults to all DOFs on the Left (X=0) face
        for backward compatibility when called from chat mode.
    design_mask : ndarray (nely, nelx) bool or None
        Boolean mask of the active design domain.  Elements where mask=False
        are pinned to density=0 (Emin stiffness) and excluded from the volume
        target.  When None the full rectangular box is used (default behaviour).

    Returns
    -------
    xPhys : ndarray (nely, nelx)
    history : list[float]
    stress_field : ndarray (nely, nelx)
    """
    E0, Emin = 1.0, 1e-9
    Ke = get_Ke(nu)
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Build element DOF map
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = [
                2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,
                2*n2,   2*n2+1, 2*n1,   2*n1+1
            ]

    iK = np.kron(edofMat, np.ones((8, 1), dtype=int)).flatten()
    jK = np.kron(edofMat, np.ones((1, 8), dtype=int)).flatten()

    if fixed_dofs is None:
        # Backward-compat default: fix all DOFs on Left (X=0) face
        fixed_dofs = np.array(
            [2*iy for iy in range(nely + 1)] +
            [2*iy + 1 for iy in range(nely + 1)]
        )
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)

    # Flatten mask in Fortran (column-major) order to match xvec layout
    if design_mask is not None:
        # Ensure 2-D (nely, nelx); if 3-D mask was passed, project to 2-D
        _dm = design_mask if design_mask.ndim == 2 else design_mask.any(axis=2)
        mask_flat_F = _dm.flatten(order='F')   # True = active design element
        n_design    = int(mask_flat_F.sum())
        vol_target  = volfrac * n_design
    else:
        mask_flat_F = None
        vol_target  = volfrac * nelx * nely

    xPhys = np.full((nely, nelx), volfrac)
    if mask_flat_F is not None:
        xPhys[~_dm] = 0.0          # void outside design domain from the start

    history = []
    change = 1.0
    U_final = np.zeros(ndof)

    for _ in range(max_iter):
        xvec = xPhys.flatten(order='F')
        E_el = Emin + xvec**penal * (E0 - Emin)
        sK = np.outer(Ke.flatten(), E_el).flatten(order='F')
        K = csc_matrix((sK, (iK, jK)), shape=(ndof, ndof))
        K = (K + K.T) / 2
        Kff = K[np.ix_(free_dofs, free_dofs)]

        obj = 0.0
        dc = np.zeros(nelx * nely)
        U_final = np.zeros(ndof)

        for (f_vec, w) in load_cases:
            u = np.zeros(ndof)
            u[free_dofs] = spsolve(Kff, f_vec[free_dofs])
            U_final += u
            ce = (np.dot(u[edofMat], Ke) * u[edofMat]).sum(axis=1)
            obj += w * float(np.dot(E_el, ce))
            dc -= w * penal * xvec**(penal - 1) * (E0 - Emin) * ce

        dc = np.asarray(H.dot(xvec * dc)).flatten() / Hs / np.maximum(1e-3, xvec)

        # Zero sensitivity outside design domain so OC ignores those elements
        if mask_flat_F is not None:
            dc[~mask_flat_F] = 0.0

        l1, l2, move = 0.0, 1e9, 0.2
        for _ in range(200):
            if (l2 - l1) < 1e-9 * (1.0 + l1):
                break
            lmid = 0.5 * (l1 + l2)
            be = np.maximum(1e-10, -dc / (np.ones(nelx * nely) * lmid))
            xnew = np.clip(
                xvec * np.sqrt(be),
                np.maximum(0., xvec - move),
                np.minimum(1., xvec + move)
            )
            # Volume check counts only active design elements
            _xsum = xnew[mask_flat_F].sum() if mask_flat_F is not None else xnew.sum()
            l1, l2 = (lmid, l2) if _xsum > vol_target else (l1, lmid)

        # Pin non-design elements to void
        if mask_flat_F is not None:
            xnew[~mask_flat_F] = 0.0

        change = float(np.max(np.abs(xnew - xvec)))
        xPhys = np.asarray(H.dot(xnew) / Hs).flatten().reshape((nely, nelx), order='F')
        if mask_flat_F is not None:
            xPhys[~_dm] = 0.0      # filter smoothing can bleed across boundary
        history.append(obj)

        if yield_callback is not None:
            yield_callback(len(history), max_iter, obj, xPhys, change)

        if change < 0.01:
            break

    # Post-convergence: von Mises stress per element (plane stress, E=1 reference)
    # Constitutive matrix D for plane stress
    _c = 1.0 / (1.0 - nu * nu)
    D_ps = _c * np.array([
        [1.0,  nu,            0.0],
        [nu,   1.0,           0.0],
        [0.0,  0.0, (1.0-nu)/2.0],
    ])

    # Strain-displacement B matrix at element centroid for unit Q4 element.
    # DOF order: [LT_x, LT_y, RT_x, RT_y, RB_x, RB_y, LB_x, LB_y]
    # matches edofMat: [2n1+2, 2n1+3, 2n2+2, 2n2+3, 2n2, 2n2+1, 2n1, 2n1+1]
    B_q4 = np.array([
        [-0.5,  0.0,  0.5,  0.0,  0.5,  0.0, -0.5,  0.0],
        [ 0.0,  0.5,  0.0,  0.5,  0.0, -0.5,  0.0, -0.5],
        [ 0.5, -0.5,  0.5,  0.5, -0.5,  0.5, -0.5, -0.5],
    ])  # (3, 8)

    xvec_final = xPhys.flatten(order='F')
    E_el_final = Emin + xvec_final**penal * (E0 - Emin)

    # Vectorised: gather element displacements (n_el, 8) → strain (n_el, 3) → stress
    u_all     = U_final[edofMat]           # (n_el, 8)
    eps_all   = u_all @ B_q4.T            # (n_el, 3): [εx, εy, γxy]
    sigma_all = E_el_final[:, None] * (eps_all @ D_ps.T)  # (n_el, 3): [σx, σy, τxy]

    sx, sy, txy = sigma_all[:, 0], sigma_all[:, 1], sigma_all[:, 2]
    vm = np.sqrt(np.maximum(sx**2 - sx*sy + sy**2 + 3.0*txy**2, 0.0))

    stress_field = vm.reshape(nely, nelx, order='F')

    return xPhys, history, stress_field


# ─────────────────────────────────────────────────────────────────────────────
#  Load / support spec helpers
# ─────────────────────────────────────────────────────────────────────────────

def _node(ix, iy, nely):
    """Global node index from grid coordinates."""
    return iy + ix * (nely + 1)


def _face_nodes(face, nelx, nely):
    """All node indices on a named bounding-box face."""
    if face == "Left (X=0)":
        return [_node(0, iy, nely) for iy in range(nely + 1)]
    if face == "Right (X=W)":
        return [_node(nelx, iy, nely) for iy in range(nely + 1)]
    if face == "Bottom (Y=0)":
        return [_node(ix, 0, nely) for ix in range(nelx + 1)]
    if face == "Top (Y=H)":
        return [_node(ix, nely, nely) for ix in range(nelx + 1)]
    # Front / Back are Z-faces; no Z DOF in 2-D SIMP → project onto all nodes
    return [_node(ix, iy, nely) for ix in range(nelx + 1) for iy in range(nely + 1)]


def _spec_to_f(spec, nelx, nely, ndof):
    """Convert one load-spec dict to a nodal force vector (ndof,)."""
    f = np.zeros(ndof)
    load_type = spec.get("type", "surface")
    face      = spec.get("face", "Right (X=W)")
    fx        = float(spec.get("fx", 0.0))
    fy        = float(spec.get("fy", 0.0))

    if load_type == "surface":
        nodes = _face_nodes(face, nelx, nely)
        n = len(nodes)
        if n:
            for nd in nodes:
                f[2 * nd]     += fx / n
                f[2 * nd + 1] += fy / n

    elif load_type == "point":
        u = float(spec.get("u", 0.5))       # 0-1 along face "height" axis
        if face in ("Left (X=0)", "Right (X=W)"):
            ix = 0 if face == "Left (X=0)" else nelx
            iy = int(round(np.clip(u * nely, 0, nely)))
        elif face in ("Bottom (Y=0)", "Top (Y=H)"):
            iy = 0 if face == "Bottom (Y=0)" else nely
            ix = int(round(np.clip(u * nelx, 0, nelx)))
        else:
            ix, iy = nelx, nely // 2
        nd = _node(ix, iy, nely)
        f[2 * nd]     += fx
        f[2 * nd + 1] += fy

    elif load_type == "line":
        u_start = float(spec.get("u_start", 0.0))
        u_end   = float(spec.get("u_end",   1.0))
        if face in ("Left (X=0)", "Right (X=W)"):
            ix       = 0 if face == "Left (X=0)" else nelx
            iy_start = int(round(np.clip(u_start * nely, 0, nely)))
            iy_end   = int(round(np.clip(u_end   * nely, 0, nely)))
            nodes    = [_node(ix, iy, nely) for iy in range(iy_start, iy_end + 1)]
        elif face in ("Bottom (Y=0)", "Top (Y=H)"):
            iy       = 0 if face == "Bottom (Y=0)" else nely
            ix_start = int(round(np.clip(u_start * nelx, 0, nelx)))
            ix_end   = int(round(np.clip(u_end   * nelx, 0, nelx)))
            nodes    = [_node(ix, iy, nely) for ix in range(ix_start, ix_end + 1)]
        else:
            nodes = _face_nodes(face, nelx, nely)
        n = len(nodes)
        if n:
            for nd in nodes:
                f[2 * nd]     += fx / n
                f[2 * nd + 1] += fy / n

    elif load_type == "moment":
        # Couple in 2-D: equal-and-opposite X-forces at Y-extremes of the face.
        # Moment (N·mm) about Z = F × box_height_elements.
        # Direction: positive magnitude → CCW (right-hand Z).
        magnitude = float(spec.get("magnitude", 0.0))
        if face in ("Left (X=0)", "Right (X=W)"):
            ix    = 0 if face == "Left (X=0)" else nelx
            n_top = _node(ix, nely, nely)
            n_bot = _node(ix, 0,    nely)
            arm   = max(nely, 1)
        else:
            iy    = 0 if face == "Bottom (Y=0)" else nely
            n_top = _node(nelx, iy, nely)
            n_bot = _node(0,    iy, nely)
            arm   = max(nelx, 1)
        f_couple = magnitude / arm
        f[2 * n_top]     += f_couple    # +X at "top"
        f[2 * n_bot]     -= f_couple    # −X at "bottom"

    return f


def _support_to_dofs(spec, nelx, nely):
    """Convert a support-spec dict to a list of constrained DOF indices."""
    face     = spec.get("face", "Left (X=0)")
    sup_type = spec.get("type", "fixed")
    nodes    = _face_nodes(face, nelx, nely)
    dofs = []
    for nd in nodes:
        if sup_type == "fixed":
            dofs += [2 * nd, 2 * nd + 1]
        elif sup_type == "roller_x":
            dofs.append(2 * nd)        # fix X only (allows sliding in Y)
        elif sup_type == "roller_y":
            dofs.append(2 * nd + 1)    # fix Y only (allows sliding in X)
    return dofs


def build_load_cases(load_specs, support_specs, nelx, nely):
    """Convert UI load/support specs to inputs for simp_core.

    All load specs are summed into one force vector and normalised to unit
    magnitude (topology is scale-invariant).  Returns:

        load_cases : list[(ndarray(ndof), float)]   — for simp_core
        fixed_dofs : ndarray                        — constrained DOF indices
        F_mag      : float                          — original force magnitude [N]
    """
    ndof = 2 * (nelx + 1) * (nely + 1)

    F = np.zeros(ndof)
    for spec in load_specs:
        F += _spec_to_f(spec, nelx, nely, ndof)

    mag = np.linalg.norm(F)
    if mag > 1e-12:
        F /= mag
    else:
        mag = 1.0
        # Fallback: downward at right-face midpoint
        mid_r = (nely + 1) * nelx + nely // 2
        F[2 * mid_r + 1] = -1.0

    fixed_set = set()
    for spec in support_specs:
        fixed_set.update(_support_to_dofs(spec, nelx, nely))

    if not fixed_set:
        # Fallback: fix all DOFs on Left (X=0)
        for iy in range(nely + 1):
            nd = _node(0, iy, nely)
            fixed_set.add(2 * nd)
            fixed_set.add(2 * nd + 1)

    return [(F, 1.0)], np.array(sorted(fixed_set)), float(mag)


# Direction string → (fx_unit, fy_unit)
_DIR_UNIT = {
    "-X": (-1.0,  0.0), "+X": (1.0,  0.0),
    "-Y": ( 0.0, -1.0), "+Y": (0.0,  1.0),
    "-Z": ( 0.0,  0.0), "+Z": (0.0,  0.0),  # Z has no DOF in 2-D
}


def build_load_cases_multi(scenarios, support_specs, nelx, nely):
    """Build multiple weighted load cases from named scenario dicts.

    Each scenario dict must have:
        name      : str
        face      : one of FACES
        direction : "-X"|"+X"|"-Y"|"+Y"|"-Z"|"+Z"
        magnitude : float (Newtons)
        weight    : float (relative; will be normalised to sum=1)

    Returns
    -------
    load_cases  : list[(ndarray(ndof), float)]
    fixed_dofs  : ndarray
    F_mag_eff   : float  — weighted-average physical force magnitude [N]
                          for use in stress scaling
    """
    ndof = 2 * (nelx + 1) * (nely + 1)

    load_cases = []
    raw_mags   = []
    raw_weights = []

    for sc in scenarios:
        ufx, ufy = _DIR_UNIT.get(sc.get("direction", "-Y"), (0.0, -1.0))
        mag_phys  = float(sc.get("magnitude", 500.0))
        face      = sc.get("face", "Right (X=W)")

        # Build a surface spec and reuse _spec_to_f
        spec = {"type": "surface", "face": face, "fx": ufx * mag_phys, "fy": ufy * mag_phys}
        F = _spec_to_f(spec, nelx, nely, ndof)

        # Normalise force vector to unit magnitude for scale-invariant topology
        vec_mag = np.linalg.norm(F)
        if vec_mag > 1e-12:
            F /= vec_mag
        else:
            vec_mag = mag_phys if mag_phys > 0 else 1.0

        w = float(sc.get("weight", 1.0))
        load_cases.append((F, w))
        raw_mags.append(vec_mag)
        raw_weights.append(w)

    # Normalise weights so they sum to 1
    total_w = sum(raw_weights)
    if total_w > 1e-12:
        load_cases = [(F, w / total_w) for (F, w), w in zip(load_cases, raw_weights)]
        F_mag_eff = sum(m * w for m, w in zip(raw_mags, raw_weights)) / total_w
    else:
        F_mag_eff = float(np.mean(raw_mags)) if raw_mags else 1.0

    # Shared supports
    fixed_set = set()
    for spec in support_specs:
        fixed_set.update(_support_to_dofs(spec, nelx, nely))
    if not fixed_set:
        for iy in range(nely + 1):
            nd = _node(0, iy, nely)
            fixed_set.add(2 * nd)
            fixed_set.add(2 * nd + 1)

    return load_cases, np.array(sorted(fixed_set)), float(F_mag_eff)


def build_filter_fast(nelx, nely, rmin):
    """Vectorised build_filter — O(nelx·nely·rmin²) but no Python loops over elements.

    Falls back to build_filter for small meshes where overhead isn't worth it.
    For nelx*nely > 400 (i.e. 20×20) the vectorised version is ~4–8× faster.
    """
    if nelx * nely <= 400:
        return build_filter(nelx, nely, rmin)

    # Grid of element (i, j) — i = x-index, j = y-index
    i_idx = np.repeat(np.arange(nelx), nely)        # (n_el,)
    j_idx = np.tile(np.arange(nely),  nelx)         # (n_el,)
    row   = j_idx + i_idx * nely                     # flat row index

    ir = int(rmin)
    iH, jH, sH = [], [], []

    for di in range(-ir, ir + 1):
        for dj in range(-ir, ir + 1):
            i2 = i_idx + di
            j2 = j_idx + dj
            fac = rmin - np.sqrt(di**2 + dj**2)
            mask = (fac > 0) & (i2 >= 0) & (i2 < nelx) & (j2 >= 0) & (j2 < nely)
            if not mask.any():
                continue
            col = j2[mask] + i2[mask] * nely
            iH.append(row[mask])
            jH.append(col)
            sH.append(np.full(mask.sum(), fac))

    iH = np.concatenate(iH)
    jH = np.concatenate(jH)
    sH = np.concatenate(sH)

    H  = csc_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely))
    return H, np.asarray(H.sum(axis=1)).flatten()


# ─────────────────────────────────────────────────────────────────────────────
#  Legacy named-scenario builder  (used by chat mode)
# ─────────────────────────────────────────────────────────────────────────────

def make_load_cases(nelx, nely, scenario, w_p, w_s):
    """Build load cases from a named scenario string.

    Returns list[(ndarray(ndof), float)] compatible with simp_core.
    Fixed DOFs are not included — simp_core uses its default (Left face).
    """
    ndof = 2 * (nelx + 1) * (nely + 1)

    def _f(dof):
        v = np.zeros(ndof)
        v[dof] = -1.0
        return v

    mid_r = (nely + 1) * nelx + nely // 2
    top_r = (nely + 1) * nelx
    bot_r = (nely + 1) * nelx + nely
    d_down_mid  = 2 * mid_r + 1
    d_right_mid = 2 * mid_r
    d_down_top  = 2 * top_r + 1
    d_down_bot  = 2 * bot_r + 1

    raw = {
        "Downward only":         [(d_down_mid, 1.0)],
        "Lateral only":          [(d_right_mid, 1.0)],
        "Down + Lateral":        [(d_down_mid, w_p), (d_right_mid, w_s)],
        "Down + Upward (top)":   [(d_down_mid, w_p), (d_down_top, w_s)],
        "Down + Downward (bot)": [(d_down_mid, w_p), (d_down_bot, w_s)],
        "Symmetric (top+bot)":   [(d_down_top, 0.5), (d_down_bot, 0.5)],
        "Torsion (top+bot opp)": [(d_down_top, w_p), (d_right_mid, -w_s)],
    }.get(scenario, [(d_down_mid, 1.0)])

    return [(_f(dof), w) for (dof, w) in raw]


# ═════════════════════════════════════════════════════════════════════════════
#  TRUE 3-D SIMP  — 8-node hexahedral elements, full (x,y,z) optimisation
# ═════════════════════════════════════════════════════════════════════════════

def get_Ke_3D(nu: float) -> np.ndarray:
    """Return the 24×24 stiffness matrix for a unit hexahedron.

    Uses 2×2×2 Gauss quadrature and 3-D isotropic linear elasticity (E=1).
    The SIMP loop scales by E_el = Emin + x^p * (E0-Emin) per element.
    """
    c  = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu))
    # 3-D isotropic constitutive matrix D (6×6)
    D = c * np.array([
        [1-nu, nu,   nu,         0,         0,         0],
        [nu,   1-nu, nu,         0,         0,         0],
        [nu,   nu,   1-nu,       0,         0,         0],
        [0,    0,    0,  (1-2*nu)/2,        0,         0],
        [0,    0,    0,          0, (1-2*nu)/2,        0],
        [0,    0,    0,          0,         0, (1-2*nu)/2],
    ])

    # Corner natural coords (ξ,η,ζ) ∈ {-1,+1}^3
    nc = np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
    ], dtype=float)

    # 2-point Gauss quadrature in [-1,1] (weights = 1)
    gp = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])

    # For a unit physical element [0,1]^3 mapped from [-1,1]^3:
    #   J = 0.5·I  →  det(J) = 1/8  →  dN_phys = 2·dN_nat
    Ke = np.zeros((24, 24))
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                dN = np.zeros((3, 8))
                for i in range(8):
                    dN[0, i] = nc[i,0] * (1 + nc[i,1]*eta)  * (1 + nc[i,2]*zeta)  / 8
                    dN[1, i] = nc[i,1] * (1 + nc[i,0]*xi)   * (1 + nc[i,2]*zeta)  / 8
                    dN[2, i] = nc[i,2] * (1 + nc[i,0]*xi)   * (1 + nc[i,1]*eta)   / 8
                dNp = 2.0 * dN          # physical derivatives (Jinv = 2·I)

                B = np.zeros((6, 24))
                for i in range(8):
                    B[0, 3*i]   = dNp[0,i]                 # ε_xx = ∂u/∂x
                    B[1, 3*i+1] = dNp[1,i]                 # ε_yy = ∂v/∂y
                    B[2, 3*i+2] = dNp[2,i]                 # ε_zz = ∂w/∂z
                    B[3, 3*i]   = dNp[1,i];  B[3, 3*i+1] = dNp[0,i]  # γ_xy
                    B[4, 3*i+1] = dNp[2,i];  B[4, 3*i+2] = dNp[1,i]  # γ_yz
                    B[5, 3*i]   = dNp[2,i];  B[5, 3*i+2] = dNp[0,i]  # γ_xz

                Ke += (1.0 / 8.0) * (B.T @ D @ B)   # det(J)=1/8, weight=1
    return Ke


def build_filter_3d(nelx: int, nely: int, nelz: int,
                    rmin: float):
    """Build the density-filter matrix H for a 3-D grid."""
    n = nelx * nely * nelz
    iH, jH, sH = [], [], []
    for ix1 in range(nelx):
        for iy1 in range(nely):
            for iz1 in range(nelz):
                row = iy1 * nelx * nelz + ix1 * nelz + iz1
                i2lo = max(0, ix1 - int(rmin))
                i2hi = min(nelx, ix1 + int(rmin) + 1)
                for ix2 in range(i2lo, i2hi):
                    j2lo = max(0, iy1 - int(rmin))
                    j2hi = min(nely, iy1 + int(rmin) + 1)
                    for iy2 in range(j2lo, j2hi):
                        k2lo = max(0, iz1 - int(rmin))
                        k2hi = min(nelz, iz1 + int(rmin) + 1)
                        for iz2 in range(k2lo, k2hi):
                            fac = rmin - np.sqrt(
                                (ix1-ix2)**2 + (iy1-iy2)**2 + (iz1-iz2)**2)
                            if fac > 0:
                                col = iy2 * nelx * nelz + ix2 * nelz + iz2
                                iH.append(row); jH.append(col); sH.append(fac)
    H = csc_matrix((sH, (iH, jH)), shape=(n, n))
    return H, np.asarray(H.sum(axis=1)).flatten()


def _node_3d(ix: int, iy: int, iz: int, nely: int, nelz: int) -> int:
    """Global node index for 3-D grid (y-major, z-fastest)."""
    return iy * (nelz + 1) + iz + ix * (nely + 1) * (nelz + 1)


def simp_core_3d(nelx: int, nely: int, nelz: int,
                 volfrac: float, nu: float, penal: float,
                 H, Hs, max_iter: int, load_cases,
                 fixed_dofs=None, yield_callback=None):
    """True 3-D SIMP topology optimisation with 8-node hex elements.

    Returns
    -------
    xPhys       : ndarray (nely, nelx, nelz)
    history     : list[float]  — compliance per iteration
    stress_field: ndarray (nely, nelx, nelz) — strain energy density
    """
    E0, Emin = 1.0, 1e-9
    Ke   = get_Ke_3D(nu)                         # 24×24
    n_el = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

    # ── DOF map  (element (ix,iy,iz) → 24 global DOFs) ────────
    edofMat = np.zeros((n_el, 24), dtype=int)
    for ix in range(nelx):
        for iy in range(nely):
            for iz in range(nelz):
                el = iy * nelx * nelz + ix * nelz + iz
                ns = [
                    _node_3d(ix,   iy,   iz,   nely, nelz),
                    _node_3d(ix+1, iy,   iz,   nely, nelz),
                    _node_3d(ix+1, iy+1, iz,   nely, nelz),
                    _node_3d(ix,   iy+1, iz,   nely, nelz),
                    _node_3d(ix,   iy,   iz+1, nely, nelz),
                    _node_3d(ix+1, iy,   iz+1, nely, nelz),
                    _node_3d(ix+1, iy+1, iz+1, nely, nelz),
                    _node_3d(ix,   iy+1, iz+1, nely, nelz),
                ]
                dofs = []
                for nd in ns:
                    dofs += [3*nd, 3*nd+1, 3*nd+2]
                edofMat[el] = dofs

    iK = np.kron(edofMat, np.ones((24, 1), dtype=int)).flatten()
    jK = np.kron(edofMat, np.ones((1, 24), dtype=int)).flatten()

    # ── Default BCs: fix Left (X=0) face ──────────────────────
    if fixed_dofs is None:
        fixed_set = set()
        for iy in range(nely + 1):
            for iz in range(nelz + 1):
                nd = _node_3d(0, iy, iz, nely, nelz)
                fixed_set.update([3*nd, 3*nd+1, 3*nd+2])
        fixed_dofs = np.array(sorted(fixed_set))

    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)

    xPhys  = np.full((nely, nelx, nelz), volfrac)
    history = []
    U_final = np.zeros(ndof)

    for _it in range(max_iter):
        # Flatten in C order: index = iy*nelx*nelz + ix*nelz + iz
        xvec  = xPhys.flatten()
        E_el  = Emin + xvec**penal * (E0 - Emin)
        sK    = np.outer(Ke.flatten(), E_el).flatten(order='F')
        K     = csc_matrix((sK, (iK, jK)), shape=(ndof, ndof))
        K     = (K + K.T) / 2
        Kff   = K[np.ix_(free_dofs, free_dofs)]

        obj = 0.0
        dc  = np.zeros(n_el)
        U_final = np.zeros(ndof)

        for (f_vec, w) in load_cases:
            u = np.zeros(ndof)
            u[free_dofs] = spsolve(Kff, f_vec[free_dofs])
            U_final += u
            ce   = (np.dot(u[edofMat], Ke) * u[edofMat]).sum(axis=1)
            obj += w * float(np.dot(E_el, ce))
            dc  -= w * penal * xvec**(penal-1) * (E0 - Emin) * ce

        dc   = np.asarray(H.dot(xvec * dc)).flatten() / Hs / np.maximum(1e-3, xvec)
        l1, l2, move, target = 0.0, 1e9, 0.2, volfrac * n_el
        for _ in range(200):
            if (l2 - l1) < 1e-9 * (1.0 + l1):
                break
            lmid = 0.5 * (l1 + l2)
            be   = np.maximum(1e-10, -dc / (np.ones(n_el) * lmid))
            xnew = np.clip(
                xvec * np.sqrt(be),
                np.maximum(0., xvec - move),
                np.minimum(1., xvec + move),
            )
            l1, l2 = (lmid, l2) if xnew.sum() > target else (l1, lmid)

        change = float(np.max(np.abs(xnew - xvec)))
        xPhys  = np.asarray(H.dot(xnew) / Hs).flatten().reshape((nely, nelx, nelz))
        history.append(obj)

        if yield_callback is not None:
            yield_callback(len(history), max_iter, obj, xPhys[:,:,nelz//2], change)

        if change < 0.01:
            break

    # ── Post-convergence strain energy density ─────────────────
    xvec_f   = xPhys.flatten()
    E_el_f   = Emin + xvec_f**penal * (E0 - Emin)
    se       = np.zeros(n_el)
    for idx in range(n_el):
        ue      = U_final[edofMat[idx]]
        se[idx] = float(ue @ Ke @ ue) * E_el_f[idx]

    stress_field = se.reshape((nely, nelx, nelz))
    return xPhys, history, stress_field


# ─────────────────────────────────────────────────────────────────────────────
#  3-D BC helpers
# ─────────────────────────────────────────────────────────────────────────────

def _face_nodes_3d(face: str, nelx: int, nely: int, nelz: int):
    """All node indices on a named bounding-box face (3-D grid)."""
    nodes = []
    if face == "Left (X=0)":
        for iy in range(nely+1):
            for iz in range(nelz+1):
                nodes.append(_node_3d(0,    iy, iz, nely, nelz))
    elif face == "Right (X=W)":
        for iy in range(nely+1):
            for iz in range(nelz+1):
                nodes.append(_node_3d(nelx, iy, iz, nely, nelz))
    elif face == "Bottom (Y=0)":
        for ix in range(nelx+1):
            for iz in range(nelz+1):
                nodes.append(_node_3d(ix, 0,    iz, nely, nelz))
    elif face == "Top (Y=H)":
        for ix in range(nelx+1):
            for iz in range(nelz+1):
                nodes.append(_node_3d(ix, nely, iz, nely, nelz))
    elif face == "Front (Z=0)":
        for ix in range(nelx+1):
            for iy in range(nely+1):
                nodes.append(_node_3d(ix, iy, 0,    nely, nelz))
    elif face == "Back (Z=D)":
        for ix in range(nelx+1):
            for iy in range(nely+1):
                nodes.append(_node_3d(ix, iy, nelz, nely, nelz))
    return nodes


def build_load_cases_3d(load_specs, support_specs, nelx, nely, nelz):
    """Convert UI load/support specs to inputs for simp_core_3d.

    Returns
    -------
    load_cases : list[(ndarray(ndof), float)]
    fixed_dofs : ndarray
    """
    ndof = 3 * (nelx+1) * (nely+1) * (nelz+1)

    F = np.zeros(ndof)
    for spec in load_specs:
        load_type = spec.get("type", "surface")
        face      = spec.get("face", "Right (X=W)")
        fx        = float(spec.get("fx", 0.0))
        fy        = float(spec.get("fy", 0.0))
        fz        = float(spec.get("fz", 0.0))
        nodes     = _face_nodes_3d(face, nelx, nely, nelz)
        n         = len(nodes)
        if n and load_type in ("surface", "line"):
            for nd in nodes:
                F[3*nd]   += fx / n
                F[3*nd+1] += fy / n
                F[3*nd+2] += fz / n
        elif n and load_type == "point":
            nd = nodes[len(nodes)//2]   # midpoint of face
            F[3*nd]   += fx
            F[3*nd+1] += fy
            F[3*nd+2] += fz

    mag = np.linalg.norm(F)
    if mag > 1e-12:
        F /= mag
    else:
        nd_mid = _node_3d(nelx, nely//2, nelz//2, nely, nelz)
        F[3*nd_mid+1] = -1.0

    fixed_set = set()
    for spec in support_specs:
        face     = spec.get("face", "Left (X=0)")
        sup_type = spec.get("type", "fixed")
        nodes    = _face_nodes_3d(face, nelx, nely, nelz)
        for nd in nodes:
            if sup_type == "fixed":
                fixed_set.update([3*nd, 3*nd+1, 3*nd+2])
            elif sup_type == "roller_x":
                fixed_set.add(3*nd)
            elif sup_type == "roller_y":
                fixed_set.add(3*nd+1)

    if not fixed_set:
        for iy in range(nely+1):
            for iz in range(nelz+1):
                nd = _node_3d(0, iy, iz, nely, nelz)
                fixed_set.update([3*nd, 3*nd+1, 3*nd+2])

    return [(F, 1.0)], np.array(sorted(fixed_set))
