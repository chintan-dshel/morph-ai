import numpy as np

MATERIALS = {
    "PLA (Bioplastic)": {
        "E_gpa": 3.5, "rho_gcc": 1.24, "nu": 0.36, "yield_strength_mpa": 50,
        "color": "#4CAF50", "printable": True, "print_temp": "200-220C", "bed_temp": "60C",
        "cost_per_kg": 20.0, "cost_level": 1, "notes": "Best for beginners. Stiff but brittle."
    },
    "PETG (Engineering Plastic)": {
        "E_gpa": 2.1, "rho_gcc": 1.27, "nu": 0.39, "yield_strength_mpa": 53,
        "color": "#2196F3", "printable": True, "print_temp": "230-250C", "bed_temp": "80C",
        "cost_per_kg": 25.0, "cost_level": 2, "notes": "Tougher than PLA. Good chemical resistance."
    },
    "ABS (Acrylonitrile Butadiene Styrene)": {
        "E_gpa": 2.3, "rho_gcc": 1.05, "nu": 0.35, "yield_strength_mpa": 44,
        "color": "#FF9800", "printable": True, "print_temp": "230-250C", "bed_temp": "100C",
        "cost_per_kg": 22.0, "cost_level": 2, "notes": "Needs enclosure. Heat resistant to ~100C."
    },
    "Nylon PA12": {
        "E_gpa": 1.6, "rho_gcc": 1.01, "nu": 0.39, "yield_strength_mpa": 50,
        "color": "#9C27B0", "printable": True, "print_temp": "240-260C", "bed_temp": "70C",
        "cost_per_kg": 40.0, "cost_level": 3, "notes": "Tough and fatigue resistant. Absorbs moisture."
    },
    "TPU (Flexible)": {
        "E_gpa": 0.05, "rho_gcc": 1.21, "nu": 0.47, "yield_strength_mpa": 29,
        "color": "#F44336", "printable": True, "print_temp": "220-240C", "bed_temp": "30-60C",
        "cost_per_kg": 30.0, "cost_level": 2, "notes": "Rubber-like. Shock absorption."
    },
    "Carbon Fiber PETG (Composite)": {
        "E_gpa": 9.5, "rho_gcc": 1.30, "nu": 0.30, "yield_strength_mpa": 110,
        "color": "#78909C", "printable": True, "print_temp": "240-260C", "bed_temp": "80C",
        "cost_per_kg": 80.0, "cost_level": 4, "notes": "High stiffness-to-weight. Hardened nozzle required."
    },
    "Titanium Ti-6Al-4V (Reference)": {
        "E_gpa": 114.0, "rho_gcc": 4.43, "nu": 0.34, "yield_strength_mpa": 880,
        "color": "#90A4AE", "printable": False, "print_temp": "N/A", "bed_temp": "N/A",
        "cost_per_kg": 500.0, "cost_level": 5, "notes": "Aerospace benchmark. Not FDM printable."
    },
    "Aluminum 6061 (Reference)": {
        "E_gpa": 68.9, "rho_gcc": 2.70, "nu": 0.33, "yield_strength_mpa": 276,
        "color": "#CFD8DC", "printable": False, "print_temp": "N/A", "bed_temp": "N/A",
        "cost_per_kg": 200.0, "cost_level": 5, "notes": "Machined metal benchmark. Not FDM printable."
    },
}

FACES = [
    "Left (X=0)", "Right (X=W)", "Bottom (Y=0)",
    "Top (Y=H)", "Front (Z=0)", "Back (Z=D)"
]

INFILL_PATTERNS = {
    "Gyroid": {
        "desc": "Isotropic TPMS. Best all-round. Excellent fatigue resistance.",
        "color": "#00BCD4",
        "fn": lambda x, y, z, p: (
            np.sin(2*np.pi*x/p)*np.cos(2*np.pi*y/p) +
            np.sin(2*np.pi*y/p)*np.cos(2*np.pi*z/p) +
            np.sin(2*np.pi*z/p)*np.cos(2*np.pi*x/p)
        )
    },
    "Schwartz-P": {
        "desc": "Stiffer in compression. Fewer triangles, faster slicing.",
        "color": "#FF5722",
        "fn": lambda x, y, z, p: (
            np.cos(2*np.pi*x/p) + np.cos(2*np.pi*y/p) + np.cos(2*np.pi*z/p)
        )
    },
    "Honeycomb": {
        "desc": "2D hex extruded in Z. Lightest weight option (up to 30% mass saving).",
        "color": "#FFC107",
        "fn": lambda x, y, z, p: (
            np.cos(2*np.pi*x/p) +
            np.cos(np.pi*x/p + np.sqrt(3)*np.pi*y/p) +
            np.cos(np.pi*x/p - np.sqrt(3)*np.pi*y/p)
        )
    },
    "Diamond": {
        "desc": "Highest surface area. Best for heat dissipation.",
        "color": "#E040FB",
        "fn": lambda x, y, z, p: (
            np.sin(2*np.pi*x/p)*np.sin(2*np.pi*y/p)*np.sin(2*np.pi*z/p) +
            np.sin(2*np.pi*x/p)*np.cos(2*np.pi*y/p)*np.cos(2*np.pi*z/p) +
            np.cos(2*np.pi*x/p)*np.sin(2*np.pi*y/p)*np.cos(2*np.pi*z/p) +
            np.cos(2*np.pi*x/p)*np.cos(2*np.pi*y/p)*np.sin(2*np.pi*z/p)
        )
    },
}


def compute_mat_scores(mat_names, box_w, box_h, box_d, vf, force_n, sf, norm_comp):
    scores = {}
    load_area_m2 = (box_h * box_d) * 1e-6
    stress_mpa = (force_n / load_area_m2) / 1e6
    for name in mat_names:
        m = MATERIALS[name]
        mass_g = (box_w/10) * (box_h/10) * (box_d/10) * vf * m["rho_gcc"]
        comp = norm_comp / m["E_gpa"]
        ok = stress_mpa <= m["yield_strength_mpa"] / sf
        ashby = (m["E_gpa"]**(1/3)) / m["rho_gcc"]
        scores[name] = {
            "mass_g": round(mass_g, 1),
            "comp": round(comp, 3),
            "stress_ok": ok,
            "ashby": round(ashby, 4),
            "allowable": round(m["yield_strength_mpa"] / sf, 1),
            "stress_applied": round(stress_mpa, 3),
        }
    return scores


def rank_materials(scores):
    names = list(scores.keys())
    if not names:
        return {}
    return {
        "stiffest":   min(names, key=lambda n: scores[n]["comp"]),
        "lightest":   min(names, key=lambda n: scores[n]["mass_g"]),
        "best_ashby": max(names, key=lambda n: scores[n]["ashby"]),
        "cheapest":   min(names, key=lambda n: MATERIALS[n]["cost_level"]),
    }
