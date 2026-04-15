import hashlib
from datetime import datetime


def estimate_print(mass_g, rho_gcc, layer_height=0.2, speed_mms=50,
                   nozzle_dia=0.4, cost_per_kg=20.0):
    vol_cm3 = mass_g / rho_gcc
    vol_mm3 = vol_cm3 * 1000
    cross_mm2 = layer_height * nozzle_dia
    length_mm = vol_mm3 / cross_mm2
    time_s = (length_mm / speed_mms) * 1.3
    return {
        "volume_cm3":        round(vol_cm3, 2),
        "filament_m":        round(length_mm / 1000, 1),
        "time_min":          round(time_s / 60, 0),
        "time_h":            round(time_s / 3600, 1),
        "material_cost_usd": round((mass_g / 1000) * cost_per_kg, 2),
    }


def make_history_record(name, export_mat, box, vf, load_scenario,
                        compliance, iters, n_faces, mass_g,
                        infill_pattern=None, infill_vf=None):
    uid = hashlib.md5(
        f"{name}{box}{export_mat}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:8]
    return {
        "id":        uid,
        "name":      name,
        "material":  export_mat,
        "box":       box,
        "vf":        vf,
        "load":      load_scenario,
        "compliance": round(compliance, 3),
        "iters":     iters,
        "faces":     n_faces,
        "mass_g":    round(mass_g, 1),
        "infill":    infill_pattern or "—",
        "infill_vf": round(infill_vf * 100, 0) if infill_vf else 0,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
