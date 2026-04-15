# MorphAI — User Manual

**Version 6 · SIMP Topology Optimizer · Multi-Load · Material Comparison · STL Export**

---

## Table of Contents

1. [What this tool does](#1-what-this-tool-does)
2. [Installation](#2-installation)
3. [Quick start — your first optimized part in 5 minutes](#3-quick-start)
4. [The sidebar — every control explained](#4-the-sidebar)
5. [Single Material mode — step by step](#5-single-material-mode)
6. [Material Comparison mode — step by step](#6-material-comparison-mode)
7. [Reading the results — all 5 tabs](#7-reading-the-results)
8. [Exporting and printing your STL](#8-exporting-and-printing)
9. [Multi-load scenarios — when to use them](#9-multi-load-scenarios)
10. [Tuning the optimizer — what to change when results look wrong](#10-tuning-the-optimizer)
11. [Material guide](#11-material-guide)
12. [Common problems and fixes](#12-common-problems-and-fixes)
13. [How the physics works](#13-how-the-physics-works)
14. [Test checklist — validate every feature](#14-test-checklist)

---

## 1. What this tool does

The MorphAI takes a box-shaped design space, a material, and a description of how your part will be loaded, then computes the mathematically optimal arrangement of material inside that box. The result is a 3D mesh you can download as an STL and send straight to your printer.

It uses SIMP (Solid Isotropic Material with Penalization), the same algorithm used in professional tools like Altair OptiStruct and ANSYS. Every control in the sidebar has a direct physical meaning, grounded in the material science properties (Young's modulus, density, Poisson's ratio) of your chosen material.

**What you get:**
- A 3D topology-optimized mesh, rendered interactively in the browser
- A downloadable `.stl` file ready for PrusaSlicer, Bambu Studio, or Cura
- A compliance convergence curve showing how the optimizer improved the design
- Material comparison rankings (stiffest, lightest, best structural efficiency)
- Post-run diagnostic warnings if the result needs more refinement

---

## 2. Installation

Open your terminal (VS Code → Terminal → New Terminal) and run:

```bash
pip install streamlit numpy pandas plotly scipy scikit-image
python -m streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`. If it does not, paste that address manually.

**If `streamlit` is not recognised as a command**, always use `python -m streamlit run app.py` instead of `streamlit run app.py`. This bypasses PATH issues on Windows.

---

## 3. Quick start

This gets you a real optimized cantilever bracket in under 5 minutes with default settings.

1. Launch the app. You will see a 3D bounding box with a blue face (fixed wall) on the left and a red face (load) on the right.
2. Leave all sidebar settings at their defaults.
3. Click **🚀 Run Optimizer** at the bottom of the sidebar.
4. Wait 10–20 seconds. A success message appears when done.
5. Look at the **🧊 3D Mesh** tab. You should see a truss-like structure — dense flanges at top and bottom, diagonal members connecting them.
6. Go to the **📦 Export STL** tab and click the download button.
7. Open the `.stl` in your slicer and print.

That is the complete workflow. Everything below explains what each control does and how to adapt it to your actual part.

---

## 4. The Sidebar

The sidebar is divided into sections from top to bottom. Work through them in order for each new design.

### Mode

**Single Material** — design one part with one material. This is the standard workflow.

**Material Comparison** — run the optimizer once and see how every selected material performs on the same geometry. Produces a ranking table, radar chart, and bar charts. Use this when you are deciding which material to buy before printing.

### Material (Single Material mode)

Click the dropdown to select your material. The line below the dropdown shows print temperature and bed temperature — these are your slicer settings.

The **printability badge** (green = printable, yellow = reference only) tells you whether a desktop FDM printer can handle this material. Titanium and Aluminum are reference materials only — they appear in the comparison charts for engineering context but cannot be printed on a desktop machine.

### Select materials to compare (Comparison mode)

Check or uncheck materials. All checked materials will be evaluated against the same geometry. You can include the reference metals for context.

### Design Space (mm)

These three sliders define the bounding box — the volume of space the optimizer is allowed to fill. Think of it as the box your part must fit inside.

- **Width (X)**: the horizontal dimension, from the fixed face to the load face
- **Height (Y)**: the vertical dimension
- **Depth (Z)**: the out-of-plane thickness

The optimizer works on a 2D cross-section and extrudes it along Z, so Depth affects mass and STL proportions but not the topology pattern.

### Boundary Conditions

This is where you tell the optimizer how your part is attached and loaded.

**Fixed face** — the face that is bolted, welded, or glued to something rigid. This face cannot move. In the 3D canvas it appears blue. For a wall bracket, this would be the back face touching the wall. For a shelf support, this is typically the left face.

**Load face** — where the external force is applied. Appears red. For a shelf bracket, this is the top face where weight sits, or the right face where a load hangs.

**Force direction** — which way the load pushes. `-Y` means downward (gravity). `-X` means pushing from right to left. If you are designing a bracket that holds weight, use `-Y`.

**Applied Force (N)** — the total force in Newtons. 500 N ≈ the weight of a 50 kg person. 100 N ≈ 10 kg. 4900 N ≈ 500 kg. This does not change the topology shape but feeds into the stress safety check.

**Safety Factor** — how much margin above the yield strength you want. 2.0 means the part is designed to handle twice the applied load before yielding. Use 2.0 for general parts, 3.0–4.0 for safety-critical applications, 1.5 for lightweight non-critical parts.

### Load Scenario

Defines how many load cases the optimizer solves simultaneously. See Section 9 for full details.

### Optimizer Settings

**Volume Fraction** — what percentage of the bounding box is filled with material. 0.4 means 40% solid. Lower values produce lighter, more truss-like structures. Higher values produce more solid, heavier parts with fewer internal voids.

- Start with 0.4 for most parts
- Use 0.2–0.3 for ultra-lightweight structures where weight is critical
- Use 0.6–0.8 for structural parts that need high rigidity with minimal optimisation

**SIMP Penalty (p)** — controls how aggressively the optimizer pushes densities toward 0 or 1. Higher penalty = crisper boundaries between solid and void. Lower penalty = smoother, more gradient result.

- Default 3.0 is correct for almost all cases
- Raise to 4.0–4.5 if the result looks grey and fuzzy (see Section 12)
- Lower to 2.0 if you want a smoother, organic topology (functional but not sharp-edged)

**Filter Radius** — sets the minimum feature size, measured in elements. Prevents checkerboard artefacts (alternating solid/void at single-element scale, which are not printable).

- Default 1.5 works well for most meshes
- Raise to 2.0–2.5 for smoother, thicker members
- Lower to 1.0 for finer detail (only useful on large meshes 30×20+)

**Max Iterations** — the optimizer stops when the density field stabilises (change < 0.01) or when it hits this limit. 60 is usually enough. If the convergence chart is still dropping steeply at the last iteration, raise to 100.

**Elements X / Elements Y** — the resolution of the finite element mesh. More elements = finer topology detail but longer run time.

| Mesh | Run time | Use for |
|---|---|---|
| 10×6 | 2–3s | Quick preview |
| 20×12 | 8–15s | Standard design |
| 30×18 | 25–40s | High detail |
| 40×25 | 60–90s | Publication quality |
| 50×25 | 2–3 min | Maximum detail |

### Export Settings

**Iso-surface level** — the density threshold at which the marching cubes algorithm draws the surface. 0.45 means: everything with density above 0.45 becomes solid in the STL. Lower this if you get a "no surface found" warning. Raise it to remove marginal material from the boundary.

**Depth slices** — how many layers the 2D cross-section is extruded into along the Z axis. 8 is a good default. Raising to 16–20 makes the end caps smoother but increases STL file size.

---

## 5. Single Material mode — step by step

### Step 1: Define your real-world problem

Before touching any slider, answer these three questions:
1. What physical object is this? (bracket, clamp, arm, hinge, housing, etc.)
2. Where is it attached? (that face becomes the Fixed face)
3. What direction does the load push? (that sets the Force direction)

**Example:** A camera mount arm. It clamps to a tripod (left face, fixed) and holds a camera (right face, load, force direction -Y for gravity).

### Step 2: Set the box dimensions

Measure or estimate the space your part occupies. Enter those dimensions in mm. The box does not need to be exact — the optimizer works within whatever space you give it.

### Step 3: Set boundary conditions

Select the fixed face and load face that match your real attachment points. Set the force direction to match gravity or the dominant load.

### Step 4: Choose material

Pick the material you plan to print with. If you are unsure, start with PLA. The optimizer will use its Poisson's ratio to compute the correct stiffness distribution.

### Step 5: Set volume fraction

Start at 0.4. After seeing the result you can re-run at 0.3 for lighter or 0.5 for heavier.

### Step 6: Run

Click 🚀 Run Optimizer. Read the post-run diagnostic messages (the blue/yellow info boxes). They tell you if the result is sharp enough to print well.

### Step 7: Evaluate the result

Go to the 3D Mesh tab. Look for:
- Continuous members connecting fixed face to load face
- Clear void regions (not grey smear)
- No floating isolated blobs

If the topology looks wrong, see Section 12.

### Step 8: Export

Go to the Export STL tab. Download the file. Open in your slicer.

---

## 6. Material Comparison mode — step by step

Use this when you want the answer to "which material should I use for this part?"

### Step 1: Switch mode

At the top of the sidebar, select **Material Comparison**.

### Step 2: Check the materials you want to compare

Check all FDM-printable materials. You can include the reference metals for context — they will appear in the comparison charts but are flagged as non-printable.

### Step 3: Set geometry and boundary conditions

Same as single material mode. These settings describe the part, not the material.

### Step 4: Run

The optimizer runs once using the average Poisson's ratio of your selected materials. This is correct because topology (the shape) is primarily driven by geometry and boundary conditions — all materials with similar Poisson's ratios produce nearly identical topologies. The differences in mass, stiffness, and safety are then computed per-material analytically.

### Step 5: Read the comparison tab

Go to **⚖️ Material Compare**. You will see:

- **Four winner badges** — stiffest, lightest, best Ashby index, cheapest
- **Comparison table** — full numbers for every material
- **Radar chart** — visual fingerprint showing each material's profile
- **Bar charts** — mass and log compliance side by side

### Step 6: Choose your material and export

The STL is exported using the best-Ashby-index winner by default. The geometry is identical for all materials — only rename the file if you decide to print a different one.

---

## 7. Reading the results — all 5 tabs

### 🧊 3D Mesh tab

The main view. Drag to rotate, scroll to zoom, right-click drag to pan.

- **Coloured solid** — your optimized part
- **Blue face** — fixed (clamped) face
- **Red face** — load application face
- **Red arrow** — force direction

What to look for: you should see load paths — continuous struts of material running from the fixed face to the load point. If you see an I-beam shape, that is correct for a cantilever under vertical load. If you see diagonal crossing members, that is correct for combined loading.

### ⚖️ Material Compare tab (comparison mode only)

The four ranking categories are independent. A material can win in one category and lose in another:

- **Stiffest** — lowest physical compliance (E normalized). Best if deflection is your concern.
- **Lightest** — lowest mass. Best if weight budget is tight.
- **Best Ashby E^(1/3)/rho** — the beam bending efficiency index from materials science. This is the correct metric for most structural parts: it accounts for both stiffness and mass. CF-PETG almost always wins here.
- **Cheapest** — lowest filament cost tier.

The **log compliance bar chart** uses a logarithmic scale because TPU's compliance is orders of magnitude higher than the structural plastics. Log scale keeps all materials visible.

### 📊 Density Map tab

The raw 2D output of the FEA solver. Bright colours = solid (density near 1), near-black = void (density near 0).

Look at the **density histogram** at the bottom. A well-converged SIMP result shows a U-shaped distribution: many elements near 0, many elements near 1, few in the middle. If the histogram shows a single large peak in the middle (grey zone), the result is under-converged. Raise the SIMP penalty or run more iterations.

The three metrics below the histogram:
- **Solid >0.5** — elements that will appear in the STL
- **Grey 0.2–0.5** — ambiguous elements (increase penalty to clear these)
- **Void <0.2** — empty space

### 📈 Convergence tab

The compliance value (y-axis) should decrease monotonically and flatten out by the end. If it is still dropping steeply at the last iteration, raise Max Iterations and re-run. If it is flat well before the last iteration, you used more iterations than needed (harmless but slower).

The dashed grey line (if present) shows the single-load reference run for comparison with multi-load results.

### 📦 Export STL tab

The download button and a preview of the mesh that was exported. Verify the shape looks as expected before printing.

---

## 8. Exporting and printing your STL

### Download

Click the green download button in the Export STL tab. The file is named automatically:
`topo_[material]_[width]x[height]x[depth]mm_[loadscenario].stl`

### Open in slicer

Drag the `.stl` file into PrusaSlicer, Bambu Studio, or Cura. The file is already in millimetres — do not rescale.

### Recommended slicer settings

| Setting | Value | Reason |
|---|---|---|
| Layer height | 0.2 mm | Standard quality |
| Walls / perimeters | 3–4 | Structural integrity on thin members |
| Infill | 15% gyroid | Topology already placed material correctly |
| Infill pattern | Gyroid | Best isotropy for optimized parts |
| Supports | Only if overhangs > 45° | |
| First layer | Slow, 0.3 mm | Adhesion on the thin base |

**The key insight:** the topology optimizer already decided where material is needed. The infill inside the solid regions is secondary — use 15% to add minimum density without wasting filament.

### Orient correctly

The blue face in the app (fixed face) should face your mounting surface. For a wall bracket, the back face goes against the wall. For a shelf support, the bottom face goes on the shelf. Orienting the part so load paths are vertical improves layer adhesion on the critical members.

### Material-specific tips

**PLA** — works out of the box. No enclosure needed. Brittle under impact but excellent stiffness-to-weight. Good for indoor structural parts.

**PETG** — slightly more flexible than PLA. Resists chemicals and moisture. Good for parts exposed to humidity.

**ABS** — needs an enclosure to prevent warping. Slightly lighter than PLA. Good for high-temperature environments (up to ~100°C).

**Nylon PA12** — absorbs moisture from air, store sealed. Tough and fatigue-resistant. Best for snap-fits and living hinges.

**CF-PETG** — use a hardened steel nozzle (0.4 mm minimum). Very stiff but abrasive. Best structural choice for performance parts.

**TPU** — very flexible. The topology optimizer will produce a different structure (more gradient, less binary) because high Poisson's ratio materials distribute stress differently. Useful for gaskets, grips, and shock mounts.

---

## 9. Multi-load scenarios — when to use them

By default the optimizer solves for a single downward load. In reality most parts see multiple forces. The load scenario selector lets you define two simultaneous load cases.

| Scenario | What it optimises for | Typical use |
|---|---|---|
| Downward only | Single vertical load at right-centre | Basic shelf bracket, simple cantilever |
| Lateral only | Single horizontal load | Lateral brace, horizontal bracket |
| Down + Lateral | Gravity + sideways force | Vehicle mount, equipment rack |
| Down + Upward (top) | Alternating vertical loads | Spring-loaded arm, bounce-prone mount |
| Down + Downward (bot) | Both ends loaded downward | Bridge-like part, symmetric loading |
| Symmetric (top+bot) | Equal loads at top and bottom right | Symmetric bending, two-point support |
| Torsion (top+bot opp) | Opposing forces at top and bottom | Torque arm, twist-loaded part |

**Primary and secondary weights** — when you pick a two-case scenario, sliders appear for how much each load matters. If your part sees mostly gravity but occasionally gets nudged sideways, set primary 0.8 and secondary 0.2. Equal importance is 0.5 / 0.5.

**What changes in the topology?** Single-load produces the most efficient structure for that one load. Multi-load produces diagonal bracing, more symmetric members, and thicker junctions. It is slightly heavier but far more robust.

---

## 10. Tuning the optimizer — what to change when results look wrong

### Problem: The 3D mesh looks like a fuzzy blob, not a clean truss

**Cause:** Under-convergence — the SIMP penalty is too low or not enough iterations.

**Fix:** Raise SIMP Penalty to 4.0. If still fuzzy, raise Max Iterations to 100.

### Problem: Grey zone warning appears (bimodality < 65%)

**Cause:** Many elements have intermediate density (0.2–0.5) rather than binary 0/1.

**Fix:** Raise SIMP Penalty to 4.0–4.5. For low volume fractions (< 0.25), also reduce Filter Radius to 1.0.

### Problem: "No STL surface found" warning

**Cause:** The iso-surface threshold is above the maximum density in the field.

**Fix:** Lower the Iso-surface level slider to 0.35 or 0.30. Check the density map — if the max density shown is below 0.45, lower the threshold to match.

### Problem: The shape doesn't make engineering sense (material in wrong places)

**Cause:** Fixed face or load face is set incorrectly, or force direction is wrong.

**Fix:** Check: Is the blue face the face that is attached to something rigid? Is the red face where the external load is applied? Is the arrow pointing in the direction the force pushes?

### Problem: Very fast convergence (under 10 iterations) but poor shape

**Cause:** Volume fraction is too high (near 0.8–0.9). The optimizer has little room to redistribute material.

**Fix:** Lower volume fraction to 0.4–0.5 for a more meaningful topology.

### Problem: STL file opens in slicer but has holes or non-manifold geometry

**Cause:** Marching cubes occasionally produces thin surface artefacts at very low density gradients.

**Fix:** Lower the Iso-surface level slightly (try -0.05). Or raise the SIMP Penalty and re-run to get sharper 0/1 boundaries, which always produce cleaner marching cubes output.

### Problem: Run takes more than 5 minutes

**Cause:** Mesh resolution is too high for the available hardware.

**Fix:** Reduce Elements X to 20 and Elements Y to 12 for a fast preview. Raise mesh only after confirming the design concept is correct.

---

## 11. Material guide

| Material | E (GPa) | Density (g/cm³) | Yield (MPa) | Poisson ν | Printable |
|---|---|---|---|---|---|
| PLA | 3.5 | 1.24 | 50 | 0.36 | ✅ |
| PETG | 2.1 | 1.27 | 53 | 0.39 | ✅ |
| ABS | 2.3 | 1.05 | 44 | 0.35 | ✅ |
| Nylon PA12 | 1.6 | 1.01 | 50 | 0.39 | ✅ |
| TPU | 0.05 | 1.21 | 29 | 0.47 | ✅ |
| CF-PETG | 9.5 | 1.30 | 110 | 0.30 | ✅ |
| Titanium Ti-6Al-4V | 114 | 4.43 | 880 | 0.34 | ❌ |
| Aluminum 6061 | 68.9 | 2.70 | 276 | 0.33 | ❌ |

**Young's Modulus (E)** — how stiff the material is. Higher = less deflection under load. CF-PETG is ~4× stiffer than PLA.

**Density (ρ)** — how heavy a solid block would be. Nylon is the lightest printable option.

**Yield Strength** — the stress at which permanent deformation begins. The Safety Factor divides this to set the allowable working stress.

**Poisson's Ratio (ν)** — how much the material squishes sideways when compressed axially. TPU is near-incompressible (ν ≈ 0.5) which produces different load paths. This is the key material property that actually changes the topology shape.

**Ashby beam bending index = E^(1/3) / ρ** — the correct way to compare materials for structural efficiency in bending. CF-PETG wins because its much higher E outweighs its slightly higher density.

---

## 12. Common problems and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| App shows code text in sidebar | Streamlit version rendering `#` comments | Already fixed in v6 |
| Optimizer crashes on run | Outdated scipy | `pip install scipy --upgrade` |
| STL is hollow / all walls | Iso threshold too high | Lower Iso-surface level to 0.35 |
| Stress shows "OVER" in red metric | Force too high for selected material | Increase safety factor or choose stiffer material |
| Shape is symmetric and boring (straight block) | Volume fraction too high | Lower to 0.3–0.4 |
| Multi-load convergence chart shows negative values | Torsion scenario with negative weight | Expected and correct — negative weight reverses force direction |
| Material comparison shows all same compliance | Mean nu used for shared topology | Correct behaviour — topology is geometry-driven, compliance scaled per material |
| Download button not visible | stl_bytes is None | Lower iso threshold and re-run |

---

## 13. How the physics works

The optimizer divides the bounding box into a grid of rectangular elements (nelx × nely). Each element gets a density value x between 0 (void) and 1 (fully solid).

The stiffness of each element follows the SIMP rule:

**E(x) = E_min + x^p × (E_0 - E_min)**

where p is the penalty parameter. With p = 3, intermediate densities (grey elements) are heavily penalised — their stiffness is much lower than their volume fraction would suggest, so the optimizer is pushed to choose either fully solid or fully void.

At each iteration the solver:
1. Assembles the global stiffness matrix from element densities
2. Solves the system K·u = f (the finite element equations) to find displacements u
3. Computes compliance C = f·u (total strain energy — inverse of stiffness)
4. Computes sensitivities dC/dx (how much each element contributes to compliance)
5. Updates densities using the Optimality Criteria method to move material toward high-sensitivity locations
6. Applies the density filter to enforce minimum feature size

This repeats until density changes are below 0.01, meaning the layout has stabilised.

The compliance the optimizer minimises is normalised (E = 1). The physical compliance for a real material is: physical compliance = normalised compliance / E_gpa. This is why CF-PETG produces a much stiffer part — the same topology, divided by a much larger E.

---

## 14. Test checklist — validate every feature

Use this checklist to verify the app is working correctly after installation or after any update.

### Installation test

- [ ] `python -m streamlit run app.py` launches without error
- [ ] Browser opens at `http://localhost:8501`
- [ ] Sidebar renders without visible code or `---` lines
- [ ] 3D canvas shows bounding box with blue and red faces

### Single material — basic run

- [ ] Set material to PLA, all other settings default
- [ ] Click Run Optimizer
- [ ] Success message appears within 30 seconds
- [ ] 3D Mesh tab shows a truss structure (not a solid block, not empty)
- [ ] Density Map shows bimodal histogram (peaks near 0 and near 1)
- [ ] Convergence chart shows declining curve that flattens
- [ ] Export STL tab shows a download button
- [ ] Downloaded `.stl` file opens in PrusaSlicer or similar

### Material comparison

- [ ] Switch mode to Material Comparison
- [ ] Check PLA, PETG, ABS, Nylon, CF-PETG
- [ ] Run Optimizer
- [ ] Material Compare tab shows 4 winner badges
- [ ] Comparison table has one row per selected material
- [ ] Radar chart overlays all materials
- [ ] CF-PETG shows lowest compliance; Nylon shows lowest mass

### Multi-load

- [ ] Set Load Scenario to "Down + Lateral"
- [ ] Run Optimizer
- [ ] Convergence tab shows multi-load compliance (solid) and single-load reference (dashed)
- [ ] Multi-Load tab shows the comparison layout

### Edge cases to verify

- [ ] Set volume fraction to 0.2 → bimodality warning appears (< 65% expected)
- [ ] Set volume fraction to 0.8 → fast convergence, dense result
- [ ] Set iso threshold to 0.90 → "No STL surface" warning appears
- [ ] Lower iso threshold to 0.30 → STL appears
- [ ] Set fixed face == load face → yellow warning appears in sidebar
- [ ] Click Clear Results → results disappear, canvas returns

### Performance

- [ ] 20×12 mesh completes in under 30 seconds
- [ ] 30×18 mesh completes in under 60 seconds
- [ ] 40×25 mesh completes in under 3 minutes

---

*MorphAI v6 · Built on SIMP (Sigmund 2001) · Marching cubes via scikit-image · Sparse FEA via scipy*
