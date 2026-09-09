# REVIT 3 SOLVER — Architecture / Implementation Assessment

**Author:** Cabudsan, Daphne Anne  
**Section:** 3H  
**Date:** September 9, 2026

---

## 1. Current Architecture

The existing solver is in `CubeModel/the cube structural model` (701 lines, single-file).

| Aspect | Current State |
|---|---|
| **Method** | Direct Stiffness Method, 3D Frame |
| **Nodes** | 8 nodes, coordinates in meters (hardcoded dict) |
| **Members** | 12 members (4 columns, 4 roof beams, 4 ground/tie beams) |
| **DOF** | 6 per node (3 translations + 3 rotations), 48 total |
| **Supports** | Pinned at nodes 1–4 (UX, UY, UZ restrained) |
| **Releases** | MZ releases at both ends of members 1, 3, 5, 7 |
| **Loads** | Nodal loads at nodes 5–8 (20 kN FX, 50 kN FY downward) |
| **Stiffness** | 12×12 local → transformation → global assembly |
| **Solver** | Direct linear solve (`np.linalg.solve`) |
| **Output** | Console text + structural diagram (matplotlib 3D) + Excel workbook |
| **Unit handling** | None — hardcoded SI (meters, Newtons, Pascals) |
| **Material handling** | None — single set of hardcoded constants (E, G, A, Iy, Iz, J) |
| **Section handling** | None — single set of hardcoded section properties |

### Structure of the single file:

```
SECTION 1:  Input Data (NODES, MEMBERS, E, G, A, Iy, Iz, J, BETA, LOADS, etc.)
SECTION 2:  DOF Assignment (build_dof_system)
SECTION 3:  Local Stiffness Matrix 12×12 (element_stiffness_local)
SECTION 4:  Transformation Matrix with Beta Angle (build_transformation)
SECTION 5:  Member End Releases / Static Condensation (apply_releases)
SECTION 6:  Global Assembly (assemble_global_K)
SECTION 7:  Solver (solve_system)
SECTION 8:  Post-Processing (compute_member_forces)
SECTION 9:  Structural Diagram Plot (plot_structural_diagram)
SECTION 10: Excel Output (write_excel_output)
SECTION 11: Main Execution
```

---

## 2. Existing Reusable Components

| Component | Exists? | Reusable? | Notes |
|---|---|---|---|
| Node representation | Yes — `NODES` dict of `int → np.array([x,y,z])` | Yes | Coordinates in meters |
| Member representation | Yes — `MEMBERS` dict of `int → (i, j)` node pairs | Yes | No property assignment yet |
| Member type detection | Yes — auto-classifies Column / Beam / Ground Beam | Yes | Can extend with member-type metadata |
| Beta angles | Yes — `BETA` dict per member | Yes | Already per-member |
| DOF system | Yes — `build_dof_system()` | Yes | Clean, generic |
| Local stiffness | Yes — `element_stiffness_local(L)` | Partial | Uses global E, G, A, Iy, Iz, J — needs parameterization |
| Transformation | Yes — `build_transformation()` | Yes | Clean, generic |
| Releases | Yes — `apply_releases()` | Yes | Clean, generic |
| Assembly | Yes — `assemble_global_K()` | Yes | Clean, generic |
| Solver | Yes — `solve_system()` | Yes | Clean, generic |
| Post-processing | Yes — `compute_member_forces()` | Yes | Clean, generic |
| Plotting | Yes — full 3D diagram | Yes | No changes needed for units/material/section |
| Excel output | Yes — multi-sheet workbook | Yes | Can add new sheets |

**Key observation:** The solver core (Sections 2–8) is well-separated from input data (Section 1). The main barrier is that material/section properties are **global constants** used directly in `element_stiffness_local()`.

---

## 3. `units.zip` Findings

**The `units.zip` file was NOT found in this repository.**

No unit database, Excel-based or otherwise, is present. No files matching `*unit*` exist anywhere in the project tree.

**Implication:** The unit system must be designed from scratch as a self-contained Python module. Standard SI and Imperial unit definitions will be hardcoded based on well-known structural engineering conventions. When `units.zip` is provided later, the module can be updated to load from it.

---

## 4. Architecture Gap

### 4a. Imperial / Metric Unit Selection

**Currently missing:**
- No concept of "unit system" anywhere in the code
- Coordinates are assumed meters, forces assumed Newtons, stress assumed Pascals
- No conversion utilities
- No way to switch between Imperial and Metric input/output

**What's needed:**
- A `UnitSystem` abstraction (Imperial vs Metric/SI)
- Conversion factors for: length, force, stress, moment, rotation
- Internal solver should use a consistent base system (SI)
- Input/output boundaries handle conversions

### 4b. Material Assignment

**Currently missing:**
- E, G are global constants (`E = 200e9`, `G = 77e9`)
- All members share the same material implicitly
- No material name, no material lookup

**What's needed:**
- A `Material` class/dataclass holding: name, E, G, density, yield_strength
- A `MaterialLibrary` dict-like structure for lookup by name
- Each member stores a reference to its material
- `element_stiffness_local()` must receive E, G from the member, not globals

### 4c. A36 Steel

**Currently missing:**
- No named material, though the hardcoded values (E=200 GPa, G=77 GPa) match A36 Steel

**What's needed:**
- Register ASTM A36 Steel in the material library with standard properties
- All members default to A36 for this activity
- Properties must be retrievable by name

### 4d. Member-Size / Section Assignment

**Currently missing:**
- A, Iy, Iz, J are global constants
- No section name, no section lookup
- All members share one section

**What's needed:**
- A `Section` class/dataclass holding: name, A, Iy, Iz, J
- A `SectionLibrary` dict-like structure for lookup by name
- Each member stores a reference to its section
- `element_stiffness_local()` must receive A, Iy, Iz, J from the member, not globals

### 4e. Future Multi-Material Models

**Currently missing:**
- No per-member property assignment
- Member is just a tuple `(i, j)` — no metadata

**What's needed:**
- A `Member` dataclass or dict that holds: node_i, node_j, material, section, member_type, beta, releases
- Model holds lists of members with full property assignment
- Solver iterates over member objects, pulling properties from each

---

## 5. Proposed Architecture

```
Model
 ├── unit_system: UnitSystem          (Imperial or Metric/SI)
 ├── material_library: MaterialLibrary (dict of name → Material)
 ├── section_library: SectionLibrary   (dict of name → Section)
 └── structural_members: list[Member]
       ├── Column (×4)
       ├── Roof Beam (×4)
       └── Tie Beam (×4)
```

### Data classes:

```python
@dataclass
class UnitSystem:
    name: str                    # "Imperial" or "Metric"
    length: str                  # "ft" or "m"
    force: str                   # "lbf" or "N"
    stress: str                  # "psi" or "Pa"
    moment: str                  # "lbf·ft" or "N·m"
    # Internal base: always SI (m, N, Pa, N·m)
    # Conversion factors stored here

@dataclass
class Material:
    name: str                    # "ASTM A36 Steel"
    E: float                     # Pa (SI internal)
    G: float                     # Pa (SI internal)
    density: float               # kg/m³ (SI internal)
    yield_strength: float        # Pa (SI internal)

@dataclass
class Section:
    name: str                    # "W10x49" or "Custom 0.01m2"
    A: float                     # m² (SI internal)
    Iy: float                    # m⁴ (SI internal)
    Iz: float                    # m⁴ (SI internal)
    J: float                     # m⁴ (SI internal)

@dataclass
class Member:
    id: int
    node_i: int
    node_j: int
    member_type: str             # "Column", "Beam", "Ground Beam"
    material: Material
    section: Section
    beta: float                  # degrees
    released: bool               # MZ release at both ends

@dataclass
class StructuralModel:
    nodes: dict[int, np.ndarray]
    members: dict[int, Member]
    unit_system: UnitSystem
    pinned_nodes: list[int]
    loads: dict[int, np.ndarray]
```

### Key principle:

**All internal solver calculations use SI base units (m, N, Pa).** Unit conversion happens only at input/output boundaries. The `element_stiffness_local()` function signature changes from using globals to receiving E, G, A, Iy, Iz, J as parameters:

```python
# BEFORE (current):
def element_stiffness_local(member_length):
    a = E * A / L        # uses globals

# AFTER (proposed):
def element_stiffness_local(member_length, E, G, A, Iy, Iz, J):
    a = E * A / L        # uses parameters from member's material + section
```

---

## 6. Files to Modify

| File | Purpose | Change |
|---|---|---|
| `CubeModel/the cube structural model` | Main solver | (a) Replace global E, G, A, Iy, Iz, J with dataclass-based lookup. (b) Change `element_stiffness_local()` to accept material/section params. (c) Update main execution to construct `StructuralModel` with A36 Steel and assigned sections. (d) Add new Excel sheets for Materials and Sections. |
| **NEW:** `CubeModel/unit_system.py` | Unit abstraction | UnitSystem class, Imperial/Metric presets, conversion functions |
| **NEW:** `CubeModel/materials.py` | Material library | Material dataclass, MaterialLibrary, A36 Steel definition |
| **NEW:** `CubeModel/sections.py` | Section library | Section dataclass, SectionLibrary, default sections |

**Only 1 existing file is modified.** 3 new small modules are added. No unrelated code is touched.

---

## 7. Implementation Sequence

| Step | Action | Files |
|---|---|---|
| 1 | Create `unit_system.py` with `UnitSystem` class + Imperial/Metric presets + conversion helpers | NEW: `unit_system.py` |
| 2 | Create `materials.py` with `Material` dataclass + `MaterialLibrary` + A36 Steel | NEW: `materials.py` |
| 3 | Create `sections.py` with `Section` dataclass + `SectionLibrary` + default sections | NEW: `sections.py` |
| 4 | Modify `element_stiffness_local()` to accept E, G, A, Iy, Iz, J as parameters instead of globals | MODIFY: `the cube structural model` |
| 5 | Update main execution to construct `StructuralModel` with unit system, material library, section library, and per-member property assignment | MODIFY: `the cube structural model` |
| 6 | Add Materials and Sections sheets to Excel output | MODIFY: `the cube structural model` |
| 7 | Add unit-aware print labels (e.g., displacements in ft or m depending on system) | MODIFY: `the cube structural model` |
| 8 | Write tests | NEW: `test_revit3_solver.py` |
| 9 | Run solver, verify identical results to current Rev. 2 | Verification |

---

## 8. Test Plan

### Unit tests (new file: `Lab03/test_revit3_solver.py`):

| Test | Description |
|---|---|
| `test_metric_unit_selection` | Selecting Metric sets correct length="m", force="N" |
| `test_imperial_unit_selection` | Selecting Imperial sets correct length="ft", force="lbf" |
| `test_unit_conversion_length` | 1 ft = 0.3048 m exactly |
| `test_unit_conversion_force` | 1 lbf = 4.44822 N exactly |
| `test_invalid_unit_system` | Unknown unit system raises error |
| `test_a36_material_lookup` | `library["ASTM A36 Steel"]` returns correct E, G |
| `test_material_assignment` | A member can have A36 assigned and properties retrieved |
| `test_unknown_material_rejection` | Unknown material name raises error |
| `test_section_lookup` | Known section name returns correct A, Iy, Iz, J |
| `test_section_assignment` | A member can have a section assigned |
| `test_invalid_section_rejection` | Unknown section name raises error |
| `test_model_construction` | Full cube model constructs with 8 nodes, 12 members, A36, selected unit system |
| `test_solver_equilibrium` | Sum of reactions + applied loads = 0 (equilibrium check) |
| `test_backward_compatibility` | Solver produces same displacements/reactions as current Rev. 2 with same inputs |

---

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| `units.zip` not available | Medium | Design unit module to be self-contained; can load from zip later |
| Changing `element_stiffness_local` signature | Low | Only caller is in main execution — single call site |
| Unit conversion rounding errors | Low | Use exact conversion factors; internal always SI |
| Excel output format changes | Low | Add new sheets, keep existing sheets unchanged |
| Performance from dataclass lookups per member | None | 12 members — negligible overhead |
| Backward compatibility breakage | Low | Run identical input, compare output numerically |

---

## 10. Approval Gate

**STOP. Architecture assessment complete.**

Awaiting approval before proceeding to implementation.
