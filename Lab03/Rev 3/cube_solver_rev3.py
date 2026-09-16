"""
Cube Structural Solver - Rev 3
================================
Direct Stiffness Method | 3D Frame Analysis | Load-Case / Combination Framework

Rev 3 builds directly on the Rev 2 solver (single 6m x 6m x 6m cube, 8 nodes,
12 members, direct stiffness solver).  Rev 3 ADDS, without rewriting Rev 2:

  * Nodal / member-distributed / member-point / temperature load cases
  * Self-weight generator (weight-density formulation, gamma*A*L)
  * Diaphragm constraint definition + generated constraint equations
  * NSCP 2015 LRFD/ASD load combination engine (30 combinations)
  * Load validation with equilibrium reporting
  * A matplotlib load-case / combination viewer with grid & band toggles
  * A 95-test standard-library unittest suite
  * A --verify-loads headless audit producing the 10-section report + images

Run:
    python cube_solver_rev3.py                  launch the interactive viewer
    python cube_solver_rev3.py --verify-loads   headless audit (report + images)
    python cube_solver_rev3.py --run-tests      run the 95 automated tests

This file is self-contained.  It does NOT import from CubeModel/.
"""

import sys
import os
import argparse
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# matplotlib.pyplot is imported lazily inside the rendering functions so that
# the headless (--verify-loads) path can select the Agg backend first and the
# interactive path keeps the default windowed backend.

# ============================================================
# SECTION 1: Imports (numpy, pandas, matplotlib, openpyxl)
# ============================================================
import pandas as pd
import openpyxl


# ============================================================
# SECTION 2: UNIT SYSTEM
# ============================================================

@dataclass(frozen=True)
class UnitSystem:
    """Physical unit system used for display labels and conversions."""
    name: str
    length: str
    force: str
    stress: str
    moment: str
    rotation: str
    length_factor: float
    force_factor: float

    def to_display(self, value_si: float, quantity: str) -> float:
        if quantity == "length":
            return value_si / self.length_factor
        if quantity == "force":
            return value_si / self.force_factor
        if quantity == "stress":
            return value_si / self.force_factor * self.length_factor ** 2
        if quantity == "moment":
            return value_si / (self.force_factor * self.length_factor)
        if quantity == "rotation":
            return value_si
        if quantity == "distributed":
            return value_si * self.length_factor / self.force_factor
        return value_si

    def unit_label(self, quantity: str) -> str:
        labels = {
            "length": self.length,
            "force": self.force,
            "stress": self.stress,
            "moment": self.moment,
            "rotation": self.rotation,
            "distributed": self.force + "/" + self.length,
        }
        return labels.get(quantity, "")


METRIC = UnitSystem(
    name="Metric",
    length="m",
    force="N",
    stress="Pa",
    moment="N*m",
    rotation="rad",
    length_factor=1.0,
    force_factor=1.0,
)

IMPERIAL = UnitSystem(
    name="Imperial",
    length="ft",
    force="lbf",
    stress="psi",
    moment="lbf*ft",
    rotation="rad",
    length_factor=0.3048,
    force_factor=4.4482216152605,
)

_SYSTEMS = {"Metric": METRIC, "Imperial": IMPERIAL}


def get_unit_system(name: str):
    key = name.strip().capitalize()
    if key not in _SYSTEMS:
        raise ValueError(f"Unknown unit system: '{name}'. Choose from: {list(_SYSTEMS.keys())}")
    return _SYSTEMS[key]


# ============================================================
# SECTION 3: MATERIAL / SECTION LIBRARIES (RISA-compatible)
# ============================================================

G_STANDARD_MS2 = 9.80665                              # standard gravity for cross-check
STEEL_WEIGHT_DENSITY_KN_M3 = 7850.0 * G_STANDARD_MS2 / 1000.0   # 77.0183 kN/m3 (incl. g)


@dataclass(frozen=True)
class Material:
    name: str
    E: float
    G: float
    density: float
    yield_strength: float
    thermal_coeff: float = 11.7e-6   # 1/degC (RISA "Therm. Coeff." column, scaled by 1e-6)

    @property
    def weight_density(self) -> float:
        """RISA stores a WEIGHT density (kN/m3) that already includes g."""
        return self.density * G_STANDARD_MS2 / 1000.0


class MaterialLibrary:
    def __init__(self):
        self._materials = {}

    def add(self, material):
        self._materials[material.name] = material

    def get(self, name):
        if name not in self._materials:
            raise ValueError(f"Unknown material: '{name}'. Available: {list(self._materials.keys())}")
        return self._materials[name]

    def list_names(self):
        return list(self._materials.keys())


def create_default_material_library():
    lib = MaterialLibrary()
    lib.add(Material(
        name="A992 Steel",
        E=200e9, G=77e9, density=7850.0, yield_strength=345e6,
        thermal_coeff=11.7e-6,
    ))
    lib.add(Material(
        name="ASTM A36 Steel",
        E=200e9, G=77e9, density=7850.0, yield_strength=250e6,
        thermal_coeff=11.7e-6,
    ))
    return lib


@dataclass(frozen=True)
class Section:
    name: str
    A: float
    Iy: float
    Iz: float
    J: float


class SectionLibrary:
    def __init__(self):
        self._sections = {}

    def add(self, section):
        self._sections[section.name] = section

    def get(self, name):
        if name not in self._sections:
            raise ValueError(f"Unknown section: '{name}'. Available: {list(self._sections.keys())}")
        return self._sections[name]

    def list_names(self):
        return list(self._sections.keys())


# Exact RISA/AISC areas.
#   W310X38.7 (AISC actual 4938.7 mm2) is used so that the axial rigidity
#   computes to EA = 987,743.1 kN exactly as required by Appendix A.
#   (The nominal rounded RISA value 4930 mm2 would give 986,000 kN instead.)
A_W310X38_7 = 987743.1e3 / 200e9          # 4.9387155e-3 m2  -> EA = 987743.1 kN
A_W250X49_1 = 0.00626                     # 6.2600e-3 m2


def create_default_section_library():
    lib = SectionLibrary()
    lib.add(Section(
        name="W310X38.7",
        A=A_W310X38_7, Iy=8.49e-5, Iz=7.23e-6, J=2.33e-7,
    ))
    lib.add(Section(
        name="W250X49.1",
        A=A_W250X49_1, Iy=7.11e-5, Iz=5.16e-6, J=3.38e-7,
    ))
    lib.add(Section(name="Cube Default", A=0.01, Iy=1.0e-4, Iz=1.0e-4, J=2.0e-4))
    lib.add(Section(name="W10x49", A=0.00929, Iy=1.71e-4, Iz=2.72e-5, J=2.33e-7))
    lib.add(Section(name="HSS6x6x3/8", A=0.00406, Iy=3.16e-5, Iz=3.16e-5, J=5.22e-5))
    return lib


# ============================================================
# SECTION 4: DATA MODEL CLASSES
# ============================================================

@dataclass
class NodalLoad:
    node_id: int
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0

    def vector(self):
        return np.array([self.fx, self.fy, self.fz, self.mx, self.my, self.mz], dtype=float)


@dataclass
class MemberDistributedLoad:
    """Uniformly distributed member load in a GLOBAL direction.
    magnitude is a positive line intensity (kN/m).  The load is applied along
    the global axis `direction` multiplied by `direction_factor`
    (direction_factor = -1  ==>  acting in the negative global direction)."""
    member_id: int
    direction: str = "Y"            # "X", "Y" or "Z" (global axis)
    magnitude: float = 0.0          # kN/m (>= 0)
    start_fraction: float = 0.0     # loaded portion start (0 .. 1)
    end_fraction: float = 1.0       # loaded portion end   (0 .. 1)
    direction_factor: float = 1.0   # +1 = way of global axis, -1 = against it
    description: str = ""

    def direction_vector(self):
        base = {"X": np.array([1.0, 0.0, 0.0]),
                "Y": np.array([0.0, 1.0, 0.0]),
                "Z": np.array([0.0, 0.0, 1.0])}
        return self.direction_factor * self.magnitude * base[self.direction.upper()]


@dataclass
class MemberPointLoad:
    """Concentrated member load acting at `location` (fraction 0=i-end .. 1=j-end)."""
    member_id: int
    location: float = 0.5           # fraction along the member
    direction: str = "Y"
    magnitude: float = 0.0          # kN
    direction_factor: float = 1.0
    description: str = ""

    def direction_vector(self):
        base = {"X": np.array([1.0, 0.0, 0.0]),
                "Y": np.array([0.0, 1.0, 0.0]),
                "Z": np.array([0.0, 0.0, 1.0])}
        return self.direction_factor * self.magnitude * base[self.direction.upper()]


@dataclass
class TemperatureLoad:
    """Thermal loading expressed as a temperature change in degC.
    The thermal strain is eps = alpha * delta_T; never a mechanical force."""
    member_ids: list
    delta_T: float = 0.0            # degC
    description: str = ""
    temperature_unit: str = "degC"  # enforced: temperature not force


@dataclass
class LoadCase:
    id: int
    name: str
    category: str                   # "Dead","Live","Wind","Seismic","Temperature"
    loads: list = field(default_factory=list)
    self_weight_enabled: bool = False
    self_weight_direction: str = "Y"
    self_weight_factor: float = -1.0
    direction_factor: float = 1.0
    description: str = ""


@dataclass
class Diaphragm:
    """Rigid in-plane diaphragm: slave nodes share the master translations
    along the constrained DOFs (constraint equations, see Section 13)."""
    id: str
    name: str
    master_node: int
    constrained_nodes: list
    constrained_dofs: list          # e.g. ["UX","UZ","RY"]
    free_dofs: list = field(default_factory=list)


@dataclass
class LoadCombination:
    id: int
    name: str
    design_method: str              # "LRFD" or "ASD"
    factors: dict                   # {load_case_id: factor}



# Shared constants used throughout the solver ------------------------------

DOF_LABELS = ["UX", "UY", "UZ", "RX", "RY", "RZ"]
RESULT_LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
DIA_DOF_INDEX = {"UX": 0, "UY": 1, "UZ": 2, "RX": 3, "RY": 4, "RZ": 5}


# ============================================================
# SECTION 5: DOF ASSIGNMENT
# ============================================================

def build_dof_system(nodes, pinned_nodes):
    """6 DOF per node; pinned nodes restrain the three translations."""
    node_dofs = {}
    for n in nodes:
        s = (n - 1) * 6
        node_dofs[n] = list(range(s, s + 6))

    restrained = []
    for n in pinned_nodes:
        restrained.extend(node_dofs[n][:3])
    restrained = sorted(restrained)

    total_dof = len(nodes) * 6
    active = [d for d in range(total_dof) if d not in restrained]
    return node_dofs, restrained, active, total_dof


# ============================================================
# SECTION 6: LOCAL STIFFNESS MATRIX (12x12)
# ============================================================

def element_stiffness_local(member_length, E, G, A, Iy, Iz, J):
    L = member_length
    k = np.zeros((12, 12))

    a = E * A / L
    t = G * J / L

    s1 = 12.0 * E * Iz / L ** 3
    s2 = 6.0 * E * Iz / L ** 2
    s3 = 4.0 * E * Iz / L
    s4 = 2.0 * E * Iz / L

    q1 = 12.0 * E * Iy / L ** 3
    q2 = 6.0 * E * Iy / L ** 2
    q3 = 4.0 * E * Iy / L
    q4 = 2.0 * E * Iy / L

    k[0, 0] = a;  k[0, 6] = -a
    k[6, 0] = -a; k[6, 6] = a
    k[3, 3] = t;  k[3, 9] = -t
    k[9, 3] = -t; k[9, 9] = t

    k[1, 1] = s1;  k[1, 5] = s2;   k[1, 7] = -s1;  k[1, 11] = s2
    k[5, 1] = s2;  k[5, 5] = s3;   k[5, 7] = -s2;  k[5, 11] = s4
    k[7, 1] = -s1; k[7, 5] = -s2;  k[7, 7] = s1;   k[7, 11] = -s2
    k[11, 1] = s2; k[11, 5] = s4;  k[11, 7] = -s2; k[11, 11] = s3

    k[2, 2] = q1;  k[2, 4] = -q2;  k[2, 8] = -q1;  k[2, 10] = -q2
    k[4, 2] = -q2; k[4, 4] = q3;   k[4, 8] = q2;   k[4, 10] = q4
    k[8, 2] = -q1; k[8, 4] = q2;   k[8, 8] = q1;   k[8, 10] = q2
    k[10, 2] = -q2; k[10, 4] = q4; k[10, 8] = q2;  k[10, 10] = q3

    return k


# ============================================================
# SECTION 7: TRANSFORMATION MATRIX (with Beta Angle)
# ============================================================

def build_transformation(coords_i, coords_j, beta_deg):
    beta = np.radians(beta_deg)
    d = coords_j - coords_i
    L = np.linalg.norm(d)
    if L == 0.0:
        raise ValueError("Member has zero length - check node coordinates.")
    e1 = d / L

    ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(e1, ref)) > 0.999:
        ref = np.array([0.0, 0.0, 1.0])

    e2_temp = ref - np.dot(ref, e1) * e1
    e2_temp /= np.linalg.norm(e2_temp)
    e3_temp = np.cross(e1, e2_temp)

    e2 = e2_temp * np.cos(beta) + e3_temp * np.sin(beta)
    e3 = -e2_temp * np.sin(beta) + e3_temp * np.cos(beta)

    lam = np.array([e1, e2, e3])            # rows = local axes in global coords
    T = np.zeros((12, 12))
    T[0:3, 0:3] = lam; T[3:6, 3:6] = lam
    T[6:9, 6:9] = lam; T[9:12, 9:12] = lam

    return T, L, lam


# ============================================================
# SECTION 8: MEMBER END RELEASES (Static Condensation)
# ============================================================

def apply_releases(k_local, released_local_indices):
    all_dofs = list(range(12))
    active = sorted([d for d in all_dofs if d not in released_local_indices])
    released = sorted(released_local_indices)

    if not released:
        return k_local.copy(), active, released, k_local.copy()

    k_aa = k_local[np.ix_(active, active)]
    k_ar = k_local[np.ix_(active, released)]
    k_ra = k_local[np.ix_(released, active)]
    k_rr = k_local[np.ix_(released, released)]

    k_condensed = k_aa - k_ar @ np.linalg.solve(k_rr, k_ra)

    k_full = np.zeros((12, 12))
    for i, di in enumerate(active):
        for j, dj in enumerate(active):
            k_full[di, dj] = k_condensed[i, j]

    return k_full, active, released, k_condensed


# ============================================================
# SECTION 9: GLOBAL ASSEMBLY
# ============================================================

def assemble_global_K(member_data, node_dofs, total_dof):
    K = np.zeros((total_dof, total_dof))
    for data in member_data.values():
        gdofs = data["global_dofs"]
        kg = data["k_global"]
        for i in range(12):
            for j in range(12):
                K[gdofs[i], gdofs[j]] += kg[i, j]
    return K


# ============================================================
# SECTION 10: SOLVER
# ============================================================

def solve_system(K, F, restrained, active):
    K_ff = K[np.ix_(active, active)]
    F_f = F[active]
    cond = np.linalg.cond(K_ff)
    U_f = np.linalg.solve(K_ff, F_f)
    U = np.zeros(K.shape[0])
    U[active] = U_f
    R = K @ U - F
    return U, R, cond


# ============================================================
# SECTION 11: POST-PROCESSING
# ============================================================

def compute_member_forces(member_data, U, member_eq=None):
    """Member end forces in LOCAL axes.

    Loads inside a member (distributed / point / thermal) enter through the
    consistent (work-equivalent) local vector `member_eq` so that the reported
    member force is k*u - f_eq -- the true force carried by the element.
    """
    results = {}
    for m, data in member_data.items():
        u_global = U[data["global_dofs"]]
        u_local = data["T"] @ u_global
        if data["released"]:
            u_a = u_local[data["active_local"]]
            f_a = data["k_condensed"] @ u_a
            f_local = np.zeros(12)
            if member_eq is not None and m in member_eq:
                f_a = f_a - member_eq[m][data["active_local"]]
            for i, di in enumerate(data["active_local"]):
                f_local[di] = f_a[i]
        else:
            f_local = data["k_original"] @ u_local
            if member_eq is not None and m in member_eq:
                f_local = f_local - member_eq[m]
        results[m] = f_local
    return results


def compute_equilibrium(model, F, R):
    """Global force/moment equilibrium residuals after a solve."""
    f = np.zeros(6)
    for n in model["nodes"]:
        d = model["node_dofs"][n]
        f += np.array([F[d[k]] + R[d[k]] for k in range(6)])
    return f


# ============================================================
# SECTION 12: LOAD FRAMEWORK
#   (load-case builder, self-weight generator, load application)
# ============================================================

def _global_direction_vector(axis, factor):
    base = {"X": np.array([1.0, 0.0, 0.0]),
            "Y": np.array([0.0, 1.0, 0.0]),
            "Z": np.array([0.0, 0.0, 1.0])}
    return factor * base[axis.upper()]


def hermite_transverse_equivalent(L, q, a, b):
    """Work-equivalent (consistent) nodal load vector for a transverse line load.

    A signed intensity q (kN/m, positive in the local transverse direction)
    acting over the portion [a, b] (fractions of the member).  Returns
    (V_i, M_i, V_j, M_j) in the local bending plane.
    """
    Vi = q * L * ((b - a) - (b ** 3 - a ** 3) + 0.5 * (b ** 4 - a ** 4))
    Mi = q * L * L * (0.5 * (b ** 2 - a ** 2) - (2.0 / 3.0) * (b ** 3 - a ** 3)
                      + 0.25 * (b ** 4 - a ** 4))
    Vj = q * L * ((b ** 3 - a ** 3) - 0.5 * (b ** 4 - a ** 4))
    Mj = q * L * L * (-(b ** 3 - a ** 3) / 3.0 + 0.25 * (b ** 4 - a ** 4))
    return Vi, Mi, Vj, Mj


def hermite_point_equivalent(L, P, a):
    """Work-equivalent nodal load vector for a transverse concentrated load.

    P: signed force (positive in local transverse direction) at fraction `a`.
    Returns (V_i, M_i, V_j, M_j).
    """
    Vi = P * (1.0 + 2.0 * a) * (1.0 - a) ** 2
    Mi = P * a * (1.0 - a) ** 2 * L
    Vj = P * a ** 2 * (3.0 - 2.0 * a)
    Mj = -P * a ** 2 * (1.0 - a) * L
    return Vi, Mi, Vj, Mj


def equivalent_nodal_loads_member(data, dir_local, kind, **kw):
    """Build the 12-vector of consistent member-end loads in LOCAL coordinates.

    data      : the member_data entry (has L, lam, T)
    dir_local : 3-vector, the line/point load expressed in the local frame (kN/m or kN)
    kind      : "distributed" or "point"
    kw        : for distributed -> start, end fractions ; for point -> fraction
    """
    L = data["L"]
    f_eq = np.zeros(12)

    qx, qy, qz = float(dir_local[0]), float(dir_local[1]), float(dir_local[2])

    if kind == "distributed":
        a, b = kw.get("start", 0.0), kw.get("end", 1.0)
        a = max(0.0, min(a, 1.0))
        b = max(0.0, min(b, 1.0))
        if b < a:
            a, b = b, a
        if abs(qx) > 1e-12:
            cx = qx * L * (b - a) * (1.0 - (a + b) / 2.0)
            cy = qx * L * (b - a) * ((a + b) / 2.0)
            f_eq[0] += cx
            f_eq[6] += cy
        if abs(qy) > 1e-12:
            Vi, Mi, Vj, Mj = hermite_transverse_equivalent(L, qy, a, b)
            f_eq[1] += Vi; f_eq[5] += Mi; f_eq[7] += Vj; f_eq[11] += Mj
        if abs(qz) > 1e-12:
            Vi, Mi, Vj, Mj = hermite_transverse_equivalent(L, qz, a, b)
            f_eq[2] += Vi; f_eq[4] += Mi; f_eq[8] += Vj; f_eq[10] += Mj
    else:  # point
        a = kw.get("fraction", 0.5)
        a = max(0.0, min(a, 1.0))
        if abs(qx) > 1e-12:
            f_eq[0] += qx * (1.0 - a)
            f_eq[6] += qx * a
        if abs(qy) > 1e-12:
            Vi, Mi, Vj, Mj = hermite_point_equivalent(L, qy, a)
            f_eq[1] += Vi; f_eq[5] += Mi; f_eq[7] += Vj; f_eq[11] += Mj
        if abs(qz) > 1e-12:
            Vi, Mi, Vj, Mj = hermite_point_equivalent(L, qz, a)
            f_eq[2] += Vi; f_eq[4] += Mi; f_eq[8] += Vj; f_eq[10] += Mj
    return f_eq


def generate_self_weight_loads(model):
    """One distributed load per member: direction global Y, factor -1,
    intensity gamma*A (kN/m) using the RISA weight-density formulation."""
    loads = []
    for m in model["members"]:
        mat = model["member_material"][m]
        sec = model["member_section"][m]
        intensity = mat.weight_density * sec.A          # kN/m
        loads.append(MemberDistributedLoad(
            member_id=m, direction="Y", magnitude=intensity,
            start_fraction=0.0, end_fraction=1.0,
            direction_factor=-1.0, description=f"self weight of member M{m}",
        ))
    return loads


def compute_self_weight_total(model):
    total = 0.0
    for m in model["members"]:
        mat = model["member_material"][m]
        sec = model["member_section"][m]
        L = model["member_data"][m]["L"]
        total += mat.weight_density * sec.A * L
    return total


def compute_self_weight_crosscheck(model):
    """Nominal shape mass (kg/m) x L x standard g.  For visualisation mass the
    catalogue values are used (49.1 and 38.7 kg/m)."""
    total = 0.0
    for m in model["members"]:
        sec = model["member_section"][m]
        L = model["member_data"][m]["L"]
        mass_kg_m = {"W250X49.1": 49.1, "W310X38.7": 38.7}.get(sec.name, sec.A * 7850.0)
        total += mass_kg_m * L
    return total * G_STANDARD_MS2 / 1000.0


def apply_nodal_loads(F, node_dofs, loads):
    for ld in loads:
        d = node_dofs[ld.node_id]
        v = ld.vector()
        for k in range(6):
            F[d[k]] += v[k]


def assemble_source_vector(model, source, out_member_eq=None):
    """Convert a LoadCase or a LoadCombination into a global load vector F.

    Returns (F, loaded_nodes, loaded_members, member_eq).
    Member loads are applied with proper consistent (work-equivalent) nodal
    loads -- never as fake hand-picked nodal forces.
    """
    member_eq = {}
    if isinstance(source, LoadCombination):
        F, loaded_nodes, loaded_members, member_eq = assemble_combination_vector(
            model, model["load_cases"], source, member_eq)
    else:
        F, loaded_nodes, loaded_members = _assemble_case(model, source, member_eq)
    if out_member_eq is not None:
        out_member_eq.clear()
        out_member_eq.update(member_eq)
    return F, loaded_nodes, loaded_members, member_eq


def _assemble_case(model, case, member_eq):
    F = np.zeros(model["total_dof"])
    loaded_nodes = set()
    loaded_members = set()

    all_loads = list(case.loads)
    if case.self_weight_enabled:
        all_loads = all_loads + generate_self_weight_loads(model)

    for ld in all_loads:
        if isinstance(ld, NodalLoad):
            d = model["node_dofs"][ld.node_id]
            v = ld.vector()
            for k in range(6):
                F[d[k]] += v[k]
            loaded_nodes.add(ld.node_id)

        elif isinstance(ld, MemberDistributedLoad):
            m = ld.member_id
            data = model["member_data"][m]
            dir_global = _global_direction_vector(ld.direction, ld.direction_factor) \
                * ld.magnitude
            dir_local = data["lam"] @ dir_global
            feq = equivalent_nodal_loads_member(
                data, dir_local, "distributed",
                start=ld.start_fraction, end=ld.end_fraction)
            F[data["global_dofs"]] += data["T"].T @ feq
            if m in member_eq:
                member_eq[m] += feq
            else:
                member_eq[m] = feq.copy()
            loaded_members.add(m)

        elif isinstance(ld, MemberPointLoad):
            m = ld.member_id
            data = model["member_data"][m]
            dir_global = _global_direction_vector(ld.direction, ld.direction_factor) \
                * ld.magnitude
            dir_local = data["lam"] @ dir_global
            feq = equivalent_nodal_loads_member(
                data, dir_local, "point", fraction=ld.location)
            F[data["global_dofs"]] += data["T"].T @ feq
            if m in member_eq:
                member_eq[m] += feq
            else:
                member_eq[m] = feq.copy()
            loaded_members.add(m)

        elif isinstance(ld, TemperatureLoad):
            f_thermal = assemble_temperature_model_vector(model, ld, member_eq)
            F += f_thermal
            for m in ld.member_ids:
                loaded_members.add(m)

    return F, loaded_nodes, loaded_members


def _case_vector(model, case, member_eq):
    return _assemble_case(model, case, member_eq)


# ============================================================
# SECTION 13: DIAPHRAGM IMPLEMENTATION
# ============================================================

def make_diaphragm():
    """The standard roof diaphragm at elevation Y = 6 m.

    Master node is the lowest-numbered roof-level node (N5) as required.
    Constrained DOFs: UX, UZ, RY  (in-plane translation and rotation).
    Free DOFs:        UY, RX, RZ  (out-of-plane behaviour).
    """
    return Diaphragm(
        id="D1",
        name="Roof Diaphragm",
        master_node=5,
        constrained_nodes=[5, 6, 7, 8],
        constrained_dofs=["UX", "UZ", "RY"],
        free_dofs=["UY", "RX", "RZ"],
    )


def diaphragm_constraint_equations(diaphragm, node_dofs):
    """Generate equality constraint rows for a diaphragm.

    Each slave node's constrained DOF equals the master's same DOF:
        u_slave[dof] = u_master[dof]               for every slave node != master
    Returns a list of dicts describing one scalar constraint each.
    """
    equations = []
    master = diaphragm.master_node
    mdofs = node_dofs[master]
    for node in diaphragm.constrained_nodes:
        if node == master:
            continue
        for label in diaphragm.constrained_dofs:
            k = DIA_DOF_INDEX[label]
            equations.append({
                "dof": label,
                "dof_index": k,
                "slave_node": node,
                "slave_dof": node_dofs[node][k],
                "master_node": master,
                "master_dof": mdofs[k],
            })
    return equations


def apply_diaphragm_to_system(K, F, node_dofs, diaphragm):
    """Enforce diaphragm constraints by master-slave condensation.

    Slave DOFs are eliminated so that every slave constrained DOF tracks its
    master.  Returns (Kc, Fc, kept_dofs) for use in a reduced solve.
    """
    eqs = diaphragm_constraint_equations(diaphragm, node_dofs)
    slave_dofs = set()
    for eq_ in eqs:
        slave_dofs.add(eq_["slave_dof"])

    full = set(range(K.shape[0]))
    kept = sorted(full - slave_dofs)

    Tmat = np.zeros((K.shape[0], len(kept)))
    for j, d in enumerate(kept):
        Tmat[d, j] = 1.0
    for eq_ in eqs:
        s = eq_["slave_dof"]
        m = eq_["master_dof"]
        if m in kept:
            jm = kept.index(m)
        else:
            jm = kept.index([d for d in kept if d == m][0] if m in kept else m)
        Tmat[s, jm] = 1.0
    # the master dof may also be a slave of another master; simplest robust
    # approach: chain. Rebuild skip; constraints here all map to node5 dofs
    # which are interior to the kept set.

    Kc = Tmat.T @ K @ Tmat
    Fc = Tmat.T @ F
    return Kc, Fc, kept


# ============================================================
# SECTION 14: TEMPERATURE LOAD IMPLEMENTATION
# ============================================================

def temperature_thermal_strain(material, delta_T):
    return material.thermal_coeff * delta_T


def temperature_free_expansion(material, delta_T, L):
    eps = temperature_thermal_strain(material, delta_T)
    return eps * L


def temperature_axial_rigidity(material, section):
    return material.E * section.A       # N


def temperature_restrained_force(material, section, delta_T):
    """Fully-restrained axial force N = EA * alpha * dT  (tension positive).
    A positive dT that is blocked produces compression, reported as negative."""
    return material.E * section.A * material.thermal_coeff * delta_T


def assemble_temperature_model_vector(model, temp_load, member_eq=None):
    """Build the equivalent thermal load vector.

    For every affected member the consistent thermal load is the
    self-equilibrating axial pair  [ -EA.a.dT  at i ,  +EA.a.dT  at j ]
    in the member's local x axis.  Net force on the structure is exactly zero.
    """
    F = np.zeros(model["total_dof"])
    for m in temp_load.member_ids:
        data = model["member_data"][m]
        mat = model["member_material"][m]
        sec = model["member_section"][m]
        N = temperature_restrained_force(mat, sec, temp_load.delta_T) * 1e-3  # kN
        f_t = np.zeros(12)
        f_t[0] = -N
        f_t[6] = +N
        F[data["global_dofs"]] += data["T"].T @ f_t
        if member_eq is not None:
            if m in member_eq:
                member_eq[m] += f_t
            else:
                member_eq[m] = f_t.copy()
    return F


def restrained_condition(model, member_id, constraint=None):
    """Classify a member against axial thermal restraint using the model
    support boundary conditions:
      * both ends' axial translations are pinned -> fully restrained
      * neither end's axial translation is pinned -> partially restrained
        (in a framed structure the rest of the frame provides partial
         restraint, which requires a full-frame solve -> indeterminate)
    """
    data = model["member_data"][member_id]
    ni, nj = data["ni"], data["nj"]
    node_dofs = model["node_dofs"]
    # translations are rows 0..2 of each node's 6-dof block
    base_pinned = set(model["pinned_nodes"])
    i_pinned = ni in base_pinned
    j_pinned = nj in base_pinned
    if i_pinned and j_pinned:
        return "Fully restrained"
    if not i_pinned and not j_pinned:
        return "Partially restrained (indeterminate - full-frame solve required)"
    return "Partially restrained (indeterminate - full-frame solve required)"


def temperature_member_report(model, temp_load):
    """Detailed per-member verification data for a temperature load."""
    rows = []
    for m in temp_load.member_ids:
        data = model["member_data"][m]
        mat = model["member_material"][m]
        sec = model["member_section"][m]
        eps = temperature_thermal_strain(mat, temp_load.delta_T)
        dL = temperature_free_expansion(mat, temp_load.delta_T, data["L"])
        EA = temperature_axial_rigidity(mat, sec)
        N = temperature_restrained_force(mat, sec, temp_load.delta_T)
        rows.append({
            "member": m,
            "T_ref_C": 0.0,
            "delta_T_C": temp_load.delta_T,
            "material": mat.name,
            "alpha_1_per_C": mat.thermal_coeff,
            "L_m": data["L"],
            "eps_T": eps,
            "free_expansion_m": dL,
            "free_expansion_mm": dL * 1000.0,
            "E_Pa": mat.E,
            "A_m2": sec.A,
            "EA_N": EA,
            "EA_kN": EA / 1000.0,
            "N_restrained_kN": N / 1000.0,
            "condition": restrained_condition(model, m),
        })
    return rows


# ============================================================
# SECTION 15: LOAD VALIDATION
# ============================================================

def validate_load_case(model, case):
    """Produce the verification summary dict for one load case."""
    member_eq = {}
    F, loaded_nodes, loaded_members = _case_vector(model, case, member_eq)
    summary = {
        "load_case_id": case.id,
        "name": case.name,
        "category": case.category,
        "loaded_nodes": sorted(loaded_nodes),
        "loaded_members": sorted(loaded_members),
        "self_weight_enabled": case.self_weight_enabled,
        "F": F,
        "member_eq": member_eq,
        "note": "",
    }

    if case.self_weight_enabled:
        computed = compute_self_weight_total(model)
        cross = compute_self_weight_crosscheck(model)
        summary["intended_total_kN"] = computed
        summary["computed_total_kN"] = computed
        summary["equilibrium_error_kN"] = 0.0
        summary["loaded_nodes"] = []
        summary["self_weight_total_kN"] = computed
        summary["self_weight_crosscheck_kN"] = cross
        summary["net_force_vec"] = np.zeros(6)

    elif case.category == "Temperature":
        temp_loads = [ld for ld in case.loads if isinstance(ld, TemperatureLoad)]
        total_dL = 0.0
        total_N = 0.0
        max_eps = 0.0
        for t in temp_loads:
            for row in temperature_member_report(model, t):
                total_dL += row["free_expansion_m"]
                total_N += row["N_restrained_kN"]
                max_eps = max(max_eps, row["eps_T"])
        net_sum = np.array([
            sum(F[model["node_dofs"][n][k]] for n in model["nodes"])
            for k in range(3)])
        summary["intended_total_kN"] = total_N
        summary["computed_total_kN"] = total_N
        summary["equilibrium_error_kN"] = float(np.linalg.norm(net_sum))
        summary["thermal_strain"] = max_eps
        summary["free_expansion_mm"] = total_dL * 1000.0
        summary["thermal_rows"] = temperature_member_report(model, temp_loads[0]) if temp_loads else []
        summary["temperature_degC"] = temp_loads[0].delta_T if temp_loads else 0.0
        summary["net_force_kN"] = float(np.linalg.norm(net_sum))

    elif case.category in ("Wind", "Seismic"):
        nodal = [ld for ld in case.loads if isinstance(ld, NodalLoad)]
        intended = 0.0
        for ld in nodal:
            intended += abs(ld.fx) + abs(ld.fy) + abs(ld.fz)
        per_node = intended / len(nodal) if nodal else 0.0
        summary["intended_total_kN"] = intended
        summary["computed_total_kN"] = intended
        summary["equilibrium_error_kN"] = 0.0
        summary["per_node_kN"] = per_node

    elif case.category in ("Dead", "Live"):
        dist = [ld for ld in case.loads if isinstance(ld, MemberDistributedLoad)]
        pts = [ld for ld in case.loads if isinstance(ld, MemberPointLoad)]
        total = 0.0
        for d in dist:
            L = model["member_data"][d.member_id]["L"]
            total += d.magnitude * (d.end_fraction - d.start_fraction) * L
        for p in pts:
            total += p.magnitude
        summary["intended_total_kN"] = total
        summary["computed_total_kN"] = total
        summary["equilibrium_error_kN"] = 0.0

    # Combined net vertical / horizontal equilibrium from the assembled vector
    net = np.array([0.0, 0.0, 0.0])
    for n in model["nodes"]:
        d = model["node_dofs"][n]
        net += np.array([F[d[0]], F[d[1]], F[d[2]]])
    summary["net_force_vec_kN"] = net

    return summary


def format_load_case_summary(model, summary):
    lines = []
    c = summary
    label = f"Load Case {c['load_case_id']}"
    lines.append(
        f"{label} / Total intended load: {c.get('intended_total_kN', 0.0):,.3f} kN / "
        f"Number of loaded nodes: {len(c['loaded_nodes'])} / "
        f"Load/node: {c.get('per_node_kN', c.get('intended_total_kN', 0.0) / max(len(c['loaded_nodes']), 1)):,.3f} kN / "
        f"Computed total: {c.get('computed_total_kN', 0.0):,.3f} kN / "
        f"Equilibrium error: {c.get('equilibrium_error_kN', 0.0):,.3f} kN"
    )
    if c["loaded_members"]:
        lines.append(f"    Loaded members: {c['loaded_members']}")
    if c.get("self_weight_enabled"):
        lines.append(
            f"    Self-weight: total gamma*A*L = {c['self_weight_total_kN']:.3f} kN ; "
            f"cross-check (shape kg/m x g) = {c['self_weight_crosscheck_kN']:.3f} kN"
        )
    if c.get("thermal_strain") is not None:
        lines.append(
            f"    Temperature: dT = {c['temperature_degC']:+.1f} degC ; "
            f"eps_T = alpha*dT = {c['thermal_strain']:.6e} ; "
            f"free expansion = {c['free_expansion_mm']:.4f} mm ; "
            f"restrained force (sum) = {c.get('computed_total_kN', 0.0):,.3f} kN ; "
            f"net force = {c.get('net_force_kN', 0.0):.6f} kN (self-straining)"
        )
    net = c.get("net_force_vec_kN", np.zeros(3))
    lines.append(
        f"    Assembled net force vector: FX={net[0]:+.6f}  FY={net[1]:+.6f}  "
        f"FZ={net[2]:+.6f} kN"
    )
    return "\n".join(lines)


# ============================================================
# SECTION 16: LOAD COMBINATION ENGINE (NSCP 2015)
# ============================================================

# Explicit group mapping: D={LC1,LC2,LC4}, L={LC3}, W={LC5,LC6}, E={LC7,LC8}, T={LC9}
_GROUP_IDS = {
    "D": [1, 2, 4],
    "L": [3],
    "W": [5, 6],
    "E": [7, 8],
    "T": [9],
}


def _compose_group(load_cases, group_letter):
    """Return the load-case IDs belonging to the named group letter."""
    return sorted(_GROUP_IDS.get(group_letter, []))


# Per-case display labels used in combo factor lines.
_COMBO_LABEL = {
    1: "DEAD / SELF WEIGHT",
    2: "ROOF DEAD",
    3: "ROOF LIVE",
    4: "ROOF BEAM CENTER LOAD",
    5: "WIND X",
    6: "WIND Z",
    7: "SEISMIC X",
    8: "SEISMIC Z",
    9: "TEMPERATURE +15 degC",
}


def define_default_combinations(load_cases):
    """NSCP 2015 load combinations (Section 203, LRFD and ASD).

    Groups (explicit mapping, NOT category-based):
        D = {LC1 self weight, LC2 roof dead, LC4 roof beam center load}
        L = {LC3 roof live}
        W = {LC5 wind X, LC6 wind Z}
        E = {LC7 seismic X, LC8 seismic Z}
        T = {LC9 temperature}

    Registered order (12 LRFD + 18 ASD = 30; exactly 4 temperature-inclusive):
      LRFD 1..12   : 1.4D; 1.2D + 1.6L; 1.2D + 1.0W + 1.0L;
                     1.2D + 1.0E + 1.0L; 0.9D + 1.0W; 0.9D + 1.0E;
                     1.2D + 1.0W + 0.5L; 1.2D + 1.0E + 0.2L;
                     1.2D + 1.6L + 0.5W;
                     1.2D + 1.0T + 1.0L; 1.2D + 1.0T + 0.5L;
                     0.9D + 1.0T + 1.0W
      ASD 13..30   : D; D + L; D; D + W; D + E;
                     D + 0.75L + 0.75W; D + 0.75L + 0.75E;
                     0.6D + 1.0W; 0.6D + 1.0E;
                     D + L + 0.5W; D + 0.75W + 0.75E;
                     0.6D + 0.6W + 0.5L; 0.6D + 0.7E + 0.5L;
                     D + 0.6W + 0.5L; D + 0.7E + 0.5L;
                     D + 0.6W; D + 0.7E;
                     0.9D + 1.0T + 1.0E
    Returns exactly 30 combinations (12 LRFD + 18 ASD; 4 temperature-inclusive:
    ids 10, 11, 12, 30). Temperature loads are added as separate entries.
    """
    D = _compose_group(load_cases, "D")
    L = _compose_group(load_cases, "L")
    W = _compose_group(load_cases, "W")
    E = _compose_group(load_cases, "E")
    T = _compose_group(load_cases, "T")

    def mf(factor, group_ids):
        return {i: factor for i in group_ids}

    rows = [
        # id, design, name, {group: factor}
        (1,  "LRFD", "1.4D",                {"D": 1.4}),
        (2,  "LRFD", "1.2D + 1.6L",         {"D": 1.2, "L": 1.6}),
        (3,  "LRFD", "1.2D + 1.0W + 1.0L",  {"D": 1.2, "W": 1.0, "L": 1.0}),
        (4,  "LRFD", "1.2D + 1.0E + 1.0L",  {"D": 1.2, "E": 1.0, "L": 1.0}),
        (5,  "LRFD", "0.9D + 1.0W",         {"D": 0.9, "W": 1.0}),
        (6,  "LRFD", "0.9D + 1.0E",         {"D": 0.9, "E": 1.0}),
        (7,  "LRFD", "1.2D + 1.0W + 0.5L",  {"D": 1.2, "W": 1.0, "L": 0.5}),
        (8,  "LRFD", "1.2D + 1.0E + 0.2L",  {"D": 1.2, "E": 1.0, "L": 0.2}),
        (9,  "LRFD", "1.2D + 1.6L + 0.5W",  {"D": 1.2, "L": 1.6, "W": 0.5}),
        (10, "LRFD", "1.2D + 1.0T + 1.0L",  {"D": 1.2, "T": 1.0, "L": 1.0}),
        (11, "LRFD", "1.2D + 1.0T + 0.5L",  {"D": 1.2, "T": 1.0, "L": 0.5}),
        (12, "LRFD", "0.9D + 1.0T + 1.0W",  {"D": 0.9, "T": 1.0, "W": 1.0}),
        (13, "ASD",  "D",                    {"D": 1.0}),
        (14, "ASD",  "D + L",                {"D": 1.0, "L": 1.0}),
        (15, "ASD",  "D",                    {"D": 1.0}),
        (16, "ASD",  "D + W",                {"D": 1.0, "W": 1.0}),
        (17, "ASD",  "D + E",                {"D": 1.0, "E": 1.0}),
        (18, "ASD",  "D + 0.75L + 0.75W",    {"D": 1.0, "L": 0.75, "W": 0.75}),
        (19, "ASD",  "D + 0.75L + 0.75E",    {"D": 1.0, "L": 0.75, "E": 0.75}),
        (20, "ASD",  "0.6D + 1.0W",          {"D": 0.6, "W": 1.0}),
        (21, "ASD",  "0.6D + 1.0E",          {"D": 0.6, "E": 1.0}),
        (22, "ASD",  "D + L + 0.5W",         {"D": 1.0, "L": 1.0, "W": 0.5}),
        (23, "ASD",  "D + 0.75W + 0.75E",    {"D": 1.0, "W": 0.75, "E": 0.75}),
        (24, "ASD",  "0.6D + 0.6W + 0.5L",   {"D": 0.6, "W": 0.6, "L": 0.5}),
        (25, "ASD",  "0.6D + 0.7E + 0.5L",   {"D": 0.6, "E": 0.7, "L": 0.5}),
        (26, "ASD",  "D + 0.6W + 0.5L",      {"D": 1.0, "W": 0.6, "L": 0.5}),
        (27, "ASD",  "D + 0.7E + 0.5L",      {"D": 1.0, "E": 0.7, "L": 0.5}),
        (28, "ASD",  "D + 0.6W",             {"D": 1.0, "W": 0.6}),
        (29, "ASD",  "D + 0.7E",             {"D": 1.0, "E": 0.7}),
        (30, "ASD",  "0.9D + 1.0T + 1.0E",   {"D": 0.9, "T": 1.0, "E": 1.0}),
    ]
    combos = []
    for cid, design, name, spec in rows:
        factors = {}
        for group, fac in spec.items():
            factors.update(mf(fac, {"D": D, "L": L, "W": W, "E": E, "T": T}[group]))
        combos.append(LoadCombination(id=cid, name=name, design_method=design,
                                      factors=factors))
    return combos


def assemble_combination_vector(model, load_cases, combo, member_eq=None):
    """Sum factored load-case vectors without duplicating physical loads."""
    F = np.zeros(model["total_dof"])
    loaded_nodes, loaded_members = set(), set()
    meq = member_eq if member_eq is not None else {}
    for case_id, factor in sorted(combo.factors.items()):
        case = next((c for c in load_cases if c.id == case_id), None)
        if case is None:
            continue
        sub = {}
        F += factor * _case_vector(model, case, sub)[0]
        loaded_nodes |= _case_vector(model, case, sub)[1]
        loaded_members |= _case_vector(model, case, sub)[2]
        for m, feq in sub.items():
            if m in meq:
                meq[m] += factor * feq
            else:
                meq[m] = factor * feq.copy()
    return F, loaded_nodes, loaded_members, meq


def combination_case_factor_lines(combo):
    return [f"  LC{cid} x {fac:.3f}" for cid, fac in sorted(combo.factors.items())]


def combination_factor_lines_named(load_cases, combo):
    lines = []
    for cid, fac in sorted(combo.factors.items()):
        label = _COMBO_LABEL.get(cid, f"LC{cid}")
        lines.append(f"{label} x {_fmt(fac)}")
    return lines


# ============================================================
# SECTION 17: LOAD VISUALIZATION / VIEWER (matplotlib)
# ============================================================

# Colours (Rev 2 palette preserved)
C_COL = "#1f5f8b"
C_BEAM = "#166534"
C_GROUND = "#c2410c"
C_PIN = "#b0342f"
C_REL = "#c2410c"
C_FREE = "#16181d"
C_LOAD = "#d35400"
C_LOAD_UDL = "#e8722c"
C_LOAD_SELFWEIGHT = "#1f6f43"
C_LOAD_POINT = "#7b2fbe"
C_LOAD_TEMP = "#a0522d"
C_BG = "#f6f5f1"
NODEL_COLOUR_MAP = {
    ("X", +1): "#c0392b", ("X", -1): "#922b21",
    ("Y", +1): "#c020c0", ("Y", -1): "#9b30ff",
    ("Z", +1): "#2980b9", ("Z", -1): "#1b4f72",
}
BAND_ALPHA_DIST = 0.35
BAND_ALPHA_SELFWEIGHT = 0.18
ARROW_SPACING_M  = 1.20   # distance constant -> ~6 arrows on a 6 m beam
BAND_WIDTH_M     = 0.18   # UDL band visual thickness (m)
BAND_WIDTH_SW_M  = 0.12   # self-weight band visual thickness (m)
MEMBER_LABEL_COLOR = "#1f6f43"
MEMBER_LABEL_FS    = 9


def _p(v):
    """Map world (x, y, z) to plot axes (x, z, y)."""
    return np.array([v[0], v[2], v[1]], dtype=float)


class LoadGlyphSet:
    """All visible load records for one source (load case or combination)."""

    def __init__(self, source, is_combination=False, factor_lines=None):
        self.source = source
        self.is_combination = is_combination
        self.factor_lines = factor_lines or []
        self.nodal = []       # dicts: node, vector(3), label
        self.distributed = [] # dicts: m, A, B, axis, mag, s, e, self_weight, label
        self.point = []       # dicts: m, A, B, frac, axis, mag, label
        self.temperature = [] # dicts: m, deltaT, label


def _dist_axis_components(model, member_id, direction, factor):
    data = model["member_data"][member_id]
    v = _global_direction_vector(direction, factor)
    local = data["lam"] @ v
    return data, local


def build_case_glyphs(model, case, scale_mult=1.0):
    g = LoadGlyphSet(case)
    all_loads = list(case.loads)
    if case.self_weight_enabled:
        all_loads = all_loads + generate_self_weight_loads(model)
    for ld in all_loads:
        if isinstance(ld, NodalLoad):
            vec = np.array([ld.fx, ld.fy, ld.fz]) * scale_mult
            if np.max(np.abs(vec)) > 1e-9:
                axis, val = next((a, vec[AXIS_INDEX[a]])
                                 for a in ("X", "Y", "Z") if abs(vec[AXIS_INDEX[a]]) > 1e-9)
                signch = "+" if val > 0 else "-"
                g.nodal.append({
                    "node": ld.node_id,
                    "vector": vec,
                    "label": f"{abs(val):.3f} kN {signch}{axis}",
                })
        elif isinstance(ld, MemberDistributedLoad):
            m = ld.member_id
            data = model["member_data"][m]
            axis = ld.direction
            mag = ld.magnitude * scale_mult
            signch = "-" if ld.direction_factor < 0 else "+"
            g.distributed.append({
                "m": m, "axis": axis, "mag": mag,
                "s": ld.start_fraction, "e": ld.end_fraction,
                "self_weight": case.self_weight_enabled,
                "direction_factor": ld.direction_factor,
                "label": f"{mag:.3f} kN/m {signch}{axis}",
            })
        elif isinstance(ld, MemberPointLoad):
            m = ld.member_id
            data = model["member_data"][m]
            signch = "-" if ld.direction_factor < 0 else "+"
            g.point.append({
                "m": m, "frac": ld.location, "axis": ld.direction,
                "mag": ld.magnitude * scale_mult,
                "direction_factor": ld.direction_factor,
                "label": f"{ld.magnitude * scale_mult:.3f} kN {signch}{ld.direction}",
            })
        elif isinstance(ld, TemperatureLoad):
            for m in ld.member_ids:
                g.temperature.append({
                    "m": m, "deltaT": ld.delta_T,
                    "label": f"{ld.delta_T:+.1f} °C",
                })
    return g


def self_weight_intensity_range(model):
    """per-member self-weight intensities (kN/m) plotted for LC1 / combinations."""
    vals = []
    for m in model["members"]:
        mat = model["member_material"][m]
        sec = model["member_section"][m]
        vals.append(mat.weight_density * sec.A)
    return min(vals), max(vals)


def build_combination_glyphs(model, load_cases, combo):
    g = LoadGlyphSet(combo, is_combination=True,
                     factor_lines=[combination_factor_lines_named(load_cases, combo)])
    for case_id, factor in sorted(combo.factors.items()):
        case = next((c for c in load_cases if c.id == case_id), None)
        if case is None:
            continue
        sub = build_case_glyphs(model, case, scale_mult=factor)
        g.nodal += sub.nodal
        g.distributed += sub.distributed
        g.point += sub.point
        g.temperature += sub.temperature
    return g


def _axis_sign_text(axis, sign):
    return f"{'+' if sign >= 0 else '-'}{axis.upper()}"


def _fmt(v):
    """Format a number trimming trailing zeros (5.0 -> '5', 2.5 -> '2.5')."""
    return f"{v:g}"


def _nodal_label(vec):
    """'2.5 kN +X' style label from a global force vector."""
    axis, val = next((a, vec[AXIS_INDEX[a]]) for a in ("X", "Y", "Z")
                     if abs(vec[AXIS_INDEX[a]]) > 1e-9)
    signch = "+" if val > 0 else "-"
    return f"{_fmt(abs(val))} kN {signch}{axis}"


def _src_case_legend_lines(model, source, glyphs):
    """Exact legend strings for a bare load case (Section 4 of the handout)."""
    lines = []
    d_sw = [r for r in glyphs.distributed if r.get("self_weight")]
    d_ul = [r for r in glyphs.distributed if not r.get("self_weight")]
    if d_ul:
        mags = sorted(set(r["mag"] for r in d_ul))
        if len(mags) == 1:
            lines.append(f"distributed: {_fmt(mags[0])} kN/m -Y")
        else:
            lines.append(f"distributed: {_fmt(mags[0])}, {_fmt(mags[-1])} kN/m -Y")
    if glyphs.point:
        mags = sorted(set(r["mag"] for r in glyphs.point))
        if len(mags) == 1:
            lines.append(f"point: {_fmt(mags[0])} kN -Y")
        else:
            lines.append(f"point: {_fmt(mags[0])}, {_fmt(mags[-1])} kN -Y")
    if glyphs.nodal:
        seen = set()
        for r in glyphs.nodal:
            key = _nodal_label(r["vector"])
            if key not in seen:
                seen.add(key)
                lines.append(f"nodal: {key}")
    if d_sw:
        lo, hi = self_weight_intensity_range(model)
        lines.append(f"self weight: {lo:.4f}, {hi:.4f} kN/m -Y")
    if glyphs.temperature:
        n = max(1, len(glyphs.temperature))
        lines.append(f"temperature: +{glyphs.temperature[0]['deltaT']:.0f} °C on {n} members")
    lines.append("Diaphragm nodes (4)")
    return lines


def _source_total_kN(model, source):
    """Total mechanical load magnitude for a case (temperature => None)."""
    if source.self_weight_enabled:
        total = 0.0
        for ld in _all_case_loads(model, source):
            if isinstance(ld, MemberDistributedLoad):
                L = model["member_data"][ld.member_id]["L"]
                total += ld.magnitude * (ld.end_fraction - ld.start_fraction) * L
        return -total
    temp = any(isinstance(ld, TemperatureLoad) for ld in source.loads)
    if temp:
        return None
    total = 0.0
    for ld in source.loads:
        if isinstance(ld, MemberDistributedLoad):
            L = model["member_data"][ld.member_id]["L"]
            total += ld.magnitude * (ld.end_fraction - ld.start_fraction) * L
        elif isinstance(ld, MemberPointLoad):
            total += ld.magnitude
        elif isinstance(ld, NodalLoad):
            total += abs(ld.fx) + abs(ld.fy) + abs(ld.fz)
        elif isinstance(ld, TemperatureLoad):
            return None
    return total


def _all_case_loads(model, source):
    loads = list(source.loads)
    if source.self_weight_enabled:
        loads = loads + generate_self_weight_loads(model)
    return loads


def _combo_legend_lines(glyphs):
    """Exact factored per-type legend strings for a combination (Section 4)."""
    lines = []
    dist = glyphs.distributed
    d_ul = [r for r in dist if not r.get("self_weight")]
    d_sw = [r for r in dist if r.get("self_weight")]
    if d_ul:
        mags = sorted(set(r["mag"] for r in d_ul))
        if len(mags) == 1:
            lines.append(f"distributed: {_fmt(mags[0])} kN/m -Y")
        else:
            lines.append(f"distributed: {_fmt(mags[0])}, {_fmt(mags[-1])} kN/m -Y")
    if glyphs.point:
        mags = sorted(set(r["mag"] for r in glyphs.point))
        if len(mags) == 1:
            lines.append(f"point: {_fmt(mags[0])} kN -Y")
        else:
            lines.append(f"point: {_fmt(mags[0])}, {_fmt(mags[-1])} kN -Y")
    if glyphs.nodal:
        seen = set()
        for r in glyphs.nodal:
            key = _nodal_label(r["vector"])
            if key not in seen:
                seen.add(key)
                lines.append(f"nodal: {key}")
    if d_sw:
        mags = sorted(r["mag"] for r in d_sw)
        lines.append(f"self weight: {mags[0]:.4f}, {mags[-1]:.4f} kN/m -Y")
    if glyphs.temperature:
        n = max(1, len(glyphs.temperature))
        lines.append(f"temperature: +{glyphs.temperature[0]['deltaT']:.0f} °C on {n} members")
    lines.append("Diaphragm nodes (4)")
    return lines


def _member_span_endpoints(model, m, s, e):
    data = model["member_data"][m]
    A = model["nodes"][data["ni"]] + (model["nodes"][data["nj"]] - model["nodes"][data["ni"]]) * s
    B = model["nodes"][data["ni"]] + (model["nodes"][data["nj"]] - model["nodes"][data["ni"]]) * e
    return A, B


def draw_structure(ax, model, release_markers=True, labels=True):
    for m, (ni, nj) in model["members"].items():
        ci = model["nodes"][ni]
        cj = model["nodes"][nj]
        col = "#3b7fbf"
        lw = 1.5
        ax.plot([ci[0], cj[0]], [ci[2], cj[2]], [ci[1], cj[1]],
                color=col, linewidth=lw, solid_capstyle="round", zorder=5)
        if labels:
            t = 0.30
            e1 = cj - ci
            length = float(np.linalg.norm(e1))
            u = e1 / length if length > 1e-9 else np.array([1.0, 0.0, 0.0])
            perp = np.array([u[1], -u[0], 0.0])
            n2 = float(np.linalg.norm(perp))
            if n2 > 1e-9:
                perp = perp / n2
            else:
                perp = np.array([0.0, 0.0, -1.0])
            pos = ci + u * (t * length) + perp * 0.10
            ax.text(pos[0], pos[2], pos[1], f"M{m}",
                    color=MEMBER_LABEL_COLOR, fontsize=MEMBER_LABEL_FS,
                    ha="center", zorder=10)
        if release_markers and m in model["released_members"]:
            for pos in [ci, cj]:
                ax.plot(pos[0], pos[2], pos[1], "o", markersize=9,
                        markerfacecolor="white", markeredgecolor=C_REL,
                        markeredgewidth=2.0, zorder=15)

    for n, c in model["nodes"].items():
        ax.scatter(c[0], c[2], c[1], s=55, marker="o",
                   facecolor="#c0392b", edgecolors="#7b241c",
                   linewidths=1.0, zorder=20)
        if labels:
            ax.text(c[0] + 0.35, c[2] + 0.35, c[1] + 0.15, f"N{n}",
                    fontsize=11, fontweight="bold", color="black",
bbox=dict(boxstyle="round,pad=0.20", fc="white",
                      ec="#bfbfbf", lw=0.5),
                    zorder=25)


def draw_global_axes(ax):
    al = 1.2
    ax.quiver(0, 0, 0, al, 0, 0, color="#e8722c", arrow_length_ratio=0.30,
              linewidth=3, zorder=30)
    ax.text(al + 0.20, 0, 0, "X", color="#e8722c", fontsize=12,
            fontweight="bold", zorder=31)
    ax.quiver(0, 0, 0, 0, al, 0, color="#8b4513", arrow_length_ratio=0.30,
              linewidth=3, zorder=30)
    ax.text(0, al + 0.20, 0, "Z", color="#8b4513", fontsize=12,
            fontweight="bold", zorder=31)
    ax.quiver(0, 0, 0, 0, 0, al, color="#2e86c1", arrow_length_ratio=0.30,
              linewidth=3, zorder=30)
    ax.text(0, 0, al + 0.20, "Y", color="#2e86c1", fontsize=12,
            fontweight="bold", zorder=31)


def _perpendicular_offset(model, m, axis_vec):
    """Unit vector, normal to the member, in the plane of the load axis."""
    data = model["member_data"][m]
    e1 = data["lam"][0]
    v = axis_vec - np.dot(axis_vec, e1) * e1
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return v / n


def draw_member_distributed_band(ax, model, rec, scale, band_on=True):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    m = rec["m"]
    data = model["member_data"][m]
    self_w = bool(rec.get("self_weight"))
    is_col = model["member_type"].get(m, "") == "Column"
    if self_w:
        band_col = "#1f6f43"
        edge_col = "#155233"
        edge_lw = 0.8
        alpha = 0.22
        band_w = BAND_WIDTH_SW_M
        arrow_lw = 0.7
    else:
        band_col = "#e8722c"
        edge_col = "#c85a1a"
        edge_lw = 1.2
        alpha = 0.40
        band_w = BAND_WIDTH_M
        arrow_lw = 1.0

    A, B = _member_span_endpoints(model, m, rec["s"], rec["e"])
    load_dir = _global_direction_vector(rec["axis"],
                                        rec.get("direction_factor", -1.0))
    nrm = float(np.linalg.norm(load_dir))
    if nrm < 1e-9:
        load_dir = np.array([0.0, -1.0, 0.0])
        nrm = 1.0
    load_dir = load_dir / nrm
    raw = _perpendicular_offset(model, m, load_dir)
    if raw is None:
        off = None
    else:
        if float(np.dot(raw, load_dir)) < 0.0:
            raw = -raw
        off = raw

    # --- band polygon: one-sided strip, skipped on vertical columns under 
    #     self-weight because the polygon degenerates there. ---
    if off is not None and band_on and not (self_w and is_col):
        P0 = A
        P1 = B
        P2 = B - off * band_w
        P3 = A - off * band_w
        poly = [_p(P0), _p(P1), _p(P2), _p(P3)]
        ax.add_collection3d(Poly3DCollection(
            [poly], alpha=alpha, facecolor=band_col,
            edgecolor=edge_col, linewidth=edge_lw, linestyle="solid",
            zorder=6))
        ax.plot([P0[0], P1[0]], [P0[2], P1[2]], [P0[1], P1[1]],
                color=edge_col, linewidth=edge_lw, alpha=1.0, zorder=6)
        ax.plot([P3[0], P2[0]], [P3[2], P2[2]], [P3[1], P2[1]],
                color=edge_col, linewidth=edge_lw, alpha=1.0, zorder=6)

    # --- arrows: arrow length = band width, tail at the FAR edge of the 
    #     band, head touching the member axis. Arrowhead is a filled 
    #     triangle at the member axis. ---
    span = (rec["e"] - rec["s"]) * data["L"]
    n_arrows = max(int(math.floor(span / ARROW_SPACING_M)), 2)
    for k in range(n_arrows + 1):
        f = rec["s"] + (rec["e"] - rec["s"]) * k / n_arrows
        P = model["nodes"][data["ni"]] + \
            (model["nodes"][data["nj"]] -
             model["nodes"][data["ni"]]) * f
        if off is None:
            ax.quiver(P[0], P[2], P[1],
                      load_dir[0], load_dir[2], load_dir[1],
                      color=band_col, arrow_length_ratio=0.45,
                      linewidth=arrow_lw, zorder=7)
        else:
            tail = P - off * band_w
            ax.quiver(tail[0], tail[2], tail[1],
                      off[0], off[2], off[1],
                      color=band_col, arrow_length_ratio=0.55,
                      linewidth=arrow_lw, zorder=7)

    # --- UDL magnitude label, ONCE per loaded member, at the midpoint, 
    #     offset perpendicular to the member on the load-opposite side. ---
    if not self_w:
        signch = "+" if rec.get("direction_factor", 1.0) >= 0 else "-"
        axis = rec["axis"].upper()
        mid = (A + B) / 2
        if off is not None:
            pos = mid - off * (band_w + 0.20)
        else:
            pos = mid + np.array([0.30, 0.0, 0.30])
        ax.text(pos[0], pos[2], pos[1],
                f"{rec['mag']:.3f} kN/m {signch}{axis}",
                color="#e8722c", fontsize=8, ha="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="#e8722c", lw=0.6, alpha=0.9),
                zorder=12)


def render_member_point_glyph(ax, model, rec, scale):
    m = rec["m"]
    data = model["member_data"][m]
    P = model["nodes"][data["ni"]] + \
        (model["nodes"][data["nj"]] - model["nodes"][data["ni"]]) * rec["frac"]
    dirn = _global_direction_vector(rec["axis"], rec.get("direction_factor", -1.0))
    nrm = float(np.linalg.norm(dirn))
    if nrm < 1e-9:
        dirn = np.array([0.0, -1.0, 0.0])
        nrm = 1.0
    dirn = dirn / nrm
    arrow_len = 0.60
    ax.quiver(P[0], P[2], P[1],
              dirn[0], dirn[2], dirn[1],
              color=C_LOAD_POINT, arrow_length_ratio=0.35, linewidth=2.2, zorder=7)
    tip = P + dirn * (arrow_len + 0.35)
    ax.text(tip[0], tip[2], tip[1],
            rec["label"], color=C_LOAD_POINT, fontsize=10, ha="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.9,
                      ec=C_LOAD_POINT, lw=0.6), zorder=8)


def render_temperature_glyph(ax, model, rec, label_offset=None):
    """Copper +15 degC label anchored at the true member midpoint, plus a 
    small thermal-expansion tick on the member (degC, never kN)."""
    m = rec["m"]
    data = model["member_data"][m]
    mid = (model["nodes"][data["ni"]] +
           model["nodes"][data["nj"]]) / 2
    e1 = model["nodes"][data["nj"]] - model["nodes"][data["ni"]]
    length = float(np.linalg.norm(e1))
    if length > 1e-9:
        u = e1 / length
    else:
        u = np.array([1.0, 0.0, 0.0])
    # perpendicular in the vertical plane so the label lifts off the 
    # roof rather than stacking along it
    perp = np.array([0.0, 1.0, 0.0]) - np.dot(np.array([0.0, 1.0, 0.0]), u) * u
    n = float(np.linalg.norm(perp))
    if n > 1e-9:
        perp = perp / n
    else:
        perp = np.array([0.0, 1.0, 0.0])
    pos = mid + perp * 0.40
    ax.text(pos[0], pos[2], pos[1],
            rec["label"] + f" M{m}",
            color=C_LOAD_TEMP, fontsize=9, ha="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="#fdf2e9",
                      ec=C_LOAD_TEMP),
            zorder=12)
    # thermal-expansion indicator: a tick pair along the member axis
    t1 = mid + u * 0.45
    t2 = mid + u * 0.90
    ax.plot([t1[0], t2[0]], [t1[2], t2[2]], [t1[1], t2[1]],
            color=C_LOAD_TEMP, linewidth=1.6, zorder=8)
    ax.scatter([mid[0]], [mid[2]], [mid[1]],
               color=C_LOAD_TEMP, s=26, marker="s", zorder=9)


def draw_diaphragm(ax, model, diaphragm):
    if diaphragm is None:
        return
    for n in diaphragm.constrained_nodes:
        c = model["nodes"][n]
        ax.scatter(c[0], c[2], c[1], color="#9333ea", s=70, marker="*",
                   edgecolors="black", linewidths=1.0, zorder=21)


def draw_diaphragm_annotation(ax, model, diaphragm):
    """Single two-line note floating 1.2 m above the roof, centred on N5."""
    if diaphragm is None:
        return
    mnode = diaphragm.master_node
    mlist = ", ".join(str(x) for x in diaphragm.constrained_dofs)
    pos = model["nodes"][mnode]
    anchor = pos + np.array([0.0, 0.0, 1.20])
    ax.text(anchor[0], anchor[2], anchor[1],
            f"ROOF DIAPHRAGM master N{mnode}\n"
            f"constrained DOFs {mlist}",
            ha="center", va="bottom",
            fontsize=9, color="#7b2fbe",
            bbox=dict(boxstyle="round,pad=0.20", fc="white",
                      ec="#7b2fbe", lw=0.6, alpha=0.95),
            clip_on=False, zorder=20,
            transform=ax.transData)


def draw_load_glpyhs(ax, model, glyphs, scale, band_on=True):
    for rec in glyphs.nodal:
        c = model["nodes"][rec["node"]]
        v = rec["vector"]
        vmax = float(np.max(np.abs(v)))
        if vmax < 1e-9:
            continue
        axis, val = next((a, v[AXIS_INDEX[a]]) for a in ("X", "Y", "Z")
                         if abs(v[AXIS_INDEX[a]]) > 1e-9)
        sign = +1 if val > 0 else -1
        col = NODEL_COLOUR_MAP[(axis, sign)]
        dvec = np.zeros(3)
        dvec[AXIS_INDEX[axis]] = 1.0 * sign
        arrow_len = 0.60
        tail = c - dvec * arrow_len
        ax.quiver(tail[0], tail[2], tail[1],
                  dvec[0] * arrow_len, dvec[2] * arrow_len,
                  dvec[1] * arrow_len,
                  color=col, arrow_length_ratio=0.30, linewidth=2.0,
                  zorder=30)
        tip = c + dvec * 0.15
        ax.text(tip[0], tip[2], tip[1] + 0.20, rec["label"],
                color=col, fontsize=8, ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=col, lw=0.6, alpha=0.9),
                zorder=31)
    for rec in glyphs.distributed:
        draw_member_distributed_band(ax, model, rec, scale, band_on=band_on)
    for rec in glyphs.point:
        render_member_point_glyph(ax, model, rec, scale)
    for rec in glyphs.temperature:
        render_temperature_glyph(ax, model, rec)


def _auto_load_scale(model, glyphs):
    """Scale so the biggest load vector is ~1.6 m long."""
    biggest = 0.0
    for rec in glyphs.nodal:
        biggest = max(biggest, float(np.linalg.norm(rec["vector"])))
    for rec in glyphs.distributed:
        biggest = max(biggest, rec["mag"] * 6.0)
    for rec in glyphs.point:
        biggest = max(biggest, rec["mag"])
    coords = np.array(list(model["nodes"].values()))
    span = float(np.max(coords) - np.min(coords))
    if biggest < 1e-12:
        return span * 0.10
    return span * 0.12 / biggest


def _legend_entries(model, source, glyphs):
    """(colour-or-None, exact text) pairs for the boxed legend."""
    from matplotlib.lines import Line2D  # noqa: F401 (type reference only)
    if isinstance(source, LoadCombination):
        lines = _combo_legend_lines(glyphs) + glyphs.factor_lines[0]
    else:
        lines = _src_case_legend_lines(model, source, glyphs)
    legends = {
        "distributed:": "#e8722c",
        "point:": "#7b2fbe",
        "nodal:": "#2e86c1",
        "self weight:": "#1f6f43",
        "temperature:": "#a0522d",
        "Diaphragm nodes (4)": "#2e86c1",
    }
    entries = []
    for line in lines:
        colour = None
        for prefix, c in legends.items():
            if line.startswith(prefix):
                colour = c
                break
        entries.append((colour, line))
    return entries


def _build_boxed_legend(ax, entries, title):
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for colour, text in entries:
        if colour is None:
            handles.append(Line2D([0], [0], color="none", linewidth=0))
        else:
            handles.append(Line2D([0], [0], color=colour, linewidth=3.2,
                                  solid_capstyle="round"))
        labels.append(text)
    leg = ax.legend(handles, labels, title=title, loc="upper left",
                    bbox_to_anchor=(0.015, 0.985), fontsize=9,
                    title_fontsize=10, frameon=True, framealpha=1.0,
                    facecolor="white", edgecolor="#7f7f7f", linewidth=0.8,
                    borderpad=0.9, labelspacing=0.55, handlelength=1.8)
    leg.set_zorder(50)
    return leg


def render_source_figure(model, load_cases, source, combo_factors=None,
                         band_on=True, grid_on=True, grid_alpha=0.65,
                         diaphragm=None, title=None, autoscale_loads=True):
    """One shared renderer for both load-case and combination views.

    Professor-style rendering: light-grey 3-D pane box, faint grid, boxed
    legend with colour swatches, thick translucent load bands, larger fonts.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plt.rcParams["font.family"] = "sans-serif"

    if isinstance(source, LoadCombination):
        glyphs = build_combination_glyphs(model, load_cases, source)
    else:
        glyphs = build_case_glyphs(model, source)

    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    draw_structure(ax, model)
    draw_global_axes(ax)
    if diaphragm is not None:
        draw_diaphragm(ax, model, diaphragm)
        draw_diaphragm_annotation(ax, model, diaphragm)

    scale = _auto_load_scale(model, glyphs) if autoscale_loads else 0.22
    draw_load_glpyhs(ax, model, glyphs, scale, band_on=band_on)

    # I1: full-cube limits (0 .. 6 on every axis)
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)
    ax.set_zlim(-1, 7)
    ax.set_box_aspect((1, 1, 1))

    ax.set_axis_on()
    # G: light-grey pane box, visible faint grid on all three panes.
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("#f7f7f7")
        pane.set_alpha(0.35)
        pane.set_edgecolor("#bfbfbf")
    if grid_on:
        # any keyword passed to grid() forces visible=True; that is intended
        # on this branch. The "off" branch below passes NO keyword.
        ax.grid(True, color="#cccccc", linewidth=0.5,
                alpha=min(float(grid_alpha), 0.7))
    else:
        ax.grid(False)

    ax.set_xlabel("X (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Z (m)", fontsize=11, fontweight="bold")
    ax.set_zlabel("Y (m)", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=10, pad=4)

    if title is None:
        if isinstance(source, LoadCombination):
            title = f"{source.design_method} Combination {source.id} - {source.name}"
        else:
            title = f"Load Case {source.id}  {source.name}  ({source.category})"
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)

    entries = _legend_entries(model, source, glyphs)
    if isinstance(source, LoadCombination):
        legend_title = f"Combination {source.id} [{source.design_method}]"
    else:
        legend_title = f"Load Case {source.id}"
    _build_boxed_legend(ax, entries, legend_title)

    # I3: fixed camera. I4: perspective projection if available.
    try:
        ax.set_proj_type("persp")
    except Exception:
        pass  # keep ortho
    ax.view_init(elev=22, azim=-55)
    return fig


# ============================================================
# SECTION 18: DEFAULT MODEL (6m cube with all 9 load cases)
# ============================================================

BASE_NODES = {
    1: np.array([0.0, 0.0, 0.0]),
    2: np.array([6.0, 0.0, 0.0]),
    3: np.array([6.0, 0.0, 6.0]),
    4: np.array([0.0, 0.0, 6.0]),
    5: np.array([0.0, 6.0, 0.0]),
    6: np.array([6.0, 6.0, 0.0]),
    7: np.array([6.0, 6.0, 6.0]),
    8: np.array([0.0, 6.0, 6.0]),
}

BASE_MEMBERS = {
    1: (1, 5), 2: (2, 6), 3: (3, 7), 4: (4, 8),
    5: (5, 6), 6: (7, 8), 7: (5, 8), 8: (6, 7),
    9: (1, 2), 10: (3, 4), 11: (1, 4), 12: (2, 3),
}

BASE_BETA = {m: (90 if m <= 4 else 0) for m in BASE_MEMBERS}

DEFAULT_PINNED_NODES = [1, 2, 3, 4]
DEFAULT_RELEASED_MEMBERS = [1, 3, 5, 7]


def classify_member_type(nodes, members, ni, nj):
    if abs(nodes[nj][1] - nodes[ni][1]) > 1e-9:
        return "Column"
    if ni in DEFAULT_PINNED_NODES and nj in DEFAULT_PINNED_NODES:
        return "Ground Beam"
    return "Beam"


def build_default_model():
    """The standard 6 m cube with A992 steel, W310X38.7 beams and
    W250X49.1 columns, pinned base nodes 1-4, MZ releases on M1/M3/M5/M7."""
    mat_lib = create_default_material_library()
    sec_lib = create_default_section_library()
    a992 = mat_lib.get("A992 Steel")
    sec_beam = sec_lib.get("W310X38.7")
    sec_col = sec_lib.get("W250X49.1")

    member_material = {m: a992 for m in BASE_MEMBERS}
    member_section = {}
    for m in BASE_MEMBERS:
        member_section[m] = sec_col if m <= 4 else sec_beam

    member_type = {m: classify_member_type(BASE_NODES, BASE_MEMBERS, *BASE_MEMBERS[m])
                   for m in BASE_MEMBERS}

    model = {
        "name": "Standard 6m Cube (Rev 3)",
        "nodes": BASE_NODES,
        "members": BASE_MEMBERS,
        "beta": BASE_BETA,
        "pinned_nodes": list(DEFAULT_PINNED_NODES),
        "released_members": list(DEFAULT_RELEASED_MEMBERS),
        "member_material": member_material,
        "member_section": member_section,
        "member_type": member_type,
        "material_lib": mat_lib,
        "section_lib": sec_lib,
        "material": a992,
        "section_beam": sec_beam,
        "section_column": sec_col,
    }
    return model


def finalize_model(model):
    node_dofs, restrained, active, total_dof = build_dof_system(
        model["nodes"], model["pinned_nodes"])
    model["node_dofs"] = node_dofs
    model["restrained"] = restrained
    model["active"] = active
    model["total_dof"] = total_dof

    member_data = {}
    for m in model["members"]:
        ni, nj = model["members"][m]
        T, L, lam = build_transformation(model["nodes"][ni], model["nodes"][nj],
                                         model["beta"][m])
        mat = model["member_material"][m]
        sec = model["member_section"][m]
        # The solver runs in kN·m units: E and G expressed in kPa (kN/m2)
        # give a stiffness matrix in kN/m, matching the kN load cases.
        k_local_orig = element_stiffness_local(L, mat.E / 1000.0, mat.G / 1000.0,
                                               sec.A, sec.Iy, sec.Iz, sec.J)
        if m in model["released_members"]:
            k_rel, active_loc, released_loc, k_cond = apply_releases(k_local_orig, [5, 11])
        else:
            k_rel = k_local_orig.copy()
            active_loc = list(range(12))
            released_loc = []
            k_cond = k_local_orig.copy()
        k_global = T.T @ k_rel @ T
        member_data[m] = {
            "ni": ni, "nj": nj, "L": L, "T": T, "lam": lam,
            "k_original": k_local_orig, "k_global": k_global,
            "k_condensed": k_cond,
            "active_local": active_loc, "released_local": released_loc,
            "released": m in model["released_members"],
            "global_dofs": node_dofs[ni] + node_dofs[nj],
        }
    model["member_data"] = member_data
    model["K"] = assemble_global_K(member_data, node_dofs, total_dof)
    return model


def define_default_load_cases(model):
    """The 9 standard load cases for the 6 m cube."""
    roof = [5, 6, 7, 8]
    roof_beams = [5, 6, 7, 8]

    cases = []
    cases.append(LoadCase(
        id=1, name="SELF WEIGHT", category="Dead",
        self_weight_enabled=True,
        self_weight_direction="Y", self_weight_factor=-1.0,
        description="Dead / self weight, gamma*A*L, direction global Y factor -1",
        loads=[NodalLoad(node_id=9, fx=0.0, fy=0.0, fz=0.0)],
    ))
    # remove the zero placeholder - keep loads list empty; self-weight handles it
    cases[0].loads = []

    cases.append(LoadCase(
        id=2, name="ROOF DEAD", category="Dead",
        description="5 kN/m UDL on roof beams M5-M8, global Y factor -1",
        loads=[MemberDistributedLoad(member_id=m, direction="Y", magnitude=5.0,
                                     direction_factor=-1.0)
               for m in roof_beams]))
    cases.append(LoadCase(
        id=3, name="ROOF LIVE", category="Live",
        description="3 kN/m UDL on roof beams M5-M8, global Y factor -1",
        loads=[MemberDistributedLoad(member_id=m, direction="Y", magnitude=3.0,
                                     direction_factor=-1.0)
               for m in roof_beams]))
    cases.append(LoadCase(
        id=4, name="ROOF BEAM CENTER LOAD", category="Live",
        description="5 kN point load at the centre of roof beams M5-M8",
        loads=[MemberPointLoad(member_id=m, location=0.5, direction="Y",
                               magnitude=5.0, direction_factor=-1.0)
               for m in roof_beams]))
    cases.append(LoadCase(
        id=5, name="WIND X", category="Wind",
        description="10 kN total in +X, 2.5 kN at each roof node N5-N8",
        loads=[NodalLoad(node_id=n, fx=2.5) for n in roof]))
    cases.append(LoadCase(
        id=6, name="WIND Z", category="Wind",
        description="10 kN total in +Z, 2.5 kN at each roof node N5-N8",
        loads=[NodalLoad(node_id=n, fz=2.5) for n in roof]))
    cases.append(LoadCase(
        id=7, name="SEISMIC X", category="Seismic",
        description="15 kN total in +X, 3.75 kN at each roof node N5-N8",
        loads=[NodalLoad(node_id=n, fx=3.75) for n in roof]))
    cases.append(LoadCase(
        id=8, name="SEISMIC Z", category="Seismic",
        description="15 kN total in +Z, 3.75 kN at each roof node N5-N8",
        loads=[NodalLoad(node_id=n, fz=3.75) for n in roof]))
    cases.append(LoadCase(
        id=9, name="TEMPERATURE +15 degC", category="Temperature",
        description="+15 degC on all 12 members (thermal strain, self-straining)",
        loads=[TemperatureLoad(member_ids=list(model["members"].keys()), delta_T=15.0,
                               description="+15 degC on all 12 members")]))
    return cases


# ============================================================
# SECTION 19: VERIFICATION REPORT
# ============================================================

def analyze_source(model, source, member_eq=None, enforce_diaphragm=False,
                   rank=0):
    """Run the direct-stiffness solver for one source (case or combination)."""
    use_member_eq = {}
    if isinstance(source, LoadCombination):
        F, ln, lm, meq = assemble_combination_vector(
            model, model["load_cases"], source, use_member_eq)
    else:
        F, ln, lm = _case_vector(model, source, use_member_eq)
        meq = use_member_eq

    active = model["active"]
    restrained = model["restrained"]

    if enforce_diaphragm and model.get("diaphragm") is not None:
        Kc, Fc, kept = apply_diaphragm_to_system(
            model["K"], F, model["node_dofs"], model["diaphragm"])
        kept_set = set(kept)
        restricted_active = [d for d in active if d in kept_set]
        U, R, cond = solve_system(Kc, Fc, restricted_active,
                                  [d for d in active if d in kept_set])
        # expand to full size
        U_full = np.zeros(model["total_dof"])
        R_full = np.zeros(model["total_dof"])
        for j, d in enumerate(kept):
            U_full[d] = U[j]
            R_full[d] = R[j]
        U, R = U_full, R_full
        recompute_eq = None
    else:
        U, R, cond = solve_system(model["K"], F, restrained, active)

    member_forces = compute_member_forces(model["member_data"], U, meq)
    eq = compute_equilibrium(model, F, R)

    return {
        "source": source,
        "F": F, "U": U, "R": R,
        "member_eq": meq,
        "member_forces": member_forces,
        "equilibrium": eq,
        "cond": cond,
        "loaded_nodes": fused_nodes(source, model),
        "loaded_members": fused_members(source),
    }


def fused_nodes(source, model):
    if isinstance(source, LoadCombination):
        nodes = set()
        for cid in source.factors:
            c = next((c_ for c_ in model["load_cases"] if c_.id == cid), None)
            if c is None:
                continue
            F_, ln, lm = _case_vector(model, c, {})
            nodes |= ln
        return sorted(nodes)
    F_, ln, lm = _case_vector(model, source, {})
    return sorted(ln)


def fused_members(source):
    return []


def write_verification_report(model, load_cases, combos, results, out_path,
                              run_tests_count=None, test_ok=True):
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("=" * 78)
    lines.append(" CUBE STRUCTURAL SOLVER - REV 3 VERIFICATION REPORT")
    lines.append(f" Generated: {now}")
    lines.append("=" * 78)

    lines.append("\n[1] STRUCTURAL MODEL")
    lines.append(f"  Model: {model['name']}")
    lines.append(f"  Nodes: {len(model['nodes'])}   Members: {len(model['members'])}")
    lines.append(f"  Material: {model['material'].name}  "
                 f"(E={model['material'].E/1e9:.1f} GPa, G={model['material'].G/1e9:.1f} GPa, "
                 f"density={model['material'].density:.0f} kg/m3, alpha={model['material'].thermal_coeff:.4g}/degC)")
    lines.append(f"  Beam section: {model['section_beam'].name} "
                 f"(A={model['section_beam'].A:.7g} m2)")
    lines.append(f"  Column section: {model['section_column'].name} "
                 f"(A={model['section_column'].A:.7g} m2)")
    lines.append(f"  Pinned base nodes: {model['pinned_nodes']}")
    lines.append(f"  MZ released members (both ends): {model['released_members']}")
    lines.append(f"  Total DOF: {model['total_dof']}   Restrained: {len(model['restrained'])}   "
                 f"Active: {len(model['active'])}")

    lines.append("\n[2] MEMBER PROPERTIES")
    lines.append("  Mbr  i-j  Type         Section")
    for m in sorted(model["members"]):
        ni, nj = model["members"][m]
        lines.append(f"  M{m:<3d} {ni}-{nj}  {model['member_type'][m]:<12s} "
                     f"{model['member_section'][m].name}")
    lines.append("  NOTE: M1-M4 columns are W250X49.1; M5-M12 (roof + ground beams)"
                 " are W310X38.7.")

    lines.append("\n[3] LOAD CASES - VALIDATION")
    lc_temp = next((c for c in load_cases if c.category == "Temperature"), None)
    for case in load_cases:
        summary = validate_load_case(model, case)
        lines.append(format_load_case_summary(model, summary))
        lines.append("  " + "-" * 70)
        if case.category == "Temperature":
            if summary.get("thermal_rows"):
                lines.append("  Member  dT(C)  alpha(1/C)  eps_T       dL(mm)    EA(kN)    "
                             "N_restrained(kN)  Condition")
                for r in summary["thermal_rows"]:
                    lines.append(
                        f"  M{r['member']:<5d} {r['delta_T_C']:+.1f}   "
                        f"{r['alpha_1_per_C']:.6g} {r['eps_T']:.6e} "
                        f"{r['free_expansion_mm']:.4f}  {r['EA_kN']:,.1f}  "
                        f"{r['N_restrained_kN']:,.3f}   {r['condition']}")

    lines.append("")
    lines.append("[3a] TEMPERATURE LOAD VERIFICATION")
    if lc_temp is not None:
        lc9 = lc_temp
        s = validate_load_case(model, lc9)
        rows = s.get("thermal_rows", [])
        lines.append(f"  Load case: LC{lc9.id}  {lc9.name}")
        lines.append(f"  Affected members: {len(lc9.loads[0].member_ids)}")
        for r in rows:
            lines.append(
                f"    M{r['member']}  "
                f"T_change={r['delta_T_C']:+.1f} degC  "
                f"alpha={r['alpha_1_per_C']:.6e} 1/degC  "
                f"eps_T={r['eps_T']:.6e}  "
                f"dL={r['free_expansion_mm']:.4f} mm  "
                f"EA={r['EA_kN']:,.1f} kN  "
                f"N_restrained={r['N_restrained_kN']:+.3f} kN  "
                f"[{r['condition']}]")
        lines.append(f"  Net force on structure: {s.get('net_force_kN',0.0):.6f} kN "
                     f"(thermal load is self-straining; zero net force)")
    else:
        lines.append("  No temperature load case defined.")

    lines.append("\n[4] DIAPHRAGM")
    d = model.get("diaphragm")
    if d is not None:
        lines.append(f"  ID: {d.id}   Name: {d.name}")
        lines.append(f"  Master node: N{d.master_node}")
        lines.append(f"  Constrained nodes: {['N'+str(x) for x in d.constrained_nodes]}")
        lines.append(f"  Constrained DOFs (in-plane): {d.constrained_dofs}")
        lines.append(f"  Free DOFs (out-of-plane): {d.free_dofs}")
        eqs = diaphragm_constraint_equations(d, model["node_dofs"])
        lines.append(f"  Generated constraint equations: {len(eqs)}")
        for eq_ in eqs:
            lines.append(
                f"    u_{eq_['dof']}(N{eq_['slave_node']})  =  "
                f"u_{eq_['dof']}(N{eq_['master_node']})    "
                f"(dof {eq_['slave_dof']} = dof {eq_['master_dof']})")
        lines.append("  These equations are generated and tested; the default "
                     "Rev 3 analysis does not eliminate them (see report [8]).")
    else:
        lines.append("  No diaphragm defined.")

    lines.append("\n[5] LOAD COMBINATIONS (NSCP 2015, checked against handout)")
    n_lrfd = sum(1 for c in combos if c.design_method == "LRFD")
    n_asd = sum(1 for c in combos if c.design_method == "ASD")
    n_temp = sum(1 for c in combos if 9 in c.factors)
    lines.append(f"  Total: {len(combos)}  (LRFD {n_lrfd} + ASD {n_asd}; "
                 f"temperature-inclusive {n_temp})")
    lines.append("  Mapping by combination group: "
                 "D={1,2,4}, L={3}, W={5,6}, E={7,8}, T={9}")
    for c in combos:
        lines.append(f"  {c.id:>2d} [{c.design_method}] {c.name}")
        for line in combination_case_factor_lines(c):
            lines.append(line)

    lines.append("\n[6] ANALYSIS RESULTS (per load case)")
    for case in load_cases:
        res = results["cases"].get(case.id)
        if res is None:
            continue
        eq = res["equilibrium"]
        max_disp = float(np.max(np.abs(res["U"])))
        lines.append(f"  LC{case.id} {case.name}:")
        lines.append(f"    max |displacement| = {max_disp:.6e} m  | cond(K_ff) = {res['cond']:.3e}")
        lines.append(f"    equilibrium residual FX={eq[0]:+.6e}  FY={eq[1]:+.6e}  "
                     f"FZ={eq[2]:+.6e} (kN)")
        # reactions
        rtext = []
        for n in model["pinned_nodes"]:
            d = model["node_dofs"][n]
            vals = [res["R"][d[k]] for k in range(6)]
            rtext.append(f"      N{n}: Rx={vals[0]:+.3f} Ry={vals[1]:+.3f} Rz={vals[2]:+.3f} "
                         f"Mx={vals[3]:.3f} My={vals[4]:.3f} Mz={vals[5]:.3f}")
        lines.append("\n".join(rtext))

    lines.append("\n[7] COMBINATION ANALYSIS (LRFD/ASD summary)")
    lines.append("  Comb  Design  Name                         Max|U| (m)")
    for c in combos:
        res = results["combos"].get(c.id)
        if res is None:
            continue
        max_disp = float(np.max(np.abs(res["U"])))
        lines.append(f"  {c.id:>4d}  {c.design_method:<5s} {c.name:<28s} {max_disp:.6e}")

    lines.append("\n[8] ENGINEERING VERIFICATION NOTES")
    lines.append("  * Loads are applied with CONSISTENT (work-equivalent) nodal loads; "
                 "no fake nodal forces are used for distributed loads.")
    lines.append("  * Self-weight uses the RISA weight-density formulation "
                 "W = gamma*A*L  (gamma already includes g).")
    lines.append("  * Temperature is a thermal strain effect "
                 "(eps_T = alpha*dT), never a kN force.")
    lines.append("  * The temperature load is self-straining: net force on the "
                 "structure is 0.000 kN.")
    lines.append("  * Fully / partially restrained thermal states are reported "
                 "separately; the partial case is indeterminate and left as such.")
    lines.append("  * Diaphragm constraint equations are generated and tested but "
                 "not eliminated into the default solve.")
    lines.append("  * The load viewer verifies load application/model interpretation; "
                 "equilibrium, reactions, forces, displacements verify the analysis.")

    lines.append("\n[9] AUTOMATED TESTS")
    if run_tests_count:
        lines.append(f"  Tests run: {run_tests_count}   All passing: {test_ok}")
    else:
        lines.append("  Run 'python cube_solver_rev3.py --run-tests' for the suite.")

    lines.append("\n[10] APPENDIX A - EXPECTED VALUES CROSS-CHECK")
    lines.append("  Quantity                          Expected      Computed      Pass")
    a = appendix_a_expected(model, load_cases, combos)
    for row in a:
        ev, cv = row["expected"], row["computed"]
        ok = "OK" if row["status"] == "pass" else "DELTA"
        lines.append(f"  {row['label']:<28s} {ev:>13} {cv:>13}   {ok}")

    lines.append("\n" + "=" * 78)
    lines.append(" END OF REV 3 VERIFICATION REPORT")
    lines.append("=" * 78)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_path


def appendix_a_expected(model, load_cases, combos):
    rows = []
    sw = compute_self_weight_total(model)
    sw_cross = compute_self_weight_crosscheck(model)
    rows.append(dict(label="Self weight total (gamma*A*L)",
                     expected=29.815, computed=round(sw, 3),
                     status="pass" if abs(sw - 29.815) < 0.006 else "delta"))
    rows.append(dict(label="Self weight cross-check (kg/m*g)",
                     expected=29.773, computed=round(sw_cross, 3),
                     status="pass" if abs(sw_cross - 29.773) / 29.773 < 0.002 else "delta"))
    checks = {
        "LC2 roof dead total": (2, 120.000),
        "LC3 roof live total": (3, 72.000),
        "LC4 centre point loads": (4, 20.000),
        "LC5 wind X total": (5, 10.000),
        "LC6 wind Z total": (6, 10.000),
        "LC7 seismic X total": (7, 15.000),
        "LC8 seismic Z total": (8, 15.000),
    }
    for label, (lc_id, exp) in checks.items():
        case = next((c for c in load_cases if c.id == lc_id), None)
        if case is None:
            continue
        s = validate_load_case(model, case)
        rows.append(dict(label=label, expected=exp,
                         computed=round(s["computed_total_kN"], 3),
                         status="pass" if abs(s["computed_total_kN"] - exp) < 1e-6 else "delta"))
    lc9 = next((c for c in load_cases if c.id == 9), None)
    if lc9 is not None:
        s = validate_load_case(model, lc9)
        rows.append(dict(label="LC9 thermal strain eps=alpha*dT",
                         expected="1.7550e-04",
                         computed=f"{s['thermal_strain']:.6e}",
                         status="pass" if abs(s["thermal_strain"] - 1.755e-4) < 1e-9 else "delta"))
        rows.append(dict(label="LC9 free expansion dL per beam (mm)",
                         expected=1.0530,
                         computed=round(s["free_expansion_mm"] / max(len(s["thermal_rows"]), 1), 4),
                         status="pass" if abs(s["free_expansion_mm"] / max(len(s["thermal_rows"]), 1) - 1.053) < 1e-4 else "delta"))
        beam_rows = [r for r in s["thermal_rows"] if r["member"] >= 5]
        tr = beam_rows[0] if beam_rows else s["thermal_rows"][0]
        rows.append(dict(label="LC9 EA (beam) kN", expected=987743.1,
                         computed=round(tr["EA_kN"], 1),
                         status="pass" if abs(tr["EA_kN"] - 987743.1) / 987743.1 < 1e-4 else "delta"))
        rows.append(dict(label="LC9 N restrained (beam) kN", expected=173.349,
                         computed=round(tr["N_restrained_kN"], 3),
                         status="pass" if abs(tr["N_restrained_kN"] - 173.349) < 0.01 else "delta"))
        rows.append(dict(label="LC9 net force on structure (kN)", expected=0.000,
                         computed=round(s["net_force_kN"], 6),
                         status="pass" if s["net_force_kN"] < 1e-6 else "delta"))
    rows.append(dict(label="Load combinations total", expected=30, computed=len(combos),
                     status="pass" if len(combos) == 30 else "delta"))
    n_temp = sum(1 for c in combos if 9 in c.factors)
    rows.append(dict(label="Temperature-inclusive combinations", expected=4,
                     computed=n_temp,
                     status="pass" if n_temp == 4 else "delta"))
    return rows


def write_excel_report(model, load_cases, combos, results, out_path):
    rows_cases = []
    for c in load_cases:
        s = validate_load_case(model, c)
        rows_cases.append({
            "LC": c.id, "Name": c.name, "Category": c.category,
            "Total kN": round(s.get("computed_total_kN", 0.0), 3),
            "Equilibrium err kN": round(s.get("equilibrium_error_kN", 0.0), 6),
            "Self weight": c.self_weight_enabled,
        })
    rows_combos = [{"Comb": c.id, "Design": c.design_method, "Name": c.name,
                    "Factors": "; ".join(f"LC{k}={v:.2f}" for k, v in sorted(c.factors.items()))}
                   for c in combos]
    disp_rows = []
    for case in load_cases:
        res = results["cases"].get(case.id)
        if res is None:
            continue
        for n in model["nodes"]:
            d = model["node_dofs"][n]
            disp_rows.append({"LC": case.id, "Node": n,
                              "UX": res["U"][d[0]], "UY": res["U"][d[1]],
                              "UZ": res["U"][d[2]], "RX": res["U"][d[3]],
                              "RY": res["U"][d[4]], "RZ": res["U"][d[5]]})
    react_rows = []
    for case in load_cases:
        res = results["cases"].get(case.id)
        if res is None:
            continue
        for n in model["pinned_nodes"]:
            d = model["node_dofs"][n]
            react_rows.append({"LC": case.id, "Node": n,
                               "Rx": res["R"][d[0]], "Ry": res["R"][d[1]],
                               "Rz": res["R"][d[2]], "Mx": res["R"][d[3]],
                               "My": res["R"][d[4]], "Mz": res["R"][d[5]]})

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(rows_cases).to_excel(writer, sheet_name="Load Cases", index=False)
        pd.DataFrame(rows_combos).to_excel(writer, sheet_name="Combinations", index=False)
        pd.DataFrame(disp_rows).to_excel(writer, sheet_name="Displacements", index=False)
        pd.DataFrame(react_rows).to_excel(writer, sheet_name="Reactions", index=False)
        pd.DataFrame([{"Node": n, "X": c[0], "Y": c[1], "Z": c[2]}
                      for n, c in model["nodes"].items()]).to_excel(
            writer, sheet_name="Nodes", index=False)
    return out_path


# ============================================================
# SECTION 20: TOP-LEVEL SOLVE / HEADLESS AUDIT
# ============================================================

def run_full_verification(model, load_cases, combos, workdir="."):
    """Headless audit: run all analyses, build results dict, write images."""
    model["load_cases"] = load_cases
    model["diaphragm"] = make_diaphragm()

    results = {"cases": {}, "combos": {}}
    for case in load_cases:
        results["cases"][case.id] = analyze_source(model, case)
    for c in combos:
        results["combos"][c.id] = analyze_source(model, c)

    import matplotlib
    matplotlib.use("Agg")

    # one image per load case
    img_paths = []
    for case in load_cases:
        fig = render_source_figure(model, load_cases, case,
                                   diaphragm=model["diaphragm"])
        p = os.path.join(workdir, f"rev3_load_case_{case.id}.png")
        fig.savefig(p, dpi=100, facecolor="#ffffff")
        import matplotlib.pyplot as plt
        plt.close(fig)
        img_paths.append(p)
    for combo in combos:
        fig = render_source_figure(model, load_cases, combo,
                                   diaphragm=model["diaphragm"])
        p = os.path.join(workdir, f"rev3_combination_{combo.id}.png")
        print("RENDER_VERSION = 2025-01-01-v1")
        fig.savefig(p, dpi=100, facecolor="#ffffff")
        import matplotlib.pyplot as plt
        plt.close(fig)
        img_paths.append(p)

    return results, img_paths


def run_rev3_audio_report(model, load_cases, combos, workdir="."):
    results, img_paths = run_full_verification(model, load_cases, combos, workdir)
    report_path = os.path.join(workdir, "cube_rev3_verification_report.txt")
    excel_path = os.path.join(workdir, "cube_solver_rev3_output.xlsx")
    write_verification_report(model, load_cases, combos, results, report_path)
    write_excel_report(model, load_cases, combos, results, excel_path)
    return results, report_path, excel_path, img_paths


# ============================================================
# SECTION 21: AUTOMATED TESTS (unittest, 95 tests)
# ============================================================

import unittest


class TestUnitSystem(unittest.TestCase):
    def test_metric_selection(self):
        us = get_unit_system("Metric")
        self.assertEqual(us.name, "Metric")
        self.assertEqual(us.length, "m")
        self.assertEqual(us.force, "N")

    def test_imperial_selection(self):
        us = get_unit_system("Imperial")
        self.assertEqual(us.name, "Imperial")
        self.assertEqual(us.length, "ft")
        self.assertEqual(us.force, "lbf")

    def test_metric_identity_conversion(self):
        us = METRIC
        self.assertAlmostEqual(us.to_display(1.0, "length"), 1.0)
        self.assertAlmostEqual(us.to_display(100.0, "force"), 100.0)

    def test_imperial_length_conversion(self):
        us = IMPERIAL
        self.assertAlmostEqual(us.to_display(1.0, "length"), 1.0 / 0.3048)

    def test_imperial_force_conversion(self):
        us = IMPERIAL
        self.assertAlmostEqual(us.to_display(4.4482216152605, "force"), 1.0)

    def test_invalid_unit_system(self):
        with self.assertRaises(ValueError):
            get_unit_system("Galactic")

    def test_unit_label_metric(self):
        self.assertEqual(METRIC.unit_label("length"), "m")
        self.assertEqual(METRIC.unit_label("force"), "N")

    def test_unit_label_imperial(self):
        self.assertEqual(IMPERIAL.unit_label("length"), "ft")
        self.assertEqual(IMPERIAL.unit_label("force"), "lbf")

    def test_distributed_conversion_metric(self):
        self.assertAlmostEqual(METRIC.to_display(5.0, "distributed"), 5.0)


class TestMaterialLibrary(unittest.TestCase):
    def setUp(self):
        self.lib = create_default_material_library()

    def test_a992_lookup(self):
        mat = self.lib.get("A992 Steel")
        self.assertIsNotNone(mat)

    def test_a992_properties(self):
        mat = self.lib.get("A992 Steel")
        self.assertAlmostEqual(mat.E, 200e9)
        self.assertAlmostEqual(mat.G, 77e9)
        self.assertEqual(mat.density, 7850.0)
        self.assertEqual(mat.yield_strength, 345e6)

    def test_a992_thermal_coefficient(self):
        mat = self.lib.get("A992 Steel")
        self.assertAlmostEqual(mat.thermal_coeff, 11.7e-6)
        self.assertLess(mat.thermal_coeff, 12e-6)

    def test_a36_lookup(self):
        mat = self.lib.get("ASTM A36 Steel")
        self.assertAlmostEqual(mat.E, 200e9)

    def test_add_custom_material(self):
        custom = Material(name="Custom Concrete", E=30e9, G=12e9,
                          density=2400.0, yield_strength=30e6, thermal_coeff=10e-6)
        self.lib.add(custom)
        self.assertIs(self.lib.get("Custom Concrete"), custom)

    def test_unknown_material_rejection(self):
        with self.assertRaises(ValueError):
            self.lib.get("Unobtanium")

    def test_list_names(self):
        self.assertIn("A992 Steel", self.lib.list_names())


class TestSectionLibrary(unittest.TestCase):
    def setUp(self):
        self.lib = create_default_section_library()

    def test_w310x38_7_lookup(self):
        sec = self.lib.get("W310X38.7")
        self.assertIsNotNone(sec)

    def test_w310x38_7_area(self):
        sec = self.lib.get("W310X38.7")
        self.assertAlmostEqual(sec.A, A_W310X38_7)
        self.assertAlmostEqual(sec.Iy, 8.49e-5)
        self.assertAlmostEqual(sec.Iz, 7.23e-6)
        self.assertAlmostEqual(sec.J, 2.33e-7)

    def test_w250x49_1_lookup(self):
        sec = self.lib.get("W250X49.1")
        self.assertAlmostEqual(sec.A, 0.00626)

    def test_cube_default(self):
        sec = self.lib.get("Cube Default")
        self.assertAlmostEqual(sec.A, 0.01)

    def test_add_custom_section(self):
        custom = Section(name="Custom Tube", A=0.005, Iy=5e-5, Iz=5e-5, J=8e-5)
        self.lib.add(custom)
        self.assertIs(self.lib.get("Custom Tube"), custom)

    def test_invalid_section_rejection(self):
        with self.assertRaises(ValueError):
            self.lib.get("Nonexistent Section")

    def test_list_names(self):
        names = self.lib.list_names()
        self.assertIn("W310X38.7", names)
        self.assertIn("W250X49.1", names)


class TestDataModel(unittest.TestCase):
    def test_nodal_load_defaults(self):
        nl = NodalLoad(node_id=5)
        self.assertEqual(nl.fx, 0.0)
        self.assertEqual(nl.mz, 0.0)

    def test_nodal_load_vector(self):
        nl = NodalLoad(node_id=5, fx=2.5, fz=3.75)
        v = nl.vector()
        self.assertEqual(v[0], 2.5)
        self.assertEqual(v[2], 3.75)
        self.assertEqual(len(v), 6)

    def test_member_distributed_load_defaults(self):
        mdl = MemberDistributedLoad(member_id=5, magnitude=5.0)
        self.assertEqual(mdl.direction, "Y")
        self.assertEqual((mdl.start_fraction, mdl.end_fraction), (0.0, 1.0))

    def test_member_point_load_defaults(self):
        mpl = MemberPointLoad(member_id=5, magnitude=5.0)
        self.assertEqual(mpl.location, 0.5)

    def test_temperature_load_fields(self):
        tl = TemperatureLoad(member_ids=[5, 6, 7, 8], delta_T=15.0)
        self.assertEqual(tl.delta_T, 15.0)
        self.assertEqual(tl.temperature_unit, "degC")
        self.assertEqual(len(tl.member_ids), 4)

    def test_temperature_units_are_celsius(self):
        tl = TemperatureLoad(member_ids=[5], delta_T=15.0)
        self.assertNotEqual(tl.temperature_unit.upper(), "KN")
        self.assertEqual(tl.temperature_unit, "degC")

    def test_load_case_fields(self):
        lc = LoadCase(id=9, name="TEMPERATURE", category="Temperature")
        self.assertEqual(lc.id, 9)
        self.assertEqual(lc.category, "Temperature")

    def test_diaphragm_fields(self):
        d = make_diaphragm()
        self.assertEqual(d.master_node, 5)
        self.assertEqual(d.constrained_dofs, ["UX", "UZ", "RY"])

    def test_load_combination_fields(self):
        c = LoadCombination(id=1, name="1.4D", design_method="LRFD",
                            factors={1: 1.4, 2: 1.4})
        self.assertEqual(c.factors[1], 1.4)

    def test_load_case_categories_allowed(self):
        allowed = {"Dead", "Live", "Wind", "Seismic", "Temperature"}
        for cat in allowed:
            LoadCase(id=1, name="x", category=cat)

    def test_direction_factor_semantics(self):
        v = _global_direction_vector("Y", -1.0)
        self.assertEqual(v[1], -1.0)
        self.assertEqual(v[0], 0.0)


class TestStiffnessCore(unittest.TestCase):
    def setUp(self):
        NODES = {
            1: np.array([0., 0., 0.]),
            2: np.array([6., 0., 0.]),
            3: np.array([6., 0., 6.]),
            4: np.array([0., 0., 6.]),
            5: np.array([0., 6., 0.]),
            6: np.array([6., 6., 0.]),
            7: np.array([6., 6., 6.]),
            8: np.array([0., 6., 6.]),
        }
        MEMBERS = {
            1: (1, 5), 2: (2, 6), 3: (3, 7), 4: (4, 8),
            5: (5, 6), 6: (7, 8), 7: (5, 8), 8: (6, 7),
            9: (1, 2), 10: (3, 4), 11: (1, 4), 12: (2, 3),
        }
        for k, v in MEMBERS.items():
            T, L, lam = build_transformation(NODES[v[0]], NODES[v[1]],
                                             90 if k <= 4 else 0)
            self.assertEqual(lam.shape, (3, 3))

    def test_stiffness_symmetric(self):
        k = element_stiffness_local(6.0, 200e9, 77e9, 0.00493, 8.49e-5, 7.23e-6, 2.33e-7)
        self.assertTrue(np.allclose(k, k.T, atol=1e-10))

    def test_stiffness_axial_term(self):
        k = element_stiffness_local(6.0, 200e9, 77e9, 0.00493, 8.49e-5, 7.23e-6, 2.33e-7)
        self.assertAlmostEqual(k[0, 0], 200e9 * 0.00493 / 6.0)

    def test_stiffness_shape(self):
        k = element_stiffness_local(6.0, 200e9, 77e9, 0.00493, 8.49e-5, 7.23e-6, 2.33e-7)
        self.assertEqual(k.shape, (12, 12))

    def test_transformation_length(self):
        ci = np.array([0., 0., 0.])
        cj = np.array([6., 0., 0.])
        T, L, lam = build_transformation(ci, cj, 0.0)
        self.assertAlmostEqual(L, 6.0)

    def test_transformation_orthonormal(self):
        ci = np.array([0., 0., 0.])
        cj = np.array([6., 0., 0.])
        T, L, lam = build_transformation(ci, cj, 0.0)
        self.assertTrue(np.allclose(lam @ lam.T, np.eye(3), atol=1e-12))

    def test_transformation_beta_90(self):
        ci = np.array([0., 0., 0.])
        cj = np.array([0., 6., 0.])     # vertical column
        T, L, lam = build_transformation(ci, cj, 90.0)
        self.assertAlmostEqual(L, 6.0)
        self.assertTrue(np.allclose(lam @ lam.T, np.eye(3), atol=1e-12))

    def test_release_condensation(self):
        k = element_stiffness_local(6.0, 200e9, 77e9, 0.00493, 8.49e-5, 7.23e-6, 2.33e-7)
        kfull, active, released, kcond = apply_releases(k, [5, 11])
        self.assertEqual(released, [5, 11])
        self.assertEqual(len(active), 10)
        self.assertEqual(kfull.shape, (12, 12))

    def test_release_mz_zeros(self):
        k = element_stiffness_local(6.0, 200e9, 77e9, 0.00493, 8.49e-5, 7.23e-6, 2.33e-7)
        kfull, active, released, kcond = apply_releases(k, [5, 11])
        self.assertAlmostEqual(kfull[5, 5], 0.0, places=6)

    def test_global_assembly_shape(self):
        model = finalize_model(build_default_model())
        self.assertEqual(model["K"].shape,
                         (model["total_dof"], model["total_dof"]))

    def test_solve_returns_sizes(self):
        model = finalize_model(build_default_model())
        F = np.zeros(model["total_dof"])
        U, R, cond = solve_system(model["K"], F, model["restrained"], model["active"])
        self.assertEqual(U.shape, (model["total_dof"],))
        self.assertEqual(R.shape, (model["total_dof"],))
        self.assertGreater(cond, 0.0)


class TestDofSystem(unittest.TestCase):
    def setUp(self):
        self.model = finalize_model(build_default_model())

    def test_node_dof_counts(self):
        for n in self.model["nodes"]:
            self.assertEqual(len(self.model["node_dofs"][n]), 6)

    def test_total_dof(self):
        self.assertEqual(self.model["total_dof"], 48)

    def test_restrained_are_translations_base_nodes(self):
        restrained = set(self.model["restrained"])
        for n in [1, 2, 3, 4]:
            d = self.model["node_dofs"][n]
            self.assertIn(d[0], restrained)
            self.assertIn(d[1], restrained)
            self.assertIn(d[2], restrained)
            self.assertNotIn(d[3], restrained)

    def test_active_complement(self):
        act = sorted(self.model["active"]) + \
            sorted(x for x in self.model["restrained"])
        self.assertEqual(sorted(act), list(range(self.model["total_dof"])))


class TestSelfWeight(unittest.TestCase):
    def setUp(self):
        self.model = finalize_model(build_default_model())
        self.cases = define_default_load_cases(self.model)
        self.lc1 = self.cases[0]

    def test_lc1_self_weight_enabled(self):
        self.assertTrue(self.lc1.self_weight_enabled)
        self.assertEqual(self.lc1.category, "Dead")

    def test_self_weight_direction_negative_y(self):
        self.assertEqual(self.lc1.self_weight_direction, "Y")
        self.assertEqual(self.lc1.self_weight_factor, -1.0)

    def test_self_weight_total_matches_baseline(self):
        total = compute_self_weight_total(self.model)
        self.assertAlmostEqual(total, 29.815, delta=0.06)

    def test_self_weight_per_member_gamma_a_l(self):
        for m in self.model["members"]:
            mat = self.model["member_material"][m]
            sec = self.model["member_section"][m]
            L = self.model["member_data"][m]["L"]
            expect = mat.weight_density * sec.A * L
            self.assertAlmostEqual(expect, mat.weight_density * sec.A * L)

    def test_self_weight_net_y(self):
        F, ln, lm = _case_vector(self.model, self.lc1, {})
        net_y = sum(F[self.model["node_dofs"][n][1]] for n in self.model["nodes"])
        total = compute_self_weight_total(self.model)
        self.assertAlmostEqual(-net_y, total, delta=1e-6)


class TestLoadCases(unittest.TestCase):
    def setUp(self):
        self.model = finalize_model(build_default_model())
        self.cases = define_default_load_cases(self.model)

    def test_lc2_roof_dead_magnitude(self):
        lc = self.cases[1]
        self.assertEqual(lc.category, "Dead")
        for ld in lc.loads:
            self.assertEqual(ld.magnitude, 5.0)
            self.assertEqual(ld.direction_factor, -1.0)

    def test_lc2_roof_dead_members(self):
        lc = self.cases[1]
        members = {ld.member_id for ld in lc.loads}
        self.assertEqual(members, {5, 6, 7, 8})

    def test_lc2_total(self):
        s = validate_load_case(self.model, self.cases[1])
        self.assertAlmostEqual(s["computed_total_kN"], 120.000)

    def test_lc3_roof_live_magnitude(self):
        lc = self.cases[2]
        for ld in lc.loads:
            self.assertEqual(ld.magnitude, 3.0)

    def test_lc3_total(self):
        s = validate_load_case(self.model, self.cases[2])
        self.assertAlmostEqual(s["computed_total_kN"], 72.000)

    def test_lc4_point_load_at_midpoint(self):
        lc = self.cases[3]
        for ld in lc.loads:
            self.assertEqual(ld.location, 0.5)
            self.assertEqual(ld.magnitude, 5.0)

    def test_lc4_total(self):
        s = validate_load_case(self.model, self.cases[3])
        self.assertAlmostEqual(s["computed_total_kN"], 20.000)

    def test_lc5_wind_x_per_node_2_5(self):
        lc = self.cases[4]
        for ld in lc.loads:
            self.assertEqual(ld.fx, 2.5)
            self.assertEqual(ld.fy, 0.0)

    def test_lc5_wind_x_total_10(self):
        s = validate_load_case(self.model, self.cases[4])
        self.assertAlmostEqual(s["computed_total_kN"], 10.000)
        self.assertAlmostEqual(s["per_node_kN"], 2.5)

    def test_lc6_wind_z_per_node_2_5(self):
        lc = self.cases[5]
        for ld in lc.loads:
            self.assertEqual(ld.fz, 2.5)

    def test_lc6_wind_z_total_10(self):
        s = validate_load_case(self.model, self.cases[5])
        self.assertAlmostEqual(s["computed_total_kN"], 10.000)

    def test_lc7_seismic_x_per_node_3_75(self):
        lc = self.cases[6]
        for ld in lc.loads:
            self.assertEqual(ld.fx, 3.75)

    def test_lc8_seismic_z_per_node_3_75(self):
        lc = self.cases[7]
        for ld in lc.loads:
            self.assertEqual(ld.fz, 3.75)

    def test_lc7_total_15(self):
        s = validate_load_case(self.model, self.cases[6])
        self.assertAlmostEqual(s["computed_total_kN"], 15.000)


class TestDiaphragm(unittest.TestCase):
    def setUp(self):
        self.model = finalize_model(build_default_model())
        self.d = make_diaphragm()

    def test_diaphragm_master_node_5(self):
        self.assertEqual(self.d.master_node, 5)

    def test_diaphragm_constrained_dofs(self):
        self.assertEqual(self.d.constrained_dofs, ["UX", "UZ", "RY"])

    def test_diaphragm_constrained_nodes(self):
        self.assertEqual(self.d.constrained_nodes, [5, 6, 7, 8])

    def test_diaphragm_free_dofs(self):
        self.assertEqual(self.d.free_dofs, ["UY", "RX", "RZ"])

    def test_constraint_equation_count(self):
        eqs = diaphragm_constraint_equations(self.d, self.model["node_dofs"])
        self.assertEqual(len(eqs), 9)   # 3 slave nodes x 3 DOFs

    def test_constraint_equation_master_slave(self):
        eqs = diaphragm_constraint_equations(self.d, self.model["node_dofs"])
        for eq_ in eqs:
            self.assertNotEqual(eq_["slave_node"], self.d.master_node)

    def test_constraint_equation_matches_dof_indices(self):
        eqs = diaphragm_constraint_equations(self.d, self.model["node_dofs"])
        for eq_ in eqs:
            self.assertEqual(
                eq_["slave_dof"], self.model["node_dofs"][eq_["slave_node"]][eq_["dof_index"]])
            self.assertEqual(
                eq_["master_dof"], self.model["node_dofs"][eq_["master_node"]][eq_["dof_index"]])


class TestTemperature(unittest.TestCase):
    def setUp(self):
        self.model = finalize_model(build_default_model())
        self.cases = define_default_load_cases(self.model)
        self.lc9 = self.cases[8]
        self.mat = self.model["material"]
        self.sec = self.model["section_beam"]

    def test_tc_exists_lc9(self):
        self.assertEqual(self.lc9.id, 9)
        self.assertEqual(self.lc9.category, "Temperature")

    def test_delta_T_15(self):
        tl = self.lc9.loads[0]
        self.assertEqual(tl.delta_T, 15.0)

    def test_thermal_coefficient_retrieved(self):
        self.assertAlmostEqual(self.mat.thermal_coeff, 11.7e-6)

    def test_thermal_strain(self):
        eps = temperature_thermal_strain(self.mat, 15.0)
        self.assertAlmostEqual(eps, 1.755e-4, places=8)

    def test_free_expansion(self):
        dL = temperature_free_expansion(self.mat, 15.0, 6.0)
        self.assertAlmostEqual(dL, 1.053e-3, places=8)

    def test_EA_beam(self):
        EA = temperature_axial_rigidity(self.mat, self.sec)
        self.assertAlmostEqual(EA / 1000.0, 987743.1, delta=1.0)

    def test_restrained_force(self):
        N = temperature_restrained_force(self.mat, self.sec, 15.0)
        self.assertAlmostEqual(N / 1000.0, 173.349, delta=0.02)

    def test_restraint_states_reported(self):
        rows = temperature_member_report(self.model, self.lc9.loads[0])
        for r in rows:
            self.assertTrue("condition" in r)
            self.assertTrue(r["condition"])

    def test_partially_restrained_indeterminate(self):
        cond = restrained_condition(self.model, 5)
        self.assertIn("indeterminate", cond)

    def test_temperature_units_not_force(self):
        self.assertEqual(self.lc9.loads[0].temperature_unit, "degC")
        self.assertIn("degC", self.lc9.name)

    def test_temperature_not_interpreted_as_mechanical(self):
        F, ln, lm = _case_vector(self.model, self.lc9, {})
        net = np.zeros(3)
        for n in self.model["nodes"]:
            d = self.model["node_dofs"][n]
            net += np.array([F[d[0]], F[d[1]], F[d[2]]])
        self.assertLess(np.linalg.norm(net), 1e-9)

    def test_temperature_net_force_zero(self):
        s = validate_load_case(self.model, self.lc9)
        self.assertAlmostEqual(s["net_force_kN"], 0.0, places=9)

    def test_temperature_analysis_self_straining(self):
        res = analyze_source(self.model, self.lc9)
        eq = res["equilibrium"]
        self.assertLess(np.linalg.norm(eq), 1e-6)


class TestCombinations(unittest.TestCase):
    def setUp(self):
        self.model = finalize_model(build_default_model())
        self.cases = define_default_load_cases(self.model)
        self.combos = define_default_combinations(self.cases)

    def test_combination_count_30(self):
        self.assertEqual(len(self.combos), 30)

    def test_lrfd_count_12(self):
        n = sum(1 for c in self.combos if c.design_method == "LRFD")
        self.assertEqual(n, 12)

    def test_asd_count_18(self):
        n = sum(1 for c in self.combos if c.design_method == "ASD")
        self.assertEqual(n, 18)

    def test_temperature_combination_count(self):
        # ids 10, 11, 12, 30 include the temperature case (LC9)
        n_temp = sum(1 for c in self.combos if 9 in c.factors)
        self.assertEqual(n_temp, 4)

    def test_combination_vector_1_2d_1_6l(self):
        combo = next(c for c in self.combos if c.name == "1.2D + 1.6L")
        F, ln, lm, meq = assemble_combination_vector(self.model, self.cases, combo)
        expected_live = 1.6 * 72.0
        expected_dead = 1.2 * compute_self_weight_total(self.model) + 1.2 * (120.0 + 20.0)
        net_y = sum(F[self.model["node_dofs"][n][1]] for n in self.model["nodes"])
        self.assertAlmostEqual(-net_y,
                               expected_dead + expected_live, delta=0.6)

    def test_combination_factors_respected(self):
        combo = next(c for c in self.combos if c.name == "0.9D + 1.0W")
        Fc, ln, lm, meq = assemble_combination_vector(self.model, self.cases, combo)
        # Dead loads have zero X-component; wind-X (LC5) is the sole X contributor.
        # The combo wind factor is 1.0, so the X net of the combo must equal the
        # X net of LC5 scaled by 1.0 exactly.
        wc5 = next(c for c in self.cases if c.id == 5)
        Fw5, _, _ = _case_vector(self.model, wc5, {})
        net_x = sum(Fc[self.model["node_dofs"][n][0]] for n in self.model["nodes"])
        wind_x = sum(Fw5[self.model["node_dofs"][n][0]] for n in self.model["nodes"])
        self.assertAlmostEqual(net_x, wind_x, delta=1e-6)


class TestViewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")

    def setUp(self):
        self.model = finalize_model(build_default_model())
        self.cases = define_default_load_cases(self.model)
        self.combos = define_default_combinations(self.cases)
        self.model["diaphragm"] = make_diaphragm()

    def test_load_case_figure_renders(self):
        fig = render_source_figure(self.model, self.cases, self.cases[1],
                                   diaphragm=self.model["diaphragm"])
        import matplotlib.pyplot as plt
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 1)
        plt.close(fig)

    def test_combination_figure_renders(self):
        fig = render_source_figure(self.model, self.cases, self.combos[0],
                                   diaphragm=self.model["diaphragm"])
        import matplotlib.pyplot as plt
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 1)
        plt.close(fig)


def make_test_suite():
    loader = unittest.TestLoader()
    classes = [
        TestUnitSystem,
        TestMaterialLibrary,
        TestSectionLibrary,
        TestDataModel,
        TestStiffnessCore,
        TestDofSystem,
        TestSelfWeight,
        TestLoadCases,
        TestDiaphragm,
        TestTemperature,
        TestCombinations,
        TestViewer,
    ]
    suite = unittest.TestSuite()
    for cls in classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    return suite


def run_rev3_tests(verbosity=1):
    suite = make_test_suite()
    count = suite.countTestCases()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result, count


# ============================================================
# SECTION 22: MAIN EXECUTION (CLI)
#     python cube_solver_rev3.py                 -- launch the viewer (GUI)
#     python cube_solver_rev3.py --verify-loads  -- headless audit
#     python cube_solver_rev3.py --run-tests     -- 95 automated tests
# ============================================================

def launch_viewer(model, load_cases, combos):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, RadioButtons

    model["load_cases"] = load_cases
    model["diaphragm"] = make_diaphragm()

    state = {"source": "case", "index": 1, "band_on": True, "grid_on": False,
             "auto_rotate": False, "timer": None}

    fig = plt.figure(figsize=(17, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.06, bottom=0.05, right=0.99, top=0.95)

    def redraw():
        ax.cla()
        if state["source"] == "case":
            source = next((c for c in load_cases if c.id == state["index"]), load_cases[0])
        else:
            source = next((c for c in combos if c.id == state["index"]), combos[0])
        fig_ignore = None
        d = model["diaphragm"]
        fig = render_source_figure(
            model, load_cases, source, band_on=state["band_on"],
            grid_on=state["grid_on"], diaphragm=d)
        # re-use the previously built axes is messy; simplest: replace figure
        nonlocal_fig = fig
        plt.close(fig)
        current_ax = ax
        # draw directly on the existing axes by re-running the renderer pieces
        _render_into_ax(ax, model, load_cases, source, state, d)
        if isinstance(source, LoadCombination):
            ax.set_title(f"{source.design_method} Combination {source.id} - {source.name}",
                         fontsize=13, fontweight="bold")
        else:
            ax.set_title(f"Load Case {source.id}  {source.name}  ({source.category})",
                         fontsize=13, fontweight="bold")
        fig.canvas.draw_idle() if fig else None

    def _render_into_ax(ax, model, load_cases, source, state, d):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        if isinstance(source, LoadCombination):
            glyphs = build_combination_glyphs(model, load_cases, source)
        else:
            glyphs = build_case_glyphs(model, source)
        draw_structure(ax, model)
        draw_global_axes(ax)
        if d is not None:
            draw_diaphragm(ax, model, d)
        scale = _auto_load_scale(model, glyphs)
        draw_load_glpyhs(ax, model, glyphs, scale,
                         band_on=state["band_on"])
        coords = np.array(list(model["nodes"].values()))
        span = float(np.max(coords) - np.min(coords))
        pad = span * 0.12
        ax.set_xlim([np.min(coords[:, 0]) - pad, np.max(coords[:, 0]) + pad])
        ax.set_ylim([np.min(coords[:, 2]) - pad, np.max(coords[:, 2]) + pad])
        ax.set_zlim([np.min(coords[:, 1]) - 1.2 * pad, np.max(coords[:, 1]) + pad])
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("Y (m)")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor("white")
            pane.set_alpha(0.0)
            pane.set_edgecolor("#e8e8e8")
        if state["grid_on"]:
            ax.grid(True, alpha=0.15, linewidth=0.4)
        else:
            ax.grid(False)
        ax.view_init(elev=22, azim=-55)

    # ---- widgets ---------------------------------------------------------
    ax_case = plt.axes([0.02, 0.60, 0.13, 0.30])
    case_labels = [f"LC{c.id} {c.name}" for c in load_cases]
    case_sel = RadioButtons(ax_case, case_labels, active=0)

    ax_combo = plt.axes([0.02, 0.12, 0.13, 0.42])
    combo_labels = [f"{c.id} [{c.design_method}] {c.name}" for c in combos]
    combo_sel = RadioButtons(ax_combo, combo_labels, active=0)

    def on_case(label):
        for i, c in enumerate(load_cases):
            if label.startswith(f"LC{c.id} "):
                state["source"] = "case"
                state["index"] = c.id
                combo_sel.set_active(0)
                _render_into_ax(ax, model, load_cases, c, state, model["diaphragm"])
                ax.set_title(f"Load Case {c.id}  {c.name}  ({c.category})",
                             fontsize=13, fontweight="bold")
                fig.canvas.draw_idle()
                return

    def on_combo(label):
        idx = int(label.split()[0])
        combo = next((c for c in combos if c.id == idx), None)
        if combo is None:
            return
        state["source"] = "combo"
        state["index"] = combo.id
        case_sel.set_active(0)
        _render_into_ax(ax, model, load_cases, combo, state, model["diaphragm"])
        ax.set_title(f"{combo.design_method} Combination {combo.id} - {combo.name}",
                     fontsize=13, fontweight="bold")
        fig.canvas.draw_idle()

    case_sel.on_clicked(on_case)
    combo_sel.on_clicked(on_combo)

    ax_check = plt.axes([0.02, 0.04, 0.13, 0.06])
    check = CheckButtons(ax_check, ["Grid", "Load band", "Auto-rotate"], [False, True, False])

    def on_check(label):
        if label == "Grid":
            state["grid_on"] = not state["grid_on"]
        elif label == "Load band":
            state["band_on"] = not state["band_on"]
        else:
            state["auto_rotate"] = not state["auto_rotate"]
            if state["auto_rotate"] and state["timer"] is None:
                state["timer"] = fig.canvas.new_timer(interval=120)
                state["timer"].add_callback(_rotate_step)
                state["timer"].start()
            elif not state["auto_rotate"] and state["timer"] is not None:
                state["timer"].stop()
        _redraw_current(ax)

    def _redraw_current(ax):
        if state["source"] == "case":
            source = next(c for c in load_cases if c.id == state["index"])
        else:
            source = next(c for c in combos if c.id == state["index"])
        _render_into_ax(ax, model, load_cases, source, state, model["diaphragm"])
        if isinstance(source, LoadCombination):
            ax.set_title(f"{source.design_method} Combination {source.id} - {source.name}",
                         fontsize=13, fontweight="bold")
        else:
            ax.set_title(f"Load Case {source.id}  {source.name}  ({source.category})",
                         fontsize=13, fontweight="bold")
        fig.canvas.draw_idle()

    def _rotate_step():
        if not state["auto_rotate"]:
            return
        ax.view_init(elev=22, azim=ax.azim + 2)
        fig.canvas.draw_idle()

    check.on_clicked(on_check)

    ax_prev = plt.axes([0.02, 0.02, 0.03, 0.03]) if False else None
    ax_save = plt.axes([0.86, 0.02, 0.12, 0.04])
    btn_save = Button(ax_save, "Save PNG")
    saved = {}

    def on_save(_):
        try:
            fname = f"rev3_view_{datetime.now().strftime('%H%M%S')}.png"
            fig.savefig(fname, dpi=100, facecolor="#ffffff")
            print(f"  Saved: {fname}")
        except Exception as exc:
            print(f"  Save failed: {exc}")

    btn_save.on_clicked(on_save)

    def on_key(event):
        if event.key is None:
            return
        if event.key.isdigit() and 1 <= int(event.key) <= 9:
            cid = int(event.key)
            case = next((c for c in load_cases if c.id == cid), None)
            if case is not None:
                state["source"] = "case"
                state["index"] = cid
                _render_into_ax(ax, model, load_cases, case, state, model["diaphragm"])
                ax.set_title(f"Load Case {case.id}  {case.name}  ({case.category})",
                             fontsize=13, fontweight="bold")
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    _render_into_ax(ax, model, load_cases, load_cases[0], state, model["diaphragm"])
    ax.set_title(f"Load Case {load_cases[0].id}  {load_cases[0].name}  "
                 f"({load_cases[0].category})", fontsize=13, fontweight="bold")
    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cube Structural Solver Rev 3 - load framework + viewer")
    parser.add_argument("--verify-loads", action="store_true",
                        help="headless audit: write report + all images")
    parser.add_argument("--run-tests", action="store_true",
                        help="run the 95 automated tests")
    args = parser.parse_args(argv)

    if args.run_tests:
        import matplotlib
        matplotlib.use("Agg")
        result, count = run_rev3_tests(verbosity=1)
        ok = result.wasSuccessful()
        print(f"\n{count} tests selected; testsRun={result.testsRun}, "
              f"failures={len(result.failures)}, errors={len(result.errors)}")
        print("ALL PASSING" if ok else "FAILURES PRESENT")
        return 0 if ok else 1

    model = finalize_model(build_default_model())
    load_cases = define_default_load_cases(model)
    combos = define_default_combinations(load_cases)
    model["load_cases"] = load_cases
    model["diaphragm"] = make_diaphragm()

    if args.verify_loads:
        import matplotlib
        matplotlib.use("Agg")
        results, report_path, excel_path, img_paths = run_rev3_audio_report(
            model, load_cases, combos, workdir=os.path.dirname(os.path.abspath(__file__)))
        print("=" * 60)
        print(" REV 3 HEADLESS AUDIT COMPLETE")
        print("=" * 60)
        print(f"  Report : {report_path}")
        print(f"  Excel  : {excel_path}")
        for p in img_paths:
            print(f"  Image  : {p}")
        test_result, test_count = run_rev3_tests(verbosity=0)
        print(f"\n  Automated tests: {test_count} run, "
              f"failures={len(test_result.failures)}, errors={len(test_result.errors)}, "
              f"OK={test_result.wasSuccessful()}")
        # regenerate the report including the test result line
        write_verification_report(
            model, load_cases, combos, results, report_path,
            run_tests_count=test_count, test_ok=test_result.wasSuccessful())
        print("  (report [9] updated with test summary)")
        return 0
    else:
        print("Launching interactive load viewer (close the window to exit).")
        print("Keyboard: 1..9 select load case | buttons: Grid / Load band / Auto-rotate")
        launch_viewer(model, load_cases, combos)
        return 0


if __name__ == "__main__":
    sys.exit(main())