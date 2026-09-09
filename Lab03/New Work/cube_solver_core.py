"""
Cube Structural Solver - Core (Repackaged for interactive use)
Direct Stiffness Method | 3D Frame Analysis

Same math as "the cube structural model", but parameterized so the
solver can be driven from Streamlit or any other caller.
"""

import numpy as np


# ============================================================
# SECTION 1: DOF ASSIGNMENT
# ============================================================

def build_dof_system(nodes, pinned_nodes):
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
# SECTION 2: LOCAL STIFFNESS MATRIX (12x12)
# ============================================================

def element_stiffness_local(member_length, E, G, A, Iy, Iz, J):
    L = member_length
    k = np.zeros((12, 12))

    a = E * A / L
    t = G * J / L

    s1 = 12.0 * E * Iz / L**3
    s2 =  6.0 * E * Iz / L**2
    s3 =  4.0 * E * Iz / L
    s4 =  2.0 * E * Iz / L

    q1 = 12.0 * E * Iy / L**3
    q2 =  6.0 * E * Iy / L**2
    q3 =  4.0 * E * Iy / L
    q4 =  2.0 * E * Iy / L

    k[0, 0] =  a;  k[0, 6] = -a
    k[6, 0] = -a;  k[6, 6] =  a
    k[3, 3] =  t;  k[3, 9] = -t
    k[9, 3] = -t;  k[9, 9] =  t

    k[1, 1] = s1;  k[1, 5]  = s2;   k[1, 7]  = -s1;  k[1, 11]  = s2
    k[5, 1] = s2;  k[5, 5]  = s3;   k[5, 7]  = -s2;  k[5, 11]  = s4
    k[7, 1] = -s1; k[7, 5]  = -s2;  k[7, 7]  =  s1;  k[7, 11]  = -s2
    k[11, 1] = s2; k[11, 5] = s4;   k[11, 7] = -s2;  k[11, 11] = s3

    k[2, 2]  = q1;  k[2, 4]  = -q2;  k[2, 8]  = -q1;  k[2, 10]  = -q2
    k[4, 2]  = -q2; k[4, 4]  =  q3;  k[4, 8]  =  q2;  k[4, 10]  =  q4
    k[8, 2]  = -q1; k[8, 4]  =  q2;  k[8, 8]  =  q1;  k[8, 10]  =  q2
    k[10, 2] = -q2; k[10, 4] =  q4;  k[10, 8] =  q2;  k[10, 10] =  q3

    return k


# ============================================================
# SECTION 3: TRANSFORMATION MATRIX (with Beta Angle)
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

    lam = np.array([e1, e2, e3])
    T = np.zeros((12, 12))
    T[0:3, 0:3] = lam;  T[3:6, 3:6] = lam
    T[6:9, 6:9] = lam;  T[9:12, 9:12] = lam

    return T, L, lam


# ============================================================
# SECTION 4: MEMBER END RELEASES (Static Condensation)
# ============================================================

def apply_releases(k_local, released_local_indices):
    all_dofs = list(range(12))
    active   = sorted([d for d in all_dofs if d not in released_local_indices])
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
# SECTION 5: GLOBAL ASSEMBLY
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
# SECTION 6: SOLVER
# ============================================================

def solve_system(K, F, restrained, active):
    K_ff = K[np.ix_(active, active)]
    F_f  = F[active]
    U_f = np.linalg.solve(K_ff, F_f)
    U = np.zeros(K.shape[0])
    U[active] = U_f
    R = K @ U - F
    return U, R


# ============================================================
# SECTION 7: POST-PROCESSING
# ============================================================

def compute_member_forces(member_data, U):
    results = {}
    for m, data in member_data.items():
        u_global = U[data["global_dofs"]]
        u_local  = data["T"] @ u_global
        if data["released"]:
            u_a = u_local[data["active_local"]]
            f_a = data["k_condensed"] @ u_a
            f_local = np.zeros(12)
            for i, di in enumerate(data["active_local"]):
                f_local[di] = f_a[i]
        else:
            f_local = data["k_original"] @ u_local
        results[m] = f_local
    return results


# ============================================================
# SECTION 8: TOP-LEVEL SOLVE
# ============================================================

def solve_cube(nodes, members, beta, pinned_nodes, released_members,
               member_material, member_section, loads):
    """Assemble and solve the full 3D frame model.

    Parameters
    ----------
    nodes : dict[int, np.ndarray]    node id -> [x, y, z] (m)
    members : dict[int, tuple[int,int]]  member id -> (i_node, j_node)
    beta : dict[int, float]          member id -> beta angle (deg)
    pinned_nodes : list[int]         nodes with pinned supports
    released_members : list[int]     members with MZ release at both ends
    member_material : dict[int, object]  member id -> Material (E, G, ...)
    member_section : dict[int, object]   member id -> Section (A, Iy, Iz, J)
    loads : dict[int, np.ndarray]    node id -> 6-component force/moment SI

    Returns
    -------
    dict with U, R, member_forces, member_data, node_dofs, restrained,
    active, K, total_dof, condition_number
    """
    node_dofs, restrained, active, total_dof = build_dof_system(nodes, pinned_nodes)

    member_data = {}
    for m in members:
        ni, nj = members[m]
        T, L, lam = build_transformation(nodes[ni], nodes[nj], beta[m])
        mat = member_material[m]
        sec = member_section[m]
        k_local_orig = element_stiffness_local(L, mat.E, mat.G,
                                               sec.A, sec.Iy, sec.Iz, sec.J)

        if m in released_members:
            k_rel, active_loc, released_loc, k_cond = apply_releases(
                k_local_orig, [5, 11])
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
            "released": m in released_members,
            "global_dofs": node_dofs[ni] + node_dofs[nj],
        }

    K = assemble_global_K(member_data, node_dofs, total_dof)

    F = np.zeros(total_dof)
    for n, lv in loads.items():
        dofs = node_dofs[n]
        for k in range(6):
            F[dofs[k]] += lv[k]

    U, R = solve_system(K, F, restrained, active)
    member_forces = compute_member_forces(member_data, U)

    return {
        "U": U, "R": R, "K": K, "F": F,
        "member_forces": member_forces, "member_data": member_data,
        "node_dofs": node_dofs, "restrained": restrained,
        "active": active, "total_dof": total_dof,
    }