"""
Cube Structural Solver - Interactive Web App (Streamlit)

Run with:
    streamlit run cube_solver_app.py
"""

import io
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "CubeModel"))

from unit_system import get_unit_system
from materials import create_default_library as create_material_library
from sections import create_default_library as create_section_library
from cube_solver_core import solve_cube

# ============================================================
# DEFAULT MODEL (6m x 6m x 6m cube)
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

DOF_LABELS = ["UX", "UY", "UZ", "RX", "RY", "RZ"]
RESULT_LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

# ============================================================
# PAGES
# ============================================================

st.set_page_config(page_title="Cube Structural Solver", layout="wide")

st.title("Cube Structural Solver")
st.caption("Direct Stiffness Method | 3D Frame Analysis | Interactive")

unit_name = st.sidebar.selectbox("Unit system", ["Metric", "Imperial"])
us = get_unit_system(unit_name)

# --- Material ---
mat_lib = create_material_library()
mat_names = mat_lib.list_names()
mat_choice = st.sidebar.selectbox("Material", mat_names + ["Custom..."])

if mat_choice == "Custom...":
    with st.sidebar.expander("Custom material (SI units)", expanded=True):
        E_m = st.number_input("E (Pa)", value=200e9, step=1e9, format="%.0f")
        G_m = st.number_input("G (Pa)", value=77e9, step=1e9, format="%.0f")
        rho_m = st.number_input("Density (kg/m3)", value=7850.0)
        fy_m = st.number_input("Yield strength (Pa)", value=250e6, step=1e6, format="%.0f")
    from materials import Material
    material = Material(name="Custom", E=E_m, G=G_m, density=rho_m,
                        yield_strength=fy_m)
else:
    material = mat_lib.get(mat_choice)

# --- Section ---
sec_lib = create_section_library()
sec_names = sec_lib.list_names()
sec_choice = st.sidebar.selectbox("Section", sec_names + ["Custom..."])

if sec_choice == "Custom...":
    with st.sidebar.expander("Custom section (SI units)", expanded=True):
        A_s = st.number_input("A (m2)", value=0.01, format="%.5f")
        Iy_s = st.number_input("Iy (m4)", value=1e-4, format="%.6f")
        Iz_s = st.number_input("Iz (m4)", value=1e-4, format="%.6f")
        J_s = st.number_input("J (m4)", value=2e-4, format="%.6f")
    from sections import Section
    section = Section(name="Custom", A=A_s, Iy=Iy_s, Iz=Iz_s, J=J_s)
else:
    section = sec_lib.get(sec_choice)

# --- Loads ---
st.sidebar.subheader("Nodal loads")
force_scale = us.force_factor
load_dict = {}
for n in [5, 6, 7, 8]:
    with st.sidebar.expander(f"Node {n} ({us.force})"):
        fx = st.number_input(f"Fx Node {n}", value=0.0, step=1000.0)
        fy = st.number_input(f"Fy Node {n}", value=-50000.0 if n != 5 else -50000.0, step=1000.0)
        fz = st.number_input(f"Fz Node {n}", value=0.0, step=1000.0)
    load_dict[n] = np.array([f for f in (fx, fy, fz)])

# --- Supports & releases ---
all_nodes = list(BASE_NODES.keys())
pinned_nodes = st.sidebar.multiselect(
    "Pinned support nodes (translations fixed)",
    [str(n) for n in all_nodes],
    default=["1", "2", "3", "4"],
)
released_members = st.sidebar.multiselect(
    "Members with MZ end release",
    [str(m) for m in BASE_MEMBERS],
    default=["1", "3", "5", "7"],
)

pinned_nodes = [int(n) for n in pinned_nodes]
released_members = [int(m) for m in released_members]

# ============================================================
# BUILD MODEL & SOLVE
# ============================================================

member_material = {m: material for m in BASE_MEMBERS}
member_section = {m: section for m in BASE_MEMBERS}

loads = {}
for n, f in load_dict.items():
    moments = np.zeros(3)
    loads[n] = np.concatenate([f * force_scale, moments])

try:
    res = solve_cube(
        BASE_NODES, BASE_MEMBERS, BASE_BETA, pinned_nodes,
        released_members, member_material, member_section, loads,
    )
except Exception as e:
    st.error(f"Solver failed: {e}")
    st.stop()

U = res["U"]
R = res["R"]
F = res["F"]
member_forces = res["member_forces"]
member_data = res["member_data"]
node_dofs = res["node_dofs"]
restrained = res["restrained"]
active = res["active"]
total_dof = res["total_dof"]

# ============================================================
# HEADER METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodes", str(len(BASE_NODES)))
c2.metric("Members", str(len(BASE_MEMBERS)))
c3.metric("Total DOF", str(total_dof))
c4.metric("Active DOF", str(len(active)))

c5, c6, c7, c8 = st.columns(4)
c5.metric("Unit system", us.name)
c6.metric("Material", material.name)
c7.metric("Section", section.name)
c8.metric("Pinned nodes", ", ".join(str(n) for n in pinned_nodes))

# ============================================================
# 3D STRUCTURAL DIAGRAM
# ============================================================

def plot_diagram(
    nodes, members, beta, pinned_nodes, released_members,
    member_data, U, R, loads, member_type,
):
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="#f6f5f1")

    C_COL = "#1f5f8b"
    C_BEAM = "#166534"
    C_GROUND = "#c2410c"
    C_PIN = "#b0342f"
    C_REL = "#c2410c"
    C_LOAD = "#d35400"

    poly = [[nodes[1][0], nodes[1][2], nodes[1][1]],
            [nodes[2][0], nodes[2][2], nodes[2][1]],
            [nodes[3][0], nodes[3][2], nodes[3][1]],
            [nodes[4][0], nodes[4][2], nodes[4][1]]]
    ax.add_collection3d(Poly3DCollection([poly], alpha=0.08,
                        facecolor="#aaaaaa", edgecolor="none"))

    max_load = 1.0
    for n, lv in loads.items():
        max_load = max(max_load, float(np.max(np.abs(lv[:3]))))

    coords = np.array([nodes[n] for n in nodes])
    span = float(np.max(coords) - np.min(coords))
    sc = span * 0.18 / max_load

    for m, (ni, nj) in members.items():
        ci = nodes[ni]
        cj = nodes[nj]
        col = C_COL if member_type[m] == "Column" else (
            C_BEAM if member_type[m] == "Beam" else C_GROUND)
        lw = 3.0 if member_type[m] == "Column" else 2.0
        ax.plot([ci[0], cj[0]], [ci[2], cj[2]], [ci[1], cj[1]],
                color=col, linewidth=lw, solid_capstyle="round", zorder=5)

        mid = (ci + cj) / 2
        ax.text(mid[0], mid[2], mid[1] + 0.25, f"M{m}",
                color=col, fontsize=8, ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          alpha=0.85, ec="none"), zorder=10)

        if m in released_members:
            for pos in [ci, cj]:
                ax.plot(pos[0], pos[2], pos[1], "o", markersize=10,
                        markerfacecolor="white", markeredgecolor=C_REL,
                        markeredgewidth=2.2, zorder=15)

    for n, c in nodes.items():
        if n in pinned_nodes:
            ax.scatter(c[0], c[2], c[1], color=C_PIN, s=40,
                       marker="o", edgecolors="black", linewidths=1.5, zorder=20)
        else:
            ax.scatter(c[0], c[2], c[1], color="#16181d", s=30,
                       marker="o", edgecolors="black", linewidths=0.8, zorder=20)
        ax.text(c[0] + 0.35, c[2] + 0.35, c[1] + 0.15, f"{n}",
                fontsize=12, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          alpha=0.92, ec="#aaaaaa"), zorder=25)

    for n, lv in loads.items():
        c = nodes[n]
        fx, fy, fz = lv[:3]
        if abs(fx) > 1e-9:
            ax.quiver(c[0], c[2], c[1], fx * sc, 0, 0, color=C_LOAD,
                      arrow_length_ratio=0.3, linewidth=2.5, zorder=30)
        if abs(fy) > 1e-9:
            ax.quiver(c[0], c[2], c[1], 0, 0, fy * sc, color=C_LOAD,
                      arrow_length_ratio=0.3, linewidth=2.5, zorder=30)
        if abs(fz) > 1e-9:
            ax.quiver(c[0], c[2], c[1], 0, 0, fz * sc, color=C_LOAD,
                      arrow_length_ratio=0.3, linewidth=2.5, zorder=30)

    pad = span * 0.12
    ax.set_xlabel("X (m)", fontsize=11, labelpad=10)
    ax.set_ylabel("Z (m)", fontsize=11, labelpad=10)
    ax.set_zlabel("Y (m)", fontsize=11, labelpad=10)
    ax.set_xlim([np.min(coords[:, 0]) - pad, np.max(coords[:, 0]) + pad])
    ax.set_ylim([np.min(coords[:, 2]) - pad, np.max(coords[:, 2]) + pad])
    ax.set_zlim([np.min(coords[:, 1]) - pad, np.max(coords[:, 1]) + pad])
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.view_init(elev=22, azim=-55)

    ax.set_title("Structural Diagram — 6m x 6m x 6m Cube Frame",
                 fontsize=14, fontweight="bold")
    return fig


member_type = {}
for m, (ni, nj) in BASE_MEMBERS.items():
    dy = abs(BASE_NODES[nj][1] - BASE_NODES[ni][1])
    dx = abs(BASE_NODES[nj][0] - BASE_NODES[ni][0])
    dz = abs(BASE_NODES[nj][2] - BASE_NODES[ni][2])
    if dy > dx and dy > dz:
        member_type[m] = "Column"
    elif ni in pinned_nodes and nj in pinned_nodes:
        member_type[m] = "Ground Beam"
    else:
        member_type[m] = "Beam"

st.subheader("Structural Diagram")
fig_struc = plot_diagram(
    BASE_NODES, BASE_MEMBERS, BASE_BETA, pinned_nodes,
    released_members, member_data, U, R, loads, member_type,
)
st.pyplot(fig_struc, clear_figure=True)

# ============================================================
# DISPLACEMENTS
# ============================================================

st.subheader("Nodal Displacements")

disp_rows = []
for n in BASE_NODES:
    d = node_dofs[n]
    row = {"Node": n}
    for k, lbl in enumerate(DOF_LABELS):
        val = us.to_display(U[d[k]], "length" if k < 3 else "rotation")
        row[lbl + f" ({us.unit_label('length' if k < 3 else 'rotation')})"] = val
    disp_rows.append(row)
df_disp = pd.DataFrame(disp_rows).set_index("Node")
st.dataframe(df_disp)

# ============================================================
# REACTIONS
# ============================================================

st.subheader("Support Reactions")

react_rows = []
for n in BASE_NODES:
    d = node_dofs[n]
    vals = [R[d[k]] for k in range(6)]
    if np.max(np.abs(vals)) < 1e-6:
        continue
    row = {"Node": n}
    for k, lbl in enumerate(RESULT_LABELS):
        qty = "force" if k < 3 else "moment"
        row[lbl + f" ({us.unit_label(qty)})"] = us.to_display(vals[k], qty)
    react_rows.append(row)

if react_rows:
    df_react = pd.DataFrame(react_rows).set_index("Node")
    st.dataframe(df_react)
else:
    st.info("No reactions (no pinned supports selected).")

# --- Equilibrium check ---
sum_fx = sum(R[node_dofs[n][0]] for n in pinned_nodes)
sum_fy = sum(R[node_dofs[n][1]] for n in pinned_nodes)
sum_fz = sum(R[node_dofs[n][2]] for n in pinned_nodes)
app_fx = sum(loads[n][0] for n in loads)
app_fy = sum(loads[n][1] for n in loads)
app_fz = sum(loads[n][2] for n in loads)

e1, e2, e3 = st.columns(3)
e1.metric("ΣFx equilibrium (N)", f"{sum_fx + app_fx:.4e}")
e2.metric("ΣFy equilibrium (N)", f"{sum_fy + app_fy:.4e}")
e3.metric("ΣFz equilibrium (N)", f"{sum_fz + app_fz:.4e}")
st.caption("Values near 0 confirm the structure is in equilibrium.")

# ============================================================
# MEMBER FORCES
# ============================================================

st.subheader("Member End Forces (Global-equivalent local output)")

force_rows = []
for m in sorted(member_forces.keys()):
    f = member_forces[m]
    force_rows.append({"Member": m, "End": "i", **{
        RESULT_LABELS[k] + f" ({us.unit_label('force' if k < 3 else 'moment')})":
            us.to_display(f[k], "force" if k < 3 else "moment") for k in range(6)}})
    force_rows.append({"Member": m, "End": "j", **{
        RESULT_LABELS[k] + f" ({us.unit_label('force' if k < 3 else 'moment')})":
            us.to_display(f[6 + k], "force" if k < 3 else "moment") for k in range(6)}})
df_forces = pd.DataFrame(force_rows)
st.dataframe(df_forces)

# ============================================================
# EXPORT
# ============================================================

st.subheader("Export Results")

excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    df_disp.to_excel(writer, sheet_name="Displacements")
    if react_rows:
        df_react.to_excel(writer, sheet_name="Reactions")
    df_forces.to_excel(writer, sheet_name="Member Forces")
excel_buf.seek(0)

st.download_button(
    "Download results (.xlsx)",
    data=excel_buf,
    file_name="cube_solver_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)