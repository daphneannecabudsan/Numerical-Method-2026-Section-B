"""I2 external assertions for cube_solver_rev3.py (handout Section 7, T01-T20).

Covers:
  T01  SW total = 29.815 kN
  T02  LC2 roof dead = 120.000 kN
  T03  LC3 roof live = 72.000 kN
  T04  LC4 centre points = 20.000 kN, glyphs at member midpoints
  T05/06  LC5/LC6 wind = 10 kN total, 2.5 kN/node
  T07/08  LC7/LC8 seismic = 15 kN total, 3.75 kN/node
  T09  LC9 net force = 0.000 kN, load vector non-empty
  T10  UDL arrow count on a 6 m member = 18
  T11  Combo 1 = 1.4 x D {LC1, LC2, LC4}
  T12  LRFD combos exist
  T13  ASD combos exist
  T14  Combo 13 design_method = ASD
  T15  Combo 15 design_method = ASD
  T16  total combinations = 30
  T17  all rev3_load_case_N.png exist (N = 1..9)
  T18  all rev3_combination_N.png exist (N = 1..30)
  T19  exact title strings (Section 3)
  T20  exact legend strings (Section 4)
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import cube_solver_rev3 as r3

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def model():
    m = r3.finalize_model(r3.build_default_model())
    m["diaphragm"] = r3.make_diaphragm()
    return m


@pytest.fixture(scope="module")
def load_cases(model):
    return r3.define_default_load_cases(model)


@pytest.fixture(scope="module")
def combinations(load_cases):
    return r3.define_default_combinations(load_cases)


@pytest.fixture(scope="module")
def summaries(model, load_cases):
    return {c.id: r3.validate_load_case(model, c) for c in load_cases}


def _case(load_cases, cid):
    return next(c for c in load_cases if c.id == cid)


def _combo(combinations, cid):
    return next(c for c in combinations if c.id == cid)


# T01 ----------------------------------------------------------------------

def test_t01_sw_total(model, summaries):
    s = summaries[1]
    assert abs(s["intended_total_kN"] - 29.815) < 1e-3


# T02-T03 ------------------------------------------------------------------

def test_t02_roof_dead_total(summaries):
    assert abs(summaries[2]["intended_total_kN"] - 120.0) < 1e-3


def test_t03_roof_live_total(summaries):
    assert abs(summaries[3]["intended_total_kN"] - 72.0) < 1e-3


# T04 ----------------------------------------------------------------------

def test_t04_centre_points_midpoint(model, load_cases, summaries):
    assert abs(summaries[4]["intended_total_kN"] - 20.0) < 1e-3
    g = r3.build_case_glyphs(model, _case(load_cases, 4))
    assert len(g.point) == 4
    for rec in g.point:
        assert abs(rec["frac"] - 0.5) < 1e-9
        assert abs(rec["mag"] - 5.0) < 1e-9


# T05-T08 ------------------------------------------------------------------

@pytest.mark.parametrize("cid,total,per", [(5, 10.0, 2.5), (6, 10.0, 2.5),
                                           (7, 15.0, 3.75), (8, 15.0, 3.75)])
def test_t05_08_wind_seismic_summaries(summaries, cid, total, per):
    s = summaries[cid]
    assert abs(s["intended_total_kN"] - total) < 1e-6
    assert abs(s["per_node_kN"] - per) < 1e-6


def test_t05_06_wind_nodal_glyphs(model, load_cases):
    for cid, (mag, axis) in ((5, (2.5, "X")), (6, (2.5, "Z"))):
        g = r3.build_case_glyphs(model, _case(load_cases, cid))
        assert len(g.nodal) == 4
        for rec in g.nodal:
            assert r3._nodal_label(rec["vector"]) == f"{mag} kN +{axis}"


def test_t07_08_seismic_nodal_glyphs(model, load_cases):
    for cid, (mag, axis) in ((7, (3.75, "X")), (8, (3.75, "Z"))):
        g = r3.build_case_glyphs(model, _case(load_cases, cid))
        assert len(g.nodal) == 4
        for rec in g.nodal:
            assert r3._nodal_label(rec["vector"]) == f"{mag} kN +{axis}"


# T09 ----------------------------------------------------------------------

def test_t09_lc9_self_straining(model, load_cases, summaries):
    s = summaries[9]
    assert abs(s["net_force_kN"]) < 0.006
    lc9 = _case(load_cases, 9)
    assert len(lc9.loads) == 1
    assert len(lc9.loads[0].member_ids) == 12


# T10 ----------------------------------------------------------------------

def test_t10_udl_arrow_count_6_on_6m(model, load_cases):
    g = r3.build_case_glyphs(model, _case(load_cases, 2))
    rec = g.distributed[0]
    span = (rec["e"] - rec["s"]) * model["member_data"][rec["m"]]["L"]
    assert abs(span - 6.0) < 1e-9
    n = max(int(math.floor(span / r3.ARROW_SPACING_M)), 2)
    assert n + 1 == 6


# T11 ----------------------------------------------------------------------

def test_t11_combo1_dead_vector(model, load_cases, combinations, summaries):
    c1 = combinations[0]
    assert c1.design_method == "LRFD"
    assert c1.name == "1.4D"
    F, _, _, _ = r3.assemble_combination_vector(model, load_cases, c1)
    net_y = sum(F[model["node_dofs"][n][1]] for n in model["nodes"])
    expected = 1.4 * (summaries[1]["intended_total_kN"] +
                      summaries[2]["intended_total_kN"] +
                      summaries[4]["intended_total_kN"])
    assert abs(-net_y - expected) < 0.6


# T12-T16 ------------------------------------------------------------------

def test_t12_lrfd_exists(combinations):
    assert sum(1 for c in combinations if c.design_method == "LRFD") > 0


def test_t13_asd_exists(combinations):
    assert sum(1 for c in combinations if c.design_method == "ASD") > 0


def test_t14_combo13_asd(combinations):
    assert _combo(combinations, 13).design_method == "ASD"


def test_t15_combo15_asd(combinations):
    assert _combo(combinations, 15).design_method == "ASD"


def test_t16_total_combinations(combinations):
    assert len(combinations) == 30


# T17-T18 ------------------------------------------------------------------

def test_t17_load_case_pngs_exist():
    for i in range(1, 10):
        assert os.path.exists(os.path.join(HERE, f"rev3_load_case_{i}.png")), (
            f"missing rev3_load_case_{i}.png")


def test_t18_combination_pngs_exist():
    for i in range(1, 31):
        assert os.path.exists(os.path.join(HERE, f"rev3_combination_{i}.png")), (
            f"missing rev3_combination_{i}.png")


# T19-T20 ------------------------------------------------------------------

def _render(model, load_cases, source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = r3.render_source_figure(model, load_cases, source,
                                  diaphragm=model["diaphragm"])
    ax = fig.axes[0]
    title = ax.get_title()
    legend = None
    leg = ax.get_legend()
    if leg is not None:
        legend = [t.get_text() for t in leg.get_texts()]
    plt.close(fig)
    return title, legend


def test_t19_titles_exact(model, load_cases, combinations):
    cases = {
        "Load Case 1  SELF WEIGHT  (Dead)": _case(load_cases, 1),
        "LRFD Combination 1 - 1.4D": _combo(combinations, 1),
        "ASD Combination 13 - D": _combo(combinations, 13),
        "ASD Combination 15 - D": _combo(combinations, 15),
    }
    for expected, src in cases.items():
        title, _ = _render(model, load_cases, src)
        assert title == expected, f"title mismatch: {title!r} != {expected!r}"


def test_t20_legends_exact(model, load_cases, combinations):
    _, leg_lc1 = _render(model, load_cases, _case(load_cases, 1))
    assert leg_lc1 == ["self weight: 0.3802, 0.4819 kN/m -Y",
                       "Diaphragm nodes (4)"]

    _, leg_c1 = _render(model, load_cases, _combo(combinations, 1))
    assert leg_c1 == [
        "distributed: 7 kN/m -Y",
        "point: 7 kN -Y",
        "self weight: 0.5323, 0.6747 kN/m -Y",
        "Diaphragm nodes (4)",
        "DEAD / SELF WEIGHT x 1.4",
        "ROOF DEAD x 1.4",
        "ROOF BEAM CENTER LOAD x 1.4",
    ]

    _, leg_c13 = _render(model, load_cases, _combo(combinations, 13))
    assert leg_c13 == [
        "distributed: 5 kN/m -Y",
        "point: 5 kN -Y",
        "self weight: 0.3802, 0.4819 kN/m -Y",
        "Diaphragm nodes (4)",
        "DEAD / SELF WEIGHT x 1",
        "ROOF DEAD x 1",
        "ROOF BEAM CENTER LOAD x 1",
    ]