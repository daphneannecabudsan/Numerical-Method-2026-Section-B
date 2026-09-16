import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CubeModel"))

from unit_system import UnitSystem, get_unit_system, METRIC, IMPERIAL
from materials import Material, MaterialLibrary, create_default_library
from sections import Section, SectionLibrary, create_default_library as create_section_library


# ============================================================
# UNIT TESTS
# ============================================================

class TestUnitSystem:
    def test_metric_selection(self):
        us = get_unit_system("Metric")
        assert us.name == "Metric"
        assert us.length == "m"
        assert us.force == "N"

    def test_imperial_selection(self):
        us = get_unit_system("Imperial")
        assert us.name == "Imperial"
        assert us.length == "ft"
        assert us.force == "lbf"

    def test_metric_identity_conversion(self):
        us = METRIC
        assert us.to_display(1.0, "length") == pytest.approx(1.0)
        assert us.to_display(100.0, "force") == pytest.approx(100.0)

    def test_imperial_length_conversion(self):
        us = IMPERIAL
        assert us.to_display(1.0, "length") == pytest.approx(1.0 / 0.3048)

    def test_imperial_force_conversion(self):
        us = IMPERIAL
        assert us.to_display(4.4482216152605, "force") == pytest.approx(1.0)

    def test_invalid_unit_system(self):
        with pytest.raises(ValueError, match="Unknown unit system"):
            get_unit_system("Galactic")

    def test_unit_label_metric(self):
        assert METRIC.unit_label("length") == "m"
        assert METRIC.unit_label("force") == "N"

    def test_unit_label_imperial(self):
        assert IMPERIAL.unit_label("length") == "ft"
        assert IMPERIAL.unit_label("force") == "lbf"


# ============================================================
# MATERIAL TESTS
# ============================================================

class TestMaterialLibrary:
    def setup_method(self):
        self.lib = create_default_library()

    def test_a36_lookup(self):
        mat = self.lib.get("ASTM A36 Steel")
        assert mat.name == "ASTM A36 Steel"
        assert mat.E == pytest.approx(200e9)
        assert mat.G == pytest.approx(77e9)

    def test_a36_density(self):
        mat = self.lib.get("ASTM A36 Steel")
        assert mat.density == pytest.approx(7850.0)

    def test_a36_yield(self):
        mat = self.lib.get("ASTM A36 Steel")
        assert mat.yield_strength == pytest.approx(250e6)

    def test_add_custom_material(self):
        custom = Material(name="Custom Concrete", E=30e9, G=12e9, density=2400.0, yield_strength=30e6)
        self.lib.add(custom)
        assert self.lib.get("Custom Concrete") is custom

    def test_unknown_material_rejection(self):
        with pytest.raises(ValueError, match="Unknown material"):
            self.lib.get("Unobtanium")

    def test_list_names(self):
        names = self.lib.list_names()
        assert "ASTM A36 Steel" in names


# ============================================================
# SECTION TESTS
# ============================================================

class TestSectionLibrary:
    def setup_method(self):
        self.lib = create_section_library()

    def test_default_section_lookup(self):
        sec = self.lib.get("Cube Default")
        assert sec.A == pytest.approx(0.01)
        assert sec.Iy == pytest.approx(1.0e-4)

    def test_w10x49_lookup(self):
        sec = self.lib.get("W10x49")
        assert sec.A == pytest.approx(0.00929)

    def test_add_custom_section(self):
        custom = Section(name="Custom Tube", A=0.005, Iy=5e-5, Iz=5e-5, J=8e-5)
        self.lib.add(custom)
        assert self.lib.get("Custom Tube") is custom

    def test_invalid_section_rejection(self):
        with pytest.raises(ValueError, match="Unknown section"):
            self.lib.get("Nonexistent Section")

    def test_list_names(self):
        names = self.lib.list_names()
        assert "Cube Default" in names
        assert "W10x49" in names


# ============================================================
# STRUCTURAL MODEL TEST
# ============================================================

class TestStructuralModel:
    def setup_method(self):
        from unit_system import get_unit_system
        from materials import create_default_library
        from sections import create_default_library as create_sec_lib

        self.unit_system = get_unit_system("Metric")
        self.mat_lib = create_default_library()
        self.sec_lib = create_sec_lib()
        self.a36 = self.mat_lib.get("ASTM A36 Steel")
        self.default_sec = self.sec_lib.get("Cube Default")

    def test_model_construction(self):
        NODES = {
            1: np.array([0.0, 0.0, 0.0]),
            2: np.array([6.0, 0.0, 0.0]),
            3: np.array([6.0, 0.0, 6.0]),
            4: np.array([0.0, 0.0, 6.0]),
            5: np.array([0.0, 6.0, 0.0]),
            6: np.array([6.0, 6.0, 0.0]),
            7: np.array([6.0, 6.0, 6.0]),
            8: np.array([0.0, 6.0, 6.0]),
        }
        MEMBERS = {
            1: (1, 5), 2: (2, 6), 3: (3, 7), 4: (4, 8),
            5: (5, 6), 6: (7, 8), 7: (5, 8), 8: (6, 7),
            9: (1, 2), 10: (3, 4), 11: (1, 4), 12: (2, 3),
        }
        assert len(NODES) == 8
        assert len(MEMBERS) == 12

    def test_element_stiffness_with_params(self):
        E, G, A, Iy, Iz, J = 200e9, 77e9, 0.01, 1.0e-4, 1.0e-4, 2.0e-4
        L = 6.0
        k = np.zeros((12, 12))
        a = E * A / L
        t = G * J / L
        s1 = 12.0 * E * Iz / L**3
        s2 = 6.0 * E * Iz / L**2
        s3 = 4.0 * E * Iz / L
        s4 = 2.0 * E * Iz / L
        q1 = 12.0 * E * Iy / L**3
        q2 = 6.0 * E * Iy / L**2
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
        k[10, 2] = -q2; k[10, 4] = q4;  k[10, 8] = q2;  k[10, 10] = q3
        assert k.shape == (12, 12)
        assert np.allclose(k, k.T, atol=1e-10)
        assert k[0, 0] == pytest.approx(E * A / L)

    def test_member_material_assignment(self):
        MEMBERS = {1: (1, 5), 2: (2, 6)}
        MEMBER_MATERIAL = {m: self.a36 for m in MEMBERS}
        for m in MEMBERS:
            assert MEMBER_MATERIAL[m].E == pytest.approx(200e9)

    def test_member_section_assignment(self):
        MEMBERS = {1: (1, 5), 2: (2, 6)}
        MEMBER_SECTION = {m: self.default_sec for m in MEMBERS}
        for m in MEMBERS:
            assert MEMBER_SECTION[m].A == pytest.approx(0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
