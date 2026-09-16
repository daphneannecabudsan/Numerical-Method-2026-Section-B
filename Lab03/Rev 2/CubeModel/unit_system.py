from dataclasses import dataclass


@dataclass(frozen=True)
class UnitSystem:
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
        return value_si

    def unit_label(self, quantity: str) -> str:
        labels = {
            "length": self.length,
            "force": self.force,
            "stress": self.stress,
            "moment": self.moment,
            "rotation": self.rotation,
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


def get_unit_system(name: str) -> UnitSystem:
    key = name.strip().capitalize()
    if key not in _SYSTEMS:
        raise ValueError(f"Unknown unit system: '{name}'. Choose from: {list(_SYSTEMS.keys())}")
    return _SYSTEMS[key]
