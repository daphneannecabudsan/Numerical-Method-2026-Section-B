from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    E: float
    G: float
    density: float
    yield_strength: float


class MaterialLibrary:
    def __init__(self):
        self._materials: dict[str, Material] = {}

    def add(self, material: Material):
        self._materials[material.name] = material

    def get(self, name: str) -> Material:
        if name not in self._materials:
            raise ValueError(f"Unknown material: '{name}'. Available: {list(self._materials.keys())}")
        return self._materials[name]

    def list_names(self) -> list[str]:
        return list(self._materials.keys())


def create_default_library() -> MaterialLibrary:
    lib = MaterialLibrary()
    lib.add(Material(
        name="ASTM A36 Steel",
        E=200e9,
        G=77e9,
        density=7850.0,
        yield_strength=250e6,
    ))
    return lib
