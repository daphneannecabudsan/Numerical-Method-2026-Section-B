from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    name: str
    A: float
    Iy: float
    Iz: float
    J: float


class SectionLibrary:
    def __init__(self):
        self._sections: dict[str, Section] = {}

    def add(self, section: Section):
        self._sections[section.name] = section

    def get(self, name: str) -> Section:
        if name not in self._sections:
            raise ValueError(f"Unknown section: '{name}'. Available: {list(self._sections.keys())}")
        return self._sections[name]

    def list_names(self) -> list[str]:
        return list(self._sections.keys())


def create_default_library() -> SectionLibrary:
    lib = SectionLibrary()
    lib.add(Section(name="Cube Default", A=0.01, Iy=1.0e-4, Iz=1.0e-4, J=2.0e-4))
    lib.add(Section(name="W10x49", A=0.00929, Iy=1.71e-4, Iz=2.72e-5, J=2.33e-7))
    lib.add(Section(name="HSS6x6x3/8", A=0.00406, Iy=3.16e-5, Iz=3.16e-5, J=5.22e-5))
    return lib
