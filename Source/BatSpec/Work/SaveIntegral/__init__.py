from typing import Protocol, Tuple

from BatSpec.Work.Units import PintUnit

class SaveIntegralProtocol(Protocol):
    def SaveIntegralEnergy(self) -> Tuple[PintUnit, float]: ...
    def SaveIntegralArea(self) -> Tuple[PintUnit, float]: ...

def SaveIntegralEnergy(f: SaveIntegralProtocol) -> Tuple[PintUnit, float]:
    return f.SaveIntegralEnergy()
    
def SaveIntegralArea(f: SaveIntegralProtocol) -> Tuple[PintUnit, float]:
    return f.SaveIntegralArea()

