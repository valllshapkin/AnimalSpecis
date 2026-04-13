from typing import Protocol

class SaveIntegralProtocol(Protocol):
    def SaveIntegralEnergy(self) -> float: ...
    def SaveIntegralArea(self) -> float: ...

def SaveIntegralEnergy(f: SaveIntegralProtocol) -> float:
    return f.SaveIntegralEnergy()
    
def SaveIntegralArea(f: SaveIntegralProtocol) -> float:
    return f.SaveIntegralArea()

