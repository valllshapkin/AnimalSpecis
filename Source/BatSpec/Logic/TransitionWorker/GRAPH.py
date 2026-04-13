from typing import Any

from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Python.Graph import GraphNode, get_shared_cache
from numpy.typing import NDArray

from . import etalonNoise as _etalonNoise

def etalonNoise(spec: GraphNode['SpecFunc'], percentile: float) -> GraphNode[NDArray[Any]]:
    return spec.cache.get_or_create_node(
        f"etalonNoise({spec.key}, {percentile})", 
        lambda: _etalonNoise(spec.value, percentile)
    )



