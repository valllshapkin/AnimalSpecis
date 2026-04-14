import pint
from functools import reduce
import operator

type PintUnit = pint.Unit
UREG = pint.UnitRegistry[float]()

def unit_mul(*units: PintUnit) -> PintUnit:
    return reduce(operator.mul, units)


# Определяем новую базовую размерность для уровня цифрового сигнала
UREG.define('[digital_level] = 1') 
# Определяем саму единицу измерения "Full Scale"
UREG.define('full_scale = [digital_level] = FS')
# Теперь UREG.FS - это полноценная единица измерения!