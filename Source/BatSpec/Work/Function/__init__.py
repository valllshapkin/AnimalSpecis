from pathlib import Path
from typing import Any, Callable, Self, Tuple, Union
from jaxtyping import Shaped, Array
from BatSpec.Work.SaveIntegral import SaveIntegralProtocol
from BatSpec.Work.Units import UREG, PintUnit, unit_mul
from BatSpec.Work.Context import fw
import numpy as np

type AnyArray = Union[np.ndarray, Array]

class Function(SaveIntegralProtocol): pass

class Function1D(Function):
    _value_a: AnyArray
    _value_u: PintUnit
    _axis__a: AnyArray
    _axis__u: PintUnit

    def __init__(self, _value_a: AnyArray, _value_u: PintUnit, _axis__a: AnyArray, _axis__u: PintUnit):
        self._value_a, self._value_u, self._axis__a, self._axis__u = _value_a, _value_u, _axis__a, _axis__u
    
    def saveNPZ(self, path: Union[str, Path]):
        import numpy as np
        if not isinstance(self._value_a, np.ndarray) or not isinstance(self._axis__a, np.ndarray): 
            raise RuntimeError("Arrays must be numpy ndarrays before saving.")
            
        np.savez_compressed(path,
            value_a = self._value_a,
            axis__a = self._axis__a,
            value_u = str(self._value_u),
            axis__u = str(self._axis__u),
        )

    @classmethod
    def loadNPZ(cls, path: Union[str, Path]):
        import numpy as np
        with np.load(path) as data:
            return cls(
                _value_a = data["value_a"],
                _value_u = UREG.Unit(str(data["value_u"])),
                _axis__a = data["axis__a"],     
                _axis__u = UREG.Unit(str(data["axis__u"])) 
            )

    @property
    def _d_axis__a(self) -> Tuple[PintUnit, float]:
        import numpy as np
        if len(self._axis__a) < 2:
            raise ValueError("Cannot calculate step size for an AnyArray with less than 2 elements.")
        step = np.mean(np.diff(self._axis__a))
        return (self._axis__u, float(step))

    def SaveIntegralEnergy(self) -> Tuple[PintUnit, float]:
        du, da = self._d_axis__a
        return unit_mul(self._value_u, self._value_u, du), float(fw().sum(self._value_a**2) * da)

    def SaveIntegralArea(self) -> Tuple[PintUnit, float]:
        du, da = self._d_axis__a
        return unit_mul(self._value_u, du), float(fw().sum(self._value_a) * da)

class TimeFunc(Function1D):

    _axis__u: PintUnit = UREG.second

    def __init__(self, values: AnyArray, axis: AnyArray, unit_values: PintUnit):
        if len(values) != len(axis): raise ValueError("values and axis must have the same length")
        self._value_a = values
        self._axis__a = axis
        self._value_u = unit_values

    @classmethod
    def from_Function1D(cls, func: Function1D) -> Self:
        return cls(func._value_a, func._axis__a, func._value_u)
    
    @classmethod
    def from_zeros(cls, unit_values: PintUnit, dtype: Any, shape_t: int = 10):
        values = fw().zeros(shape_t, dtype=dtype)
        axis = fw().arange(shape_t)
        return cls(values=values, axis=axis, unit_values=unit_values)
    
    @property
    def time(self) -> Tuple[PintUnit, AnyArray]:
        return self._axis__u, self._axis__a
    
    @time.setter
    def time(self, value: AnyArray):
        if len(value) != len(self._axis__a):
            raise ValueError(f"New time AnyArray must have length {len(self._axis__a)}, got {len(value)}")
        self._axis__a = value

    @property
    def start(self) -> float:
        return float(self._axis__a[0])
    
    @property
    def end(self) -> float:
        return float(self._axis__a[-1])
    
    @property
    def values(self) -> Tuple[PintUnit, AnyArray]:
        return self._value_u, self._value_a

    @values.setter
    def values(self, value: AnyArray):
        if len(value) != len(self._axis__a):
            raise ValueError(f"New values must have length {len(self._axis__a)}, got {len(value)}")
        self._value_a = value
        
    @property
    def dt(self) -> Tuple[PintUnit, float]:
        return self._d_axis__a

    @property
    def sr(self) -> int:
        _, dt = self.dt
        return round(1.0 / dt)

    def saveNPZ(self, path: Union[str, Path]):
        Function1D.saveNPZ(self, path)
        
    @classmethod
    def loadNPZ(cls, path: Union[str, Path]):
        return cls.from_Function1D(Function1D.loadNPZ(path))

class FreqFunc(Function1D):
    """
    Представляет одномерную функцию в частотной области (спектр).
    """

    _axis__u: PintUnit = UREG.hertz  # Единица измерения оси по умолчанию - Герцы

    def __init__(self, values: AnyArray, axis: AnyArray, unit_values: PintUnit):
        """
        Инициализирует FreqFunc.
        
        :param values: Массив значений (амплитуд) спектра.
        :param axis: Массив частот, соответствующий значениям.
        :param unit_values: Единица измерения значений (например, Па/Гц).
        """
        if len(values) != len(axis):
            raise ValueError("values и axis должны иметь одинаковую длину")
        self._value_a = values
        self._axis__a = axis
        self._value_u = unit_values

    @classmethod
    def from_Function1D(cls, func: Function1D) -> Self:
        """Создает экземпляр FreqFunc из базового Function1D."""
        if not func._axis__u.is_compatible_with(UREG.hertz):
            # Можно добавить предупреждение, если единицы оси не Герцы
            # import warnings
            # warnings.warn(f"Creating FreqFunc from Function1D with non-frequency axis unit: {func._axis__u}")
            pass
        return cls(func._value_a, func._axis__a, func._value_u)
    
    @classmethod
    def from_zeros(cls, unit_values: PintUnit, dtype: Any, shape_f: int = 10):
        """Создает FreqFunc, заполненный нулями."""
        values = fw().zeros(shape_f, dtype=dtype)
        axis = fw().arange(shape_f)  # По умолчанию создаем ось 0, 1, 2, ... Гц
        return cls(values=values, axis=axis, unit_values=unit_values)
    
    @property
    def freq(self) -> Tuple[PintUnit, AnyArray]:
        """Возвращает кортеж (единица измерения, массив частот)."""
        return self._axis__u, self._axis__a
    
    @freq.setter
    def freq(self, value: AnyArray):
        """Устанавливает новый массив частот."""
        if len(value) != len(self._value_a):
            raise ValueError(f"Новый массив частот должен иметь длину {len(self._value_a)}, получено {len(value)}")
        self._axis__a = value  # Исправлено по аналогии с TimeFunc

    @property
    def min_freq(self) -> float:
        """Возвращает минимальное значение частоты."""
        return float(self._axis__a[0])
    
    @property
    def max_freq(self) -> float:
        """Возвращает максимальное значение частоты."""
        return float(self._axis__a[-1])
    
    @property
    def values(self) -> Tuple[PintUnit, AnyArray]:
        """Возвращает кортеж (единица измерения, массив значений)."""
        return self._value_u, self._value_a

    @values.setter
    def values(self, value: AnyArray):
        """Устанавливает новый массив значений."""
        if len(value) != len(self._axis__a):
            raise ValueError(f"Новый массив значений должен иметь длину {len(self._axis__a)}, получено {len(value)}")
        self._value_a = value
        
    @property
    def df(self) -> Tuple[PintUnit, float]:
        """Возвращает шаг по частоте (delta frequency)."""
        return self._d_axis__a

    def saveNPZ(self, path: Union[str, Path]):
        """Сохраняет объект в .npz файл."""
        Function1D.saveNPZ(self, path)
        
    @classmethod
    def loadNPZ(cls, path: Union[str, Path]):
        """Загружает объект из .npz файла."""
        return cls.from_Function1D(Function1D.loadNPZ(path))
    
class Function2D(Function):
    _matrx_a: Shaped[AnyArray, "X Y"]
    _matrx_u: PintUnit
    _first_a: Shaped[AnyArray, "X"]
    _first_u: PintUnit
    _sec___a: Shaped[AnyArray, "Y"]
    _sec___u: PintUnit

    def __init__(self, 
                 _matrx_a: Shaped[AnyArray, "X Y"], _matrx_u: PintUnit, 
                 _first_a: Shaped[AnyArray, "X"], _first_u: PintUnit, 
                 _sec___a: Shaped[AnyArray, "Y"], _sec___u: PintUnit):
        self._matrx_a = _matrx_a
        self._matrx_u = _matrx_u
        self._first_a = _first_a
        self._first_u = _first_u
        self._sec___a = _sec___a
        self._sec___u = _sec___u

    def saveNPZ(self, path: Union[str, Path]):
        import numpy as np
        if (not isinstance(self._matrx_a, np.ndarray) or 
            not isinstance(self._first_a, np.ndarray) or 
            not isinstance(self._sec___a, np.ndarray)): 
            raise RuntimeError("Arrays must be numpy ndarrays before saving.")
        
        np.savez_compressed(path,
            _matrx_a = self._matrx_a,
            _first_a = self._first_a,
            _sec___a = self._sec___a,
            _matrx_u = str(self._matrx_u),
            _first_u = str(self._first_u),
            _sec___u = str(self._sec___u),
        )

    @classmethod
    def loadNPZ(cls, path: Union[str, Path]):
        import numpy as np
        with np.load(path) as data:
            return cls(
                _matrx_a = data["_matrx_a"],
                _matrx_u = UREG.Unit(str(data["_matrx_u"])),
                _first_a = data["_first_a"],
                _first_u = UREG.Unit(str(data["_first_u"])),
                _sec___a = data["_sec___a"],
                _sec___u = UREG.Unit(str(data["_sec___u"]))
            )

    @property
    def _d_first_a(self) -> Tuple[PintUnit, float]:
        import numpy as np
        if len(self._first_a) < 2:
            raise ValueError("Cannot calculate step size for an AnyArray with less than 2 elements.")
        
        # Вычисляем средний шаг между элементами
        step = np.mean(np.diff(self._first_a))
        return (self._first_u, float(step))

    @property
    def _d_sec___a(self) -> Tuple[PintUnit, float]:
        import numpy as np
        if len(self._sec___a) < 2:
            raise ValueError("Cannot calculate step size for an AnyArray with less than 2 elements.")
        
        # Вычисляем средний шаг между элементами
        step = np.mean(np.diff(self._sec___a))
        return (self._sec___u, float(step))

    def SaveIntegralArea(self) -> Tuple[PintUnit, float]:
        fdu, fda = self._d_first_a
        sdu, sda = self._d_sec___a
        return unit_mul(self._matrx_u, fdu, sdu), float(fw().sum(self._matrx_a) * fda * sda)
    
    def SaveIntegralEnergy(self) -> Tuple[PintUnit, float]:
        fdu, fda = self._d_first_a
        sdu, sda = self._d_sec___a
        return unit_mul(self._matrx_u, self._matrx_u, fdu, sdu), float(fw().sum(self._matrx_a ** 2) * fda * sda)


class SpecFunc(Function2D):
    _first_u: PintUnit = UREG.second # Явно указал UREG.second для ясности
    _sec___u: PintUnit = UREG.hertz  # И UREG.hertz

    def __init__(self, matrix: Shaped[AnyArray, "T H"], time: Shaped[AnyArray, "T"], freq: Shaped[AnyArray, "H"], unit: PintUnit):
        # Проверка соответствия размеров
        if matrix.shape[0] != len(time):
            raise ValueError(f"Размер матрицы по оси 0 ({matrix.shape[0]}) не совпадает с длиной оси времени ({len(time)})")
        if matrix.shape[1] != len(freq):
            raise ValueError(f"Размер матрицы по оси 1 ({matrix.shape[1]}) не совпадает с длиной оси частот ({len(freq)})")
        
        self._matrx_a = matrix
        self._first_a = time
        self._sec___a = freq
        self._matrx_u = unit

    @classmethod
    def from_Function2D(cls, func: Function2D) -> Self:
        return cls(func._matrx_a, func._first_a, func._sec___a, func._matrx_u)

    @classmethod
    def from_zeros(cls, unit: PintUnit, dtype: Any, shape_t: int = 2, shape_h: int = 2):
        matrix = fw().zeros((shape_t, shape_h), dtype=dtype)
        time = fw().arange(shape_t)
        freq = fw().arange(shape_h)
        return cls(matrix=matrix, time=time, freq=freq, unit=unit)

    # --- Свойства для доступа к данным (без изменений) ---
    @property
    def time(self) -> Tuple[PintUnit, Shaped[AnyArray, "T"]]:
        return self._first_u, self._first_a
    
    @time.setter
    def time(self, value: Shaped[AnyArray, "T"]): 
        self._first_a = value
    
    @property
    def freq(self) -> Tuple[PintUnit, Shaped[AnyArray, "H"]]:
        return self._sec___u, self._sec___a
    
    @freq.setter
    def freq(self, value: Shaped[AnyArray, "H"]):
        self._sec___a = value

    @property
    def values(self) -> Tuple[PintUnit, Shaped[AnyArray, "T H"]]:
        return self._matrx_u, self._matrx_a
    
    @values.setter
    def values(self, value: Shaped[AnyArray, "T H"]):
        self._matrx_a = value

    @property
    def dt(self) -> Tuple[PintUnit, float]:
        return self._d_first_a

    @property
    def df(self) -> Tuple[PintUnit, float]:
        return self._d_sec___a

    # --- Методы сохранения/загрузки (без изменений) ---
    def saveNPZ(self, path: Union[str, Path]):
        Function2D.saveNPZ(self, path)
        
    @classmethod
    def loadNPZ(cls, path: Union[str, Path]):
        return cls.from_Function2D(Function2D.loadNPZ(path))

    def integrateOverTime(self) -> FreqFunc:
        """
        Интегрирует спектрограмму по оси времени, получая усредненный спектр (FreqFunc).
        
        Физический смысл: общая "площадь" сигнала на каждой частоте за весь период времени.
        
        :return: Объект FreqFunc, где ось - частота, а значения - результат интегрирования.
        """
        # Получаем шаг по времени (единица, значение)
        dt_unit, dt_value = self.dt
        
        # Суммируем значения матрицы по оси времени (axis=0) и умножаем на шаг dt
        integrated_values = fw().sum(self._matrx_a, axis=0) * dt_value
        
        # Вычисляем новую единицу измерения для значений
        # Например, (Па/√Гц) * с  ->  Па*с/√Гц
        new_unit = unit_mul(self._matrx_u, dt_unit)
        
        # Создаем и возвращаем новый FreqFunc
        # Осью для него будет ось частот изначальной спектрограммы
        return FreqFunc(
            values=integrated_values,
            axis=self._sec___a,  # ось частот
            unit_values=new_unit
        )

    def integrateOverFreq(self) -> TimeFunc:
        """
        Интегрирует спектрограмму по оси частот, получая временной ход мощности (TimeFunc).
        
        Физический смысл: общая "площадь" сигнала во всем частотном диапазоне в каждый момент времени.
        
        :return: Объект TimeFunc, где ось - время, а значения - результат интегрирования.
        """
        # Получаем шаг по частоте (единица, значение)
        df_unit, df_value = self.df
        
        # Суммируем значения матрицы по оси частот (axis=1) и умножаем на шаг df
        integrated_values = fw().sum(self._matrx_a, axis=1) * df_value
        
        # Вычисляем новую единицу измерения для значений
        # Например, (Па/√Гц) * Гц  ->  Па*√Гц (что соответствует размерности амплитуды)
        new_unit = unit_mul(self._matrx_u, df_unit)
        
        # Создаем и возвращаем новый TimeFunc
        # Осью для него будет ось времени изначальной спектрограммы
        return TimeFunc(
            values=integrated_values,
            axis=self._first_a, # ось времени
            unit_values=new_unit
        )
    
    def cloneApply(self, func: Callable[[AnyArray], AnyArray], new_unit: PintUnit):
        return SpecFunc(func(self._matrx_a), self._first_a, self._sec___a, new_unit)

    