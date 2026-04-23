from BatSpec.Work.Context import fw
from BatSpec.Work.Function import TimeFunc
from BatSpec.Work.Units import UREG, PintUnit
from BatSpec.Work.ConvWindow import Window, WindowNorm, WindowNormMismatchError
from pathlib import Path
from typing import Any, cast, Tuple

type T = Any

def loadRecord(path: Path, unit: PintUnit = UREG.FS) -> 'TimeFunc':
    """Load audio file as TimeFunc (mono, float64)."""
    
    import numpy as np
    import soundfile as sf # type: ignore

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    data, samplerate = cast(
        Tuple[np.typing.NDArray[np.float64], int], 
        sf.read(str(path), dtype='float64', always_2d=True) # type: ignore
    )

    # Force mono
    if data.ndim == 2:
        if data.shape[1] > 1:
            data = np.mean(data, axis=1)        # stereo -> mono
        else:
            data = data.flatten()               # (N, 1) -> (N,)

    # Time axis in seconds
    n_samples = len(data)
    time_axis = np.arange(n_samples) / samplerate

    return TimeFunc(
        values=data,
        axis=time_axis,
        unit_values=unit,
    )

def trimRecord(signal: TimeFunc, t_start: float = 0.0, t_end: float | None = None) -> TimeFunc:
    """
    Обрезает запись по времени (в секундах).

    Параметры
    ----------
    signal : TimeFunc
        Исходный сигнал.
    t_start : float, optional
        Начальное время обрезки в секундах (по умолчанию 0.0).
    t_end : float | None, optional
        Конечное время обрезки в секундах. 
        Если None — обрезается до конца сигнала.

    Возвращает
    ----------
    TimeFunc
        Новый обрезанный сигнал.
    """
    if t_start < 0:
        raise ValueError("t_start не может быть отрицательным")

    _, time_array = signal.time
    value_unit, value_array = signal.values

    # Если t_end не указан — берём конец сигнала
    if t_end is None:
        t_end = float(time_array[-1])

    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) должен быть больше t_start ({t_start})")

    # Находим индексы для обрезки
    mask = (time_array >= t_start) & (time_array <= t_end)
    
    if not fw().any(mask):
        raise ValueError(f"Интервал [{t_start}, {t_end}] не пересекается с сигналом")

    # Обрезаем массивы
    new_time = time_array[mask]
    new_values = value_array[mask]

    # Создаём новый объект TimeFunc
    return TimeFunc(
        values=new_values,
        axis=new_time,
        unit_values=value_unit
    )

def resampleRecord(signal: TimeFunc, new_sr: int) -> TimeFunc:
    from scipy.signal import resample_poly # type: ignore
    import numpy as np
    from math import gcd


    """
    Изменяет частоту дискретизации сигнала (resampling).

    Параметры
    ----------
    signal : TimeFunc
        Исходный сигнал.
    new_sr : int
        Новая частота дискретизации в Гц (например, 16000, 44100, 48000).

    Возвращает
    ----------
    TimeFunc
        Новый сигнал с изменённой частотой дискретизации.
    """
    if new_sr <= 0:
        raise ValueError("new_sr должен быть положительным целым числом")

    _, orig_time = signal.time
    value_unit, values = signal.values
    orig_sr = signal.sr

    if orig_sr == new_sr:
        # Нет необходимости в ресэмплинге
        return TimeFunc(values=values.copy(), 
                        axis=orig_time.copy(), 
                        unit_values=value_unit)

    # Вычисляем коэффициенты up/down для resample_poly
    # Пример: 44100 → 16000 → up=160, down=441 (после сокращения)
    g = gcd(orig_sr, new_sr)
    up = new_sr // g
    down = orig_sr // g

    # Выполняем ресэмплинг
    new_values = cast(np.typing.NDArray[Any], 
        resample_poly(values, up=up, down=down, axis=0)
    )

    # Создаём новый временной массив
    new_length = len(new_values)
    new_dt = 1.0 / new_sr
    new_time = np.arange(new_length, dtype=fw().float64) * new_dt

    return TimeFunc(
        values=new_values,
        axis=new_time,
        unit_values=value_unit
    )


def localRMS(f: TimeFunc, window: Window) -> TimeFunc:
    if window.norm != WindowNorm.AREA:
        raise WindowNormMismatchError(WindowNorm.AREA, window.norm)

    import numpy as np
    
    value_unit, value_array = f.values
    _, time_axis = f.time
    win_array = window.get_array(f.sr)

    # RMS = sqrt( mean(signal^2) ). mean(x) = convolve(x, window_norm_area)
    rms_values = np.sqrt(np.convolve(value_array ** 2, win_array, mode="same"))

    # RMS имеет ту же размерность, что и исходный сигнал
    return TimeFunc(
        values=rms_values,
        axis=time_axis.copy(),
        unit_values=value_unit
    )

def correctDC(f: TimeFunc) -> TimeFunc:
    import numpy as np
    
    value_unit, value_array = f.values
    _, time_axis = f.time

    # Вычитание медианы не меняет размерность
    corrected_values = value_array - np.median(value_array)
    
    return TimeFunc(
        values=corrected_values,
        axis=time_axis.copy(),
        unit_values=value_unit
    )


def makeLog(f: TimeFunc, base: str = "natural", add_one: bool = False) -> TimeFunc:
    """
    Применяет логарифм к значениям TimeFunc.

    Параметры
    ----------
    f : TimeFunc
        Входной сигнал.
    base : str, optional
        Основание логарифма: 'natural' (по умолчанию), '10' или '2'.
    add_one : bool, optional
        Если True, вычисляет log(x + 1). Полезно для сигналов, содержащих нули, 
        или для реализации log1p. По умолчанию False.

    Возвращает
    ----------
    TimeFunc
        Сигнал с логарифмированными значениями.

    Raises
    ------
    ValueError
        Если аргумент логарифма (x или x+1) <= 0 или указано неподдерживаемое основание.
    """
    import numpy as np

    value_unit, value_array = f.values
    _, time_axis = f.time

    # Определяем смещение
    offset = 1.0 if add_one else 0.0

    # Проверка: аргумент логарифма должен быть строго положительным
    if np.any(value_array + offset <= 0):
        if add_one:
            raise ValueError("makeLog: при add_one=True значения должны быть строго больше -1")
        else:
            raise ValueError("makeLog: значения должны быть строго положительными (или используйте add_one=True)")

    # Выбор функции логарифма
    if base == "natural":
        # Используем np.log1p для лучшей точности, если add_one=True
        log_func = np.log1p if add_one else np.log
    elif base == "10":
        log_func = np.log10
    elif base == "2":
        log_func = np.log2
    else:
        raise ValueError(f"Неподдерживаемое основание логарифма: {base}")

    # Вычисление
    if add_one and base == "natural":
        log_values = log_func(value_array) # np.log1p уже учитывает +1
    else:
        log_values = log_func(value_array + offset)

    new_unit = UREG.dimensionless

    return TimeFunc(
        values=log_values,
        axis=time_axis.copy(),
        unit_values=new_unit
    )