from BatSpec.Work.Context import fw
from BatSpec.Work.Function import TimeFunc
from BatSpec.Work.Units import UREG, PintUnit
from pathlib import Path
from typing import Any

type T = Any

def loadRecord(path: Path, unit: PintUnit = UREG.FS) -> 'TimeFunc':
    """Load audio file as TimeFunc (mono, float64)."""
    
    import numpy as np
    import soundfile as sf

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    data, samplerate = sf.read(str(path), dtype='float64', always_2d=True)

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
    from scipy.signal import resample_poly
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
    new_values = resample_poly(values, up=up, down=down, axis=0)

    # Создаём новый временной массив
    new_length = len(new_values)
    new_dt = 1.0 / new_sr
    new_time = np.arange(new_length, dtype=fw().float64) * new_dt

    return TimeFunc(
        values=new_values,
        axis=new_time,
        unit_values=value_unit
    )












    