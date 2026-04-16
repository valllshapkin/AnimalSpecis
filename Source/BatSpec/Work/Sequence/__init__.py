from BatSpec.Work.Function import SpecFunc
from BatSpec.Work.Function import TimeFunc
from typing import Union
from BatSpec.Work.Units import UREG

import numpy as np
from jaxtyping import Array
type AnyArray = Union[np.ndarray, Array]

class Sequenogram(TimeFunc):
    def __init__(self, values: AnyArray, axis: AnyArray):
        super().__init__(values, axis, unit_values=UREG.dimensionless)

        
def convolveSequenogramWithPattern(seq: 'Sequenogram', call_patch: 'SpecFunc') -> 'SpecFunc':
    """
    Свёртка 1D секвенограммы (TimeFunc) с 2D патчем (эталонным SpecFunc) через FFT.
    Выравнивает физический нуль (t=0) патча с пиками секвенограммы.
    """
    import numpy as np
    from scipy.signal import fftconvolve

    _, seq_vals = seq.values
    patch_unit, patch_vals = call_patch.values
    
    # Безопасно извлекаем numpy-массивы
    if hasattr(seq_vals, "numpy"):
        seq_vals = seq_vals.numpy()
    else:
        seq_vals = np.asarray(seq_vals)
        
    if hasattr(patch_vals, "numpy"):
        patch_vals = patch_vals.numpy()
    else:
        patch_vals = np.asarray(patch_vals)
        
    seq_expanded = seq_vals[:, np.newaxis]
    
    # Сворачиваем (mode='full' генерирует массив длины T + P_t - 1)
    conv_matrix = fftconvolve(seq_expanded, patch_vals, mode='full', axes=0)
    
    # === ИСПРАВЛЕНИЕ СДВИГА ===
    # Находим индекс "нуля" на оси времени патча
    _, patch_time = call_patch.time
    if hasattr(patch_time, "numpy"):
        patch_time = patch_time.numpy()
        
    # Ищем индекс, где значение времени максимально близко к 0.0
    zero_idx = int(np.abs(patch_time).argmin())
    
    # Чтобы "нулевая" точка патча легла ровно в пик секвенограммы,
    # мы сдвигаем "окно" вырезки вправо на величину zero_idx.
    # Длина секвенограммы T. Берем срез [zero_idx : zero_idx + T]
    T = len(seq_vals)
    conv_matrix = conv_matrix[zero_idx : zero_idx + T, :]
    
    # Извлекаем оригинальные оси
    _, time_axis = seq.time
    _, freq_axis = call_patch.freq
    
    return SpecFunc(
        matrix=conv_matrix.astype(np.float32), 
        time=time_axis, 
        freq=freq_axis, 
        unit=patch_unit
    )