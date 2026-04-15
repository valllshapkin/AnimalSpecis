from pathlib import Path
from typing import Optional, cast

from BatSpec.Work.Function import TimeFunc, SpecFunc
from BatSpec.Work.ConvWindow import Window, WindowNorm, WindowNormMismatchError
from BatSpec.Work.Units import UREG, unit_devide, unit_sqrt, unit_mul


def makeComplexSpec(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)
    
    import numpy as np
    import tensorflow as tf

    win        = window.get_array(signal.sr)
    fft_length = (bins - 1) * 2
    win_length = len(win)
    hop        = max(1, int(win_length * (1 - overlap)))

    if fft_length < win_length:
        print(
            f"[WARNING] fft_length={fft_length} меньше win_length={win_length}. "
            f"Фрейм будет обрезан — это потеря данных."
        )
    if fft_length > win_length:
        print(
            f"[WARNING] fft_length={fft_length} больше win_length={win_length}. "
            f"Это увеличивает разрешение по частоте но не приносит реальные данные."
        )
    print(f"fft_length={fft_length} win_length={win_length} hop={hop}")

    def win_func(length: int, dtype: tf.DType = tf.float32):
        assert length == win_length
        return tf.convert_to_tensor(win, dtype=dtype)

    # Извлекаем оригинальный unit сигнала
    signal_unit, signal_a = signal.values
    
    stft = cast(tf.Tensor, tf.signal.stft( # type: ignore
        signal_a.astype(np.float32),
        frame_length=win_length,
        frame_step=hop,
        fft_length=fft_length,
        window_fn=win_func,
    ))

    complex_matrix  = stft.numpy()                             # (T, H) комплексный
    scale           = np.sqrt(2 / signal.sr)
    num_frames: int = complex_matrix.shape[0]
    freq_axis  = np.fft.rfftfreq(fft_length, d=1.0 / signal.sr)
    time_axis  = (np.arange(num_frames) * hop) / signal.sr

    # Вычисляем новый unit: X / sqrt(Hz)
    spec_unit = unit_devide(signal_unit, unit_sqrt(UREG.Hz))

    return SpecFunc(
        matrix = complex_matrix * scale,
        time   = time_axis,
        freq   = freq_axis,
        unit   = spec_unit,                                   # Физически корректный unit
    )

def inverseComplexSpec(spec: SpecFunc, window: Window, overlap: float = 0.5) -> TimeFunc:
    # Window[norm="energy"]
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)
    
    import numpy as np
    import tensorflow as tf

    spec_unit, complex_matrix = spec.values
    _, _ = spec.time
    _, freq_axis = spec.freq

    # 1. Восстанавливаем параметры FFT
    # freq_axis имеет размер (fft_length // 2) + 1
    fft_length = (len(freq_axis) - 1) * 2
    
    # 2. Восстанавливаем SR (Частота дискретизации)
    # sr_f — это шаг по частоте (delta f)
    _, sr_f = spec.df 
    sr = int(round(sr_f * fft_length)) # Правильная формула: SR = df * fft_length
    
    # 3. Готовим окно и шаг
    win        = window.get_array(sr)
    win_length = len(win)
    hop        = max(1, int(win_length * (1 - overlap)))

    # Масштабирование (обратное тому, что в makeComplexSpec)
    scale          = np.sqrt(2 / sr)
    unscaled       = complex_matrix / scale

    def win_func(length: int, dtype: tf.DType = tf.float32):
        assert length == win_length
        return tf.convert_to_tensor(win, dtype=dtype)

    # 4. Обратный STFT
    signal_tf = cast(tf.Tensor, tf.signal.inverse_stft( # type: ignore
        tf.convert_to_tensor(unscaled, dtype=tf.complex64),
        frame_length = win_length,
        frame_step   = hop,
        fft_length   = fft_length,
        window_fn    = tf.signal.inverse_stft_window_fn(hop, forward_window_fn=win_func), # type: ignore
    ))

    signal_a  = signal_tf.numpy().astype(np.float32)
    
    # Создаем новую временную ось на основе восстановленного SR
    new_time_axis = np.arange(len(signal_a)) / sr

    # 5. Восстанавливаем юнит
    signal_unit = unit_mul(spec_unit, unit_sqrt(UREG.Hz))

    return TimeFunc(
        values      = signal_a,
        axis        = new_time_axis,
        unit_values = signal_unit,
    )

def makeSpec(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    """Амплитудная спектрограмма (|STFT| * scale)."""
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)

    import numpy as np
    import tensorflow as tf

    win        = window.get_array(signal.sr)
    fft_length = (bins - 1) * 2
    win_length = len(win)
    hop        = max(1, int(win_length * (1 - overlap)))

    def win_func(length: int, dtype: tf.DType = tf.float32):
        assert length == win_length
        return tf.convert_to_tensor(win, dtype=dtype)

    signal_unit, signal_a = signal.values

    stft = cast(tf.Tensor, tf.signal.stft(  # type: ignore
        signal_a.astype(np.float32),
        frame_length=win_length,
        frame_step=hop,
        fft_length=fft_length,
        window_fn=win_func,
    ))

    amplitude_matrix = np.abs(stft.numpy())          # (T, F) вещественный
    scale            = np.sqrt(2 / signal.sr)
    num_frames: int  = amplitude_matrix.shape[0]
    freq_axis        = np.fft.rfftfreq(fft_length, d=1.0 / signal.sr)
    time_axis        = (np.arange(num_frames) * hop) / signal.sr

    spec_unit = unit_devide(signal_unit, unit_sqrt(UREG.Hz))

    return SpecFunc(
        matrix=amplitude_matrix * scale,
        time=time_axis,
        freq=freq_axis,
        unit=spec_unit,
    )

# Константы для уровней (reference values)
# 20 мкПа для звукового давления (в воздухе)
REF_PA_ASD = 2e-5  
# 1.0 для Full Scale (цифровой ноль)
REF_FS_ASD = 1.0   
# 1.0 для Вольт (если сигнал снимается прямо с АЦП)
REF_V_ASD  = 1.0   

def makeLogDB(spec: SpecFunc) -> SpecFunc:
    import numpy as np

    old_unit, _ = spec.values

    # 1. Определяем, в каких физических величинах находится спектрограмма
    # и подбираем правильный референсный ноль.
    # is_compatible_with проверяет физическую размерность, а не точное совпадение строки!
    if old_unit.is_compatible_with(UREG.Pa / (UREG.Hz ** 0.5)):
        ref_val = REF_PA_ASD
        
    elif old_unit.is_compatible_with(UREG.FS / (UREG.Hz ** 0.5)):
        ref_val = REF_FS_ASD
        
    elif old_unit.is_compatible_with(UREG.V / (UREG.Hz ** 0.5)):
        ref_val = REF_V_ASD
        
    else:
        # Если размерность неизвестна (или безразмерная) — берем 1.0
        ref_val = 1.0

    # 2. Логарифмируем: 20 * log10( X / X_ref )
    def to_db(arr: np.ndarray) -> np.ndarray:
        # Модуль на случай, если спектр все еще комплексный (хотя обычно сюда подают уже амплитудный)
        amplitude = np.abs(arr)
        
        # Делим на константу и ограничиваем снизу, чтобы не поймать log(0)
        scaled_amplitude = np.clip(amplitude / ref_val, 1e-9, None)
        
        return 20 * np.log10(scaled_amplitude)

    # 3. Возвращаем новый объект, применяя функцию и задав юнит dB
    return spec.cloneApply(func=to_db, new_unit=UREG.dB)

def saveSpecToPNG(spec: SpecFunc, filepath: str | Path, cmap: str = 'magma'):
    """
    Сохраняет спектрограмму в PNG с подписями осей, юнитами и цветовой шкалой.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    val_u, mat = spec.values
    t_u, t_arr = spec.time
    f_u, f_arr = spec.freq

    # Если матрица комплексная (до взятия логарифма или модуля), берем модуль
    if np.iscomplexobj(mat):
        plot_mat = np.abs(mat)
    else:
        plot_mat = mat

    fig, ax = plt.subplots(figsize=(10, 6))

    # Вычисляем границы для осей изображения [left, right, bottom, top]
    extent = [t_arr[0], t_arr[-1], f_arr[0], f_arr[-1]]

    # Рисуем (матрицу транспонируем, так как imshow ожидает Y=Частота, X=Время)
    im = ax.imshow(
        plot_mat.T, 
        aspect='auto', 
        origin='lower', 
        extent=extent, 
        cmap=cmap,
        interpolation='nearest'
    )

    # Добавляем Colorbar и подписываем юнит значений (например, dB или Pa/sqrt(Hz))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Amplitude / Power [{val_u}]")

    # Подписываем оси
    ax.set_xlabel(f"Time [{t_u}]")
    ax.set_ylabel(f"Frequency [{f_u}]")
    ax.set_title("Spectrogram")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)

def cropSpecRect(
    spec: SpecFunc, 
    t_start: float, 
    t_end: float, 
    f_start: float, 
    f_end: float
) -> SpecFunc:
    """
    Вырезает прямоугольник из спектрограммы по физическим координатам.
    Если координаты выходят за пределы, дополняет нулями (или минимумом для dB).
    """
    import numpy as np

    val_u, mat = spec.values
    t_u, t_arr = spec.time
    f_u, f_arr = spec.freq

    dt = t_arr[1] - t_arr[0]
    df = f_arr[1] - f_arr[0]

    # Находим индексы относительно начала оригинального массива (могут быть отрицательными!)
    idx_t_start = int(round((t_start - t_arr[0]) / dt))
    idx_t_end   = int(round((t_end - t_arr[0]) / dt))
    
    idx_f_start = int(round((f_start - f_arr[0]) / df))
    idx_f_end   = int(round((f_end - f_arr[0]) / df))

    # Размеры новой матрицы
    N_t = max(0, idx_t_end - idx_t_start)
    N_f = max(0, idx_f_end - idx_f_start)

    # Определяем, чем заполнять пустые области (padding)
    # Если данные вещественные и содержат отрицательные числа (вероятно это Децибелы), 
    # паддинг нулями даст ложный яркий фон. Лучше заполнять минимальным значением.
    if not np.iscomplexobj(mat) and np.min(mat) < 0:
        pad_value = np.min(mat)
    else:
        pad_value = 0

    new_mat = np.full((N_t, N_f), fill_value=pad_value, dtype=mat.dtype)

    # Вычисляем границы доступных данных в оригинальной матрице
    orig_t_start = max(0, idx_t_start)
    orig_t_end   = min(len(t_arr), idx_t_end)
    orig_f_start = max(0, idx_f_start)
    orig_f_end   = min(len(f_arr), idx_f_end)

    # Вычисляем, куда вставить эти данные в новую матрицу
    new_t_start = max(0, -idx_t_start)
    new_t_end   = new_t_start + (orig_t_end - orig_t_start)
    
    new_f_start = max(0, -idx_f_start)
    new_f_end   = new_f_start + (orig_f_end - orig_f_start)

    # Копируем пересечение, если оно есть
    if (orig_t_start < orig_t_end) and (orig_f_start < orig_f_end):
        new_mat[new_t_start:new_t_end, new_f_start:new_f_end] = \
            mat[orig_t_start:orig_t_end, orig_f_start:orig_f_end]

    # Генерируем новые оси, продолжая оригинальную сетку
    new_t_arr = t_arr[0] + np.arange(idx_t_start, idx_t_end) * dt
    new_f_arr = f_arr[0] + np.arange(idx_f_start, idx_f_end) * df

    return SpecFunc(
        matrix = new_mat,
        time   = new_t_arr,
        freq   = new_f_arr,
        unit   = val_u
    )

def extractPeakContext(
    spec: SpecFunc, 
    duration_s: float = 0.100, 
    f_start: Optional[float] = None, 
    f_end: Optional[float] = None
) -> SpecFunc:
    """
    Интегрирует спектрограмму по частоте, находит временной пик и 
    вырезает окно (duration_s) вокруг этого пика.
    Если границы частот не переданы — берет весь частотный диапазон.
    """
    import numpy as np

    # 1. Сворачиваем спектрограмму в TimeFunc
    # Предполагается, что метод возвращает профиль энергии/мощности во времени
    time_func: TimeFunc = spec.integrateOverFreq()
    
    _, val_arr = time_func.values
    _, t_arr = time_func.time

    # 2. Ищем индекс максимального значения (пик энергии/амплитуды)
    peak_idx = np.argmax(val_arr)
    peak_time = float(t_arr[peak_idx])

    # 3. Вычисляем временное окно
    half_dur = duration_s / 2.0
    t_start = peak_time - half_dur
    t_end   = peak_time + half_dur

    # 4. Если частоты не заданы, берем оригинальные границы
    _, f_arr = spec.freq
    if f_start is None:
        f_start = float(f_arr[0])
    if f_end is None:
        f_end = float(f_arr[-1])

    # 5. Вызываем ранее написанную функцию вырезания прямоугольника
    return cropSpecRect(spec, t_start=t_start, t_end=t_end, f_start=f_start, f_end=f_end)