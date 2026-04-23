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

def makeLogDB(spec: SpecFunc, add_one: bool = False) -> SpecFunc:
    import numpy as np

    old_unit, _ = spec.values

    # 1. Определяем референсный ноль в зависимости от физики
    if old_unit.is_compatible_with(UREG.Pa / (UREG.Hz ** 0.5)):
        ref_val = REF_PA_ASD
    elif old_unit.is_compatible_with(UREG.FS / (UREG.Hz ** 0.5)):
        ref_val = REF_FS_ASD
    elif old_unit.is_compatible_with(UREG.V / (UREG.Hz ** 0.5)):
        ref_val = REF_V_ASD
    else:
        ref_val = 1.0

    # 2. Логарифмируем
    def to_db(arr: np.ndarray) -> np.ndarray:
        # Работаем с амплитудой
        amplitude = np.abs(arr)
        
        # Отношение к референсу
        ratio = amplitude / ref_val
        
        if add_one:
            # 20 * log10(ratio + 1)
            # Здесь clip не обязателен, так как ratio >= 0, значит (ratio + 1) >= 1
            return 20 * np.log10(ratio + 1)
        else:
            # Стандартный логарифм с ограничением снизу, чтобы не получить -inf
            scaled_ratio = np.clip(ratio, 1e-9, None)
            return 20 * np.log10(scaled_ratio)

    # 3. Возвращаем новый объект
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

def addSpecToSpec(target: 'SpecFunc', source: 'SpecFunc') -> None:
    """
    Накладывает (прибавляет) source на target с учетом их физических осей (времени и частоты).
    Автоматически обрезает source, если он выходит за границы target.
    """
    # Извлекаем матрицы
    _, target_mat = target.values
    _, source_mat = source.values
    
    # Извлекаем оси
    _, target_time = target.time
    _, target_freq = target.freq
    _, source_time = source.time
    _, source_freq = source.freq
    
    # Получаем шаги (dt, df) из target для расчета индексов
    _, dt = target.dt
    _, df = target.df
    
    # Вычисляем смещение начала source относительно target в индексах
    t_offset = int(round(float(source_time[0] - target_time[0]) / float(dt)))
    f_offset = int(round(float(source_freq[0] - target_freq[0]) / float(df)))
    
    # Размеры матриц
    len_t, len_f = source_mat.shape
    max_t, max_f = target_mat.shape
    
    # Находим границы области пересечения в координатах target
    tgt_t_start = max(0, t_offset)
    tgt_t_end   = min(max_t, t_offset + len_t)
    tgt_f_start = max(0, f_offset)
    tgt_f_end   = min(max_f, f_offset + len_f)
    
    # Если пересечения нет (source полностью вне target), прерываем
    if tgt_t_start >= tgt_t_end or tgt_f_start >= tgt_f_end:
        return
        
    # Вычисляем те же границы, но в локальных координатах source
    src_t_start = tgt_t_start - t_offset
    src_t_end   = tgt_t_end - t_offset
    src_f_start = tgt_f_start - f_offset
    src_f_end   = tgt_f_end - f_offset
    
    # Сложение матриц (inplace)
    target_mat[tgt_t_start:tgt_t_end, tgt_f_start:tgt_f_end] += source_mat[src_t_start:src_t_end, src_f_start:src_f_end]


def localZNorm(spec: SpecFunc, window_t: int = 50, window_f: int = 50) -> SpecFunc:
    """
    Локальная Z-нормировка спектрограммы.
    Для каждого пикселя вычитает локальное среднее и делит на локальное СКО
    в окне (window_t x window_f).
    """
    import numpy as np
    from scipy.ndimage import uniform_filter

    _, mat = spec.values

    # Локальное среднее
    local_mean = uniform_filter(mat.astype(np.float32), size=(window_t, window_f))

    # Локальное СКО через E[X^2] - E[X]^2
    local_mean_sq = uniform_filter(mat.astype(np.float32) ** 2, size=(window_t, window_f))
    local_std = np.sqrt(np.clip(local_mean_sq - local_mean ** 2, 0, None))

    # Нормируем, защищаемся от деления на ноль
    normed = (mat - local_mean) / np.where(local_std < 1e-9, 1e-9, local_std)

    return spec.cloneApply(func=lambda _: normed, new_unit=UREG.dimensionless)


def correlationTransform(spec: SpecFunc) -> SpecFunc:
    """
    Корреляционное преобразование: для каждого пикселя (t, f)
    перемножает все 9 элементов окна 3x3.
    
    Знак сохраняется через произведение знаков.
    Амплитуда вычисляется как геометрическое среднее |x_i|.
    """
    import numpy as np

    _, mat = spec.values
    T, F = mat.shape

    # Pad матрицу (reflect, чтобы не терять края)
    padded = np.pad(mat, pad_width=2, mode='reflect')

    result = np.ones((T, F), dtype=np.float64)

    # Перемножаем все 9 элементов окна
    for dt in range(5):
        for df in range(5):
            result *= padded[dt:dt + T, df:df + F]

    return spec.cloneApply(func=lambda _: (result).astype(mat.dtype), new_unit=spec.values[0] ** 9)


def padSpecFreq(
    spec: SpecFunc,
    new_f_start: float,
    new_f_end: float
) -> SpecFunc:
    """
    Расширяет (дополняет) спектрограмму по частотной оси до новых границ.
    
    Области за пределами оригинального частотного диапазона заполняются
    подходящим значением (0 для комплексных/линейных данных или минимум 
    для логарифмических данных в dB).

    Пример:
        # spec_30_100: спектрограмма с диапазоном [30000.0, 100000.0] Гц
        # Расширяем её до стандартного диапазона [0, 120000.0] Гц
        spec_0_120 = padSpecFreq(spec_30_100, new_f_start=0.0, new_f_end=120000.0)

    Args:
        spec: Исходная спектрограмма.
        new_f_start: Новая начальная частота (в тех же юнитах, что и у spec.freq).
        new_f_end: Новая конечная частота.

    Returns:
        Новая, расширенная по частоте спектрограмма.
        
    Raises:
        ValueError: Если новый диапазон не охватывает полностью оригинальный.
    """
    import numpy as np

    # 1. Извлекаем исходные данные и параметры
    val_u, mat = spec.values
    _, t_arr   = spec.time
    _, f_arr   = spec.freq
    
    # Шаг по частоте
    _, df = spec.df
    
    # 2. Проверка корректности: новый диапазон должен полностью включать старый
    orig_f_start, orig_f_end = f_arr[0], f_arr[-1]
    if new_f_start > orig_f_start or new_f_end < orig_f_end:
        raise ValueError(
            f"Новый частотный диапазон [{new_f_start}, {new_f_end}] "
            f"не охватывает полностью оригинальный [{orig_f_start}, {orig_f_end}]."
        )

    # 3. Определяем, чем заполнять пустые области (padding)
    # Та же логика, что и в `cropSpecRect`: для dB-шкалы берем минимум, для остального — 0.
    if not np.iscomplexobj(mat) and np.min(mat) < 0:
        pad_value = np.min(mat)
    else:
        pad_value = 0
        
    # 4. Создаем новую ось частот
    # Используем `linspace` для большей точности с float
    num_new_f_bins = int(round((new_f_end - new_f_start) / float(df))) + 1
    new_f_arr = np.linspace(new_f_start, new_f_end, num_new_f_bins)
    
    # 5. Создаем новую матрицу, заполненную значением для паддинга
    num_t_bins = mat.shape[0]
    new_mat = np.full((num_t_bins, num_new_f_bins), fill_value=pad_value, dtype=mat.dtype)

    # 6. Вычисляем, куда в новую матрицу нужно вставить старые данные
    # Находим индекс, соответствующий началу старой частотной оси
    start_idx = int(round((orig_f_start - new_f_start) / float(df)))
    end_idx = start_idx + len(f_arr)
    
    # 7. Копируем оригинальную матрицу в вычисленный "слот"
    new_mat[:, start_idx:end_idx] = mat

    # 8. Собираем и возвращаем новый объект SpecFunc
    return SpecFunc(
        matrix = new_mat,
        time   = t_arr,
        freq   = new_f_arr,
        unit   = val_u
    )


def stackSpecsMax(*specs: SpecFunc) -> SpecFunc:
    """
    Создает новую спектрограмму, где каждый пиксель является максимумом 
    от соответствующих пикселей из набора входных спектрограмм.

    Все входные спектрограммы должны иметь одинаковый размер (shape),
    одинаковые оси времени и частоты, а также совместимые физические юниты.

    Пример:
        # spec1, spec2, spec3 - три спектрограммы одного и того же события
        max_spec = stackSpecsMax(spec1, spec2, spec3)
        # max_spec теперь содержит "пиковую огибающую" всех трех спектрограмм

    Args:
        *specs: Переменное количество объектов SpecFunc для обработки.

    Returns:
        Новый объект SpecFunc, являющийся поэлементным максимумом входных данных.
        
    Raises:
        ValueError: Если не передано ни одной спектрограммы, или если
                    спектрограммы несовместимы по размеру, осям или юнитам.
    """
    import numpy as np

    # 1. Проверка на пустой ввод
    if not specs:
        raise ValueError("Необходимо передать хотя бы одну спектрограмму.")

    # 2. Если передана только одна, просто возвращаем её клон
    if len(specs) == 1:
        return specs[0].clone()

    # 3. Берем первую спектрограмму как эталон для сравнения
    ref_spec = specs[0]
    ref_shape = ref_spec.values[1].shape
    ref_unit = ref_spec.values[0]
    ref_time = ref_spec.time[1]
    ref_freq = ref_spec.freq[1]

    # 4. Проверяем совместимость всех остальных спектрограмм с эталоном
    for i, spec in enumerate(specs[1:], start=1):
        if spec.values[1].shape != ref_shape:
            raise ValueError(
                f"Несовместимость размеров: спектрограмма #{i} имеет shape {spec.matrix.shape}, "
                f"ожидался {ref_shape}."
            )
        if not spec.values[0].is_compatible_with(ref_unit):
            raise ValueError(
                f"Несовместимость юнитов: спектрограмма #{i} имеет юнит '{spec.unit}', "
                f"ожидался совместимый с '{ref_unit}'."
            )
        if not np.allclose(spec.time[1], ref_time):
            raise ValueError(f"Ось времени спектрограммы #{i} не совпадает с эталонной.")
        if not np.allclose(spec.freq[1], ref_freq):
            raise ValueError(f"Ось частоты спектрограммы #{i} не совпадает с эталонной.")
    
    # 5. Если все проверки пройдены, выполняем операцию
    
    # Создаем стек из матриц. Получится 3D-массив (N, T, F)
    # где N - количество спектрограмм.
    all_matrices = [s.values[1] for s in specs]
    stacked_matrices = np.stack(all_matrices, axis=0)
    
    # Находим максимум вдоль первой оси (оси, по которой мы "сложили" спектрограммы)
    max_matrix = np.max(stacked_matrices, axis=0)

    # 6. Создаем и возвращаем новый объект SpecFunc с результатом
    return SpecFunc(
        matrix = max_matrix,
        time   = ref_time,
        freq   = ref_freq,
        unit   = ref_unit,
    )