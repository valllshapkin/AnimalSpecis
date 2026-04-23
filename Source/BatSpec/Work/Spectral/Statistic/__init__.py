
from BatSpec.Work.Function import SpecFunc

def noiseZNormByFreq(spec: SpecFunc, noise_percentile: float = 10.0) -> SpecFunc:
    """
    Адаптивная Z-нормировка по частотным полосам на основе профиля шума.
    
    1. Интегрирует спектрограмму по частоте для получения энергии по времени.
    2. Находит временные кадры, энергия которых ниже заданного процентиля (считаем их шумом).
    3. Вычисляет среднее и СКО (сигму) для каждой частоты только на основе этих "шумовых" кадров.
    4. Применяет Z-нормировку ко всей спектрограмме вдоль оси времени для каждой частоты.
    
    Args:
        spec: Исходная спектрограмма.
        noise_percentile: Процентиль (от 0 до 100), определяющий границу "шумовых" кадров.
                          Например, 10.0 означает 10% самых тихих кадров.
    Returns:
        Спектрограмма после Z-нормировки (безразмерная величина).
    """
    import numpy as np
    from BatSpec.Work.Units import UREG

    # 1. Сворачиваем (интегрируем) по оси частот для получения профиля энергии во времени
    time_func = spec.integrateOverFreq()
    _, energy_profile = time_func.values

    # 2. Находим порог для самых тихих вырезок
    threshold = np.percentile(energy_profile, noise_percentile)
    
    # Создаем булеву маску для шумовых кадров
    noise_mask = energy_profile <= threshold

    # Защита от случая, если маска оказалась пустой (например, при странных данных или percentile=0)
    if not np.any(noise_mask):
        noise_mask[np.argmin(energy_profile)] = True

    # 3. Извлекаем матрицу спектрограммы (T, F)
    _, mat = spec.values
    
    # Выделяем только те кадры, которые относятся к шуму
    noise_mat = mat[noise_mask, :]  # shape: (N_noise_frames, F)

    # 4. Считаем среднее и сигму вдоль оси времени (axis=0) для каждой частоты
    noise_mean = np.mean(noise_mat, axis=0)  # shape: (F,)
    noise_std = np.std(noise_mat, axis=0)    # shape: (F,)

    # Защита от деления на ноль (ограничиваем минимальное значение СКО)
    safe_std = np.where(noise_std < 1e-9, 1e-9, noise_std)

    # 5. Применяем Z-нормировку ко всей исходной матрице
    # Благодаря механизму broadcasting в numpy, операция (T, F) - (F,) 
    # корректно вычтет среднее и поделит на СКО для каждого столбца (частоты).
    normalized_mat = (mat - noise_mean) / safe_std

    # Возвращаем новый объект спектрограммы с примененной матрицей и безразмерным юнитом
    return spec.cloneApply(
        func=lambda _: normalized_mat.astype(np.float32), 
        new_unit=UREG.dimensionless
    )


from typing import Optional
import numpy as np
from BatSpec.Work.Function import TimeFunc, SpecFunc
from BatSpec.Work.Units import unit_devide
from BatSpec.Work.Record import resampleRecord

def integrateAndClipEnvelope(
    spec: SpecFunc, 
    threshold: float = 5.0, 
    resample: Optional[int] = None,
    make_zero: bool = True,
    log: bool = False
) -> TimeFunc:
    """
    Интегрирует спектрограмму, нормализует её, выполняет опциональный ресемплинг 
    и отсекает шум по порогу сигм.

    Args:
        spec: Входная спектрограмма.
        threshold: Коэффициент сигма для клиппинга (по умолчанию 5).
        resample: Целевая частота дискретизации (SR). Если None - ресемплинг не проводится.
    """
    # 1. Интегрируем по частотам (результат в TimeFunc)
    # Внутри integrateOverFreq уже произошло умножение на df
    time_func = spec.integrateOverFreq()
    
    # 2. Подготовка параметров нормализации
    f_unit, f_arr = spec.freq
    N = len(f_arr)
    f_range = f_arr[-1] - f_arr[0]
    if f_range <= 0:
        f_range = 1e-9

    # Извлекаем значения для масштабирования
    val_unit, val_arr = time_func.values
    
    # Масштабируем: делим на sqrt(N) и на размах частот
    # Физически: (X * Hz) / (1 * Hz) -> возвращаемся к исходному юниту X
    norm_factor = np.sqrt(N) * float(f_range)
    scaled_values = val_arr / norm_factor
    new_unit = unit_devide(val_unit, f_unit)

    # Создаем временный объект TimeFunc для дальнейших манипуляций
    envelope = TimeFunc(
        values      = scaled_values.astype(np.float32),
        axis        = time_func.time[1],
        unit_values = new_unit
    )

    # 3. Ресемплинг (выполняется ДО расчета сигмы и клиппинга)
    if resample is not None:
        # resampleRecord принимает TimeFunc и возвращает новый TimeFunc
        envelope = resampleRecord(envelope, resample)

    # 4. Клиппинг на основе статистики уже (возможно) ресемплированного сигнала
    _, final_values = envelope.values
    
    sigma = np.std(final_values)
    limit = threshold * sigma
    
    # Применяем клиппинг: всё что ниже 'limit' становится равным 'limit'
    # Отрицательные значения также подтягиваются к этому порогу
    clipped_values = np.clip(final_values, a_min=limit, a_max=None)
    clipped_values = clipped_values - limit if make_zero else clipped_values
    
    # Возвращаем финальный TimeFunc
    return TimeFunc(
        values      = clipped_values.astype(np.float32),
        axis        = envelope.time[1],
        unit_values = envelope.values[0]
    )
