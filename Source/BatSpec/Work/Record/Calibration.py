from typing import Callable
from BatSpec.Work.Units import UREG, unit_mul, unit_devide
from BatSpec.Work.Context import fw
from BatSpec.Work.Function import TimeFunc, FreqFunc, SpecFunc, AnyArray
from BatSpec.Work.Spectral import makeComplexSpec, inverseComplexSpec
from BatSpec.Work.ConvWindow import Window, WindowNorm, TEST_HANN_WINODW

# =====================================================================
# 1. Аналитические модели приборов (базовый класс + реализации)
# =====================================================================

class CalibrationModel:
    """Базовый класс для аналитической модели АЧХ прибора."""
    def __call__(self, freq_hz: float) -> float:
        """
        Возвращает калибровочный коэффициент (Pa / FS) для одного частотного бина.
        :param freq_hz: Частота в Герцах.
        :return: Коэффициент в Па/FS.
        """
        raise NotImplementedError

    def generate_calibration_curve(self, freq_axis: AnyArray) -> FreqFunc:
        """
        Принимает ось частот и модель, вызывая её побиново,
        и собирает FreqFunc с единицами Pa / FS.
        """
        import numpy as np

        coef_values = np.array([self(float(f)) for f in freq_axis])
        unit_pa_per_fs = unit_devide(UREG.Pa, UREG.FS)

        return FreqFunc(
            values=coef_values,
            axis=freq_axis,
            unit_values=unit_pa_per_fs,
        )

        
class FlatResponseModel(CalibrationModel):
    """
    Идеальный прибор с ровной АЧХ.
    1.0 FS всегда равен sensitivity_pa Паскалей на любой частоте.
    """
    def __init__(self, sensitivity_pa: float = 20.0):
        self.sensitivity_pa = sensitivity_pa

    def __call__(self, freq_hz: float) -> float:
        return self.sensitivity_pa


class PetterssonM500Model(CalibrationModel):
    """
    Примерная выдуманная модель для ультразвукового микрофона.
    Чувствительность падает на высоких частотах — коэффициент
    (сколько Паскалей в одной цифровой единице) растёт выше 20 кГц.
    """
    def __init__(self, base_pa_per_fs: float = 15.0, rolloff_start_hz: float = 20_000.0,
                 rolloff_rate: float = 5.0 / 10_000.0):
        self.base_pa_per_fs   = base_pa_per_fs
        self.rolloff_start_hz = rolloff_start_hz
        self.rolloff_rate     = rolloff_rate  # Па/FS на Гц выше порога

    def __call__(self, freq_hz: float) -> float:
        boost = max(0.0, freq_hz - self.rolloff_start_hz) * self.rolloff_rate
        return self.base_pa_per_fs + boost


class ResonanceMicModel(CalibrationModel):
    """
    Модель микрофона с резонансом.
    На резонансной частоте микрофон выдаёт больший сигнал —
    коэффициент Pa/FS там меньше (Гауссиан вычитается из базы).
    """
    def __init__(self, base_pa: float = 25.0, resonance_hz: float = 40_000.0,
                 resonance_width_hz: float = 5_000.0, resonance_depth_pa: float = 15.0):
        self.base_pa            = base_pa
        self.resonance_hz       = resonance_hz
        self.resonance_width_hz = resonance_width_hz
        self.resonance_depth_pa = resonance_depth_pa

    def __call__(self, freq_hz: float) -> float:
        import math
        exponent   = -0.5 * ((freq_hz - self.resonance_hz) / self.resonance_width_hz) ** 2
        resonance  = self.resonance_depth_pa * math.exp(exponent)
        return self.base_pa - resonance



# =====================================================================
# 2. Главная функция: TimeFunc(FS) -> TimeFunc(Pa)
# =====================================================================

def applyСalibration(signal_fs: TimeFunc, model: CalibrationModel, 
    window: Window = TEST_HANN_WINODW, overlap: float = 0.5, bins: int = 300
) -> TimeFunc:
    """
    Переводит сырой цифровой сигнал (FS) в физические Паскали (Pa),
    учитывая частотно-зависимую калибровку (АЧХ) прибора.
    """
    # 1. Проверяем единицы входного сигнала
    u, _ = signal_fs.values
    if not u.is_compatible_with(UREG.FS):
        raise ValueError(f"Ожидался сигнал в единицах FS, получено: {u}")

    # 2. Переводим в частотно-временную область (STFT)
    complex_spec = makeComplexSpec(signal_fs, window=window, overlap=overlap, bins=bins)

    # 3. Генерируем калибровочную кривую по оси частот спектрограммы
    _, freq_axis = complex_spec.freq
    calib_curve  = model.generate_calibration_curve(freq_axis)

    # 4. Применяем коэффициенты к спектру (broadcasting: (T, H) * (H,))
    spec_u,  spec_matrix  = complex_spec.values
    coef_u,  coef_vector  = calib_curve.values

    calibrated_matrix = spec_matrix * coef_vector
    new_spec_unit     = unit_mul(spec_u, coef_u)

    # 5. Собираем откалиброванную спектрограмму
    calibrated_spec = SpecFunc(
        matrix=calibrated_matrix,
        time=complex_spec.time[1],
        freq=complex_spec.freq[1],
        unit=new_spec_unit,
    )

    # 6. Обратный STFT -> временной сигнал в Паскалях
    return inverseComplexSpec(calibrated_spec, window, overlap=overlap)