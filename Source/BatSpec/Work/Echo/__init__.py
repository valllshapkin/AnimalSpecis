import numpy as np
from scipy.signal import fftconvolve
from typing import Union, List

# =====================================================================
# 1. МОДЕЛИ ЭХА
# =====================================================================

from BatSpec.Work.Function import SpecFunc, TimeFunc
from BatSpec.Work.Units import UREG

class CausalEchoModel:
    def __init__(self, decay_rate_base: float, f_ref: float, freq_exp: float, time_power: float = 0.0, max_duration: float = 10.0):
        self.decay_rate_base = decay_rate_base
        self.f_ref = f_ref
        self.freq_exp = freq_exp
        self.time_power = time_power
        self.max_duration = max_duration

    def get(self, freq_axis: np.ndarray, dt: float, threshold: float = 1e-4) -> 'SpecFunc':
        """
        Генерирует SpecFunc (окно эха) для заданного массива частот и шага по времени dt.
        """
        # 1. Вычисляем rate для каждой частоты
        rates = self.decay_rate_base * (np.abs(freq_axis) / self.f_ref) ** self.freq_exp
        
        # 2. Оцениваем длительность хвоста (когда сигнал упадет до threshold)
        with np.errstate(divide='ignore'):
            durations = np.where(rates > 0, -np.log(threshold) / rates, self.max_duration)
        
        # 3. Находим общую максимальную длительность окна (срезаем по max_duration)
        max_t = min(np.max(durations), self.max_duration)
        
        # 4. Создаем ось времени: строго каузальная, начинается с t=0
        time_axis = np.arange(0, max_t, dt, dtype=np.float32)
        
        # 5. Векторизованное вычисление (Time x Freq) без циклов
        t_grid = time_axis[:, np.newaxis]
        rates_grid = rates[np.newaxis, :]
        
        power_decay = 1.0 / ((t_grid + 1.0) ** self.time_power)
        exp_decay = np.exp(-rates_grid * t_grid)
        
        # Результирующая матрица эха
        matrix = (power_decay * exp_decay).astype(np.float32)

        return SpecFunc(
            matrix=matrix,
            time=time_axis,
            freq=freq_axis,
            unit=UREG.dimensionless
        )


import numpy as np
# from BatSpec.Work.Function import SpecFunc
# from BatSpec.Work.Units import UREG


class AsymmetricEchoModel:
    def __init__(self, decay_rate_base: float, f_ref: float, freq_exp: float, 
                 time_power: float = 0.0, attack_tau: float = 0.0002):
        self.decay_rate_base = decay_rate_base
        self.f_ref = f_ref
        self.freq_exp = freq_exp
        self.time_power = time_power
        self.attack_tau = attack_tau

    def get(self, freq_axis: np.ndarray, dt: float, duration: float) -> SpecFunc:
        """
        Генерирует SpecFunc (окно эха) с плавной атакой.
        Максимум в каждой частотной полосе выровнен ровно на t=0 и отнормирован на 1.0.
        """
        time_axis = np.arange(0, duration, dt, dtype=np.float32)
        
        t_grid = time_axis[:, np.newaxis]
        f_grid = freq_axis[np.newaxis, :]
        
        rates = self.decay_rate_base * (np.abs(f_grid) / self.f_ref) ** self.freq_exp
        
        attack_factor = 1.0 - np.exp(-t_grid / self.attack_tau)
        power_decay = 1.0 / ((t_grid + 1.0) ** self.time_power)
        exp_decay = np.exp(-rates * t_grid)
        
        matrix = (attack_factor * power_decay * exp_decay).astype(np.float32)

        # === 1. ВЫРАВНИВАНИЕ ПИКОВ НА t=0 ===
        # Ищем индекс максимума для каждой частотной полосы (axis=0 означает "в каждом столбце")
        peak_indices = np.argmax(matrix, axis=0)
        
        shifted_matrix = np.zeros_like(matrix)
        T_len, F_len = matrix.shape
        
        for f_idx in range(F_len):
            shift = peak_indices[f_idx]
            if shift < T_len:
                # Сдвигаем хвост эха вверх к t=0. Остаток снизу останется нулями (np.zeros_like).
                shifted_matrix[:T_len - shift, f_idx] = matrix[shift:, f_idx]
                
        matrix = shifted_matrix

        # === 2. НОРМИРОВКА ===
        # Находим максимум в каждом столбце (теперь он лежит прямо на t=0)
        max_vals = np.max(matrix, axis=0, keepdims=True)
        max_vals[max_vals == 0] = 1.0  # Защита от деления на ноль
        matrix /= max_vals

        # === 3. СГЛАЖИВАНИЕ ХВОСТА (Fade-out) ===
        fade_len = min(int(len(time_axis) * 0.05), 100)
        if fade_len > 0:
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)[:, np.newaxis]
            matrix[-fade_len:] *= fade

        return SpecFunc(
            matrix=matrix,
            time=time_axis,
            freq=freq_axis,
            unit=UREG.dimensionless
        )

# class AsymmetricEchoModel(EchoModel):
#     """
#     Твоя оригинальная модель с медленной "атакой" (нарастанием) и затуханием.
#     Пик НЕ находится на t=0, что вызывает фазовые сдвиги в деконволюции.
#     """
#     def __init__(self, decay_rate_base: float, f_ref: float, freq_exp: float, time_power: float = 0.0, max_duration: float = 10.0):
#         self.decay_rate_base = decay_rate_base
#         self.f_ref = f_ref
#         self.freq_exp = freq_exp
#         self.time_power = time_power
#         self.max_duration = max_duration

#     def __repr__(self) -> str:
#         return (f"{self.__class__.__name__}(decay_rate_base={self.decay_rate_base!r}, ...)")

#     def _get_decay_rate(self, f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         return self.decay_rate_base * (np.abs(f) / self.f_ref) ** self.freq_exp

#     def compute_value(self, t: Union[float, np.ndarray], f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         decay_rate = self._get_decay_rate(f)
#         exp_decay = np.exp(-decay_rate * t)
#         power_decay = 1.0 / ((t + 1.0) ** self.time_power)
#         attack_tau = 0.0002
#         attack_factor = 1.0 - np.exp(-t / attack_tau)
#         return attack_factor * power_decay * exp_decay

#     def get_duration(self, f: Union[float, np.ndarray], threshold: float = 1e-4) -> Union[float, np.ndarray]:
#         # (Логика get_duration остается той же, что и у тебя)
#         # ...
#         f_array = np.atleast_1d(f)
#         decay_rates = self._get_decay_rate(f_array)
#         durations = np.zeros_like(f_array, dtype=float)
#         for i, rate in enumerate(decay_rates):
#             if rate > 0:
#                 t_est = -np.log(threshold) / rate
#                 durations[i] = min(t_est, self.max_duration)
#             else:
#                 durations[i] = self.max_duration
#         return float(durations[0]) if np.isscalar(f) else durations


# class CausalEchoModel(EchoModel):
#     """
#     Новая, строго каузальная модель. Пик всегда на t=0.
#     Не имеет "атаки", сразу начинается с максимального затухания.
#     Идеально подходит для деконволюции (Винер, Ричардсон-Люси).
#     """
#     def __init__(self, decay_rate_base: float, f_ref: float, freq_exp: float, time_power: float = 0.0, max_duration: float = 10.0):
#         self.decay_rate_base = decay_rate_base
#         self.f_ref = f_ref
#         self.freq_exp = freq_exp
#         self.time_power = time_power
#         self.max_duration = max_duration

#     def __repr__(self) -> str:
#         return (f"{self.__class__.__name__}(decay_rate_base={self.decay_rate_base!r}, ...)")

#     def _get_decay_rate(self, f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         return self.decay_rate_base * (np.abs(f) / self.f_ref) ** self.freq_exp

#     def compute_value(self, t: Union[float, np.ndarray], f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         # Убираем attack_factor. Теперь при t=0, результат равен 1.0
#         decay_rate = self._get_decay_rate(f)
#         exp_decay = np.exp(-decay_rate * t)
#         power_decay = 1.0 / ((t + 1.0) ** self.time_power)
#         return power_decay * exp_decay

#     def get_duration(self, f: Union[float, np.ndarray], threshold: float = 1e-4) -> Union[float, np.ndarray]:
#         # (Логика get_duration та же)
#         f_array = np.atleast_1d(f)
#         decay_rates = self._get_decay_rate(f_array)
#         durations = np.zeros_like(f_array, dtype=float)
#         for i, rate in enumerate(decay_rates):
#             if rate > 0:
#                 t_est = -np.log(threshold) / rate
#                 durations[i] = min(t_est, self.max_duration)
#             else:
#                 durations[i] = self.max_duration
#         return float(durations[0]) if np.isscalar(f) else durations

# # =====================================================================
# # 2. ХЕЛПЕРЫ И ФУНКЦИИ ОЧИСТКИ (Без изменений!)
# # =====================================================================

# # Типизацию TimeDomainEchoModel лучше заменить на более общую, чтобы подходили оба класса
# def get_echo_lengths(model, frequencies: Union[float, List[float], np.ndarray], sample_rate: float, threshold: float = 1e-4) -> Union[int, np.ndarray]:
#     durations = model.get_duration(frequencies, threshold)
#     lengths = np.ceil(np.atleast_1d(durations) * sample_rate).astype(int)
#     lengths = np.maximum(lengths, 1)
#     return int(lengths[0]) if np.isscalar(frequencies) else lengths

# def generate_echos(model, frequencies: Union[float, List[float], np.ndarray], sample_rate: float, threshold: float = 1e-4) -> Union[np.ndarray, List[np.ndarray]]:
#     is_scalar = np.isscalar(frequencies)
#     freqs = np.atleast_1d(frequencies)
#     lengths = get_echo_lengths(model, freqs, sample_rate, threshold)
#     lengths = np.atleast_1d(lengths)
    
#     echos = []
#     for f, length in zip(freqs, lengths):
#         t_axis = np.arange(length) / sample_rate
#         echo_array = model.compute_value(t_axis, f)
        
#         k_sum = np.sum(echo_array)
#         if k_sum > 0:
#             echo_array = echo_array / k_sum
#         echos.append(echo_array)
        
#     return echos[0] if is_scalar else echos

# def remove_echo_wiener2(
#     signal: np.ndarray, 
#     frequency: float, 
#     sample_rate: float, 
#     model, # <-- Теперь можно передавать любой из классов
#     eps_factor: float = 1e-2,
# ) -> np.ndarray:
#     N = len(signal)
    
#     # Теперь peak_idx будет всегда 0, но на всякий случай оставим `np.roll`
#     # для совместимости со старой моделью, если вдруг понадобится.
#     # Для CausalEchoModel он ничего не сделает, т.к. argmax(kernel) == 0.
#     raw_kernel = generate_echos(model, frequency, sample_rate)
#     peak_idx = int(np.argmax(raw_kernel))
#     kernel = np.roll(raw_kernel, -peak_idx)
#     K = len(kernel)
        
#     M = N + K - 1 
#     H = np.fft.rfft(kernel, n=M)
#     X = np.fft.rfft(signal, n=M)
    
#     eps = eps_factor * np.max(np.abs(H))
#     H_inv = np.conj(H) / (np.abs(H)**2 + eps**2)
#     signal_restored_full = np.fft.irfft(X * H_inv, n=M)
    
#     signal_restored = signal_restored_full[:N]
#     signal_restored = np.clip(signal_restored, 0, None)

#     return signal_restored

# def richardson_lucy_1d(signal: np.ndarray, frequency: float, sample_rate: float, model, iterations: int = 15) -> np.ndarray:
#     # (Эта функция без изменений, она будет работать с CausalEchoModel)
#     signal = np.clip(signal, 1e-24, None)
#     kernel = generate_echos(model, frequency, sample_rate)
#     peak_idx = int(np.argmax(kernel))
#     kernel_rev = kernel[::-1]
#     estimate = np.copy(signal)
#     for _ in range(iterations):
#         blurred = fftconvolve(estimate, kernel, mode='full')[peak_idx : peak_idx + len(signal)]
#         blurred[blurred == 0] = 1e-24
#         ratio = signal / blurred
#         rev_peak_idx = len(kernel) - 1 - peak_idx
#         correction = fftconvolve(ratio, kernel_rev, mode='full')[rev_peak_idx : rev_peak_idx + len(signal)]
#         estimate = estimate * correction
#     return estimate

# from BatSpec.Work.Function import SpecFunc

def convolveSpecWithEcho(spec: 'SpecFunc', echo: 'SpecFunc') -> 'SpecFunc':
    """
    Размывает базовую спектрограмму (spec) с помощью матрицы эха (echo).
    Свёртка происходит ТОЛЬКО по оси времени (независимо для каждой частоты).
    """
    import numpy as np
    from scipy.signal import fftconvolve

    # Безопасно извлекаем значения
    spec_unit, spec_vals = spec.values
    _, echo_vals = echo.values
    
    if hasattr(spec_vals, "numpy"): spec_vals = spec_vals.numpy()
    else: spec_vals = np.asarray(spec_vals)
        
    if hasattr(echo_vals, "numpy"): echo_vals = echo_vals.numpy()
    else: echo_vals = np.asarray(echo_vals)

    # Оси
    _, spec_time = spec.time
    _, spec_freq = spec.freq
    _, echo_time = echo.time
    _, echo_freq = echo.freq
    
    if hasattr(spec_freq, "numpy"): spec_freq = spec_freq.numpy()
    if hasattr(echo_freq, "numpy"): echo_freq = echo_freq.numpy()
    if hasattr(echo_time, "numpy"): echo_time = echo_time.numpy()

    # Получаем шаг по частоте
    _, df = spec.df
    
    # Подготавливаем результирующую матрицу (копия оригинала)
    out_vals = np.copy(spec_vals)
    
    # 1. Вычисляем смещение по частоте для поиска перекрывающихся столбцов
    f_offset = int(round(float(echo_freq[0] - spec_freq[0]) / float(df)))
    
    tgt_f_start = max(0, f_offset)
    tgt_f_end   = min(spec_vals.shape[1], f_offset + echo_vals.shape[1])
    
    # Если есть перекрытие по частотам
    if tgt_f_start < tgt_f_end:
        src_f_start = tgt_f_start - f_offset
        src_f_end   = tgt_f_end - f_offset
        
        # Извлекаем только совпадающие частотные полосы
        spec_overlap = spec_vals[:, tgt_f_start:tgt_f_end]
        echo_overlap = echo_vals[:, src_f_start:src_f_end]
        
        # 2. Быстрая свёртка по времени (axes=0 выполняет 1D свёртку для каждого столбца)
        conv_overlap = fftconvolve(spec_overlap, echo_overlap, mode='full', axes=0)
        
        # 3. Компенсация сдвига (выравнивание t=0)
        # Находим индекс "нуля" на оси времени эха
        zero_idx = int(np.abs(echo_time).argmin())
        
        # Обрезаем так, чтобы длина совпала с оригинальной
        T = len(spec_time)
        conv_overlap = conv_overlap[zero_idx : zero_idx + T, :]
        
        # 4. Записываем результат обратно в матрицу
        out_vals[:, tgt_f_start:tgt_f_end] = conv_overlap

    # Возвращаем новую SpecFunc с теми же осями, что и оригинал
    # (Обычно импульсная характеристика эха считается безразмерным множителем, 
    # поэтому оставляем единицу измерения от исходного spec)
    return SpecFunc(
        matrix=out_vals.astype(np.float32),
        time=spec.time[1], 
        freq=spec.freq[1], 
        unit=spec_unit
    )


TEST_ECHO_MODEL = AsymmetricEchoModel(
    decay_rate_base=200, f_ref=40000.0,
    freq_exp=2.0, time_power=20.0, attack_tau = 0.01
)