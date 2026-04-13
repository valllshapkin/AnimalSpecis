import numpy as np
from scipy.signal import fftconvolve
from typing import Union, List

# =====================================================================
# 1. МОДЕЛИ ЭХА
# =====================================================================

class AsymmetricEchoModel:
    """
    Твоя оригинальная модель с медленной "атакой" (нарастанием) и затуханием.
    Пик НЕ находится на t=0, что вызывает фазовые сдвиги в деконволюции.
    """
    def __init__(self, decay_rate_base: float, f_ref: float, freq_exp: float, time_power: float = 0.0, max_duration: float = 10.0):
        self.decay_rate_base = decay_rate_base
        self.f_ref = f_ref
        self.freq_exp = freq_exp
        self.time_power = time_power
        self.max_duration = max_duration

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(decay_rate_base={self.decay_rate_base!r}, ...)")

    def _get_decay_rate(self, f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.decay_rate_base * (np.abs(f) / self.f_ref) ** self.freq_exp

    def compute_value(self, t: Union[float, np.ndarray], f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        decay_rate = self._get_decay_rate(f)
        exp_decay = np.exp(-decay_rate * t)
        power_decay = 1.0 / ((t + 1.0) ** self.time_power)
        attack_tau = 0.0002
        attack_factor = 1.0 - np.exp(-t / attack_tau)
        return attack_factor * power_decay * exp_decay

    def get_duration(self, f: Union[float, np.ndarray], threshold: float = 1e-4) -> Union[float, np.ndarray]:
        # (Логика get_duration остается той же, что и у тебя)
        # ...
        f_array = np.atleast_1d(f)
        decay_rates = self._get_decay_rate(f_array)
        durations = np.zeros_like(f_array, dtype=float)
        for i, rate in enumerate(decay_rates):
            if rate > 0:
                t_est = -np.log(threshold) / rate
                durations[i] = min(t_est, self.max_duration)
            else:
                durations[i] = self.max_duration
        return float(durations[0]) if np.isscalar(f) else durations


class CausalEchoModel:
    """
    Новая, строго каузальная модель. Пик всегда на t=0.
    Не имеет "атаки", сразу начинается с максимального затухания.
    Идеально подходит для деконволюции (Винер, Ричардсон-Люси).
    """
    def __init__(self, decay_rate_base: float, f_ref: float, freq_exp: float, time_power: float = 0.0, max_duration: float = 10.0):
        self.decay_rate_base = decay_rate_base
        self.f_ref = f_ref
        self.freq_exp = freq_exp
        self.time_power = time_power
        self.max_duration = max_duration

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(decay_rate_base={self.decay_rate_base!r}, ...)")

    def _get_decay_rate(self, f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.decay_rate_base * (np.abs(f) / self.f_ref) ** self.freq_exp

    def compute_value(self, t: Union[float, np.ndarray], f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # Убираем attack_factor. Теперь при t=0, результат равен 1.0
        decay_rate = self._get_decay_rate(f)
        exp_decay = np.exp(-decay_rate * t)
        power_decay = 1.0 / ((t + 1.0) ** self.time_power)
        return power_decay * exp_decay

    def get_duration(self, f: Union[float, np.ndarray], threshold: float = 1e-4) -> Union[float, np.ndarray]:
        # (Логика get_duration та же)
        f_array = np.atleast_1d(f)
        decay_rates = self._get_decay_rate(f_array)
        durations = np.zeros_like(f_array, dtype=float)
        for i, rate in enumerate(decay_rates):
            if rate > 0:
                t_est = -np.log(threshold) / rate
                durations[i] = min(t_est, self.max_duration)
            else:
                durations[i] = self.max_duration
        return float(durations[0]) if np.isscalar(f) else durations

# =====================================================================
# 2. ХЕЛПЕРЫ И ФУНКЦИИ ОЧИСТКИ (Без изменений!)
# =====================================================================

# Типизацию TimeDomainEchoModel лучше заменить на более общую, чтобы подходили оба класса
def get_echo_lengths(model, frequencies: Union[float, List[float], np.ndarray], sample_rate: float, threshold: float = 1e-4) -> Union[int, np.ndarray]:
    durations = model.get_duration(frequencies, threshold)
    lengths = np.ceil(np.atleast_1d(durations) * sample_rate).astype(int)
    lengths = np.maximum(lengths, 1)
    return int(lengths[0]) if np.isscalar(frequencies) else lengths

def generate_echos(model, frequencies: Union[float, List[float], np.ndarray], sample_rate: float, threshold: float = 1e-4) -> Union[np.ndarray, List[np.ndarray]]:
    is_scalar = np.isscalar(frequencies)
    freqs = np.atleast_1d(frequencies)
    lengths = get_echo_lengths(model, freqs, sample_rate, threshold)
    lengths = np.atleast_1d(lengths)
    
    echos = []
    for f, length in zip(freqs, lengths):
        t_axis = np.arange(length) / sample_rate
        echo_array = model.compute_value(t_axis, f)
        
        k_sum = np.sum(echo_array)
        if k_sum > 0:
            echo_array = echo_array / k_sum
        echos.append(echo_array)
        
    return echos[0] if is_scalar else echos

def remove_echo_wiener2(
    signal: np.ndarray, 
    frequency: float, 
    sample_rate: float, 
    model, # <-- Теперь можно передавать любой из классов
    eps_factor: float = 1e-2,
) -> np.ndarray:
    N = len(signal)
    
    # Теперь peak_idx будет всегда 0, но на всякий случай оставим `np.roll`
    # для совместимости со старой моделью, если вдруг понадобится.
    # Для CausalEchoModel он ничего не сделает, т.к. argmax(kernel) == 0.
    raw_kernel = generate_echos(model, frequency, sample_rate)
    peak_idx = int(np.argmax(raw_kernel))
    kernel = np.roll(raw_kernel, -peak_idx)
    K = len(kernel)
        
    M = N + K - 1 
    H = np.fft.rfft(kernel, n=M)
    X = np.fft.rfft(signal, n=M)
    
    eps = eps_factor * np.max(np.abs(H))
    H_inv = np.conj(H) / (np.abs(H)**2 + eps**2)
    signal_restored_full = np.fft.irfft(X * H_inv, n=M)
    
    signal_restored = signal_restored_full[:N]
    signal_restored = np.clip(signal_restored, 0, None)

    return signal_restored

def richardson_lucy_1d(signal: np.ndarray, frequency: float, sample_rate: float, model, iterations: int = 15) -> np.ndarray:
    # (Эта функция без изменений, она будет работать с CausalEchoModel)
    signal = np.clip(signal, 1e-24, None)
    kernel = generate_echos(model, frequency, sample_rate)
    peak_idx = int(np.argmax(kernel))
    kernel_rev = kernel[::-1]
    estimate = np.copy(signal)
    for _ in range(iterations):
        blurred = fftconvolve(estimate, kernel, mode='full')[peak_idx : peak_idx + len(signal)]
        blurred[blurred == 0] = 1e-24
        ratio = signal / blurred
        rev_peak_idx = len(kernel) - 1 - peak_idx
        correction = fftconvolve(ratio, kernel_rev, mode='full')[rev_peak_idx : rev_peak_idx + len(signal)]
        estimate = estimate * correction
    return estimate