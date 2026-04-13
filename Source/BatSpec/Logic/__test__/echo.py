import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import binary_dilation, zoom
from numpy.fft import rfft, irfft

from BatSpec.API.spec import getSpectrogramm, updateVisLayer
from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise
from BatSpec.Logic.SpecWorker import subEtalonNoise, makeLog
from BatSpec.API.spec import getSpectrogramm, updateVisLayer
from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Logic.TransitionWorker import (
    clusterCallsDBSCAN, createClusterBasis, extractCalls, extractClusterPattern, findPeaks,
    specDistanceNormalized
)
from BatSpec.Logic.SpecWorker import (
    subEtalonNoise, normalizeFrecZ, makeLog
)
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise

from BatSpec.Logic.SpecWorker import subEtalonNoise
from BatSpec.Logic.TransitionWorker import etalonNoise, integrateFreq

# ==========================================
# СТАНДАРТ
# ==========================================
STD_HZ_PER_PX = 1200.0
STD_S_PER_PX  = 0.001
STD_MAX_FREQ  = 120_000.0

def make_std_transforms(freq: np.ndarray, time: np.ndarray):
    freq = np.asarray(freq, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)

    t_max    = float(time[-1])
    std_F    = int(round(STD_MAX_FREQ / STD_HZ_PER_PX))
    std_T    = int(round(t_max        / STD_S_PER_PX))
    std_freq = np.linspace(0.0, STD_MAX_FREQ, std_F, endpoint=False)
    std_time = np.linspace(0.0, t_max,        std_T, endpoint=False)

    src_f_min = float(freq[0])
    src_F     = len(freq)

    def _zoom_FT(mat_FT, dst_F, dst_T):
        zf = dst_F / mat_FT.shape[0]
        zt = dst_T / mat_FT.shape[1]
        zoomed = zoom(mat_FT.astype(np.float32), (zf, zt), order=1)
        out = np.zeros((dst_F, dst_T), dtype=np.float32)
        cf = min(zoomed.shape[0], dst_F)
        ct = min(zoomed.shape[1], dst_T)
        out[:cf, :ct] = zoomed[:cf, :ct]
        return out

    def to_std(spec) -> SpecFunc:
        mat_FT = spec.matrix.astype(np.float32).T     # (F, T)
        zoomed = _zoom_FT(mat_FT, std_F, std_T)
        f_offset = int(round(src_f_min / STD_HZ_PER_PX))
        if f_offset > 0:
            shifted = np.zeros_like(zoomed)
            end = min(f_offset + zoomed.shape[0], std_F)
            shifted[f_offset:end, :] = zoomed[:end - f_offset, :]
            zoomed = shifted
        return SpecFunc(zoomed.T, std_freq, std_time)

    def from_std(spec) -> SpecFunc:
        mat_FT = spec.matrix.astype(np.float32).T     # (F_std, T_std)
        f_offset = int(round(src_f_min / STD_HZ_PER_PX))
        if f_offset > 0:
            shifted = np.zeros_like(mat_FT)
            end = min(f_offset + src_F, std_F)
            shifted[:end - f_offset, :] = mat_FT[f_offset:end, :]
            mat_FT = shifted
        restored = _zoom_FT(mat_FT, src_F, len(time))
        return SpecFunc(restored.T, freq, time)

    return to_std, from_std

# ==========================================
# МАСКА ВЕСОВ
# ==========================================
def make_weight_mask(V_FT: np.ndarray, threshold_quantile: float = 0.75) -> np.ndarray:
    """
    V_FT : (F, T) нормализованная [0,1]
    Возвращает (F, T) маску: 1 = сигнал, 0 = шум
    """
    threshold = float(np.quantile(V_FT, threshold_quantile))
    mask = (V_FT >= threshold).astype(np.float32)
    mask = binary_dilation(mask, iterations=3).astype(np.float32)
    return mask

# ==========================================
# ЗАГРУЗКА И ПРЕДОБРАБОТКА
# ==========================================
spec     = getSpectrogramm().value
noise    = etalonNoise(spec, 5)
denoised = makeLog(subEtalonNoise(spec, noise, alpha=1))
updateVisLayer("DENOISED", denoised)

to_std, from_std = make_std_transforms(denoised.freq, denoised.time)
denoised_std     = to_std(denoised)
updateVisLayer("DENOISED_STD", denoised_std)

# ==========================================
# ПОДГОТОВКА МАТРИЦ
# ==========================================
V_raw  = denoised_std.matrix.astype(np.float32)   # (T, F)
time   = denoised_std.time
freq   = denoised_std.freq

T_len  = V_raw.shape[0]
F_bins = V_raw.shape[1]

dt = float(denoised_std.dt)

# Размеры окон
L_w = max(1, int(round(0.060 / dt)))  # Длина эталонного колла (например 60 мс)
L_e = max(1, int(round(0.040 / dt)))  # Длина окна эха/размытия (например 40 мс)

V_shifted = V_raw - V_raw.min()
V = (V_shifted / (V_shifted.max() + 1e-9)).T      # (F, T) для алгоритма

# --- маска ---
M = make_weight_mask(V, threshold_quantile=0.96)
updateVisLayer("MASK", SpecFunc(M.T, freq, time))

K          = 10  # Количество видов (коллов в библиотеке)
epochs     = 100
sparsity_H = 1.5
sparsity_E = 0.5 # Регуляризация для эха (чтобы оно не разрасталось)

# N_fft должен вмещать сигнал + колл + эхо
N_fft = int(2 ** np.ceil(np.log2(T_len + L_w + L_e)))

print(f"Nested Conv-NMF | F={F_bins}, T={T_len}, K={K}, L_w={L_w}, L_e={L_e}, N_fft={N_fft}")

# ==========================================
# ИНИЦИАЛИЗАЦИЯ
# ==========================================
rng = np.random.default_rng(42)

# 1. ФИКСИРОВАННАЯ БИБЛИОТЕКА ЭТАЛОНОВ (W)
# ЗДЕСЬ НУЖНО ЗАГРУЗИТЬ ТВОИ ЭТАЛОНЫ. 
# Для примера я генерирую их случайно, но в реальности они НЕ обновляются.
W = rng.random((F_bins, K, L_w)).astype(np.float32)
# Нормализуем библиотеку один раз
for k in range(K):
    mx = float(W[:, k, :].max())
    if mx > 1e-9: W[:, k, :] /= mx

# Предрасчет FFT для W (так как W не меняется, делаем это один раз!)
W_pad = np.zeros((F_bins, K, N_fft), dtype=np.float32)
W_pad[:, :, :L_w] = W
W_f = rfft(W_pad, axis=2) # (F, K, N_fft/2 + 1)

# 2. ОБУЧАЕМЫЕ ПАРАМЕТРЫ (H и E)
H = rng.random((K, T_len)).astype(np.float32)
E = rng.random((F_bins, L_e)).astype(np.float32)

# ==========================================
# ОБУЧЕНИЕ (ОПТИМИЗИРОВАННОЕ ЧЕРЕЗ FFT)
# ==========================================
for epoch in range(epochs):
    # --- Шаг 1: Подготовка общих FFT ---
    H_f = rfft(H, n=N_fft, axis=1) # (K, N_fft)
    
    E_pad = np.zeros((F_bins, N_fft), dtype=np.float32)
    E_pad[:, :L_e] = E
    E_f = rfft(E_pad, axis=1)      # (F, N_fft)

    # Комбинированный шаблон: W_tilde = E * W (свертка эха и колла)
    # E_f[:, np.newaxis, :] делает broadcast по оси K
    W_tilde_f = E_f[:, np.newaxis, :] * W_f # (F, K, N_fft)

    # Реконструкция V_hat = W_tilde * H
    V_hat_f = np.sum(W_tilde_f * H_f[np.newaxis, :, :], axis=1) # (F, N_fft)
    V_hat = irfft(V_hat_f, n=N_fft, axis=1)[:, :T_len]
    V_hat = np.clip(V_hat, 0, None)

    # Ошибка с учетом маски
    MV  = M * V
    MVh = M * V_hat
    MV_f  = rfft(MV, n=N_fft, axis=1)
    MVh_f = rfft(MVh, n=N_fft, axis=1)

    # --- Шаг 2: Обновление активаций (H) ---
    num_H = np.zeros((K, T_len), dtype=np.float32)
    den_H = np.zeros((K, T_len), dtype=np.float32)
    for k in range(K):
        # Корреляция V и W_tilde
        num_H[k] = irfft(np.sum(np.conj(W_tilde_f[:, k, :]) * MV_f,  axis=0), n=N_fft)[:T_len]
        den_H[k] = irfft(np.sum(np.conj(W_tilde_f[:, k, :]) * MVh_f, axis=0), n=N_fft)[:T_len]
    
    H = np.clip(H * (np.clip(num_H, 0, None) / (den_H + sparsity_H + 1e-9)), 0, None)
    
    # Обновляем H_f после изменения H для шага с E
    H_f = rfft(H, n=N_fft, axis=1)

    # --- Шаг 3: Обновление Эха (E) ---
    # Считаем чистые пики без эха: P = W * H
    P_f = np.sum(W_f * H_f[np.newaxis, :, :], axis=1) # (F, N_fft)
    
    # Корреляция P с V_hat и V
    num_E = irfft(np.conj(P_f) * MV_f,  n=N_fft, axis=1)[:, :L_e]
    den_E = irfft(np.conj(P_f) * MVh_f, n=N_fft, axis=1)[:, :L_e]

    E = np.clip(E * (np.clip(num_E, 0, None) / (den_E + sparsity_E + 1e-9)), 0, None)

    # --- Шаг 4: Нормализация (чтобы H и E не улетали в космос) ---
    mx_E = float(E.max())
    if mx_E > 1e-9:
        E /= mx_E
        H *= mx_E  # Переносим масштаб в H

    # --- Логирование ---
    if (epoch + 1) % 10 == 0:
        err = float(np.linalg.norm((V - V_hat) * M))
        print(f"Эпоха {epoch+1:3d} | Ошибка на маске: {err:.4f}")

print("Обучение завершено!")

# ==========================================
# ВИЗУАЛИЗАЦИЯ
# ==========================================

# 1. Показать окно ЭХА (оно общее для всех)
t_echo = time[0] + np.arange(L_e) * dt
updateVisLayer("ECHO_WINDOW", SpecFunc(E.T, freq, t_echo))

# 2. Восстановление (чистые коллы + с эхом)
for k in range(K):
    act      = H[k, :]
    max_val  = float(act.max()) or 1.0
    peaks, _ = find_peaks(act, height=max_val * 0.1, distance=L_w)

    # Чистые коллы (без эха)
    rec_pure_FT = np.zeros((F_bins, T_len), dtype=np.float32)
    for t_peak in peaks:
        t_end = min(t_peak + L_w, T_len)
        rec_pure_FT[:, t_peak:t_end] += W[:, k, :t_end - t_peak] * act[t_peak]
    
    updateVisLayer(f"REC_PURE_{k}", from_std(SpecFunc(rec_pure_FT.T, freq, time)))

    # Коллы с учетом эха
    rec_echo_FT = np.zeros((F_bins, T_len), dtype=np.float32)
    # Считаем W_tilde = E * W_k
    w_k_tilde = irfft(E_f * W_f[:, k, :], n=N_fft, axis=1)[:, :L_w + L_e - 1]
    
    for t_peak in peaks:
        t_end = min(t_peak + (L_w + L_e - 1), T_len)
        chunk_len = t_end - t_peak
        rec_echo_FT[:, t_peak:t_end] += w_k_tilde[:, :chunk_len] * act[t_peak]

    updateVisLayer(f"REC_WITH_ECHO_{k}", from_std(SpecFunc(rec_echo_FT.T, freq, time)))