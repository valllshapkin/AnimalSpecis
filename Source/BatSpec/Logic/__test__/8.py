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
L  = max(1, int(round(0.080 / dt)))

V_shifted = V_raw - V_raw.min()
V = (V_shifted / (V_shifted.max() + 1e-9)).T      # (F, T) для алгоритма

# --- маска ---
M = make_weight_mask(V, threshold_quantile=0.96)
updateVisLayer("MASK", SpecFunc(M.T, freq, time))  # показываем маску

K          = 5
epochs     = 100
sparsity_H = 1.5
sparsity_W = 0.5
N_fft      = int(2 ** np.ceil(np.log2(T_len + L)))

print(f"Conv-NMF  |  F={F_bins}, T={T_len}, K={K}, L={L}, N_fft={N_fft}")
print(f"Маска     |  сигнал={M.mean()*100:.1f}% пикселей")

# ==========================================
# FFT-СВЁРТКИ С МАСКОЙ
# ==========================================
def fft_reconstruct(W_mat, H_mat):
    H_f = rfft(H_mat, n=N_fft, axis=1)
    out = np.zeros((F_bins, N_fft // 2 + 1), dtype=np.complex64)
    for k in range(K):
        w_pad = np.zeros((F_bins, N_fft), dtype=np.float32)
        w_pad[:, :L] = W_mat[:, k, :]
        out += rfft(w_pad, axis=1) * H_f[k]
    return irfft(out, n=N_fft, axis=1)[:, :T_len].astype(np.float32)

def fft_correlate_VW_for_H(V_mat, W_mat, V_hat, M):
    MV   = M * V_mat
    MVh  = M * V_hat
    V_f  = rfft(MV,  n=N_fft, axis=1)
    Vh_f = rfft(MVh, n=N_fft, axis=1)
    num_H = np.zeros((K, T_len), dtype=np.float32)
    den_H = np.zeros((K, T_len), dtype=np.float32)
    for k in range(K):
        w_pad = np.zeros((F_bins, N_fft), dtype=np.float32)
        w_pad[:, :L] = W_mat[:, k, :]
        W_f = rfft(w_pad, axis=1)
        num_H[k] = irfft(np.sum(np.conj(W_f) * V_f,  axis=0), n=N_fft)[:T_len]
        den_H[k] = irfft(np.sum(np.conj(W_f) * Vh_f, axis=0), n=N_fft)[:T_len]
    return num_H, den_H

def fft_correlate_VH_for_W(V_mat, H_mat, V_hat, M):
    MV   = M * V_mat
    MVh  = M * V_hat
    V_f  = rfft(MV,  n=N_fft, axis=1)
    Vh_f = rfft(MVh, n=N_fft, axis=1)
    H_f  = rfft(H_mat, n=N_fft, axis=1)
    num_W = np.zeros((F_bins, K, L), dtype=np.float32)
    den_W = np.zeros((F_bins, K, L), dtype=np.float32)
    for k in range(K):
        num_W[:, k, :] = irfft(np.conj(H_f[k]) * V_f,  n=N_fft, axis=1)[:, :L]
        den_W[:, k, :] = irfft(np.conj(H_f[k]) * Vh_f, n=N_fft, axis=1)[:, :L]
    return num_W, den_W

# ==========================================
# ИНИЦИАЛИЗАЦИЯ
# ==========================================
rng = np.random.default_rng(42)
W = rng.random((F_bins, K, L)).astype(np.float32)
H = rng.random((K, T_len)).astype(np.float32)

# ==========================================
# ОБУЧЕНИЕ
# ==========================================
for epoch in range(epochs):
    V_hat = fft_reconstruct(W, H)

    num_H, den_H = fft_correlate_VW_for_H(V, W, V_hat, M)
    H = np.clip(H * (np.clip(num_H, 0, None) / (den_H + sparsity_H + 1e-9)), 0, None)

    V_hat = fft_reconstruct(W, H)
    num_W, den_W = fft_correlate_VH_for_W(V, H, V_hat, M)
    W = np.clip(W * (np.clip(num_W, 0, None) / (den_W + sparsity_W + 1e-9)), 0, None)

    for k in range(K):
        mx = float(W[:, k, :].max())
        if mx > 1e-9:
            W[:, k, :] /= mx
            H[k, :]    *= mx

    if (epoch + 1) % 10 == 0:
        err = float(np.linalg.norm((V - V_hat) * M))   # ошибка только на сигнале
        print(f"Эпоха {epoch+1:3d} | Ошибка (сигнал): {err:.4f}")

print("Готово!")

# ==========================================
# ПАТТЕРНЫ
# ==========================================
for k in range(K):
    t_pat = time[0] + np.arange(L) * dt
    updateVisLayer(f"PATTERN_{k}", SpecFunc(W[:, k, :].T, freq, t_pat))

# ==========================================
# РЕКОНСТРУКЦИИ
# ==========================================
for k in range(K):
    act      = H[k, :]
    max_val  = float(act.max()) or 1.0
    peaks, _ = find_peaks(act, height=max_val * 0.2, distance=L)

    rec_FT = np.zeros((F_bins, T_len), dtype=np.float32)
    for t_peak in peaks:
        t_end = min(t_peak + L, T_len)
        rec_FT[:, t_peak:t_end] += W[:, k, :t_end - t_peak] * act[t_peak]

    rec_src = from_std(SpecFunc(rec_FT.T, freq, time))
    updateVisLayer(f"RECONSTRUCTED_{k}", rec_src)