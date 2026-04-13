import sys
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

spec = getSpectrogramm().value

noise = etalonNoise(spec, 5)
denoised =  makeLog(subEtalonNoise(spec, noise, alpha=1))
updateVisLayer("DENOISED", denoised)
import numpy as np
from scipy.signal import find_peaks
from numpy.fft import rfft, irfft

# ==========================================
# РАЗМЕРЫ  (matrix shape = T × H)
# ==========================================
V_raw  = denoised.matrix.astype(np.float32)   # (T, F)
time   = denoised.time                         # длина T
freq   = denoised.freq                         # длина F

T_len  = V_raw.shape[0]   # время
F_bins = V_raw.shape[1]   # частота

dt = float(denoised.dt)
L  = max(1, int(round(0.060 / dt)))

# NMF требует V >= 0; после makeLog могут быть отрицательные значения
V_shifted = V_raw - V_raw.min()
V = (V_shifted / (V_shifted.max() + 1e-9)).T  # алгоритм работает в (F, T)

K = 5
epochs = 100
sparsity_H = 1.5
sparsity_W = 0.5

N_fft = int(2 ** np.ceil(np.log2(T_len + L)))

print(f"Conv-NMF  |  F={F_bins}, T={T_len}, K={K}, L={L}, N_fft={N_fft}")

# ==========================================
# FFT-СВЁРТКИ  (всё внутри работает с (F, T))
# ==========================================
def fft_reconstruct(W_mat, H_mat):
    H_f = rfft(H_mat, n=N_fft, axis=1)                        # (K, Nf)
    out = np.zeros((F_bins, N_fft // 2 + 1), dtype=np.complex64)
    for k in range(K):
        w_pad = np.zeros((F_bins, N_fft), dtype=np.float32)
        w_pad[:, :L] = W_mat[:, k, :]
        out += rfft(w_pad, axis=1) * H_f[k]
    return irfft(out, n=N_fft, axis=1)[:, :T_len].astype(np.float32)

def fft_correlate_VW_for_H(V_mat, W_mat, V_hat):
    V_f  = rfft(V_mat,  n=N_fft, axis=1)                      # (F, Nf)
    Vh_f = rfft(V_hat,  n=N_fft, axis=1)
    num_H = np.zeros((K, T_len), dtype=np.float32)
    den_H = np.zeros((K, T_len), dtype=np.float32)
    for k in range(K):
        w_pad = np.zeros((F_bins, N_fft), dtype=np.float32)
        w_pad[:, :L] = W_mat[:, k, :]
        W_f = rfft(w_pad, axis=1)
        num_H[k] = irfft(np.sum(np.conj(W_f) * V_f,  axis=0), n=N_fft)[:T_len]
        den_H[k] = irfft(np.sum(np.conj(W_f) * Vh_f, axis=0), n=N_fft)[:T_len]
    return num_H, den_H

def fft_correlate_VH_for_W(V_mat, H_mat, V_hat):
    V_f  = rfft(V_mat,  n=N_fft, axis=1)                      # (F, Nf)
    Vh_f = rfft(V_hat,  n=N_fft, axis=1)
    H_f  = rfft(H_mat,  n=N_fft, axis=1)                      # (K, Nf)
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

    num_H, den_H = fft_correlate_VW_for_H(V, W, V_hat)
    H = np.clip(H * (np.clip(num_H, 0, None) / (den_H + sparsity_H + 1e-9)), 0, None)

    V_hat = fft_reconstruct(W, H)
    num_W, den_W = fft_correlate_VH_for_W(V, H, V_hat)
    W = np.clip(W * (np.clip(num_W, 0, None) / (den_W + sparsity_W + 1e-9)), 0, None)

    for k in range(K):
        mx = float(W[:, k, :].max())
        if mx > 1e-9:
            W[:, k, :] /= mx
            H[k, :]    *= mx

    if (epoch + 1) % 1 == 0:
        err = float(np.linalg.norm(V - V_hat))
        print(f"Эпоха {epoch+1:3d} | Ошибка: {err:.4f}")

print("Готово!")

# ==========================================
# ПАТТЕРНЫ — возвращаем в (T, F) для SpecFunc
# ==========================================
for k in range(K):
    t_pat   = time[0] + np.arange(L) * dt
    pattern = W[:, k, :].T                    # (L, F) → (T=L, H=F)
    updateVisLayer(f"PATTERN_{k}", SpecFunc(pattern, freq, t_pat))

# ==========================================
# РЕКОНСТРУКЦИИ — тоже (T, F)
# ==========================================
for k in range(K):
    act      = H[k, :]
    max_val  = float(act.max()) or 1.0
    peaks, _ = find_peaks(act, height=max_val * 0.2, distance=L)

    rec = np.zeros((F_bins, T_len), dtype=np.float32)
    for t_peak in peaks:
        t_end = min(t_peak + L, T_len)
        rec[:, t_peak:t_end] += W[:, k, :t_end - t_peak] * act[t_peak]

    updateVisLayer(f"RECONSTRUCTED_{k}", SpecFunc(rec.T, freq, time))  # .T → (T, F)