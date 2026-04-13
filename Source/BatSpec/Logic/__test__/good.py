import numpy as np

from BatSpec.API.spec import getSpectrogramm, updateVisLayer
from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Logic.SpecWorker import subEtalonNoise, makeLog, normalizeFrecZ, gaussianBlur
from BatSpec.Logic.TransitionWorker import etalonNoise
from scipy.ndimage import zoom

spec       = getSpectrogramm().value
noise      = etalonNoise(spec, 5)
denoised   = subEtalonNoise(spec, noise, alpha=1)
normalized = normalizeFrecZ(denoised, noise)

STD_HZ_PER_PX = 2400.0
STD_S_PER_PX  = 0.001
STD_MAX_FREQ  = 120_000.0

t_max  = float(normalized.time[-1])
std_F  = int(round(STD_MAX_FREQ / STD_HZ_PER_PX))
std_T  = int(round(t_max        / STD_S_PER_PX))

mat_src = normalized.matrix.astype(np.float32)
mat_std = zoom(mat_src, (std_T / mat_src.shape[0], std_F / mat_src.shape[1]), order=1)

std_freq = np.linspace(0.0, STD_MAX_FREQ, std_F, endpoint=False)
std_time = np.linspace(0.0, t_max,        std_T, endpoint=False)
dt       = float(std_time[1] - std_time[0])

print(f"STD: T={std_T}, F={std_F}, dt={dt*1000:.1f}ms")
updateVisLayer("NORMALIZED_STD", SpecFunc(mat_std, std_freq, std_time))

max_shift = int(round(0.200 / dt))
W         = 1
T, F      = mat_std.shape

t0 = max_shift + W
t1 = T - max_shift - W
N  = t1 - t0

t_idx = np.arange(t0, t1)[:, None]
s_idx = np.arange(1, max_shift + 1)[None, :]
zeros = np.zeros_like(s_idx)

offsets = np.stack([
    t_idx - s_idx - 1,
    t_idx - s_idx,
    t_idx - s_idx + 1,
    t_idx - 1 + zeros,    # broadcast до (N, max_shift)
    t_idx     + zeros,
    t_idx + 1 + zeros,
    t_idx + s_idx - 1,
    t_idx + s_idx,
    t_idx + s_idx + 1,
], axis=2)  # (N, max_shift, 9)

shift_axis = np.arange(1, max_shift + 1) * dt
time_corr  = std_time[t0:t1]

freq_idx_40 = int(np.argmin(np.abs(std_freq - 40_000.0)))
gathered_40 = mat_std[:, freq_idx_40][offsets]
corr_40     = np.abs(np.prod(gathered_40, axis=2)).astype(np.float32)
updateVisLayer("AUTOCORR_40kHz", makeLog(SpecFunc(corr_40, shift_axis, time_corr)))

BATCH    = 10
corr_all = np.zeros((N, max_shift), dtype=np.float64)

for f_start in range(0, F, BATCH):
    f_end    = min(f_start + BATCH, F)
    chunk    = mat_std[:, f_start:f_end]
    gathered = chunk[offsets, :]
    corr_all += np.abs(np.prod(gathered, axis=2)).sum(axis=2)
    print(f"  freq {f_start}..{f_end}/{F}")

S = SpecFunc(corr_all.astype(np.float32), shift_axis, time_corr)
updateVisLayer("AUTOCORR_ALL", makeLog(gaussianBlur(S)))