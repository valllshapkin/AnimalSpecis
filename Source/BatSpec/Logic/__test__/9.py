import numpy as np

from BatSpec.API.spec import getSpectrogramm, updateVisLayer
from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Logic.SpecWorker import subEtalonNoise, makeLog, normalizeFrecZ
from BatSpec.Logic.TransitionWorker import etalonNoise

# ==========================================
# ЗАГРУЗКА
# ==========================================
spec     = getSpectrogramm().value
noise    = etalonNoise(spec, 5)
denoised = subEtalonNoise(spec, noise, alpha=1)
normalized = normalizeFrecZ(denoised, noise)

# ==========================================
# КРОКОЗЯБРА: mat[t-3]*mat[t-2]*mat[t-1]*mat[t+1]*mat[t+2]*mat[t+3]
# ==========================================
dt    = float(normalized.dt)
shift = int(round(0.005 / dt))

mat = normalized.matrix  # (T, F)
T   = mat.shape[0]
s   = shift

# все 6 слагаемых выровнены по времени: берём срез [3s : T-3s]
m_n3 = mat[:T - 6*s:  ][::1]  # проще через явные срезы:
m_n3 = mat[0*s : T - 6*s, :]
m_n2 = mat[1*s : T - 5*s, :]
m_n1 = mat[2*s : T - 4*s, :]
m_p1 = mat[4*s : T - 2*s, :]  # пропускаем 3s (центр t=3s)
m_p2 = mat[5*s : T - 1*s, :]
m_p3 = mat[6*s : T - 0*s, :]

corr      = m_n3 * m_n2 * m_n1 * m_p1 * m_p2 * m_p3
time_corr = normalized.time[3*s : T - 3*s]

updateVisLayer("CROKOZЯБРА_6x", makeLog(SpecFunc(corr  / 85, denoised.freq, time_corr)))