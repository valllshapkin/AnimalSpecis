import librosa
import numpy as np

# Предположим, D - ваша спектрограмма (600000, 300)
# librosa обычно ожидает форму (Частоты, Время). 
# Если у вас 600000 - это время, а 300 - частоты, нужно транспонировать матрицу.

from BatSpec.API import getSpectrogramm, updateVisLayer
from BatSpec.Logic.SpecWorker import (
    subEtalonNoise, makeLog, normalizeFrecZ, makeBinarization,
    gradientSquared, removeEchoWiener, enhanceCurvesGabor, labelWithHDBSCAN
)
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise
from BatSpec.Logic.Echo import CausalEchoModel

from pathlib import Path
from BatSpec.Logic.Functions import TimeFunc, SpecFunc
from BatSpec.Logic.ConvWindow import Window
from BatSpec.Logic.SoundWorker import localRMS
from BatSpec.Logic.SpecWorker import subEtalonNoise, normalizeFrecZ, harmonicPercussive
from BatSpec.Logic.TransitionWorker import makeSpec, etalonNoise
from BatSpec.Logic.Loader import loadRecord
from BatSpec.Logic.SaveIntegral import SaveIntegralEnergy
from scipy.signal import windows
import numpy as np

record: TimeFunc = loadRecord(Path("/MainData/Repo/golonchenroppi/BatSpecNew/Resources/r09_23/MYODAS_20230624_005134.wav"))
locrms: TimeFunc = localRMS(record, Window(windows.blackmanharris, 0.001, norm="area"))
spec: SpecFunc = makeSpec(record, Window(windows.hann, 0.002, norm="energy"), overlap=0.8, bins=300)

print(
    SaveIntegralEnergy(record), 
    SaveIntegralEnergy(locrms), 
    SaveIntegralEnergy(spec),
    "Эти числа должны быть равны в аналитике быть почти равными", 
    sep="\n"
)

noise = etalonNoise(spec, 5)

DENOISED = subEtalonNoise(spec, noise, alpha=1)
# updateVisLayer("DENOISED", makeLog(DENOISED))

GABOR = enhanceCurvesGabor(
    DENOISED,
    ksize=17, sigma=1, 
    lambd=5, pre_blur=0
)
# updateVisLayer("GABOR", makeLog(GABOR))

WINNER = removeEchoWiener(GABOR, CausalEchoModel(
    decay_rate_base=20, f_ref=40000.0,
    freq_exp=2.0, time_power=20.0,
), eps_factor=1e-5)
# updateVisLayer("WINNER", makeLog(WINNER))

BINARY = makeBinarization(normalizeFrecZ(WINNER, noise), trashhold=5)
# updateVisLayer("BINARY", BINARY)

LABELS = labelWithHDBSCAN(
    bin_spec=BINARY, 
    spec=makeLog(DENOISED), 
    min_cluster_size = 50, 
    min_samples = 50, 
    epsilon = 0,
    scale_weights = (10.0, 10.0, 20)
)
# updateVisLayer("LABELS", LABELS)



from BatSpec.Logic.VisDebug import show_spec_labels,  RegisterGlobFunc; RegisterGlobFunc()

show_spec_labels(spec, LABELS)
