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

from BatSpec.Logic.TransitionWorker import integrateFreq
func = integrateFreq(DENOISED)
from scipy.signal import find_peaks

peaks, props = find_peaks(func.data,
                          height=None,      # ← можно задать минимальную высоту
                          prominence=0.2,   # очень рекомендуется
                          width=10,         # минимальная ширина
                          rel_height=0.5,   # на какой высоте считать ширину (0.5 = половина)
                          distance=None)    # можно добавить

print("Доступные ключи:", list(props.keys()))

import matplotlib.pyplot as plt
plt.plot(func.time_axis, func.data, 'b-', label='Сигнал')
plt.plot(func.time_axis[peaks], func.data[peaks], 'ro', label='Пики')

# Подписи с информацией
for i, peak in enumerate(peaks):
    plt.text(func.time_axis[peak], func.data[peak]+0.1, 
             f'h={props["peak_heights"][i]:.2f}\np={props["prominences"][i]:.2f}\nw={props["widths"][i]:.1f}',
             ha='center', fontsize=9)

plt.legend()
plt.grid()
plt.show()


# from BatSpec.Logic.VisDebug import show_spec_labels,  RegisterGlobFunc; RegisterGlobFunc()

# show_spec_labels(spec, LABELS)
