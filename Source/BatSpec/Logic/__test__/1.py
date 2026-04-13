from pathlib import Path
from BatSpec.Logic.Functions import TimeFunc, SpecFunc
from BatSpec.Logic.ConvWindow import Window
from BatSpec.Logic.SoundWorker import localRMS
from BatSpec.Logic.SpecWorker import subEtalonNoise, normalizeFrecZ, harmonicPercussive
from BatSpec.Logic.TransitionWorker import makeSpec, etalonNoise
from BatSpec.Logic.Loader import loadRecord
from BatSpec.Logic.SaveIntegral import SaveIntegralEnergy
from BatSpec.Logic.Echo import TimeDomainEchoModel, generate_echos
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

A, B = harmonicPercussive(spec)

from BatSpec.Logic.VisDebug import RegisterGlobFunc; RegisterGlobFunc()

A.show()
B.show()
A.showLog()
B.showLog()

# noise = etalonNoise(spec, 20)
# nspec = subEtalonNoise(spec, noise, alpha=1.0)

# print(f"После вычитания шума: {SaveIntegralEnergy(nspec)} \nСигнал/(Сигнал + Шум): {SaveIntegralEnergy(nspec)/SaveIntegralEnergy(spec)}")

# model = TimeDomainEchoModel(
#     decay_rate_base=40,
#     f_ref=40000.0, 
#     freq_exp=2.0,
#     time_power=20.0,
# )

# cspec = removeEchoRichardsonLucy(nspec, model, iterations=20)
# gspec = enhanceCurvesGabor(cspec, ksize=16, sigma=1.0, lambd=8.0, pre_blur=2.0)
# bspec = normalizeFrecZ(gspec, noise).cloneApply(lambda arr: np.where(arr > 10, 1, 0))

