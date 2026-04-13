from BatSpec.Logic.Functions import SpecFunc

from BatSpec.Logic.SpecWorker import (
    subEtalonNoise, makeLog, enhanceCurvesGabor, normalizeFrecZ, makeBinarization,
    filterLabelsCurves, fillFromSpec
)
from BatSpec.Logic.SpecWorker.Morphology import morphDilate
from BatSpec.Logic.TransitionWorker import etalonNoise, extractLabels


from pathlib import Path
import os
import glob

from scipy.signal import windows

os.chdir(Path(__file__).parent)

for wav_file in glob.glob("./Specs/*.npz"):
    file = Path(wav_file)
    print(file)

    SPEC = SpecFunc.load(str(file))
    noise = etalonNoise(SPEC, 5)

    DENOISED = subEtalonNoise(SPEC, noise, alpha=1)

    GABOR = enhanceCurvesGabor(
        DENOISED,
        ksize=17, sigma=1, 
        lambd=5, pre_blur=0
    )

    BINARY = makeBinarization(normalizeFrecZ(GABOR, noise), trashhold=10)

    CURVES = filterLabelsCurves(BINARY)

    DILATE = morphDilate(CURVES, radius=2)

    for n, i in enumerate(extractLabels(DILATE, padding_time=0, padding_freq=0, min_area=3)):
        fillFromSpec(i, DENOISED, apply_mask=True)
        i.saveDebugLogPngWithStats(f"./Calls/{file.stem}.{n:03d}.png")
        i.savez_compressed(f"./Calls/{file.stem}.{n:03d}.npz")


