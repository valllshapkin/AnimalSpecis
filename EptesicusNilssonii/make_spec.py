from BatSpec.Logic.ConvWindow import Window
from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Logic.Loader import loadRecord
from BatSpec.Logic.TransitionWorker import makeSpec

from pathlib import Path
import os
import glob

from scipy.signal import windows

os.chdir(Path(__file__).parent)

for wav_file in glob.glob("./Audio/*.WAV"):
    file = Path(wav_file)
    record = loadRecord(file)
    spec: SpecFunc = makeSpec(record, Window(windows.hann, 0.002, norm="energy"), overlap=0.8, bins=300)
    spec.savez_compressed(f"./Spec/{file.stem}.npz")



