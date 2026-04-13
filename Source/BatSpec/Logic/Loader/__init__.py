from BatSpec.Logic.Functions import TimeFunc
from BatSpec.ArrType import Array, Float64

from pathlib import Path
import soundfile as sf
from typing import Any

type T = Any

import numpy as np

def loadRecord(path: Path) -> TimeFunc[Float64]:
    if not path.exists(): raise FileNotFoundError(f"Файл не найден: {path}")
    data: Array[Float64, T]; samplerate: int

    data, samplerate = sf.read(str(path), dtype='float64', always_2d=False)

    if data.ndim == 2 and data.shape[1] > 1:
        data = np.mean(data, axis=1)
    elif data.ndim == 2 and data.shape[1] == 1:
        data = data.flatten()

    # data = data - np.median(data)
    
    return TimeFunc[Float64](data, samplerate).withFrameWork(np)