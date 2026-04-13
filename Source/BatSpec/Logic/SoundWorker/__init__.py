from BatSpec.Logic.Functions import TimeFunc
from BatSpec.Logic.ConvWindow import Window
import numpy as np

def localRMS(f: TimeFunc, window: Window) -> TimeFunc:
    # Window[norm="area"]

    return TimeFunc(np.sqrt(np.convolve(
        f.data ** 2, window.get_array(f.sr), mode="same"
    )), f.sr)

def correctDC(f: TimeFunc) -> TimeFunc:
    return TimeFunc(f.data - np.median(f.data), f.sr)