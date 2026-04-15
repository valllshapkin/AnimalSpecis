from pathlib import Path
from typing import Any, Dict, Callable
from enum import Enum

import numpy as np
from pathlib import Path
ScriptDir = Path(__file__).parent

class WindowNorm(Enum):
    NONE   = "none"
    ENERGY = "energy"
    AREA   = "area"

class Window:
    def __init__(self, func: Callable[[int], np.ndarray], time: float, norm: WindowNorm = WindowNorm.NONE):
        self.func = func
        self.time = time
        self.norm = norm

    def get_array(self, sr: int) -> np.ndarray:
        size = int(self.time * sr)

        if not 16 < size < 2048:
            print(f"Странный размер окна {size}")

        row = self.func(size)

        match self.norm:
            case WindowNorm.NONE:   return row
            case WindowNorm.ENERGY: return row / np.sqrt(np.sum(row ** 2))
            case WindowNorm.AREA:   return row / np.sum(row)


def load_file_window(file: Path) -> Dict[str, Window]:
    namespace: Dict[str, Any] = {}

    with file.open("r", encoding="utf-8") as f:
        exec(f.read(), namespace)

    name     = namespace.get("EXPORT_NAME")
    window   = namespace.get("EXPORT_WINDOW")
    duration = namespace.get("EXPORT_DURATION")

    if not name:     raise RuntimeError(f"EXPORT_NAME не задан в {file}")
    if not window:   raise RuntimeError(f"EXPORT_WINDOW не задан в {file}")
    if not duration: raise RuntimeError(f"EXPORT_DURATION не задан в {file}")

    return {name: Window(func=window, time=duration, norm=WindowNorm.ENERGY)}

TEST_HANN_WINODW = load_file_window(ScriptDir / "__assets__" / "Hann.STFT.2.py")["Hann STFT 2ms"]

class WindowNormMismatchError(Exception):
    """Исключение, сигнализирующее о несовпадении ожидаемой и фактической нормировки окна."""

    def __init__(self, expected: WindowNorm, actual: WindowNorm, message: str | None = None) -> None:
        self.expected = expected
        self.actual = actual
        if message is None:
            message = f"Несовпадение нормировки окна: ожидалась '{expected.value}', получена '{actual.value}'."
        super().__init__(message)
