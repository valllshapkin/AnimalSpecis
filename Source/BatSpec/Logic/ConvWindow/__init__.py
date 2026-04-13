from pathlib import Path

import numpy as np

class Window:
    def __init__(self, func, time, norm=None):
        self.func = func
        self.time = time
        self.norm = norm

    def get_array(self, sr):
        size = int(self.time * sr)
        if not 16 < size < 2048: print(f"Странный размер окна {size}")
        row = self.func(size) 
        if self.norm == None:
            return row
        elif self.norm == "energy":
            return row / np.sqrt(np.sum(row ** 2))
        elif self.norm == "area":
            return row / np.sum(row)
        else: raise RuntimeError()

    
def load_file_window(file: Path) -> dict[str, Window]:
    namespace = {}
    
    with file.open("r", encoding="utf-8") as f:
        exec(f.read(), namespace)

    if not namespace.get("EXPORT_NAME"): raise RuntimeError()
    if not namespace.get("EXPORT_WINDOW"): raise RuntimeError()
    if not namespace.get("EXPORT_DURATION"): raise RuntimeError()

    return {namespace.get("EXPORT_NAME"): Window(
        namespace.get("EXPORT_WINDOW"), 
        namespace.get("EXPORT_DURATION"),
        norm="energy"
    )}