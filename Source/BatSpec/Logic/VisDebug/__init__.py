import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow
from BatSpec.Logic.Functions import SpecFunc
from BatSpec.QtUp.Image import AdaptiveImageItem 

def show_spec(spec: SpecFunc, title="Spectrogram", levels=None):
    app = QApplication.instance() or QApplication(sys.argv)

    win = QMainWindow()
    win.setWindowTitle(title)
    win.resize(1200, 500)

    plot_widget = pg.PlotWidget()
    win.setCentralWidget(plot_widget)
    plot_item = plot_widget.getPlotItem()

    plot_item.setLabel("left",   "Frequency", units="Hz")
    plot_item.setLabel("bottom", "Time",       units="s")

    img = AdaptiveImageItem()

    if levels is None:
        levels = (float(spec.matrix.min()), float(spec.matrix.max()))
    img._levels = levels

    plot_item.addItem(img)
    img.attachTo(plot_item)

    matrix_T = spec.matrix.T  # -> (n_freq, n_time)

    x_range = (float(spec.time[0]), float(spec.time[-1]))
    y_range = (float(spec.freq[0]), float(spec.freq[-1]))
    img.setFullData(matrix_T, x_range, y_range)

    plot_item.setLimits(xMin=x_range[0], xMax=x_range[1],
                        yMin=y_range[0], yMax=y_range[1])
    plot_item.setRange(xRange=x_range, yRange=y_range, padding=0)

    color_bar = pg.ColorBarItem(
        values=levels,
        colorMap=pg.colormap.get("viridis"),
        label="Power",
    )
    color_bar.setImageItem(img, insert_in=plot_item)

    win.show()
    app.exec()
    
def show_spec_labels(spec: SpecFunc, labels: SpecFunc, title="Spectrogram Labels"):
    """
    spec   : SpecFunc — спектрограмма (float, для отображения)
    labels : SpecFunc — метки кластеров (int matrix, shape n_time x n_freq)
    """
    app = QApplication.instance() or QApplication(sys.argv)

    win = QMainWindow()
    win.setWindowTitle(title)
    win.resize(1200, 500)

    plot_widget = pg.PlotWidget()
    win.setCentralWidget(plot_widget)
    plot_item = plot_widget.getPlotItem()

    plot_item.setLabel("left",   "Frequency", units="Hz")
    plot_item.setLabel("bottom", "Time",       units="s")

    # --- спектрограмма ---
    img = AdaptiveImageItem()
    levels = (float(spec.matrix.min()), float(spec.matrix.max()))
    img._levels = levels

    plot_item.addItem(img)
    img.attachTo(plot_item)

    matrix_T = spec.matrix.T
    x_range = (float(spec.time[0]),  float(spec.time[-1]))
    y_range = (float(spec.freq[0]),  float(spec.freq[-1]))
    matrix_T = _burn_labels_into_matrix(matrix_T, labels)  # <- вот здесь
    img.setFullData(matrix_T, x_range, y_range)


    plot_item.setLimits(xMin=x_range[0], xMax=x_range[1],
                        yMin=y_range[0], yMax=y_range[1])
    plot_item.setRange(xRange=x_range, yRange=y_range, padding=0)

    color_bar = pg.ColorBarItem(
        values=levels,
        colorMap=pg.colormap.get("viridis"),
        label="Power",
    )
    color_bar.setImageItem(img, insert_in=plot_item)


    win.show()
    app.exec()


def _burn_labels_into_matrix(matrix_T: np.ndarray, labels: SpecFunc) -> np.ndarray:
    """
    matrix_T : (n_freq, n_time) float — спектрограмма
    Возвращает копию с вожжёнными рамками и полупрозрачными масками кластеров.
    """
    result = matrix_T.copy().astype(np.float32)
    lmat = labels.matrix  # (n_time, n_freq)
    lmat_T = lmat.T       # (n_freq, n_time) — совпадает с matrix_T

    vmin, vmax = result.min(), result.max()
    box_val  = vmax                          # цвет рамки — максимум (яркий)
    mask_val = vmin + (vmax - vmin) * 0.3   # маска — чуть приглушённее

    unique = np.unique(lmat_T)
    unique = unique[unique != -1]

    for label_id in unique:
        mask = lmat_T == label_id  # (n_freq, n_time)

        # маска — повышаем яркость внутри кластера
        result[mask] = result[mask] * 0.5 + mask_val * 0.5

        # bbox
        f_idx, t_idx = np.where(mask)
        f0, f1 = f_idx.min(), f_idx.max()
        t0, t1 = t_idx.min(), t_idx.max()

        # рамка — вжигаем border в 1px
        result[f0,    t0:t1+1] = box_val  # низ
        result[f1,    t0:t1+1] = box_val  # верх
        result[f0:f1+1, t0  ] = box_val  # лево
        result[f0:f1+1, t1  ] = box_val  # право

    return result


def RegisterGlobFunc():
    from BatSpec.Logic import Functions
    Functions.GlobalShowSpecFunc = show_spec

