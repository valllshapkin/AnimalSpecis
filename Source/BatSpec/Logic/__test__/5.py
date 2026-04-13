from matplotlib import pyplot as plt
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox
import sys
from BatSpec.Logic.Functions import TimeFunc
from BatSpec.Logic.Disсrete import Peaks
from BatSpec.Logic.TransitionWorker import clusterCalls, createClusterBasis, extractCalls, findPeaks, separatePattern
from BatSpec.Logic.SpecWorker import (
    subEtalonNoise, normalizeFrecZ
)
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise

from pathlib import Path
from BatSpec.Logic.Functions import TimeFunc, SpecFunc
from BatSpec.Logic.ConvWindow import Window
from BatSpec.Logic.SpecWorker import subEtalonNoise
from BatSpec.Logic.TransitionWorker import makeSpec, etalonNoise, integrateFreq
from BatSpec.Logic.Loader import loadRecord
from scipy.signal import windows
import numpy as np

# ==========================================================
# Визуализация (В стиле вашей архитектуры)
# ==========================================================

class SignalPeaksGraph(QGroupBox):
    """
    UI Hierarchy:
    SignalPeaksGraph (QGroupBox)
    └── self.main_layout : QVBoxLayout
        └── self.plot_widget : pg.PlotWidget (или ваш ThemedPlotWidget)
            ├── self.plot_curve : pg.PlotDataItem (Сам сигнал)
            ├── self.peak_widths : pg.ErrorBarItem (Линии ширины пиков)
            └── self.peak_scatter : pg.ScatterPlotItem (Точки центров пиков)
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setTitle("Осциллограмма и Пики")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 4, 4, 4)

        # Plot widget setup (Можете заменить на свой ThemedPlotWidget)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', "Amplitude", units='FS')
        self.plot_widget.setLabel('bottom', "Time", units='s')
        self.main_layout.addWidget(self.plot_widget)

        # 1. Линия сигнала
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen(color=(100, 150, 255), width=1.5))
        self.plot_curve.setDownsampling(ds=True, auto=True, method='peak')
        self.plot_curve.setClipToView(True)

        # 2. Отрезки для отображения ширины пиков
        self.peak_widths = pg.ErrorBarItem(pen=pg.mkPen(color=(255, 100, 100, 150), width=3))
        self.plot_widget.addItem(self.peak_widths)

        # 3. Маркеры самих пиков
        self.peak_scatter = pg.ScatterPlotItem(
            size=10, 
            pen=pg.mkPen('w', width=1), 
            brush=pg.mkBrush(255, 50, 50),
            symbol='o'
        )
        self.plot_widget.addItem(self.peak_scatter)


    def clearPlot(self):
        self.plot_curve.setData([], [])
        self.peak_scatter.setData([], [])
        self.peak_widths.setData(x=np.array([]), y=np.array([]), left=np.array([]), right=np.array([]))
        self.plot_widget.setTitle("No data")

    def updatePlotData(self, signal: TimeFunc, peaks: Peaks):
        if signal is None:
            return self.clearPlot()

        # 1. Отрисовываем сам сигнал
        self.plot_curve.setData(x=signal.time_axis, y=signal.data)

        if peaks is not None and len(peaks.center) > 0:
            # Поскольку в `Peaks` хранятся только значения X (время),
            # нам нужно найти соответствующие значения Y (амплитуду) из сигнала.
            # searchsorted работает очень быстро для отсортированных массивов (коим является time_axis).
            indices = np.searchsorted(signal.time_axis, peaks.center)
            indices = np.clip(indices, 0, len(signal.data) - 1)
            y_peaks = signal.data[indices]

            # 2. Отрисовываем маркеры (точки) пиков
            self.peak_scatter.setData(x=peaks.center, y=y_peaks)

            # 3. Отрисовываем ширину пиков (горизонтальные линии)
            half_widths = peaks.width / 2
            self.peak_widths.setData(
                x=peaks.center, 
                y=y_peaks, 
                left=half_widths, 
                right=half_widths,
                beam=0.05 # Вертикальные засечки на концах линий ширины
            )
        else:
            self.peak_scatter.setData([], [])
            self.peak_widths.setData(x=np.array([]), y=np.array([]), left=np.array([]), right=np.array([]))

        # Обновляем границы графика
        self.plot_widget.setXRange(0, signal.duration, padding=0.01)
        
        ymin = float(signal.data.min())
        ymax = float(signal.data.max())
        yrange = ymax - ymin
        pad = max(0.08, yrange * 0.12)
        self.plot_widget.setYRange(ymin - pad, ymax + pad)


def save_calls_to_png(calls: list[SpecFunc], output_folder: str):
    """
    Сохраняет список вырезок SpecFunc в виде PNG изображений.
    """
    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    
    print(f"Сохраняем {len(calls)} вырезок в {path.absolute()} ...")
    
    for i, call in enumerate(calls):
        # matrix имеет форму [T, H] (Время, Частота). 
        # Для отрисовки транспонируем в [H, T]
        data = call.matrix.T
        
        # Границы для осей графика
        t0, t1 = call.time[0], call.time[-1]
        f0, f1 = call.freq[0], call.freq[-1]
        
        fig, ax = plt.subplots(figsize=(3, 3)) # Небольшой квадратный размер
        
        # Отрисовка. origin='lower' ставит низкие частоты вниз.
        # extent привязывает пиксели к реальным секундам и герцам.
        cax = ax.imshow(
            data, 
            aspect='auto', 
            origin='lower', 
            extent=[t0, t1, f0, f1],
            cmap='magma' # Отличная палитра для спектрограмм
        )
        
        ax.set_title(f"Call {i:03d}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Freq (Hz)")
        
        # Убираем лишние белые поля
        plt.tight_layout()
        
        # Сохраняем в файл (например: call_000.png, call_001.png...)
        plt.savefig(path / f"call_{i:03d}.png", dpi=100)
        
        # Обязательно закрываем фигуру, чтобы не забить оперативную память!
        plt.close(fig)
        
    print("Сохранение завершено!")
# ==========================================================
# Тестовый запуск приложения
# ==========================================================

# Функция сохранения, которую мы писали в прошлом ответе
def save_calls_to_png(calls: list[SpecFunc], output_folder: str, prefix: str = "call"):
    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    for i, call in enumerate(calls):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(call.matrix.T, aspect='auto', origin='lower', cmap='magma')
        plt.tight_layout()
        plt.savefig(path / f"{prefix}_{i:03d}.png", dpi=100)
        plt.close(fig)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # 1. Загрузка и препроцессинг
    record = loadRecord(Path("/MainData/Repo/golonchenroppi/BatSpecNew/Resources/r09_23/MYODAS_20230624_004924.wav"))
    spec = makeSpec(record, Window(windows.hann, 0.002, norm="energy"), overlap=0.8, bins=300)
    noise = etalonNoise(spec, 5)
    denoised = subEtalonNoise(spec, noise, alpha=1)
    func = integrateFreq(denoised)
    
    peaks = findPeaks(func, width=0.0005, distance=0.0005, prominence=0.1)
    
    # 2. Вырезание (width_multiplier=3 как вы просили)
    normalized_spec = normalizeFrecZ(denoised, noise)
    calls = extractCalls(peaks, normalized_spec, width_multiplier=3.0)
    
    if len(calls) == 0:
        print("Коллы не найдены!")
        sys.exit()

    # 3. Кластеризация (допустим, ищем 3 типа сигналов)
    n_clust = 3
    labels = clusterCalls(calls, n_clusters=n_clust)
    
    # 4. Сохраняем вырезки по папкам кластеров и считаем базис
    clusters_data = {i: [] for i in range(n_clust)}
    for call, label in zip(calls, labels):
        clusters_data[label].append(call)
        
    best_cluster_id = -1
    max_len = 0
    
    for cluster_id, cluster_calls in clusters_data.items():
        if len(cluster_calls) == 0: continue
            
        print(f"Кластер {cluster_id}: {len(cluster_calls)} коллов")
        save_calls_to_png(cluster_calls, f"./clusters/cluster_{cluster_id}")
        
        # Считаем базис для кластера
        basis = createClusterBasis(cluster_calls)
        
        # Сохраним картинку базиса
        save_calls_to_png([basis], f"./clusters/cluster_{cluster_id}", prefix="BASIS")
        
        # Выбираем самый большой кластер для теста сепарации
        if len(cluster_calls) > max_len:
            max_len = len(cluster_calls)
            best_cluster_id = cluster_id

    # 5. Тестируем сепарацию (ищем базис самого большого кластера в целой спектрограмме)
    if best_cluster_id != -1:
        print(f"Тестируем сепарацию паттерном из кластера {best_cluster_id}...")
        best_basis = createClusterBasis(clusters_data[best_cluster_id])
        
        # Разделяем исходную (нормализованную) спектрограмму
        extracted, cleaned = separatePattern(normalized_spec, best_basis, prominence=0.4)
        
        # Для визуальной проверки сохраним фрагменты
        # (целиком спектрограмма может быть слишком длинной для png)
        # Вы можете отдать `extracted` и `cleaned` в ваш MainGraph!
        
        print("Сепарация завершена. Extracted и Cleaned готовы для UI.")

    # (Тут ваш код создания UI окна)
    sys.exit(app.exec())