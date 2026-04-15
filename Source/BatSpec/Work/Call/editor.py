import sys
import json
import numpy as np
from scipy.interpolate import splprep, splev
from dataclasses import asdict

# === ИЗМЕНЕНИЕ 1: Импортируем готовые классы ===
from BatSpec.Work.Call import Call, Trace
from BatSpec.Work.Function import SpecFunc
from BatSpec.Work.Units import UREG

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QListWidget, QSlider, QLabel, QFileDialog, QGroupBox,
    QDoubleSpinBox  # Добавили виджет для чисел
)
from PySide6.QtCore import Qt, QRectF
import pyqtgraph as pg

# ==========================================
# 1. ФУНКЦИИ СОХРАНЕНИЯ/ЗАГРУЗКИ (остаются без изменений)
# ==========================================

def save_call_to_json(call: Call, filepath: str):
    # asdict должен работать, если ваш Trace - это dataclass
    data = [asdict(tr) for tr in call.traces]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_call_from_json(filepath: str) -> Call:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Предполагаем, что у Trace такой же конструктор
    traces = [Trace(t=tr['t'], f=tr['f'], i=tr['i']) for tr in data]
    return Call(traces)

# ==========================================
# 2. ГРАФИЧЕСКИЙ ЭЛЕМЕНТ (остается без изменений)
# ==========================================

class EditableTraceItem(pg.GraphItem):
    """
    Кастомный элемент PyQtGraph. 
    Отображает точки (узлы) и гладкий сплайн между ними.
    Позволяет перетаскивать точки мышью.
    """
    def __init__(self, trace: Trace, on_select_cb=None):
        super().__init__()
        self.trace = trace
        self.on_select_cb = on_select_cb
        
        self.dragPointIndex = None
        self.selectedPointIndex = None
        
        self.spline_curve = pg.PlotCurveItem(pen=pg.mkPen(color='y', width=2))
        self.update_graph()

    def update_graph(self):
        n = len(self.trace.t)
        if n >= 2:
            try:
                k = min(3, n - 1)
                tck, _ = splprep([self.trace.t, self.trace.f], s=0, k=k)
                u_new = np.linspace(0, 1, 100)
                t_smooth, f_smooth = splev(u_new, tck)
                self.spline_curve.setData(x=t_smooth, y=f_smooth)
            except Exception as e:
                print(f"Ошибка построения сплайна: {e}")
                self.spline_curve.setData(x=[], y=[])
        else:
            self.spline_curve.setData(x=[], y=[])

        if n > 0:
            pos = np.column_stack((self.trace.t, self.trace.f))
            sizes = [max(5, i * 20) for i in self.trace.i]
            brushes = []
            for idx in range(n):
                if idx == self.selectedPointIndex:
                    brushes.append(pg.mkBrush('r'))
                else:
                    brushes.append(pg.mkBrush('g'))
            self.setData(pos=pos, symbolBrush=brushes, symbolSize=sizes, symbolPen='w')
        else:
            self.setData(pos=np.empty((0,2)))

    def set_intensity_for_selected(self, intensity: float):
        if self.selectedPointIndex is not None:
            self.trace.i[self.selectedPointIndex] = intensity
            self.update_graph()

    def mouseDragEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        if ev.isStart():
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) > 0:
                self.dragPointIndex = pts[0].data()[0]
                self.selectedPointIndex = self.dragPointIndex
                if self.on_select_cb:
                    self.on_select_cb(self.trace.i[self.selectedPointIndex])
                self.update_graph()
                ev.accept()
            else:
                ev.ignore()
        elif ev.isFinish():
            self.dragPointIndex = None
            return
        else:
            if self.dragPointIndex is not None:
                new_pos = ev.pos()
                self.trace.t[self.dragPointIndex] = new_pos.x()
                self.trace.f[self.dragPointIndex] = new_pos.y()
                self.update_graph()

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            pts = self.scatter.pointsAt(ev.pos())
            if len(pts) > 0:
                self.selectedPointIndex = pts[0].data()[0]
                if self.on_select_cb:
                    self.on_select_cb(self.trace.i[self.selectedPointIndex])
                self.update_graph()
                ev.accept()

# ==========================================
# 3. ГЛАВНОЕ ОКНО (UI)
# ==========================================

class CallEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bat Call Annotator")
        self.resize(1000, 700)

        self.call = Call()
        self.active_trace_item = None
        self.trace_items = []
        self.spectrogram_item = None # Для хранения фона

        self._init_ui()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # === ЛЕВАЯ ПАНЕЛЬ (ГРАФИК) ===
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setLabel('left', 'Frequency', units='Hz')
        self.plot.scene().sigMouseClicked.connect(self.on_plot_clicked)
        layout.addWidget(self.plot, stretch=3)

        # === ПРАВАЯ ПАНЕЛЬ (УПРАВЛЕНИЕ) ===
        control_panel = QWidget()
        vbox = QVBoxLayout(control_panel)
        layout.addWidget(control_panel, stretch=1)

        # Список кривых
        vbox.addWidget(QLabel("<b>Кривые (Traces):</b>"))
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_trace_selected)
        vbox.addWidget(self.list_widget)
        btn_add_trace = QPushButton("Добавить Trace")
        btn_add_trace.clicked.connect(self.add_new_trace)
        vbox.addWidget(btn_add_trace)

        # Панель интенсивности
        group_int = QGroupBox("Параметры точки")
        group_vbox = QVBoxLayout(group_int)
        self.lbl_int = QLabel("Интенсивность: 0.5")
        group_vbox.addWidget(self.lbl_int)
        self.slider_int = QSlider(Qt.Orientation.Horizontal)
        self.slider_int.setRange(1, 100)
        self.slider_int.setValue(50)
        self.slider_int.valueChanged.connect(self.on_slider_changed)
        group_vbox.addWidget(self.slider_int)
        vbox.addWidget(group_int)

        # === ИЗМЕНЕНИЕ 2: Новая панель для растеризации ===
        group_render = QGroupBox("Параметры растеризации")
        render_vbox = QVBoxLayout(group_render)
        
        self.spin_dt = self._create_spinbox(render_vbox, "Шаг dt (сек):", 0.0001, 0.0005, 4)
        self.spin_df = self._create_spinbox(render_vbox, "Шаг df (Гц):", 1.0, 200.0, 1)
        self.spin_sigma_t = self._create_spinbox(render_vbox, "Размытие t (сек):", 0.0001, 0.002, 4)
        self.spin_sigma_f = self._create_spinbox(render_vbox, "Размытие f (Гц):", 1.0, 1500.0, 1)

        btn_render = QPushButton("Сгенерировать и показать")
        btn_render.clicked.connect(self.render_call_to_spectrogram)
        render_vbox.addWidget(btn_render)
        vbox.addWidget(group_render)
        # ===================================================

        vbox.addStretch()
        btn_save = QPushButton("Сохранить Call")
        btn_save.clicked.connect(self.save_file)
        vbox.addWidget(btn_save)
        btn_load = QPushButton("Загрузить Call")
        btn_load.clicked.connect(self.load_file)
        vbox.addWidget(btn_load)

    def _create_spinbox(self, layout, label, min_val, val, decimals):
        """Хелпер для создания полей ввода чисел"""
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(min_val, 1e9)
        spin.setValue(val)
        spin.setSingleStep(val / 10.0)
        hbox.addWidget(spin)
        layout.addLayout(hbox)
        return spin

    # --- ЛОГИКА ---
    def add_new_trace(self, trace=None):
        if not isinstance(trace, Trace):
            trace = Trace(t=[], f=[], i=[])
        self.call.traces.append(trace)
        item = EditableTraceItem(trace, on_select_cb=self.update_slider_from_point)
        self.plot.addItem(item.spline_curve)
        self.plot.addItem(item)
        self.trace_items.append(item)
        idx = len(self.call.traces)
        self.list_widget.addItem(f"Trace {idx}")
        self.list_widget.setCurrentRow(idx - 1)

    # === ИЗМЕНЕНИЕ 3: Логика кнопки рендера ===
    def render_call_to_spectrogram(self):
        """Вызывает ваш растеризатор и отображает результат."""
        if not self.call.traces:
            print("Нечего рендерить, добавьте хотя бы один Trace.")
            return

        dt = self.spin_dt.value()
        df = self.spin_df.value()
        sigma_t = self.spin_sigma_t.value()
        sigma_f = self.spin_sigma_f.value()

        # Вызываем ваш метод из BatSpec.Work.Call
        spec: SpecFunc = self.call.toSpecFunc(
            dt=dt, df=df, sigma_t=sigma_t, sigma_f=sigma_f, unit=UREG.dimensionless
        )

        # Получаем данные для отрисовки
        _, matrix = spec.values
        _, t_arr = spec.time
        _, f_arr = spec.freq

        # Удаляем старый фон, если он есть
        if self.spectrogram_item:
            self.plot.removeItem(self.spectrogram_item)

        # Создаем и настраиваем новый фон
        self.spectrogram_item = pg.ImageItem(matrix.T) # .T - транспонируем!
        
        t_start, f_start = t_arr[0], f_arr[0]
        width, height = t_arr[-1] - t_start, f_arr[-1] - f_start
        self.spectrogram_item.setRect(QRectF(t_start, f_start, width, height))

        # Применяем цветовую карту
        cmap = pg.colormap.get('magma')
        self.spectrogram_item.setColorMap(cmap)

        self.spectrogram_item.setZValue(-10) # На задний план
        self.plot.addItem(self.spectrogram_item)
    # ====================================================

    def on_trace_selected(self, row: int):
        if row >= 0 and row < len(self.trace_items):
            self.active_trace_item = self.trace_items[row]

    def on_plot_clicked(self, event):
        if event.double() and event.button() == Qt.MouseButton.LeftButton:
            if self.active_trace_item is None: return
            pos = self.plot.plotItem.vb.mapSceneToView(event.scenePos())
            t_new, f_new = pos.x(), pos.y()
            self.active_trace_item.trace.t = np.append(self.active_trace_item.trace.t, t_new)
            self.active_trace_item.trace.f = np.append(self.active_trace_item.trace.f, f_new)
            self.active_trace_item.trace.i = np.append(self.active_trace_item.trace.i, 0.5)
            self.active_trace_item.update_graph()
            
    def update_slider_from_point(self, intensity: float):
        self.slider_int.blockSignals(True)
        self.slider_int.setValue(int(intensity * 100))
        self.lbl_int.setText(f"Интенсивность: {intensity:.2f}")
        self.slider_int.blockSignals(False)

    def on_slider_changed(self, value: int):
        if self.active_trace_item:
            intensity = value / 100.0
            self.lbl_int.setText(f"Интенсивность: {intensity:.2f}")
            self.active_trace_item.set_intensity_for_selected(intensity)

    def save_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить", "", "JSON Files (*.json)")
        if path:
            save_call_to_json(self.call, path)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить", "", "JSON Files (*.json)")
        if path:
            self.call = Call()
            for item in self.trace_items:
                self.plot.removeItem(item.spline_curve)
                self.plot.removeItem(item)
            self.trace_items.clear()
            self.list_widget.clear()
            loaded_call = load_call_from_json(path)
            for trace in loaded_call.traces:
                self.add_new_trace(trace)

if __name__ == '__main__':
    # Убедитесь, что ваш BatSpec находится в PYTHONPATH
    # (например, запускайте из корня проекта)
    app = QApplication(sys.argv)
    window = CallEditorWindow()
    window.show()
    sys.exit(app.exec())