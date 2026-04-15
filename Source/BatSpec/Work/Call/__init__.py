import numpy as np
from scipy.interpolate import splprep, splev
from dataclasses import dataclass
from typing import List, Tuple
from BatSpec.Work.Function import SpecFunc
from BatSpec.Work.Units import UREG, PintUnit

@dataclass
class Trace:
    """
    Один непрерывный компонент сигнала (например, основная частота или один обертон).
    """
    t: np.ndarray  # Время (сек)
    f: np.ndarray  # Частота (Гц)
    i: np.ndarray  # Интенсивность в точке (амплитуда)

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.f = np.asarray(self.f, dtype=float)
        self.i = np.asarray(self.i, dtype=float)
        assert len(self.t) == len(self.f) == len(self.i), "Массивы t, f, i должны быть одной длины"


class Call:
    """
    Абстрактный акустический сигнал, состоящий из одной или нескольких кривых (Trace).
    """
    def __init__(self, traces: List[Trace] | None = None):
        self.traces = traces if traces else []

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Возвращает (t_min, t_max, f_min, f_max) по всем контрольным точкам."""
        if not self.traces:
            return 0.0, 0.1, 0.0, 1000.0
        
        t_min = min(np.min(tr.t) for tr in self.traces)
        t_max = max(np.max(tr.t) for tr in self.traces)
        f_min = min(np.min(tr.f) for tr in self.traces)
        f_max = max(np.max(tr.f) for tr in self.traces)
        return t_min, t_max, f_min, f_max

    def toSpecFunc(
        self, 
        dt: float, 
        df: float, 
        sigma_t: float, 
        sigma_f: float, 
        unit: PintUnit = UREG.dimensionless,
        padding_s: float = 0.05,
        padding_hz: float = 5000.0
    ) -> SpecFunc:
        """
        Рендерит параметрические кривые в растровую спектрограмму (SpecFunc).
        
        dt, df       : Шаг сетки (сек, Гц)
        sigma_t, f   : Размытие Гаусса по времени и частоте ("толщина" линии)
        padding      : Отступы вокруг bounding box
        """
        # 1. Формируем сетку
        t_min, t_max, f_min, f_max = self.get_bounding_box()
        t_start, t_end = t_min - padding_s, t_max + padding_s
        f_start, f_end = max(0, f_min - padding_hz), f_max + padding_hz

        t_arr = np.arange(t_start, t_end, dt)
        f_arr = np.arange(f_start, f_end, df)
        
        matrix = np.zeros((len(t_arr), len(f_arr)), dtype=np.float32)

        # Подготовка "кисти" (Gaussian Patch)
        # Ограничиваем размер патча до 3 сигм (дальше гауссиан практически равен нулю)
        rad_t = max(1, int(3 * sigma_t / dt))
        rad_f = max(1, int(3 * sigma_f / df))
        
        patch_t = np.arange(-rad_t, rad_t + 1) * dt
        patch_f = np.arange(-rad_f, rad_f + 1) * df
        PT, PF = np.meshgrid(patch_t, patch_f, indexing='ij')
        
        # Нормированный 2D Гауссиан с максимумом в центре = 1.0
        base_patch = np.exp(-0.5 * ((PT / sigma_t)**2 + (PF / sigma_f)**2))

        # 2. Рендерим каждый Trace
        for trace in self.traces:
            n_points = len(trace.t)
            if n_points < 2:
                continue
                
            # Степень сплайна (максимум 3 - кубический, но не больше кол-ва точек минус 1)
            k = min(3, n_points - 1)
            
            # splprep строит параметрическую кривую
            # u - параметр вдоль кривой (от 0 до 1)
            tck, _ = splprep([trace.t, trace.f, trace.i], s=0, k=k)
            
            # Генерируем плотный набор точек (чем тоньше dt/df, тем плотнее надо брать)
            # Берем с запасом, чтобы линия не выглядела пунктирной
            u_dense = np.linspace(0, 1, max(1000, int((t_max-t_min)/dt)*2))
            t_dense, f_dense, i_dense = splev(u_dense, tck)

            # Наносим точки на матрицу
            for t_val, f_val, i_val in zip(t_dense, f_dense, i_dense):
                if i_val <= 0:
                    continue

                # Находим индексы центра в большой матрице
                idx_t = int(round((t_val - t_start) / dt))
                idx_f = int(round((f_val - f_start) / df))

                # Границы для врезки патча в матрицу
                mt_start = max(0, idx_t - rad_t)
                mt_end   = min(matrix.shape[0], idx_t + rad_t + 1)
                mf_start = max(0, idx_f - rad_f)
                mf_end   = min(matrix.shape[1], idx_f + rad_f + 1)

                # Границы самого патча (если он обрезается краем матрицы)
                pt_start = mt_start - (idx_t - rad_t)
                pt_end   = pt_start + (mt_end - mt_start)
                pf_start = mf_start - (idx_f - rad_f)
                pf_end   = pf_start + (mf_end - mf_start)

                if mt_start >= mt_end or mf_start >= mf_end:
                    continue

                # Масштабируем патч на локальную интенсивность
                scaled_patch = base_patch[pt_start:pt_end, pf_start:pf_end] * i_val
                
                # Используем maximum, чтобы пересекающиеся линии не складывали яркость в 2 раза,
                # а сливались в единый гладкий гребень
                matrix[mt_start:mt_end, mf_start:mf_end] = np.maximum(
                    matrix[mt_start:mt_end, mf_start:mf_end],
                    scaled_patch
                )

        return SpecFunc(
            matrix = matrix,
            time   = t_arr,
            freq   = f_arr,
            unit   = unit
        )