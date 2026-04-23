import numpy as np
from scipy.interpolate import splprep, splev
from dataclasses import dataclass
from typing import List, Tuple, Optional
from BatSpec.Work.Function import SpecFunc
from BatSpec.Work.Units import UREG, PintUnit

@dataclass
class Trace:
    """
    Один непрерывный компонент сигнала (например, основная частота или один обертон).
    """
    t: np.ndarray | List  # Время (сек)
    f: np.ndarray | List  # Частота (Гц)
    i: np.ndarray | List  # Интенсивность в точке (амплитуда)

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
    
    def _prepare_axes(
        self,
        dt: Optional[float],
        df: Optional[float],
        time_axis: Optional[np.ndarray],
        freq_axis: Optional[np.ndarray],
        padding_s: float,
        padding_hz: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Вспомогательная функция для подготовки осей и шагов."""
        # --- Проверка входных данных ---
        if (dt is None) == (time_axis is None):
            raise ValueError("Для временной оси необходимо указать либо 'dt', либо 'time_axis', но не оба сразу.")
        if (df is None) == (freq_axis is None):
            raise ValueError("Для частотной оси необходимо указать либо 'df', либо 'freq_axis', но не оба сразу.")

        t_min, t_max, f_min, f_max = self.get_bounding_box()
        t_start_padded, t_end_padded = t_min - padding_s, t_max + padding_s
        f_start_padded, f_end_padded = max(0, f_min - padding_hz), f_max + padding_hz

        # --- Подготовка временной оси ---
        if time_axis is not None:
            t_arr = np.asarray(time_axis)
            if t_arr[0] > t_start_padded or t_arr[-1] < t_end_padded:
                raise ValueError("Предоставленная ось 'time_axis' не покрывает весь временной диапазон сигнала с учетом padding.")
            dt_calc = t_arr[1] - t_arr[0] if len(t_arr) > 1 else 1.0
            if dt_calc <= 0: raise ValueError("Шаг по времени в 'time_axis' должен быть положительным.")
        else:
            t_arr = np.arange(t_start_padded, t_end_padded, dt)
            dt_calc = dt

        # --- Подготовка частотной оси ---
        if freq_axis is not None:
            f_arr = np.asarray(freq_axis)
            if f_arr[0] > f_start_padded or f_arr[-1] < f_end_padded:
                raise ValueError("Предоставленная ось 'freq_axis' не покрывает весь частотный диапазон сигнала с учетом padding.")
            df_calc = f_arr[1] - f_arr[0] if len(f_arr) > 1 else 1.0
            if df_calc <= 0: raise ValueError("Шаг по частоте в 'freq_axis' должен быть положительным.")
        else:
            f_arr = np.arange(f_start_padded, f_end_padded, df)
            df_calc = df
            
        return t_arr, f_arr, dt_calc, df_calc


    def toSpecFunc(
        self, 
        sigma_t: float, 
        sigma_f: float, 
        dt: Optional[float] = None, 
        df: Optional[float] = None,
        time_axis: Optional[np.ndarray] = None,
        freq_axis: Optional[np.ndarray] = None,
        unit: PintUnit = UREG.dimensionless,
        padding_s: float = 0.05,
        padding_hz: float = 5000.0
    ) -> SpecFunc:
        """
        Рендерит параметрические кривые в растровую спектрограмму (SpecFunc).
        Сетку можно задать либо шагами (dt, df), либо готовыми осями (time_axis, freq_axis).
        
        sigma_t, sigma_f : Размытие Гаусса по времени и частоте ("толщина" линии).
        padding_s, hz    : Отступы вокруг bounding box (используются только при генерации осей по dt/df).
        """
        # 1. Формируем сетку
        t_arr, f_arr, dt_calc, df_calc = self._prepare_axes(
            dt, df, time_axis, freq_axis, padding_s, padding_hz
        )
        t_start, f_start = t_arr[0], f_arr[0]
        matrix = np.zeros((len(t_arr), len(f_arr)), dtype=np.float32)

        # 2. Подготовка "кисти" (Gaussian Patch)
        rad_t = max(1, int(3 * sigma_t / dt_calc))
        rad_f = max(1, int(3 * sigma_f / df_calc))
        
        patch_t = np.arange(-rad_t, rad_t + 1) * dt_calc
        patch_f = np.arange(-rad_f, rad_f + 1) * df_calc
        PT, PF = np.meshgrid(patch_t, patch_f, indexing='ij')
        
        base_patch = np.exp(-0.5 * ((PT / sigma_t)**2 + (PF / sigma_f)**2))

        # 3. Рендерим каждый Trace
        for trace in self.traces:
            if len(trace.t) < 2: continue
                
            k = min(3, len(trace.t) - 1)
            tck, _ = splprep([trace.t, trace.f, trace.i], s=0, k=k)
            
            # Плотность интерполяции зависит от шага сетки
            n_interp = max(1000, int((t_arr[-1] - t_arr[0]) / dt_calc) * 2)
            u_dense = np.linspace(0, 1, n_interp)
            t_dense, f_dense, i_dense = splev(u_dense, tck)

            for t_val, f_val, i_val in zip(t_dense, f_dense, i_dense):
                if i_val <= 0: continue

                idx_t = int(round((t_val - t_start) / dt_calc))
                idx_f = int(round((f_val - f_start) / df_calc))

                mt_start, mt_end = max(0, idx_t - rad_t), min(matrix.shape[0], idx_t + rad_t + 1)
                mf_start, mf_end = max(0, idx_f - rad_f), min(matrix.shape[1], idx_f + rad_f + 1)

                pt_start, pt_end = mt_start - (idx_t - rad_t), mt_start - (idx_t - rad_t) + (mt_end - mt_start)
                pf_start, pf_end = mf_start - (idx_f - rad_f), mf_start - (idx_f - rad_f) + (mf_end - mf_start)

                if mt_start >= mt_end or mf_start >= mf_end: continue

                scaled_patch = base_patch[pt_start:pt_end, pf_start:pf_end] * i_val
                
                matrix[mt_start:mt_end, mf_start:mf_end] = np.maximum(
                    matrix[mt_start:mt_end, mf_start:mf_end],
                    scaled_patch
                )

        return SpecFunc(matrix=matrix, time=t_arr, freq=f_arr, unit=unit)

    def to_correlation_kernel(
        self, 
        sigma_t: float, 
        sigma_f: float,
        dt: Optional[float] = None, 
        df: Optional[float] = None,
        time_axis: Optional[np.ndarray] = None,
        freq_axis: Optional[np.ndarray] = None,
        unit: PintUnit = UREG.dimensionless,
        padding_s: float = 0.05,
        padding_hz: float = 5000.0,
        zero_mean: bool = True
    ) -> SpecFunc:
        """
        Генерирует 2D ядро (шаблон) для поиска. Игнорирует амплитуду (i).
        Сетку можно задать либо шагами (dt, df), либо готовыми осями (time_axis, freq_axis).
        
        sigma_t, f   : Размытие Гаусса ("толщина" линии).
        zero_mean    : Если True, из ядра вычитается среднее (сумма элементов становится 0).
        """
        # 1. Формируем сетку
        t_arr, f_arr, dt_calc, df_calc = self._prepare_axes(
            dt, df, time_axis, freq_axis, padding_s, padding_hz
        )
        t_start, f_start = t_arr[0], f_arr[0]
        matrix = np.zeros((len(t_arr), len(f_arr)), dtype=np.float32)

        # 2. Подготовка "кисти"
        rad_t = max(1, int(3 * sigma_t / dt_calc))
        rad_f = max(1, int(3 * sigma_f / df_calc))
        
        patch_t = np.arange(-rad_t, rad_t + 1) * dt_calc
        patch_f = np.arange(-rad_f, rad_f + 1) * df_calc
        PT, PF = np.meshgrid(patch_t, patch_f, indexing='ij')
        
        base_patch = np.exp(-0.5 * ((PT / sigma_t)**2 + (PF / sigma_f)**2))

        # 3. Рендерим форму (без интенсивности i)
        for trace in self.traces:
            if len(trace.t) < 2: continue
            k = min(3, len(trace.t) - 1)
            tck, _ = splprep([trace.t, trace.f], s=0, k=k)
            
            n_interp = max(1000, int((t_arr[-1] - t_arr[0]) / dt_calc) * 2)
            u_dense = np.linspace(0, 1, n_interp)
            t_dense, f_dense = splev(u_dense, tck)

            for t_val, f_val in zip(t_dense, f_dense):
                idx_t = int(round((t_val - t_start) / dt_calc))
                idx_f = int(round((f_val - f_start) / df_calc))

                mt_start, mt_end = max(0, idx_t - rad_t), min(matrix.shape[0], idx_t + rad_t + 1)
                mf_start, mf_end = max(0, idx_f - rad_f), min(matrix.shape[1], idx_f + rad_f + 1)
                
                pt_start, pt_end = mt_start - (idx_t - rad_t), mt_start - (idx_t - rad_t) + (mt_end - mt_start)
                pf_start, pf_end = mf_start - (idx_f - rad_f), mf_start - (idx_f - rad_f) + (mf_end - mf_start)

                if mt_start >= mt_end or mf_start >= mf_end: continue
                
                scaled_patch = base_patch[pt_start:pt_end, pf_start:pf_end]
                
                matrix[mt_start:mt_end, mf_start:mf_end] = np.maximum(
                    matrix[mt_start:mt_end, mf_start:mf_end],
                    scaled_patch
                )

        # 4. НОРМИРОВКА НА НОЛЬ (Zero-Mean)
        if zero_mean and matrix.size > 0:
            matrix -= np.mean(matrix)

        return SpecFunc(matrix=matrix, time=t_arr, freq=f_arr, unit=unit)


def centrateCorpMass(corp: SpecFunc):
    u, m = corp.values
    _, t = corp.time
    _, f = corp.freq 
    # Считаем взвешенное среднее по времени
    # Суммируем по частоте, чтобы получить 1D проекцию на время
    time_projection = m.sum(axis=1)
    # Защита от деления на ноль, если вся проекция нулевая
    total_mass_t = time_projection.sum()
    if total_mass_t > 0:
        center_of_mass_t = np.average(t, weights=time_projection)
    else:
        center_of_mass_t = np.mean(t) # Если масс нет, берем геометрический центр

    # Аналогично для частоты
    freq_projection = m.sum(axis=0)
    total_mass_f = freq_projection.sum()
    if total_mass_f > 0:
        center_of_mass_f = np.average(f, weights=freq_projection)
    else:
        center_of_mass_f = np.mean(f)

    # Смещаем оси так, чтобы центр масс оказался в (0, 0)
    new_t = t - center_of_mass_t
    new_f = f - center_of_mass_f
    
    return SpecFunc(m.copy(), new_t, new_f, u)