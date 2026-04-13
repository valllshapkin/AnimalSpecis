import numpy as np
from scipy.ndimage import gaussian_filter
from typing import List, Dict

from tqdm import tqdm
from BatSpec.Logic.Disсrete import Peaks
from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Logic.TransitionWorker import createClusterBasis

class BatTemplateGenerator:
    """
    Генератор аналитических ядер (шаблонов) для поиска летучих мышей.
    Основан на относительном времени и частоте.
    """
    def __init__(self, dt: float, df: float, blur_sigma: float = 1.5):
        """
        :param dt: Разрешение по времени (секунд в 1 пикселе)
        :param df: Разрешение по частоте (Гц в 1 пикселе)
        :param blur_sigma: Силу размытия (симуляция оконного сглаживания FFT)
        """
        self.dt = dt
        self.df = df
        self.blur_sigma = blur_sigma

    def _create_specfunc(self, t_arr: np.ndarray, f_arr: np.ndarray) -> 'SpecFunc':
        """Внутренний метод: отрисовывает кривую и упаковывает в SpecFunc"""
        duration = t_arr[-1]
        bandwidth = np.max(f_arr)
        
        # Считаем размеры в пикселях с небольшим запасом (padding)
        w_px = max(2, int(duration / self.dt))
        h_px = max(2, int(bandwidth / self.df) + 10)
        
        canvas = np.zeros((h_px, w_px))
        
        # Отрисовка линии
        for i in range(len(t_arr)):
            x_px = min(w_px - 1, int(t_arr[i] / self.dt))
            y_px = min(h_px - 1, int(f_arr[i] / self.df))
            canvas[y_px, x_px] = 1.0
            
        # Размываем (эмулируем гармонику на спектрограмме)
        kernel = gaussian_filter(canvas, sigma=self.blur_sigma)
        
        # ВАЖНО: Делаем сумму ядра равной 0 (zero-mean).
        # Это нужно, чтобы фильтр не реагировал на сплошной белый шум на фоне.
        kernel = kernel - np.mean(kernel)
        
        # Генерируем локальные оси для SpecFunc
        local_time = np.linspace(0, duration, w_px)
        local_freq = np.linspace(0, h_px * self.df, h_px)
        
        return SpecFunc(kernel, local_freq, local_time)

    # ================== АНАЛИТИЧЕСКИЕ ПАТТЕРНЫ ==================

    def generate_CF(self, duration: float = 0.01) -> 'SpecFunc':
        """Постоянная частота (горизонтальная линия)"""
        t = np.linspace(0, duration, 100)
        f = np.zeros_like(t) + (50 / self.df) # Слегка приподнимаем над нулем
        return self._create_specfunc(t, f)

    def generate_FM(self, duration: float = 0.005, bandwidth: float = 30000) -> 'SpecFunc':
        """Крутой частотно-модулированный спад (линейный или экспоненциальный)"""
        t = np.linspace(0, duration, 100)
        # Классический крутой спад по экспоненте
        tau = duration / 3 
        f = bandwidth * np.exp(-t / tau)
        return self._create_specfunc(t, f)

    def generate_qCF(self, duration: float = 0.01, bandwidth: float = 2000) -> 'SpecFunc':
        """Квазипостоянная частота (очень пологий спад)"""
        t = np.linspace(0, duration, 100)
        # Линейное падение на небольшую дельту частоты
        f = bandwidth * (1 - t / duration)
        return self._create_specfunc(t, f)

    def generate_FM_qCF(self, duration: float = 0.008, bandwidth: float = 20000, flat_ratio: float = 0.5) -> 'SpecFunc':
        """
        Крутой спад переходящий в полку (гладконосые летучие мыши).
        flat_ratio: какую долю времени занимает плоская часть (qCF)
        """
        t = np.linspace(0, duration, 100)
        # Асимптотическая экспонента. Подбираем tau так, чтобы к flat_ratio кривая легла в горизонт
        tau = (duration * (1 - flat_ratio)) / 5
        f = bandwidth * np.exp(-t / tau)
        return self._create_specfunc(t, f)

    def generate_qCF_FM(self, duration: float = 0.008, bandwidth: float = 15000, flat_ratio: float = 0.6) -> 'SpecFunc':
        """
        Полка, в конце резко падающая вниз.
        """
        t = np.linspace(0, duration, 100)
        t_drop_start = duration * flat_ratio
        f = np.zeros_like(t)
        
        for i, time_val in enumerate(t):
            if time_val < t_drop_start:
                f[i] = bandwidth  # Полка наверху
            else:
                # Падение вниз по экспоненте
                tau = (duration - t_drop_start) / 3
                f[i] = bandwidth * np.exp(-(time_val - t_drop_start) / tau)
                
        return self._create_specfunc(t, f)

    def generate_FM_CF_FM(self, duration: float = 0.02, bandwidth: float = 15000) -> 'SpecFunc':
        """
        Сложный крик (подковоносые). Вниз -> Полка -> Вниз.
        """
        t = np.linspace(0, duration, 100)
        f = np.zeros_like(t)
        
        t1 = duration * 0.2 # Конец первой фазы FM
        t2 = duration * 0.8 # Начало второй фазы FM
        
        for i, time_val in enumerate(t):
            if time_val < t1:
                # Взлет (или спад, зависит от конкретного вида, сделаем классический взлет к полке)
                f[i] = bandwidth * (time_val / t1)
            elif time_val < t2:
                # CF полка
                f[i] = bandwidth
            else:
                # Спад в конце
                f[i] = bandwidth * (1 - (time_val - t2) / (duration - t2))
                
        return self._create_specfunc(t, f)

    # ================== BATCH МЕТОДЫ ==================

    def generate_all_dict(self) -> Dict[str, 'SpecFunc']:
        """Возвращает словарь со всеми типами сигналов"""
        return {
            "CF": self.generate_CF(),
            "FM": self.generate_FM(),
            "qCF": self.generate_qCF(),
            "FM_qCF": self.generate_FM_qCF(),
            "qCF_FM": self.generate_qCF_FM(),
            "FM_CF_FM": self.generate_FM_CF_FM()
        }

    def generate_all_list(self) -> List['SpecFunc']:
        """Возвращает список SpecFunc (если названия тебе не нужны)"""
        return list(self.generate_all_dict().values())


import numpy as np
from skimage.feature import match_template

def applyPatternFilter(main_spec: SpecFunc, pattern_spec: SpecFunc) -> SpecFunc:
    """
    Скользит шаблоном (pattern_spec) по основной спектрограмме (main_spec).
    Возвращает тепловую карту совпадений (от 0.0 до 1.0).
    """
    
    # 1. Извлекаем матрицы
    image = main_spec.matrix
    template = pattern_spec.matrix
    
    # Проверка на случай, если шаблон оказался больше спектрограммы 
    # (например, мышь кричит 20мс, а файл длится 10мс)
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        # Возвращаем пустую матрицу, так как шаблон физически не влезает
        return SpecFunc(np.zeros_like(image), main_spec.freq, main_spec.time)

    # 2. Нормализованная кросс-корреляция
    # pad_input=True гарантирует, что размер выходной матрицы будет 1-в-1 как у main_spec.
    # mode='reflect' аккуратно обрабатывает края спектрограммы.
    corr_map = match_template(image, template, pad_input=True)
    
    # 3. Нормализация результата
    # match_template возвращает значения от -1.0 до 1.0.
    # Нас интересуют только прямые совпадения (положительная корреляция).
    # Поэтому мы отсекаем всё, что ниже 0.
    norm_map = np.clip(corr_map, 0.0, 1.0)
    
    # Возвращаем новую "спектрограмму", которая на самом деле является картой вероятностей
    return SpecFunc(norm_map, main_spec.freq, main_spec.time)

def applyPatternFilterTime(main_spec: SpecFunc, pattern_spec: SpecFunc) -> SpecFunc:
    """
    Скользит шаблоном (pattern_spec) по основной спектрограмме (main_spec) 
    СТРОГО по оси времени. Обе матрицы должны иметь одинаковый размер по оси частот.
    """
    
    # 1. Извлекаем матрицы. Формат: (Time, Freq)
    image = main_spec.matrix
    template = pattern_spec.matrix
    
    T_m, F_m = image.shape
    T_t, F_t = template.shape
    
    # Проверка на то, что шаблон не длиннее самой спектрограммы по времени
    if T_t > T_m:
        return main_spec.cloneApply(lambda _: np.zeros_like(image))

    # Защита от ошибок: спектрограмма и шаблон должны совпадать по частотной оси
    if F_m != F_t:
        raise ValueError(f"Несовпадение частотных осей: спектрограмма {F_m}, шаблон {F_t}")

    # 2. Нормализованная кросс-корреляция (NCC) БЕЗ автоматического паддинга.
    # Так как F_m == F_t, шаблон физически не может двигаться по вертикали.
    # Он будет скользить только по времени (T).
    # Размер результата будет: (T_m - T_t + 1, 1)
    corr_map = match_template(image, template, pad_input=False)
    
    # 3. Отсекаем отрицательные корреляции (нас интересуют только совпадения)
    corr_map = np.clip(corr_map, 0.0, 1.0)
    
    # 4. Восстанавливаем размер по оси времени (центрируем результат)
    # Чтобы длина стала снова T_m, добавляем нули в начало и в конец.
    pad_front = (T_t - 1) // 2
    pad_back = T_m - corr_map.shape[0] - pad_front
    
    # Паддинг только по оси Time. Ось Freq (которая сейчас равна 1) не трогаем.
    corr_padded = np.pad(corr_map, ((pad_front, pad_back), (0, 0)), mode='constant', constant_values=0)
    
    # 5. Возвращаем форму к (Time, Freq). 
    # Так как вероятность совпадения шаблона в момент времени 't' относится 
    # ко всему сигналу, мы дублируем (растягиваем) этот score на все частоты.
    # Это создаст четкие "вертикальные полосы" вероятностей на итоговой матрице.
    final_score_matrix = np.tile(corr_padded, (1, F_m))
    
    # Возвращаем новую SpecFunc с картой вероятностей
    return SpecFunc(final_score_matrix, main_spec.freq, main_spec.time)

def enhanceCallsWithEtalons(
    main_spec: SpecFunc, 
    peaks: Peaks,
    calls: list[SpecFunc], 
    labels: np.ndarray, 
    alpha: float = 0.5
) -> SpecFunc:
    """
    Усиливает каждый крик, смешивая его с идеальным эталоном своего кластера.
    
    :param main_spec: Исходная полная спектрограмма.
    :param peaks: Объект Peaks с координатами центров.
    :param calls: Список всех вырезанных коллов.
    :param labels: Массив меток кластеров (от hdbscan).
    :param alpha: Коэффициент смешивания. 0.0 - без изменений, 1.0 - полная замена на эталон.
    """
    FW = main_spec.FW if hasattr(main_spec, 'FW') else np
    dt = main_spec.dt
    t0 = main_spec.time[0]
    
    # 1. Создаем копию исходной матрицы, которую будем изменять
    enhanced_matrix = main_spec.matrix.copy()
    
    # 2. Предварительно вычисляем базисы для каждого кластера
    # Это намного быстрее, чем считать его для каждого колла в цикле
    unique_clusters = set(labels)
    valid_clusters = [c for c in unique_clusters if c != -1]
    
    print("Создаем эталонные базисы для кластеров...")
    etalons = {}
    for cluster_id in valid_clusters:
        indices = np.where(labels == cluster_id)[0]
        cluster_calls = [calls[i] for i in indices]
        etalons[cluster_id] = createClusterBasis(cluster_calls)

    print(f"Смешиваем {len(calls)} коллов с их эталонами...")
    # 3. Проходим по КАЖДОМУ исходному коллу
    for i in tqdm(range(len(calls))):
        label = labels[i]
        
        # Если колл - это шум (-1), мы его не трогаем, оставляем как есть
        if label == -1:
            continue
            
        call = calls[i]
        call_mat = call.matrix
        T_call, F_call = call_mat.shape
        
        # Получаем соответствующий эталон
        etalon = etalons[label]
        etalon_mat = etalon.matrix
        T_etalon, _ = etalon_mat.shape
        
        # 4. Выравниваем эталон под размер текущего колла
        # Вырезаем из центра эталона кусок, равный по длине текущему коллу
        start = (T_etalon - T_call) // 2
        etalon_part = etalon_mat[start : start + T_call, :]
        
        # 5. Смешиваем! Это и есть твоя магия.
        blended_mat = (1 - alpha) * call_mat + alpha * etalon_part
        
        # 6. Вставляем усиленный колл обратно на его место в общую матрицу
        center_time = peaks.center[i]
        c_idx = int(round((center_time - t0) / dt))
        half_T_call = T_call // 2
        
        start_idx = c_idx - half_T_call
        end_idx = start_idx + T_call
        
        # Защита от краев (хотя extractCalls уже должен был это сделать)
        if start_idx < 0 or end_idx > main_spec.matrix.shape[0]:
            continue
        
        # ВАЖНО: для обработки перекрытий используем np.maximum.
        # Если два усиленных колла пересекаются, в месте пересечения останется более яркий пиксель.
        # Это предотвращает артефакты от простого сложения или перезаписи.
        enhanced_matrix[start_idx:end_idx, :] = FW.maximum(
            enhanced_matrix[start_idx:end_idx, :],
            blended_mat
        )

    return SpecFunc(enhanced_matrix, main_spec.freq, main_spec.time)