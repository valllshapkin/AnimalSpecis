from BatSpec.Logic.Functions import TimeFunc
from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Logic.Disсrete import Peaks
from BatSpec.Logic.ConvWindow import Window
import numpy as np

def etalonNoise(spec: SpecFunc, percentile: float):
    rms = np.sqrt(np.sum(spec.matrix ** 2, axis=1))    # форма (T,)
    noise_threshold = np.percentile(rms, percentile)
    noise_idx = np.where(rms <= noise_threshold)[0]          # индексы "тихих" кадров
    return noise_idx

def integrateFreq(spec: SpecFunc) -> TimeFunc:
    """
    Integrate a spectrogram over the frequency axis to obtain a time-varying function.

    For each time bin, the integral is computed as the sum over frequency bins
    multiplied by the frequency step (df). The resulting time series is returned
    as a TimeFunc object with sampling rate derived from the time axis.

    Parameters
    ----------
    spec : SpecFunc
        Spectrogram object containing matrix (time x frequency), freq array, and time array.

    Returns
    -------
    TimeFunc
        Time-domain function where data[i] = ∫ spec.matrix[i, :] df.
    """
    # Sum over frequency axis (axis=1) using the framework's sum function
    integrated = spec.FW.sum(spec.matrix, axis=1) * spec.df  # shape (T,)
    # Compute sampling rate from the uniform time step
    sr = int(round(1.0 / spec.dt))
    return TimeFunc(integrated, sr)

def makeSpec(signal: TimeFunc, window: Window, overlap=0.5, bins=600) -> SpecFunc:
    # Window[norm="energy"]

    import tensorflow as tf
    
    win = window.get_array(signal.sr)
    fft_length = (bins - 1) * 2
    win_length = len(win)
    hop = int(win_length * (1 - overlap))

    if fft_length < win_length:
        print(
            f"[WARNING] fft_length={fft_length} меньше win_length={win_length}. "
            f"Фрейм будет обрезан — это потеря данных."
        )
    if fft_length > win_length:
        print(
            f"[WARNING] fft_length={fft_length} больше win_length={win_length}. "
            f"Это увеличивает разрешение по частоте но не приносит реальные данные"
        )
    print(
        f"fft_length={fft_length} win_length={win_length}. "
        f"win_length={win_length} hop={hop}"
    )
        

    def win_func(length, dtype=tf.float32):
        assert length == win_length
        return tf.convert_to_tensor(win, dtype=dtype)

    stft = tf.signal.stft(
        signal.data.astype(np.float32),
        frame_length=win_length,
        frame_step=hop,
        fft_length=fft_length,
        window_fn=win_func
    )
    spectrogram = np.abs(stft.numpy()) * np.sqrt(2 / signal.sr)
    freq_axis = np.fft.rfftfreq(fft_length, d=1.0/signal.sr)

    num_frames = spectrogram.shape[0]
    time_axis = (np.arange(num_frames) * hop) / signal.sr

    # print(spectrogram.shape)

    return SpecFunc(matrix=spectrogram, freq=freq_axis, time=time_axis)

def findPeaks(signal: TimeFunc, width: float, distance: float | None = None, prominence=0.5, ) -> Peaks:
    from scipy.signal import find_peaks

    peaks, props = find_peaks(
        signal.data,
        prominence=prominence,
        width=round(width * signal.sr),        
        distance=round(distance * signal.sr) if distance else None
    )

    return Peaks(center=signal.time_axis[peaks], width=props['widths'] / signal.sr, time_axis=signal.time_axis)
    
def extractCalls(peaks: Peaks, spec: SpecFunc, width_multiplier: float = 1.5) -> list[SpecFunc]:
    """
    Вырезает фрагменты спектрограммы на основе найденных пиков.
    
    :param peaks: Объект Peaks с центрами и ширинами в секундах.
    :param spec: Исходная спектрограмма.
    :param width_multiplier: Множитель ширины, чтобы захватить сигнал целиком (включая тихие хвосты).
    :return: Список объектов SpecFunc (вырезанные коллы).
    """
    extracted_calls = []
    
    # dt - шаг по времени (в секундах)
    dt = spec.dt
    t0 = spec.time[0]
    total_T = spec.matrix.shape[0]
    
    for i in range(peaks.count):
        # Получаем время центра и ширину из Peaks
        c_time = peaks.center[i]
        w_time = peaks.width[i] * width_multiplier
        
        # Переводим секунды в индексы оси T
        # Использование round и int надежно переводит время в индекс
        c_idx = int(round((c_time - t0) / dt))
        half_w_idx = int(round((w_time / 2) / dt))
        
        # Вычисляем границы с защитой от выхода за пределы массива
        start_idx = max(0, c_idx - half_w_idx)
        end_idx = min(total_T, c_idx + half_w_idx + 1)
        
        # Делаем срезы (slices). Это NumPy views, они не копируют данные в памяти!
        sub_matrix = spec.matrix[start_idx:end_idx, :]
        sub_time = spec.time[start_idx:end_idx]
        
        # Создаем новый объект SpecFunc для вырезки
        call_spec = SpecFunc(matrix=sub_matrix, freq=spec.freq, time=sub_time)
        extracted_calls.append(call_spec)
        
    return extracted_calls


def specDistance(spec1: SpecFunc, spec2: SpecFunc, bg_value: float = 0.0) -> float:
    """Вычисляет MAE расстояние между центрированными спектрограммами."""
    FW = spec1.FW
    T1, H = spec1.matrix.shape
    T2, _ = spec2.matrix.shape
    
    max_T = max(T1, T2)
    canvas1 = FW.full((max_T, H), bg_value, dtype=spec1.matrix.dtype)
    canvas2 = FW.full((max_T, H), bg_value, dtype=spec2.matrix.dtype)
    
    start1 = (max_T - T1) // 2
    canvas1[start1 : start1 + T1, :] = spec1.matrix
    
    start2 = (max_T - T2) // 2
    canvas2[start2 : start2 + T2, :] = spec2.matrix
    
    return float(FW.mean(FW.abs(canvas1 - canvas2)))


def clusterCalls(calls: list[SpecFunc], n_clusters: int = 3) -> np.ndarray:
    """
    Кластеризует список вырезок. 
    Возвращает массив меток (labels), где каждый элемент - номер кластера (0, 1, 2...).
    """
    from sklearn.cluster import AgglomerativeClustering

    N = len(calls)
    dist_matrix = np.zeros((N, N))
    
    print("Вычисляем матрицу расстояний...")
    for i in range(N):
        for j in range(i + 1, N):
            d = specDistance(calls[i], calls[j], bg_value=0.0)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
    print("Кластеризация...")
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters, 
        metric='precomputed', 
        linkage='average'
    )
    labels = clusterer.fit_predict(dist_matrix)
    return labels

def createClusterBasis(calls: list[SpecFunc], bg_value: float = 0.0) -> SpecFunc:
    """
    Создает усредненный паттерн (базис) из списка коллов.
    """
    if not calls:
        raise ValueError("Список вырезок пуст!")
        
    FW = calls[0].FW
    # Находим максимальную длину среди всех коллов
    max_T = max(c.matrix.shape[0] for c in calls)
    H = calls[0].matrix.shape[1]
    
    sum_canvas = FW.zeros((max_T, H), dtype=np.float32)
    
    # Складываем все спектрограммы по центру
    for c in calls:
        T, _ = c.matrix.shape
        start = (max_T - T) // 2
        sum_canvas[start : start + T, :] += c.matrix
        
    # Усредняем
    avg_matrix = sum_canvas / len(calls)
    
    # Создаем локальную ось времени для паттерна (начинается с 0)
    dt = calls[0].dt
    rel_time = FW.arange(max_T, dtype=np.float32) * dt
    
    return SpecFunc(matrix=avg_matrix, freq=calls[0].freq, time=rel_time)


def separatePattern(main_spec: SpecFunc, pattern: SpecFunc, prominence: float = 0.5) -> tuple[SpecFunc, SpecFunc]:
    """
    Ищет паттерн в главной спектрограмме и разделяет ее на две.
    :return: (Extracted_SpecFunc, Cleaned_SpecFunc)
    """
    from scipy.signal import correlate, find_peaks

    FW = main_spec.FW
    T_main, H = main_spec.matrix.shape
    T_pat, H_pat = pattern.matrix.shape
    
    if H != H_pat:
        raise RuntimeError("Оси частот не совпадают!")

    print("Вычисление кросс-корреляции (это может занять пару секунд)...")
    # Считаем 1D корреляцию для каждого частотного бина и суммируем их.
    # mode='same' вернет массив размера T_main (центры совпадений)
    corr_1d = np.zeros(T_main, dtype=np.float32)
    for h in range(H):
        corr_1d += correlate(main_spec.matrix[:, h], pattern.matrix[:, h], mode='same')
        
    # Нормализуем корреляцию к максимуму (чтобы prominence работал предсказуемо)
    corr_max = np.max(corr_1d)
    if corr_max > 0:
        corr_1d = corr_1d / corr_max

    # Ищем пики (моменты времени, где паттерн совпал)
    peaks_indices, _ = find_peaks(corr_1d, prominence=prominence)
    print(f"Паттерн найден в {len(peaks_indices)} местах.")

    # Готовим холсты для результатов
    cleaned_matrix = np.copy(main_spec.matrix)
    extracted_matrix = np.zeros_like(main_spec.matrix)
    
    half_T = T_pat // 2

    # Вычитание паттерна
    for p_idx in peaks_indices:
        # Вычисляем границы в главной матрице
        start_idx = p_idx - half_T
        end_idx = start_idx + T_pat
        
        # Защита от выхода за края
        if start_idx < 0 or end_idx > T_main:
            continue
            
        # Для более чистого вычитания можно масштабировать паттерн под локальную энергию
        # (упрощенный вариант: берем паттерн "как есть", либо умножаем на коэффициент)
        local_chunk = cleaned_matrix[start_idx:end_idx, :]
        
        # Вычитаем паттерн из главной спектрограммы. np.clip не дает энергии уйти в минус.
        subtracted = np.clip(local_chunk - pattern.matrix, a_min=0, a_max=None)
        
        # То, что мы удалили, записываем в extracted (отделенный сигнал)
        extracted_matrix[start_idx:end_idx, :] += (local_chunk - subtracted)
        
        # Обновляем очищенную матрицу
        cleaned_matrix[start_idx:end_idx, :] = subtracted

    cleaned_spec = SpecFunc(matrix=cleaned_matrix, freq=main_spec.freq, time=main_spec.time)
    extracted_spec = SpecFunc(matrix=extracted_matrix, freq=main_spec.freq, time=main_spec.time)
    
    return extracted_spec, cleaned_spec


def extractClusterPattern(main_spec: SpecFunc, pattern: SpecFunc, prominence: float = 0.2) -> SpecFunc:
    """
    Ищет паттерн в спектрограмме. Возвращает спектрограмму, содержащую 
    только совпавшие фрагменты (остальное - тишина).
    """
    from scipy.signal import correlate, find_peaks

    FW = main_spec.FW
    T_main, H = main_spec.matrix.shape
    T_pat, H_pat = pattern.matrix.shape
    
    if H != H_pat:
        raise RuntimeError("Оси частот не совпадают!")

    print(f"  > Считаем 1D кросс-корреляцию...")
    corr_1d = np.zeros(T_main, dtype=np.float32)
    for h in range(H):
        corr_1d += correlate(main_spec.matrix[:, h], pattern.matrix[:, h], mode='same')
        
    corr_max = np.max(corr_1d)
    if corr_max > 0:
        corr_1d = corr_1d / corr_max

    # Снизил prominence до 0.2, чтобы находилось больше совпадений
    peaks_indices, _ = find_peaks(corr_1d, prominence=prominence)
    print(f"  > Паттерн найден в {len(peaks_indices)} местах.")

    # Создаем пустую спектрограмму (фон)
    extracted_matrix = np.zeros_like(main_spec.matrix)
    half_T = T_pat // 2

    # Копируем найденные куски из оригинала в пустую матрицу
    for p_idx in peaks_indices:
        start_idx = p_idx - half_T
        end_idx = start_idx + T_pat
        
        if start_idx < 0 or end_idx > T_main:
            continue
            
        # Просто переносим оригинальный сигнал в эти временные рамки
        extracted_matrix[start_idx:end_idx, :] = main_spec.matrix[start_idx:end_idx, :]

    return SpecFunc(matrix=extracted_matrix, freq=main_spec.freq, time=main_spec.time)


from sklearn.cluster import DBSCAN
from skimage.metrics import structural_similarity as ssim
import numpy as np


# ВАРИАНТ А: Нормализованный MAE
def specDistanceNormalized(spec1: SpecFunc, spec2: SpecFunc) -> float:
    FW = spec1.FW
    T1, H = spec1.matrix.shape; T2, _ = spec2.matrix.shape
    max_T = max(T1, T2)
    canvas1 = FW.full((max_T, H), 0.0, dtype=np.float32)
    canvas2 = FW.full((max_T, H), 0.0, dtype=np.float32)
    start1 = (max_T - T1) // 2; canvas1[start1 : start1 + T1, :] = spec1.matrix
    start2 = (max_T - T2) // 2; canvas2[start2 : start2 + T2, :] = spec2.matrix
    
    # --- НОВАЯ ЧАСТЬ ---
    # L1 нормализация (сумма модулей равна 1)
    norm1 = np.sum(np.abs(canvas1))
    if norm1 > 1e-6: canvas1 /= norm1
        
    norm2 = np.sum(np.abs(canvas2))
    if norm2 > 1e-6: canvas2 /= norm2
    # --- КОНЕЦ НОВОЙ ЧАСТИ ---
    
    return float(FW.mean(FW.abs(canvas1 - canvas2)))

# ВАРИАНТ Б (РЕКОМЕНДУЕМЫЙ): Расстояние на основе SSIM
def specDistanceSSIM(spec1: SpecFunc, spec2: SpecFunc) -> float:
    # ... (код для padding остается тот же) ...
    FW = spec1.FW
    T1, H = spec1.matrix.shape; T2, _ = spec2.matrix.shape
    max_T = max(T1, T2)
    canvas1 = FW.full((max_T, H), 0.0, dtype=np.float32)
    canvas2 = FW.full((max_T, H), 0.0, dtype=np.float32)
    start1 = (max_T - T1) // 2; canvas1[start1 : start1 + T1, :] = spec1.matrix
    start2 = (max_T - T2) // 2; canvas2[start2 : start2 + T2, :] = spec2.matrix

    # SSIM возвращает схожесть от -1 до 1. Превращаем в расстояние от 0 до 2.
    # data_range - размах значений в данных (от 0 до макс. яркости)
    max_val = max(np.max(canvas1), np.max(canvas2))
    if max_val < 1e-6: return 0.0 # если обе картинки черные
    
    similarity = ssim(canvas1, canvas2, data_range=max_val)
    distance = 1.0 - similarity # Расстояние от 0 (идентичны) до 2 (макс. разные)
    return distance

def clusterCallsDBSCAN(calls: list[SpecFunc], metric_func, eps: float, min_samples: int = 5) -> np.ndarray:
    """
    Кластеризует с помощью DBSCAN.
    :param eps: Максимальное расстояние между двумя сэмплами, чтобы один считался соседом другого.
                ЭТО ГЛАВНЫЙ ПАРАМЕТР, КОТОРЫЙ НУЖНО ПОДБИРАТЬ!
    :param min_samples: Количество сэмплов в окрестности, чтобы точка считалась ядром.
    :return: Массив меток. Метка -1 означает "выброс" (шум).
    """
    N = len(calls)
    dist_matrix = np.zeros((N, N))
    
    print("Вычисляем матрицу расстояний...")
    for i in range(N):
        for j in range(i + 1, N):
            d = metric_func(calls[i], calls[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
    print("Кластеризация DBSCAN...")
    # Главный параметр тут `eps`. Его нужно подбирать.
    # Для SSIM дистанции (0..2) хорошее начало - eps=0.4..0.7
    # Для Normalized MAE (0..1) хорошее начало - eps=0.1..0.2
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(dist_matrix)
    return labels

def specDistanceSignal(spec1fillFromSpec: SpecFunc, spec2: SpecFunc,
                       snr_threshold: float = 2.0) -> float:
    """
    MAE только по значимым пикселям, с нормализацией по L2-энергии.
    Инвариантен к амплитуде, чувствителен к форме/частоте/длительности.
    """
    FW = spec1.FW
    T1, H = spec1.matrix.shape
    T2, _ = spec2.matrix.shape
    max_T = max(T1, T2)

    canvas1 = FW.full((max_T, H), 0.0, dtype=np.float32)
    canvas2 = FW.full((max_T, H), 0.0, dtype=np.float32)
    start1 = (max_T - T1) // 2; canvas1[start1:start1+T1, :] = spec1.matrix
    start2 = (max_T - T2) // 2; canvas2[start2:start2+T2, :] = spec2.matrix

    # 1. Обнуляем шум — оставляем только значимый сигнал
    c1 = np.where(canvas1 > snr_threshold, canvas1, 0.0)
    c2 = np.where(canvas2 > snr_threshold, canvas2, 0.0)

    # 2. Нормализуем по L2 — убираем влияние амплитуды/яркости
    n1 = np.linalg.norm(c1)
    n2 = np.linalg.norm(c2)
    if n1 > 1e-6: c1 = c1 / n1
    if n2 > 1e-6: c2 = c2 / n2

    # 3. MAE по объединённой маске значимых пикселей
    mask = (c1 > 0) | (c2 > 0)
    if mask.sum() == 0:
        return 0.0

    return float(np.mean(np.abs(c1[mask] - c2[mask])))


import numpy as np
from skimage.feature import match_template
from tqdm import tqdm # Крайне советую для прогресс-бара, так как матрица 800x800 это ~320 000 сравнений
import numpy as np
from skimage.feature import match_template
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def _compute_pair(i, j, calls):
    """Функция-воркер для одного сравнения"""
    mat_i = calls[i].matrix
    mat_j = calls[j].matrix
    
    if mat_i.shape[0] <= mat_j.shape[0]:
        template, image = mat_i, mat_j
    else:
        template, image = mat_j, mat_i
        
    if template.size == 0 or image.size == 0:
        return i, j, 0.0
        
    res = match_template(image, template, pad_input=False)
    return i, j, np.max(res)

def buildCorrelationMatrix(peaks: Peaks, spec: SpecFunc, width_multiplier: float = 3.0, max_workers: int = 4):
    """
    Параллельная версия построения матрицы сходства.
    max_workers - количество потоков (поставь равным числу ядер твоего процессора)
    """
    calls = extractCalls(peaks, spec, width_multiplier=width_multiplier)
    n = len(calls)
    
    corr_matrix = np.eye(n, dtype=np.float32)
    print(f"Строим матрицу {n}x{n} в {max_workers} потоков...")
    
    # Собираем все пары (i, j), которые нужно посчитать (верхний треугольник)
    tasks = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Запускаем параллельно
        futures = [executor.submit(_compute_pair, i, j, calls) for i, j in tasks]
        
        # tqdm будет обновляться по мере завершения задач
        for future in tqdm(futures, total=len(tasks)):
            i, j, max_corr = future.result()
            corr_matrix[i, j] = max_corr
            corr_matrix[j, i] = max_corr
            
    return corr_matrix, calls


import hdbscan

def clusterCorrelationMatrix(corr_matrix: np.ndarray, min_cluster_size: int = 5):
    distance_matrix = np.clip(1.0 - corr_matrix, 0.0, 1.0).astype(np.float64)
    
    clusterer = hdbscan.HDBSCAN(
        metric='precomputed', 
        min_cluster_size=min_cluster_size,
        min_samples=2, # Делает алгоритм более чувствительным к тонким различиям
        cluster_selection_method='leaf' # ВАЖНО: заставляет выделять мелкие уникальные подгруппы
    )
    
    return clusterer.fit_predict(distance_matrix)




from BatSpec.Logic.Calls import applyPatternFilterTime
def applyClusterFilters(
    main_spec: SpecFunc, 
    calls: list[SpecFunc], 
    labels: np.ndarray, 
    apply_as_mask: bool = False
) -> SpecFunc:
    """
    Создает усредненные базисы для каждого кластера и прогоняет их по всей спектрограмме.
    
    :param main_spec: Исходная полная спектрограмма
    :param calls: Список всех вырезанных коллов
    :param labels: Массив меток кластеров (от hdbscan)
    :param apply_as_mask: Если True - умножает исходную спектрограмму на вероятности (усиливая сигнал визуально).
                          Если False - возвращает чистую карту вероятностей (от 0 до 1).
    """
    unique_clusters = set(labels)
    
    # Создаем пустую матрицу-аккумулятор того же размера, что и исходная
    # FW - используем логику твоего фреймворка
    FW = main_spec.FW if hasattr(main_spec, 'FW') else np
    max_scores = FW.zeros_like(main_spec.matrix, dtype=np.float32)
    
    # Отбрасываем шум (-1)
    valid_clusters = [c for c in unique_clusters if c != -1]
    
    print(f"Применяем {len(valid_clusters)} кластерных фильтров к основной записи...")
    
    for cluster_id in tqdm(valid_clusters):
        # 1. Собираем все вырезки этого кластера
        indices = np.where(labels == cluster_id)[0]
        cluster_calls = [calls[i] for i in indices]
        
        # 2. Создаем идеальный базис (паттерн)
        basis = createClusterBasis(cluster_calls)
        
        # 3. Скользим этим базисом по всей записи
        prob_map_spec = applyPatternFilterTime(main_spec, basis)
        
        # 4. Накапливаем максимальный отклик
        # Если этот кластер "узнал" крик лучше, чем предыдущие, берем его значение
        max_scores = FW.maximum(max_scores, prob_map_spec.matrix)
        
    if apply_as_mask:
        # УМНОЖАЕМ исходную спектрограмму на карту вероятностей.
        # Там где был шум (score ~ 0.1), интенсивность упадет в 10 раз.
        # Там где крик мыши (score ~ 0.95), интенсивность останется почти нетронутой.
        enhanced_matrix = main_spec.matrix * max_scores
        return SpecFunc(enhanced_matrix, main_spec.freq, main_spec.time)
    else:
        # Возвращаем саму карту совпадений (вертикальные полосы)
        return SpecFunc(max_scores, main_spec.freq, main_spec.time)


from skimage.measure import label, regionprops
import numpy as np
from typing import List, Tuple

def extractLabels(
    bin_spec: SpecFunc,
    padding_time: int = 8,
    padding_freq: int = 12,
    min_area: int = 5,
    min_width: int = 3,   # минимальная ширина по времени после padding
    min_height: int = 3   # минимальная высота по частоте
) -> List[SpecFunc]:
    """
    Вырезает отдельные лейблы в список SpecFunc с защитой от пустых массивов.
    Оси: [0] - Время (T), [1] - Частоты (H).
    """
    if np.all(bin_spec.matrix == 0):
        return []

    mask = (bin_spec.matrix > 0).astype(np.uint8)
    labeled = label(mask)
    
    if labeled.max() == 0:
        return []

    props = regionprops(labeled)

    result: List[SpecFunc] = []

    for prop in props:
        if prop.area < min_area:
            continue

        # row = Время (T), col = Частота (H)
        minr, minc, maxr, maxc = prop.bbox  

        # Добавляем padding с защитой границ
        # Время: ось 0
        t_start = max(0, minr - padding_time)
        t_end   = min(mask.shape[0], maxr + padding_time + 1)
        
        # Частота: ось 1
        f_start = max(0, minc - padding_freq)
        f_end   = min(mask.shape[1], maxc + padding_freq + 1)

        # Защита от нулевого/недостаточного размера
        if (t_end - t_start < min_width) or (f_end - f_start < min_height):
            
            # Корректируем время
            if t_end - t_start < min_width:
                center_t = (minr + maxr) // 2
                t_start = max(0, center_t - min_width // 2)
                t_end   = min(mask.shape[0], t_start + min_width)
                
            # Корректируем частоту
            if f_end - f_start < min_height:
                center_f = (minc + maxc) // 2
                f_start = max(0, center_f - min_height // 2)
                f_end   = min(mask.shape[1], f_start + min_height)

            # Если после попытки исправления размер всё ещё мал — пропускаем
            if (t_end - t_start < min_width) or (f_end - f_start < min_height):
                continue

        # Вырезаем матрицу (размер: Время х Частота)
        sub_matrix = np.zeros((t_end - t_start, f_end - f_start), 
                              dtype=bin_spec.matrix.dtype)
        
        # Копируем только пиксели текущего лейбла (индексация: [Время, Частота])
        label_mask = (labeled[t_start:t_end, f_start:f_end] == prop.label)
        sub_matrix[label_mask] = bin_spec.matrix[t_start:t_end, f_start:f_end][label_mask]

        cropped_time = bin_spec.time[t_start:t_end]
        cropped_freq = bin_spec.freq[f_start:f_end]

        # Финальная защита от пустых осей
        if len(cropped_time) == 0 or len(cropped_freq) == 0:
            continue

        # Инициализируем SpecFunc: matrix(T, H), freq(H), time(T)
        result.append(SpecFunc(sub_matrix, cropped_freq, cropped_time))

    return result