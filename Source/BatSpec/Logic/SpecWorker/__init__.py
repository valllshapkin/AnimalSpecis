import math
from typing import List, Tuple


from BatSpec.Logic.Functions import SpecFunc, TimeFunc
import numpy as np
from skimage.measure import label, regionprops
import hdbscan
from sklearn.preprocessing import StandardScaler



def normalizeFrecZ(spec: SpecFunc, noise_idx) -> SpecFunc:

    if len(noise_idx) < 2:
        print("Too few noise frames — skipping global normalisation.")
        return spec.cloneApply(lambda arr: arr.copy())

    # 2. Выбираем шумовые кадры
    noise_frames = spec.matrix[noise_idx, :]                 # форма (n_noise, F)

    # 3. Статистика по времени (ось 0) для каждой частоты
    mean = np.mean(noise_frames, axis=0)                     # форма (F,)
    std  = np.std(noise_frames,  axis=0)                     # форма (F,)
    std[std < 1e-8] = 1.0

    # (опционально сглаживание по частотам – работает корректно)
    # if freq_smooth_window > 1:
    #     kernel = np.ones(freq_smooth_window) / freq_smooth_window
    #     mean = np.convolve(mean, kernel, mode="same")
    #     std  = np.convolve(std,  kernel, mode="same")
    #     std[std < 1e-6] = 1.0

    # 4. Нормализация: (spectrum - mean) / std для каждой частоты
    return spec.cloneApply(lambda arr: (arr - mean[np.newaxis, :]) / std[np.newaxis, :])

def subEtalonNoise(spec: SpecFunc, noise_idx, alpha: float = 1) -> SpecFunc:
    """
    Вычитание шума по энергии (Spectral Subtraction).
    matrix**2 = signal**2 + noise**2  =>  signal = sqrt(matrix**2 - noise**2)
    
    alpha (oversubtraction factor): коэффициент силы подавления шума.
          1.0 - стандартное вычитание. 
          1.5 или 2.0 - давит шум сильнее (полезно, если остается много "мусорных" точек), 
          но может немного "съесть" края полезного сигнала.
    """
    if len(noise_idx) < 2:
        print("Too few noise frames — skipping noise subtraction.")
        return spec.cloneApply(lambda arr: arr.copy())

    # 1. Извлекаем кадры шума (амплитуды)
    noise_frames = spec.matrix[noise_idx, :]                 # форма (n_noise, F)

    # 2. Вычисляем среднюю ЭНЕРГИЮ шума для каждой частоты
    # Возводим в квадрат и берем среднее по оси времени
    mean_noise_energy = np.mean(noise_frames ** 2, axis=0)   # форма (F,)

    def apply_subtraction(arr):
        # 3. Переводим весь исходный сигнал в энергию
        full_energy = arr ** 2
        
        # 4. Вычитаем энергию шума
        clean_energy = full_energy - alpha * mean_noise_energy[np.newaxis, :]
        
        # 5. Ограничиваем снизу. Энергия не может быть отрицательной!
        # Ставим 1e-12 вместо 0, чтобы избежать проблем с делением или логарифмами в будущем
        clean_energy = np.clip(clean_energy, a_min=1e-22, a_max=None)
        
        # 6. Возвращаемся к амплитуде
        clean_amplitude = np.sqrt(clean_energy)
        
        return clean_amplitude

    return spec.cloneApply(apply_subtraction)


from BatSpec.Logic.Echo import CausalEchoModel, generate_echos, remove_echo_wiener2, richardson_lucy_1d
import numpy as np
from scipy.signal import find_peaks, fftconvolve

def removeEchoWiener(spec: SpecFunc, model: CausalEchoModel, eps_factor: float = 1e-2) -> SpecFunc:
    """
    Применяет фильтр Винера ко всей спектрограмме (по каждой частоте отдельно).
    """
    T_len, F_len = spec.matrix.shape
    sample_rate = 1.0 / spec.dt
    
    # Создаем пустую матрицу для результата
    new_matrix = np.zeros_like(spec.matrix)
    
    for i in range(F_len):
        freq_val = spec.freq[i]
        signal_1d = spec.matrix[:, i]
        
        # Применяем 1D деконволюцию к текущей частоте
        cleaned_1d = remove_echo_wiener2(
            signal=signal_1d,
            frequency=freq_val,
            sample_rate=sample_rate,
            model=model,
            eps_factor=eps_factor,
        )
        new_matrix[:, i] = cleaned_1d
        
    return SpecFunc(new_matrix, spec.freq, spec.time)


def removeEchoRichardsonLucy(spec: SpecFunc, model: CausalEchoModel, iterations: int = 15) -> SpecFunc:
    """
    Применяет деконволюцию Ричардсона-Люси ко всей спектрограмме.
    Идеально работает после применения функции sub_etalon_noise, 
    так как алгоритм требует строго положительных значений.
    
    iterations: Количество проходов алгоритма. 
                Меньше (5-10) - мягкая очистка. 
                Больше (20-30) - сильное сжатие эха, но может расти высокочастотный шум.
    """
    T_len, F_len = spec.matrix.shape
    sample_rate = 1.0 / spec.dt
    
    # Создаем пустую матрицу для результата
    new_matrix = np.zeros_like(spec.matrix)
    
    for i in range(F_len):
        print(f"bin {i}")
        freq_val = spec.freq[i]
        signal_1d = spec.matrix[:, i]
        
        # Применяем 1D деконволюцию Ричардсона-Люси к текущей частоте
        cleaned_1d = richardson_lucy_1d(
            signal=signal_1d,
            frequency=freq_val,
            sample_rate=sample_rate,
            model=model,
            iterations=iterations
        )
        new_matrix[:, i] = cleaned_1d
        
    return SpecFunc(new_matrix, spec.freq, spec.time)

# Если генерация ядра эха лежит в другом модуле, импортируй её:
# from BatSpec.Logic.Echo import generate_echos 
from scipy.signal import find_peaks

def createReferenceEcho(spec: SpecFunc, model: CausalEchoModel, num_segments: int = 100) -> SpecFunc:
    T_len, F_len = spec.matrix.shape
    sample_rate = 1.0 / spec.dt
    
    power_over_time = np.sqrt(np.sum(spec.matrix**2, axis=1))
    
    min_dist_samples = max(1, int(0.003 * sample_rate))
    prominence_thresh = np.max(power_over_time) * 0.15 
    
    peaks, _ = find_peaks(power_over_time, distance=min_dist_samples, prominence=prominence_thresh)
    
    if len(peaks) == 0:
        print("Warning: No broadband peaks found. Using center of the file as reference.")
        peaks = [T_len // 2]
        
    ref_matrix = np.zeros_like(spec.matrix)
    seg_size = max(1, F_len // num_segments)
    
    for i in range(F_len):
        freq_val = spec.freq[i]
        seg_idx = i // seg_size
        
        if seg_idx % 2 == 0:
            ref_matrix[:, i] = spec.matrix[:, i]
        else:
            impulse_signal = np.zeros(T_len)
            impulse_signal[peaks] = np.max(spec.matrix[peaks, :])
            
            kernel = generate_echos(model, freq_val, sample_rate)

            # Ищем максимум горба, чтобы предотвратить изгиб (дугу)
            peak_idx = int(np.argmax(kernel))
            
            convolved = fftconvolve(impulse_signal, kernel, mode='full')
            
            # Компенсируем сдвиг! Вырезаем сигнал так, чтобы горб совпал с исходной позицией пика
            shifted = convolved[peak_idx : peak_idx + T_len]
            
            # Если из-за сдвига не хватило длины в конце (редкий случай), добиваем нулями
            if len(shifted) < T_len:
                shifted = np.pad(shifted, (0, T_len - len(shifted)))
                
            ref_matrix[:, i] = shifted
            
    return SpecFunc(ref_matrix, spec.freq, spec.time)

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def _build_gabor_filters(ksize: int = 25, sigma: float = 3.0, lambd: float = 10.0, gamma: float = 0.5):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        
        # 1. Делаем сумму ядра строго равной 0 (чтобы фильтр не реагировал на ровный фон)
        kern = kern - kern.mean()
        
        # 2. Правильная нормализация (делим на сумму только положительных частей)
        # Это гарантирует, что яркость сигнала не улетит в космос
        pos_sum = kern[kern > 0].sum()
        if pos_sum > 1e-6:
            kern /= pos_sum
            
        filters.append(kern)
    return filters

def enhanceCurvesGabor(spec: SpecFunc, ksize: int = 16, sigma: float = 3.0, lambd: float = 8.0, gamma=0.5, pre_blur: float = 1.0) -> SpecFunc:
    """
    Применяет фильтры Габора для выделения кривых.
    
    pre_blur: Легкое размытие по Гауссу ПЕРЕД Габором. 
              Критически важно после деконволюции, чтобы "склеить" 
              отдельные точки обратно в сплошные линии и избежать артефакта "звезд".
    """
    filters = _build_gabor_filters(ksize=ksize, sigma=sigma, lambd=lambd, gamma=gamma)
    
    def apply_gabor(arr):
        img = arr.astype(np.float32)
        
        # Сглаживаем "точечность" от деконволюции (помогает от звезд)
        if pre_blur > 0:
            img = gaussian_filter(img, sigma=pre_blur)
            
        accum = np.zeros_like(img, dtype=np.float32)
        
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_32F, kern)
            
            # Отсекаем отрицательные значения (Габор может давать минус вокруг линий)
            fimg = np.maximum(fimg, 0)
            
            # Сохраняем максимальный отклик среди всех углов
            np.maximum(accum, fimg, accum)
            
        return accum

    return spec.cloneApply(apply_gabor)


def harmonicPercussive(spec: SpecFunc) -> Tuple[SpecFunc, SpecFunc]:
    import librosa
    D_transposed = spec.matrix.T
    harmonic, percussive = librosa.decompose.hpss(D_transposed)
    return SpecFunc(harmonic.T, spec.freq, spec.time), SpecFunc(percussive.T, spec.freq, spec.time)

def makeLog(spec: SpecFunc) -> SpecFunc:
    return spec.cloneApply(lambda arr: 20 * np.log10(np.clip(arr, 1e-9, None)))

def makeBinarization(spec: SpecFunc, trashhold=32) -> SpecFunc:
    return spec.cloneApply(lambda arr: np.where(arr > trashhold, 1, 0))


def makeLabels(bin_spec: SpecFunc) -> SpecFunc:
    """
    Превращает бинарную маску (0 и 1) в матрицу лейблов.
    Каждому связному компоненту (пятну) присваивается уникальный integer ID.
    Используется 8-связность (соседи по диагонали тоже считаются).
    """
    def apply_labeling(arr):
        # Гарантируем, что матрица бинарная
        binary_matrix = (arr > 0).astype(np.uint8)
        # return_num=False, так как нам нужна только сама матрица лейблов
        labeled_matrix = label(binary_matrix, connectivity=2)
        return labeled_matrix

    return bin_spec.cloneApply(apply_labeling)

import numpy as np
from skimage.measure import label, regionprops
import hdbscan
from sklearn.preprocessing import StandardScaler
import math

# from BatSpec.Logic.Functions import SpecFunc
import numpy as np
from skimage.measure import label, regionprops
import hdbscan
from sklearn.preprocessing import StandardScaler
import math

# from BatSpec.Logic.Functions import SpecFunc

def filterLabelsHDBSCAN(
    bin_spec: SpecFunc, 
    spec: SpecFunc, 
    min_cluster_size: int = 10, 
    retention_percentile: int = 50
) -> SpecFunc:
    
    labeled_matrix = makeLabels(bin_spec).matrix 
    intensity_matrix = spec.matrix
    
    regions = regionprops(labeled_matrix, intensity_image=intensity_matrix)
    
    features = []
    valid_labels = []
    
    for r in regions:
        if r.area < 5: 
            continue
            
        height = r.bbox[2] - r.bbox[0]
        width = r.bbox[3] - r.bbox[1]
        ratio = height / max(1, width)
        
        eccentricity = r.eccentricity
        solidity = r.solidity
        t_center = r.centroid[1] 
        mean_energy = r.intensity_mean
        
        # Индексы: 0:Area, 1:Ratio, 2:Energy, 3:Ecc, 4:Solidity, 5:t_center
        features.append([r.area, ratio, mean_energy, eccentricity, solidity, t_center])
        valid_labels.append(r.label)
        
    if len(features) == 0:
        return bin_spec.cloneApply(lambda arr: np.zeros_like(arr))
        
    features = np.array(features)
    
    if len(features) < min_cluster_size * 2:
        good_labels = set()
        for i, f in enumerate(features):
            if f[1] > 0.5 and f[4] < 0.8: 
                good_labels.add(valid_labels[i])
        clean_mask = np.isin(labeled_matrix, list(good_labels)).astype(np.uint8)
        return SpecFunc(clean_mask, bin_spec.freq, bin_spec.time)

    # 1. Нормализация и Кластеризация
    X_to_cluster = features[:, :]
    X_scaled = StandardScaler().fit_transform(X_to_cluster)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
    cluster_ids = clusterer.fit_predict(X_scaled)
    
    unique_clusters = list(set(cluster_ids) - {-1})
    if not unique_clusters:
        return bin_spec.cloneApply(lambda arr: np.zeros_like(arr))
        
    # 2. Оцениваем каждый кластер
    cluster_scores = []
    for c_id in unique_clusters:
        idx = np.where(cluster_ids == c_id)[0]
        
        # Средняя ПЛОЩАДЬ объектов в кластере (Признак 0)
        avg_area = np.mean(features[idx, 0])
        
        # --- ТОПОЛОГИЯ ---
        avg_ecc = np.mean(features[idx, 3]) 
        avg_solidity = np.mean(features[idx, 4]) 
        topo_score = avg_ecc * avg_solidity
        
        # --- РИТМИЧНОСТЬ ---
        times = np.sort(features[idx, 5])
        if len(times) > 2:
            dt = np.diff(times)
            mean_dt = np.mean(dt)
            std_dt = np.std(dt)
            # Чтобы не было бесконечности, берем корень от кол-ва элементов
            # Это дает бонус длинным сериям, но не позволяет им перевесить саму топологию
            rhythm_score = math.sqrt(len(times)) * (mean_dt / (std_dt + 1.0))
        else:
            rhythm_score = 0.5 
            
        # --- ФИНАЛЬНАЯ ОЦЕНКА ---
        # Теперь формула: Топология * Ритм * Площадь
        # Для безопасности (если площадь исчисляется тысячами пикселей) берем от нее корень или логарифм.
        # Я использую np.sqrt(avg_area), чтобы она плавно повышала балл, а не умножала его сразу на 5000.
        score = topo_score * rhythm_score * np.sqrt(avg_area)
        
        cluster_scores.append({'id': c_id, 'score': score})
        
    # 3. Сортируем кластеры (Лучшие сверху)
    cluster_scores.sort(key=lambda x: x['score'], reverse=True)
    
    if retention_percentile <= 0:
        num_to_keep = 1
    elif retention_percentile >= 100:
        num_to_keep = len(cluster_scores)
    else:
        num_to_keep = max(1, math.ceil(len(cluster_scores) * (retention_percentile / 100.0)))
        
    kept_cluster_ids = [c['id'] for c in cluster_scores[:num_to_keep]]
    
    good_indices = np.where(np.isin(cluster_ids, kept_cluster_ids))[0]
    good_label_ids = [valid_labels[i] for i in good_indices]
    
    clean_mask = np.isin(labeled_matrix, good_label_ids).astype(np.uint8)
    
    return SpecFunc(clean_mask, bin_spec.freq, bin_spec.time)

# def filterLabelsCurves(
#     bin_spec: SpecFunc, 
#     spec: SpecFunc, 
#     threshold: float = 50.0  # 0-100, чем выше — тем строже фильтр
# ) -> SpecFunc:
    
#     labeled_matrix = makeLabels(bin_spec).matrix 
#     intensity_matrix = spec.matrix
    
#     regions = regionprops(labeled_matrix, intensity_image=intensity_matrix)
    
#     scores = []
#     valid_labels = []
    
#     for r in regions:
#         if r.area < 5:
#             continue
        
#         score = r.eccentricity * r.solidity * np.sqrt(r.area)
        
#         scores.append(score)
#         valid_labels.append(r.label)
        
#     if len(scores) == 0:
#         return bin_spec.cloneApply(lambda arr: np.zeros_like(arr))
    
#     scores = np.array(scores)
#     cutoff = np.percentile(scores, threshold)
    
#     good_labels = [label for label, s in zip(valid_labels, scores) if s >= cutoff]
    
#     clean_mask = np.isin(labeled_matrix, good_labels).astype(np.uint8)
#     return SpecFunc(clean_mask, bin_spec.freq, bin_spec.time)

def filterLabelsCurves(
    bin_spec: SpecFunc, 
    min_score: float = 2.56  # Твой порог: 0.8 (ecc) * 0.8 (sol) * sqrt(16)
) -> SpecFunc:
    
    labeled_matrix = makeLabels(bin_spec).matrix 
    
    # Считаем только то, что нужно формуле, и сразу отдаем в виде быстрых массивов numpy
    props = regionprops_table(
        labeled_matrix,
        properties=('label', 'area', 'eccentricity', 'solidity')
    )
    
    # 1. Отсеиваем мелочь (площадь < 5)
    valid_mask = props['area'] >= 5
    
    if not np.any(valid_mask):
        return bin_spec.cloneApply(lambda arr: np.zeros_like(arr))
    
    # Берем только валидные данные
    labels = props['label'][valid_mask]
    areas = props['area'][valid_mask]
    eccs = props['eccentricity'][valid_mask]
    sols = props['solidity'][valid_mask]
    
    # 2. Вычисляем твой score (математика 1-в-1 как в твоем цикле, но считается мгновенно)
    scores = eccs * sols * np.sqrt(areas)
    
    # 3. Применяем АБСОЛЮТНЫЙ ПОРОГ
    good_labels = labels[scores >= min_score]
    
    if len(good_labels) == 0:
        return bin_spec.cloneApply(lambda arr: np.zeros_like(arr))
    
    # 4. Быстрое наложение маски (вместо медленного np.isin)
    lut = np.zeros(labeled_matrix.max() + 1, dtype=np.uint8)
    lut[good_labels] = 1
    clean_mask = lut[labeled_matrix]
    
    return SpecFunc(clean_mask, bin_spec.freq, bin_spec.time)


import numpy as np
import hdbscan
from skimage.measure import regionprops_table
from sklearn.preprocessing import StandardScaler

def filterLabelsCurvesHDBSCAN(
    bin_spec: SpecFunc, 
    min_cluster_size: int = 5,
    top_clusters_to_keep: int = 1
) -> SpecFunc:
    
    labeled_matrix = makeLabels(bin_spec).matrix 
    
    # 1. ОПТИМИЗАЦИЯ СКОРОСТИ: Получаем свойства сразу в виде numpy-массивов
    # Без intensity_image это работает невероятно быстро
    props = regionprops_table(
        labeled_matrix, 
        properties=('label', 'area', 'eccentricity', 'solidity')
    )
    
    # Фильтруем откровенный мусор (площадь < 5)
    valid_mask = props['area'] >= 5
    if not np.any(valid_mask):
        return bin_spec.cloneApply(lambda arr: np.zeros_like(arr))
    
    valid_labels = props['label'][valid_mask]
    areas = props['area'][valid_mask]
    eccs = props['eccentricity'][valid_mask]
    sols = props['solidity'][valid_mask]
    
    # 2. ПОДГОТОВКА ПРИЗНАКОВ ДЛЯ HDBSCAN
    # Логарифмируем площадь, чтобы огромные регионы не ломали метрику расстояний
    features = np.column_stack([eccs, sols, np.log1p(areas)])
    
    # Нормализуем данные (HDBSCAN чувствителен к масштабу признаков)
    features_scaled = StandardScaler().fit_transform(features)
    
    # 3. КЛАСТЕРИЗАЦИЯ
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    cluster_labels = clusterer.fit_predict(features_scaled)
    
    # Вычисляем твой изначальный "скор" для оценки качества кластеров
    scores = eccs * sols * np.sqrt(areas)
    
    unique_clusters = set(cluster_labels) - {-1}  # -1 в HDBSCAN это выбросы (шум)
    
    good_labels = []
    
    if not unique_clusters:
        # Fallback: Если HDBSCAN счел ВСЁ шумом (бывает при малом кол-ве данных),
        # просто берем верхние 20% по старому методу
        cutoff = np.percentile(scores, 80)
        good_labels = valid_labels[scores >= cutoff]
    else:
        # 4. ВЫБОР ЛУЧШИХ КЛАСТЕРОВ
        cluster_scores = []
        for cluster_id in unique_clusters:
            # Находим средний score для элементов этого кластера
            mean_score = np.mean(scores[cluster_labels == cluster_id])
            cluster_scores.append((cluster_id, mean_score))
        
        # Сортируем кластеры по убыванию их среднего score
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Берем нужное количество лучших кластеров
        best_clusters = [c_id for c_id, _ in cluster_scores[:top_clusters_to_keep]]
        
        # Собираем лейблы, которые попали в эти лучшие кластеры
        good_labels = valid_labels[np.isin(cluster_labels, best_clusters)]
    
    if len(good_labels) == 0:
        return bin_spec.cloneApply(lambda arr: np.zeros_like(arr))
    
    # 5. ОПТИМИЗАЦИЯ СКОРОСТИ: Замена np.isin на Lookup Table (LUT)
    # Это работает за О(1) для каждого пикселя, в отличие от долгого поиска в np.isin
    lut = np.zeros(labeled_matrix.max() + 1, dtype=np.uint8)
    lut[good_labels] = 1
    clean_mask = lut[labeled_matrix]
    
    return SpecFunc(clean_mask, bin_spec.freq, bin_spec.time)

def gradientSquared(spec: SpecFunc, ksize: int = 3, blur_sigma: float = 0.0) -> SpecFunc:
    """
    Вычисляет квадрат модуля градиента спектрограммы.
    Отлично выделяет контуры сигналов (перепады энергии) и полностью гасит монотонный стационарный фон.
    
    :param ksize: Размер ядра Собеля (1, 3, 5, 7). 3 - стандарт для выделения тонких краев.
    :param blur_sigma: Опциональное предварительное размытие по Гауссу. 
                       В вычислении производных любое "зерно" шума усиливается. 
                       Легкий блюр (например, 0.5 или 1.0) перед градиентом - золотое правило CV.
    """
    def apply_gradient(arr):
        # Обязательно переводим во float32, иначе при возведении в квадрат будет переполнение (overflow)
        img = arr.astype(np.float32)
        
        # Легкий блюр для подавления высокочастотного шума перед взятием производной
        if blur_sigma > 0:
            img = gaussian_filter(img, sigma=blur_sigma)
            
        # Производная по оси X (по времени)
        # cv2.CV_32F позволяет получать отрицательные значения производной
        gx = cv2.Sobel(img, cv2.CV_32F, dx=1, dy=0, ksize=ksize)
        
        # Производная по оси Y (по частоте)
        gy = cv2.Sobel(img, cv2.CV_32F, dx=0, dy=1, ksize=ksize)
        
        # Квадрат модуля градиента: |∇I|^2 = gx^2 + gy^2
        grad_sq = (gx ** 2) + (gy ** 2)
        
        return grad_sq

    return spec.cloneApply(apply_gradient)

import numpy as np
from scipy.linalg import pinv
from skimage.util import view_as_windows
from BatSpec.Logic.Functions import SpecFunc

def adaptiveMahalanobisTexture(
    spec: SpecFunc, 
    patch_size: int = 3, 
    block_time: int = 256, 
    block_freq: int = 64
) -> SpecFunc:
    """
    Адаптивный фильтр Махаланобиса для поиска аномалий текстуры (летучих мышей).
    Танцует от шума: локально оценивает структуру шума в блоках и подсвечивает то, что в нее не вписывается.
    
    :param patch_size: Размер окна текстуры (3 или 5). 3x3 = 9 признаков. Идеально ловит края и изгибы.
    :param block_time: Размер локального окна адаптации по времени (кадры).
    :param block_freq: Размер локального окна адаптации по частоте (бины).
    Хуйня
    """
    
    def apply_mahalanobis(matrix):
        T_len, F_len = matrix.shape
        
        # 1. Паддинг, чтобы после извлечения патчей размер остался исходным
        pad_w = patch_size // 2
        padded = np.pad(matrix, pad_width=pad_w, mode='reflect')
        
        # 2. Создаем 3D тензор признаков: каждый пиксель теперь описывается патчем (patch_size x patch_size)
        # Форма: (T_len, F_len, patch_size, patch_size)
        patches = view_as_windows(padded, (patch_size, patch_size))
        
        # Сплющиваем патчи: Форма (T_len, F_len, N_features), где N_features = 9 (при patch_size=3)
        features = patches.reshape(T_len, F_len, -1)
        n_features = features.shape[-1]
        
        # Результирующая матрица аномалий
        anomaly_map = np.zeros((T_len, F_len), dtype=np.float32)
        
        # Матрица регуляризации (чтобы ковариация никогда не "падала" с ошибкой деления на ноль)
        reg_matrix = np.eye(n_features) * 1e-6
        
        # 3. Скользящие блоки адаптации (считаем локальный фон)
        for t in range(0, T_len, block_time):
            for f in range(0, F_len, block_freq):
                # Вырезаем текущий блок
                t_end = min(t + block_time, T_len)
                f_end = min(f + block_freq, F_len)
                
                block_features = features[t:t_end, f:f_end, :] # (H, W, 9)
                H, W, _ = block_features.shape
                
                # Вытягиваем блок в 2D матрицу для линейной алгебры: (H*W, 9)
                X = block_features.reshape(-1, n_features)
                
                # --- Математика Махаланобиса для локального шума ---
                # Среднее по блоку
                mu = np.mean(X, axis=0)
                
                # Ковариация шума в этом конкретном блоке!
                cov = np.cov(X, rowvar=False) + reg_matrix
                
                # Обратная матрица ковариации
                inv_cov = pinv(cov)
                
                # Считаем расстояния для каждого пикселя в блоке
                diff = X - mu
                
                # Векторизованное (diff * inv_cov * diff^T)
                # Это в сотни раз быстрее, чем вызов scipy.spatial.distance.mahalanobis
                left_term = np.dot(diff, inv_cov)
                dist_sq = np.sum(left_term * diff, axis=1)
                
                # Записываем результат обратно в карту
                anomaly_map[t:t_end, f:f_end] = np.sqrt(np.maximum(dist_sq, 0)).reshape(H, W)
                
        return anomaly_map

    return spec.cloneApply(apply_mahalanobis)


def mahalanobisEtalonTexture(
    spec: SpecFunc, 
    noise_idx: list, 
    patch_shape: Tuple[int, int] = (5, 5)
) -> SpecFunc:
    """
    Текстурный фильтр Махаланобиса со строгим эталонным шумом и скользящим окном.
    
    1. Обучает модель текстуры (ковариацию) ТОЛЬКО на строках noise_idx.
    2. Скользит окном (patch_shape) по всей спектрограмме пиксель за пикселем.
    3. Выдает карту расстояний: чем выше значение, тем меньше текстура похожа на шум.
    """
    if len(noise_idx) < 2:
        print("Too few noise frames — skipping Mahalanobis texture filter.")
        return spec.cloneApply(lambda arr: arr.copy())

    def apply_mahalanobis(matrix):
        T_len, F_len = matrix.shape
        pt, pf = patch_shape
        pad_t, pad_f = pt // 2, pf // 2
        
        # 1. Делаем паддинг, чтобы размер спектрограммы не изменился после скользящего окна
        padded = np.pad(matrix, ((pad_t, pad_t), (pad_f, pad_f)), mode='reflect')
        
        # 2. Создаем виртуальное представление всех окон (без копирования в память!)
        # Форма: (T_len, F_len, pt, pf)
        patches = view_as_windows(padded, patch_shape)
        K = pt * pf  # Длина вектора признаков (например, 25 для окна 5x5)
        
        # 3. --- ОБУЧЕНИЕ НА ЭТАЛОНЕ ---
        # Вытаскиваем окна только для тех кадров времени, где есть шум
        noise_patches = patches[noise_idx, :, :, :]  # Форма: (n_noise, F_len, pt, pf)
        
        # Вытягиваем все шумовые окна в плоские векторы (N, 25)
        X_noise = noise_patches.reshape(-1, K)
        
        # Считаем среднее
        mu = np.mean(X_noise, axis=0)
        
        # Считаем ковариацию. Добавляем маленькую диагональ для защиты от деления на ноль
        cov = np.cov(X_noise, rowvar=False) + np.eye(K) * 1e-6
        inv_cov = pinv(cov)  # Обратная матрица ковариации
        
        # 4. --- ПРИМЕНЕНИЕ КО ВСЕЙ СПЕКТРОГРАММЕ ---
        # Чтобы не съесть всю оперативную память (MemoryError), обрабатываем чанками по времени
        result = np.zeros((T_len, F_len), dtype=np.float32)
        chunk_size = 5000  # Обрабатываем по 5000 кадров за раз
        
        for i in range(0, T_len, chunk_size):
            end = min(i + chunk_size, T_len)
            
            # Достаем чанк и вытягиваем окна в векторы: (chunk_size * F_len, 25)
            X_chunk = patches[i:end, :, :, :].reshape(-1, K)
            
            # Считаем дистанцию
            diff = X_chunk - mu
            
            # Векторизованное перемножение матриц: (diff * inv_cov) * diff^T
            left_term = np.dot(diff, inv_cov)
            dist_sq = np.sum(left_term * diff, axis=1)
            
            # Извлекаем корень и кладем обратно в матрицу
            dist = np.sqrt(np.maximum(dist_sq, 0))
            result[i:end, :] = dist.reshape(end - i, F_len)
            
        return result

    return spec.cloneApply(apply_mahalanobis)



from skimage.filters import frangi
from BatSpec.Logic.Functions import SpecFunc
import numpy as np


def extractRidgesFrangi(spec: SpecFunc, sigmas=(1, 2, 3), alpha=0.5, beta=0.5, gamma=15.0) -> SpecFunc:
    """
    Выделяет непрерывные кривые линии (писки летучих мышей), игнорируя бесформенный шум.
    Не размазывает сигнал, а наоборот, сужает его до "хребта" (ridge).
    
    sigmas: Масштабы линий (в пикселях). Если мышь толщиной 1-3 пикселя, используем (1, 2, 3).
    alpha: Чувствительность к отклонениям от формы линии (ближе к 0 - строже).
    beta: Отсечение "пятен" (клякс).
    gamma: Порог контрастности. Если сделать адаптивным к шуму - будет scale-invariant.
    """
    def apply_frangi(matrix):
        # Желательно пропустить матрицу через логарифм или нормализацию перед этим
        # black_ridges=False, так как наши писки светлые на темном фоне
        ridges = frangi(
            matrix, 
            sigmas=sigmas, 
            alpha=alpha, 
            beta=beta, 
            gamma=gamma, 
            black_ridges=False
        )
        return ridges

    return spec.cloneApply(apply_frangi)

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Union, Tuple

def gaussianBlur(spec: SpecFunc, sigma: Union[float, Tuple[float, float]] = 1.0) -> SpecFunc:
    """
    Применяет гауссовское сглаживание к спектрограмме.
    
    :param sigma: Радиус размытия (стандартное отклонение Гауссианы). 
                  Можно передать одно число (например, 1.0) для равномерного размытия,
                  или кортеж (sigma_time, sigma_freq), если оси нужно размыть по-разному.
    """
    def apply_blur(arr):
        # Обязательно переводим во float32, чтобы scipy не ругался на типы данных
        return gaussian_filter(arr.astype(np.float32), sigma=sigma)

    return spec.cloneApply(apply_blur)
    

import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler

# from BatSpec.Logic.Functions import SpecFunc

def labelWithHDBSCAN(
    bin_spec: SpecFunc, 
    spec: SpecFunc, 
    min_cluster_size: int = 15, 
    min_samples: int = 1, 
    epsilon: float = 5.0,
    scale_weights: tuple = (1.0, 1.0, 0.5)
) -> SpecFunc:
    """
    Создает матрицу лейблов, используя HDBSCAN вместо connectedComponents.
    Кластеризует пиксели в пространстве (время, частота, энергия).

    :param bin_spec: Бинарная маска, откуда берутся координаты пикселей.
    :param spec: Исходная (логарифмированная) спектрограмма для получения энергии.
    :param min_cluster_size: Минимальное количество пикселей для формирования кластера.
    :param min_samples: Параметр HDBSCAN, влияющий на консервативность. 1 - самый мягкий.
    :param epsilon: Максимальное расстояние для "сшивания" кластеров. 
                    Если расстояние между облаками пикселей меньше epsilon, они сольются.
    :param scale_weights: Веса для осей (время, частота, энергия) перед масштабированием.
                          Позволяет сделать расстояние по одной оси "важнее" других.
    """
    
    binary_matrix = bin_spec.matrix > 0
    intensity_matrix = spec.matrix
    
    # 1. Извлекаем координаты и значения "активных" пикселей
    coords_y, coords_x = np.where(binary_matrix)
    if len(coords_x) == 0:
        print("[DEBUG] Нет активных пикселей, возвращаем пустую матрицу лейблов")
        return bin_spec.cloneApply(lambda arr: np.zeros_like(arr, dtype=np.int32))
        
    values = intensity_matrix[coords_y, coords_x]
    
    # Собираем данные в одну матрицу (N_points, 3)
    # y = частота, x = время
    points = np.vstack([coords_x, coords_y, values]).T
    
    print(f"[DEBUG] Всего точек для кластеризации: {len(points)}")
    print(f"[DEBUG] Параметры: min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={epsilon}")
    print(f"[DEBUG] Веса scale_weights={scale_weights}")
    
    # 2. Масштабирование признаков — только взвешивание, StandardScaler НЕ используется
    weighted_points = points * np.array(scale_weights)
    
    # 3. Кластеризация
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(weighted_points)
    
    # Отладочная информация о результате кластеризации
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("[DEBUG] Результаты кластеризации:")
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"  Шум (-1): {count} точек")
        else:
            print(f"  Кластер {label}: {count} точек")
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"[DEBUG] Всего кластеров (без шума): {n_clusters}")
    
    # 4. Воссоздание матрицы лейблов
    labeled_matrix = np.zeros_like(binary_matrix, dtype=np.int32)
    
    # Фильтруем шумный кластер (-1)
    valid_indices = np.where(labels != -1)[0]
    
    # Получаем координаты и новые лейблы только для валидных точек
    valid_coords_y = coords_y[valid_indices]
    valid_coords_x = coords_x[valid_indices]
    valid_labels = labels[valid_indices]
    
    # Присваиваем лейблы. +1, чтобы лейблы начинались с 1 (как в skimage.label)
    # и не было конфликта с фоном (0).
    labeled_matrix[valid_coords_y, valid_coords_x] = valid_labels + 1
    
    print(f"[DEBUG] В матрицу лейблов записано {len(valid_indices)} точек (кластеры +1)")
    
    return SpecFunc(labeled_matrix, bin_spec.freq, bin_spec.time)


import os
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

# from BatSpec.Logic.Functions import SpecFunc

def extractPatternsNCC(
    spec: SpecFunc, 
    pattern_length: int = 200, 
    n_candidates: int = 500,
    proj_dim: int = 64, 
    dbscan_eps: float = 0.2, 
    ncc_threshold: float = 0.6,
    save_path: str = None
) -> List[SpecFunc]:
    import matplotlib.pyplot as plt

    """
    Находит повторяющиеся 2D паттерны (с трансляцией только по оси времени) через 
    Random Projections, кластеризацию DBSCAN и точный поиск NCC.
    
    Возвращает СПИСОК SpecFunc, где каждый элемент — это отдельный найденный паттерн,
    расставленный по своим координатам на пустом фоне.
    
    :param pattern_length: Длина паттерна по оси времени (в кадрах).
    :param n_candidates: Сколько топ-участков по энергии брать для кластеризации.
    :param proj_dim: Размерность случайных проекций.
    :param dbscan_eps: Порог косинусного расстояния для DBSCAN.
    :param ncc_threshold: Порог качества совпадения при финальном поиске.
    :param save_path: Если указать путь, туда сохранятся картинки базисов.
    """
    T_len, F_len = spec.matrix.shape
    W = pattern_length
    
    if T_len < W + 10:
        print("Изображение слишком короткое для заданного pattern_length.")
        return []

    # Защита от Out of Memory
    estimated_ram_mb = (n_candidates * W * F_len * 4) / (1024**2)
    if estimated_ram_mb > 1500: 
        safe_candidates = int((1500 * (1024**2)) / (W * F_len * 4))
        print(f"[Warning] n_candidates={n_candidates} потребует ~{estimated_ram_mb:.0f} MB RAM. Снижаю до {safe_candidates}.")
        n_candidates = safe_candidates

    img_f32 = spec.matrix.astype(np.float32)

    # --- 1. Грубый поиск кандидатов по энергии ---
    print("1. Поиск участков-кандидатов по энергии...")
    energy = np.convolve(np.sum(img_f32**2, axis=1), np.ones(W, dtype=np.float32), 'valid')
    
    peaks, _ = find_peaks(energy, distance=W//2)
    if len(peaks) > n_candidates:
        top_peak_indices = peaks[np.argsort(energy[peaks])[-n_candidates:]]
    else:
        top_peak_indices = peaks
        
    if len(top_peak_indices) < 10:
        print("Слишком мало кандидатов для кластеризации.")
        return []

    # --- 2. Извлечение и Random Projections ---
    print(f"2. Сжатие {len(top_peak_indices)} патчей случайными проекциями...")
    patches = np.array([img_f32[i:i+W] for i in top_peak_indices])
    vectorized_patches = patches.reshape(len(patches), -1)
    
    np.random.seed(42)
    proj_matrix = np.random.randn(W * F_len, proj_dim).astype(np.float32) / np.sqrt(proj_dim)
    reduced_vectors = vectorized_patches @ proj_matrix

    # --- 3. Кластеризация ---
    print("3. Кластеризация патчей (DBSCAN)...")
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=3, metric='cosine')
    labels = dbscan.fit_predict(reduced_vectors)
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    if n_clusters == 0:
        print("Не найдено кластеров. Попробуйте увеличить dbscan_eps.")
        return []

    # --- 4. Формирование базисов и ТОЧНЫЙ поиск NCC ---
    print(f"4. Найдено {n_clusters} кластеров. Запуск точного поиска (NCC)...")
    
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # Список, в который будем собирать отдельные SpecFunc
    component_specs: List[SpecFunc] = []

    for k in unique_labels:
        if k == -1:
            continue

        class_mask = (labels == k)
        cluster_patches = patches[class_mask]
        
        # Усредняем патчи кластера -> получаем идеальный чистый базис
        clean_basis = np.mean(cluster_patches, axis=0)
        
        basis_norm = np.linalg.norm(clean_basis)
        if basis_norm == 0:
            continue
            
        # --- NCC (Normalized Cross-Correlation) ---
        S = np.zeros(T_len - W + 1, dtype=np.float32)
        for c in range(F_len):
            S += np.correlate(img_f32[:, c], clean_basis[:, c], mode='valid')
            
        window_norms = np.sqrt(energy)
        NCC = S / (window_norms * basis_norm + 1e-9)
        
        final_coords, _ = find_peaks(NCC, height=ncc_threshold, distance=W//2)
        
        if len(final_coords) == 0:
            print(f"  -> Базис {k+1}: не прошел финальный порог NCC.")
            continue
            
        print(f"  -> Базис {k+1}: найдено {len(final_coords)} вхождений (макс. NCC = {NCC[final_coords].max():.2f})")
        
        # Создаем пустую матрицу ТОЛЬКО для этого конкретного паттерна
        component_image = np.zeros_like(img_f32, dtype=np.float32)
        for coord in final_coords:
            # Умножаем на NCC[coord] для подгонки амплитуды под локальный сигнал
            component_image[coord:coord+W] += clean_basis * NCC[coord]
            
        # Создаем объект SpecFunc и добавляем в список
        comp_spec = SpecFunc(component_image, spec.freq, spec.time)
        component_specs.append(comp_spec)
        
        # Опциональное сохранение визуалов на диск
        if save_path:
            def norm_img(img):
                mn, mx = img.min(), img.max()
                return (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)

            plt.imsave(os.path.join(save_path, f'basis_{k+1}.png'), norm_img(clean_basis), cmap='gray')
            plt.imsave(os.path.join(save_path, f'component_{k+1}.png'), norm_img(component_image), cmap='gray')

    return component_specs

import numpy as np
from scipy.signal import find_peaks, peak_widths
from sklearn.cluster import DBSCAN
from typing import Tuple

# from BatSpec.Logic.Functions import SpecFunc

def _auto_estimate_length(energy_profile: np.ndarray, min_len: int = 30, max_len: int = 800) -> int:
    """
    Автоматически определяет длину паттерна по профилю энергии.
    Ищет пики и измеряет их ширину на половине высоты (FWHM).
    """
    # Находим выраженные пики энергии
    peaks, properties = find_peaks(energy_profile, prominence=np.max(energy_profile) * 0.15)
    
    if len(peaks) < 3:
        return min_len # Если пиков мало, возвращаем безопасный минимум
        
    # Измеряем ширину каждого пика на уровне половины его высоты (rel_height=0.5 для FWHM)
    widths, _, _, _ = peak_widths(energy_profile, peaks, rel_height=0.5)
    
    # Берем медиану, чтобы игнорировать аномально широкие или узкие артефакты
    median_width = int(np.median(widths))
    
    # Добавляем 20% запаса, так как энергия часто "хвостами" размазывается шире основного узора
    estimated_len = int(median_width * 1.2)
    
    # Ограничиваем рамками разумного
    return np.clip(estimated_len, min_len, max_len)


def splitDominantPattern(
    spec: SpecFunc, 
    n_candidates: int = 300,
    dbscan_eps: float = 0.2, 
    ncc_threshold: float = 0.5,
    min_pattern_len: int = 30,
    max_pattern_len: int = 800
) -> Tuple[SpecFunc, SpecFunc]:
    """
    Итеративно находит САМЫЙ сильный паттерн на спектрограмме, извлекает его 
    и возвращает кортеж (Очищенная_спектрограмма, Извлеченный_паттерн).
    
    Математическое правило: Исходная.matrix == Очищенная.matrix + Извлеченный.matrix
    (Без отрицательных значений и артефактов вычитания).
    """
    T_len, F_len = spec.matrix.shape
    img_f32 = spec.matrix.astype(np.float32)
    
    # --- 1. Автоопределение длины паттерна ---
    print("1. Автоматическое определение длины паттерна...")
    energy_profile = np.sum(img_f32**2, axis=1)
    auto_W = _auto_estimate_length(energy_profile, min_len=min_pattern_len, max_len=max_pattern_len)
    print(f"   Определена длина паттерна: {auto_W} кадров")
    
    if T_len < auto_W + 10:
        return spec.cloneApply(lambda arr: arr.copy()), spec.cloneApply(lambda arr: np.zeros_like(arr))

    # Защита от OOM
    estimated_ram_mb = (n_candidates * auto_W * F_len * 4) / (1024**2)
    if estimated_ram_mb > 1000: 
        n_candidates = int((1000 * (1024**2)) / (auto_W * F_len * 4))

    # --- 2. Поиск кандидатов ---
    print("2. Извлечение кандидатов...")
    energy_conv = np.convolve(energy_profile, np.ones(auto_W, dtype=np.float32), 'valid')
    peaks, _ = find_peaks(energy_conv, distance=auto_W//2)
    
    if len(peaks) > n_candidates:
        top_peak_indices = peaks[np.argsort(energy_conv[peaks])[-n_candidates:]]
    else:
        top_peak_indices = peaks
        
    if len(top_peak_indices) < 5:
        print("   Слишком мало кандидатов. Возвращаем исходник.")
        return spec.cloneApply(lambda arr: arr.copy()), spec.cloneApply(lambda arr: np.zeros_like(arr))

    # --- 3. Random Projections и кластеризация ---
    print("3. Поиск доминирующего кластера...")
    patches = np.array([img_f32[i:i+auto_W] for i in top_peak_indices])
    vectorized_patches = patches.reshape(len(patches), -1)
    
    np.random.seed(42)
    proj_dim = min(64, auto_W * F_len)
    proj_matrix = np.random.randn(auto_W * F_len, proj_dim).astype(np.float32) / np.sqrt(proj_dim)
    reduced_vectors = vectorized_patches @ proj_matrix

    dbscan = DBSCAN(eps=dbscan_eps, min_samples=3, metric='cosine')
    labels = dbscan.fit_predict(reduced_vectors)
    
    unique_labels = set(labels) - {-1}
    if not unique_labels:
        print("   Кластеры не найдены.")
        return spec.cloneApply(lambda arr: arr.copy()), spec.cloneApply(lambda arr: np.zeros_like(arr))

    # Находим КРУПНЕЙШИЙ кластер (самый частый паттерн)
    dominant_label = max(unique_labels, key=lambda lbl: np.sum(labels == lbl))
    class_mask = (labels == dominant_label)
    cluster_patches = patches[class_mask]
    
    # Формируем чистый базис
    clean_basis = np.mean(cluster_patches, axis=0)
    basis_norm = np.linalg.norm(clean_basis)
    
    if basis_norm == 0:
         return spec.cloneApply(lambda arr: arr.copy()), spec.cloneApply(lambda arr: np.zeros_like(arr))

    # --- 4. Точный NCC поиск по ВСЕЙ спектрограмме ---
    print("4. Точный поиск NCC по всему файлу...")
    S = np.zeros(T_len - auto_W + 1, dtype=np.float32)
    for c in range(F_len):
        S += np.correlate(img_f32[:, c], clean_basis[:, c], mode='valid')
        
    window_norms = np.sqrt(energy_conv)
    NCC = S / (window_norms * basis_norm + 1e-9)
    
    final_coords, _ = find_peaks(NCC, height=ncc_threshold, distance=auto_W//2)
    
    if len(final_coords) == 0:
        print("   Паттерн не прошел финальный порог NCC.")
        return spec.cloneApply(lambda arr: arr.copy()), spec.cloneApply(lambda arr: np.zeros_like(arr))
        
    print(f"   Найдено {len(final_coords)} вхождений доминирующего паттерна.")

    # --- 5. Формирование результатов со строгим соблюдением баланса ---
    extracted_matrix = np.zeros_like(img_f32, dtype=np.float32)
    
    for coord in final_coords:
        # Ожидаемая амплитуда паттерна в данной точке
        expected_signal = clean_basis * NCC[coord]
        
        # ВАЖНО: Ограничиваем извлекаемый сигнал локальным максимумом исходника.
        # Это гарантирует, что мы не вычтем больше, чем там есть (не будет отрицательного шума).
        safe_extracted = np.minimum(expected_signal, img_f32[coord:coord+auto_W])
        
        extracted_matrix[coord:coord+auto_W] += safe_extracted

    # Вычитание. Так как extracted_matrix <= img_f32 поэлементно, 
    # cleaned_matrix гарантированно не содержит отрицательных значений!
    cleaned_matrix = img_f32 - extracted_matrix
    
    # Создаем объекты SpecFunc
    cleaned_spec = SpecFunc(cleaned_matrix, spec.freq, spec.time)
    extracted_spec = SpecFunc(extracted_matrix, spec.freq, spec.time)

    return cleaned_spec, extracted_spec

import numpy as np
from scipy.ndimage import median_filter
from typing import Union, Tuple
from BatSpec.Logic.Functions import SpecFunc

def localNormalizeMAD(
    spec: SpecFunc, 
    window_size: Union[int, Tuple[int, int]] = (7, 7), 
    epsilon: float = 1e-6
) -> SpecFunc:
    """
    Локальная робастная Z-нормализация на основе медианы и MAD.
    Идеально подходит для поиска аномалий/пиков, так как яркие сигналы
    не искажают оценку локального фона (не создают черных ореолов).

    :param window_size: Размер скользящего окна (время, частота).
    :param epsilon: Защита от деления на ноль в зонах с нулевым шумом.
    """
    # Приводим к кортежу, если передано одно число
    if isinstance(window_size, int):
        window_size = (window_size, window_size)

    def apply_local_mad(arr):
        # 1. Считаем локальную медиану (оценка фона без учета пиков)
        local_median = median_filter(arr, size=window_size, mode='reflect')
        
        # 2. Считаем абсолютное отклонение каждого пикселя от локальной медианы
        abs_deviation = np.abs(arr - local_median)
        
        # 3. Считаем локальный MAD (медиана от абсолютных отклонений)
        local_mad = median_filter(abs_deviation, size=window_size, mode='reflect')
        
        # 4. Переводим MAD в приближенное стандартное отклонение (std).
        # Коэффициент 1.4826 делает MAD эквивалентным std для нормального распределения.
        # Это полезно, чтобы итоговые значения были похожи на классические Z-scores.
        local_std = local_mad * 1.4826
        
        # Защита от деления на ноль (в местах, где фон абсолютно ровный)
        local_std[local_std < epsilon] = 1.0
        
        # 5. Вычисляем робастный Z-score: (X - Median) / (MAD * 1.4826)
        robust_z_score = (arr - local_median) / local_std
        
        return robust_z_score

    return spec.cloneApply(apply_local_mad)


# from skimage.restoration import denoise_bilateral

# def bilateralBlur(
#     spec: SpecFunc,
#     sigma_spatial: float = 3.0,
#     sigma_intensity: float = 0.1,
#     win_size: int | None = None
# ) -> SpecFunc:
#     """
#     Применяет билатеральный фильтр к спектрограмме.
    
#     :param sigma_spatial:   пространственное сглаживание (в пикселях).
#                             Аналог σ_s в классическом bilateral filter.
#     :param sigma_intensity: сглаживание по интенсивности (очень важный параметр!).
#                             Для спектрограмм обычно ставят в диапазоне 0.05–0.3
#                             (зависит от нормализации спектрограммы).
#     :param win_size:        размер окна (diameter). Если None — skimage выберет автоматически 
#                             (обычно 3–5 × sigma_spatial).
#     """
#     def apply_bilateral(arr: np.ndarray) -> np.ndarray:
#         # skimage.restoration.denoise_bilateral ожидает float32/float64 в диапазоне [0, 1]
#         # или любого float. Поэтому приводим аккуратно.
#         arr = arr.astype(np.float32)
        
#         # Если спектрограмма в децибелах (логарифмическая) — sigma_intensity должен быть небольшим
#         filtered = denoise_bilateral(
#             arr,
#             win_size=win_size,
#             sigma_spatial=sigma_spatial,
#             sigma_color=sigma_intensity,   # sigma_color = sigma_intensity
#             channel_axis=None              # 2D массив, не изображение с каналами
#         )
#         return filtered

#     return spec.cloneApply(apply_bilateral)

import cv2

def bilateralBlur(
    spec: SpecFunc,
    sigma_spatial: float = 5.0,
    sigma_intensity: float = 0.1,
    d: int = 9
) -> SpecFunc:
    """
    Билатеральный фильтр через OpenCV (быстрее).
    """
    def apply_bilateral(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        
        # cv2.bilateralFilter требует uint8 или float32, но лучше привести к 0-1 или 0-255
        # Простой вариант — нормализуем:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            normalized = (arr - arr_min) / (arr_max - arr_min)
        else:
            normalized = arr
            
        filtered = cv2.bilateralFilter(
            normalized,
            d=d,                    # диаметр окна
            sigmaColor=sigma_intensity * 255,   # т.к. normalized в [0,1]
            sigmaSpace=sigma_spatial
        )
        
        # Возвращаем в исходный диапазон
        return filtered * (arr_max - arr_min) + arr_min

    return spec.cloneApply(apply_bilateral)


import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# from BatSpec.Logic.Functions import SpecFunc

def makeLabelsWatershed(
    spec: SpecFunc, 
    threshold: float = 0.3, 
    min_distance: int = 5
) -> SpecFunc:
    """
    Создает матрицу лейблов с помощью маркерного водораздела (Watershed).
    Разделяет слипшиеся сигналы, проводя границу строго по локальному минимуму (седлу) между пиками.
    
    :param threshold: Минимальный уровень энергии (внешняя "пленка"). Все что ниже - фон (0).
    :param min_distance: Минимальное расстояние (в пикселях) между двумя пиками. 
                         Чем больше значение, тем сильнее защита от ложного дробления 
                         одного сигнала на два из-за мелкого шума на макушке.
    """
    def apply_watershed(arr):
        # Гарантируем корректный тип
        img = arr.astype(np.float32)
        
        # 1. Задаем маску: отсекаем весь фон, оставляем только области > threshold
        mask = img >= threshold
        
        # 2. Ищем локальные максимумы (центры "зарядов") СТРОГО внутри маски
        peaks_coords = peak_local_max(img, min_distance=min_distance, labels=mask)
        
        # Если сигналов нет — возвращаем пустую матрицу нулей
        if len(peaks_coords) == 0:
            return np.zeros_like(img, dtype=np.int32)
            
        # 3. Маркируем пики уникальными ID (1, 2, 3...)
        markers = np.zeros_like(img, dtype=np.int32)
        for i, (r, c) in enumerate(peaks_coords):
            markers[r, c] = i + 1
            
        # 4. Научно обоснованный Flood Fill (распространение фронта)
        # Передаем -img, чтобы алгоритм тек с верхушек (максимумов) вниз к границам маски.
        # В месте столкновения двух зон образуется граница (по седловой точке).
        labels = watershed(-img, markers, mask=mask)
        
        return labels.astype(np.int32)

    return spec.cloneApply(apply_watershed)


def fillFromSpec(
    crop: SpecFunc, 
    big_spec: SpecFunc, 
    apply_mask: bool = False
) -> SpecFunc:
    """
    Находит положение вырезки `crop` внутри `big_spec` по осям времени и частоты.
    Берет оригинальные значения из `big_spec.matrix` и копирует их в вырезку.
    
    Оси: [0] - Время (T), [1] - Частоты (H)
    """
    if len(crop.time) == 0 or len(crop.freq) == 0:
        return crop

    # 1. Ищем начальные индексы в большой спектрограмме.
    # Используем argmin(abs(...)) для безопасного поиска float-значений
    t_start_idx = np.argmin(np.abs(big_spec.time - crop.time[0]))
    f_start_idx = np.argmin(np.abs(big_spec.freq - crop.freq[0]))
    
    # 2. Длины осей вырезки
    len_t = len(crop.time)
    len_f = len(crop.freq)
    
    # 3. Вычисляем конечные индексы (гарантирует точное совпадение размерностей)
    t_end_idx = t_start_idx + len_t
    f_end_idx = f_start_idx + len_f
    
    # 4. Вырезаем кусок матрицы из большой спектрограммы
    real_sub_matrix = big_spec.matrix[t_start_idx:t_end_idx, f_start_idx:f_end_idx]
    
    # Защита от выхода за границы
    if real_sub_matrix.shape != crop.matrix.shape:
        raise ValueError(
            f"Размеры не совпали! Ожидали {crop.matrix.shape}, "
            f"получили из большой спеки {real_sub_matrix.shape}. "
            "Возможно, вырезка выходит за границы big_spec."
        )

    # 5. Копируем в себя
    if apply_mask:
        # Копируем оригинальные значения только туда, где была разметка (> 0)
        # Остальной фон в crop.matrix останется нулями
        mask = (crop.matrix > 0)
        new_matrix = np.zeros_like(real_sub_matrix, dtype=big_spec.matrix.dtype)
        new_matrix[mask] = real_sub_matrix[mask]
        crop.matrix = new_matrix
    else:
        # Полностью перезаписываем матрицу вырезки оригинальным прямоугольником (вместе с фоном)
        crop.matrix = real_sub_matrix.copy()

    # Если нужно обновить тип данных
    crop.matrix = crop.matrix.astype(big_spec.matrix.dtype)

    return crop

def centrateCorpMass(corp: SpecFunc):
    return SpecFunc(
        corp.matrix.copy(), 
        freq=corp.freq.copy(), 
        time=corp.time - np.average(corp.time, weights=corp.matrix.sum(axis=1))
    )
    