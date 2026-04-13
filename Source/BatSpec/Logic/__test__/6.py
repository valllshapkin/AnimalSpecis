import sys
from BatSpec.API.spec import getSpectrogramm, updateVisLayer
from BatSpec.Logic.TransitionWorker import (
    clusterCallsDBSCAN, createClusterBasis, extractCalls, extractClusterPattern, findPeaks,
    specDistanceNormalized
)
from BatSpec.Logic.SpecWorker import (
    subEtalonNoise, normalizeFrecZ
)
from BatSpec.Logic.TransitionWorker.GRAPH import etalonNoise

from BatSpec.Logic.SpecWorker import subEtalonNoise
from BatSpec.Logic.TransitionWorker import etalonNoise, integrateFreq



# 1. Препроцессинг (без изменений)
spec = getSpectrogramm().value

noise = etalonNoise(spec, 5)
denoised = subEtalonNoise(spec, noise, alpha=1)
updateVisLayer("DENOISED", denoised)

func = integrateFreq(denoised)
peaks = findPeaks(func, width=0.0005, distance=0.0005, prominence=0.1)
print(f"Найдено пиков: {peaks.count}")

# 2. Вырезание с width_multiplier=3.0, как вы и хотели
normalized_spec = normalizeFrecZ(denoised, noise)
calls = extractCalls(peaks, normalized_spec, width_multiplier=3.0)

if len(calls) < 10: # DBSCAN не любит мало данных
    print("Найдено слишком мало коллов для кластеризации.")
    sys.exit()

# 3. КЛАСТЕРИЗАЦИЯ НОВЫМ МЕТОДОМ
# -------------------------------------------------------------------

METRIC = specDistanceNormalized
EPS = 0.01 # Подбирается экспериментально

labels = clusterCallsDBSCAN(calls, metric_func=METRIC, eps=EPS, min_samples=5)
# -------------------------------------------------------------------

# 4. Анализ результатов
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print(f'Найдено кластеров: {n_clusters_}')
print(f'Выбросов (шума): {n_noise_}')

# 5. Группировка и отправка в UI
unique_labels = set(labels)
for k in unique_labels:
    if k == -1: # Пропускаем выбросы
        continue
    
    cluster_calls = [call for i, call in enumerate(calls) if labels[i] == k]
    count = len(cluster_calls)
    
    print(f"\n--- Обработка Кластера {k} (Содержит {count} коллов) ---")
    
    basis = createClusterBasis(cluster_calls)
    extracted_spec = extractClusterPattern(normalized_spec, basis, prominence=0.3)
    
    layer_name = f"Cluster_{k}_({count}_items)"
    updateVisLayer(layer_name, extracted_spec)
    
print("\nГотово! Все кластеры отправлены в UI.")