import math
from typing import List, Tuple, Any, Union

import librosa

from BatSpec.Logic.Functions import SpecFunc, TimeFunc
import numpy as np
from skimage.measure import label, regionprops
import hdbscan
from sklearn.preprocessing import StandardScaler

from BatSpec.Logic.Echo import AsymmetricEchoModel, CausalEchoModel
from BatSpec.Python.Graph import GraphNode, get_shared_cache
from numpy.typing import NDArray

# Добавил импорт gradientSquared
from . import (
    subEtalonNoise as _subEtalonNoise, removeEchoRichardsonLucy as _removeEchoRichardsonLucy, 
    makeLog as _makeLog, enhanceCurvesGabor as _enhanceCurvesGabor, normalizeFrecZ as _normalizeFrecZ,
    makeBinarization as _makeBinarization, harmonicPercussive as _harmonicPercussive, 
    mahalanobisEtalonTexture as _mahalanobisEtalonTexture, extractRidgesFrangi as _extractRidgesFrangi,
    makeLabels as _makeLabels, filterLabelsHDBSCAN as _filterLabelsHDBSCAN, 
    filterLabelsCurves as _filterLabelsCurves, removeEchoWiener as _removeEchoWiener,
    gradientSquared as _gradientSquared, gaussianBlur as _gaussianBlur,
    labelWithHDBSCAN as _labelWithHDBSCAN, extractPatternsNCC as _extractPatternsNCC,
    splitDominantPattern as _splitDominantPattern, localNormalizeMAD as _localNormalizeMAD,
    bilateralBlur as _bilateralBlur, makeLabelsWatershed as _makeLabelsWatershed
)

def makeLabelsWatershed(
    spec: GraphNode[SpecFunc], 
    threshold: float = 0.3, 
    min_distance: int = 5
) -> GraphNode[SpecFunc]:
    """
    Обёртка для алгоритма watershed, строит маску меток на основе спектрограммы.
    
    Параметры
    ----------
    spec : GraphNode[SpecFunc]
        Входная спектрограмма.
    threshold : float, default=0.3
        Порог для бинаризации перед применением watershed.
    min_distance : int, default=5
        Минимальное расстояние между пиками (для поиска локальных максимумов).

    Возвращает
    -------
    GraphNode[SpecFunc]
        Узел графа с результирующей маской меток (каждый объект помечен уникальным целым числом).
    """
    return spec.cache.get_or_create_node(
        f"makeLabelsWatershed({spec.key}, threshold={threshold}, min_distance={min_distance})",
        lambda: _makeLabelsWatershed(spec.value, threshold=threshold, min_distance=min_distance)
    )

def subEtalonNoise(spec: GraphNode[SpecFunc], noise_idx: GraphNode[NDArray[Any]], alpha: float = 1) -> GraphNode[SpecFunc]:
    return get_shared_cache(spec, noise_idx).get_or_create_node(
        f"subEtalonNoise({spec.key}, {noise_idx.key}, {alpha})", 
        lambda: _subEtalonNoise(spec.value, noise_idx.value, alpha=alpha)
    )

def removeEchoRichardsonLucy(spec: GraphNode[SpecFunc], model: CausalEchoModel | AsymmetricEchoModel, iterations: int = 15) -> GraphNode[SpecFunc]:
    return spec.cache.get_or_create_node(
        f"removeEchoRichardsonLucy({spec.key}, {model}, iterations={iterations})",
        lambda: _removeEchoRichardsonLucy(spec.value, model, iterations=iterations)
    )

def removeEchoWiener(spec: GraphNode[SpecFunc], model: CausalEchoModel, eps_factor: float = 1e-2) -> GraphNode[SpecFunc]:
    return spec.cache.get_or_create_node(
        f"removeEchoRichardsonLucy({spec.key}, {model}, iterations={eps_factor})",
        lambda: _removeEchoWiener(spec.value, model, eps_factor=eps_factor)
    )

def makeLog(spec: GraphNode[SpecFunc]) -> GraphNode[SpecFunc]:
    return spec.cache.get_or_create_node(
        f"makeLog({spec.key})",
        lambda: _makeLog(spec.value)
    )

def enhanceCurvesGabor(spec: GraphNode[SpecFunc], ksize: int = 16, sigma: float = 3.0, lambd: float = 8.0, gamma=0.5, pre_blur: float = 1.0) -> GraphNode[SpecFunc]:
    return spec.cache.get_or_create_node(
        f"enhanceCurvesGabor({spec.key}, ksize={ksize}, sigma={sigma}, lambd={lambd}, gamma={gamma}, pre_blur={pre_blur})",
        lambda: _enhanceCurvesGabor(spec.value, ksize=ksize, sigma=sigma, lambd=lambd, gamma=gamma, pre_blur=pre_blur)
    )

def normalizeFrecZ(spec: GraphNode[SpecFunc], noise_idx: GraphNode[NDArray[Any]]) -> GraphNode[SpecFunc]:
    return get_shared_cache(spec, noise_idx).get_or_create_node(
        f"normalizeFrecZ({spec.key}, {noise_idx.key})",
        lambda: _normalizeFrecZ(spec.value, noise_idx.value)  
    )

def makeBinarization(spec: GraphNode[SpecFunc], trashhold=32) -> GraphNode[SpecFunc]:
    return spec.cache.get_or_create_node(
        f"makeBinarization({spec.key}, trashhold={trashhold})",
        lambda: _makeBinarization(spec.value, trashhold=trashhold)
    )

def makeLabels(spec: GraphNode[SpecFunc]) -> GraphNode[SpecFunc]:
    return spec.cache.get_or_create_node(
        f"makeLabels({spec.key})",
        lambda: _makeLabels(spec.value)
    )


def labelWithHDBSCAN(
    bin_spec: GraphNode[SpecFunc], 
    spec: GraphNode[SpecFunc], 
    min_cluster_size: int = 15, 
    min_samples: int = 1, 
    epsilon: float = 5.0,
    scale_weights: tuple = (1.0, 1.0, 0.5)
) -> GraphNode[SpecFunc]:
        
    return get_shared_cache(bin_spec, spec).get_or_create_node(
        f"labelWithHDBSCAN({bin_spec.key}, {spec.key}, "
        f"min_size={min_cluster_size}, min_samples={min_samples}, "
        f"eps={epsilon}, weights={scale_weights})",
        
        lambda: _labelWithHDBSCAN(
            bin_spec=bin_spec.value, 
            spec=spec.value, 
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            epsilon=epsilon,
            scale_weights=scale_weights
        )
    )

def filterLabelsHDBSCAN(labels: GraphNode[SpecFunc], spec: GraphNode[SpecFunc], min_cluster_size=3, retention_percentile=50) -> GraphNode[SpecFunc]:
    return spec.cache.get_or_create_node(
        f"filterLabelsHDBSCAN({labels.key}, {spec.key}, min_cluster_size={min_cluster_size}, retention_percentile={retention_percentile})",
        lambda: _filterLabelsHDBSCAN(labels.value, spec.value, min_cluster_size=min_cluster_size, retention_percentile=retention_percentile)
    )

def filterLabelsCurves(bin: GraphNode[SpecFunc], min_score: float = 2.56):
    return bin.cache.get_or_create_node(
        f"filterLabelsCurves({bin.key}, min_score={min_score})", 
        lambda: _filterLabelsCurves(bin.value, min_score=min_score)
    )

from . import filterLabelsCurvesHDBSCAN as _filterLabelsCurvesHDBSCAN
def filterLabelsCurvesHDBSCAN(bin: GraphNode[SpecFunc], min_cluster_size: int = 3, top_clusters_to_keep: int = 5):
    return bin.cache.get_or_create_node(
        f"filterLabelsCurvesHDBSCAN({bin.key}, min_cluster_size={min_cluster_size}, top_clusters_to_keep={top_clusters_to_keep})", 
        lambda: _filterLabelsCurvesHDBSCAN(bin.value, min_cluster_size=min_cluster_size, top_clusters_to_keep=top_clusters_to_keep)
    )

def harmonicPercussive(
    spec: GraphNode[SpecFunc]
) -> Tuple[GraphNode[SpecFunc], GraphNode[SpecFunc]]:
    
    parent_key = f"harmonicPercussive({spec.key})"
    parent_node = spec.cache.get_or_create_node(
        parent_key, 
        lambda: _harmonicPercussive(spec.value)
    )

    l_node = spec.cache.get_or_create_node(
        f"{parent_key}[0]", 
        lambda: parent_node.value[0]
    )

    s_node = spec.cache.get_or_create_node(
        f"{parent_key}[1]", 
        lambda: parent_node.value[1]
    )

    return l_node, s_node

def adaptiveMahalanobisTexture(
    spec: GraphNode[SpecFunc], 
    patch_size: int = 3, 
    block_time: int = 256, 
    block_freq: int = 64
) -> GraphNode[SpecFunc]:
    return spec.cache.get_or_create_node(
        f"adaptiveMahalanobisTexture({spec.key}, patch_size={patch_size}, block_time={block_time}, block_freq={block_freq})",
        lambda: _adaptiveMahalanobisTexture(spec.value, patch_size=patch_size, block_time=block_time, block_freq=block_freq)
    )

def mahalanobisEtalonTexture(
    spec: GraphNode[SpecFunc], 
    noise_idx: GraphNode[NDArray[Any]], 
    patch_shape: Tuple[int, int] = (5, 5)
) -> GraphNode[SpecFunc]:
    return get_shared_cache(spec, noise_idx).get_or_create_node(
        f"mahalanobisEtalonTexture({spec.key}, {noise_idx.key}, patch_shape={patch_shape})",
        lambda: _mahalanobisEtalonTexture(spec.value, noise_idx.value, patch_shape=patch_shape)
    )

def extractRidgesFrangi(
    spec: GraphNode[SpecFunc], 
    sigmas: tuple = (1, 2, 3), 
    alpha: float = 0.5, 
    beta: float = 0.5, 
    gamma: float = 15.0
) -> GraphNode[SpecFunc]: 
    return spec.cache.get_or_create_node(
        f"extractRidgesFrangi({spec.key}, sigmas={sigmas}, alpha={alpha}, beta={beta}, gamma={gamma})",
        lambda: _extractRidgesFrangi(spec.value, sigmas=sigmas, alpha=alpha, beta=beta, gamma=gamma)
    )

def gradientSquared(
    spec: GraphNode[SpecFunc], 
    ksize: int = 3, 
    blur_sigma: float = 0.0
) -> GraphNode[SpecFunc]:
    return spec.cache.get_or_create_node(
        f"gradientSquared({spec.key}, ksize={ksize}, blur_sigma={blur_sigma})",
        lambda: _gradientSquared(spec.value, ksize=ksize, blur_sigma=blur_sigma)
    )

# Добавьте в список импортов:
# gaussianBlur as _gaussianBlur

def gaussianBlur(
    spec: GraphNode[SpecFunc], 
    sigma: float | tuple[float, float] = 1.0
) -> GraphNode[SpecFunc]:
    
    # Если sigma это кортеж, он корректно преобразуется в строку для ключа кэша
    return spec.cache.get_or_create_node(
        f"gaussianBlur({spec.key}, sigma={sigma})",
        lambda: _gaussianBlur(spec.value, sigma=sigma)
    )

def bilateralBlur(
    spec: GraphNode[SpecFunc], 
    sigma_spatial: float = 5.0,
    sigma_intensity: float = 0.1,
    d: int = 9
) -> GraphNode[SpecFunc]:
    
    # Если нужно, можно добавить поддержку кортежа для sigma_spatial позже
    return spec.cache.get_or_create_node(
        f"bilateralBlur({spec.key}, sigma_spatial={sigma_spatial}, "
        f"sigma_intensity={sigma_intensity}, d={d})",
        lambda: _bilateralBlur(spec.value, 
            sigma_spatial=sigma_spatial,
            sigma_intensity=sigma_intensity,
            d=d
        )
    )

def extractPatternsNCC(
    spec: GraphNode[SpecFunc], 
    pattern_length: int = 200, 
    n_candidates: int = 1000,
    proj_dim: int = 64, 
    dbscan_eps: float = 0.2, 
    ncc_threshold: float = 0.6,
    save_path: str = None
) -> GraphNode[List[SpecFunc]]:
    return spec.cache.get_or_create_node(
        f"extractPatternsNCC({spec.key}, "
        f"pattern_length={pattern_length}, "
        f"n_candidates={n_candidates},"
        f"proj_dim={proj_dim}, "
        f"dbscan_eps={dbscan_eps}, "
        f"ncc_threshold={ncc_threshold}, "
        f"save_path={save_path})",
        lambda: _extractPatternsNCC(spec.value, 
            pattern_length=pattern_length, 
            n_candidates=n_candidates,
            proj_dim=proj_dim, 
            dbscan_eps=dbscan_eps, 
            ncc_threshold=ncc_threshold,
            save_path=save_path
        )
    )


def splitDominantPattern(
    spec: GraphNode[SpecFunc],
    n_candidates: int = 300,
    dbscan_eps: float = 0.2,
    ncc_threshold: float = 0.5,
    min_pattern_len: int = 30,
    max_pattern_len: int = 800
) -> Tuple[GraphNode[SpecFunc], GraphNode[SpecFunc]]:
    """
    Обёртка для splitDominantPattern с кэшированием.
    """
    parent_key = (
        f"splitDominantPattern({spec.key}, "
        f"n_candidates={n_candidates}, dbscan_eps={dbscan_eps}, "
        f"ncc_threshold={ncc_threshold}, min_pattern_len={min_pattern_len}, "
        f"max_pattern_len={max_pattern_len})"
    )
    
    parent_node = spec.cache.get_or_create_node(
        parent_key,
        lambda: _splitDominantPattern(
            spec.value,
            n_candidates=n_candidates,
            dbscan_eps=dbscan_eps,
            ncc_threshold=ncc_threshold,
            min_pattern_len=min_pattern_len,
            max_pattern_len=max_pattern_len
        )
    )
    
    left_node = spec.cache.get_or_create_node(
        f"{parent_key}[0]",
        lambda: parent_node.value[0]
    )
    
    right_node = spec.cache.get_or_create_node(
        f"{parent_key}[1]",
        lambda: parent_node.value[1]
    )
    
    return left_node, right_node



def localNormalizeMAD(
    spec: GraphNode[SpecFunc],
    window_size: Union[int, Tuple[int, int]] = (7, 7),
    epsilon: float = 1e-6
) -> GraphNode[SpecFunc]:
    """
    Обёртка для localNormalizeMAD с кэшированием.
    """
    # Приводим window_size к строковому представлению для ключа кэша
    ws_repr = str(window_size) if isinstance(window_size, tuple) else f"({window_size},{window_size})"
    
    return spec.cache.get_or_create_node(
        f"localNormalizeMAD({spec.key}, window_size={ws_repr}, epsilon={epsilon})",
        lambda: _localNormalizeMAD(spec.value, window_size=window_size, epsilon=epsilon)
    )