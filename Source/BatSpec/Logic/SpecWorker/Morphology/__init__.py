import cv2
import numpy as np
from BatSpec.Logic.Functions import SpecFunc


def morphErode(spec: SpecFunc, radius: int = 2, 
               shape: int = cv2.MORPH_ELLIPSE) -> SpecFunc:
    """Эрозия"""
    binary = (spec.matrix > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(shape, (2*radius + 1, 2*radius + 1))
    result = cv2.erode(binary, kernel)
    return SpecFunc(result * spec.matrix.max(), spec.freq, spec.time)


def morphDilate(spec: SpecFunc, radius: int = 2, 
                shape: int = cv2.MORPH_ELLIPSE) -> SpecFunc:
    """Дилатация"""
    binary = (spec.matrix > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(shape, (2*radius + 1, 2*radius + 1))
    result = cv2.dilate(binary, kernel)
    return SpecFunc(result * spec.matrix.max(), spec.freq, spec.time)


def morphOpen(spec: SpecFunc, radius: int = 2, 
              shape: int = cv2.MORPH_ELLIPSE) -> SpecFunc:
    """Открытие"""
    binary = (spec.matrix > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(shape, (2*radius + 1, 2*radius + 1))
    result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return SpecFunc(result * spec.matrix.max(), spec.freq, spec.time)


def morphClose(spec: SpecFunc, radius: int = 2, 
               shape: int = cv2.MORPH_ELLIPSE) -> SpecFunc:
    """Закрытие"""
    binary = (spec.matrix > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(shape, (2*radius + 1, 2*radius + 1))
    result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return SpecFunc(result * spec.matrix.max(), spec.freq, spec.time)


def morphGradient(spec: SpecFunc, radius: int = 2, 
                  shape: int = cv2.MORPH_ELLIPSE) -> SpecFunc:
    """Морфологический градиент"""
    binary = (spec.matrix > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(shape, (2*radius + 1, 2*radius + 1))
    result = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    return SpecFunc(result * spec.matrix.max(), spec.freq, spec.time)


def morphSmooth(spec: SpecFunc, open_radius: int = 2, close_radius: int = 2) -> SpecFunc:
    """Сглаживание (Open + Close)"""
    binary = (spec.matrix > 0).astype(np.uint8)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*open_radius+1, 2*open_radius+1))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_radius+1, 2*close_radius+1))
    
    temp = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k1)
    result = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, k2)
    return SpecFunc(result * spec.matrix.max(), spec.freq, spec.time)


def morphHorizontalConnect(spec: SpecFunc, width: int = 10) -> SpecFunc:
    """Соединение по времени"""
    binary = (spec.matrix > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, width))
    result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return SpecFunc(result * spec.matrix.max(), spec.freq, spec.time)


def morphVerticalConnect(spec: SpecFunc, height: int = 10) -> SpecFunc:
    """Соединение по частоте"""
    binary = (spec.matrix > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (height, 1))
    result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return SpecFunc(result * spec.matrix.max(), spec.freq, spec.time)


def morphRemoveSmall(spec: SpecFunc, min_size: int = 10) -> SpecFunc:
    """Удаление мелких объектов"""
    binary = (spec.matrix > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 1
            
    return SpecFunc(cleaned * spec.matrix.max(), spec.freq, spec.time)


def morphSkeleton(spec: SpecFunc) -> SpecFunc:
    """Скелетизация"""
    binary = (spec.matrix > 0).astype(np.uint8)
    skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return SpecFunc(skeleton * spec.matrix.max(), spec.freq, spec.time)