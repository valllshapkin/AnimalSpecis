import cv2

from BatSpec.Logic.Functions import SpecFunc
from BatSpec.Python.Graph import GraphNode

# ====================== Морфологические операции (GraphNode wrappers) ======================


from . import morphErode as _morphErode
def morphErode(
    spec: GraphNode[SpecFunc], 
    radius: int = 2, 
    shape: int = cv2.MORPH_ELLIPSE
) -> GraphNode[SpecFunc]:
    """Эрозия"""
    return spec.cache.get_or_create_node(
        f"morphErode({spec.key}, radius={radius}, shape={shape})",
        lambda: _morphErode(spec.value, radius=radius, shape=shape)
    )


from . import morphDilate as _morphDilate
def morphDilate(
    spec: GraphNode[SpecFunc], 
    radius: int = 2, 
    shape: int = cv2.MORPH_ELLIPSE
) -> GraphNode[SpecFunc]:
    """Дилатация"""
    return spec.cache.get_or_create_node(
        f"morphDilate({spec.key}, radius={radius}, shape={shape})",
        lambda: _morphDilate(spec.value, radius=radius, shape=shape)
    )


from . import morphOpen as _morphOpen
def morphOpen(
    spec: GraphNode[SpecFunc], 
    radius: int = 2, 
    shape: int = cv2.MORPH_ELLIPSE
) -> GraphNode[SpecFunc]:
    """Открытие"""
    return spec.cache.get_or_create_node(
        f"morphOpen({spec.key}, radius={radius}, shape={shape})",
        lambda: _morphOpen(spec.value, radius=radius, shape=shape)
    )


from . import morphClose as _morphClose
def morphClose(
    spec: GraphNode[SpecFunc], 
    radius: int = 2, 
    shape: int = cv2.MORPH_ELLIPSE
) -> GraphNode[SpecFunc]:
    """Закрытие"""
    return spec.cache.get_or_create_node(
        f"morphClose({spec.key}, radius={radius}, shape={shape})",
        lambda: _morphClose(spec.value, radius=radius, shape=shape)
    )


from . import morphGradient as _morphGradient
def morphGradient(
    spec: GraphNode[SpecFunc], 
    radius: int = 2, 
    shape: int = cv2.MORPH_ELLIPSE
) -> GraphNode[SpecFunc]:
    """Морфологический градиент"""
    return spec.cache.get_or_create_node(
        f"morphGradient({spec.key}, radius={radius}, shape={shape})",
        lambda: _morphGradient(spec.value, radius=radius, shape=shape)
    )


from . import morphSmooth as _morphSmooth
def morphSmooth(
    spec: GraphNode[SpecFunc], 
    open_radius: int = 2, 
    close_radius: int = 2
) -> GraphNode[SpecFunc]:
    """Сглаживание (Opening + Closing)"""
    return spec.cache.get_or_create_node(
        f"morphSmooth({spec.key}, open_radius={open_radius}, close_radius={close_radius})",
        lambda: _morphSmooth(spec.value, open_radius=open_radius, close_radius=close_radius)
    )

from . import morphHorizontalConnect as _morphHorizontalConnect
def morphHorizontalConnect(
    spec: GraphNode[SpecFunc], 
    width: int = 10
) -> GraphNode[SpecFunc]:
    """Соединение разрывов по времени (горизонтально)"""
    return spec.cache.get_or_create_node(
        f"morphHorizontalConnect({spec.key}, width={width})",
        lambda: _morphHorizontalConnect(spec.value, width=width)
    )


from . import morphVerticalConnect as _morphVerticalConnect
def morphVerticalConnect(
    spec: GraphNode[SpecFunc], 
    height: int = 10
) -> GraphNode[SpecFunc]:
    """Соединение разрывов по частоте (вертикально)"""
    return spec.cache.get_or_create_node(
        f"morphVerticalConnect({spec.key}, height={height})",
        lambda: _morphVerticalConnect(spec.value, height=height)
    )


from . import morphRemoveSmall as _morphRemoveSmall
def morphRemoveSmall(
    spec: GraphNode[SpecFunc], 
    min_size: int = 10
) -> GraphNode[SpecFunc]:
    """Удаление мелких объектов"""
    return spec.cache.get_or_create_node(
        f"morphRemoveSmall({spec.key}, min_size={min_size})",
        lambda: _morphRemoveSmall(spec.value, min_size=min_size)
    )


from . import morphSkeleton as _morphSkeleton
def morphSkeleton(
    spec: GraphNode[SpecFunc]
) -> GraphNode[SpecFunc]:
    """Скелетизация"""
    return spec.cache.get_or_create_node(
        f"morphSkeleton({spec.key})",
        lambda: _morphSkeleton(spec.value)
    )