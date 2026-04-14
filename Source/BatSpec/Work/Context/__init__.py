from contextvars import ContextVar
from contextlib import contextmanager
import numpy as np
import torch

from jaxtyping import Array
import numpy as np
from typing import Union

AnyArray = Union[np.ndarray, Array]

# Создаем переменную контекста. Можно задать дефолтный бэкенд (например, np), 
# либо оставить None и кидать ошибку, если бэкенд не выбран.
_active_backend = ContextVar("active_backend", default=np)

def fw():
    """Возвращает текущий активный бэкенд."""
    backend = _active_backend.get()
    if backend is None:
        raise RuntimeError("Бэкенд не установлен! Используйте 'with ArrayBackend(...):'")
    return backend

@contextmanager
def ArrayBackend(backend):
    """Контекстный менеджер для переключения бэкенда."""
    # set() меняет значение в текущем контексте и возвращает токен
    token = _active_backend.set(backend)
    try:
        yield
    finally:
        # reset() возвращает значение, которое было ДО вызова set()
        # Это позволяет делать вложенные with
        _active_backend.reset(token)

# --- Проверка работы ---

if __name__ == "__main__":
    with ArrayBackend(np):
        arr_np = fw().zeros((2, 2))
        print(f"1. Outer context: {type(arr_np)}")

        with ArrayBackend(torch):
            arr_torch = fw().zeros((2, 2))
            print(f"2. Inner context: {type(arr_torch)}")
            
        # Проверяем, что после выхода из внутреннего with вернулся numpy
        arr_np2 = fw().zeros((2, 2))
        print(f"3. Back to outer: {type(arr_np2)}")