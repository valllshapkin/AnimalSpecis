import numpy as np

def rpca(D, lambda_val=None, max_iter=100, tol=1e-7):
    """
    Алгоритм RPCA (Robust PCA) через Inexact ALM.
    D - исходная матрица спектрограммы (только амплитуды, без фазы!).
    """
    # Если на вход подана комплексная спектрограмма, берем модуль
    if np.iscomplexobj(D):
        print("Внимание: RPCA работает с амплитудами. Беру np.abs().")
        D_mag = np.abs(D)
    else:
        D_mag = D

    n1, n2 = D_mag.shape
    
    # Лямбда по умолчанию: 1 / sqrt(max(n1, n2))
    if lambda_val is None:
        lambda_val = 1.0 / np.sqrt(max(n1, n2))
        
    # Инициализация матриц
    L = np.zeros((n1, n2)) # Низкоранговая (фон)
    S = np.zeros((n1, n2)) # Разреженная (удары/голос)
    Y = np.zeros((n1, n2)) # Матрица множителей Лагранжа
    
    # Нормы для проверки сходимости
    norm_D = np.linalg.norm(D_mag, 'fro')
    
    # Параметры шага
    mu = 1.25 / np.linalg.norm(D_mag, 2)
    mu_bar = mu * 1e7
    rho = 1.5
    
    def soft_thresholding(x, penalty):
        return np.sign(x) * np.maximum(x - penalty, 0)

    for i in range(max_iter):
        # 1. Обновляем L (SVD)
        # ВАЖНО: full_matrices=False спасает оперативную память для (600000, 300)
        U, s, Vt = np.linalg.svd(D_mag - S + (1/mu) * Y, full_matrices=False)
        s_thresholded = soft_thresholding(s, 1/mu)
        L = U @ np.diag(s_thresholded) @ Vt
        
        # 2. Обновляем S
        S = soft_thresholding(D_mag - L + (1/mu) * Y, lambda_val / mu)
        
        # 3. Обновляем множители и mu
        Z = D_mag - L - S
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)
        
        # Проверка сходимости
        err = np.linalg.norm(Z, 'fro') / norm_D
        print(f"Итерация {i+1}, Ошибка: {err:.6f}")
        
        if err < tol:
            print("Алгоритм сошелся!")
            break
            
    return L, S

# === ПРИМЕР ИСПОЛЬЗОВАНИЯ ===

# 1. Создаем случайную "спектрограмму" ваших размеров для теста
# (В реальности тут будет ваша загруженная спектрограмма)
print("Генерация тестовых данных...")
spectrogram = np.random.rand(600000, 300) 

# 2. Запускаем RPCA
print("Запуск RPCA (это может занять несколько минут)...")
L, S = rpca(spectrogram, max_iter=50)

print("Готово!")
print("Размер L (Фон/Гармоники):", L.shape)
print("Размер S (Удары/Голос):", S.shape)