import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

# Настройка стиля
plt.figure(figsize=(12, 7))
plt.style.use('seaborn-v0_8-whitegrid')

# Диапазон значений по оси X (от -20 до 20)
x = np.linspace(-20, 20, 2000)

# Список количества множителей
n_factors = [1, 2, 3, 4, 5, 6]
colors = ['gray', 'blue', 'green', 'orange', 'purple', 'red']

def get_product_pdf(x_vals, n):
    """
    Вычисляет точную плотность вероятности произведения n нормальных величин.
    Использует свойство логарифма: произведение -> сумма логарифмов.
    """
    if n == 1:
        # Просто стандартное нормальное распределение
        return norm.pdf(x_vals, loc=0, scale=1)
    
    # Параметры распределения логарифма одной величины |X|
    # Для X ~ N(0, 1): ln|X| имеет мат. ожидание mu_log и дисперсию sigma_log^2
    # Это известные константы (константа Эйлера-Маскерони и т.д.)
    # mu_log = -0.635... (смещение), sigma_log = pi/2 (точнее sqrt(pi/2), зависит от определения)
    # Но точнее для N(0,1):
    # E[ln|X|] = -np.euler_gamma / 2 - np.log(2) / 2  (около -0.635)
    # Var[ln|X|] = np.pi**2 / 8 (около 1.23)
    
    # Сумма n таких величин будет иметь параметры:
    mu_sum = n * (-np.euler_gamma / 2 - np.log(2) / 2)
    sigma_sum = np.sqrt(n * (np.pi**2 / 8))
    
    # Произведение P = X1 * ... * Xn.
    # |P| имеет логнормальное распределение с параметрами mu_sum, sigma_sum.
    # Знак P случаен (положителен с вероятностью 1/2, отрицателен с 1/2).
    
    # Вычисляем плотность для модуля
    pdf_abs = lognorm.pdf(x_vals[x_vals > 0], s=sigma_sum, scale=np.exp(mu_sum))
    
    # Создаем массив для результата (симметричный относительно 0)
    pdf_final = np.zeros_like(x_vals)
    
    # Заполняем положительную часть (с вероятностью 1/2)
    pdf_final[x_vals > 0] = 0.5 * pdf_abs
    # Заполняем отрицательную часть (симметрично)
    pdf_final[x_vals < 0] = 0.5 * lognorm.pdf(-x_vals[x_vals < 0], s=sigma_sum, scale=np.exp(mu_sum))
    
    # В нуле плотность не определена (или бесконечна для четных n), ставим 0 для графика
    return pdf_final

# Строим графики
for n, color in zip(n_factors, colors):
    if n == 1:
        y = norm.pdf(x, 0, 1)
        plt.plot(x, y, color=color, linewidth=2, label=f'N = 1 (Нормальное)')
    else:
        # Вычисляем аналитическую функцию
        y = get_product_pdf(x, n)
        plt.plot(x, y, color=color, linewidth=2, label=f'N = {n} (Произведение)')

plt.title('Точные теоретические распределения произведения N величин', fontsize=14)
plt.xlabel('Значение', fontsize=12)
plt.ylabel('Плотность вероятности f(x)', fontsize=12)
plt.ylim(0, 1.5) # Ограничиваем Y, чтобы пики N=2 не "раздавили" остальные
plt.xlim(-15, 15)
plt.legend()
plt.show()