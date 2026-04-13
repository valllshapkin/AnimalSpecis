import numpy as np
import cv2
import matplotlib.pyplot as plt
from BatSpec.Logic.SpecWorker import _build_gabor_filters

ksize = 31
sigma = 3
lambd = 13
gamma=0.5
filters = _build_gabor_filters(ksize=ksize, sigma=sigma, lambd=lambd, gamma=gamma)

# Визуализация: сетка 2x4
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.ravel()

for idx, kern in enumerate(filters):
    ax = axes[idx]
    # Для красивого отображения используем симметричную цветовую карту,
    # так как ядро содержит как положительные, так и отрицательные значения.
    im = ax.imshow(kern, cmap='RdBu', interpolation='bilinear')
    # ax.set_title(f'θ = {theta:.2f} rad ({np.rad2deg(theta):.0f}°)')
    ax.axis('off')
    # Добавляем цветовую шкалу только для первого графика, чтобы не загромождать
    if idx == 0:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f'Gabor kernels (ksize={ksize}, σ={sigma}, λ={lambd}, γ={gamma})', fontsize=14)
plt.tight_layout()
plt.show()

# Дополнительно: показать 3D-поверхность одного ядра (например, первого)
from mpl_toolkits.mplot3d import Axes3D

kern0 = filters[0]
x = np.arange(ksize)
y = np.arange(ksize)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, kern0, cmap='RdBu', edgecolor='none')
# ax.set_title(f'3D surface of kernel θ = {theta0:.2f} rad')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('amplitude')
plt.show()