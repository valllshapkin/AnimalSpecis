import torch
import random
import numpy as np
from typing import Union
from pathlib import Path

from BatSpec.Work.Function import TimeFunc, SpecFunc
from BatSpec.Work.Sequence import Sequenogram
from BatSpec.Work.Units import UREG
from BatSpec.Work.Sequence import convolveSequenogramWithPattern
from BatSpec.Work.Spectral import addSpecToSpec
from BatSpec.Work.Call import centrateCorpMass
from AnimalSpecies.TestBat.Etalons import MAIN_CALL, SECOND_CALL
from BatSpec.QtApp.Visualize import update_function, update_spec2d, run_visualizer

# ==========================================
# 1. ГЕНЕРАЦИЯ БАЗОВЫХ ДАННЫХ
# ==========================================
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

T = 100_000
F_bins = 300

print("1. Generating synthetic ground truth...")
H_true = torch.zeros(2, T)

num_positions_0 = 200
num_positions_1 = 100

positions_0 = random.sample(range(T), num_positions_0)
positions_1 = random.sample(range(T), num_positions_1)

values_0 = torch.empty(num_positions_0).uniform_(0.8, 1.0)
values_1 = torch.empty(num_positions_1).uniform_(0.8, 1.0)

H_true[0, positions_0] = values_0
H_true[1, positions_1] = values_1

# ==========================================
# 2. НАСТРОЙКА ПАРАМЕТРОВ СЕТКИ И SEQUENOGRAM
# ==========================================
time_max = 12.0  # сек
freq_max = 120000.0  # Гц

dt = time_max / T         
df = freq_max / F_bins    

global_time_axis = np.arange(T, dtype=np.float32) * dt
global_freq_axis = np.arange(F_bins, dtype=np.float32) * df

seq_main = Sequenogram(values=H_true[0].numpy(), axis=global_time_axis)
seq_second = Sequenogram(values=H_true[1].numpy(), axis=global_time_axis)

# ==========================================
# 3. РЕНДЕРИНГ И ЦЕНТРИРОВАНИЕ ПАТЧЕЙ
# ==========================================
print("2. Рендеринг эталонных патчей...")
patch_main = MAIN_CALL.toSpecFunc(dt=dt, df=df, sigma_t=0.0002, sigma_f=1100.0, padding_s=0.002, padding_hz=10_000)
patch_second = SECOND_CALL.toSpecFunc(dt=dt, df=df, sigma_t=0.0002, sigma_f=1100.0, padding_s=0.002, padding_hz=10_000)

patch_main = centrateCorpMass(patch_main)
patch_second = centrateCorpMass(patch_second)

main_part = convolveSequenogramWithPattern(seq_main, patch_main)
second_part = convolveSequenogramWithPattern(seq_second, patch_second)

# ==========================================
# 4. СБОРКА ЧИСТОГО СИГНАЛА
# ==========================================
print("3. Сборка чистой спектрограммы...")
# Создаем ЧИСТУЮ (нулевую) матрицу
clean_matrix = np.zeros((T, F_bins), dtype=np.float32)
SPSL_clean = SpecFunc(
    matrix=clean_matrix, 
    time=global_time_axis, 
    freq=global_freq_axis, 
    unit=UREG.pascal
)

addSpecToSpec(SPSL_clean, main_part)
addSpecToSpec(SPSL_clean, second_part)

# ==========================================
# 5. ПРИМЕНЕНИЕ ЭХА (РАЗМЫТИЕ ЧИСТОГО СИГНАЛА)
# ==========================================
print("4. Размытие эхом...")
from BatSpec.Work.Echo import TEST_ECHO_MODEL, convolveSpecWithEcho

echo = TEST_ECHO_MODEL.get(SPSL_clean.freq[1], SPSL_clean.dt[1], duration=0.3)
RESULT = convolveSpecWithEcho(SPSL_clean, echo)

# ==========================================
# 6. НАЛОЖЕНИЕ ШУМА В САМОМ КОНЦЕ
# ==========================================
print("5. Добавление фонового шума...")
noise_matrix = np.abs(np.random.normal(loc=0.02*40, scale=0.015*40, size=(T, F_bins))).astype(np.float32)

# Достаем матрицу из результата свёртки и плюсуем к ней шум
_, result_matrix = RESULT.values
result_matrix += noise_matrix

# ==========================================
# 7. ВИЗУАЛИЗАЦИЯ
# ==========================================
print("6. Запуск визуализатора...")

update_spec2d("main_part", main_part)
update_spec2d("second_part", second_part)
update_spec2d("echo", echo)
update_spec2d("RESULT_clean_and_noisy", RESULT)

update_function("Sequenogram_Main", seq_main)
update_function("Sequenogram_Second", seq_second)
update_spec2d("Patch_Main", patch_main)
update_spec2d("Patch_Second", patch_second)

run_visualizer()