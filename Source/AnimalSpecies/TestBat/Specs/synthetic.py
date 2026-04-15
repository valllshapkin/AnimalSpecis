import torch
import torch.optim as optim
import random 

torch.manual_seed(42)
random.seed(42)

T = 500_000
F_bins = 300
K = 2
W_len = 150
E_len = 400
bounds = [(50, 150), (180, 260)]

print("1. Generating synthetic ground truth...")
H_true = torch.zeros(K, T)

# --- Генерация позиций для каждого канала ---
num_positions_0 = 200  # для канала [0, :]
num_positions_1 = 100  # для канала [1, :]

positions_0 = random.sample(range(T), num_positions_0)  # уникальные позиции
positions_1 = random.sample(range(T), num_positions_1)

# --- Случайные амплитуды в диапазоне [0.8, 1.0] ---
values_0 = torch.empty(num_positions_0).uniform_(0.8, 1.0)
values_1 = torch.empty(num_positions_1).uniform_(0.8, 1.0)

H_true[0, positions_0] = values_0
H_true[1, positions_1] = values_1

print(f"Канал 0: {num_positions_0} пиков, позиции (первые 10): {positions_0[:10]}")
print(f"Канал 1: {num_positions_1} пиков, позиции (первые 10): {positions_1[:10]}")
print(f"Канал 0 — мин/макс значений: {H_true[0][positions_0].min():.3f} / {H_true[0][positions_0].max():.3f}")
print(f"Канал 1 — мин/макс значений: {H_true[1][positions_1].min():.3f} / {H_true[1][positions_1].max():.3f}")
print("Готово. H_true.shape =", H_true.shape)