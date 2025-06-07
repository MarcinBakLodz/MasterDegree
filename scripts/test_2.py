import torch

# Dane wejściowe
x = torch.tensor([
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
], dtype=torch.float32)  # kształt [3, 3, 3]

y = torch.tensor([1, 2, 3], dtype=torch.float32)  # kształt [3]

# Broadcasting y -> [1, 3, 1], wynik -> [3, 3, 3]
result = x - y.view(1, 3, 1)

print("Wynik:")
print(result)