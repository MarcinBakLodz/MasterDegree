import torch
print(torch.version.cuda)       # Powinno zwrócić np. '11.8'
print(torch.backends.cudnn.enabled)