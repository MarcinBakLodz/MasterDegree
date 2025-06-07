import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDTW(nn.Module):
    def __init__(self, gamma=1.0, normalize=False):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x, y):
        """
        x, y: [B, 1, T]
        """
        print("x:",x.shape)
        B, T = x.size()
        D = (x - y).pow(2).squeeze(1)  # [B, T]

        dtw = torch.zeros((B, T + 1, T + 1), device=x.device) + float('inf')
        dtw[:, 0, 0] = 0

        for i in range(1, T + 1):
            for j in range(1, T + 1):
                cost = D[:, i - 1] + self._min_soft(
                    dtw[:, i - 1, j],
                    dtw[:, i, j - 1],
                    dtw[:, i - 1, j - 1]
                )
                dtw[:, i, j] = cost

        result = dtw[:, -1, -1]
        if self.normalize:
            result = result / T
        return result.mean()

    def _min_soft(self, a, b, c):
        stacked = torch.stack([a, b, c], dim=0)
        return -self.gamma * torch.logsumexp(-stacked / self.gamma, dim=0)