import torch
import torch.nn as nn


class CnnEncoder(nn.Module):
    """
    Nature DQN-style CNN. Input: (B, C, H, W) uint8, output: (B, feature_dim).
    Normalises pixels to [0, 1] internally.
    """

    def __init__(self, obs_shape: tuple, feature_dim: int = 256):
        super().__init__()
        C, H, W = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, C, H, W)).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat_dim, feature_dim), nn.ReLU())
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x.float() / 255.0))
