import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd:
    """Welford's online running mean and variance."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4  # avoid div-by-zero on first update

    def update(self, x: np.ndarray):
        b_mean, b_var, b_count = x.mean(0), x.var(0), x.shape[0]
        tot = self.count + b_count
        delta = b_mean - self.mean
        self.mean += delta * b_count / tot
        self.var = (self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / tot) / tot
        self.count = tot

    @property
    def std(self):
        return np.sqrt(np.maximum(self.var, 1e-8))


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 512), nn.LeakyReLU(),
        nn.Linear(512, 512), nn.LeakyReLU(),
        nn.Linear(512, out_dim),
    )


class RND(nn.Module):
    """
    Random Network Distillation — Burda et al., 2018.
    Intrinsic reward = L2 distance between a fixed random target network
    and a trained predictor, both mapping flattened observations to a
    fixed-dim feature space.

    Observations are normalised with a running mean/std before being fed
    to both networks. Intrinsic rewards are also normalised online.
    """

    def __init__(self, obs_shape: tuple, output_dim: int = 512):
        super().__init__()
        flat_dim = int(np.prod(obs_shape))
        self.target = _mlp(flat_dim, output_dim)
        self.predictor = _mlp(flat_dim, output_dim)

        for p in self.target.parameters():
            p.requires_grad = False

        # Initialise predictor weights slightly smaller so early rewards aren't huge
        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))

        self.obs_rms = RunningMeanStd(shape=(flat_dim,))
        self.reward_rms = RunningMeanStd()
        self.flat_dim = flat_dim

    def _normalise_obs(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float().flatten(1) / 255.0
        mean = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=obs.device)
        std = torch.as_tensor(self.obs_rms.std, dtype=torch.float32, device=obs.device)
        return ((x - mean) / std).clamp(-5.0, 5.0)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        obs : (B, C, H, W) uint8
        Returns: (intrinsic_rewards (B,), loss scalar)
        """
        x_np = obs.float().flatten(1).cpu().numpy() / 255.0
        self.obs_rms.update(x_np)

        x = self._normalise_obs(obs)
        target_feat = self.target(x)
        pred_feat = self.predictor(x)

        loss_per = 0.5 * F.mse_loss(pred_feat, target_feat.detach(), reduction="none").sum(-1)
        loss = loss_per.mean()

        rewards = loss_per.detach()
        self.reward_rms.update(rewards.cpu().numpy())
        std = torch.as_tensor(self.reward_rms.std, dtype=torch.float32, device=obs.device)
        rewards = rewards / (std + 1e-8)

        return rewards, loss
