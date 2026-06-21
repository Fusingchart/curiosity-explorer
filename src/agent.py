import torch
import torch.nn as nn
from torch.distributions import Categorical

from .cnn_encoder import CnnEncoder


class PPOAgent(nn.Module):
    """PPO actor-critic with shared CNN encoder."""

    def __init__(self, obs_shape: tuple, action_dim: int, feature_dim: int = 256):
        super().__init__()
        self.encoder = CnnEncoder(obs_shape, feature_dim)
        self.actor = nn.Linear(feature_dim, action_dim)
        self.critic = nn.Linear(feature_dim, 1)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encoder(obs))

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor = None
    ) -> tuple:
        features = self.encoder(obs)
        dist = Categorical(logits=self.actor(features))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(features)
