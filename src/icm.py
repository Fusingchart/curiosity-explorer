import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_encoder import CnnEncoder


class ICM(nn.Module):
    """
    Intrinsic Curiosity Module — Pathak et al., 2017.
    Intrinsic reward = forward model prediction error in latent space.

    The encoder is separate from the policy encoder so policy gradients don't
    interfere with the curiosity feature space.

    Args:
        beta:  weight of forward loss vs. inverse loss (0 = inverse only, 1 = forward only)
    """

    def __init__(
        self,
        obs_shape: tuple,
        action_dim: int,
        feature_dim: int = 256,
        beta: float = 0.2,
    ):
        super().__init__()
        self.beta = beta
        self.action_dim = action_dim

        self.encoder = CnnEncoder(obs_shape, feature_dim)

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        obs, next_obs : (B, C, H, W) uint8
        actions       : (B,) long
        Returns       : (intrinsic_rewards (B,), loss scalar)
        """
        phi = self.encoder(obs)
        phi_next = self.encoder(next_obs)

        # Inverse loss — predict action from transition
        inv_logits = self.inverse_model(torch.cat([phi, phi_next], dim=-1))
        inv_loss = F.cross_entropy(inv_logits, actions)

        # Forward loss — predict next latent from current latent + action
        a_onehot = F.one_hot(actions, self.action_dim).float()
        phi_next_hat = self.forward_model(torch.cat([phi.detach(), a_onehot], dim=-1))
        fwd_loss_per = 0.5 * F.mse_loss(phi_next_hat, phi_next.detach(), reduction="none").sum(-1)
        fwd_loss = fwd_loss_per.mean()

        loss = (1 - self.beta) * inv_loss + self.beta * fwd_loss
        intrinsic_rewards = fwd_loss_per.detach()

        return intrinsic_rewards, loss
