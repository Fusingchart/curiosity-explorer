import torch
import numpy as np


class RolloutBuffer:
    """
    Fixed-size rollout buffer for on-policy PPO.
    Stores next_obs alongside obs so curiosity modules can compute rewards
    over the full rollout in a single batched forward pass.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: tuple,
        device: torch.device,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.device = device
        self.ptr = 0

        S, E = n_steps, n_envs
        self.obs = torch.zeros(S, E, *obs_shape, dtype=torch.uint8)
        self.next_obs = torch.zeros(S, E, *obs_shape, dtype=torch.uint8)
        self.actions = torch.zeros(S, E, dtype=torch.long)
        self.rewards = torch.zeros(S, E)
        self.values = torch.zeros(S, E)
        self.log_probs = torch.zeros(S, E)
        self.dones = torch.zeros(S, E)
        self.advantages = torch.zeros(S, E)
        self.returns = torch.zeros(S, E)
        self.positions: list[list] = []  # shape: [n_steps][n_envs]

    def store(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: np.ndarray,
        positions: list | None = None,
    ):
        t = self.ptr
        self.obs[t] = obs.cpu()
        self.next_obs[t] = next_obs.cpu()
        self.actions[t] = action.cpu()
        self.rewards[t] = torch.as_tensor(reward)
        self.values[t] = value.cpu()
        self.log_probs[t] = log_prob.cpu()
        self.dones[t] = torch.as_tensor(done, dtype=torch.float32)
        if positions is not None:
            self.positions.append(positions)
        self.ptr += 1

    def add_intrinsic(self, int_rewards: torch.Tensor, int_coef: float):
        """int_rewards: (n_steps, n_envs) on CPU."""
        self.rewards += int_coef * int_rewards

    def compute_advantages(
        self, last_value: torch.Tensor, gamma: float, gae_lambda: float
    ):
        last_value = last_value.cpu()
        gae = torch.zeros(self.n_envs)
        for t in reversed(range(self.n_steps)):
            next_val = last_value if t == self.n_steps - 1 else self.values[t + 1]
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * mask - self.values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Yield shuffled flat minibatches. Each batch is (obs, next_obs, actions,
        old_log_probs, advantages, returns) — all on self.device."""
        total = self.n_steps * self.n_envs
        idx = torch.randperm(total)
        obs_f = self.obs.reshape(total, *self.obs_shape)
        nxt_f = self.next_obs.reshape(total, *self.obs_shape)
        act_f = self.actions.reshape(total)
        lp_f = self.log_probs.reshape(total)
        adv_f = self.advantages.reshape(total)
        ret_f = self.returns.reshape(total)

        for start in range(0, total, batch_size):
            b = idx[start : start + batch_size]
            yield (
                obs_f[b].to(self.device),
                nxt_f[b].to(self.device),
                act_f[b].to(self.device),
                lp_f[b].to(self.device),
                adv_f[b].to(self.device),
                ret_f[b].to(self.device),
            )

    def reset(self):
        self.ptr = 0
        self.positions = []
