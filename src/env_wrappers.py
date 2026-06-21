import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MiniGridPixels(gym.Wrapper):
    """
    Wraps a MiniGrid env to return RGB pixel observations instead of the
    default dict/symbolic obs. Resizes to (obs_size x obs_size) and
    optionally converts to grayscale.

    Resulting obs shape: (C, obs_size, obs_size) — channels-first for PyTorch.
    """

    def __init__(self, env: gym.Env, obs_size: int = 64, grayscale: bool = False):
        super().__init__(env)
        self.obs_size = obs_size
        self.grayscale = grayscale
        self.channels = 1 if grayscale else 3

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.channels, obs_size, obs_size),
            dtype=np.uint8,
        )

    def _process(self, obs) -> np.ndarray:
        import cv2

        # MiniGrid's render() gives the full env view; get_frame() gives agent view
        frame = self.env.get_frame(highlight=False, tile_size=8)

        frame = cv2.resize(frame, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA)

        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = frame[np.newaxis]  # (1, H, W)
        else:
            frame = np.transpose(frame, (2, 0, 1))  # (3, H, W)

        return frame

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process(obs), reward, terminated, truncated, info


def make_minigrid_env(env_id: str, obs_size: int = 64, grayscale: bool = False, seed: int = 0):
    """Factory for a wrapped, seeded MiniGrid pixel env."""
    import minigrid  # noqa: F401 — registers envs

    env = gym.make(env_id, render_mode="rgb_array")
    env = MiniGridPixels(env, obs_size=obs_size, grayscale=grayscale)
    env.reset(seed=seed)
    return env
