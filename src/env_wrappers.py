import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MiniGridPixels(gym.ObservationWrapper):
    """
    Converts MiniGrid's symbolic dict obs to channels-first pixel obs.
    Applies RGBImgPartialObsWrapper internally for egocentric RGB view.
    Output shape: (C, obs_size, obs_size) uint8.
    """

    def __init__(self, env: gym.Env, obs_size: int = 64, grayscale: bool = False):
        from minigrid.wrappers import RGBImgPartialObsWrapper

        env = RGBImgPartialObsWrapper(env)
        super().__init__(env)
        self.obs_size = obs_size
        self.grayscale = grayscale
        channels = 1 if grayscale else 3
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(channels, obs_size, obs_size), dtype=np.uint8
        )

    def observation(self, obs) -> np.ndarray:
        import cv2

        frame = obs["image"]  # (H, W, 3) uint8 from RGBImgPartialObsWrapper
        frame = cv2.resize(frame, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[np.newaxis]
        else:
            frame = np.transpose(frame, (2, 0, 1))
        return frame


def make_minigrid_env(env_id: str, obs_size: int = 64, grayscale: bool = False) -> gym.Env:
    import minigrid  # noqa: F401

    env = gym.make(env_id, render_mode="rgb_array")
    return MiniGridPixels(env, obs_size=obs_size, grayscale=grayscale)


def make_env_thunk(env_id: str, seed: int, obs_size: int = 64):
    """Returns a thunk for use with gym.vector.SyncVectorEnv."""

    def _init():
        env = make_minigrid_env(env_id, obs_size=obs_size)
        env.reset(seed=seed)
        return env

    return _init


def get_grid_size(vec_env) -> tuple[int, int]:
    """Returns (width, height) of the underlying MiniGrid env."""
    base = vec_env.envs[0].unwrapped
    return base.width, base.height


def get_agent_positions(vec_env) -> list[tuple[int, int]]:
    """Returns current agent (x, y) position for each sub-env."""
    return [tuple(env.unwrapped.agent_pos) for env in vec_env.envs]
