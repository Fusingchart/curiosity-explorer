"""
Smoke test: random agent on MiniGrid-Empty-8x8.
Confirms obs shape, action space, and wrapper correctness.
Run: python scripts/smoke_test.py
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from src.env_wrappers import make_minigrid_env


def main():
    env = make_minigrid_env("MiniGrid-Empty-8x8-v0", obs_size=64, grayscale=False)

    print(f"Obs space:    {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset(seed=42)
    print(f"Reset obs shape: {obs.shape}, dtype: {obs.dtype}, range: [{obs.min()}, {obs.max()}]")
    assert obs.shape == (3, 64, 64), f"Unexpected obs shape: {obs.shape}"

    total_reward = 0.0
    steps = 0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            obs, info = env.reset()

    print(f"Ran {steps} steps, total reward: {total_reward:.2f}")
    print("Smoke test passed.")
    env.close()


if __name__ == "__main__":
    main()
