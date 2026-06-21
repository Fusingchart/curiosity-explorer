"""
Smoke test: confirms the full stack works end-to-end before a real training run.
Run: python scripts/smoke_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gymnasium as gym
import numpy as np

from src.env_wrappers import make_env_thunk, get_grid_size, get_agent_positions
from src.agent import PPOAgent
from src.icm import ICM
from src.rnd import RND
from src.buffer import RolloutBuffer
from src.visualizer import HeatmapTracker


def test_env():
    print("--- Env wrapper ---")
    envs = gym.vector.SyncVectorEnv([make_env_thunk("MiniGrid-Empty-8x8-v0", seed=0)])
    obs, _ = envs.reset()
    assert obs.shape == (1, 3, 64, 64), obs.shape
    assert obs.dtype == np.uint8
    w, h = get_grid_size(envs)
    pos = get_agent_positions(envs)
    print(f"  obs={obs.shape}  grid={w}x{h}  pos={pos}")
    obs, _, _, _, _ = envs.step(envs.action_space.sample())
    assert obs.shape == (1, 3, 64, 64)
    envs.close()
    print("  PASS")


def test_agent():
    print("--- PPO agent ---")
    obs_shape = (3, 64, 64)
    action_dim = 7
    agent = PPOAgent(obs_shape, action_dim)
    obs = torch.zeros(4, *obs_shape, dtype=torch.uint8)
    a, lp, ent, v = agent.get_action_and_value(obs)
    print(f"  action={a.shape}  log_prob={lp.shape}  value={v.shape}")
    assert a.shape == (4,)
    print("  PASS")


def test_icm():
    print("--- ICM ---")
    obs_shape = (3, 64, 64)
    icm = ICM(obs_shape, action_dim=7)
    obs = torch.zeros(8, *obs_shape, dtype=torch.uint8)
    nxt = torch.zeros(8, *obs_shape, dtype=torch.uint8)
    act = torch.randint(0, 7, (8,))
    r, loss = icm(obs, nxt, act)
    print(f"  rewards={r.shape}  loss={loss.item():.4f}")
    assert r.shape == (8,)
    print("  PASS")


def test_rnd():
    print("--- RND ---")
    obs_shape = (3, 64, 64)
    rnd = RND(obs_shape)
    obs = torch.zeros(8, *obs_shape, dtype=torch.uint8)
    r, loss = rnd(obs)
    print(f"  rewards={r.shape}  loss={loss.item():.4f}")
    assert r.shape == (8,)
    print("  PASS")


def test_buffer():
    print("--- RolloutBuffer ---")
    obs_shape = (3, 64, 64)
    device = torch.device("cpu")
    buf = RolloutBuffer(128, 4, obs_shape, device)
    obs = torch.zeros(4, *obs_shape, dtype=torch.uint8)
    for _ in range(128):
        buf.store(obs, obs, torch.zeros(4, dtype=torch.long),
                  np.zeros(4), torch.zeros(4), torch.zeros(4), np.zeros(4, dtype=bool))
    buf.compute_advantages(torch.zeros(4), gamma=0.99, gae_lambda=0.95)
    batches = list(buf.get_batches(64))
    print(f"  {len(batches)} batches of 64 from 128x4={128*4} transitions")
    assert len(batches) == 8
    print("  PASS")


def test_quick_train():
    print("--- Quick training loop (10 updates) ---")
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/train.py",
         "--total-steps", "2048",
         "--n-envs", "2",
         "--n-steps", "128",
         "--n-epochs", "1",
         "--batch-size", "128",
         "--curiosity", "icm"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Training script failed")
    print("  PASS")


if __name__ == "__main__":
    test_env()
    test_agent()
    test_icm()
    test_rnd()
    test_buffer()
    test_quick_train()
    print("\nAll smoke tests passed.")
