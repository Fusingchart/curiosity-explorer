"""
Visualisation tools: curiosity heatmap, reward decomposition plot, GIF export.
All render functions return (H, W, 3) uint8 numpy arrays suitable for W&B / imageio.
"""

import os
from collections import defaultdict

import imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class HeatmapTracker:
    """
    Accumulates intrinsic rewards keyed by agent grid position and renders
    a per-cell mean-reward heatmap.

    Args:
        grid_w, grid_h : MiniGrid environment dimensions
    """

    def __init__(self, grid_w: int, grid_h: int):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self._rewards: dict = defaultdict(list)

    def update(self, positions: list[tuple[int, int]], rewards: np.ndarray):
        """
        positions : list of (x, y) tuples, one per env
        rewards   : (n_envs,) float array of intrinsic rewards
        """
        for pos, r in zip(positions, rewards):
            self._rewards[pos].append(float(r))

    def render(self, title: str = "") -> np.ndarray:
        grid = np.zeros((self.grid_h, self.grid_w))
        for (x, y), rs in self._rewards.items():
            if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                grid[y, x] = np.mean(rs)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(grid, cmap="hot", interpolation="nearest", vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean intrinsic reward")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return frame

    def reset(self):
        self._rewards.clear()

    @property
    def n_cells_visited(self) -> int:
        return len(self._rewards)


def make_reward_plot(
    steps: list[int],
    ext_rewards: list[float],
    int_rewards: list[float],
    label_ext: str = "Extrinsic",
    label_int: str = "Intrinsic (scaled)",
) -> np.ndarray:
    """Reward decomposition curve. Returns (H, W, 3) uint8."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(steps, ext_rewards, label=label_ext, color="steelblue", linewidth=1.5)
    ax.plot(steps, int_rewards, label=label_int, color="tomato", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Mean reward")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame


def save_gif(frames: list[np.ndarray], path: str, fps: int = 8):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)
