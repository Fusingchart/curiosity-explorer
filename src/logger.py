"""
W&B logging scaffold. Call init_run() once at the start of training,
then log() anywhere to push metrics. log_image() and log_gif() handle
visualization artifacts.
"""
import os
import numpy as np


_run = None


def init_run(project: str = "curiosity-explorer", name: str = None, config: dict = None):
    import wandb

    global _run
    _run = wandb.init(project=project, name=name, config=config or {})
    return _run


def log(metrics: dict, step: int = None):
    import wandb

    if _run is None:
        return
    wandb.log(metrics, step=step)


def log_image(key: str, image: np.ndarray, step: int = None, caption: str = ""):
    import wandb

    if _run is None:
        return
    wandb.log({key: wandb.Image(image, caption=caption)}, step=step)


def log_gif(key: str, frames: list[np.ndarray], step: int = None, fps: int = 10):
    """
    frames: list of (H, W, C) uint8 RGB arrays.
    """
    import tempfile
    import imageio
    import wandb

    if _run is None:
        return

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        path = f.name

    imageio.mimsave(path, frames, fps=fps, loop=0)
    wandb.log({key: wandb.Video(path, fps=fps, format="gif")}, step=step)
    os.unlink(path)


def finish():
    import wandb

    if _run is not None:
        wandb.finish()
