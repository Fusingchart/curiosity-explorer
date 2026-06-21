"""
Main training script — PPO with optional ICM or RND curiosity.

Usage:
  # Baseline (no curiosity)
  python scripts/train.py --env-id MiniGrid-FourRooms-v0 --curiosity none

  # PPO + ICM
  python scripts/train.py --env-id MiniGrid-FourRooms-v0 --curiosity icm --int-coef 0.1 --use-wandb

  # PPO + RND
  python scripts/train.py --env-id MiniGrid-FourRooms-v0 --curiosity rnd --int-coef 0.3 --use-wandb

  # Quick smoke run (1 update, no W&B)
  python scripts/train.py --total-steps 1024 --n-envs 2 --n-steps 128
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env_wrappers import make_env_thunk, get_grid_size, get_agent_positions
from src.agent import PPOAgent
from src.buffer import RolloutBuffer
from src.icm import ICM
from src.rnd import RND
from src.visualizer import HeatmapTracker, make_reward_plot, save_gif
import src.logger as logger


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Environment
    p.add_argument("--env-id", default="MiniGrid-FourRooms-v0")
    p.add_argument("--obs-size", type=int, default=64)
    # Curiosity
    p.add_argument("--curiosity", default="icm", choices=["none", "icm", "rnd"])
    p.add_argument("--int-coef", type=float, default=0.1, help="Intrinsic reward coefficient")
    p.add_argument("--icm-beta", type=float, default=0.2, help="ICM forward/inverse balance")
    p.add_argument("--rnd-output-dim", type=int, default=512)
    # PPO
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--feature-dim", type=int, default=256)
    # Infra
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--checkpoint-dir", default="runs")
    p.add_argument("--vis-interval", type=int, default=50_000, help="Steps between heatmap snapshots")
    p.add_argument("--save-interval", type=int, default=200_000, help="Steps between model checkpoints")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    run_name = args.run_name or f"{args.env_id}__{args.curiosity}__s{args.seed}__{int(time.time())}"
    ckpt_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run: {run_name}\nDevice: {device}")

    if args.use_wandb:
        logger.init_run("curiosity-explorer", name=run_name, config=vars(args))

    # --- Envs ---
    envs = gym.vector.SyncVectorEnv(
        [make_env_thunk(args.env_id, args.seed + i, args.obs_size) for i in range(args.n_envs)]
    )
    obs_shape: tuple = envs.single_observation_space.shape  # (C, H, W)
    action_dim: int = envs.single_action_space.n
    grid_w, grid_h = get_grid_size(envs)
    print(f"obs={obs_shape}  actions={action_dim}  grid={grid_w}x{grid_h}")

    # --- Models ---
    agent = PPOAgent(obs_shape, action_dim, args.feature_dim).to(device)

    curiosity = None
    if args.curiosity == "icm":
        curiosity = ICM(obs_shape, action_dim, args.feature_dim, beta=args.icm_beta).to(device)
    elif args.curiosity == "rnd":
        curiosity = RND(obs_shape, args.rnd_output_dim).to(device)

    all_params = list(agent.parameters())
    if curiosity is not None:
        all_params += list(curiosity.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr, eps=1e-5)

    # --- Buffer + Visualizer ---
    buffer = RolloutBuffer(args.n_steps, args.n_envs, obs_shape, device)
    heatmap = HeatmapTracker(grid_w, grid_h)
    heatmap_frames: list[np.ndarray] = []

    # Reward tracking
    ext_history, int_history, step_history = [], [], []
    ep_ext = np.zeros(args.n_envs)
    completed_ep_returns: list[float] = []

    obs_np, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs_np, dtype=torch.uint8, device=device)

    n_updates = args.total_steps // (args.n_steps * args.n_envs)
    total_steps = 0
    last_vis_step = 0
    last_save_step = 0

    print(f"Training for {n_updates} updates ({args.total_steps:,} steps)...")

    for update in range(1, n_updates + 1):
        # Linear LR annealing
        frac = 1.0 - (update - 1) / n_updates
        optimizer.param_groups[0]["lr"] = frac * args.lr

        # ------------------------------------------------------------------ #
        #  Rollout collection                                                  #
        # ------------------------------------------------------------------ #
        for step in range(args.n_steps):
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs)

            next_obs_np, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            done = terminated | truncated
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.uint8, device=device)

            positions = get_agent_positions(envs)
            buffer.store(obs, next_obs, action, reward, value.squeeze(-1), log_prob, done, positions)

            obs = next_obs
            total_steps += args.n_envs
            ep_ext += reward

            for i, d in enumerate(done):
                if d:
                    completed_ep_returns.append(ep_ext[i])
                    ep_ext[i] = 0.0

        # ------------------------------------------------------------------ #
        #  Intrinsic rewards (full-rollout batch, no grad)                    #
        # ------------------------------------------------------------------ #
        mean_int = 0.0
        if curiosity is not None:
            with torch.no_grad():
                obs_flat = buffer.obs.reshape(-1, *obs_shape).to(device)
                nxt_flat = buffer.next_obs.reshape(-1, *obs_shape).to(device)
                act_flat = buffer.actions.reshape(-1).to(device)

                if args.curiosity == "icm":
                    int_r, _ = curiosity(obs_flat, nxt_flat, act_flat)
                else:
                    int_r, _ = curiosity(nxt_flat)

            int_r = int_r.reshape(args.n_steps, args.n_envs).cpu()
            buffer.add_intrinsic(int_r, args.int_coef)
            mean_int = int_r.mean().item()

            # Update heatmap with (position, intrinsic_reward) per step
            int_r_np = int_r.numpy()
            for s, positions in enumerate(buffer.positions):
                heatmap.update(positions, int_r_np[s])

        # ------------------------------------------------------------------ #
        #  Advantages + PPO update                                            #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            last_val = agent.get_value(obs).squeeze(-1)
        buffer.compute_advantages(last_val, args.gamma, args.gae_lambda)

        pg_losses, v_losses, ent_losses, cur_losses = [], [], [], []

        for _ in range(args.n_epochs):
            for b_obs, b_nxt, b_act, b_lp, b_adv, b_ret in buffer.get_batches(args.batch_size):
                _, new_lp, entropy, new_val = agent.get_action_and_value(b_obs, b_act)
                ratio = (new_lp - b_lp).exp()

                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -b_adv * ratio,
                    -b_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                v_loss = 0.5 * ((new_val.squeeze() - b_ret) ** 2).mean()
                ent_loss = entropy.mean()

                cur_loss = torch.tensor(0.0, device=device)
                if curiosity is not None:
                    if args.curiosity == "icm":
                        _, cur_loss = curiosity(b_obs, b_nxt, b_act)
                    else:
                        _, cur_loss = curiosity(b_nxt)

                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent_loss + cur_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())
                cur_losses.append(cur_loss.item())

        buffer.reset()

        # ------------------------------------------------------------------ #
        #  Logging                                                            #
        # ------------------------------------------------------------------ #
        mean_ep_ret = np.mean(completed_ep_returns[-50:]) if completed_ep_returns else 0.0
        completed_ep_returns = completed_ep_returns[-200:]  # keep rolling window

        if update % 5 == 0:
            print(
                f"[{total_steps:>9,}] ep_ret={mean_ep_ret:.2f}  "
                f"int={mean_int:.4f}  pg={np.mean(pg_losses):.4f}  "
                f"vf={np.mean(v_losses):.4f}  ent={np.mean(ent_losses):.3f}  "
                f"cells={heatmap.n_cells_visited}"
            )
            if args.use_wandb:
                logger.log(
                    {
                        "charts/ep_return": mean_ep_ret,
                        "charts/cells_visited": heatmap.n_cells_visited,
                        "rewards/mean_intrinsic": mean_int,
                        "losses/policy": np.mean(pg_losses),
                        "losses/value": np.mean(v_losses),
                        "losses/entropy": np.mean(ent_losses),
                        "losses/curiosity": np.mean(cur_losses),
                        "charts/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=total_steps,
                )

        ext_history.append(mean_ep_ret)
        int_history.append(mean_int * args.int_coef)
        step_history.append(total_steps)

        # ------------------------------------------------------------------ #
        #  Visualisation snapshot                                             #
        # ------------------------------------------------------------------ #
        if total_steps - last_vis_step >= args.vis_interval:
            last_vis_step = total_steps
            frame = heatmap.render(
                title=f"{args.curiosity} heatmap | {total_steps:,} steps | {heatmap.n_cells_visited} cells"
            )
            heatmap_frames.append(frame)
            if args.use_wandb:
                logger.log_image("viz/heatmap", frame, step=total_steps,
                                  caption=f"{total_steps:,} steps")

        # ------------------------------------------------------------------ #
        #  Model checkpoint                                                   #
        # ------------------------------------------------------------------ #
        if total_steps - last_save_step >= args.save_interval:
            last_save_step = total_steps
            path = os.path.join(ckpt_dir, f"agent_{total_steps}.pt")
            torch.save({"agent": agent.state_dict(), "args": vars(args), "step": total_steps}, path)
            print(f"  -> checkpoint: {path}")

    # ------------------------------------------------------------------ #
    #  Final artefacts                                                    #
    # ------------------------------------------------------------------ #
    # Final heatmap frame
    frame = heatmap.render(title=f"{args.curiosity} heatmap | FINAL | {total_steps:,} steps")
    heatmap_frames.append(frame)

    gif_path = os.path.join(ckpt_dir, "heatmap_evolution.gif")
    save_gif(heatmap_frames, gif_path, fps=4)
    print(f"Saved heatmap GIF: {gif_path}")

    reward_plot = make_reward_plot(step_history, ext_history, int_history)
    plot_path = os.path.join(ckpt_dir, "reward_curves.png")
    import imageio
    imageio.imwrite(plot_path, reward_plot)
    print(f"Saved reward plot: {plot_path}")

    if args.use_wandb:
        logger.log_gif("viz/heatmap_evolution", heatmap_frames, fps=4)
        logger.log_image("viz/reward_curves", reward_plot)

    torch.save({"agent": agent.state_dict(), "args": vars(args), "step": total_steps},
               os.path.join(ckpt_dir, "agent_final.pt"))

    envs.close()
    logger.finish()
    print("Done.")


if __name__ == "__main__":
    train(parse_args())
