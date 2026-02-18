"""Per-task expert training: train a small model per ARC task, save demonstrations."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, GRID_SIZE, NUM_COLORS
from .dataset import ARCTask, augment_colors, augment_geometry, Grid, Pair
from .env import BatchedARCEnv, reconstruct_obs
from .model import (
    ARCPolicy,
    StepActions,
    sample_resize,
    sample_paint,
    compute_log_probs_resize,
    compute_log_probs_paint,
    compute_entropy_resize,
    compute_entropy_paint,
)


@dataclass
class Demo:
    """A solved trajectory for one task."""
    task_id: str
    solved: bool
    best_reward: float
    iterations: int
    resize_h: int
    resize_w: int
    paint_colors: List[int]
    paint_positions: List[int]

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls, path: str | Path) -> Demo:
        with open(path) as f:
            return cls(**json.load(f))


def train_task(
    task: ARCTask,
    model_cfg: ModelConfig,
    device: torch.device,
    K: int = 128,
    T: int = 50,
    max_iters: int = 2000,
    lr: float = 1e-3,
    patience: int = 200,
    entropy_coeff: float = 0.01,
    grad_clip: float = 1.0,
    num_grad_steps: int = 16,
) -> Demo:
    """Train a small expert model on a single task.

    Returns a Demo with the best trajectory found.
    """
    policy = ARCPolicy(model_cfg).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    best_reward = 0.0
    best_demo: Optional[Dict] = None
    no_improve = 0

    for it in range(max_iters):
        examples, test_input, target_output = task.get_training_instance(
            model_cfg.max_examples
        )
        examples, test_input, target_output = augment_colors(
            examples, test_input, target_output
        )
        examples, test_input, target_output = augment_geometry(
            examples, test_input, target_output
        )
        instances = [(examples, test_input, target_output)]

        # ---- Collect rollouts ----
        env = BatchedARCEnv(
            instances, K, T, max_examples=model_cfg.max_examples, device=device
        )
        N = env.N  # K (B=1)

        stored_grids = torch.zeros(T, N, GRID_SIZE, GRID_SIZE, dtype=torch.long, device=device)
        stored_grid_h = torch.zeros(T, N, dtype=torch.long, device=device)
        stored_grid_w = torch.zeros(T, N, dtype=torch.long, device=device)

        policy.eval()
        with torch.no_grad():
            obs = env.reset()

            stored_grids[0] = env.grids
            stored_grid_h[0] = env.grid_h
            stored_grid_w[0] = env.grid_w

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = policy(obs)
            actions_0 = sample_resize(outputs)
            env.resize(actions_0.resize_h + 1, actions_0.resize_w + 1)

            all_colors = []
            all_positions = []
            for step in range(1, T):
                stored_grids[step] = env.grids
                stored_grid_h[step] = env.grid_h
                stored_grid_w[step] = env.grid_w

                obs = env.get_obs()
                masks = env.get_grid_masks()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = policy(obs)
                actions_t = sample_paint(outputs, masks)
                y = actions_t.position // GRID_SIZE
                x = actions_t.position % GRID_SIZE
                env.paint(actions_t.color, x, y)
                all_colors.append(actions_t.color)
                all_positions.append(actions_t.position)

            rewards = env.compute_rewards()
            paint_colors = torch.stack(all_colors, dim=0)    # [T-1, N]
            paint_positions = torch.stack(all_positions, dim=0)  # [T-1, N]

        # ---- Track best trajectory ----
        max_r = rewards.max().item()
        if max_r > best_reward:
            best_reward = max_r
            no_improve = 0
            best_idx = rewards.argmax().item()
            best_demo = {
                "resize_h": (actions_0.resize_h[best_idx] + 1).item(),
                "resize_w": (actions_0.resize_w[best_idx] + 1).item(),
                "paint_colors": paint_colors[:, best_idx].cpu().tolist(),
                "paint_positions": paint_positions[:, best_idx].cpu().tolist(),
            }
            if best_reward >= 2.0:
                break
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        # ---- GRPO update ----
        advantages = rewards.clone()
        mean = advantages.mean()
        std = advantages.std().clamp(min=1e-8)
        advantages = (advantages - mean) / std

        policy.train()
        optimizer.zero_grad()

        if num_grad_steps >= T:
            step_indices = list(range(T))
        else:
            step_indices = [0] + sorted(random.sample(range(1, T), min(num_grad_steps - 1, T - 1)))

        scale = T / len(step_indices)

        for step_idx in step_indices:
            obs_r, grid_masks = reconstruct_obs(
                stored_grids[step_idx], stored_grid_h[step_idx], stored_grid_w[step_idx],
                env.context_channels, step_idx, T, device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = policy(obs_r)
                if step_idx == 0:
                    stored = StepActions(resize_h=actions_0.resize_h, resize_w=actions_0.resize_w)
                    lp = compute_log_probs_resize(out, stored)
                    ent = compute_entropy_resize(out)
                else:
                    pidx = step_idx - 1
                    stored = StepActions(color=paint_colors[pidx], position=paint_positions[pidx])
                    lp = compute_log_probs_paint(out, stored, grid_masks)
                    ent = compute_entropy_paint(out, grid_masks)
                loss = -(advantages * lp).mean()

            total = (loss - entropy_coeff * ent) * scale
            total.backward()

        nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()

    if best_demo is None:
        best_demo = {"resize_h": 1, "resize_w": 1, "paint_colors": [0] * (T - 1), "paint_positions": [0] * (T - 1)}

    return Demo(
        task_id=task.task_id,
        solved=best_reward >= 2.0,
        best_reward=best_reward,
        iterations=it + 1,
        resize_h=best_demo["resize_h"],
        resize_w=best_demo["resize_w"],
        paint_colors=best_demo["paint_colors"],
        paint_positions=best_demo["paint_positions"],
    )
