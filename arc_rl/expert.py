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
from .env import BatchedARCEnv, reconstruct_obs, _grid_to_tensor, encode_context
from .fast_collect import collect_rollouts_fast
from .model import (
    ARCPolicy,
    StepActions,
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


def _encode_instances(
    instances: List[Tuple[List[Pair], Grid, Grid]],
    K: int,
    max_examples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode task instances into tensors for fast collection.

    Returns: (context_channels [N, C, 30, 30], target_h [N], target_w [N], targets [N, 30, 30])
    """
    B = len(instances)
    contexts = []
    target_h = torch.zeros(B, dtype=torch.long, device=device)
    target_w = torch.zeros(B, dtype=torch.long, device=device)
    targets = torch.zeros(B, GRID_SIZE, GRID_SIZE, dtype=torch.long, device=device)

    for b, (examples, test_input, target_output) in enumerate(instances):
        contexts.append(encode_context(examples, test_input, max_examples, device))
        th, tw = len(target_output), len(target_output[0])
        target_h[b] = th
        target_w[b] = tw
        targets[b] = _grid_to_tensor(target_output, device)

    ctx = torch.stack(contexts, dim=0).repeat_interleave(K, dim=0)
    return (
        ctx,
        target_h.repeat_interleave(K),
        target_w.repeat_interleave(K),
        targets.repeat_interleave(K, dim=0),
    )


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

        # Encode once, reuse for collection
        ctx, t_h, t_w, tgts = _encode_instances(instances, K, model_cfg.max_examples, device)

        # ---- Collect rollouts (fast path) ----
        policy.eval()
        with torch.no_grad():
            result = collect_rollouts_fast(policy, ctx, t_h, t_w, tgts, K, T, device)

        rewards = result["rewards"]
        resize_h = result["resize_h"]
        resize_w = result["resize_w"]
        paint_colors = result["paint_colors"]
        paint_positions = result["paint_positions"]

        # ---- Track best trajectory ----
        max_r = rewards.max().item()
        if max_r > best_reward:
            best_reward = max_r
            no_improve = 0
            best_idx = rewards.argmax().item()
            best_demo = {
                "resize_h": (resize_h[best_idx] + 1).item(),
                "resize_w": (resize_w[best_idx] + 1).item(),
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
                result["stored_grids"][step_idx],
                result["stored_grid_h"][step_idx],
                result["stored_grid_w"][step_idx],
                ctx, step_idx, T, device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = policy(obs_r)
                if step_idx == 0:
                    stored = StepActions(resize_h=resize_h, resize_w=resize_w)
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
        best_demo = {
            "resize_h": 1, "resize_w": 1,
            "paint_colors": [0] * (T - 1), "paint_positions": [0] * (T - 1),
        }

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
