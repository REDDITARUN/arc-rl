"""GRPO trainer: rollout collection, advantage computation, policy update."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig, GRID_SIZE, NUM_COLORS
from .dataset import ARCDataset, ARCTask, augment_colors, augment_geometry, Pair, Grid
from .env import BatchedARCEnv
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
class RolloutData:
    """Stored trajectory data from a batch of rollouts."""
    resize_h: torch.Tensor         # [N]
    resize_w: torch.Tensor         # [N]
    paint_colors: torch.Tensor     # [T-1, N]
    paint_positions: torch.Tensor  # [T-1, N]
    rewards: torch.Tensor          # [N]
    stored_grids: torch.Tensor     # [T, N, 30, 30]
    stored_grid_h: torch.Tensor    # [T, N]
    stored_grid_w: torch.Tensor    # [T, N]
    context_channels: torch.Tensor # [N, C_ctx, 30, 30]
    B: int = 0
    K: int = 0


@dataclass
class TrainMetrics:
    loss: float = 0.0
    policy_loss: float = 0.0
    entropy_loss: float = 0.0
    mean_reward: float = 0.0
    exact_match_rate: float = 0.0
    best_reward: float = 0.0
    grad_norm: float = 0.0
    steps_per_sec: float = 0.0


class GRPOTrainer:
    """Group Relative Policy Optimization for ARC."""

    def __init__(
        self,
        policy: ARCPolicy,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        train_cfg: TrainConfig,
        model_cfg: ModelConfig,
        dataset: ARCDataset,
        device: torch.device,
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = train_cfg
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.device = device
        self.amp_dtype = torch.bfloat16 if train_cfg.bf16 else torch.float32

        # Pre-allocate coordinate indices for obs reconstruction
        self._y_idx = torch.arange(GRID_SIZE, device=device).view(1, GRID_SIZE, 1)
        self._x_idx = torch.arange(GRID_SIZE, device=device).view(1, 1, GRID_SIZE)

    def _prepare_task_instances(
        self, tasks: List[ARCTask]
    ) -> List[Tuple[List[Pair], Grid, Grid]]:
        """Create leave-one-out training instances, with optional augmentation."""
        instances = []
        for task in tasks:
            examples, test_input, target_output = task.get_training_instance(
                self.model_cfg.max_examples
            )
            if self.cfg.augment_colors:
                examples, test_input, target_output = augment_colors(
                    examples, test_input, target_output
                )
            if self.cfg.augment_geometry:
                examples, test_input, target_output = augment_geometry(
                    examples, test_input, target_output
                )
            instances.append((examples, test_input, target_output))
        return instances

    def _reconstruct_obs(
        self, rollout: RolloutData, step_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct observation and grid masks from stored state.

        Returns (obs [N, C, 30, 30], grid_masks [N, 30, 30]).
        """
        grids = rollout.stored_grids[step_idx]       # [N, 30, 30]
        grid_h = rollout.stored_grid_h[step_idx]     # [N]
        grid_w = rollout.stored_grid_w[step_idx]     # [N]

        one_hot = F.one_hot(grids, NUM_COLORS).permute(0, 3, 1, 2).float()
        grid_masks = (self._y_idx < grid_h.view(-1, 1, 1)) & (
            self._x_idx < grid_w.view(-1, 1, 1)
        )
        masks_f = grid_masks.unsqueeze(1).float()
        current = torch.cat([one_hot, masks_f], dim=1)  # [N, 11, 30, 30]

        step_val = step_idx / max(self.cfg.max_steps, 1)
        N = grids.shape[0]
        step_ch = torch.full(
            (N, 1, GRID_SIZE, GRID_SIZE), step_val, device=self.device
        )

        obs = torch.cat([rollout.context_channels, current, step_ch], dim=1)
        return obs, grid_masks

    @torch.no_grad()
    def collect_rollouts(
        self, task_instances: List[Tuple[List[Pair], Grid, Grid]]
    ) -> RolloutData:
        """Run K rollouts per task, store grid states for gradient replay."""
        B = len(task_instances)
        K = self.cfg.num_rollouts
        N = B * K
        T = self.cfg.max_steps

        env = BatchedARCEnv(
            task_instances, K, T,
            max_examples=self.model_cfg.max_examples,
            device=self.device,
        )

        # Pre-allocate state storage
        stored_grids = torch.zeros(T, N, GRID_SIZE, GRID_SIZE, dtype=torch.long, device=self.device)
        stored_grid_h = torch.zeros(T, N, dtype=torch.long, device=self.device)
        stored_grid_w = torch.zeros(T, N, dtype=torch.long, device=self.device)

        self.policy.eval()
        obs = env.reset()

        # Store initial state (step 0)
        stored_grids[0] = env.grids
        stored_grid_h[0] = env.grid_h
        stored_grid_w[0] = env.grid_w

        # --- Step 0: RESIZE ---
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.cfg.bf16):
            outputs = self.policy(obs)
        actions_0 = sample_resize(outputs)
        env.resize(actions_0.resize_h + 1, actions_0.resize_w + 1)

        # --- Steps 1..T-1: PAINT ---
        all_colors = []
        all_positions = []
        for step in range(1, T):
            stored_grids[step] = env.grids
            stored_grid_h[step] = env.grid_h
            stored_grid_w[step] = env.grid_w

            obs = env.get_obs()
            masks = env.get_grid_masks()
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.cfg.bf16):
                outputs = self.policy(obs)
            actions_t = sample_paint(outputs, masks)
            y = actions_t.position // GRID_SIZE
            x = actions_t.position % GRID_SIZE
            env.paint(actions_t.color, x, y)
            all_colors.append(actions_t.color)
            all_positions.append(actions_t.position)

        rewards = env.compute_rewards()
        self.policy.train()

        return RolloutData(
            resize_h=actions_0.resize_h,
            resize_w=actions_0.resize_w,
            paint_colors=torch.stack(all_colors, dim=0),
            paint_positions=torch.stack(all_positions, dim=0),
            rewards=rewards,
            stored_grids=stored_grids,
            stored_grid_h=stored_grid_h,
            stored_grid_w=stored_grid_w,
            context_channels=env.context_channels,
            B=B,
            K=K,
        )

    def compute_advantages(self, rollout: RolloutData) -> torch.Tensor:
        """GRPO: normalize rewards within each task group. Returns [N]."""
        rewards = rollout.rewards.view(rollout.B, rollout.K)  # [B, K]
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = ((rewards - mean) / std).reshape(-1)  # [N]
        return advantages

    def update_policy(self, rollout: RolloutData) -> Tuple[float, float, float]:
        """Replay from stored states with gradients, compute GRPO loss, update.

        Uses subsampled steps if cfg.num_grad_steps < max_steps.
        """
        T = self.cfg.max_steps
        advantages = self.compute_advantages(rollout)

        self.policy.train()
        self.optimizer.zero_grad()

        # Determine which steps to compute gradients for
        num_grad_steps = self.cfg.num_grad_steps
        if num_grad_steps <= 0 or num_grad_steps >= T:
            step_indices = list(range(T))
        else:
            paint_steps = sorted(random.sample(range(1, T), min(num_grad_steps - 1, T - 1)))
            step_indices = [0] + paint_steps

        # Scale factor so total gradient magnitude approximates full-trajectory gradient
        scale = T / len(step_indices)

        total_policy_loss = 0.0
        total_entropy = 0.0

        for step_idx in step_indices:
            obs, grid_masks = self._reconstruct_obs(rollout, step_idx)

            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.cfg.bf16):
                outputs = self.policy(obs)

                if step_idx == 0:
                    stored = StepActions(resize_h=rollout.resize_h, resize_w=rollout.resize_w)
                    log_probs = compute_log_probs_resize(outputs, stored)
                    ent = compute_entropy_resize(outputs)
                else:
                    paint_idx = step_idx - 1
                    stored = StepActions(
                        color=rollout.paint_colors[paint_idx],
                        position=rollout.paint_positions[paint_idx],
                    )
                    log_probs = compute_log_probs_paint(outputs, stored, grid_masks)
                    ent = compute_entropy_paint(outputs, grid_masks)

                step_loss = -(advantages * log_probs).mean()

            step_total = (step_loss - self.cfg.entropy_coeff * ent) * scale
            step_total.backward()

            total_policy_loss += step_loss.item()
            total_entropy += ent.item()

        grad_norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.cfg.grad_clip
        ).item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        n_steps = len(step_indices)
        avg_policy_loss = total_policy_loss / n_steps
        avg_entropy = total_entropy / n_steps
        return avg_policy_loss, avg_entropy, grad_norm

    def train_step(self, iteration: int) -> TrainMetrics:
        """One full GRPO training step: sample → rollout → update."""
        t0 = time.time()

        tasks = self.dataset.sample(self.cfg.tasks_per_batch)
        task_instances = self._prepare_task_instances(tasks)

        rollout = self.collect_rollouts(task_instances)
        policy_loss, entropy, grad_norm = self.update_policy(rollout)

        rewards = rollout.rewards
        exact = (rewards >= 2.0).float()

        elapsed = time.time() - t0
        N = rollout.B * rollout.K
        steps_total = N * self.cfg.max_steps
        steps_per_sec = steps_total / max(elapsed, 1e-6)

        return TrainMetrics(
            loss=policy_loss - self.cfg.entropy_coeff * entropy,
            policy_loss=policy_loss,
            entropy_loss=entropy,
            mean_reward=rewards.mean().item(),
            exact_match_rate=exact.mean().item(),
            best_reward=rewards.max().item(),
            grad_norm=grad_norm,
            steps_per_sec=steps_per_sec,
        )
