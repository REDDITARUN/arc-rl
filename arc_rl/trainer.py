"""GRPO trainer: rollout collection, advantage computation, policy update."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig, GRID_SIZE
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

    @torch.no_grad()
    def collect_rollouts(
        self, task_instances: List[Tuple[List[Pair], Grid, Grid]]
    ) -> RolloutData:
        """Run K rollouts per task, return stored actions + rewards."""
        B = len(task_instances)
        K = self.cfg.num_rollouts
        N = B * K
        T = self.cfg.max_steps

        env = BatchedARCEnv(
            task_instances, K, T,
            max_examples=self.model_cfg.max_examples,
            device=self.device,
        )

        self.policy.eval()

        obs = env.reset()  # [N, C, 30, 30]

        # --- Step 0: RESIZE ---
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.cfg.bf16):
            outputs = self.policy(obs)
        actions_0 = sample_resize(outputs)
        env.resize(actions_0.resize_h + 1, actions_0.resize_w + 1)  # convert 0-indexed → 1-indexed

        # --- Steps 1..T-1: PAINT ---
        all_colors = []
        all_positions = []
        for step in range(1, T):
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
            paint_colors=torch.stack(all_colors, dim=0),      # [T-1, N]
            paint_positions=torch.stack(all_positions, dim=0), # [T-1, N]
            rewards=rewards,
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

    def update_policy(
        self, task_instances: List[Tuple[List[Pair], Grid, Grid]], rollout: RolloutData
    ) -> Tuple[float, float, float]:
        """Replay trajectories with gradients, compute GRPO loss, update."""
        B, K, T = rollout.B, rollout.K, self.cfg.max_steps
        N = B * K

        advantages = self.compute_advantages(rollout)  # [N]

        env = BatchedARCEnv(
            task_instances, K, T,
            max_examples=self.model_cfg.max_examples,
            device=self.device,
        )
        obs = env.reset()

        self.policy.train()
        self.optimizer.zero_grad()

        total_policy_loss = 0.0
        total_entropy = 0.0

        # --- Step 0: RESIZE (with grad) ---
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.cfg.bf16):
            outputs = self.policy(obs)
            stored = StepActions(resize_h=rollout.resize_h, resize_w=rollout.resize_w)
            log_probs = compute_log_probs_resize(outputs, stored)
            step_loss = -(advantages * log_probs).mean()
            ent = compute_entropy_resize(outputs)

        step_loss_total = step_loss - self.cfg.entropy_coeff * ent
        step_loss_total.backward()

        total_policy_loss += step_loss.item()
        total_entropy += ent.item()

        with torch.no_grad():
            env.resize(rollout.resize_h + 1, rollout.resize_w + 1)

        # --- Steps 1..T-1: PAINT (with grad) ---
        for step_idx in range(T - 1):
            obs = env.get_obs()
            masks = env.get_grid_masks()

            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.cfg.bf16):
                outputs = self.policy(obs)
                stored = StepActions(
                    color=rollout.paint_colors[step_idx],
                    position=rollout.paint_positions[step_idx],
                )
                log_probs = compute_log_probs_paint(outputs, stored, masks)
                step_loss = -(advantages * log_probs).mean()
                ent = compute_entropy_paint(outputs, masks)

            step_loss_total = step_loss - self.cfg.entropy_coeff * ent
            step_loss_total.backward()

            total_policy_loss += step_loss.item()
            total_entropy += ent.item()

            with torch.no_grad():
                y = rollout.paint_positions[step_idx] // GRID_SIZE
                x = rollout.paint_positions[step_idx] % GRID_SIZE
                env.paint(rollout.paint_colors[step_idx], x, y)

        grad_norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.cfg.grad_clip
        ).item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        avg_policy_loss = total_policy_loss / T
        avg_entropy = total_entropy / T
        return avg_policy_loss, avg_entropy, grad_norm

    def train_step(self, iteration: int) -> TrainMetrics:
        """One full GRPO training step: sample → rollout → update."""
        t0 = time.time()

        tasks = self.dataset.sample(self.cfg.tasks_per_batch)
        task_instances = self._prepare_task_instances(tasks)

        rollout = self.collect_rollouts(task_instances)
        policy_loss, entropy, grad_norm = self.update_policy(task_instances, rollout)

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
