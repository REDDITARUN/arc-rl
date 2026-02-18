"""Vectorized ARC environment for batched RL rollouts."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .config import GRID_SIZE, MAX_EXAMPLES, NUM_COLORS
from .dataset import Grid, Pair


def _grid_to_tensor(grid: Grid, device: torch.device) -> torch.Tensor:
    """Convert a python grid to a padded 30x30 long tensor."""
    h, w = len(grid), len(grid[0])
    t = torch.zeros(GRID_SIZE, GRID_SIZE, dtype=torch.long, device=device)
    src = torch.tensor(grid, dtype=torch.long, device=device)
    t[:h, :w] = src
    return t


def encode_grid(
    grid: torch.Tensor, h: int, w: int, device: torch.device
) -> torch.Tensor:
    """Encode a 30x30 int grid as (NUM_COLORS + 1) channels: one-hot + valid mask."""
    one_hot = F.one_hot(grid.long(), NUM_COLORS).permute(2, 0, 1).float()  # [10, 30, 30]
    mask = torch.zeros(1, GRID_SIZE, GRID_SIZE, device=device)
    mask[0, :h, :w] = 1.0
    return torch.cat([one_hot, mask], dim=0)  # [11, 30, 30]


def encode_context(
    examples: List[Pair],
    test_input: Grid,
    max_examples: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode task context (examples + test input) as channels.

    Returns: [C_context, 30, 30] where C_context = (max_examples*2 + 1) * 11
    """
    channels: List[torch.Tensor] = []
    zeros = torch.zeros(NUM_COLORS + 1, GRID_SIZE, GRID_SIZE, device=device)
    for i in range(max_examples):
        if i < len(examples):
            inp, out = examples[i]
            channels.append(encode_grid(_grid_to_tensor(inp, device), len(inp), len(inp[0]), device))
            channels.append(encode_grid(_grid_to_tensor(out, device), len(out), len(out[0]), device))
        else:
            channels.append(zeros)
            channels.append(zeros)

    channels.append(encode_grid(
        _grid_to_tensor(test_input, device), len(test_input), len(test_input[0]), device
    ))

    return torch.cat(channels, dim=0)  # [(max_ex*2+1)*11, 30, 30]


class BatchedARCEnv:
    """Manages B*K parallel ARC environments for batched GRPO rollouts.

    Terminology:
        B = number of tasks in the batch
        K = number of rollouts per task
        N = B * K = total parallel environments
    """

    def __init__(
        self,
        task_instances: List[Tuple[List[Pair], Grid, Grid]],
        K: int,
        max_steps: int,
        max_examples: int = MAX_EXAMPLES,
        device: torch.device = torch.device("cpu"),
    ):
        self.B = len(task_instances)
        self.K = K
        self.N = self.B * K
        self.max_steps = max_steps
        self.max_examples = max_examples
        self.device = device

        # Pre-encode shared task contexts: [B, C_context, 30, 30]
        contexts = []
        self.target_h = torch.zeros(self.B, dtype=torch.long, device=device)
        self.target_w = torch.zeros(self.B, dtype=torch.long, device=device)
        self.targets = torch.zeros(self.B, GRID_SIZE, GRID_SIZE, dtype=torch.long, device=device)
        for b, (examples, test_input, target_output) in enumerate(task_instances):
            contexts.append(encode_context(examples, test_input, max_examples, device))
            th, tw = len(target_output), len(target_output[0])
            self.target_h[b] = th
            self.target_w[b] = tw
            self.targets[b] = _grid_to_tensor(target_output, device)

        # [B, C_ctx, 30, 30] → expand to [N, C_ctx, 30, 30]
        ctx_stack = torch.stack(contexts, dim=0)  # [B, C, 30, 30]
        self.context_channels = ctx_stack.repeat_interleave(K, dim=0)  # [N, C, 30, 30]

        # Expand targets to N for vectorized reward
        self._target_h_n = self.target_h.repeat_interleave(K)  # [N]
        self._target_w_n = self.target_w.repeat_interleave(K)  # [N]
        self._targets_n = self.targets.repeat_interleave(K, dim=0)  # [N, 30, 30]

        # Per-rollout mutable state
        self.grids = torch.zeros(self.N, GRID_SIZE, GRID_SIZE, dtype=torch.long, device=device)
        self.grid_h = torch.ones(self.N, dtype=torch.long, device=device)
        self.grid_w = torch.ones(self.N, dtype=torch.long, device=device)
        self.step_count = 0

        # Coordinate lookup for masking
        self._y_idx = torch.arange(GRID_SIZE, device=device).view(1, GRID_SIZE, 1)
        self._x_idx = torch.arange(GRID_SIZE, device=device).view(1, 1, GRID_SIZE)

    def reset(self) -> torch.Tensor:
        self.grids.zero_()
        self.grid_h.fill_(1)
        self.grid_w.fill_(1)
        self.step_count = 0
        return self.get_obs()

    def get_grid_masks(self) -> torch.Tensor:
        """Boolean mask [N, 30, 30]: True for cells inside the grid."""
        return (self._y_idx < self.grid_h.view(-1, 1, 1)) & (
            self._x_idx < self.grid_w.view(-1, 1, 1)
        )

    def get_obs(self) -> torch.Tensor:
        """Build full observation tensor [N, in_channels, 30, 30]."""
        one_hot = F.one_hot(self.grids, NUM_COLORS).permute(0, 3, 1, 2).float()
        masks = self.get_grid_masks().unsqueeze(1).float()  # [N, 1, 30, 30]
        current_channels = torch.cat([one_hot, masks], dim=1)

        step_val = self.step_count / max(self.max_steps, 1)
        step_channel = torch.full(
            (self.N, 1, GRID_SIZE, GRID_SIZE), step_val, device=self.device
        )

        return torch.cat([self.context_channels, current_channels, step_channel], dim=1)

    def resize(self, h: torch.Tensor, w: torch.Tensor) -> None:
        """Set grid dimensions. h, w: [N] long tensors (1-indexed sizes)."""
        self.grid_h = h.clamp(1, GRID_SIZE).long()
        self.grid_w = w.clamp(1, GRID_SIZE).long()
        self.grids.zero_()
        self.step_count += 1

    def paint(self, color: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
        """Paint cells. color: [N], x: [N], y: [N] (all long tensors)."""
        valid = (y >= 0) & (y < self.grid_h) & (x >= 0) & (x < self.grid_w)
        idx = torch.arange(self.N, device=self.device)
        y_safe = y.clamp(0, GRID_SIZE - 1)
        x_safe = x.clamp(0, GRID_SIZE - 1)
        self.grids[idx[valid], y_safe[valid], x_safe[valid]] = color[valid]
        self.step_count += 1

    def compute_rewards(self) -> torch.Tensor:
        """Fully vectorized reward computation. Returns [N] float tensor."""
        # Check size match
        size_match = (self.grid_h == self._target_h_n) & (self.grid_w == self._target_w_n)

        # Compute pixel accuracy using full 30x30 grids (masked to valid region)
        target_mask = (self._y_idx < self._target_h_n.view(-1, 1, 1)) & (
            self._x_idx < self._target_w_n.view(-1, 1, 1)
        )  # [N, 30, 30]
        cell_correct = (self.grids == self._targets_n) & target_mask  # [N, 30, 30]
        num_correct = cell_correct.sum(dim=(1, 2)).float()  # [N]
        num_total = target_mask.sum(dim=(1, 2)).float().clamp(min=1)  # [N]
        pixel_acc = num_correct / num_total  # [N]

        exact = (pixel_acc == 1.0)
        rewards = torch.where(exact, torch.tensor(2.0, device=self.device), pixel_acc)
        rewards = torch.where(size_match, rewards, torch.tensor(0.0, device=self.device))
        return rewards

    def get_predicted_grids(self) -> List[List[List[int]]]:
        """Extract predicted grids as python lists (for evaluation)."""
        results = []
        for i in range(self.N):
            h = self.grid_h[i].item()
            w = self.grid_w[i].item()
            grid = self.grids[i, :h, :w].cpu().tolist()
            results.append(grid)
        return results
