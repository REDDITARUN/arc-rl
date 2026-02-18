"""Compiled collection loop: fuses obs → forward → sample → paint into minimal Python."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .config import GRID_SIZE, NUM_COLORS


def _build_obs_and_masks(
    grids: torch.Tensor,
    grid_h: torch.Tensor,
    grid_w: torch.Tensor,
    context_channels: torch.Tensor,
    step: int,
    max_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build observation and grid masks from current state. Pure tensor ops."""
    N = grids.shape[0]
    device = grids.device
    one_hot = F.one_hot(grids, NUM_COLORS).permute(0, 3, 1, 2).float()
    y_idx = torch.arange(GRID_SIZE, device=device).view(1, GRID_SIZE, 1)
    x_idx = torch.arange(GRID_SIZE, device=device).view(1, 1, GRID_SIZE)
    grid_masks = (y_idx < grid_h.view(-1, 1, 1)) & (x_idx < grid_w.view(-1, 1, 1))
    current = torch.cat([one_hot, grid_masks.unsqueeze(1).float()], dim=1)
    step_ch = torch.full((N, 1, GRID_SIZE, GRID_SIZE), step / max(max_steps, 1),
                         device=device, dtype=torch.float32)
    obs = torch.cat([context_channels, current, step_ch], dim=1)
    return obs, grid_masks


def _sample_paint_flat(action_logits: torch.Tensor, spatial_logits: torch.Tensor,
                       grid_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample color and flat position. Avoids Categorical object creation overhead."""
    color = torch.multinomial(F.softmax(action_logits, dim=-1), 1).squeeze(-1)
    spatial = spatial_logits.masked_fill(~grid_masks, -1e9)
    flat = spatial.reshape(spatial.size(0), -1)
    position = torch.multinomial(F.softmax(flat, dim=-1), 1).squeeze(-1)
    return color, position


def _sample_resize_flat(size_h_logits: torch.Tensor,
                        size_w_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample resize dimensions."""
    h = torch.multinomial(F.softmax(size_h_logits, dim=-1), 1).squeeze(-1)
    w = torch.multinomial(F.softmax(size_w_logits, dim=-1), 1).squeeze(-1)
    return h, w


def _paint_grids(grids: torch.Tensor, grid_h: torch.Tensor, grid_w: torch.Tensor,
                 color: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Apply paint action to grids. Returns modified grids."""
    y = position // GRID_SIZE
    x = position % GRID_SIZE
    valid = (y >= 0) & (y < grid_h) & (x >= 0) & (x < grid_w)
    idx = torch.arange(grids.shape[0], device=grids.device)
    y_safe = y.clamp(0, GRID_SIZE - 1)
    x_safe = x.clamp(0, GRID_SIZE - 1)
    grids[idx[valid], y_safe[valid], x_safe[valid]] = color[valid]
    return grids


def collect_rollouts_fast(
    policy: torch.nn.Module,
    context_channels: torch.Tensor,
    target_h: torch.Tensor,
    target_w: torch.Tensor,
    targets: torch.Tensor,
    K: int,
    T: int,
    device: torch.device,
) -> dict:
    """Fast rollout collection with minimal Python overhead.

    Args:
        policy: the ARCPolicy model
        context_channels: [N, C_ctx, 30, 30] pre-encoded task contexts (already repeated for K)
        target_h, target_w: [N] target grid dimensions
        targets: [N, 30, 30] target grids
        K: rollouts per task
        T: max steps per episode
        device: CUDA device

    Returns dict with:
        resize_h, resize_w: [N] (0-indexed)
        paint_colors: [T-1, N]
        paint_positions: [T-1, N]
        rewards: [N]
        stored_grids: [T, N, 30, 30]
        stored_grid_h: [T, N]
        stored_grid_w: [T, N]
    """
    N = context_channels.shape[0]

    grids = torch.zeros(N, GRID_SIZE, GRID_SIZE, dtype=torch.long, device=device)
    grid_h = torch.ones(N, dtype=torch.long, device=device)
    grid_w = torch.ones(N, dtype=torch.long, device=device)

    stored_grids = torch.zeros(T, N, GRID_SIZE, GRID_SIZE, dtype=torch.long, device=device)
    stored_grid_h = torch.zeros(T, N, dtype=torch.long, device=device)
    stored_grid_w = torch.zeros(T, N, dtype=torch.long, device=device)

    paint_colors = torch.zeros(T - 1, N, dtype=torch.long, device=device)
    paint_positions = torch.zeros(T - 1, N, dtype=torch.long, device=device)

    # Step 0: RESIZE
    stored_grids[0] = grids
    stored_grid_h[0] = grid_h
    stored_grid_w[0] = grid_w

    obs, _ = _build_obs_and_masks(grids, grid_h, grid_w, context_channels, 0, T)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = policy(obs)
    resize_h, resize_w = _sample_resize_flat(outputs["size_h_logits"], outputs["size_w_logits"])

    # Apply resize
    grid_h = (resize_h + 1).clamp(1, GRID_SIZE)
    grid_w = (resize_w + 1).clamp(1, GRID_SIZE)
    grids.zero_()

    # Steps 1..T-1: PAINT (tight loop, no object creation)
    for step in range(1, T):
        stored_grids[step] = grids
        stored_grid_h[step] = grid_h
        stored_grid_w[step] = grid_w

        obs, masks = _build_obs_and_masks(grids, grid_h, grid_w, context_channels, step, T)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = policy(obs)
        color, position = _sample_paint_flat(
            outputs["action_logits"], outputs["spatial_logits"], masks
        )
        grids = _paint_grids(grids, grid_h, grid_w, color, position)

        paint_colors[step - 1] = color
        paint_positions[step - 1] = position

    # Vectorized reward computation
    y_idx = torch.arange(GRID_SIZE, device=device).view(1, GRID_SIZE, 1)
    x_idx = torch.arange(GRID_SIZE, device=device).view(1, 1, GRID_SIZE)
    size_match = (grid_h == target_h) & (grid_w == target_w)
    target_mask = (y_idx < target_h.view(-1, 1, 1)) & (x_idx < target_w.view(-1, 1, 1))
    cell_correct = (grids == targets) & target_mask
    num_correct = cell_correct.sum(dim=(1, 2)).float()
    num_total = target_mask.sum(dim=(1, 2)).float().clamp(min=1)
    pixel_acc = num_correct / num_total
    exact = pixel_acc == 1.0
    rewards = torch.where(exact, torch.tensor(2.0, device=device), pixel_acc)
    rewards = torch.where(size_match, rewards, torch.tensor(0.0, device=device))

    return {
        "resize_h": resize_h,
        "resize_w": resize_w,
        "paint_colors": paint_colors,
        "paint_positions": paint_positions,
        "rewards": rewards,
        "stored_grids": stored_grids,
        "stored_grid_h": stored_grid_h,
        "stored_grid_w": stored_grid_w,
    }
