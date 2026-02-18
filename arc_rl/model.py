"""ResNet policy network (AlphaZero-style) for ARC-RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .config import ModelConfig, GRID_SIZE


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ARCPolicy(nn.Module):
    """ResNet backbone with four output heads:
    - action_head:  color logits  [B, num_colors]
    - spatial_head: cell logits   [B, 30, 30]
    - size_head:    height/width  [B, 30] each
    - value_head:   state value   [B]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.hidden_channels

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(C) for _ in range(cfg.num_blocks)])

        # Color selection (which of 10 colors to paint)
        self.action_fc = nn.Linear(C, cfg.num_colors)

        # Spatial selection (which cell to paint) — 1×1 conv → heatmap
        self.spatial_conv = nn.Conv2d(C, 1, 1)

        # Output grid size prediction
        self.size_h_fc = nn.Linear(C, cfg.grid_size)
        self.size_w_fc = nn.Linear(C, cfg.grid_size)

        # State value
        self.value_fc = nn.Sequential(
            nn.Linear(C, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.tower(self.stem(x))  # [B, C, 30, 30]
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)  # [B, C]

        return {
            "action_logits": self.action_fc(pooled),                        # [B, 10]
            "spatial_logits": self.spatial_conv(features).squeeze(1),       # [B, 30, 30]
            "size_h_logits": self.size_h_fc(pooled),                       # [B, 30]
            "size_w_logits": self.size_w_fc(pooled),                       # [B, 30]
            "value": self.value_fc(pooled).squeeze(-1),                    # [B]
        }


# ---------------------------------------------------------------------------
# Action sampling & log-prob helpers
# ---------------------------------------------------------------------------

@dataclass
class StepActions:
    """Container for one step's sampled actions across N rollouts."""
    # RESIZE (step 0 only)
    resize_h: torch.Tensor | None = None   # [N] long, 0-indexed (actual size = val + 1)
    resize_w: torch.Tensor | None = None
    # PAINT (step 1+ only)
    color: torch.Tensor | None = None      # [N] long 0..9
    position: torch.Tensor | None = None   # [N] long 0..899 (flat index y*30+x)


def sample_resize(outputs: Dict[str, torch.Tensor]) -> StepActions:
    """Sample grid size from size heads."""
    h_dist = Categorical(logits=outputs["size_h_logits"])
    w_dist = Categorical(logits=outputs["size_w_logits"])
    return StepActions(resize_h=h_dist.sample(), resize_w=w_dist.sample())


def sample_paint(
    outputs: Dict[str, torch.Tensor],
    grid_masks: torch.Tensor,
) -> StepActions:
    """Sample color and position from action & spatial heads."""
    color_dist = Categorical(logits=outputs["action_logits"])
    spatial = outputs["spatial_logits"].masked_fill(~grid_masks, float("-inf"))
    pos_dist = Categorical(logits=spatial.reshape(spatial.size(0), -1))
    return StepActions(color=color_dist.sample(), position=pos_dist.sample())


def compute_log_probs_resize(
    outputs: Dict[str, torch.Tensor],
    actions: StepActions,
) -> torch.Tensor:
    """Log probability of stored resize actions. Returns [N]."""
    log_h = F.log_softmax(outputs["size_h_logits"], dim=-1)
    log_w = F.log_softmax(outputs["size_w_logits"], dim=-1)
    lp_h = log_h.gather(1, actions.resize_h.unsqueeze(1)).squeeze(1)
    lp_w = log_w.gather(1, actions.resize_w.unsqueeze(1)).squeeze(1)
    return lp_h + lp_w


def compute_log_probs_paint(
    outputs: Dict[str, torch.Tensor],
    actions: StepActions,
    grid_masks: torch.Tensor,
) -> torch.Tensor:
    """Log probability of stored paint actions. Returns [N]."""
    # Color log-prob
    log_color = F.log_softmax(outputs["action_logits"], dim=-1)
    lp_color = log_color.gather(1, actions.color.unsqueeze(1)).squeeze(1)

    # Spatial log-prob (masked)
    spatial = outputs["spatial_logits"].masked_fill(~grid_masks, float("-inf"))
    log_spatial = F.log_softmax(spatial.reshape(spatial.size(0), -1), dim=-1)
    lp_pos = log_spatial.gather(1, actions.position.unsqueeze(1)).squeeze(1)

    return lp_color + lp_pos


def compute_entropy_paint(
    outputs: Dict[str, torch.Tensor],
    grid_masks: torch.Tensor,
) -> torch.Tensor:
    """Mean entropy of color + spatial distributions. Returns scalar."""
    color_ent = Categorical(logits=outputs["action_logits"]).entropy().mean()
    spatial = outputs["spatial_logits"].masked_fill(~grid_masks, float("-inf"))
    spatial_ent = Categorical(logits=spatial.reshape(spatial.size(0), -1)).entropy().mean()
    return color_ent + spatial_ent


def compute_entropy_resize(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean entropy of size distributions. Returns scalar."""
    h_ent = Categorical(logits=outputs["size_h_logits"]).entropy().mean()
    w_ent = Categorical(logits=outputs["size_w_logits"]).entropy().mean()
    return h_ent + w_ent
