"""Configuration dataclasses for ARC-RL."""

from dataclasses import dataclass, field
from typing import Optional


GRID_SIZE = 30
NUM_COLORS = 10
MAX_EXAMPLES = 3


@dataclass
class ModelConfig:
    hidden_channels: int = 128
    num_blocks: int = 20
    num_colors: int = NUM_COLORS
    grid_size: int = GRID_SIZE
    max_examples: int = MAX_EXAMPLES

    @property
    def in_channels(self) -> int:
        num_grids = self.max_examples * 2 + 2  # example pairs + test_input + current_output
        per_grid = self.num_colors + 1  # one-hot colors + valid mask
        return num_grids * per_grid + 1  # +1 for step counter channel


@dataclass
class TrainConfig:
    data_dir: str = "references/ARC-AGI/data"

    num_rollouts: int = 32
    tasks_per_batch: int = 16
    max_steps: int = 150

    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    entropy_coeff: float = 0.01

    num_iterations: int = 50000
    warmup_steps: int = 1000

    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000

    device: str = "cuda"
    compile_model: bool = True
    bf16: bool = True
    seed: int = 42

    checkpoint_dir: str = "checkpoints"
    resume: Optional[str] = None

    wandb_project: str = "arc-rl"
    wandb_enabled: bool = False

    augment_colors: bool = True
    augment_geometry: bool = True


@dataclass
class EvalConfig:
    data_dir: str = "references/ARC-AGI/data"
    split: str = "evaluation"
    num_rollouts: int = 64
    max_steps: int = 300
    num_attempts: int = 3
    device: str = "cuda"
    checkpoint: str = ""
    bf16: bool = True
