"""ARC dataset loading and augmentation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

Grid = List[List[int]]
Pair = Tuple[Grid, Grid]


@dataclass
class ARCTask:
    task_id: str
    train_pairs: List[Pair]
    test_pairs: List[Pair]

    def get_training_instance(self, max_examples: int = 3) -> Tuple[List[Pair], Grid, Grid]:
        """Leave-one-out: hold out a random train pair as the target."""
        idx = random.randint(0, len(self.train_pairs) - 1)
        examples = [p for i, p in enumerate(self.train_pairs) if i != idx]
        if len(examples) > max_examples:
            examples = random.sample(examples, max_examples)
        target_input, target_output = self.train_pairs[idx]
        return examples, target_input, target_output

    def get_eval_instance(
        self, test_idx: int = 0, max_examples: int = 3
    ) -> Tuple[List[Pair], Grid, Grid]:
        """Use all train pairs as examples, test pair as target."""
        examples = self.train_pairs[:max_examples]
        target_input, target_output = self.test_pairs[test_idx]
        return examples, target_input, target_output


class ARCDataset:
    def __init__(self, data_dir: str | Path, split: str = "training"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tasks = self._load_tasks()

    def _load_tasks(self) -> List[ARCTask]:
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        tasks: List[ARCTask] = []
        for f in sorted(split_dir.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
            train_pairs = [(p["input"], p["output"]) for p in data["train"]]
            test_pairs = [(p["input"], p["output"]) for p in data["test"]]
            tasks.append(ARCTask(f.stem, train_pairs, test_pairs))
        return tasks

    def sample(self, n: int) -> List[ARCTask]:
        return random.sample(self.tasks, min(n, len(self.tasks)))

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> ARCTask:
        return self.tasks[idx]


def augment_colors(
    examples: List[Pair], test_input: Grid, target_output: Grid
) -> Tuple[List[Pair], Grid, Grid]:
    """Apply a random color permutation consistently across all grids."""
    perm = list(range(10))
    random.shuffle(perm)

    def remap(grid: Grid) -> Grid:
        return [[perm[c] for c in row] for row in grid]

    new_examples = [(remap(inp), remap(out)) for inp, out in examples]
    return new_examples, remap(test_input), remap(target_output)


def _rot90(grid: Grid) -> Grid:
    h, w = len(grid), len(grid[0])
    return [[grid[h - 1 - j][i] for j in range(h)] for i in range(w)]


def _flip_h(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def _flip_v(grid: Grid) -> Grid:
    return grid[::-1]


_TRANSFORMS = [
    lambda g: g,                                     # identity
    _rot90,                                          # 90 CW
    lambda g: _rot90(_rot90(g)),                     # 180
    lambda g: _rot90(_rot90(_rot90(g))),             # 270 CW
    _flip_h,                                         # horizontal flip
    _flip_v,                                         # vertical flip
    lambda g: _rot90(_flip_h(g)),                    # diagonal
    lambda g: _rot90(_flip_v(g)),                    # anti-diagonal
]


def augment_geometry(
    examples: List[Pair], test_input: Grid, target_output: Grid
) -> Tuple[List[Pair], Grid, Grid]:
    """Apply a random geometric transform (from D4 group) to all grids."""
    tfm = random.choice(_TRANSFORMS)

    new_examples = [(tfm(inp), tfm(out)) for inp, out in examples]
    return new_examples, tfm(test_input), tfm(target_output)
