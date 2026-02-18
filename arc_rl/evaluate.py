"""Evaluation and ARC-AGI benchmarking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .config import EvalConfig, ModelConfig, GRID_SIZE
from .dataset import ARCDataset, ARCTask, Grid, Pair
from .env import BatchedARCEnv
from .model import ARCPolicy, sample_resize, sample_paint


@dataclass
class TaskResult:
    task_id: str
    test_idx: int
    correct: bool
    best_pixel_acc: float
    predictions: List[Grid]


@dataclass
class BenchmarkResult:
    total_tasks: int
    solved: int
    accuracy: float
    mean_pixel_acc: float
    task_results: List[TaskResult]


@torch.no_grad()
def evaluate_task(
    policy: ARCPolicy,
    task: ARCTask,
    eval_cfg: EvalConfig,
    model_cfg: ModelConfig,
    device: torch.device,
    test_idx: int = 0,
) -> TaskResult:
    """Evaluate a single ARC task: run K rollouts, pick the best."""
    policy.eval()
    amp_dtype = torch.bfloat16 if eval_cfg.bf16 else torch.float32

    examples, test_input, target_output = task.get_eval_instance(
        test_idx=test_idx, max_examples=model_cfg.max_examples
    )
    K = eval_cfg.num_rollouts
    T = eval_cfg.max_steps

    env = BatchedARCEnv(
        [(examples, test_input, target_output)],
        K, T,
        max_examples=model_cfg.max_examples,
        device=device,
    )

    obs = env.reset()

    # Step 0: RESIZE
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=eval_cfg.bf16):
        outputs = policy(obs)
    actions = sample_resize(outputs)
    env.resize(actions.resize_h + 1, actions.resize_w + 1)

    # Steps 1..T-1: PAINT
    for step in range(1, T):
        obs = env.get_obs()
        masks = env.get_grid_masks()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=eval_cfg.bf16):
            outputs = policy(obs)
        actions = sample_paint(outputs, masks)
        y = actions.position // GRID_SIZE
        x = actions.position % GRID_SIZE
        env.paint(actions.color, x, y)

    rewards = env.compute_rewards()
    predicted_grids = env.get_predicted_grids()

    best_idx = rewards.argmax().item()
    best_reward = rewards[best_idx].item()
    correct = best_reward >= 2.0

    # Collect top-N unique predictions for multi-attempt submission
    unique_preds: List[Grid] = []
    seen = set()
    sorted_indices = rewards.argsort(descending=True)
    for idx in sorted_indices:
        grid = predicted_grids[idx.item()]
        key = str(grid)
        if key not in seen:
            seen.add(key)
            unique_preds.append(grid)
        if len(unique_preds) >= eval_cfg.num_attempts:
            break

    return TaskResult(
        task_id=task.task_id,
        test_idx=test_idx,
        correct=correct,
        best_pixel_acc=min(best_reward, 1.0),
        predictions=unique_preds,
    )


def run_benchmark(
    policy: ARCPolicy,
    eval_cfg: EvalConfig,
    model_cfg: ModelConfig,
    device: torch.device,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run full ARC-AGI benchmark on a split."""
    dataset = ARCDataset(eval_cfg.data_dir, eval_cfg.split)
    task_results: List[TaskResult] = []
    solved = 0

    for i, task in enumerate(dataset.tasks):
        all_test_correct = True
        for test_idx in range(len(task.test_pairs)):
            result = evaluate_task(policy, task, eval_cfg, model_cfg, device, test_idx)
            task_results.append(result)

            if not result.correct:
                all_test_correct = False

        if all_test_correct:
            solved += 1

        if verbose and (i + 1) % 20 == 0:
            print(
                f"  [{i+1}/{len(dataset)}] solved={solved} "
                f"({100*solved/(i+1):.1f}%)"
            )

    total = len(dataset)
    pixel_accs = [r.best_pixel_acc for r in task_results]
    mean_pa = sum(pixel_accs) / max(len(pixel_accs), 1)

    result = BenchmarkResult(
        total_tasks=total,
        solved=solved,
        accuracy=solved / max(total, 1),
        mean_pixel_acc=mean_pa,
        task_results=task_results,
    )

    if verbose:
        print(f"\n{'='*50}")
        print(f"Benchmark: {eval_cfg.split}")
        print(f"  Tasks:       {total}")
        print(f"  Solved:      {solved}")
        print(f"  Accuracy:    {100*result.accuracy:.1f}%")
        print(f"  Mean PixAcc: {100*mean_pa:.1f}%")
        print(f"{'='*50}")

    return result


def save_predictions(
    benchmark: BenchmarkResult,
    output_path: str | Path,
) -> None:
    """Save predictions in ARC submission format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission: Dict[str, Dict] = {}
    for result in benchmark.task_results:
        if result.task_id not in submission:
            submission[result.task_id] = {}
        key = f"attempt_{result.test_idx + 1}"

        attempts = {}
        for attempt_idx, pred in enumerate(result.predictions):
            attempts[f"attempt_{attempt_idx + 1}"] = pred
        submission[result.task_id][key] = attempts

    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)
