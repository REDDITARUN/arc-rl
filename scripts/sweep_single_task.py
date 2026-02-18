#!/usr/bin/env python3
"""Sweep RL configs on a single ARC task to find a strong expert setup.

This is a fast calibration step:
1) Pick one training task
2) Try several RL configs (K/T/lr/patience/etc.)
3) Rank by solve rate -> mean reward -> speed
4) Save best demo + JSON report
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc_rl.config import ModelConfig
from arc_rl.dataset import ARCDataset
from arc_rl.expert import Demo, train_task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-task RL config sweep for ARC experts")
    p.add_argument("--data-dir", type=str, default="references/ARC-AGI/data")
    p.add_argument("--task-id", type=str, default=None, help="Task id (json stem). Default: random")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--trials-per-config", type=int, default=3)
    p.add_argument("--inner-log-every", type=int, default=10,
                   help="Update inner train_task progress every N iterations")
    p.add_argument("--seed", type=int, default=42)

    # Model
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument(
        "--archs",
        type=str,
        default="resnet,unet,vit,trm",
        help="Comma-separated architectures to test (resnet,unet,vit,trm)",
    )

    # Sweep setup
    p.add_argument(
        "--preset",
        type=str,
        default="quick",
        choices=["quick", "balanced", "aggressive"],
        help="Predefined config family to test",
    )
    p.add_argument(
        "--configs-json",
        type=str,
        default=None,
        help="Optional path to a JSON list of config dicts. Overrides --preset.",
    )
    p.add_argument("--output", type=str, default="logs/single_task_sweep.json")
    p.add_argument("--save-best-demo", type=str, default="demos/single_task_best.json")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preset_configs(name: str) -> List[Dict]:
    # All keep model fixed; only RL knobs differ.
    if name == "quick":
        return [
            {"K": 128, "T": 20, "max_iters": 120, "patience": 30, "lr": 1e-3, "num_grad_steps": 8, "entropy_coeff": 0.01},
            {"K": 256, "T": 20, "max_iters": 120, "patience": 30, "lr": 1e-3, "num_grad_steps": 8, "entropy_coeff": 0.01},
            {"K": 256, "T": 30, "max_iters": 150, "patience": 40, "lr": 8e-4, "num_grad_steps": 10, "entropy_coeff": 0.01},
            {"K": 512, "T": 20, "max_iters": 120, "patience": 35, "lr": 8e-4, "num_grad_steps": 8, "entropy_coeff": 0.01},
        ]
    if name == "aggressive":
        return [
            {"K": 512, "T": 30, "max_iters": 250, "patience": 60, "lr": 1e-3, "num_grad_steps": 12, "entropy_coeff": 0.01},
            {"K": 512, "T": 40, "max_iters": 300, "patience": 75, "lr": 8e-4, "num_grad_steps": 16, "entropy_coeff": 0.01},
            {"K": 768, "T": 30, "max_iters": 250, "patience": 60, "lr": 8e-4, "num_grad_steps": 12, "entropy_coeff": 0.01},
            {"K": 1024, "T": 30, "max_iters": 220, "patience": 60, "lr": 7e-4, "num_grad_steps": 12, "entropy_coeff": 0.008},
        ]
    # balanced
    return [
        {"K": 256, "T": 30, "max_iters": 180, "patience": 45, "lr": 1e-3, "num_grad_steps": 10, "entropy_coeff": 0.01},
        {"K": 384, "T": 30, "max_iters": 180, "patience": 45, "lr": 8e-4, "num_grad_steps": 10, "entropy_coeff": 0.01},
        {"K": 512, "T": 30, "max_iters": 220, "patience": 60, "lr": 8e-4, "num_grad_steps": 12, "entropy_coeff": 0.01},
        {"K": 512, "T": 40, "max_iters": 250, "patience": 75, "lr": 7e-4, "num_grad_steps": 16, "entropy_coeff": 0.01},
    ]


def load_configs(args: argparse.Namespace) -> List[Dict]:
    if args.configs_json:
        with open(args.configs_json) as f:
            cfgs = json.load(f)
        if not isinstance(cfgs, list) or not cfgs:
            raise ValueError("--configs-json must be a non-empty JSON list")
        return cfgs
    return preset_configs(args.preset)


def pick_task(dataset: ARCDataset, task_id: str | None):
    if task_id is None:
        return random.choice(dataset.tasks)
    for t in dataset.tasks:
        if t.task_id == task_id:
            return t
    raise ValueError(f"Task id not found in training split: {task_id}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    dataset = ARCDataset(args.data_dir, split="training")
    task = pick_task(dataset, args.task_id)
    print(f"Selected task: {task.task_id}")
    print(f"Train pairs: {len(task.train_pairs)} | Test pairs: {len(task.test_pairs)}")

    cfgs = load_configs(args)
    archs = [a.strip().lower() for a in args.archs.split(",") if a.strip()]
    model_cfg = ModelConfig(hidden_channels=args.hidden_channels, num_blocks=args.num_blocks)
    print(
        f"Sweeping {len(archs)} archs x {len(cfgs)} configs x {args.trials_per_config} trials"
    )
    print()

    all_rows = []
    best_demo: Demo | None = None
    best_run_score = (-1.0, -1.0, -1e9)  # solved, reward, -time

    sweep_grid = [(arch, cfg_idx, cfg) for arch in archs for cfg_idx, cfg in enumerate(cfgs)]
    outer = tqdm(sweep_grid, desc="Sweep", dynamic_ncols=True)
    for arch, cfg_idx, cfg in outer:
        rewards = []
        solved = 0
        times = []
        iters = []

        print(f"[Arch {arch}] [Config {cfg_idx}] {cfg}")
        trial_bar = tqdm(
            range(args.trials_per_config),
            desc=f"{arch}/cfg{cfg_idx}",
            leave=False,
            dynamic_ncols=True,
        )
        for trial in trial_bar:
            trial_seed = args.seed + cfg_idx * 1000 + trial
            set_seed(trial_seed)
            t0 = time.time()

            def on_inner_progress(cur_it: int, max_it: int, best_r: float, no_improve: int):
                trial_bar.set_postfix(
                    inner=f"{cur_it}/{max_it}",
                    best=f"{best_r:.3f}",
                    stall=no_improve,
                    ordered=True,
                )

            demo = train_task(
                task=task,
                model_cfg=model_cfg,
                device=device,
                arch=arch,
                K=int(cfg["K"]),
                T=int(cfg["T"]),
                max_iters=int(cfg["max_iters"]),
                lr=float(cfg["lr"]),
                patience=int(cfg["patience"]),
                entropy_coeff=float(cfg.get("entropy_coeff", 0.01)),
                num_grad_steps=int(cfg.get("num_grad_steps", 16)),
                progress_cb=on_inner_progress,
                progress_every=max(1, args.inner_log_every),
            )
            dt = time.time() - t0

            rewards.append(float(demo.best_reward))
            solved += int(demo.solved)
            times.append(dt)
            iters.append(int(demo.iterations))

            run_score = (1.0 if demo.solved else 0.0, float(demo.best_reward), -dt)
            if run_score > best_run_score:
                best_run_score = run_score
                best_demo = demo

            print(
                f"  trial {trial+1}/{args.trials_per_config}: "
                f"reward={demo.best_reward:.3f} solved={demo.solved} "
                f"iters={demo.iterations} time={dt:.1f}s"
            )
            trial_bar.set_postfix(
                inner=f"{demo.iterations}/{int(cfg['max_iters'])}",
                best=f"{demo.best_reward:.3f}",
                done=f"{trial+1}/{args.trials_per_config}",
                ordered=True,
            )
            outer.set_postfix(
                arch=arch,
                cfg=cfg_idx,
                trial=f"{trial+1}/{args.trials_per_config}",
                reward=f"{demo.best_reward:.3f}",
                solved=int(demo.solved),
                ordered=True,
            )
        trial_bar.close()

        row = {
            "arch": arch,
            "config_index": cfg_idx,
            "config": cfg,
            "solve_rate": solved / args.trials_per_config,
            "mean_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_time_sec": float(np.mean(times)),
            "mean_iters": float(np.mean(iters)),
        }
        all_rows.append(row)
        print(
            "  -> "
            f"solve_rate={100*row['solve_rate']:.1f}% "
            f"mean_reward={row['mean_reward']:.3f} "
            f"max_reward={row['max_reward']:.3f} "
            f"mean_time={row['mean_time_sec']:.1f}s"
        )
        print()

    all_rows.sort(
        key=lambda r: (r["solve_rate"], r["mean_reward"], -r["mean_time_sec"]),
        reverse=True,
    )
    best_cfg = all_rows[0]

    print("==== Ranked configs ====")
    for r in all_rows:
        print(
            f"[{r['arch']}][{r['config_index']}] "
            f"solve={100*r['solve_rate']:.1f}% "
            f"mean_reward={r['mean_reward']:.3f} "
            f"max_reward={r['max_reward']:.3f} "
            f"time={r['mean_time_sec']:.1f}s "
            f"cfg={r['config']}"
        )
    print()
    print(f"Best config index: {best_cfg['config_index']}")
    print(f"Best config: {best_cfg['config']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "task_id": task.task_id,
        "device": str(device),
        "model_cfg": {"hidden_channels": args.hidden_channels, "num_blocks": args.num_blocks},
        "archs": archs,
        "trials_per_config": args.trials_per_config,
        "results": all_rows,
        "best_config_index": best_cfg["config_index"],
        "best_config": best_cfg["config"],
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved sweep report: {out_path}")

    if best_demo is not None:
        demo_path = Path(args.save_best_demo)
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        best_demo.save(demo_path)
        print(f"Saved best run demo: {demo_path}")


if __name__ == "__main__":
    main()

