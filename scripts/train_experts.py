#!/usr/bin/env python3
"""Phase 1: Train per-task expert models in parallel, save demonstrations.

Usage:
    python scripts/train_experts.py --num-workers 8 --output-dir demos
    python scripts/train_experts.py --num-workers 4 --K 256 --T 50 --max-iters 2000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc_rl.config import ModelConfig
from arc_rl.dataset import ARCDataset


def worker_fn(
    rank: int,
    task_data: list,
    model_cfg_dict: dict,
    args_dict: dict,
    output_dir: str,
    counter_file: str,
):
    """Worker process: train a subset of tasks sequentially."""
    import random
    import numpy as np
    from tqdm import tqdm

    seed = args_dict["seed"] + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device_name = args_dict.get("device", "cuda")
    device = torch.device(device_name if (device_name != "cuda" or torch.cuda.is_available()) else "cpu")
    from arc_rl.config import ModelConfig
    from arc_rl.dataset import ARCTask
    from arc_rl.expert import train_task, Demo

    model_cfg = ModelConfig(**model_cfg_dict)
    out = Path(output_dir)

    solved = 0
    total = len(task_data)

    desc = f"W{rank}" if args_dict.get("num_workers", 1) > 1 else "Tasks"
    pbar = tqdm(
        task_data,
        desc=desc,
        position=rank,
        leave=True,
        dynamic_ncols=True,
    )

    for i, (task_id, train_pairs, test_pairs) in enumerate(pbar):
        task = ARCTask(task_id=task_id, train_pairs=train_pairs, test_pairs=test_pairs)

        # Skip if already done (allows resuming)
        demo_path = out / f"{task_id}.json"
        if demo_path.exists():
            try:
                d = Demo.load(demo_path)
                if d.solved:
                    solved += 1
                pbar.set_postfix(solved=solved, last="skip", ordered=True)
                continue
            except Exception:
                pass

        tqdm.write(f"[W{rank}] ({i+1}/{total}) Training {task_id} ...")
        t_start = __import__("time").time()

        def on_inner_progress(cur_it: int, max_it: int, best_r: float, no_improve: int):
            pbar.set_postfix(
                solved=solved,
                task=task_id[:8],
                inner=f"{cur_it}/{max_it}",
                best=f"{best_r:.2f}",
                stall=no_improve,
                ordered=True,
            )

        demo = train_task(
            task=task,
            model_cfg=model_cfg,
            device=device,
            K=args_dict["K"],
            T=args_dict["T"],
            max_iters=args_dict["max_iters"],
            lr=args_dict["lr"],
            patience=args_dict["patience"],
            entropy_coeff=args_dict["entropy_coeff"],
            num_grad_steps=args_dict["num_grad_steps"],
            progress_cb=on_inner_progress,
            progress_every=args_dict["inner_log_every"],
        )
        demo.save(demo_path)

        if demo.solved:
            solved += 1

        elapsed_task = __import__("time").time() - t_start
        status = "SOLVED" if demo.solved else f"r={demo.best_reward:.2f}"
        tqdm.write(
            f"[W{rank}] ({i+1}/{total}) {task_id}: {status} "
            f"in {demo.iterations}it ({elapsed_task:.0f}s)"
        )
        pbar.set_postfix(
            solved=solved,
            last=f"{task_id[:8]}:{status}",
            ordered=True,
        )


def parse_args():
    p = argparse.ArgumentParser(description="Train per-task expert models")
    p.add_argument("--data-dir", type=str, default="references/ARC-AGI/data")
    p.add_argument("--output-dir", type=str, default="demos")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")

    # Expert model architecture (small and fast)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=4)

    # Per-task training
    p.add_argument("--K", type=int, default=128, help="Rollouts per task")
    p.add_argument("--T", type=int, default=50, help="Max steps per episode")
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--num-grad-steps", type=int, default=16)
    p.add_argument("--inner-log-every", type=int, default=10,
                   help="Update per-task inner progress every N iterations")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tasks from {args.data_dir}/training ...")
    dataset = ARCDataset(args.data_dir, split="training")
    print(f"  {len(dataset)} tasks")

    model_cfg = ModelConfig(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
    )
    num_params = sum(
        p.numel()
        for p in __import__("arc_rl.model", fromlist=["ARCPolicy"]).ARCPolicy(model_cfg).parameters()
    )
    print(f"Expert model: {num_params/1e6:.2f}M params "
          f"({args.hidden_channels}ch, {args.num_blocks}blk)")
    print(f"Per-task: K={args.K}, T={args.T}, max_iters={args.max_iters}, "
          f"patience={args.patience}, lr={args.lr}")
    print(f"Workers: {args.num_workers}")
    print()

    # Serialize tasks as plain Python data (picklable)
    task_data = []
    for task in dataset.tasks:
        task_data.append((task.task_id, task.train_pairs, task.test_pairs))

    # Split tasks round-robin among workers
    chunks = [[] for _ in range(args.num_workers)]
    for i, td in enumerate(task_data):
        chunks[i % args.num_workers].append(td)

    model_cfg_dict = {
        "hidden_channels": model_cfg.hidden_channels,
        "num_blocks": model_cfg.num_blocks,
        "num_colors": model_cfg.num_colors,
        "grid_size": model_cfg.grid_size,
        "max_examples": model_cfg.max_examples,
    }
    args_dict = {
        "K": args.K, "T": args.T, "max_iters": args.max_iters,
        "patience": args.patience, "lr": args.lr,
        "entropy_coeff": args.entropy_coeff,
        "num_grad_steps": args.num_grad_steps, "seed": args.seed,
        "num_workers": args.num_workers,
        "device": args.device,
        "inner_log_every": args.inner_log_every,
    }

    counter_file = str(output_dir / "_progress.log")
    open(counter_file, "w").close()

    t0 = time.time()

    if args.num_workers <= 1:
        worker_fn(0, task_data, model_cfg_dict, args_dict, str(output_dir), counter_file)
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(args.num_workers):
            p = mp.Process(
                target=worker_fn,
                args=(rank, chunks[rank], model_cfg_dict, args_dict,
                      str(output_dir), counter_file),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        failed = [p.exitcode for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"One or more workers failed with exit codes: {failed}")

    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed/60:.1f} minutes")

    # Summary
    solved = 0
    total = 0
    rewards = []
    for f in sorted(output_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        total += 1
        with open(f) as fh:
            d = json.load(fh)
        if d.get("solved"):
            solved += 1
        rewards.append(d.get("best_reward", 0))

    print(f"\nResults: {solved}/{total} tasks solved ({100*solved/max(total,1):.1f}%)")
    if rewards:
        print(f"  Mean best reward: {sum(rewards)/len(rewards):.3f}")
        print(f"  Reward > 0: {sum(1 for r in rewards if r > 0)}/{total}")

    summary = {"solved": solved, "total": total, "elapsed_minutes": elapsed / 60}
    with open(output_dir / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
