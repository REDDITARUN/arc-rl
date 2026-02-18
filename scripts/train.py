#!/usr/bin/env python3
"""Main training script for ARC-RL."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc_rl.config import ModelConfig, TrainConfig, EvalConfig
from arc_rl.dataset import ARCDataset
from arc_rl.model import ARCPolicy
from arc_rl.trainer import GRPOTrainer
from arc_rl.evaluate import run_benchmark


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ARC-RL agent with GRPO")

    # Model
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=20)

    # Training
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--num-rollouts", type=int, default=32)
    p.add_argument("--tasks-per-batch", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--num-iterations", type=int, default=50000)
    p.add_argument("--num-grad-steps", type=int, default=32,
                   help="Subsample this many steps for gradient (0=all steps)")
    p.add_argument("--warmup-steps", type=int, default=1000)

    # Data & augmentation
    p.add_argument("--data-dir", type=str, default="references/ARC-AGI/data")
    p.add_argument("--augment-colors", action="store_true", default=True)
    p.add_argument("--no-augment-colors", dest="augment_colors", action="store_false")
    p.add_argument("--augment-geometry", action="store_true", default=True)
    p.add_argument("--no-augment-geometry", dest="augment_geometry", action="store_false")

    # Infrastructure
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--seed", type=int, default=42)

    # Logging & checkpointing
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", type=str, default="arc-rl")

    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Configs
    model_cfg = ModelConfig(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
    )
    train_cfg = TrainConfig(
        data_dir=args.data_dir,
        num_rollouts=args.num_rollouts,
        tasks_per_batch=args.tasks_per_batch,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        entropy_coeff=args.entropy_coeff,
        num_grad_steps=args.num_grad_steps,
        num_iterations=args.num_iterations,
        warmup_steps=args.warmup_steps,
        device=args.device,
        compile_model=args.compile,
        bf16=args.bf16,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        wandb_project=args.wandb_project,
        wandb_enabled=args.wandb,
        augment_colors=args.augment_colors,
        augment_geometry=args.augment_geometry,
    )

    # Dataset
    print(f"Loading training data from {train_cfg.data_dir}/training ...")
    dataset = ARCDataset(train_cfg.data_dir, split="training")
    print(f"  {len(dataset)} training tasks loaded")

    # Model
    policy = ARCPolicy(model_cfg).to(device)
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Model: {num_params/1e6:.2f}M parameters")
    print(f"  in_channels={model_cfg.in_channels}, hidden={model_cfg.hidden_channels}, "
          f"blocks={model_cfg.num_blocks}")

    if train_cfg.compile_model and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        policy = torch.compile(policy)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < train_cfg.warmup_steps:
            return step / max(train_cfg.warmup_steps, 1)
        progress = (step - train_cfg.warmup_steps) / max(
            train_cfg.num_iterations - train_cfg.warmup_steps, 1
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_iter = 0
    if train_cfg.resume:
        ckpt = torch.load(train_cfg.resume, map_location=device, weights_only=False)
        raw_policy = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
        raw_policy.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_iter = ckpt["iteration"] + 1
        print(f"Resumed from {train_cfg.resume} at iteration {start_iter}")

    # Wandb
    if train_cfg.wandb_enabled:
        import wandb
        wandb.init(
            project=train_cfg.wandb_project,
            config={**vars(model_cfg.__class__(**vars(model_cfg))), **vars(train_cfg)},
        )

    # Trainer
    trainer = GRPOTrainer(
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        dataset=dataset,
        device=device,
    )

    # Checkpoint dir
    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\nStarting training for {train_cfg.num_iterations} iterations")
    print(f"  B={train_cfg.tasks_per_batch}, K={train_cfg.num_rollouts}, "
          f"T={train_cfg.max_steps}, grad_steps={train_cfg.num_grad_steps}")
    print(f"  lr={train_cfg.lr}, entropy_coeff={train_cfg.entropy_coeff}")
    print()

    best_eval_acc = 0.0
    pbar = tqdm(
        range(start_iter, train_cfg.num_iterations),
        initial=start_iter,
        total=train_cfg.num_iterations,
        desc="Training",
        unit="it",
        dynamic_ncols=True,
    )
    for it in pbar:
        metrics = trainer.train_step(it)

        # Update progress bar
        pbar.set_postfix(
            loss=f"{metrics.loss:.3f}",
            reward=f"{metrics.mean_reward:.3f}",
            exact=f"{100*metrics.exact_match_rate:.1f}%",
            gnorm=f"{metrics.grad_norm:.1f}",
            sps=f"{metrics.steps_per_sec:.0f}",
            ordered=True,
        )

        # Detailed logging
        if (it + 1) % train_cfg.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            tqdm.write(
                f"[{it+1:>6d}] "
                f"loss={metrics.loss:.4f}  "
                f"reward={metrics.mean_reward:.3f}  "
                f"exact={100*metrics.exact_match_rate:.1f}%  "
                f"ent={metrics.entropy_loss:.2f}  "
                f"gnorm={metrics.grad_norm:.2f}  "
                f"lr={lr:.2e}  "
                f"steps/s={metrics.steps_per_sec:.0f}"
            )
            if train_cfg.wandb_enabled:
                import wandb
                wandb.log({
                    "train/loss": metrics.loss,
                    "train/policy_loss": metrics.policy_loss,
                    "train/entropy": metrics.entropy_loss,
                    "train/mean_reward": metrics.mean_reward,
                    "train/exact_match_rate": metrics.exact_match_rate,
                    "train/best_reward": metrics.best_reward,
                    "train/grad_norm": metrics.grad_norm,
                    "train/lr": lr,
                    "train/steps_per_sec": metrics.steps_per_sec,
                }, step=it + 1)

        # Evaluation
        if (it + 1) % train_cfg.eval_interval == 0:
            tqdm.write(f"\n--- Evaluation at iteration {it+1} ---")
            eval_cfg = EvalConfig(
                data_dir=train_cfg.data_dir,
                split="training",
                num_rollouts=32,
                max_steps=train_cfg.max_steps,
                device=train_cfg.device,
                bf16=train_cfg.bf16,
            )
            raw_policy = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
            bench = run_benchmark(raw_policy, eval_cfg, model_cfg, device, verbose=True)

            if bench.accuracy > best_eval_acc:
                best_eval_acc = bench.accuracy
                path = ckpt_dir / "best.pt"
                torch.save({
                    "model": raw_policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "iteration": it,
                    "accuracy": bench.accuracy,
                    "model_cfg": vars(model_cfg),
                    "train_cfg": vars(train_cfg),
                }, path)
                tqdm.write(f"  New best! Saved to {path}")

            if train_cfg.wandb_enabled:
                import wandb
                wandb.log({
                    "eval/accuracy": bench.accuracy,
                    "eval/solved": bench.solved,
                    "eval/mean_pixel_acc": bench.mean_pixel_acc,
                }, step=it + 1)

        # Periodic save
        if (it + 1) % train_cfg.save_interval == 0:
            raw_policy = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
            path = ckpt_dir / f"iter_{it+1:06d}.pt"
            torch.save({
                "model": raw_policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "iteration": it,
                "model_cfg": vars(model_cfg),
                "train_cfg": vars(train_cfg),
            }, path)
            tqdm.write(f"  Checkpoint saved: {path}")

    pbar.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
