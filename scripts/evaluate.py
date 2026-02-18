#!/usr/bin/env python3
"""Evaluation and benchmarking script for ARC-RL."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc_rl.config import EvalConfig, ModelConfig
from arc_rl.model import ARCPolicy
from arc_rl.evaluate import run_benchmark, save_predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ARC-RL agent")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    p.add_argument("--data-dir", type=str, default="references/ARC-AGI/data")
    p.add_argument("--split", type=str, default="evaluation", choices=["training", "evaluation"])
    p.add_argument("--num-rollouts", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--num-attempts", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--save-predictions", type=str, default=None,
                    help="Path to save predictions JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model_cfg = ModelConfig(**ckpt["model_cfg"])
    policy = ARCPolicy(model_cfg).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Model: {num_params/1e6:.2f}M parameters")

    eval_cfg = EvalConfig(
        data_dir=args.data_dir,
        split=args.split,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        num_attempts=args.num_attempts,
        device=args.device,
        bf16=args.bf16,
    )

    # Run benchmark
    print(f"\nRunning benchmark on {args.split} split ...")
    result = run_benchmark(policy, eval_cfg, model_cfg, device, verbose=True)

    if args.save_predictions:
        save_predictions(result, args.save_predictions)
        print(f"\nPredictions saved to {args.save_predictions}")


if __name__ == "__main__":
    main()
