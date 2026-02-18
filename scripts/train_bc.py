#!/usr/bin/env python3
"""Phase 2: Behavior-clone a shared model from expert demonstrations.

Replays expert trajectories, collects (obs, action) pairs, and trains
the shared policy with supervised cross-entropy loss.

Usage:
    python scripts/train_bc.py --demos-dir demos --epochs 100
    python scripts/train_bc.py --demos-dir demos --hidden-channels 128 --num-blocks 10 --wandb
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc_rl.config import ModelConfig, GRID_SIZE
from arc_rl.env import BatchedARCEnv
from arc_rl.expert import Demo
from arc_rl.model import ARCPolicy


@torch.no_grad()
def replay_demo(
    demo: Demo,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[dict]]:
    """Replay a demo trajectory to collect (obs, action) pairs.

    Returns:
        observations: list of [1, C, 30, 30] tensors (one per step)
        targets: list of dicts with action labels for each step
    """
    if demo.examples is None or demo.test_input is None or demo.target_output is None:
        raise ValueError(f"Demo for task {demo.task_id} is missing stored context fields.")

    instances = [(demo.examples, demo.test_input, demo.target_output)]
    env = BatchedARCEnv(instances, K=1, max_steps=len(demo.paint_colors) + 1,
                        max_examples=3, device=device)

    observations = []
    targets = []

    obs = env.reset()
    observations.append(obs)

    # Step 0: RESIZE
    rh = demo.resize_h - 1  # convert 1-indexed to 0-indexed
    rw = demo.resize_w - 1
    targets.append({"type": "resize", "resize_h": rh, "resize_w": rw})

    env.resize(
        torch.tensor([demo.resize_h], device=device),
        torch.tensor([demo.resize_w], device=device),
    )

    # Steps 1..T-1: PAINT
    for step_idx in range(len(demo.paint_colors)):
        obs = env.get_obs()
        observations.append(obs)

        color = demo.paint_colors[step_idx]
        position = demo.paint_positions[step_idx]

        targets.append({"type": "paint", "color": color, "position": position})

        y = position // GRID_SIZE
        x = position % GRID_SIZE
        env.paint(
            torch.tensor([color], device=device),
            torch.tensor([x], device=device),
            torch.tensor([y], device=device),
        )

    return observations, targets


def build_bc_batch(
    demos: List[Demo],
    device: torch.device,
    max_steps_sample: int = 16,
) -> Tuple[torch.Tensor, dict]:
    """Build a batch of (obs, target) from multiple demos.

    Randomly samples `max_steps_sample` steps per demo to keep batch size bounded.
    """
    all_obs = []
    all_resize_h = []
    all_resize_w = []
    all_color = []
    all_position = []
    all_is_resize = []

    for demo in demos:
        if not demo.solved:
            continue

        observations, targets = replay_demo(demo, device)

        # Sample a subset of steps
        n_steps = len(observations)
        if n_steps > max_steps_sample:
            indices = [0] + sorted(random.sample(range(1, n_steps), max_steps_sample - 1))
        else:
            indices = list(range(n_steps))

        for idx in indices:
            all_obs.append(observations[idx].squeeze(0))  # [C, 30, 30]
            t = targets[idx]
            is_resize = t["type"] == "resize"
            all_is_resize.append(is_resize)
            all_resize_h.append(t.get("resize_h", 0))
            all_resize_w.append(t.get("resize_w", 0))
            all_color.append(t.get("color", 0))
            all_position.append(t.get("position", 0))

    if not all_obs:
        return None, None

    obs = torch.stack(all_obs)  # [batch, C, 30, 30]
    target = {
        "is_resize": torch.tensor(all_is_resize, device=device),
        "resize_h": torch.tensor(all_resize_h, dtype=torch.long, device=device),
        "resize_w": torch.tensor(all_resize_w, dtype=torch.long, device=device),
        "color": torch.tensor(all_color, dtype=torch.long, device=device),
        "position": torch.tensor(all_position, dtype=torch.long, device=device),
    }
    return obs, target


def bc_loss(outputs: dict, targets: dict) -> torch.Tensor:
    """Cross-entropy loss for behavior cloning."""
    is_resize = targets["is_resize"]
    loss = torch.tensor(0.0, device=targets["color"].device)
    count = 0

    # Resize steps
    resize_mask = is_resize
    if resize_mask.any():
        n = resize_mask.sum()
        loss_h = F.cross_entropy(
            outputs["size_h_logits"][resize_mask],
            targets["resize_h"][resize_mask],
        )
        loss_w = F.cross_entropy(
            outputs["size_w_logits"][resize_mask],
            targets["resize_w"][resize_mask],
        )
        loss = loss + (loss_h + loss_w) * n
        count += n.item()

    # Paint steps
    paint_mask = ~is_resize
    if paint_mask.any():
        n = paint_mask.sum()
        loss_color = F.cross_entropy(
            outputs["action_logits"][paint_mask],
            targets["color"][paint_mask],
        )
        loss_pos = F.cross_entropy(
            outputs["spatial_logits"][paint_mask].reshape(n, -1),
            targets["position"][paint_mask],
        )
        loss = loss + (loss_color + loss_pos) * n
        count += n.item()

    return loss / max(count, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Behavior-clone shared model from expert demos")
    p.add_argument("--demos-dir", type=str, default="demos")
    p.add_argument("--data-dir", type=str, default="references/ARC-AGI/data")
    p.add_argument("--output", type=str, default="checkpoints/bc_model.pt")

    # Shared model architecture
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=10)

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-tasks", type=int, default=32,
                   help="Number of demo tasks per batch")
    p.add_argument("--steps-per-demo", type=int, default=16,
                   help="Max observation-action pairs sampled per demo per batch")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", type=str, default="arc-rl")

    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = args.bf16 and device.type == "cuda"
    print(f"Device: {device}")

    # Load demos
    demos_dir = Path(args.demos_dir)
    demo_files = sorted(f for f in demos_dir.glob("*.json") if not f.name.startswith("_"))
    print(f"Loading {len(demo_files)} demos from {demos_dir} ...")

    demos = {}
    solved_count = 0
    for f in demo_files:
        d = Demo.load(f)
        demos[d.task_id] = d
        if d.solved:
            solved_count += 1
    print(f"  {solved_count} solved demos available for BC training")

    if solved_count == 0:
        print("No solved demos! Run train_experts.py first.")
        return

    # Filter to solved demos with exact stored context
    solved_demos = []
    missing_context = 0
    for task_id, demo in demos.items():
        if demo.solved:
            if demo.examples is None or demo.test_input is None or demo.target_output is None:
                missing_context += 1
                continue
            solved_demos.append(demo)
    print(f"  {len(solved_demos)} solved demos with stored context")
    if missing_context:
        print(f"  Skipped {missing_context} old demos missing context; regenerate with latest train_experts.py")

    if not solved_demos:
        print("No usable solved demos with stored context. Re-run train_experts.py to regenerate demos.")
        return

    # Model
    model_cfg = ModelConfig(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
    )
    policy = ARCPolicy(model_cfg).to(device)
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Shared model: {num_params/1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args), tags=["bc"])

    # Training
    print(f"\nBC training for {args.epochs} epochs")
    print(f"  batch_tasks={args.batch_tasks}, steps_per_demo={args.steps_per_demo}")
    n_demos = len(solved_demos)
    batches_per_epoch = max(n_demos // args.batch_tasks, 1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        policy.train()
        epoch_loss = 0.0
        epoch_batches = 0

        indices = list(range(n_demos))
        random.shuffle(indices)

        pbar = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}",
                    leave=False)
        for batch_idx in pbar:
            start = (batch_idx * args.batch_tasks) % n_demos
            batch_indices = [indices[(start + j) % n_demos] for j in range(args.batch_tasks)]
            batch_demos = [solved_demos[i] for i in batch_indices]

            obs, targets = build_bc_batch(
                batch_demos, device,
                max_steps_sample=args.steps_per_demo,
            )
            if obs is None:
                continue

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                outputs = policy(obs)
                loss = bc_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(epoch_batches, 1)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}  lr={lr:.2e}")

        if args.wandb:
            import wandb
            wandb.log({"bc/loss": avg_loss, "bc/lr": lr}, step=epoch + 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model": policy.state_dict(),
                "model_cfg": {
                    "hidden_channels": model_cfg.hidden_channels,
                    "num_blocks": model_cfg.num_blocks,
                },
                "epoch": epoch + 1,
                "loss": avg_loss,
            }, args.output)
            print(f"  Saved best → {args.output} (loss={avg_loss:.4f})")

    print(f"\nBC training complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {args.output}")
    print(f"\nNext step: RL fine-tune with:")
    print(f"  python scripts/train.py --resume {args.output} "
          f"--hidden-channels {args.hidden_channels} --num-blocks {args.num_blocks} ...")


if __name__ == "__main__":
    main()
