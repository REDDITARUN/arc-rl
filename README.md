# ARC-RL: Vision + Reinforcement Learning for ARC-AGI

A novel approach to the Abstraction and Reasoning Corpus (ARC-AGI) benchmark that treats each task as a sequential decision-making problem. An RL agent observes the task examples, then paints the output grid one cell at a time — trained with Group Relative Policy Optimization (GRPO).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  INPUT: Channel-stacked grids  [B, 89, 30, 30]         │
│    • 3 example pairs (one-hot + mask)   = 66 ch         │
│    • test input (one-hot + mask)        = 11 ch         │
│    • current output (one-hot + mask)    = 11 ch         │
│    • step counter                       =  1 ch         │
└──────────────────────┬──────────────────────────────────┘
                       │
                 ResNet Backbone
              (20 blocks, 128 ch)
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼            ▼
    Action Head   Spatial Head   Size Head   Value Head
     [B, 10]      [B, 30, 30]   [B, 30]×2    [B, 1]
    which color   which cell    grid dims    state value
```

## Action Space

| Action | Parameters | When |
|--------|-----------|------|
| RESIZE(h, w) | height, width (1-30) | Step 0 only |
| COLOR_c(x, y) | color (0-9), position | Steps 1+ |

11 total actions. No macros, no DSL. The model learns everything from RL.

## Training: GRPO

```
For each batch of B=16 tasks:
    For each task: run K=32 parallel rollouts
    Each rollout: RESIZE + paint for T steps
    Reward: pixel accuracy + exact match bonus
    Advantages: normalized within each task's K rollouts
    Update: policy gradient with entropy bonus
```

## Project Structure

```
arc_rl/
├── arc_rl/               # Core package
│   ├── config.py         # Configuration dataclasses
│   ├── dataset.py        # ARC task loading + augmentation
│   ├── env.py            # Vectorized ARC environment
│   ├── model.py          # ResNet policy network (AlphaZero-style)
│   ├── trainer.py        # GRPO trainer + rollout logic
│   ├── expert.py         # Per-task expert RL training + demo schema
│   ├── fast_collect.py   # Lower-overhead rollout collection path
│   ├── renderer.py       # Grid visualization
│   └── evaluate.py       # Evaluation + benchmarking
├── scripts/
│   ├── train.py          # Main training entry point
│   ├── train_experts.py  # Phase 1: train per-task expert demos
│   ├── train_bc.py       # Phase 2: behavior-clone from expert demos
│   └── evaluate.py       # Evaluation entry point
├── notebooks/
│   └── explore.ipynb     # Interactive exploration notebook
├── references/           # Reference implementations
│   ├── ARC-AGI/          # Official ARC-AGI repo + data
│   ├── VARC/             # Vision ARC (image-to-image)
│   ├── NVARC/            # NVIDIA ARC solution
│   └── TinyRecursiveModels/  # Recursive reasoning (7M params)
├── checkpoints/          # Saved model checkpoints
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# Full training (A100 recommended)
python scripts/train.py \
    --hidden-channels 128 \
    --num-blocks 20 \
    --num-rollouts 32 \
    --tasks-per-batch 16 \
    --max-steps 150 \
    --lr 3e-4 \
    --num-iterations 50000 \
    --bf16 \
    --compile \
    --wandb

# Quick test run (smaller model, fewer rollouts)
python scripts/train.py \
    --hidden-channels 64 \
    --num-blocks 8 \
    --num-rollouts 8 \
    --tasks-per-batch 4 \
    --max-steps 50 \
    --num-iterations 100 \
    --no-compile \
    --device cpu
```

### 3-Phase Pipeline (recommended)

```bash
# Phase 1: train per-task experts and save demos
python scripts/train_experts.py \
    --num-workers 8 \
    --hidden-channels 64 --num-blocks 4 \
    --K 512 --T 30 \
    --max-iters 300 --patience 75 \
    --lr 1e-3 --output-dir demos

# Phase 2: behavior-clone a shared model from solved demos
python scripts/train_bc.py \
    --demos-dir demos \
    --hidden-channels 128 --num-blocks 10 \
    --epochs 100 --batch-tasks 32 --steps-per-demo 16 \
    --lr 3e-4 --bf16

# Phase 3: RL fine-tune from the BC checkpoint
python scripts/train.py \
    --resume checkpoints/bc_model.pt \
    --hidden-channels 128 --num-blocks 10 \
    --num-rollouts 64 --tasks-per-batch 16 \
    --max-steps 50 --num-grad-steps 16 \
    --lr 1e-4 --num-iterations 5000 \
    --bf16 --compile
```

`scripts/train.py --resume` supports both full training checkpoints and model-only checkpoints (e.g. BC output).

## Evaluation

```bash
# Benchmark on evaluation split
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --split evaluation \
    --num-rollouts 64 \
    --max-steps 300 \
    --save-predictions predictions.json

# Benchmark on training split (sanity check)
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --split training \
    --num-rollouts 32
```

## Key Design Decisions

**Why ResNet (not ViT/UNet)?**
- AlphaZero proved ResNets are optimal for board-game RL
- CNN spatial bias is perfect for 30×30 grids
- 5-10× faster than ViT per forward pass — critical for RL rollouts
- ~6M params — small, fast, doesn't overfit

**Why GRPO (not PPO)?**
- No separate critic needed (advantages from group statistics)
- Embarrassingly parallel (K rollouts per task)
- Proven for reasoning tasks (DeepSeek-R1)
- Works well with sparse reward

**Why channel stacking (not image tiling)?**
- CNN directly correlates same-position features across grids
- No tokenization overhead
- Fixed input size regardless of grid dimensions
- The "task description" IS the channel data

## Compute Requirements

| Phase | GPU | Time | Notes |
|-------|-----|------|-------|
| Training (full) | 1× A100 80GB | ~2-3 days | BF16, compiled |
| Training (test) | CPU / any GPU | ~5 min | Small config |
| Evaluation | 1× A100 | ~30 min | 400 tasks × 64 rollouts |

## Reference Scores (ARC-AGI-1)

| Method | Params | Score |
|--------|--------|-------|
| VARC-ViT | 18M | 52-56% |
| TRM | 7M | 45% |
| NVARC (ensemble) | mixed | ~55%+ |
| **ARC-RL (ours)** | **~6M** | **TBD** |
