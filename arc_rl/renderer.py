"""Grid rendering and visualization utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .dataset import Grid

# Standard ARC colour palette (RGB)
ARC_COLORS: List[Tuple[int, int, int]] = [
    (0, 0, 0),        # 0  black
    (0, 116, 217),     # 1  blue
    (255, 65, 54),     # 2  red
    (46, 204, 64),     # 3  green
    (255, 220, 0),     # 4  yellow
    (170, 170, 170),   # 5  grey
    (240, 18, 190),    # 6  magenta
    (255, 133, 27),    # 7  orange
    (127, 219, 255),   # 8  cyan
    (135, 12, 37),     # 9  maroon
]

BORDER_COLOR = (64, 64, 64)
CELL_PX = 20
BORDER_PX = 1


def grid_to_image(grid: Grid, cell_px: int = CELL_PX, border_px: int = BORDER_PX) -> np.ndarray:
    """Render a grid as an RGB numpy array [H_px, W_px, 3]."""
    h, w = len(grid), len(grid[0])
    img_h = h * cell_px + (h + 1) * border_px
    img_w = w * cell_px + (w + 1) * border_px
    img = np.full((img_h, img_w, 3), BORDER_COLOR, dtype=np.uint8)

    for r in range(h):
        for c in range(w):
            y0 = border_px + r * (cell_px + border_px)
            x0 = border_px + c * (cell_px + border_px)
            color = ARC_COLORS[grid[r][c] % len(ARC_COLORS)]
            img[y0 : y0 + cell_px, x0 : x0 + cell_px] = color
    return img


def render_pair(inp: Grid, out: Grid, cell_px: int = CELL_PX) -> np.ndarray:
    """Render an input→output pair side by side with an arrow gap."""
    img_in = grid_to_image(inp, cell_px)
    img_out = grid_to_image(out, cell_px)
    gap = 10
    max_h = max(img_in.shape[0], img_out.shape[0])
    combined = np.full((max_h, img_in.shape[1] + gap + img_out.shape[1], 3), 30, dtype=np.uint8)
    combined[: img_in.shape[0], : img_in.shape[1]] = img_in
    combined[: img_out.shape[0], img_in.shape[1] + gap :] = img_out
    return combined


def render_task(
    examples: List[Tuple[Grid, Grid]],
    test_input: Grid,
    prediction: Optional[Grid] = None,
    target: Optional[Grid] = None,
    cell_px: int = CELL_PX,
) -> np.ndarray:
    """Render full task: examples, test input, and optional prediction/target."""
    rows: List[np.ndarray] = []
    for inp, out in examples:
        rows.append(render_pair(inp, out, cell_px))

    # Test row
    test_img = grid_to_image(test_input, cell_px)
    test_row_parts = [test_img]
    gap = 10
    if prediction is not None:
        pred_img = grid_to_image(prediction, cell_px)
        test_row_parts.append(np.full((pred_img.shape[0], gap, 3), 30, dtype=np.uint8))
        test_row_parts.append(pred_img)
    if target is not None:
        tgt_img = grid_to_image(target, cell_px)
        test_row_parts.append(np.full((tgt_img.shape[0], gap, 3), 30, dtype=np.uint8))
        test_row_parts.append(tgt_img)

    max_h = max(p.shape[0] for p in test_row_parts)
    padded = []
    for p in test_row_parts:
        if p.shape[0] < max_h:
            pad = np.full((max_h - p.shape[0], p.shape[1], 3), 30, dtype=np.uint8)
            p = np.concatenate([p, pad], axis=0)
        padded.append(p)
    test_row = np.concatenate(padded, axis=1)
    rows.append(test_row)

    # Stack rows vertically
    max_w = max(r.shape[1] for r in rows)
    final_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 30, dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        final_rows.append(r)
        final_rows.append(np.full((6, max_w, 3), 30, dtype=np.uint8))  # separator

    return np.concatenate(final_rows, axis=0)
