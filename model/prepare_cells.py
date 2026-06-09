# =============================================================================
# prepare_cells.py — Cell Extraction Preprocessing Pipeline
# =============================================================================
# Member 5 runs this ONCE before training to extract individual cell crops
# from the full puzzle images and save them as a ready-to-train dataset.
#
# What this script does:
#   1. Reads metadata.jsonl for train / val / test splits
#   2. Uses the keypoints field to perspective-warp each puzzle image to a
#      flat 450x450 square (50px per cell)
#   3. Crops out all 81 cells per image with a 4-pixel margin on each edge
#      (MUST match the margin used in app/image_processor.py so training
#      and inference inputs see the same content)
#   4. Derives each cell's class label from cells[r][c] in the metadata
#   5. Saves every cropped cell as a 28x28 grayscale PNG into:
#        data/cells/<split>/<class>/  (0 through 9)
#   6. Writes a summary CSV for each split: data/cells/<split>/labels.csv
#
# Output layout (used by train_model.py via image_dataset_from_directory):
#   data/cells/
#   ├── train/
#   │   ├── 0/   ← empty cells
#   │   ├── 1/   ← digit 1 cells
#   │   ...
#   │   └── 9/
#   ├── val/
#   └── test/
#
# Run from the project root:
#   python model/prepare_cells.py
#
# Dependencies: opencv-python, Pillow, numpy, tqdm
# Source (perspective transform approach):
# https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# =============================================================================

import json
import os
import cv2
import numpy as np
from PIL import Image
import csv
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT   = "data"                   # raw dataset root
OUTPUT_ROOT = os.path.join(DATA_ROOT, "cells")   # extracted cells go here
SPLITS      = ["train", "val", "test"]
CELL_SIZE   = 28                       # output size per cell (px) — matches CNN input
WARP_SIZE   = 450                      # intermediate warp canvas (divisible by 9 → 50px/cell)
CELL_PX     = WARP_SIZE // 9          # 50px per cell before final resize

# Margin cropped from each edge of a raw cell crop.
# This trims grid lines that bleed into the cell boundary.
# IMPORTANT: this value MUST match the margin in app/image_processor.py
# so that training images and inference inputs look the same to the model.
MARGIN = 4


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Re-order four corner points to: top-left, top-right, bottom-right, bottom-left.
    The keypoints in metadata arrive as: top-left, bottom-left, bottom-right, top-right.
    We re-order so getPerspectiveTransform maps to a sensible rectangle.

    # Source: https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def perspective_warp(image: np.ndarray, keypoints: list) -> np.ndarray:
    """
    Use the four corner keypoints to warp the puzzle region to a flat
    WARP_SIZE x WARP_SIZE square.

    Parameters
    ----------
    image     : BGR image as loaded by OpenCV
    keypoints : flat list of 8 floats [x0,y0, x1,y1, x2,y2, x3,y3]
                (top-left, bottom-left, bottom-right, top-right in pixel coords)

    Returns
    -------
    np.ndarray : grayscale warped image, shape (WARP_SIZE, WARP_SIZE)

    # Source: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
    """
    pts = np.array(keypoints, dtype=np.float32).reshape(4, 2)
    src = order_points(pts)

    dst = np.array([
        [0,           0          ],
        [WARP_SIZE-1, 0          ],
        [WARP_SIZE-1, WARP_SIZE-1],
        [0,           WARP_SIZE-1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (WARP_SIZE, WARP_SIZE))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return gray


def extract_cells_from_warp(gray_warp: np.ndarray) -> list:
    """
    Divide a WARP_SIZE x WARP_SIZE grayscale image into 81 cell crops.
    Each crop is trimmed by MARGIN pixels on all four edges (to remove grid
    line bleed) then resized to CELL_SIZE x CELL_SIZE.

    Returns a flat list of 81 uint8 arrays, row-major (index = r*9 + c).
    """
    cells = []
    for r in range(9):
        for c in range(9):
            y1 = r * CELL_PX + MARGIN
            y2 = (r + 1) * CELL_PX - MARGIN
            x1 = c * CELL_PX + MARGIN
            x2 = (c + 1) * CELL_PX - MARGIN
            crop = gray_warp[y1:y2, x1:x2]
            resized = cv2.resize(crop, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)
            cells.append(resized)
    return cells


def derive_class(cell_label: list) -> int:
    """
    Convert a single cell's 10-element label array to a class integer 0–9.

    Label format: [is_given, flag_1, flag_2, ..., flag_9]
      is_given == 1  → this cell is pre-printed in the puzzle (a "given")
                       → return whichever of flag_1..flag_9 is set
      is_given == 0  → this cell is blank in the puzzle
                       → return 0 (empty class)

    Parameters
    ----------
    cell_label : list of 10 ints from metadata cells[r][c]

    Returns
    -------
    int : 0 (empty) or 1–9 (digit)
    """
    if cell_label[0] == 1:
        for digit in range(1, 10):
            if cell_label[digit] == 1:
                return digit
        return 0  # malformed label fallback
    else:
        return 0


def load_image_robust(path: str):
    """
    Load an image from disk, handling webp/jpeg/png/jpg and RGBA.
    Returns a BGR numpy array, or None if loading fails.

    # Source: https://pillow.readthedocs.io/en/stable/reference/Image.html
    """
    try:
        pil_img = Image.open(path).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as e:
        print(f"  WARNING: could not load {path}: {e}")
        return None


def process_split(split: str) -> dict:
    """
    Run the full extraction pipeline for one split (train / val / test).
    Returns a stats dict with counts per class.
    """
    images_dir  = os.path.join(DATA_ROOT, split, "images")
    jsonl_path  = os.path.join(DATA_ROOT, split, "metadata.jsonl")
    out_dir     = os.path.join(OUTPUT_ROOT, split)

    for cls in range(10):
        os.makedirs(os.path.join(out_dir, str(cls)), exist_ok=True)

    csv_path = os.path.join(out_dir, "labels.csv")
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(["filename", "class", "puzzle_image", "row", "col"])

    stats   = {cls: 0 for cls in range(10)}
    skipped = 0

    with open(jsonl_path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for rec in tqdm(records, desc=f"  {split}", unit="puzzle"):
        raw_fn   = rec["file_name"].replace("\\", "/").replace("images/", "")
        img_path = os.path.join(images_dir, raw_fn)

        if not os.path.exists(img_path):
            print(f"  WARNING: image not found: {img_path}")
            skipped += 1
            continue

        bgr = load_image_robust(img_path)
        if bgr is None:
            skipped += 1
            continue

        keypoints   = rec["keypoints"]
        cells_label = rec["cells"]

        try:
            gray_warp = perspective_warp(bgr, keypoints)
        except Exception as e:
            print(f"  WARNING: warp failed for {raw_fn}: {e}")
            skipped += 1
            continue

        cell_images = extract_cells_from_warp(gray_warp)

        puzzle_stem = os.path.splitext(raw_fn)[0]

        for idx, cell_img in enumerate(cell_images):
            r = idx // 9
            c = idx % 9
            cls = derive_class(cells_label[r][c])

            fname    = f"{puzzle_stem}_r{r}_c{c}.png"
            out_path = os.path.join(out_dir, str(cls), fname)

            cv2.imwrite(out_path, cell_img)
            writer.writerow([fname, cls, raw_fn, r, c])
            stats[cls] += 1

    csv_file.close()

    if skipped > 0:
        print(f"  Skipped {skipped} images due to load/warp errors.")

    return stats


def print_stats(split: str, stats: dict):
    total = sum(stats.values())
    print(f"\n  {split} — {total} cells extracted")
    print(f"  {'Class':<8} {'Count':>7}  {'%':>6}")
    print(f"  {'-'*24}")
    for cls in range(10):
        label = "empty" if cls == 0 else f"digit {cls}"
        pct   = stats[cls] / total * 100 if total > 0 else 0
        print(f"  {cls} ({label:<7}) {stats[cls]:>7}  {pct:>5.1f}%")


def main():
    print("=" * 60)
    print("PhotoSudoku — Cell Extraction Preprocessing")
    print("=" * 60)
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Cell size:   {CELL_SIZE}x{CELL_SIZE} px (grayscale)")
    print(f"Margin:      {MARGIN}px per edge (matches app/image_processor.py)")
    print()

    for split in SPLITS:
        jsonl_path = os.path.join(DATA_ROOT, split, "metadata.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"Skipping {split} — metadata.jsonl not found at {jsonl_path}")
            continue
        print(f"Processing {split}...")
        stats = process_split(split)
        print_stats(split, stats)

    print("\n" + "=" * 60)
    print("Done. Cell images saved to data/cells/")
    print("Next step: run  python model/train_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
