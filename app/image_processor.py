# =============================================================================
# app/image_processor.py — OpenCV Grid Detection & Cell Extraction
# Owner: Member 2
#
# Pipeline:
#   1. Load uploaded image bytes via Pillow → BGR numpy array
#   2. Grayscale + Gaussian blur + adaptive threshold
#   3. Find the largest 4-sided contour (the outer grid border)
#   4. Perspective-warp to a flat 450×450 square
#   5. Divide into 81 cells, resize each to 28×28
#   6. Reshape to (28,28,1) float32 — values kept in [0,255] so the model's
#      own Rescaling(1/255) layer normalises correctly at inference time.
#
# Returns a flat list of 81 np.ndarray, shape (28,28,1), dtype float32
# =============================================================================

import cv2
import numpy as np
from PIL import Image
import io

# ── Constants ────────────────────────────────────────────────────────────────
WARP_SIZE = 450          # intermediate canvas — divisible by 9 → 50px/cell
CELL_PX   = WARP_SIZE // 9   # 50px per cell before resize
CELL_SIZE = 28           # final cell size fed to the CNN


# ── Helpers ──────────────────────────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Re-order four corner points to: top-left, top-right, bottom-right, bottom-left.
    Works for any quadrilateral regardless of how corners were detected.

    # Source: https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left  (smallest x+y)
    rect[2] = pts[np.argmax(s)]   # bottom-right (largest x+y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right (smallest y-x)
    rect[3] = pts[np.argmax(diff)]  # bottom-left (largest y-x)
    return rect


def _perspective_warp(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Warp the detected puzzle region to a flat WARP_SIZE×WARP_SIZE grayscale image.

    Parameters
    ----------
    bgr     : BGR image (H, W, 3) uint8
    corners : (4, 2) float32 array of the grid's four corner pixel coordinates

    Returns
    -------
    np.ndarray : grayscale warped image, shape (WARP_SIZE, WARP_SIZE), uint8
    """
    src = _order_points(corners)
    dst = np.array([
        [0,           0           ],
        [WARP_SIZE-1, 0           ],
        [WARP_SIZE-1, WARP_SIZE-1 ],
        [0,           WARP_SIZE-1 ],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, M, (WARP_SIZE, WARP_SIZE))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return gray


def _find_grid_corners(bgr: np.ndarray) -> np.ndarray:
    """
    Detect the largest quadrilateral contour in the image — assumed to be the
    outer border of the Sudoku grid.

    Parameters
    ----------
    bgr : BGR image (H, W, 3) uint8

    Returns
    -------
    np.ndarray : (4, 2) float32 corner points

    Raises
    ------
    ValueError : if no valid 4-sided contour can be found
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Denoise then threshold — adaptive threshold handles uneven lighting well
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )

    # Dilate to close small gaps in grid lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Find all external contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found. Is this a Sudoku image?")

    # Sort by area descending — the grid is the biggest thing on the page
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:   # check top-5 largest contours
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    raise ValueError(
        "Could not detect a Sudoku grid in this photo.\n"
        "Try a clearer photo with better lighting and less rotation."
    )


def _extract_cells_from_warp(gray_warp: np.ndarray) -> list:
    """
    Divide a WARP_SIZE×WARP_SIZE grayscale image into 81 cell arrays.

    Each cell is pre-processed:
      - Cropped with a 4-pixel inner margin to reduce grid-line bleed
        (MUST match the margin used in model/prepare_cells.py so training
        and inference inputs are consistent)
      - Resized to CELL_SIZE×CELL_SIZE
      - Cast to float32 — values stay in [0, 255] so the model's
        Rescaling(1/255) layer handles normalisation

    Returns a flat list of 81 np.ndarray, row-major (r*9 + c).
    """
    cells = []
    margin = 4  # pixels cropped from each edge — must match prepare_cells.py

    for r in range(9):
        for c in range(9):
            y1 = r * CELL_PX + margin
            y2 = (r + 1) * CELL_PX - margin
            x1 = c * CELL_PX + margin
            x2 = (c + 1) * CELL_PX - margin

            crop = gray_warp[y1:y2, x1:x2]

            # Resize to CNN input size
            resized = cv2.resize(crop, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)

            # Cast to float32 — do NOT divide by 255 here.
            # The model's first layer is Rescaling(1/255), which expects raw [0,255].
            # Dividing here and then having the model divide again gives ~[0,0.004],
            # which the model has never seen and produces garbage predictions.
            cell_arr = resized.astype(np.float32)
            cell_arr = cell_arr.reshape(CELL_SIZE, CELL_SIZE, 1)

            cells.append(cell_arr)

    return cells


# ── Public API ────────────────────────────────────────────────────────────────

def load_image_from_upload(uploaded_file) -> np.ndarray:
    """
    Convert a Streamlit UploadedFile (or any file-like object) to a BGR
    numpy array suitable for OpenCV processing.

    Parameters
    ----------
    uploaded_file : Streamlit UploadedFile / BytesIO / bytes

    Returns
    -------
    np.ndarray : (H, W, 3) uint8 BGR image
    """
    raw = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return bgr


def extract_cells(image: np.ndarray) -> list:
    """
    Detect the Sudoku grid in a photo and extract all 81 cell images.

    This is the main entry point matching the contracts.py interface.

    Parameters
    ----------
    image : np.ndarray
        BGR image as returned by load_image_from_upload() or cv2.imread().
        Shape: (H, W, 3), dtype uint8.

    Returns
    -------
    list[np.ndarray]
        Flat list of exactly 81 arrays.
        Each: shape (28, 28, 1), dtype float32, values in [0.0, 255.0].
        (The model's Rescaling layer normalises to [0,1] internally.)
        Ordering: row-major, top-to-bottom, left-to-right.
        Cell at (row r, col c) → cells[r * 9 + c].

    Raises
    ------
    ValueError
        If no Sudoku grid can be detected. The UI catches this and asks the
        user to retake the photo.
    """
    corners = _find_grid_corners(image)
    gray_warp = _perspective_warp(image, corners)
    cells = _extract_cells_from_warp(gray_warp)
    return cells
