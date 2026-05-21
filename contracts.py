# =============================================================================
# contracts.py — PhotoSudoku Shared Interface Contracts
# =============================================================================
# This file defines the EXACT input/output signatures for every cross-module
# function in the project. Each member must implement their function to match
# the signature and docstring here — do not change the function names or
# return types without telling the whole group.
#
# HOW TO USE:
#   - Read the docstring for your function carefully.
#   - Implement it in your own module file (NOT here).
#   - Test it using the test stubs at the bottom of this file.
#   - Member 1 will import from your module into main.py / ui.py.
#
# GRID FORMAT (used everywhere):
#   A Sudoku grid is always represented as a list of 9 lists, each with 9 ints.
#   0  = empty cell
#   1-9 = given or filled digit
#
#   Example:
#   [
#     [5, 3, 0, 0, 7, 0, 0, 0, 0],
#     [6, 0, 0, 1, 9, 5, 0, 0, 0],
#     ...
#   ]
#   Access a cell: grid[row][col], where row=0 is the top row, col=0 is left.
# =============================================================================

from __future__ import annotations
import numpy as np


# -----------------------------------------------------------------------------
# MODULE 2 — image_processor.py
# Owner: Member 2
# -----------------------------------------------------------------------------

def extract_cells(image: np.ndarray) -> list[np.ndarray]:
    """
    Detect the Sudoku grid in a photo and extract all 81 cell images.

    Parameters
    ----------
    image : np.ndarray
        A BGR image as loaded by OpenCV (cv2.imread or cv2.imdecode).
        Shape: (H, W, 3), dtype uint8.

    Returns
    -------
    list[np.ndarray]
        A flat list of exactly 81 numpy arrays.
        Each array:  shape (28, 28, 1), dtype float32, values in [0.0, 1.0]
        Ordering:    row-by-row, left-to-right, top-to-bottom.
                     Index 0 = top-left cell, index 80 = bottom-right cell.
                     Cell at grid row r, col c → cells[r * 9 + c]

    Raises
    ------
    ValueError
        If no Sudoku grid can be detected in the image.
        The caller (UI) will catch this and prompt the user to retake the photo.

    Notes
    -----
    Steps to implement (suggested):
      1. Convert to grayscale and apply adaptive thresholding.
      2. Find the largest 4-sided contour (the grid border).
      3. Apply perspective warp to get a flat top-down view.
      4. Divide the warped image into a 9x9 grid of cells.
      5. For each cell: crop, resize to 28x28, normalize to [0,1], reshape to (28,28,1).

    Referenced tutorial (add your own URL when implemented):
    # Source: <paste URL here>
    """
    raise NotImplementedError("Member 2: implement extract_cells() in image_processor.py")


# -----------------------------------------------------------------------------
# MODULE 3 — digit_recognizer.py
# Owner: Member 3
# -----------------------------------------------------------------------------

def load_model(model_path: str = "model/digit_model.h5"):
    """
    Load the trained TensorFlow CNN model from disk.

    Parameters
    ----------
    model_path : str
        Relative path to the saved .h5 model file.
        Default: "model/digit_model.h5"
        Always use os.path.join() — never hardcode an absolute path.

    Returns
    -------
    tensorflow.keras.Model
        The loaded Keras model, ready for inference.

    Notes
    -----
    Call this ONCE when the app starts (in main.py), not on every prediction.
    Store the result in st.session_state so Streamlit doesn't reload it on
    every interaction.

    Referenced tutorial (add your own URL when implemented):
    # Source: <paste URL here>
    """
    raise NotImplementedError("Member 3: implement load_model() in digit_recognizer.py")


def predict_grid(cells: list[np.ndarray], model) -> list[list[int]]:
    """
    Run the CNN model on all 81 cell images and return the detected puzzle grid.

    Parameters
    ----------
    cells : list[np.ndarray]
        Exactly 81 arrays from extract_cells(). Shape per array: (28, 28, 1).
    model : tensorflow.keras.Model
        The loaded model from load_model().

    Returns
    -------
    list[list[int]]
        A 9x9 grid in the standard grid format.
        0 = cell was detected as empty.
        1-9 = detected digit.

    Notes
    -----
    - Stack all 81 cells into a single batch: np.stack(cells) → shape (81,28,28,1)
    - Run model.predict() on the batch (faster than 81 individual calls).
    - Map class 0 → 0 (empty), classes 1-9 → digit 1-9.
    - Reshape the flat list of 81 predictions into a 9x9 list of lists.

    Referenced tutorial (add your own URL when implemented):
    # Source: <paste URL here>
    """
    raise NotImplementedError("Member 3: implement predict_grid() in digit_recognizer.py")


# -----------------------------------------------------------------------------
# MODULE 4 — solver.py
# Owner: Member 4
# -----------------------------------------------------------------------------

def is_valid(grid: list[list[int]], row: int, col: int, num: int) -> bool:
    """
    Check whether placing `num` at grid[row][col] is valid under Sudoku rules.

    Parameters
    ----------
    grid : list[list[int]]
        Current 9x9 grid state (0 = empty).
    row : int
        Row index (0–8).
    col : int
        Column index (0–8).
    num : int
        Digit to check (1–9).

    Returns
    -------
    bool
        True if placing num at (row, col) does not violate any Sudoku rule.
        Rules: num must not already exist in the same row, column, or 3x3 box.

    Notes
    -----
    This is a helper used internally by solve(). Also useful for validating
    user input during regular play mode.

    Referenced tutorial (add your own URL when implemented):
    # Source: <paste URL here>
    """
    raise NotImplementedError("Member 4: implement is_valid() in solver.py")


def find_empty(grid: list[list[int]]) -> tuple[int, int] | None:
    """
    Find the next empty cell (value 0) in the grid.

    Parameters
    ----------
    grid : list[list[int]]
        Current 9x9 grid state.

    Returns
    -------
    tuple[int, int] or None
        (row, col) of the first empty cell found (scanned row by row, left to right).
        None if no empty cell exists (puzzle is fully filled).

    Notes
    -----
    This is a helper used internally by solve().

    Referenced tutorial (add your own URL when implemented):
    # Source: <paste URL here>
    """
    raise NotImplementedError("Member 4: implement find_empty() in solver.py")


def solve(grid: list[list[int]]) -> bool:
    """
    Solve the Sudoku puzzle in-place using backtracking.

    Parameters
    ----------
    grid : list[list[int]]
        A 9x9 grid. IMPORTANT: this is modified in-place.
        Pass a copy (copy.deepcopy) if you need to keep the original.

    Returns
    -------
    bool
        True  — puzzle solved successfully; grid now contains the solution.
        False — puzzle has no valid solution (e.g. bad detection from camera).

    Notes
    -----
    Algorithm:
      1. Find the next empty cell using find_empty().
      2. Try digits 1-9; for each, check is_valid().
      3. If valid, place the digit and recurse.
      4. If recursion returns False, reset the cell to 0 (backtrack).
      5. If no digit works, return False.
      6. If no empty cell remains, return True.

    Referenced tutorial (add your own URL when implemented):
    # Source: <paste URL here>
    """
    raise NotImplementedError("Member 4: implement solve() in solver.py")


def get_hint(grid: list[list[int]], row: int, col: int) -> int | None:
    """
    Return the correct digit for a specific empty cell, without modifying the grid.

    Parameters
    ----------
    grid : list[list[int]]
        The ORIGINAL puzzle grid (only given digits, 0s for empty).
        Do NOT pass the in-progress player grid directly — pass a deepcopy.
    row : int
        Row index of the cell the user wants a hint for (0–8).
    col : int
        Column index of the cell the user wants a hint for (0–8).

    Returns
    -------
    int or None
        The correct digit (1–9) for that cell.
        None if the cell already has a digit (user shouldn't be asking for a hint).

    Notes
    -----
    Implementation: solve a deepcopy of the grid, then read off grid_copy[row][col].
    This avoids mutating the player's current game state.

    Referenced tutorial (add your own URL when implemented):
    # Source: <paste URL here>
    """
    raise NotImplementedError("Member 4: implement get_hint() in solver.py")


# -----------------------------------------------------------------------------
# MODULE 5 — train_model.py
# Owner: Member 5
# Contract: the output file, not a callable function.
# -----------------------------------------------------------------------------

# Member 5 does not expose functions called by other modules.
# The contract is the SAVED FILE at: model/digit_model.h5
#
# Requirements for the saved model:
#   Input shape:  (None, 28, 28, 1)  — batch of grayscale cell images
#   Output shape: (None, 10)         — softmax probabilities for classes 0-9
#   Class meaning: 0=empty, 1=digit 1, ..., 9=digit 9
#   File path:    model/digit_model.h5  (relative to project root)
#
# Member 3 will call:
#   model = tf.keras.models.load_model("model/digit_model.h5")
#   predictions = model.predict(batch)  # batch shape: (81, 28, 28, 1)
#
# Make sure the model is saved AFTER training with:
#   model.save("model/digit_model.h5")
#
# DO NOT save it anywhere else or with a different filename.


# =============================================================================
# INTEGRATION TESTS
# Run this file directly to check your implementations against the contracts.
# Usage: python contracts.py
#
# Each test imports from the real module file — replace the stubs with your
# actual implementations before running.
# =============================================================================

def _make_dummy_grid() -> list[list[int]]:
    """A known solvable Sudoku puzzle for testing."""
    return [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]


def _run_tests():
    print("=" * 60)
    print("PhotoSudoku — Contract Integration Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    # ── Test 1: extract_cells output format ──────────────────────
    print("\n[Test 1] extract_cells() — output format")
    try:
        from app.image_processor import extract_cells
        import cv2
        dummy_img = np.zeros((450, 450, 3), dtype=np.uint8)
        try:
            cells = extract_cells(dummy_img)
            assert isinstance(cells, list), "Return type must be list"
            assert len(cells) == 81, f"Must return 81 cells, got {len(cells)}"
            assert cells[0].shape == (28, 28, 1), f"Each cell must be (28,28,1), got {cells[0].shape}"
            assert cells[0].dtype == np.float32, f"dtype must be float32, got {cells[0].dtype}"
            assert 0.0 <= cells[0].max() <= 1.0, "Values must be normalized to [0, 1]"
            print("  PASS — returns 81 x (28,28,1) float32 arrays normalized to [0,1]")
            passed += 1
        except ValueError:
            print("  PASS (acceptable) — raised ValueError for undetectable grid")
            passed += 1
    except NotImplementedError:
        print("  SKIP — Member 2 has not implemented this yet")
    except ImportError:
        print("  SKIP — app/image_processor.py not found")

    # ── Test 2: predict_grid output format ───────────────────────
    print("\n[Test 2] predict_grid() — output format")
    try:
        from app.digit_recognizer import predict_grid, load_model
        import os
        if os.path.exists("model/digit_model.h5"):
            model = load_model()
            dummy_cells = [np.zeros((28, 28, 1), dtype=np.float32)] * 81
            result = predict_grid(dummy_cells, model)
            assert isinstance(result, list), "Return type must be list"
            assert len(result) == 9, f"Must have 9 rows, got {len(result)}"
            assert all(len(row) == 9 for row in result), "Each row must have 9 cols"
            assert all(isinstance(v, int) for row in result for v in row), "All values must be int"
            assert all(0 <= v <= 9 for row in result for v in row), "All values must be 0-9"
            print("  PASS — returns 9x9 list[list[int]] with values 0-9")
            passed += 1
        else:
            print("  SKIP — model/digit_model.h5 not found (Member 5 must train first)")
    except NotImplementedError:
        print("  SKIP — Member 3 has not implemented this yet")
    except ImportError:
        print("  SKIP — app/digit_recognizer.py not found")

    # ── Test 3: is_valid() correctness ───────────────────────────
    print("\n[Test 3] is_valid() — correctness")
    try:
        from app.solver import is_valid
        grid = _make_dummy_grid()
        assert is_valid(grid, 0, 2, 4) == True,  "4 should be valid at (0,2)"
        assert is_valid(grid, 0, 2, 5) == False, "5 already in row 0"
        assert is_valid(grid, 0, 2, 3) == False, "3 already in column 2 (row 2)"
        assert is_valid(grid, 0, 2, 9) == False, "9 already in top-left box"
        print("  PASS — correctly validates row, column, and box constraints")
        passed += 1
    except NotImplementedError:
        print("  SKIP — Member 4 has not implemented is_valid() yet")
    except ImportError:
        print("  SKIP — app/solver.py not found")

    # ── Test 4: find_empty() correctness ─────────────────────────
    print("\n[Test 4] find_empty() — correctness")
    try:
        from app.solver import find_empty
        grid = _make_dummy_grid()
        result = find_empty(grid)
        assert result == (0, 2), f"First empty should be (0,2), got {result}"
        full_grid = [[1]*9 for _ in range(9)]
        assert find_empty(full_grid) is None, "Should return None for full grid"
        print("  PASS — correctly finds first empty cell and returns None when full")
        passed += 1
    except NotImplementedError:
        print("  SKIP — Member 4 has not implemented find_empty() yet")
    except ImportError:
        print("  SKIP — app/solver.py not found")

    # ── Test 5: solve() correctness ──────────────────────────────
    print("\n[Test 5] solve() — solves a known puzzle correctly")
    try:
        from app.solver import solve
        import copy
        grid = _make_dummy_grid()
        grid_copy = copy.deepcopy(grid)
        result = solve(grid_copy)
        assert result == True, "Should return True for solvable puzzle"
        assert all(grid_copy[r][c] != 0 for r in range(9) for c in range(9)), \
            "All cells must be filled after solving"
        assert all(grid_copy[r][c] == grid[r][c]
                   for r in range(9) for c in range(9)
                   if grid[r][c] != 0), \
            "Given digits must not be changed"
        print("  PASS — solves correctly, preserves given digits, fills all cells")
        passed += 1
    except NotImplementedError:
        print("  SKIP — Member 4 has not implemented solve() yet")
    except ImportError:
        print("  SKIP — app/solver.py not found")

    # ── Test 6: get_hint() correctness ───────────────────────────
    print("\n[Test 6] get_hint() — returns correct digit without mutating grid")
    try:
        from app.solver import get_hint
        import copy
        grid = _make_dummy_grid()
        original = copy.deepcopy(grid)
        hint = get_hint(grid, 0, 2)
        assert isinstance(hint, int), f"Hint must be int, got {type(hint)}"
        assert 1 <= hint <= 9, f"Hint must be 1-9, got {hint}"
        assert grid == original, "get_hint() must not modify the original grid"
        assert get_hint(grid, 0, 0) is None, "Should return None for a non-empty cell"
        print(f"  PASS — hint for (0,2) = {hint}, original grid unchanged")
        passed += 1
    except NotImplementedError:
        print("  SKIP — Member 4 has not implemented get_hint() yet")
    except ImportError:
        print("  SKIP — app/solver.py not found")

    # ── Test 7: model file contract ──────────────────────────────
    print("\n[Test 7] model/digit_model.h5 — file and shape contract")
    try:
        import os
        import tensorflow as tf
        if os.path.exists("model/digit_model.h5"):
            model = tf.keras.models.load_model("model/digit_model.h5")
            assert model.input_shape == (None, 28, 28, 1), \
                f"Input shape must be (None,28,28,1), got {model.input_shape}"
            assert model.output_shape == (None, 10), \
                f"Output shape must be (None,10), got {model.output_shape}"
            print("  PASS — model loads, input (None,28,28,1), output (None,10)")
            passed += 1
        else:
            print("  SKIP — model/digit_model.h5 not found (Member 5 must train first)")
    except ImportError:
        print("  SKIP — TensorFlow not installed in this environment")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed} passed, {failed} failed, {7 - total} skipped")
    if failed == 0:
        print("All implemented functions match their contracts.")
    else:
        print("Fix the failures above before integrating.")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
