# ============================================
# Sudoku Solver
# Owner: Member 4

# This module implements a Sudoku solver using the backtracking algorithm.

# Functions:
# 1. find_empty() : Find the first empty cell
# 2. is_valid()   : Check whether a number can be placed
# 3. solve()      : Solve Sudoku using recursion and backtracking
# 4. get_hint()   : Return the correct digit for a selected cell
# ============================================
from copy import deepcopy
# Find the next empty cell (0)
def find_empty(grid):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                return (row, col)
    return None

# Check whether a number can be placed in a cell
def is_valid(grid, num, pos):
    row, col = pos
# Check row
    for c in range(9):
        if grid[row][c] == num and c != col:
            return False
# Check column
    for r in range(9):
        if grid[r][col] == num and r != row:
            return False
# Check 3x3 box
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] == num and (r, c) != pos:
                return False

    return True  # Number is valid

# Solve the puzzle using backtracking
def solve(grid):
    empty = find_empty(grid)    # 1-Find an empty cell
    if not empty:
        return True # No empty cell means puzzle solved

    row, col = empty
    # 2-Try numbers 1-9
    for num in range(1, 10):
        # 3-Check if the number is valid
        if is_valid(grid, num, (row, col)):
            grid[row][col] = num

            #Solve the remaining cells recursively
            if solve(grid):
                return True
            grid[row][col] = 0 # Wrong choice, undo it and try another number

    return False # No valid number found

# Give the correct number for a selected empty cell
def get_hint(grid, row, col):
    # Check if the position is valid
    if row < 0 or row > 8 or col < 0 or col > 8:
        return None

    # Cell already contains a number
    if grid[row][col] != 0:
        return None

    # Create a copy of the board
    new_grid = deepcopy(grid)

    # Solve the copied board
    if solve(new_grid):
        return new_grid[row][col] # Solve the copied puzzle
    else:
        return None # No solution found