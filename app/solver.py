# ============================================
# Module 4: Sudoku Solver
# Owner: Member 4: YU JIE
#
# This module implements a Sudoku solver using
# the backtracking algorithm.
#
# Functions:
# - find_empty() : Find the first empty cell
# - is_valid()   : Check whether a number can be placed
# - solve()      : Solve Sudoku using recursion and backtracking
# - get_hint()   : Return the correct digit for a selected cell
# ============================================
from copy import deepcopy
# ============================================
# Find the first empty cell in the Sudoku grid
# Returns:
#   (row, col) if an empty cell exists
#   None if the puzzle is complete
# ============================================
def find_empty(grid):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                return (row, col)
    return None

# ============================================
# Check whether a number can be placed in a cell
#
# Parameters:
#   grid : Sudoku board
#   num  : Number to test
#   pos  : (row, col)
#
# Returns:
#   True  if valid
#   False if invalid
# ============================================
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
# Check the 3×3 subgrid
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] == num and (r, c) != pos:
                return False

    return True  # Number is valid
# ============================================
# Solve Sudoku using Backtracking
#
# Returns:
#   True  if a solution is found
#   False if no solution exists
# ============================================
def solve(grid):
    # Find an empty cell
    empty = find_empty(grid)

    # If there is no empty cell, Sudoku is solved
    if not empty:
        return True

    row, col = empty
    # Try numbers 1 to 9
    for num in range(1, 10):
        if is_valid(grid, num, (row, col)):
            # Place the number temporarily
            grid[row][col] = num
            if solve(grid): # Recursively solve the remaining puzzle
                return True
            grid[row][col] = 0 # Backtracking: remove the number
            # Undo the move and try another number

    return False # No valid number found
# ============================================
# Generate a hint for a selected cell
#
# Parameters:
#   grid : Current Sudoku board
#   row  : Row index
#   col  : Column index
#
# Returns:
#   Correct digit for that position
#   None if invalid or unsolvable
# ============================================
def get_hint(grid, row, col):
    # Check if the position is inside the Sudoku board
    if row < 0 or row > 8 or col < 0 or col > 8:
        return None

    # If the cell already has a number, no hint is needed
    if grid[row][col] != 0:
        return None

    # Make a copy of the Sudoku board
    new_grid = deepcopy(grid)

    # Solve the copied board
    if solve(new_grid):
        return new_grid[row][col] # Return the correct number for that position
    else:
        return None # No solution found