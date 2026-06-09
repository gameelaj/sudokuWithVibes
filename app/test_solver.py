# ============================================
# Module 4: Solver Testing
# Owner: Member 4: YU JIE
#
# Run from the project root:
#   python app/test_solver.py
#
# Test Cases:
# 1. Solve a normal Sudoku puzzle
# 2. Verify a completed Sudoku puzzle
# 3. Test the hint generation function
# ============================================
from app.solver import solve, get_hint

# ==========================
# Test 1: Solve a normal Sudoku puzzle
# ==========================

print("TEST 1: Solve Sudoku")
puzzle1 = [
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9]
]
if solve(puzzle1):
    print("Sudoku solved successfully!")
    for row in puzzle1:
        print(row)
else:
    print("No solution found.")

# ==========================
# Test 2: Already solved Sudoku
print("\nTEST 2: Completed Sudoku")
# ==========================

puzzle2 = [
    [5,3,4,6,7,8,9,1,2],
    [6,7,2,1,9,5,3,4,8],
    [1,9,8,3,4,2,5,6,7],
    [8,5,9,7,6,1,4,2,3],
    [4,2,6,8,5,3,7,9,1],
    [7,1,3,9,2,4,8,5,6],
    [9,6,1,5,3,7,2,8,4],
    [2,8,7,4,1,9,6,3,5],
    [3,4,5,2,8,6,1,7,9]
]

if solve(puzzle2):
    print("Completed Sudoku is valid.")
else:
    print("Error detected.")

# ==========================
# Test 3: Hint Function
# ==========================

print("\nTEST 3: Hint Function")

puzzle3 = [
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9]
]

hint = get_hint(puzzle3, 0, 2)
print("Hint for position (0, 2):", hint)
