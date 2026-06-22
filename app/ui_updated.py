# contains functions to upload file, preview photo, scan it to recognize grid and solve, reset or get hints
# referenced from: streamlit guide, https://www.youtube.com/watch?v=VqgUkExPvLY
#                  st.data_editor docs,   https://docs.streamlit.io/develop/api-reference/data/st.data_editor

import copy # to copy the content of the grid without changing the original, will later be used for restart option.
import pandas as pd
import streamlit as st

# imports from other files for image processing and sudoku solving 
from app.digit_recognizer import load_model, predict_grid
from app.image_processor import extract_cells, load_image_from_upload
from app.solver import find_empty, get_hint, solve

# sets the default grid which is displayed before any photo is uploaded, 0 represents empty cell
_DEFAULT_GRID = [
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

# label for columns for each of the 9 boxes as A, B and C
_BOX_COLS = ["A", "B", "C"]

# sets the value for each of the grids ranging from 0 to 9 that are integer values 
_BOX_COL_CONFIG = {
    col: st.column_config.NumberColumn(
        label=col, min_value=0, max_value=9, step=1, format="%d"
    )
    for col in _BOX_COLS
}

# Session state guide:   https://docs.streamlit.io/develop/concepts/architecture/session-state
def _init_state():# function to persist the game progress whenever the page is refreshed, prevents from data loss. 
    if "grid" not in st.session_state:
        st.session_state.grid = copy.deepcopy(_DEFAULT_GRID)# saves the changes whenever the numbers are entered 
    if "original" not in st.session_state:
        st.session_state.original = copy.deepcopy(_DEFAULT_GRID)# saves the original puzzle so that it can be used when the reset/hint button is pressed 


# divides the sudoku from 9x9 box to 3x3 boxes for display as sudoku consists of 9 boxes.
def _extract_box(grid, box_row, box_col):
    r0, c0 = box_row * 3, box_col * 3
    return [[grid[r0 + r][c0 + c] for c in range(3)] for r in range(3)]


# combines the 9 3x3 boxes back into a single sudoku puzzle to solve it 
def _reassemble(edited_boxes):
    grid = [[0] * 9 for _ in range(9)]
    # loops through the 3x3 boxes one by one and places it in the respective position for a complete sudoku puzzle 
    for (br, bc), df in edited_boxes.items():
        vals = df.fillna(0).astype(int).values
        r0, c0 = br * 3, bc * 3 # determines the starting position for each box 
        for r in range(3):
            for c in range(3):
                grid[r0 + r][c0 + c] = vals[r][c] # copies the values to the puzzle
    return grid


# function to display the sudoku grid as separate 3x3 boxes, gets the number input from the user and returns the values as dict
def _display_boxes(grid):
    edited_boxes = {} # empty dictionary for storing values later 
    for br in range(3):   # box row  (0, 1, 2)
        cols = st.columns(3)
        for bc in range(3):  # box col  (0, 1, 2)
            subgrid = _extract_box(grid, br, bc) # extracting the numbers from the big grid 
            df = pd.DataFrame(subgrid, columns=_BOX_COLS) # links those numbers into respective columns 
            with cols[bc]:  # displays the box inside the respective column on screen
                edited = st.data_editor(
                    df, column_config=_BOX_COL_CONFIG, hide_index=True, use_container_width=True, key=f"box_{br}_{bc}",
                )
            edited_boxes[(br, bc)] = edited

        st.write("")  # puts small gap between boxes

    return edited_boxes


# main entry point that is called from the main file
def run():
    st.title("SudokuWithVibes")
    st.caption("Upload a Sudoku photo → Scan → Solve ")

    _init_state()# calling the function to persist the session state 

    # section for uploading the photo 
    uploaded = st.file_uploader(
        "Upload a sudoku photo",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded:
        col_img, col_btn = st.columns([2, 1]) # displays two columns, one for image and other for buttons

        with col_img:
            uploaded.seek(0)
            st.image(uploaded, caption="Uploaded photo", use_container_width=True) # displays the uploaded photo

        # button to control image scanning 
        with col_btn:
            st.write("")
            if st.button("Scan Image", type="primary", use_container_width=True):  
                with st.spinner("Detecting grid and reading digits…"):
                    try:
                        uploaded.seek(0)
                        bgr = load_image_from_upload(uploaded) # converts the uploaded image into image compatible with python
                        cells = extract_cells(bgr) # extracts 81 cells from one photo
                        model = st.session_state.get("model") or load_model() # loads the AI model for digit recognition
                        grid  = predict_grid(cells, model) 

                        st.session_state.grid = grid # saves the scanned grid 
                        st.session_state.original = copy.deepcopy(grid)
                        st.success("Scan complete! Fix any wrong digits in the grid below.")
                    except Exception as e:
                        st.error(f"Scan failed: {e}") # displays error message 

    st.divider()

    # displays the sudoku puzzle in the form of 9 smaller 3x3 boxes 
    st.subheader("Puzzle Grid")
    st.caption("Use **0** for empty cells · double-click any cell to edit")

    edited_boxes = _display_boxes(st.session_state.grid) # calling function to display edited box 

    st.divider()

    # 3 bottom-most buttons to solve, get hint or reset 
    c1, c2, c3 = st.columns(3)

    # solve button 
    with c1:
        if st.button("Solve", type="primary", use_container_width=True):
            current = _reassemble(edited_boxes)
            solution = copy.deepcopy(current)
            if solve(solution):
                st.session_state.grid = solution
                # Clear box editor keys so they re-render with the solution
                for br in range(3):
                    for bc in range(3):
                        st.session_state.pop(f"box_{br}_{bc}", None)
                st.success("Solved!")
                st.rerun()
            else:
                st.error("No solution found — check the grid for mistakes.")

    # hint button 
    with c2:
        if st.button("Hint", use_container_width=True):
            current = _reassemble(edited_boxes)
            st.session_state.grid = current

            empty = find_empty(current) # finds the empty cell in the puzzle 
            if not empty:
                st.success("No empty cells — puzzle is complete!")
            else:
                row, col = empty # if there are empty cells, finds the position of that cell 
                val = get_hint(st.session_state.original, row, col) # detremines the correct answer for that cell
                if val:
                    st.session_state.grid[row][col] = val # places the answer in the cell 
                    for br in range(3):
                        for bc in range(3):
                            st.session_state.pop(f"box_{br}_{bc}", None)
                    st.success(f"Hint: row {row + 1}, col {col + 1} = {val}") # adds the row and column number by 1 
                    st.rerun()
                else:
                    st.error("Could not generate a hint — the puzzle may have errors.")


    # reset button 
    with c3:
        if st.button("Reset", use_container_width=True):
            st.session_state.grid = copy.deepcopy(st.session_state.original) # refers to the copy made of original grid 
            for br in range(3):
                for bc in range(3):
                    st.session_state.pop(f"box_{br}_{bc}", None)
            st.rerun()
