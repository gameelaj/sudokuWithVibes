# contains functions to upload file, preview photo, scan it to recognize grid and solve, reset or get hints
# referenced from: tkinter docs, https://docs.python.org/3/library/tkinter.html
#                  tkinter grid layout, https://www.pythontutorial.net/tkinter/tkinter-grid/

import copy  # to copy the content of the grid without changing the original, will later be used for restart option.
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk  # used to display the uploaded photo in the tkinter window

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


class SudokuApp(tk.Tk):
    """Main application window for SudokuWithVibes."""

    def __init__(self):
        super().__init__()
        # displays the title 
        self.title("SudokuWithVibes")
        self.resizable(False, False)
        self.configure(bg="#f0f0f0")

        # saves the changes whenever the numbers are entered
        self.grid_state = copy.deepcopy(_DEFAULT_GRID)
        # saves the original puzzle so that it can be used when the reset/hint button is pressed
        self.original = copy.deepcopy(_DEFAULT_GRID)

        # 9x9 grid of StringVar, one per cell, used to read/write cell values
        self.cell_vars = [[tk.StringVar() for _ in range(9)] for _ in range(9)]
        # holds the Entry widgets so we can style them (e.g. mark pre-filled cells)
        self.cell_entries = [[None] * 9 for _ in range(9)]

        # loaded AI model is cached here after first use so we don't reload it every scan
        self._model = None

        # path to the currently uploaded image file
        self._uploaded_path = None

        self._build_ui()
        self._load_grid_into_ui(self.grid_state)

  
    #builds the widgets and puts them in the window                                                     
    def _build_ui(self):
        # title display
        tk.Label(
            self, text="SudokuWithVibes", font=("Helvetica", 20, "bold"), bg="#f0f0f0"
        ).pack(pady=(12, 0))
        tk.Label(
            self, text="Upload a Sudoku photo → Scan → Solve",font=("Helvetica", 10), fg="#555555", bg="#f0f0f0"
        ).pack(pady=(0, 8))

        # row for uploading the photo and file name display 
        upload_frame = tk.Frame(self, bg="#f0f0f0")
        upload_frame.pack(fill="x", padx=20, pady=(0, 4))

        #upload button display 
        tk.Button(
            upload_frame, text="Upload Sudoku Photo",
            command=self._on_upload, width=22,
            bg="#4a90d9", fg="white", relief="flat", cursor="hand2"
        ).pack(side="left")

        self._filename_label = tk.Label(
            upload_frame, text="No file selected",
            font=("Helvetica", 9), fg="#777777", bg="#f0f0f0"
        )
        self._filename_label.pack(side="left", padx=8)

        # area for image preview
        # main container frame
        main_container = tk.Frame(self, bg="#f0f0f0")
        main_container.pack(padx=20, pady=4)

        # left side frame for image preview and scan button
        left_frame = tk.Frame(main_container, bg="#f0f0f0")
        left_frame.pack(side="left", padx=10, anchor="center")

        # area for image preview
        self._img_label = tk.Label(left_frame, bg="#f0f0f0")
        self._img_label.pack(pady=(0, 4))

        # scan button which would work only after photo is uploaded 
        self._scan_btn = tk.Button(
            left_frame, text="Scan Image",
            command=self._on_scan, width=22,
            bg="#2ecc71", fg="black", relief="flat", cursor="hand2"
        )

        # right side frame for the sudoku puzzle grid
        right_frame = tk.Frame(main_container, bg="#f0f0f0")
        right_frame.pack(side="left", padx=10, anchor="center")

        # label and description for puzzle grid 
        tk.Label(
            right_frame, text="Puzzle Grid",
            font=("Helvetica", 13, "bold"), bg="#f0f0f0"
        ).pack()
        tk.Label(
            right_frame, text="Click any cell to edit",
            font=("Helvetica", 9), fg="#555555", bg="#f0f0f0"
        ).pack(pady=(0, 6))

        # 9×9 sudoku grid display area 
        self._grid_frame = tk.Frame(right_frame, bg="#333333", bd=2, relief="solid")
        self._grid_frame.pack(padx=20, pady=(0, 8))
        self._build_grid_widgets()

        # buttons for solving, getting hint and resetting 
        btn_frame = tk.Frame(self, bg="#f0f0f0")
        btn_frame.pack(pady=(4, 14))

        tk.Button(
            btn_frame, text="Solve", width=12,
            command=self._on_solve,
            bg="#4a90d9", fg="black", relief="flat", cursor="hand2"
        ).pack(side="left", padx=6)

        tk.Button(
            btn_frame, text="Hint", width=12,
            command=self._on_hint,
            bg="#f39c12", fg="black", relief="flat", cursor="hand2"
        ).pack(side="left", padx=6)

        tk.Button(
            btn_frame, text="Reset", width=12,
            command=self._on_reset,
            bg="#e74c3c", fg="black", relief="flat", cursor="hand2"
        ).pack(side="left", padx=6)

        # Status bar at the bottom 
        self._status_var = tk.StringVar(value="")
        tk.Label(
            self, textvariable=self._status_var,
            font=("Helvetica", 9), fg="#2c7a2c", bg="#f0f0f0"
        ).pack(pady=(0, 8))

    def _build_grid_widgets(self):
        """Creates 81 Entry widgets arranged as nine 3×3 boxes inside the grid frame."""
        # iterates over the 3×3 arrangement of boxes; br stands for box row and bc for box column.
        for br in range(3):
            for bc in range(3):
                # outer frame gives each 3×3 box a visible border
                box_frame = tk.Frame(
                    self._grid_frame,
                    bg="#333333", bd=1, relief="solid",
                    padx=1, pady=1
                )
                box_frame.grid(row=br, column=bc, padx=1, pady=1)

                # inner cells of the 3×3 box
                for r in range(3):
                    for c in range(3):
                        row_idx = br * 3 + r  # row index in 9×9 grid
                        col_idx = bc * 3 + c  # column index in 9×9 grid

                        # getting the input number from the user for the grid
                        entry = tk.Entry(
                            box_frame,
                            textvariable=self.cell_vars[row_idx][col_idx],
                            width=2,
                            font=("Helvetica", 16, "bold"),
                            justify="center",
                            bd=0, relief="flat",
                            bg="white", fg="#222222",
                            insertbackground="#222222",
                        )
                        entry.grid(row=r, column=c, padx=1, pady=1, ipady=4)

                        # set allowed values entered to be from 0 to 9 
                        vcmd = (self.register(self._validate_cell), "%P")
                        entry.config(validate="key", validatecommand=vcmd) # put the number in the grid if is valid

                        self.cell_entries[row_idx][col_idx] = entry

    #  Grid helpers                                                      
    def _load_grid_into_ui(self, grid):
        """Writes a 9×9 list-of-lists into the cell StringVars for display."""
        for r in range(9):
            for c in range(9):
                val = grid[r][c]
                # display 0 as blank  
                self.cell_vars[r][c].set("" if val == 0 else str(val))

    def _read_grid_from_ui(self):
        """Reads the current cell values from the UI and returns a 9×9 list."""
        grid = []
        for r in range(9):
            row = []
            for c in range(9):
                raw = self.cell_vars[r][c].get().strip()
                row.append(int(raw) if raw.isdigit() else 0)
            grid.append(row)
        return grid

    @staticmethod

    # function to validate the cell entry 
    def _validate_cell(value_after):
        #Allows only a single digit (0-9) or an empty string in each cell.
        return value_after == "" or (value_after.isdigit() and len(value_after) == 1)

    #  Opens a file dialog to let the user pick a sudoku photo.                                                    
    def _on_upload(self):
        path = filedialog.askopenfilename(
            title="Select a Sudoku photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp"), ("All files", "*.*")]
        )
        if not path:
            return  # returns if the user cancels to upload any file 

        self._uploaded_path = path
        self._filename_label.config(text=path.split("/")[-1])  # show only the filename

        # displays the uploaded photo which fits within 300×300 display box
        img = Image.open(path)
        img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(img)
        self._img_label.config(image=photo)
        self._img_label.image = photo  # keeps a reference to prevent it from being garbage collected 
         
        self._scan_btn.pack(pady=(10, 0))# the scan button aappears once the image is uploaded 
        self._set_status("")

    def _on_scan(self):
        #Reads digits from the uploaded photo using the AI model and fills the grid.
        if not self._uploaded_path:
            messagebox.showwarning("No image", "Please upload a sudoku photo first.")
            return

        self._set_status("Detecting grid and reading digits…")
        self.update_idletasks()  # flush UI so the status message appears immediately

        try:
            # converts the uploaded image into a format compatible with the image processor
            with open(self._uploaded_path, "rb") as fh:
                bgr = load_image_from_upload(fh)

            cells = extract_cells(bgr)  # extracts 81 cells from the photo

            # loads the AI model for digit recognition; cache it to avoid reloading
            if self._model is None:
                self._model = load_model()
            grid = predict_grid(cells, self._model)

            # saves the scanned grid as both the working state and the original baseline
            self.grid_state = grid
            self.original = copy.deepcopy(grid)
            self._load_grid_into_ui(grid)
            self._set_status("Scan complete! Fix any wrong digits in the grid below.")
        except Exception as exc:
            messagebox.showerror("Scan failed", str(exc))  # displays error message
            self._set_status("")

    def _on_solve(self):
        """Solves the current puzzle and updates the grid display with the solution."""
        current = self._read_grid_from_ui()
        solution = copy.deepcopy(current)

        if solve(solution):
            self.grid_state = solution
            self._load_grid_into_ui(solution)
            self._set_status("Solved!")
        else:
            messagebox.showerror("No solution", "No solution found. Check the grid for mistakes.")

    #function to call whenever the hint button is pressed 
    def _on_hint(self):
        #Correct value is filled starting from the top left empty cell each time the hint button is pressed.
        current = self._read_grid_from_ui()
        self.grid_state = current

        empty = find_empty(current)  # sees which cell is empty 
        if not empty:
            messagebox.showinfo("Complete", "All cells are filled. Puzzle is complete!")
            return

        row, col = empty  # stores the position of the empty cell where hint will be displayed 
        val = get_hint(self.original, row, col)  # determines the correct answer for that cell
        if val:
            self.grid_state[row][col] = val  # places the answer in the cell
            self._load_grid_into_ui(self.grid_state)
            # adds 1 to row and column so the display uses 1-based numbering
            self._set_status(f"Hint: row {row + 1}, col {col + 1} = {val}")
        else:
            messagebox.showerror("Hint error", "Unable to generate a hint, the puzzle may have errors!")

    #Restores the grid to the original puzzle, discarding any user edits
    def _on_reset(self):
        self.grid_state = copy.deepcopy(self.original)  # refers to the copy made of the original grid
        self._load_grid_into_ui(self.grid_state)
        self._set_status("")

    #  Utility                                                            
    def _set_status(self, message):
        """Updates the status bar text at the bottom of the window."""
        self._status_var.set(message)


# main entry point that is called from the main file
def run():
    app = SudokuApp()
    app.mainloop()

if __name__ == "__main__":
    run()
