
# The file does:
#   1. Load uploaded image bytes via Pillow → BGR numpy array
#   2. Grayscale + Gaussian blur + adaptive threshold
#   3. Find the largest 4-sided contour (the outer grid border)
#   4. Perspective-warp to a flat 450×450 square
#   5. Divide into 81 cells, resize each to 28×28
#   6. Reshape to (28,28,1) float32 — values kept in [0,255] so the model's
#      own Rescaling(1/255) layer normalises correctly at inference time.
#
# Returns a flat list of 81 np.ndarray, shape (28,28,1), dtype float32
# Primary tutorial is: https://youtu.be/oXlwWbU8l2o?si=xP8NzUDFqJJfwhZD
import cv2
import numpy as np
from PIL import Image
import io

#Constants, same as prepare_cells.py file
WARP_SIZE = 450          # intermediate canvas — divisible by 9 → 50px/cell
CELL_PX   = WARP_SIZE // 9   # 50px per cell before resize
CELL_SIZE = 28           # final cell size fed to the CNN


#Helpers

def _order_points(pts: np.ndarray) -> np.ndarray: #reordering the corners to a standard format
    # Source: https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left  (smallest x+y)
    rect[2] = pts[np.argmax(s)]   # bottom-right (largest x+y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right (smallest y-x)
    rect[3] = pts[np.argmax(diff)]  # bottom-left (largest y-x)
    return rect
#final format is top-left,top-right,bottom-right,bottom-left.

def _perspective_warp(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    src = _order_points(corners) #get the matrix with set coord from order_points
    dst = np.array([
        [0,           0           ],
        [WARP_SIZE-1, 0           ],
        [WARP_SIZE-1, WARP_SIZE-1 ],
        [0,           WARP_SIZE-1 ],
    ], dtype=np.float32) #create a matrix with desired coord mapping

    M = cv2.getPerspectiveTransform(src, dst) #create a matrix for transformation(warping)
    warped = cv2.warpPerspective(bgr, M, (WARP_SIZE, WARP_SIZE)) #run a matrix multiplication
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) #convert to grayscale from BGR
    return gray


def _find_grid_corners(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) #convert the img to grayscale
    # Denoise then threshold — adaptive threshold handles uneven lighting well
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    ) #threshold(binarise) the image

    # Dilate to close small gaps in grid lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    #dilation is widening certain pixels
    #makes the contour detection better, especially after birarisation of the image

    # Find all external contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours that we get is just a list of coordinates of contours
    #skip the second return variable since it just shows the hierarchies of contours
    #but since we are looking only for the external contours(the grid itself) we don't need it
    #if the list of contours is empty, there is prolly no sudoku in the image
    if not contours:
        raise ValueError("No contours found. Is this a Sudoku image?")

    # Sort by area descending — the grid is the biggest thing on the page
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:   # check top-5 largest contours
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    raise ValueError( #raise the value error in case we couldn't properly detect the image
        "Could not detect a Sudoku grid in this photo.\n"
        "Try a clearer photo with better lighting and less rotation."
    )


def _extract_cells_from_warp(gray_warp: np.ndarray) -> list:
    cells = []
    margin = 4  # ixels cropped from each edge — must match prepare_cells.py

    for r in range(9):
        for c in range(9): #get each of the 81 cells
            y1 = r * CELL_PX + margin #get the cord of each cell corner
            y2 = (r + 1) * CELL_PX - margin #trim the margin to delete the grid lines
            x1 = c * CELL_PX + margin
            x2 = (c + 1) * CELL_PX - margin

            crop = gray_warp[y1:y2, x1:x2] #perform list slicing, which is basically cropping the image

            #Resize to CNN input size
            resized = cv2.resize(crop, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)

            #Cast to float32 — do NOT divide by 255 here.
            #The model's first layer is Rescaling(1/255), which expects raw [0,255].
            #Dividing here and then having the model divide again gives ~[0,0.004],
            #which the model has never seen and produces garbage predictions.
            cell_arr = resized.astype(np.float32)
            cell_arr = cell_arr.reshape(CELL_SIZE, CELL_SIZE, 1)

            cells.append(cell_arr)

    return cells


#Public API

def load_image_from_upload(uploaded_file) -> np.ndarray:
    raw = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    #since openCV works with BGR and not RGB images, we have to reconvert the RGB to BGR
    #since streamlit uploads the image in bytes, we need to convert it into RGB using Image function from PIL lib
    return bgr

def extract_cells(image: np.ndarray) -> list: #this function acts as a driver kinda
    #uses all of the functions in this file to get all 81 cells
    #returns a list of cells, each cell is of size (28,28) and since they are in grayscale
    #there is only one color channel, so .shape will return (28,28,1)
    corners = _find_grid_corners(image)
    gray_warp = _perspective_warp(image, corners)
    cells = _extract_cells_from_warp(gray_warp)
    return cells
    #raises value error in case it couldn't detect a grid
