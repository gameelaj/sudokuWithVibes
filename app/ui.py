# =============================================================================
# app/ui.py — Streamlit UI (all pages, CSS, state management)
# Owner: Member 1
#
# Pipeline wired here:
#   Uploaded image
#     → app.image_processor.extract_cells()    (OpenCV grid detection)
#     → app.digit_recognizer.predict_grid()    (CNN digit recognition)
#     → app.solver.solve() / get_hint()        (backtracking solver)
#     → Streamlit grid display                 (interactive play)
# =============================================================================

import streamlit as st
import copy
import time
import numpy as np

from app.image_processor import load_image_from_upload, extract_cells
from app.digit_recognizer import load_model, predict_grid
from app.solver import solve, get_hint, find_empty
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: linear-gradient(145deg, #0d0d1a 0%, #111128 50%, #0d0d1a 100%);
}

header[data-testid="stHeader"] { background: transparent !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

/* ── Cards ── */
.glass-card {
    background: rgba(26,26,46,0.7);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 2.5rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.glass-card-sm {
    background: rgba(26,26,46,0.55);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}

/* ── Typography ── */
.hero-title {
    font-size: 3rem; font-weight: 900;
    background: linear-gradient(135deg, #00e676 0%, #00c853 40%, #e6ff00 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem; letter-spacing: -1px; white-space: nowrap;
}
.hero-subtitle {
    font-size: 1.3rem; color: rgba(224,224,224,0.6);
    font-weight: 300; margin-bottom: 2rem; letter-spacing: 0.5px;
}
.hero-tagline {
    font-size: 1.1rem; color: rgba(224,224,224,0.45);
    font-weight: 400; line-height: 1.8;
}
.section-title-center {
    font-size: 2rem; font-weight: 700; color: #e0e0e0;
    margin-bottom: 0.3rem; text-align: center;
}
.section-title {
    font-size: 2rem; font-weight: 700; color: #e0e0e0; margin-bottom: 0.3rem;
}
.section-caption { font-size:1rem; color:rgba(224,224,224,0.45); font-weight:300; margin-bottom:1.5rem; }
.section-caption-center { font-size:1rem; color:rgba(224,224,224,0.45); font-weight:300; margin-bottom:1.5rem; text-align:center; }

/* ── Neon divider ── */
.neon-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #00e676 30%, #e6ff00 70%, transparent 100%);
    border: none; margin: 1.5rem 0 2rem 0; opacity: 0.5; border-radius: 2px;
}

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, rgba(26,26,46,0.8), rgba(40,40,70,0.8)) !important;
    border: 1px solid rgba(0,230,118,0.25) !important;
    color: #e0e0e0 !important; border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important; font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important; letter-spacing: 0.3px !important;
}
div.stButton > button:hover {
    border-color: #00e676 !important;
    box-shadow: 0 0 20px rgba(0,230,118,0.2) !important;
    transform: translateY(-1px) !important; color: #00e676 !important;
}
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #00e676 0%, #00c853 100%) !important;
    border: none !important; color: #0d0d1a !important; font-weight: 700 !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 0 30px rgba(0,230,118,0.4) !important;
    transform: translateY(-2px) !important; color: #0d0d1a !important;
}

/* ── Data editor ── */
[data-testid="stDataEditor"], [data-testid="stDataFrame"] {
    border: 2px solid rgba(0,230,118,0.15) !important;
    border-radius: 14px !important; overflow: hidden !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
}
[data-testid="stDataEditor"] [data-testid="glideDataEditor"],
[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {
    font-family: 'Inter', monospace !important;
    font-size: 1.2rem !important; font-weight: 600 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(13,13,26,0.95) !important;
    border-right: 1px solid rgba(0,230,118,0.1) !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2 { color: #00e676 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,230,118,0.2) !important;
    border-radius: 14px !important; background: rgba(26,26,46,0.3) !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.05) !important;
}

/* ── Page dots ── */
.page-indicator { display:flex; justify-content:center; gap:10px; margin:1rem 0 2rem 0; }
.page-dot {
    width:10px; height:10px; border-radius:50%;
    background:rgba(224,224,224,0.15); transition:all 0.3s;
}
.page-dot.active {
    background:#00e676; box-shadow:0 0 10px rgba(0,230,118,0.5);
    width:28px; border-radius:5px;
}

/* ── Stats ── */
.stats-container { display:flex; justify-content:center; gap:1.5rem; margin:1rem 0; flex-wrap:wrap; }
.stat-card {
    background:rgba(26,26,46,0.6); border:1px solid rgba(255,255,255,0.06);
    border-radius:14px; padding:1rem 1.8rem; text-align:center;
    min-width:120px; backdrop-filter:blur(10px);
}
.stat-value { font-size:1.8rem; font-weight:800; margin-bottom:0.1rem; }
.stat-value.green { color:#00e676; }
.stat-value.red   { color:#ff5252; }
.stat-value.yellow{ color:#e6ff00; }
.stat-value.purple{ color:#b388ff; }
.stat-label { font-size:0.75rem; color:rgba(224,224,224,0.4); text-transform:uppercase; letter-spacing:1.5px; font-weight:500; }

/* ── Timer ── */
.timer-display { text-align:center; margin:0.5rem 0 1rem 0; }
.timer-value { font-size:2.2rem; font-weight:300; color:rgba(224,224,224,0.7); letter-spacing:4px; font-variant-numeric:tabular-nums; }

/* ── Pipeline progress ── */
.pipeline-step {
    display:inline-flex; align-items:center; gap:6px;
    font-size:0.82rem; font-weight:500; padding:4px 12px;
    border-radius:20px; margin:2px;
}
.pipeline-step.done { background:rgba(0,230,118,0.12); color:#00e676; border:1px solid rgba(0,230,118,0.3); }
.pipeline-step.fail { background:rgba(255,82,82,0.12); color:#ff5252; border:1px solid rgba(255,82,82,0.3); }
.pipeline-step.wait { background:rgba(255,255,255,0.05); color:rgba(224,224,224,0.4); border:1px solid rgba(255,255,255,0.1); }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Shared widget helpers
# ─────────────────────────────────────────────────────────────────────────────

def _page_dots(active_index: int):
    dots = ""
    for i, label in enumerate(["Home", "Scan", "Play"]):
        cls = "active" if i == active_index else ""
        dots += f'<div class="page-dot {cls}" title="{label}"></div>'
    st.markdown(f'<div class="page-indicator">{dots}</div>', unsafe_allow_html=True)


def _render_timer():
    if "timer_start" not in st.session_state:
        st.session_state.timer_start = time.time()
    elapsed = int(time.time() - st.session_state.timer_start)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    t = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    st.markdown(f'<div class="timer-display"><div class="timer-value">{t}</div></div>', unsafe_allow_html=True)


def _render_stats():
    w = st.session_state.get("stats_wins", 0)
    l = st.session_state.get("stats_losses", 0)
    h = st.session_state.get("stats_hints", 0)
    p = st.session_state.get("stats_played", 0)
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-card"><div class="stat-value green">{p}</div><div class="stat-label">Played</div></div>
        <div class="stat-card"><div class="stat-value green">{w}</div><div class="stat-label">Wins</div></div>
        <div class="stat-card"><div class="stat-value red">{l}</div><div class="stat-label">Gave Up</div></div>
        <div class="stat-card"><div class="stat-value purple">{h}</div><div class="stat-label">Hints</div></div>
    </div>""", unsafe_allow_html=True)


def _display_grid(grid_matrix):
    """Editable 9×9 grid using st.data_editor."""
    df = pd.DataFrame(grid_matrix, columns=[f"C{i}" for i in range(1, 10)])
    config = {f"C{i}": st.column_config.NumberColumn(min_value=0, max_value=9, step=1, label="") for i in range(1, 10)}
    edited = st.data_editor(df, column_config=config, hide_index=True, use_container_width=True)
    return edited.values.tolist()


def _render_sidebar():
    st.sidebar.title("PhotoSudoku")
    st.sidebar.write("AIT102 Group Project")
    uploaded = st.sidebar.file_uploader(
        "Upload a Sudoku puzzle photo",
        type=["jpg", "jpeg", "png", "webp"]
    )
    return uploaded


# ─────────────────────────────────────────────────────────────────────────────
# Image → Grid pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(uploaded_file) -> tuple:
    """
    Run the full pipeline on an uploaded image file.

    Returns
    -------
    (grid, error_message)
        grid         : 9×9 list of ints if successful, None on failure
        error_message: str if failed, None on success
    """
    # Step 1 — load image
    try:
        uploaded_file.seek(0)
        bgr = load_image_from_upload(uploaded_file)
    except Exception as e:
        return None, f"Could not read image: {e}"

    # Step 2 — detect grid & extract cells
    try:
        cells = extract_cells(bgr)
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Grid detection failed: {e}"

    # Step 3 — load model & recognise digits
    try:
        model = load_model()
        grid = predict_grid(cells, model)
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Digit recognition failed: {e}"

    return grid, None


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MATRIX = [
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


def _init_state():
    defaults = {
        "current_page": "front",
        "matrix": copy.deepcopy(_DEFAULT_MATRIX),
        "original_matrix": copy.deepcopy(_DEFAULT_MATRIX),
        "stats_wins": 0,
        "stats_losses": 0,
        "stats_hints": 0,
        "stats_played": 0,
        "scan_error": None,
        "scan_success": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

def _page_front():
    _page_dots(0)
    _, hero, _ = st.columns([1, 2, 1])
    with hero:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div class="hero-title">SudokuWithVibes</div>
            <div class="hero-subtitle">Your instant puzzle solver</div>
            <div class="neon-divider"></div>
            <p class="hero-tagline">
                Having a hard time solving sudoku on a newspaper or book?<br/>
                Upload a photo and we'll solve it for you instantly!
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        _, btn, _ = st.columns([1.2, 1, 1.2])
        with btn:
            if st.button("Start Upload", use_container_width=True, type="primary"):
                st.session_state.current_page = "upload_and_confirm"
                st.rerun()


def _page_upload():
    _page_dots(1)
    st.markdown('<div class="section-title">Scan & Verify</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Upload a photo — we\'ll detect the grid and fill in the numbers automatically</div>', unsafe_allow_html=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    uploaded = _render_sidebar()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card-sm"><span style="font-size:1.1rem;font-weight:600;color:#e0e0e0;">Original Picture</span></div>', unsafe_allow_html=True)
        if uploaded:
            uploaded.seek(0)
            st.image(uploaded.read(), caption="Uploaded photo", use_container_width=True)
        else:
            st.info("Upload a Sudoku puzzle photo using the sidebar.")

    with col2:
        st.markdown("""
        <div class="glass-card-sm">
            <span style="font-size:1.1rem;font-weight:600;color:#e0e0e0;">Scanned Matrix</span>
            <p style="color:rgba(224,224,224,0.4);font-size:0.85rem;margin-top:4px;">
                Double-click any cell to fix recognition errors
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Run pipeline when user clicks Scan ───────────────────────────────
        if uploaded:
            scan_col, _ = st.columns([1, 2])
            with scan_col:
                if st.button("Scan Image", use_container_width=True, type="primary"):
                    with st.spinner("Detecting grid and recognising digits..."):
                        grid, err = _run_pipeline(uploaded)
                    if err:
                        st.session_state.scan_error = err
                        st.session_state.scan_success = False
                    else:
                        st.session_state.matrix = grid
                        st.session_state.original_matrix = copy.deepcopy(grid)
                        st.session_state.scan_error = None
                        st.session_state.scan_success = True

            if st.session_state.scan_error:
                st.error(f"Scan failed: {st.session_state.scan_error}")

            if st.session_state.scan_success:
                st.success("Grid detected! Review below and correct any errors.")

        updated_board = _display_grid(st.session_state.matrix)

    # ── Control buttons ───────────────────────────────────────────────────────
    st.write("")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Start Again", use_container_width=True):
            st.session_state.current_page = "front"
            st.session_state.scan_error = None
            st.session_state.scan_success = False
            st.rerun()
    with b2:
        if st.button("Confirm Puzzle", use_container_width=True, type="primary"):
            st.session_state.matrix = updated_board
            st.session_state.original_matrix = copy.deepcopy(updated_board)
            st.session_state.timer_start = time.time()
            st.session_state.stats_played += 1
            st.session_state.scan_error = None
            st.session_state.scan_success = False
            st.session_state.current_page = "play"
            st.rerun()


def _page_play():
    _page_dots(2)
    st.markdown('<div class="section-title-center">Game Arena</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption-center">Play normally or get hints — fill in the empty cells below</div>', unsafe_allow_html=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    _render_timer()

    # ── Grid ─────────────────────────────────────────────────────────────────
    _, grid_col, _ = st.columns([1, 2, 1])
    with grid_col:
        updated_board = _display_grid(st.session_state.matrix)

    st.write("")

    # ── Action buttons ────────────────────────────────────────────────────────
    _, c1, c2, c3, _ = st.columns([0.5, 1, 1, 1, 0.5])

    with c1:
        if st.button("Save Board", use_container_width=True):
            st.session_state.matrix = updated_board
            st.success("Progress saved!")

    with c2:
        if st.button("Get Hint", use_container_width=True):
            st.session_state.matrix = updated_board
            empty = find_empty(st.session_state.matrix)
            if empty:
                row, col = empty
                val = get_hint(st.session_state.original_matrix, row, col)
                if val:
                    st.session_state.matrix[row][col] = val
                    st.session_state.stats_hints += 1
                    st.success(f"Hint: Cell ({row+1}, {col+1}) = {val}")
                    st.rerun()
                else:
                    st.error("Could not generate a hint — puzzle may be unsolvable.")
            else:
                st.success("No empty cells — puzzle is complete!")

    with c3:
        if st.button("Show Full Solution", use_container_width=True, type="primary"):
            solution = copy.deepcopy(st.session_state.original_matrix)
            if solve(solution):
                st.session_state.matrix = solution
                st.session_state.stats_wins += 1
                st.success("Puzzle Solved!")
                st.rerun()
            else:
                st.error("No valid solution exists for this puzzle.")

    # ── Restart ───────────────────────────────────────────────────────────────
    st.write("")
    _, rc, _ = st.columns([1.5, 1, 1.5])
    with rc:
        if st.button("Restart Scan", use_container_width=True):
            st.session_state.stats_losses += 1
            st.session_state.current_page = "front"
            st.session_state.pop("matrix", None)
            st.session_state.pop("original_matrix", None)
            st.session_state.pop("timer_start", None)
            st.rerun()

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.write("")
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    _render_stats()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point (called from main.py)
# ─────────────────────────────────────────────────────────────────────────────

def run():
    """Initialise and render the current page. Call this from main.py."""
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()

    page = st.session_state.current_page
    if page == "front":
        _page_front()
    elif page == "upload_and_confirm":
        _page_upload()
    elif page == "play":
        _page_play()
