"""Paper Monitoring — Streamlit Dashboard.

Single-page app that displays classified papers from the SQLite graph store
and provides a "Run Weekly Update" button that launches the pipeline as a
subprocess.  The subprocess approach ensures the pipeline survives if the
browser tab is accidentally refreshed or the Streamlit script reruns.

Launch:  streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import streamlit as st

from src.config import settings as default_settings
from src.store.graph_store import GraphStore

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Paper Monitoring", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

TIER_COLORS: dict[int | None, str] = {
    1: "#B8860B",   # gold
    2: "#808080",   # silver
    3: "#2196F3",   # blue
    4: "#9E9E9E",   # gray
    5: "#4CAF50",   # green
    None: "#e57373",  # red — classification failed
}


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.process = None          # subprocess.Popen
    st.session_state.event_queue = queue.Queue()
    st.session_state.progress_log: list[str] = []
    st.session_state.final_status = None     # "done" | "error" | None


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _reader_thread(proc: subprocess.Popen, q: queue.Queue) -> None:
    """Read stdout from the pipeline subprocess line-by-line, push to queue."""
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = {"type": "progress", "message": line}
        q.put(event)
    # Process has exited — check return code
    proc.wait()
    if proc.returncode != 0 and q.empty():
        stderr_text = proc.stderr.read() if proc.stderr else ""
        q.put({"type": "error", "message": f"Pipeline exited with code {proc.returncode}. {stderr_text}"})


def _start_pipeline() -> None:
    """Launch the pipeline as a subprocess with --progress."""
    q: queue.Queue = queue.Queue()
    proc = subprocess.Popen(
        [_PYTHON, "-m", "src.pipeline", "--progress"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    reader = threading.Thread(target=_reader_thread, args=(proc, q), daemon=True)
    reader.start()

    st.session_state.process = proc
    st.session_state.event_queue = q
    st.session_state.running = True
    st.session_state.progress_log = []
    st.session_state.final_status = None


def _drain_events() -> None:
    """Drain all queued events and update session state."""
    while True:
        try:
            event = st.session_state.event_queue.get_nowait()
        except queue.Empty:
            break

        etype = event.get("type", "progress")
        msg = event.get("message", "")

        if etype == "done":
            st.session_state.final_status = "done"
            st.session_state.running = False
            st.session_state.progress_log.append(msg)
        elif etype == "error":
            st.session_state.final_status = "error"
            st.session_state.running = False
            st.session_state.progress_log.append(f"ERROR: {msg}")
        else:
            st.session_state.progress_log.append(msg)


def _is_still_running() -> bool:
    """Check whether the subprocess is still alive."""
    proc = st.session_state.process
    if proc is None:
        return False
    return proc.poll() is None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Paper Monitoring")

store = GraphStore(default_settings.db_path)
latest_run = store.get_latest_run()

if latest_run is not None:
    st.caption(
        f"Last run: {latest_run.run_date} — "
        f"{latest_run.papers_classified} classified, "
        f"{latest_run.papers_failed} failed — "
        f"status: **{latest_run.status}**"
    )

# ---------------------------------------------------------------------------
# Run button + progress
# ---------------------------------------------------------------------------

col_btn, col_status = st.columns([1, 3])

with col_btn:
    if st.button(
        "Run Weekly Update" if not st.session_state.running else "Pipeline running...",
        disabled=st.session_state.running,
    ):
        _start_pipeline()
        st.rerun()

# While running, drain events and auto-refresh
if st.session_state.running:
    _drain_events()
    with col_status:
        st.info("Pipeline is running — keep this tab open.")
    if st.session_state.progress_log:
        with st.expander("Progress", expanded=True):
            for line in st.session_state.progress_log:
                st.text(line)
    # Auto-refresh every 2 seconds to pick up new progress
    time.sleep(2)
    st.rerun()

# Show final status after completion
if st.session_state.final_status == "done":
    _drain_events()  # catch any trailing events
    st.success(
        f"Pipeline complete! {st.session_state.progress_log[-1] if st.session_state.progress_log else ''}"
    )
    if st.session_state.progress_log:
        with st.expander("Run log"):
            for line in st.session_state.progress_log:
                st.text(line)
    st.session_state.final_status = None  # clear after displaying

elif st.session_state.final_status == "error":
    _drain_events()
    st.error("Pipeline failed.")
    if st.session_state.progress_log:
        with st.expander("Error details", expanded=True):
            for line in st.session_state.progress_log[-5:]:
                st.text(line)
    st.session_state.final_status = None

# ---------------------------------------------------------------------------
# Tier filter
# ---------------------------------------------------------------------------

st.subheader("Classified Papers")

tier_options = ["T1", "T2", "T3", "T4", "T5", "Failed"]
selected_tiers = st.multiselect("Filter by tier", tier_options, default=tier_options)

# Map selection back to int|None for filtering
_tier_map: dict[str, int | None] = {
    "T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5, "Failed": None,
}
active_tiers = {_tier_map[t] for t in selected_tiers}

# ---------------------------------------------------------------------------
# Paper table
# ---------------------------------------------------------------------------

papers = store.get_all_papers(limit=500)

if not papers:
    st.info("No papers classified yet. Click **Run Weekly Update** to get started.")
else:
    # Filter by tier
    filtered = [
        p for p in papers
        if p["properties"].get("tier") in active_tiers
    ]

    if not filtered:
        st.warning("No papers match the selected tiers.")
    else:
        rows = []
        for p in filtered:
            props = p["properties"]
            tier = props.get("tier")
            tier_label = f"T{tier}" if tier is not None else "FAILED"
            authors = props.get("authors", [])
            first_author = authors[0] if authors else ""
            if len(authors) > 1:
                first_author += " et al."

            rows.append({
                "Tier": tier_label,
                "Title": p["label"][:80] + ("..." if len(p["label"]) > 80 else ""),
                "Authors": first_author,
                "Published": props.get("published_date", ""),
                "Category": props.get("primary_category", ""),
                "HF Upvotes": props.get("hf_upvotes", 0),
                "Confidence": props.get("confidence", ""),
                "arXiv": props.get("arxiv_url", ""),
            })

        st.dataframe(
            rows,
            use_container_width=True,
            column_config={
                "arXiv": st.column_config.LinkColumn("arXiv"),
            },
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

store.close()

st.divider()
st.caption(
    "Built with [Claude Code](https://claude.ai/code) | "
    f"model: `{default_settings.ollama_model}` | "
    f"db: `{default_settings.db_path}`"
)
