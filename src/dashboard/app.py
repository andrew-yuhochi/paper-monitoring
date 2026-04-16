"""Paper Monitoring — Streamlit Dashboard.

Single-page app with two tabs:
  1. Classified Papers — weekly pipeline results (seed papers hidden)
  2. Knowledge Bank — foundational concepts with sources and prerequisites

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

# Ensure the project root is on sys.path so `src.*` imports resolve
# regardless of the working directory Streamlit uses.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.config import settings as default_settings  # noqa: E402
from src.store.graph_store import GraphStore  # noqa: E402

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Paper Monitoring", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.process = None
    st.session_state.event_queue = queue.Queue()
    st.session_state.progress_log: list[str] = []
    st.session_state.final_status = None


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


# ---------------------------------------------------------------------------
# Header + run button
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

col_btn, col_status = st.columns([1, 3])

with col_btn:
    if st.button(
        "Run Weekly Update" if not st.session_state.running else "Pipeline running...",
        disabled=st.session_state.running,
    ):
        _start_pipeline()
        st.rerun()

if st.session_state.running:
    _drain_events()
    with col_status:
        st.info("Pipeline is running — keep this tab open.")
    if st.session_state.progress_log:
        with st.expander("Progress", expanded=True):
            for line in st.session_state.progress_log:
                st.text(line)
    time.sleep(2)
    st.rerun()

if st.session_state.final_status == "done":
    _drain_events()
    st.success(
        f"Pipeline complete! {st.session_state.progress_log[-1] if st.session_state.progress_log else ''}"
    )
    if st.session_state.progress_log:
        with st.expander("Run log"):
            for line in st.session_state.progress_log:
                st.text(line)
    st.session_state.final_status = None
elif st.session_state.final_status == "error":
    _drain_events()
    st.error("Pipeline failed.")
    if st.session_state.progress_log:
        with st.expander("Error details", expanded=True):
            for line in st.session_state.progress_log[-5:]:
                st.text(line)
    st.session_state.final_status = None

# ---------------------------------------------------------------------------
# Tabs: Classified Papers | Knowledge Bank
# ---------------------------------------------------------------------------

tab_papers, tab_concepts = st.tabs(["Classified Papers", "Knowledge Bank"])

# ---------------------------------------------------------------------------
# Tab 1: Classified Papers (seed papers hidden)
# ---------------------------------------------------------------------------

with tab_papers:
    all_papers = store.get_all_papers(limit=500)

    # Only show papers from the weekly pipeline (have run_date in properties).
    # Seed papers don't have run_date — they're knowledge bank sources, not digest entries.
    pipeline_papers = [
        p for p in all_papers
        if "run_date" in p["properties"]
    ]

    if not pipeline_papers:
        st.info("No papers classified yet. Click **Run Weekly Update** to get started.")
    else:
        tier_options = ["T1", "T2", "T3", "T4", "T5", "Failed"]
        selected_tiers = st.multiselect("Filter by tier", tier_options, default=tier_options)
        _tier_map: dict[str, int | None] = {
            "T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5, "Failed": None,
        }
        active_tiers = {_tier_map[t] for t in selected_tiers}

        filtered = [
            p for p in pipeline_papers
            if p["properties"].get("tier") in active_tiers
        ]

        if not filtered:
            st.warning("No papers match the selected tiers.")
        else:
            for p in filtered:
                props = p["properties"]
                tier = props.get("tier")
                tier_label = f"T{tier}" if tier is not None else "FAILED"
                summary = props.get("summary", "No summary available.")
                contributions = props.get("key_contributions", [])
                concepts = p.get("linked_concepts", [])
                arxiv_url = props.get("arxiv_url", "")

                with st.container(border=True):
                    col_tier, col_content = st.columns([1, 12])
                    with col_tier:
                        st.markdown(f"### {tier_label}")
                    with col_content:
                        st.markdown(f"**{p['label']}**")
                        st.write(summary)
                        if contributions:
                            st.markdown("**Key contributions:** " + " | ".join(contributions))
                        if concepts:
                            st.markdown("**Related concepts:** " + ", ".join(f"`{c}`" for c in concepts))
                        if arxiv_url:
                            st.markdown(f"[arXiv]({arxiv_url})")

# ---------------------------------------------------------------------------
# Tab 2: Knowledge Bank (concepts)
# ---------------------------------------------------------------------------

with tab_concepts:
    concepts = store.get_all_concepts()

    if not concepts:
        st.info("Knowledge bank is empty. Run the seed pipeline first: `python -m src.seed`")
    else:
        st.metric("Total concepts", len(concepts))

        search = st.text_input("Search concepts", placeholder="e.g. attention, transformer, backpropagation")

        display = concepts
        if search:
            query = search.lower()
            display = [
                c for c in concepts
                if query in c["label"].lower()
                or query in c["properties"].get("description", "").lower()
            ]

        if not display:
            st.warning(f"No concepts match '{search}'.")
        else:
            for c in display:
                props = c["properties"]
                desc = props.get("description", "No description.")
                tags = props.get("domain_tags", [])
                sources = c["source_papers"]
                prereqs = c["prerequisites"]

                with st.expander(f"**{c['label']}**  {'  '.join(f'`{t}`' for t in tags)}"):
                    st.write(desc)
                    if sources:
                        st.caption(f"Source papers: {', '.join(sources)}")
                    if prereqs:
                        st.caption(f"Prerequisites: {', '.join(prereqs)}")

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
