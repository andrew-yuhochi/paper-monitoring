"""Paper Monitoring — Streamlit Dashboard.

Single-page app with three tabs:
  1. Classified Papers — weekly pipeline results (seed papers hidden)
  2. Knowledge Bank — foundational concepts with sources and prerequisites
  3. Graph — interactive knowledge graph visualization

Launch:  streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports resolve
# regardless of the working directory Streamlit uses.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402
from streamlit_agraph import Config as AgraphConfig  # noqa: E402
from streamlit_agraph import Edge as AgraphEdge  # noqa: E402
from streamlit_agraph import Node as AgraphNode  # noqa: E402
from streamlit_agraph import agraph  # noqa: E402

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

# Node type -> color mapping
_NODE_COLORS: dict[str, str] = {
    "problem": "#E74C3C",     # red
    "technique": "#3498DB",   # blue
    "concept": "#2ECC71",     # green
    "category": "#95A5A6",    # gray
    "paper": "#F1C40F",       # yellow
}

_NODE_SIZES: dict[str, int] = {
    "problem": 30,
    "technique": 28,
    "concept": 22,
    "category": 20,
    "paper": 18,
}

# Edge relationship type -> color
_EDGE_COLORS: dict[str, str] = {
    "PREREQUISITE_OF": "#95A5A6",
    "BUILDS_ON": "#2ECC71",
    "INTRODUCES": "#F1C40F",
    "ADDRESSES": "#E74C3C",
    "BASELINE_OF": "#8E44AD",
    "ALTERNATIVE_TO": "#E67E22",
    "BELONGS_TO": "#BDC3C7",
}


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.process = None
    st.session_state.event_queue = queue.Queue()
    st.session_state.progress_log: list[str] = []
    st.session_state.final_status = None

if "graph_center_node" not in st.session_state:
    st.session_state.graph_center_node = None


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
# Tabs: Classified Papers | Knowledge Bank | Graph
# ---------------------------------------------------------------------------

tab_papers, tab_concepts, tab_graph = st.tabs(["Classified Papers", "Knowledge Bank", "Graph"])

# ---------------------------------------------------------------------------
# Tab 1: Classified Papers (seed papers hidden)
# ---------------------------------------------------------------------------

with tab_papers:
    all_papers = store.get_all_papers(limit=500)

    # Only show papers from the weekly pipeline (have run_date in properties).
    # Seed papers don't have run_date — they're knowledge bank sources, not digest entries.
    # Default: last 30 days only.
    cutoff = (date.today() - timedelta(days=30)).isoformat()
    pipeline_papers = [
        p for p in all_papers
        if "run_date" in p["properties"]
        and p["properties"].get("run_date", "") >= cutoff
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
                        published = props.get("published_date", "")
                        run_dt = props.get("run_date", "")
                        date_parts = []
                        if published:
                            date_parts.append(f"Published: {published}")
                        if run_dt:
                            date_parts.append(f"Classified: {run_dt}")
                        if date_parts:
                            st.caption(" · ".join(date_parts))
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
                    elif props.get("seeded_from"):
                        st.caption(f"Source: {props['seeded_from']}")
                    if prereqs:
                        st.caption(f"Prerequisites: {', '.join(prereqs)}")

# ---------------------------------------------------------------------------
# Tab 3: Graph — interactive knowledge graph visualization
# ---------------------------------------------------------------------------

with tab_graph:
    # --- Node type filter ---
    st.markdown("##### Node type filter")
    filter_cols = st.columns(5)
    type_labels = ["problem", "technique", "concept", "category", "paper"]
    visible_types: set[str] = set()
    for i, ntype in enumerate(type_labels):
        with filter_cols[i]:
            color_dot = f'<span style="color:{_NODE_COLORS[ntype]}">&#9679;</span>'
            if st.checkbox(
                f"{ntype.capitalize()}",
                value=True,
                key=f"graph_filter_{ntype}",
            ):
                visible_types.add(ntype)

    # --- Legend ---
    legend_parts = [
        f'<span style="color:{_NODE_COLORS[t]}">&#9679;</span> {t.capitalize()}'
        for t in type_labels
    ]
    st.markdown(" &nbsp;&nbsp; ".join(legend_parts), unsafe_allow_html=True)

    # --- Node selector for centering ---
    all_nodes_by_type = {}
    for ntype in visible_types:
        nodes_of_type = store.get_nodes_by_type(ntype)
        for n in nodes_of_type:
            all_nodes_by_type[f"{n.label} ({ntype})"] = n.id

    if not all_nodes_by_type:
        st.info("No nodes in the knowledge bank yet. Run the seed pipeline or add nodes manually.")
    else:
        # Build select options sorted alphabetically
        node_options = sorted(all_nodes_by_type.keys())

        # Determine the current center
        center_node_id = st.session_state.graph_center_node
        # Find the display label for the current center
        center_label = None
        for label, nid in all_nodes_by_type.items():
            if nid == center_node_id:
                center_label = label
                break

        selected_label = st.selectbox(
            "Center on node",
            options=node_options,
            index=node_options.index(center_label) if center_label in node_options else 0,
            key="graph_node_selector",
        )
        center_node_id = all_nodes_by_type.get(selected_label)

        if center_node_id:
            st.session_state.graph_center_node = center_node_id

            # Fetch neighborhood
            neighborhood = store.get_node_neighborhood(center_node_id, depth=1)
            graph_nodes = neighborhood["nodes"]
            graph_edges = neighborhood["edges"]

            # Filter nodes by visible types
            graph_nodes = [n for n in graph_nodes if n["node_type"] in visible_types]
            visible_node_ids = {n["id"] for n in graph_nodes}

            # Filter edges to only connect visible nodes
            graph_edges = [
                e for e in graph_edges
                if e["source_id"] in visible_node_ids and e["target_id"] in visible_node_ids
            ]

            if not graph_nodes:
                st.warning("No visible nodes after filtering. Try enabling more node types.")
            else:
                # Build streamlit-agraph nodes
                agraph_nodes = []
                for n in graph_nodes:
                    ntype = n["node_type"]
                    is_center = n["id"] == center_node_id
                    agraph_nodes.append(
                        AgraphNode(
                            id=n["id"],
                            label=n["label"],
                            color=_NODE_COLORS.get(ntype, "#CCCCCC"),
                            size=_NODE_SIZES.get(ntype, 20) + (8 if is_center else 0),
                            title=f"[{ntype}] {n['label']}",
                            shape="dot",
                        )
                    )

                # Build streamlit-agraph edges
                agraph_edges = []
                for e in graph_edges:
                    rtype = e["relationship_type"]
                    agraph_edges.append(
                        AgraphEdge(
                            source=e["source_id"],
                            target=e["target_id"],
                            label=rtype,
                            color=_EDGE_COLORS.get(rtype, "#CCCCCC"),
                        )
                    )

                # Render graph
                config = AgraphConfig(
                    height=600,
                    width=900,
                    directed=True,
                    physics=True,
                    hierarchical=False,
                )

                clicked_node = agraph(
                    nodes=agraph_nodes,
                    edges=agraph_edges,
                    config=config,
                )

                # Handle click to re-center
                if clicked_node and clicked_node != center_node_id:
                    st.session_state.graph_center_node = clicked_node
                    st.rerun()

                # --- Node details panel ---
                center = store.get_node(center_node_id)
                if center:
                    st.markdown("---")
                    st.markdown(f"**Selected: {center.label}** ({center.node_type})")
                    props = center.properties
                    if props.get("description"):
                        st.write(props["description"])
                    if props.get("approach"):
                        st.write(f"**Approach:** {props['approach']}")
                    if props.get("domain_tags"):
                        st.write("**Tags:** " + ", ".join(f"`{t}`" for t in props["domain_tags"]))
                    if props.get("edited_by"):
                        st.caption(f"Edited by: {props['edited_by']}")

                    # Show connected edges
                    edges_from = store.get_edges_from(center_node_id)
                    edges_to = store.get_edges_to(center_node_id)
                    if edges_from or edges_to:
                        with st.expander(f"Connections ({len(edges_from) + len(edges_to)})"):
                            for e in edges_from:
                                target = store.get_node(e.target_id)
                                target_label = target.label if target else e.target_id
                                st.text(f"  -> [{e.relationship_type}] -> {target_label}")
                            for e in edges_to:
                                source = store.get_node(e.source_id)
                                source_label = source.label if source else e.source_id
                                st.text(f"  <- [{e.relationship_type}] <- {source_label}")

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
