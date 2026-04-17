"""Paper Monitoring — Streamlit Dashboard.

Single-page app with three tabs:
  1. Classified Papers — weekly pipeline results (seed papers hidden)
  2. Knowledge Bank — foundational concepts with sources and prerequisites
  3. Graph — interactive, editable knowledge graph visualization

Launch:  streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import logging
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
from src.utils.normalize import normalize_concept_name  # noqa: E402

logger = logging.getLogger(__name__)

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

_ALL_RELATIONSHIP_TYPES = list(_EDGE_COLORS.keys())

# Property fields by node type (for the editing form)
_NODE_PROPERTY_FIELDS: dict[str, list[str]] = {
    "problem": ["description"],
    "technique": ["approach", "innovation_type", "practical_relevance", "limitations"],
    "concept": ["description", "domain_tags"],
    "category": ["description"],
    "paper": ["description"],
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
if "confirm_delete_node" not in st.session_state:
    st.session_state.confirm_delete_node = False


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
# Tab 3: Graph — interactive, editable knowledge graph visualization
# ---------------------------------------------------------------------------

with tab_graph:
    # --- Node type filter ---
    st.markdown("##### Node type filter")
    filter_cols = st.columns(5)
    type_labels = ["problem", "technique", "concept", "category", "paper"]
    visible_types: set[str] = set()
    for i, ntype in enumerate(type_labels):
        with filter_cols[i]:
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

    # Even when the graph is empty, show the "Add Node" form
    graph_col, edit_col = st.columns([3, 1])

    with graph_col:
        if not all_nodes_by_type:
            st.info("No nodes in the knowledge bank yet. Use the editing panel to add nodes.")
        else:
            # Build select options sorted alphabetically
            node_options = sorted(all_nodes_by_type.keys())

            # Determine the current center
            center_node_id = st.session_state.graph_center_node
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
                        edited_by = n.get("properties", {}).get("edited_by", "llm")
                        # Solid border for user-edited, dashed for LLM
                        border_width = 3 if is_center else 1
                        agraph_nodes.append(
                            AgraphNode(
                                id=n["id"],
                                label=n["label"],
                                color=_NODE_COLORS.get(ntype, "#CCCCCC"),
                                size=_NODE_SIZES.get(ntype, 20) + (8 if is_center else 0),
                                title=f"[{ntype}] {n['label']}\nedited by: {edited_by}",
                                shape="dot",
                                borderWidth=border_width,
                                borderWidthSelected=3,
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

    # --- Editing panel (right column) ---
    with edit_col:
        st.markdown("#### Edit Graph")

        # =====================================================================
        # ADD NODE
        # =====================================================================
        with st.expander("Add Node", expanded=False):
            new_node_type = st.selectbox(
                "Type",
                options=type_labels,
                key="add_node_type",
            )
            new_node_label = st.text_input("Label", key="add_node_label")

            # Type-specific property fields
            new_props: dict = {}
            fields = _NODE_PROPERTY_FIELDS.get(new_node_type, ["description"])
            for field in fields:
                if field == "domain_tags":
                    val = st.text_input(
                        "Domain tags (comma-separated)",
                        key=f"add_node_{field}",
                    )
                    if val.strip():
                        new_props[field] = [t.strip() for t in val.split(",") if t.strip()]
                elif field == "innovation_type":
                    new_props[field] = st.selectbox(
                        "Innovation type",
                        options=["architecture", "problem_framing", "loss_trick",
                                 "eval_methodology", "dataset", "training_technique"],
                        key=f"add_node_{field}",
                    )
                else:
                    val = st.text_area(field.replace("_", " ").capitalize(), key=f"add_node_{field}", height=68)
                    if val.strip():
                        new_props[field] = val.strip()

            if st.button("Create Node", key="btn_add_node", disabled=not new_node_label.strip()):
                label = new_node_label.strip()
                node_id = f"{new_node_type}:{normalize_concept_name(label)}"
                new_props["edited_by"] = "user"
                try:
                    store.add_node(node_id, new_node_type, label, new_props)
                    logger.info("User created node %s (%s)", node_id, new_node_type)
                    st.session_state.graph_center_node = node_id
                    st.rerun()
                except ValueError:
                    st.error(f"Node `{node_id}` already exists.")

        # =====================================================================
        # ADD EDGE
        # =====================================================================
        with st.expander("Add Edge", expanded=False):
            # Build a list of all nodes for source/target selection
            all_node_labels_for_edges: dict[str, str] = {}
            for ntype in type_labels:
                for n in store.get_nodes_by_type(ntype):
                    all_node_labels_for_edges[f"{n.label} ({ntype})"] = n.id

            if len(all_node_labels_for_edges) < 2:
                st.caption("Need at least 2 nodes to create an edge.")
            else:
                edge_node_options = sorted(all_node_labels_for_edges.keys())
                edge_source_label = st.selectbox("Source", options=edge_node_options, key="add_edge_source")
                edge_target_label = st.selectbox("Target", options=edge_node_options, key="add_edge_target")
                edge_rel_type = st.selectbox("Relationship", options=_ALL_RELATIONSHIP_TYPES, key="add_edge_rel")

                source_id = all_node_labels_for_edges.get(edge_source_label, "")
                target_id = all_node_labels_for_edges.get(edge_target_label, "")

                if st.button("Create Edge", key="btn_add_edge", disabled=(source_id == target_id)):
                    try:
                        store.add_edge(
                            source_id, target_id, edge_rel_type,
                            properties={"edited_by": "user"},
                        )
                        logger.info(
                            "User created edge %s -[%s]-> %s",
                            source_id, edge_rel_type, target_id,
                        )
                        st.rerun()
                    except ValueError:
                        st.error("Edge already exists.")

        # =====================================================================
        # EDIT SELECTED NODE / DELETE NODE / DELETE EDGES
        # =====================================================================
        center_node_id_edit = st.session_state.graph_center_node
        if center_node_id_edit:
            center = store.get_node(center_node_id_edit)
            if center:
                st.markdown("---")
                st.markdown(f"**{center.label}**")
                edited_by = center.properties.get("edited_by", "llm")
                st.caption(f"{center.node_type} · edited by: {edited_by}")

                # --- Edit properties ---
                with st.expander("Edit Properties", expanded=False):
                    edit_label = st.text_input(
                        "Label",
                        value=center.label,
                        key="edit_node_label",
                    )
                    edit_props: dict = {}
                    fields = _NODE_PROPERTY_FIELDS.get(center.node_type, ["description"])
                    for field in fields:
                        current_val = center.properties.get(field, "")
                        if field == "domain_tags":
                            current_tags = ", ".join(current_val) if isinstance(current_val, list) else str(current_val)
                            val = st.text_input(
                                "Domain tags (comma-separated)",
                                value=current_tags,
                                key=f"edit_node_{field}",
                            )
                            edit_props[field] = [t.strip() for t in val.split(",") if t.strip()]
                        elif field == "innovation_type":
                            options = ["architecture", "problem_framing", "loss_trick",
                                       "eval_methodology", "dataset", "training_technique"]
                            idx = options.index(current_val) if current_val in options else 0
                            edit_props[field] = st.selectbox(
                                "Innovation type",
                                options=options,
                                index=idx,
                                key=f"edit_node_{field}",
                            )
                        else:
                            val = st.text_area(
                                field.replace("_", " ").capitalize(),
                                value=str(current_val),
                                key=f"edit_node_{field}",
                                height=68,
                            )
                            edit_props[field] = val.strip()

                    if st.button("Save Changes", key="btn_save_node"):
                        edit_props["edited_by"] = "user"
                        store.update_node_properties(center_node_id_edit, edit_props)
                        # Update label if changed
                        if edit_label.strip() and edit_label.strip() != center.label:
                            store.upsert_node(
                                center_node_id_edit,
                                center.node_type,
                                edit_label.strip(),
                                {**center.properties, **edit_props},
                            )
                        logger.info("User edited node %s", center_node_id_edit)
                        st.rerun()

                # --- Delete node ---
                if st.session_state.confirm_delete_node:
                    st.warning(f"Delete **{center.label}** and all its edges?")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes, delete", key="btn_confirm_del_node"):
                            removed = store.delete_node(center_node_id_edit)
                            logger.info(
                                "User deleted node %s (%d edges removed)",
                                center_node_id_edit, removed,
                            )
                            st.session_state.graph_center_node = None
                            st.session_state.confirm_delete_node = False
                            st.rerun()
                    with col_no:
                        if st.button("Cancel", key="btn_cancel_del_node"):
                            st.session_state.confirm_delete_node = False
                            st.rerun()
                else:
                    if st.button("Delete Node", key="btn_delete_node", type="secondary"):
                        st.session_state.confirm_delete_node = True
                        st.rerun()

                # --- Connected edges with delete ---
                edges_from = store.get_edges_from(center_node_id_edit)
                edges_to = store.get_edges_to(center_node_id_edit)
                total_edges = len(edges_from) + len(edges_to)
                if total_edges > 0:
                    with st.expander(f"Connections ({total_edges})"):
                        for e in edges_from:
                            target = store.get_node(e.target_id)
                            target_label = target.label if target else e.target_id
                            ecol_info, ecol_del = st.columns([4, 1])
                            with ecol_info:
                                st.text(f"-> [{e.relationship_type}] -> {target_label}")
                            with ecol_del:
                                edge_key = f"del_e_{e.source_id}_{e.target_id}_{e.relationship_type}"
                                if st.button("X", key=edge_key, help="Delete this edge"):
                                    store.delete_edge(e.source_id, e.target_id, e.relationship_type)
                                    logger.info(
                                        "User deleted edge %s -[%s]-> %s",
                                        e.source_id, e.relationship_type, e.target_id,
                                    )
                                    st.rerun()
                        for e in edges_to:
                            source = store.get_node(e.source_id)
                            source_label = source.label if source else e.source_id
                            ecol_info, ecol_del = st.columns([4, 1])
                            with ecol_info:
                                st.text(f"<- [{e.relationship_type}] <- {source_label}")
                            with ecol_del:
                                edge_key = f"del_e_{e.source_id}_{e.target_id}_{e.relationship_type}"
                                if st.button("X", key=edge_key, help="Delete this edge"):
                                    store.delete_edge(e.source_id, e.target_id, e.relationship_type)
                                    logger.info(
                                        "User deleted edge %s -[%s]-> %s",
                                        e.source_id, e.relationship_type, e.target_id,
                                    )
                                    st.rerun()

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
