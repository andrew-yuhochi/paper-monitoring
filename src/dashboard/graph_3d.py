"""3D WebGL knowledge graph renderer.

Returns a self-contained HTML string that embeds a 3d-force-graph
visualization designed to be injected via ``st.components.v1.html()``.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_NODE_COLORS: dict[str, str] = {
    "problem": "#E74C3C",
    "technique": "#3498DB",
    "concept": "#2ECC71",
    "category": "#95A5A6",
    "paper": "#F1C40F",
}

_NODE_SIZES: dict[str, int] = {
    "problem": 10,
    "technique": 8,
    "concept": 5,
    "category": 6,
    "paper": 6,
}

_GRAPH_CDN = "https://unpkg.com/3d-force-graph@1.77.0/dist/3d-force-graph.min.js"
# SpriteText provides always-visible 3D text labels that scale with camera distance
_SPRITE_CDN = "https://unpkg.com/three-spritetext@1.9.0/dist/three-spritetext.min.js"


def _safe_json(obj: object) -> str:
    """Serialize to JSON and escape </script> to prevent HTML injection."""
    raw = json.dumps(obj, ensure_ascii=False)
    return raw.replace("</script>", "<\\/script>")


def render_graph_3d(
    nodes: list[dict],
    edges: list[dict],
    height: int = 750,
) -> str:
    """Return a self-contained HTML string rendering a 3D force-directed graph.

    Parameters
    ----------
    nodes:
        Each dict must have keys ``id``, ``node_type``, ``label``, ``properties``.
    edges:
        Each dict must have keys ``source_id``, ``target_id``,
        ``relationship_type``, ``weight``, ``properties``.
    height:
        Pixel height of the graph container.
    """
    enriched_nodes: list[dict] = []
    for n in nodes:
        ntype = n.get("node_type", "concept")
        enriched_nodes.append({
            "id": n["id"],
            "node_type": ntype,
            "label": n.get("label", n["id"]),
            "color": _NODE_COLORS.get(ntype, "#CCCCCC"),
            "val": _NODE_SIZES.get(ntype, 5),
            "properties": n.get("properties", {}),
        })

    # _sid/_tid preserve the original string IDs — 3d-force-graph mutates
    # edge.source and edge.target from strings to node objects after the first
    # graphData() call, which breaks BFS and filter lookups on re-render.
    enriched_edges: list[dict] = []
    for e in edges:
        enriched_edges.append({
            "source": e["source_id"],
            "target": e["target_id"],
            "_sid": e["source_id"],
            "_tid": e["target_id"],
            "relationship_type": e.get("relationship_type", ""),
            "weight": e.get("weight", 1.0),
            "properties": e.get("properties", {}),
        })

    nodes_json = _safe_json(enriched_nodes)
    edges_json = _safe_json(enriched_edges)
    node_types_json = _safe_json(list(_NODE_COLORS.keys()))
    node_colors_json = _safe_json(_NODE_COLORS)

    return f"""<!DOCTYPE html>
<html>
<head>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #0a0a1a; overflow: hidden; }}
    #graph {{ width: 100%; height: {height}px; position: relative; }}
    #controls {{
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 10;
      color: white;
      font-family: sans-serif;
      font-size: 13px;
      background: rgba(0,0,0,0.7);
      padding: 10px 14px;
      border-radius: 6px;
      max-width: 220px;
      line-height: 1.8;
    }}
    #controls label {{ display: flex; align-items: center; gap: 6px; cursor: pointer; }}
    #controls input[type=range] {{ width: 120px; }}
    #center-row {{ display: none; margin-top: 4px; font-size: 11px; color: #aaa; }}
    #center-row span {{ word-break: break-word; flex: 1; }}
    #clear-btn {{
      cursor: pointer; background: rgba(255,255,255,0.15);
      border: none; color: white; border-radius: 3px;
      padding: 1px 6px; font-size: 11px; flex-shrink: 0;
    }}
    #clear-btn:hover {{ background: rgba(255,255,255,0.3); }}
  </style>
</head>
<body>
  <div id="controls">
    <div>
      <span>Depth: <b id="hop-val">2</b> hops</span><br>
      <input type="range" id="hop-slider" min="1" max="5" value="2">
    </div>
    <div id="center-row">
      <span id="center-label"></span>
      <button id="clear-btn">✕ All</button>
    </div>
    <div style="margin-top:8px; border-top:1px solid #444; padding-top:6px;">
      <b>Node types</b>
      <div id="type-filters"></div>
    </div>
  </div>
  <div id="graph"></div>

  <script src="{_GRAPH_CDN}"></script>
  <script src="{_SPRITE_CDN}"></script>
  <script>
    const allNodes = {nodes_json};
    const allEdges = {edges_json};
    const NODE_TYPES = {node_types_json};
    const NODE_COLORS = {node_colors_json};

    // ── State ────────────────────────────────────────────────────────────────
    let selectedNodeId = null;
    let hopDepth = 2;
    let visibleTypes = new Set(NODE_TYPES);

    // ── Type-filter checkboxes ───────────────────────────────────────────────
    const filterContainer = document.getElementById('type-filters');
    NODE_TYPES.forEach(t => {{
      const lbl = document.createElement('label');
      const cb  = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = true;
      cb.dataset.type = t;
      cb.addEventListener('change', () => {{
        if (cb.checked) visibleTypes.add(t); else visibleTypes.delete(t);
        applyFilters();
      }});
      const dot = document.createElement('span');
      dot.style.cssText = 'display:inline-block;width:10px;height:10px;border-radius:50%;background:'
        + (NODE_COLORS[t] || '#ccc') + ';flex-shrink:0;';
      lbl.appendChild(cb);
      lbl.appendChild(dot);
      lbl.appendChild(document.createTextNode(' ' + t.charAt(0).toUpperCase() + t.slice(1)));
      filterContainer.appendChild(lbl);
    }});

    // ── BFS: _sid/_tid are the immutable original string IDs ─────────────────
    function bfsNeighborhood(centerId, depth) {{
      const adjOut = {{}};
      const adjIn  = {{}};
      allEdges.forEach(e => {{
        if (!adjOut[e._sid]) adjOut[e._sid] = [];
        adjOut[e._sid].push(e._tid);
        if (!adjIn[e._tid]) adjIn[e._tid] = [];
        adjIn[e._tid].push(e._sid);
      }});
      const visited = new Set([centerId]);
      let frontier = [centerId];
      for (let d = 0; d < depth; d++) {{
        const next = [];
        frontier.forEach(nid => {{
          (adjOut[nid] || []).forEach(t => {{ if (!visited.has(t)) {{ visited.add(t); next.push(t); }} }});
          (adjIn[nid]  || []).forEach(s => {{ if (!visited.has(s)) {{ visited.add(s); next.push(s); }} }});
        }});
        frontier = next;
        if (!frontier.length) break;
      }}
      return visited;
    }}

    // ── Compute visible subgraph ─────────────────────────────────────────────
    function computeVisible() {{
      const hopSet = selectedNodeId !== null
        ? bfsNeighborhood(selectedNodeId, hopDepth)
        : null;

      const visibleNodes = allNodes.filter(n => {{
        if (!visibleTypes.has(n.node_type)) return false;
        if (hopSet !== null && !hopSet.has(n.id)) return false;
        return true;
      }});
      const visibleIds = new Set(visibleNodes.map(n => n.id));

      // Use _sid/_tid — immune to 3d-force-graph's source/target mutation
      const visibleEdges = allEdges.filter(e =>
        visibleIds.has(e._sid) && visibleIds.has(e._tid)
      );
      return {{ nodes: visibleNodes, links: visibleEdges }};
    }}

    // ── Graph init (deferred for iframe layout) ──────────────────────────────
    const graphEl = document.getElementById('graph');
    let graph;

    function initGraph() {{
      const w = graphEl.offsetWidth  || window.innerWidth  || 800;
      const h = graphEl.offsetHeight || {height};

      graph = ForceGraph3D({{ rendererConfig: {{ antialias: true }} }})(graphEl)
        .width(w)
        .height(h)
        .backgroundColor('#0a0a1a')
        .nodeId('id')
        .nodeLabel('label')
        .nodeColor(n => n.color || '#cccccc')
        .nodeVal(n => n.val || 5)
        .linkSource('source')
        .linkTarget('target')
        .linkLabel(l => l.relationship_type || '')
        .linkColor(() => '#888888')
        .linkOpacity(0.6)
        .linkWidth(0.8)
        .linkCurvature(0.15)
        .graphData(computeVisible());

      // Always-visible node labels via SpriteText (scale with camera distance)
      if (typeof SpriteText !== 'undefined') {{
        graph
          .nodeThreeObject(node => {{
            const sprite = new SpriteText(node.label);
            sprite.color = '#ffffff';
            sprite.textHeight = 5;
            sprite.backgroundColor = 'rgba(0,0,0,0.45)';
            sprite.padding = 2;
            sprite.borderRadius = 3;
            return sprite;
          }})
          .nodeThreeObjectExtend(true);
      }}

      // Node click: fly camera to node + apply hop filter
      graph.onNodeClick(node => {{
        selectedNodeId = node.id;
        document.getElementById('center-label').textContent = node.label;
        document.getElementById('center-row').style.display = 'flex';

        // Place camera at fixed distance along origin→node axis
        const distance = 120;
        const mag = Math.hypot(node.x || 0.1, node.y || 0.1, node.z || 0.1);
        const ratio = 1 + distance / mag;
        graph.cameraPosition(
          {{ x: node.x * ratio, y: node.y * ratio, z: node.z * ratio }},
          node,
          1000
        );
        applyFilters();
      }});

      // Resize
      window.addEventListener('resize', () => {{
        graph.width(graphEl.offsetWidth).height(graphEl.offsetHeight);
      }});
    }}

    // ── Filter helper ────────────────────────────────────────────────────────
    function applyFilters() {{
      if (graph) graph.graphData(computeVisible());
    }}

    // ── Hop slider ───────────────────────────────────────────────────────────
    document.getElementById('hop-slider').addEventListener('input', e => {{
      hopDepth = parseInt(e.target.value, 10);
      document.getElementById('hop-val').textContent = hopDepth;
      if (selectedNodeId !== null) applyFilters();
    }});

    // ── Clear selection ──────────────────────────────────────────────────────
    document.getElementById('clear-btn').addEventListener('click', () => {{
      selectedNodeId = null;
      document.getElementById('center-row').style.display = 'none';
      applyFilters();
    }});

    setTimeout(initGraph, 50);
  </script>
</body>
</html>"""
