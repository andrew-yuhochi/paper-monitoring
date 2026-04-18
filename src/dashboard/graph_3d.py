"""3D WebGL knowledge graph renderer.

Returns a self-contained HTML string that embeds a 3d-force-graph
visualization. The HTML is designed to be injected via
``st.components.v1.html()``.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# Node color and size by type — mirrors the app.py constants
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

_THREE_CDN = "https://unpkg.com/three@0.166.0/build/three.min.js"
_GRAPH_CDN = "https://unpkg.com/3d-force-graph@1.77.0/dist/3d-force-graph.min.js"


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
        Each dict must have keys ``id``, ``node_type``, ``label``,
        ``properties``.
    edges:
        Each dict must have keys ``source_id``, ``target_id``,
        ``relationship_type``, ``weight``, ``properties``.
    height:
        Pixel height of the graph container.
    """
    # Enrich nodes with color and size for the renderer
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

    # Remap source_id/target_id -> source/target for 3d-force-graph
    enriched_edges: list[dict] = []
    for e in edges:
        enriched_edges.append({
            "source": e["source_id"],
            "target": e["target_id"],
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
    #graph {{ width: 100%; height: {height}px; }}
    #controls {{
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 10;
      color: white;
      font-family: sans-serif;
      font-size: 13px;
      background: rgba(0,0,0,0.65);
      padding: 10px 14px;
      border-radius: 6px;
      max-width: 220px;
      line-height: 1.7;
    }}
    #controls label {{ display: flex; align-items: center; gap: 6px; cursor: pointer; }}
    #controls input[type=range] {{ width: 120px; }}
    #center-label {{ margin-top: 6px; font-size: 11px; color: #aaa; word-break: break-word; }}
  </style>
</head>
<body>
  <div id="controls">
    <div>
      <label>
        <input type="checkbox" id="fly-mode"> Fly mode (WASD)
      </label>
    </div>
    <div style="margin-top:6px;">
      <span>Depth: <b id="hop-val">2</b> hops</span><br>
      <input type="range" id="hop-slider" min="1" max="5" value="2">
    </div>
    <div id="center-label"></div>
    <div style="margin-top:8px; border-top:1px solid #444; padding-top:6px;">
      <b>Node types</b>
      <div id="type-filters"></div>
    </div>
  </div>
  <div id="graph"></div>

  <script src="{_THREE_CDN}"></script>
  <script src="{_GRAPH_CDN}"></script>
  <script>
    const allNodes = {nodes_json};
    const allEdges = {edges_json};
    const NODE_TYPES = {node_types_json};
    const NODE_COLORS = {node_colors_json};

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    let selectedNodeId = null;
    let hopDepth = 2;
    let visibleTypes = new Set(NODE_TYPES);

    // -----------------------------------------------------------------------
    // Build type-filter checkboxes dynamically
    // -----------------------------------------------------------------------
    const filterContainer = document.getElementById('type-filters');
    NODE_TYPES.forEach(t => {{
      const lbl = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = true;
      cb.dataset.type = t;
      cb.addEventListener('change', () => {{
        if (cb.checked) visibleTypes.add(t); else visibleTypes.delete(t);
        applyFilters();
      }});
      const dot = document.createElement('span');
      dot.style.cssText = 'display:inline-block;width:10px;height:10px;border-radius:50%;background:' + (NODE_COLORS[t] || '#ccc') + ';flex-shrink:0;';
      lbl.appendChild(cb);
      lbl.appendChild(dot);
      lbl.appendChild(document.createTextNode(' ' + t.charAt(0).toUpperCase() + t.slice(1)));
      filterContainer.appendChild(lbl);
    }});

    // -----------------------------------------------------------------------
    // BFS to compute visible node set from selected center up to N hops
    // -----------------------------------------------------------------------
    function bfsNeighborhood(centerId, depth) {{
      const adjOut = {{}};
      const adjIn = {{}};
      allEdges.forEach(e => {{
        if (!adjOut[e.source]) adjOut[e.source] = [];
        adjOut[e.source].push(e.target);
        if (!adjIn[e.target]) adjIn[e.target] = [];
        adjIn[e.target].push(e.source);
      }});

      const visited = new Set([centerId]);
      let frontier = [centerId];
      for (let d = 0; d < depth; d++) {{
        const next = [];
        frontier.forEach(nid => {{
          (adjOut[nid] || []).forEach(t => {{ if (!visited.has(t)) {{ visited.add(t); next.push(t); }} }});
          (adjIn[nid] || []).forEach(s => {{ if (!visited.has(s)) {{ visited.add(s); next.push(s); }} }});
        }});
        frontier = next;
        if (frontier.length === 0) break;
      }}
      return visited;
    }}

    // -----------------------------------------------------------------------
    // Compute the currently visible node and edge sets
    // -----------------------------------------------------------------------
    function computeVisible() {{
      let hopSet = null;
      if (selectedNodeId !== null) {{
        hopSet = bfsNeighborhood(selectedNodeId, hopDepth);
      }}

      const nodeById = {{}};
      allNodes.forEach(n => {{ nodeById[n.id] = n; }});

      const visibleNodes = allNodes.filter(n => {{
        if (!visibleTypes.has(n.node_type)) return false;
        if (hopSet !== null && !hopSet.has(n.id)) return false;
        return true;
      }});

      const visibleNodeIds = new Set(visibleNodes.map(n => n.id));

      const visibleEdges = allEdges.filter(e =>
        visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target)
      );

      return {{ nodes: visibleNodes, links: visibleEdges }};
    }}

    // -----------------------------------------------------------------------
    // Graph initialisation
    // -----------------------------------------------------------------------
    const graphEl = document.getElementById('graph');
    const graph = ForceGraph3D({{ rendererConfig: {{ antialias: true }} }})(graphEl)
      .backgroundColor('#0a0a1a')
      .nodeId('id')
      .nodeLabel('label')
      .nodeColor(n => n.color || '#cccccc')
      .nodeVal(n => n.val || 5)
      .linkSource('source')
      .linkTarget('target')
      .linkLabel(l => l.relationship_type || '')
      .linkColor(() => '#aaaaaa')
      .linkOpacity(0.5)
      .linkWidth(0.5)
      .d3AlphaDecay(0.05)
      .controlType('orbit')
      .graphData(computeVisible());

    // -----------------------------------------------------------------------
    // Fog — requires THREE from the global CDN load
    // -----------------------------------------------------------------------
    graph.onEngineStop(() => {{
      try {{
        const scene = graph.scene();
        if (scene && !scene.fog && window.THREE) {{
          scene.fog = new THREE.FogExp2(0x000011, 0.002);
        }}
      }} catch (e) {{
        // THREE not available yet; fog skipped
      }}
    }});

    // -----------------------------------------------------------------------
    // Node click: fly-to + BFS centering
    // -----------------------------------------------------------------------
    graph.onNodeClick((node) => {{
      selectedNodeId = node.id;
      document.getElementById('center-label').textContent = 'Center: ' + node.label;
      // Animated camera fly-to
      graph.cameraPosition(
        {{ x: node.x + 40, y: node.y + 40, z: node.z + 40 }},
        node,
        1000
      );
      applyFilters();
    }});

    // -----------------------------------------------------------------------
    // Filters: recompute and push to graph
    // -----------------------------------------------------------------------
    function applyFilters() {{
      graph.graphData(computeVisible());
    }}

    // -----------------------------------------------------------------------
    // Hop slider
    // -----------------------------------------------------------------------
    const hopSlider = document.getElementById('hop-slider');
    const hopVal = document.getElementById('hop-val');
    hopSlider.addEventListener('input', () => {{
      hopDepth = parseInt(hopSlider.value, 10);
      hopVal.textContent = hopDepth;
      if (selectedNodeId !== null) applyFilters();
    }});

    // -----------------------------------------------------------------------
    // Fly-mode toggle
    // -----------------------------------------------------------------------
    document.getElementById('fly-mode').addEventListener('change', (e) => {{
      graph.controlType(e.target.checked ? 'fly' : 'orbit');
    }});
  </script>
</body>
</html>"""
