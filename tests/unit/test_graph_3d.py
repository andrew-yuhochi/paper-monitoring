"""Unit tests for render_graph_3d (src/dashboard/graph_3d.py)."""
from __future__ import annotations

import json
import re

import pytest

from src.dashboard.graph_3d import render_graph_3d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_node(
    node_id: str,
    node_type: str,
    label: str,
    properties: dict | None = None,
) -> dict:
    base = {"id": node_id, "node_type": node_type, "label": label}
    if properties is not None:
        base["properties"] = properties
    return base


def _make_edge(
    source_id: str,
    target_id: str,
    relationship_type: str = "relates_to",
    weight: float = 1.0,
) -> dict:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "relationship_type": relationship_type,
        "weight": weight,
        "properties": {},
    }


@pytest.fixture
def representative_graph() -> tuple[list[dict], list[dict]]:
    """One node per type plus a handful of edges — used in most tests."""
    nodes = [
        _make_node("n:problem:001", "problem", "Vanishing Gradient", {"description": "Gradient issues"}),
        _make_node("n:technique:001", "technique", "Batch Normalisation", {}),
        _make_node("n:concept:001", "concept", "Backpropagation", {}),
        _make_node("n:category:001", "category", "Training Methods", {}),
        _make_node("n:paper:001", "paper", "ResNet (He et al. 2016)", {}),
    ]
    edges = [
        _make_edge("n:technique:001", "n:problem:001", "solves"),
        _make_edge("n:concept:001", "n:technique:001", "enables"),
        _make_edge("n:paper:001", "n:technique:001", "introduces"),
    ]
    return nodes, edges


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_return_type_is_str(representative_graph: tuple) -> None:
    """render_graph_3d must return a str."""
    nodes, edges = representative_graph
    result = render_graph_3d(nodes, edges)
    assert isinstance(result, str)


def test_output_contains_doctype(representative_graph: tuple) -> None:
    """Output must start with <!DOCTYPE html>."""
    nodes, edges = representative_graph
    result = render_graph_3d(nodes, edges)
    assert "<!DOCTYPE html>" in result


def test_output_contains_closing_html_tag(representative_graph: tuple) -> None:
    """Output must contain </html>."""
    nodes, edges = representative_graph
    result = render_graph_3d(nodes, edges)
    assert "</html>" in result


def test_three_cdn_url_pinned(representative_graph: tuple) -> None:
    """The THREE.js CDN URL must reference exactly three@0.166.0."""
    nodes, edges = representative_graph
    result = render_graph_3d(nodes, edges)
    assert "three@0.166.0" in result


def test_graph_cdn_url_pinned(representative_graph: tuple) -> None:
    """The 3d-force-graph CDN URL must reference exactly 3d-force-graph@1.77.0."""
    nodes, edges = representative_graph
    result = render_graph_3d(nodes, edges)
    assert "3d-force-graph@1.77.0" in result


def test_height_parameter_controls_pixel_height() -> None:
    """height=800 should produce '800px' in the HTML output."""
    result = render_graph_3d([], [], height=800)
    assert "800px" in result


def test_height_default_is_750() -> None:
    """Default height (750) should produce '750px' in the HTML output."""
    result = render_graph_3d([], [])
    assert "750px" in result


def test_custom_height_does_not_embed_default() -> None:
    """When a custom height is supplied, the default 750px must not appear."""
    result = render_graph_3d([], [], height=600)
    assert "600px" in result
    assert "750px" not in result


# ---------------------------------------------------------------------------
# Node serialization
# ---------------------------------------------------------------------------


def test_node_id_appears_in_json(representative_graph: tuple) -> None:
    """Every node id must appear verbatim in the rendered JSON."""
    nodes, edges = representative_graph
    result = render_graph_3d(nodes, edges)
    for node in nodes:
        assert node["id"] in result, f"Node id {node['id']!r} not found in output"


def test_node_label_appears_in_output(representative_graph: tuple) -> None:
    """Every node label must appear verbatim in the rendered output."""
    nodes, edges = representative_graph
    result = render_graph_3d(nodes, edges)
    for node in nodes:
        assert node["label"] in result, f"Label {node['label']!r} not found in output"


def test_problem_node_color() -> None:
    """problem node_type must map to hex color #E74C3C."""
    nodes = [_make_node("p1", "problem", "A Problem", {})]
    result = render_graph_3d(nodes, [])
    assert "#E74C3C" in result


def test_technique_node_color() -> None:
    """technique node_type must map to hex color #3498DB."""
    nodes = [_make_node("t1", "technique", "A Technique", {})]
    result = render_graph_3d(nodes, [])
    assert "#3498DB" in result


def test_concept_node_color() -> None:
    """concept node_type must map to hex color #2ECC71."""
    nodes = [_make_node("c1", "concept", "A Concept", {})]
    result = render_graph_3d(nodes, [])
    assert "#2ECC71" in result


def test_category_node_color() -> None:
    """category node_type must map to hex color #95A5A6."""
    nodes = [_make_node("cat1", "category", "A Category", {})]
    result = render_graph_3d(nodes, [])
    assert "#95A5A6" in result


def test_paper_node_color() -> None:
    """paper node_type must map to hex color #F1C40F."""
    nodes = [_make_node("ppr1", "paper", "A Paper", {})]
    result = render_graph_3d(nodes, [])
    assert "#F1C40F" in result


def test_problem_node_size() -> None:
    """problem node_type must map to size 10 in the serialized 'val' field."""
    nodes = [_make_node("p1", "problem", "A Problem", {})]
    result = render_graph_3d(nodes, [])
    # Extract the allNodes JSON array from the output
    match = re.search(r"const allNodes = (\[.*?\]);", result, re.DOTALL)
    assert match, "allNodes assignment not found in output"
    data = json.loads(match.group(1))
    assert data[0]["val"] == 10, f"Expected val=10 for problem, got {data[0]['val']}"


def test_technique_node_size() -> None:
    """technique node_type must map to size 8."""
    nodes = [_make_node("t1", "technique", "A Technique", {})]
    result = render_graph_3d(nodes, [])
    match = re.search(r"const allNodes = (\[.*?\]);", result, re.DOTALL)
    assert match
    data = json.loads(match.group(1))
    assert data[0]["val"] == 8


def test_concept_node_size() -> None:
    """concept node_type must map to size 5."""
    nodes = [_make_node("c1", "concept", "A Concept", {})]
    result = render_graph_3d(nodes, [])
    match = re.search(r"const allNodes = (\[.*?\]);", result, re.DOTALL)
    assert match
    data = json.loads(match.group(1))
    assert data[0]["val"] == 5


def test_category_node_size() -> None:
    """category node_type must map to size 6."""
    nodes = [_make_node("cat1", "category", "A Category", {})]
    result = render_graph_3d(nodes, [])
    match = re.search(r"const allNodes = (\[.*?\]);", result, re.DOTALL)
    assert match
    data = json.loads(match.group(1))
    assert data[0]["val"] == 6


def test_paper_node_size() -> None:
    """paper node_type must map to size 6."""
    nodes = [_make_node("ppr1", "paper", "A Paper", {})]
    result = render_graph_3d(nodes, [])
    match = re.search(r"const allNodes = (\[.*?\]);", result, re.DOTALL)
    assert match
    data = json.loads(match.group(1))
    assert data[0]["val"] == 6


def test_unknown_node_type_default_color() -> None:
    """An unrecognized node_type must receive default color #CCCCCC."""
    nodes = [_make_node("x1", "widget", "Mystery Node", {})]
    result = render_graph_3d(nodes, [])
    match = re.search(r"const allNodes = (\[.*?\]);", result, re.DOTALL)
    assert match
    data = json.loads(match.group(1))
    assert data[0]["color"] == "#CCCCCC", f"Expected #CCCCCC, got {data[0]['color']!r}"


def test_unknown_node_type_default_size() -> None:
    """An unrecognized node_type must receive default val 5."""
    nodes = [_make_node("x1", "widget", "Mystery Node", {})]
    result = render_graph_3d(nodes, [])
    match = re.search(r"const allNodes = (\[.*?\]);", result, re.DOTALL)
    assert match
    data = json.loads(match.group(1))
    assert data[0]["val"] == 5, f"Expected val=5, got {data[0]['val']}"


# ---------------------------------------------------------------------------
# Edge serialization
# ---------------------------------------------------------------------------


def test_edges_remapped_source_key() -> None:
    """source_id must be remapped to source in the serialized edges."""
    nodes = [
        _make_node("a", "concept", "A", {}),
        _make_node("b", "concept", "B", {}),
    ]
    edges = [_make_edge("a", "b", "relates_to")]
    result = render_graph_3d(nodes, edges)
    match = re.search(r"const allEdges = (\[.*?\]);", result, re.DOTALL)
    assert match, "allEdges assignment not found in output"
    data = json.loads(match.group(1))
    assert "source" in data[0], "Key 'source' missing from serialized edge"
    assert data[0]["source"] == "a"


def test_edges_remapped_target_key() -> None:
    """target_id must be remapped to target in the serialized edges."""
    nodes = [
        _make_node("a", "concept", "A", {}),
        _make_node("b", "concept", "B", {}),
    ]
    edges = [_make_edge("a", "b", "relates_to")]
    result = render_graph_3d(nodes, edges)
    match = re.search(r"const allEdges = (\[.*?\]);", result, re.DOTALL)
    assert match
    data = json.loads(match.group(1))
    assert "target" in data[0], "Key 'target' missing from serialized edge"
    assert data[0]["target"] == "b"


def test_relationship_type_preserved() -> None:
    """relationship_type value must appear in the serialized edge output."""
    nodes = [
        _make_node("a", "concept", "A", {}),
        _make_node("b", "technique", "B", {}),
    ]
    edges = [_make_edge("a", "b", "enables")]
    result = render_graph_3d(nodes, edges)
    match = re.search(r"const allEdges = (\[.*?\]);", result, re.DOTALL)
    assert match
    data = json.loads(match.group(1))
    assert data[0]["relationship_type"] == "enables"


def test_source_id_key_absent_in_serialized_edges() -> None:
    """The original source_id key must NOT appear in the serialized allEdges JSON."""
    nodes = [
        _make_node("a", "concept", "A", {}),
        _make_node("b", "concept", "B", {}),
    ]
    edges = [_make_edge("a", "b")]
    result = render_graph_3d(nodes, edges)
    match = re.search(r"const allEdges = (\[.*?\]);", result, re.DOTALL)
    assert match
    data = json.loads(match.group(1))
    for edge in data:
        assert "source_id" not in edge, "source_id must not appear in serialized edges"


def test_target_id_key_absent_in_serialized_edges() -> None:
    """The original target_id key must NOT appear in the serialized allEdges JSON."""
    nodes = [
        _make_node("a", "concept", "A", {}),
        _make_node("b", "concept", "B", {}),
    ]
    edges = [_make_edge("a", "b")]
    result = render_graph_3d(nodes, edges)
    match = re.search(r"const allEdges = (\[.*?\]);", result, re.DOTALL)
    assert match
    data = json.loads(match.group(1))
    for edge in data:
        assert "target_id" not in edge, "target_id must not appear in serialized edges"


# ---------------------------------------------------------------------------
# HTML injection safety
# ---------------------------------------------------------------------------


def test_script_tag_in_label_is_escaped() -> None:
    """A node label containing </script> must be escaped so the HTML stays valid."""
    malicious_label = 'Safe</script><script>alert("xss")</script>'
    nodes = [_make_node("evil", "concept", malicious_label, {})]
    result = render_graph_3d(nodes, [])
    # The raw closing </script> sequence injected by the label must not appear
    # unescaped inside the JS block (the first </script> closes the real tag).
    # After escaping it becomes <\/script>.
    assert "<\\/script>" in result, "Escaped form <\\/script> not found in output"
    # Ensure the unescaped form does not appear inside the allNodes assignment
    match = re.search(r"const allNodes = (\[.*?\]);", result, re.DOTALL)
    assert match
    raw_nodes_json = match.group(1)
    assert "</script>" not in raw_nodes_json, (
        "Unescaped </script> found inside allNodes JSON — HTML injection possible"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_inputs_produce_valid_html() -> None:
    """Empty nodes and edges must not raise and must produce valid HTML."""
    result = render_graph_3d([], [])
    assert isinstance(result, str)
    assert "<!DOCTYPE html>" in result
    assert "</html>" in result


def test_empty_nodes_json_is_empty_array() -> None:
    """allNodes must be serialized as an empty array when no nodes are passed."""
    result = render_graph_3d([], [])
    match = re.search(r"const allNodes = (\[.*?\]);", result, re.DOTALL)
    assert match
    assert json.loads(match.group(1)) == []


def test_empty_edges_json_is_empty_array() -> None:
    """allEdges must be serialized as an empty array when no edges are passed."""
    result = render_graph_3d([], [])
    match = re.search(r"const allEdges = (\[.*?\]);", result, re.DOTALL)
    assert match
    assert json.loads(match.group(1)) == []


def test_node_without_properties_key_does_not_raise() -> None:
    """A node dict lacking a 'properties' key must be handled gracefully."""
    node_without_props = {"id": "noprops", "node_type": "concept", "label": "No Props"}
    result = render_graph_3d([node_without_props], [])
    assert isinstance(result, str)
    assert "noprops" in result


def test_node_without_properties_defaults_to_empty_dict() -> None:
    """When 'properties' is absent the enriched node should carry an empty dict."""
    node_without_props = {"id": "noprops", "node_type": "concept", "label": "No Props"}
    result = render_graph_3d([node_without_props], [])
    match = re.search(r"const allNodes = (\[.*?\]);", result, re.DOTALL)
    assert match
    data = json.loads(match.group(1))
    assert data[0]["properties"] == {}, (
        f"Expected empty dict for missing properties, got {data[0]['properties']!r}"
    )
