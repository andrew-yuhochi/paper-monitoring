"""GraphExporter: converts the concept graph to Obsidian vault, Neo4j Cypher, and Cytoscape JSON.

Reads from concepts + concept_relationships tables (read-only).
Three export targets:
  - Obsidian vault  : one Markdown file per concept with YAML frontmatter + wikilinks
  - Neo4j Cypher    : .cypher file with CREATE / MATCH statements, loadable in cypher-shell
  - Cytoscape JSON  : {nodes, edges} JSON readable by Cytoscape.js / the desktop app
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

from src.models.concepts import Concept, ConceptRelationship
from src.store.graph_store import GraphStore
from src.utils.normalize import normalize_concept_name

logger = logging.getLogger(__name__)

# Relationship types surfaced as Obsidian wikilinks / section headers
_WIKILINK_RELATIONSHIP_TYPES = ("BUILDS_ON", "ALTERNATIVE_TO", "PREREQUISITE_OF")


def _slug(name: str) -> str:
    """Convert a concept name to a lowercase hyphenated slug for file names."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


class GraphExporter:
    """Exports the concept graph to three formats: Obsidian, Neo4j Cypher, Cytoscape JSON."""

    def __init__(self, store: GraphStore, user_id: str = "default") -> None:
        self._store = store
        self._user_id = user_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_obsidian_vault(self, vault_dir: Path) -> int:
        """Write one Markdown file per concept into vault_dir/concepts/.

        Returns the number of files written.
        """
        concepts_dir = vault_dir / "concepts"
        concepts_dir.mkdir(parents=True, exist_ok=True)

        concepts = self._store.list_concepts(self._user_id)
        if not concepts:
            logger.warning("No concepts found for user_id=%r — vault will be empty", self._user_id)

        # Build slug → name map for wikilinks (resolve target concept names)
        slug_to_name: dict[str, str] = {_slug(c.name): c.name for c in concepts}

        written = 0
        for concept in concepts:
            outfile = concepts_dir / f"{_slug(concept.name)}.md"
            content = self._render_obsidian_note(concept, slug_to_name)
            outfile.write_text(content, encoding="utf-8")
            written += 1
            logger.debug("Wrote Obsidian note: %s", outfile)

        logger.info("Obsidian vault: %d concepts written to %s", written, concepts_dir)
        return written

    def to_neo4j_cypher(self, output_dir: Path) -> Path:
        """Write a .cypher file with CREATE / MATCH statements for all concepts + relationships.

        Returns the path of the written file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        today = date.today().strftime("%Y%m%d")
        out_path = output_dir / f"graph-{today}.cypher"

        concepts = self._store.list_concepts(self._user_id)
        lines: list[str] = [
            "// Paper Monitoring — concept graph export",
            f"// Generated: {date.today().isoformat()}",
            f"// user_id: {self._user_id}",
            "// Load with: cypher-shell -f <this file>",
            "",
            "// ── Concepts ──────────────────────────────────────────────",
        ]

        for concept in concepts:
            slug_val = _slug(concept.name)
            year_val = str(concept.introduced_year) if concept.introduced_year else "null"
            tags_val = _cypher_str(", ".join(concept.domain_tags))
            lines.append(
                f"CREATE (:{_neo4j_label(concept.concept_type)} {{"
                f"slug: {_cypher_str(slug_val)}, "
                f"name: {_cypher_str(concept.name)}, "
                f"concept_type: {_cypher_str(concept.concept_type)}, "
                f"introduced_year: {year_val}, "
                f"domain_tags: {tags_val}, "
                f"what_it_is: {_cypher_str(concept.what_it_is)}"
                f"}});"
            )

        lines += [
            "",
            "// ── Relationships ─────────────────────────────────────────",
        ]

        for concept in concepts:
            rels = self._store.get_relationships(concept.name, self._user_id)
            for rel in rels:
                src_slug = _slug(rel.from_concept)
                tgt_slug = _slug(rel.to_concept)
                if rel.label:
                    rel_props = f"{{label: {_cypher_str(rel.label)}}}"
                else:
                    rel_props = "{}"
                lines.append(
                    f"MATCH (a:Concept {{slug: {_cypher_str(src_slug)}}}), "
                    f"(b:Concept {{slug: {_cypher_str(tgt_slug)}}}) "
                    f"CREATE (a)-[:{rel.relationship_type} {rel_props}]->(b);"
                )

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Neo4j Cypher export written to %s", out_path)
        return out_path

    def to_cytoscape_json(self, output_dir: Path) -> Path:
        """Write a Cytoscape-compatible JSON file ({nodes, edges}).

        Returns the path of the written file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        today = date.today().strftime("%Y%m%d")
        out_path = output_dir / f"graph-{today}.json"

        concepts = self._store.list_concepts(self._user_id)

        nodes: list[dict] = []
        for concept in concepts:
            nodes.append({
                "data": {
                    "id": _slug(concept.name),
                    "label": concept.name,
                    "concept_type": concept.concept_type,
                    "introduced_year": concept.introduced_year,
                    "domain_tags": concept.domain_tags,
                    "what_it_is": concept.what_it_is,
                }
            })

        edges: list[dict] = []
        edge_id = 0
        for concept in concepts:
            rels = self._store.get_relationships(concept.name, self._user_id)
            for rel in rels:
                edge_id += 1
                edges.append({
                    "data": {
                        "id": f"e{edge_id}",
                        "source": _slug(rel.from_concept),
                        "target": _slug(rel.to_concept),
                        "relationship_type": rel.relationship_type,
                        "label": rel.label,
                    }
                })

        payload = {"nodes": nodes, "edges": edges}
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "Cytoscape JSON export written to %s (%d nodes, %d edges)",
            out_path, len(nodes), len(edges),
        )
        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_obsidian_note(
        self,
        concept: Concept,
        slug_to_name: dict[str, str],
    ) -> str:
        """Render a single Obsidian Markdown note for a concept."""
        # YAML frontmatter
        year_str = str(concept.introduced_year) if concept.introduced_year else ""
        domain_yaml = "\n".join(f"  - {t}" for t in concept.domain_tags)
        refs_yaml = "\n".join(f"  - {r}" for r in concept.source_refs)

        frontmatter_lines = [
            "---",
            f"name: {concept.name}",
            f"concept_type: {concept.concept_type}",
        ]
        if year_str:
            frontmatter_lines.append(f"introduced_year: {year_str}")
        if concept.domain_tags:
            frontmatter_lines += ["domain_tags:", domain_yaml]
        if concept.source_refs:
            frontmatter_lines += ["source_refs:", refs_yaml]
        frontmatter_lines.append("---")

        # Body: what it is, problem solved
        body_lines: list[str] = [
            "",
            f"# {concept.name}",
            "",
        ]
        if concept.what_it_is:
            body_lines += [concept.what_it_is, ""]
        if concept.what_problem_it_solves:
            body_lines += [f"**Problem solved:** {concept.what_problem_it_solves}", ""]

        # Relationships grouped by type (only wikilink-friendly types shown as wikilinks)
        rels = self._store.get_relationships(concept.name, self._user_id)
        rels_by_type: dict[str, list[ConceptRelationship]] = {}
        for rel in rels:
            rels_by_type.setdefault(rel.relationship_type, []).append(rel)

        for rel_type in _WIKILINK_RELATIONSHIP_TYPES:
            if rel_type not in rels_by_type:
                continue
            section_title = rel_type.replace("_", " ").title()
            body_lines.append(f"## {section_title}")
            body_lines.append("")
            for rel in rels_by_type[rel_type]:
                target_slug = _slug(rel.to_concept)
                # Resolve display name from slug map; fall back to stored name
                display_name = slug_to_name.get(target_slug, rel.to_concept)
                wikilink = f"[[{target_slug}|{display_name}]]"
                if rel.label:
                    body_lines.append(f"- {wikilink} — {rel.label}")
                else:
                    body_lines.append(f"- {wikilink}")
            body_lines.append("")

        # Other relationship types (not wikilinked, shown as plain text)
        other_types = [t for t in rels_by_type if t not in _WIKILINK_RELATIONSHIP_TYPES]
        if other_types:
            body_lines.append("## Related")
            body_lines.append("")
            for rel_type in other_types:
                for rel in rels_by_type[rel_type]:
                    body_lines.append(
                        f"- **{rel_type}**: {rel.to_concept}"
                        + (f" — {rel.label}" if rel.label else "")
                    )
            body_lines.append("")

        # Content angles
        if concept.content_angles:
            body_lines.append("## Content Angles")
            body_lines.append("")
            for angle in concept.content_angles:
                body_lines.append(f"- {angle}")
            body_lines.append("")

        return "\n".join(frontmatter_lines + body_lines)


# ---------------------------------------------------------------------------
# Cypher formatting helpers
# ---------------------------------------------------------------------------

def _cypher_str(value: str) -> str:
    """Escape a string value for use as a Cypher string literal."""
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _neo4j_label(concept_type: str) -> str:
    """Return a Neo4j node label. Always includes :Concept; adds the specific type."""
    return f"Concept:{concept_type}"
