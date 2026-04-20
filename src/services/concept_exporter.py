# Concept Explorer engine: resolves concept names, traverses BUILDS_ON lineage,
# and renders 8-field Markdown or JSON exports matching UX-SPEC §7/§11.
# Called from src/explore.py — not wired into the weekly pipeline.
# Fuzzy matching uses rapidfuzz.WRatio (handles abbreviations like "XGB" → "XGBoost").

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import NamedTuple

from rapidfuzz import fuzz

from src.config import settings
from src.models.concepts import Concept
from src.store.graph_store import GraphStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class LineageStep(NamedTuple):
    name: str
    label: str  # narrative "why" prose
    depth: int
    what_it_is: str


class ConceptNotFoundError(ValueError):
    """Raised when the queried concept cannot be fuzzy-matched."""

    def __init__(self, query: str, suggestions: list[str]) -> None:
        self.query = query
        self.suggestions = suggestions
        super().__init__(f"Concept not found: {query!r}")


# ---------------------------------------------------------------------------
# ConceptExporter
# ---------------------------------------------------------------------------


def _normalize(name: str) -> str:
    return " ".join(name.lower().split())


# rapidfuzz WRatio returns 0-100; translate to 0-1 for comparison with threshold
_RAPIDFUZZ_SCALE = 100.0


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    return re.sub(r"[^a-z0-9]+", "-", slug).strip("-")


class ConceptExporter:
    """Resolve, traverse, and export concepts from the knowledge graph."""

    def __init__(
        self,
        store: GraphStore,
        match_threshold: float | None = None,
        user_id: str = "default",
    ) -> None:
        self._store = store
        # Threshold is in 0-1 range (config default 0.85).
        # rapidfuzz WRatio is 0-100, so multiply threshold by 100.
        self._threshold = match_threshold or settings.concept_match_threshold
        self._threshold_rf = self._threshold * _RAPIDFUZZ_SCALE
        self._user_id = user_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_concept(self, query: str) -> Concept:
        """Fuzzy-match query against all concepts.

        Returns the matched Concept or raises ConceptNotFoundError with
        top-3 suggestions if no match is within threshold.
        """
        all_concepts = self._store.list_concepts(self._user_id)
        if not all_concepts:
            raise ConceptNotFoundError(query, [])

        norm_query = _normalize(query)
        best_ratio = 0.0
        best_concept: Concept | None = None
        scored: list[tuple[float, str]] = []

        for concept in all_concepts:
            # Use WRatio: handles abbreviations (XGB→XGBoost), partial matches, transpositions
            ratio = fuzz.WRatio(norm_query, _normalize(concept.name))
            scored.append((ratio, concept.name))
            if ratio > best_ratio:
                best_ratio = ratio
                best_concept = concept

        scored.sort(reverse=True)
        top3 = [name for _, name in scored[:3]]

        if best_concept is None or best_ratio < self._threshold_rf:
            raise ConceptNotFoundError(query, top3)

        logger.debug(
            "Resolved %r → %r (ratio=%.3f)", query, best_concept.name, best_ratio
        )
        return best_concept

    def traverse_lineage(
        self, concept: Concept, max_depth: int = 10
    ) -> list[LineageStep]:
        """Walk BUILDS_ON edges INBOUND to concept recursively (oldest → newest).

        Inbound BUILDS_ON means: source BUILDS_ON target — i.e., the source
        was built on top of the target. So to find ancestors of XGBoost, we
        look for concepts that XGBoost's predecessors BUILDS_ON, following the
        chain back.

        Actually: XGBoost has BUILDS_ON edges pointing to its predecessors
        (XGBoost BUILDS_ON GradientBoosting). So we need to follow outgoing
        BUILDS_ON edges recursively to find the lineage (walk forward from
        oldest to newest requires reversing). Let me follow the recursive CTE
        from TDD §2.3.1 which walks source→target of BUILDS_ON from the given
        concept_id. That gives us what XGBoost was built on.

        We then need to reverse the order (oldest → newest).
        """
        concept_id = self._store._get_concept_id(concept.name, self._user_id)
        steps: list[LineageStep] = []
        visited: set[int] = {concept_id}

        # BFS/iterative DFS following outgoing BUILDS_ON edges
        # Each edge: source=XGBoost, target=GradientBoosting (XGBoost BUILDS_ON GBM)
        # We want to collect ancestors in order
        queue: list[tuple[int, str, int]] = [(concept_id, "", 0)]

        while queue:
            current_id, label, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            rows = self._store._conn.execute(
                """
                SELECT cr.target_concept_id, cr.label, c.name, c.what_it_is
                FROM concept_relationships cr
                JOIN concepts c ON cr.target_concept_id = c.id
                WHERE cr.source_concept_id = ?
                  AND cr.relationship_type = 'BUILDS_ON'
                  AND cr.user_id = ?
                """,
                (current_id, self._user_id),
            ).fetchall()

            for row in rows:
                tid = row["target_concept_id"]
                if tid in visited:
                    continue
                visited.add(tid)
                steps.append(
                    LineageStep(
                        name=row["name"],
                        label=row["label"] or "",
                        depth=depth + 1,
                        what_it_is=row["what_it_is"] or "",
                    )
                )
                queue.append((tid, row["label"] or "", depth + 1))

        # Reverse so order is oldest → newest (deepest ancestors first)
        steps.sort(key=lambda s: -s.depth)
        logger.debug(
            "Traversed %d lineage steps for %r", len(steps), concept.name
        )
        return steps

    def get_alternatives(self, concept: Concept) -> list[str]:
        """Return names of concepts connected by ALTERNATIVE_TO edges."""
        rows = self._store._conn.execute(
            """
            SELECT c.name
            FROM concept_relationships cr
            JOIN concepts c ON cr.target_concept_id = c.id
            WHERE cr.source_concept_id = (
                SELECT id FROM concepts WHERE slug = ? AND user_id = ?
            )
            AND cr.relationship_type = 'ALTERNATIVE_TO'
            AND cr.user_id = ?
            """,
            (_slugify(concept.name), self._user_id, self._user_id),
        ).fetchall()
        return [r["name"] for r in rows]

    def get_related(self, concept: Concept) -> list[tuple[str, str]]:
        """Return (name, relationship_type) for all outgoing relationships."""
        try:
            rels = self._store.get_relationships(concept.name, self._user_id)
        except ValueError:
            return []
        return [(r.to_concept, r.relationship_type) for r in rels]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_markdown(
        self,
        concept: Concept,
        out_dir: Path,
        lineage: list[LineageStep] | None = None,
    ) -> Path:
        """Render the 8-field Markdown export per UX-SPEC §7."""
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify(concept.name)
        filename = f"{slug}-{date.today().strftime('%Y%m%d')}.md"
        out_path = out_dir / filename

        if lineage is None:
            lineage = self.traverse_lineage(concept)
        alternatives = self.get_alternatives(concept)
        related = self.get_related(concept)

        content = self._render_markdown(concept, lineage, alternatives, related)
        out_path.write_text(content, encoding="utf-8")
        logger.info("Markdown export written to %s", out_path)
        return out_path

    def export_json(
        self,
        concept: Concept,
        out_dir: Path,
        lineage: list[LineageStep] | None = None,
    ) -> Path:
        """Render structured JSON with the same 8 fields as the Markdown export."""
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify(concept.name)
        filename = f"{slug}-{date.today().strftime('%Y%m%d')}.json"
        out_path = out_dir / filename

        if lineage is None:
            lineage = self.traverse_lineage(concept)
        alternatives = self.get_alternatives(concept)
        related = self.get_related(concept)

        data = self._build_json_payload(concept, lineage, alternatives, related)
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("JSON export written to %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_markdown(
        self,
        concept: Concept,
        lineage: list[LineageStep],
        alternatives: list[str],
        related: list[tuple[str, str]],
    ) -> str:
        lines: list[str] = []

        # Header
        lines.append(f"# CONCEPT EXPLORER — {concept.name}")
        lines.append("")
        lines.append(f"**CONCEPT**: {concept.name}")
        lines.append(
            f"**DEFINITION**: {concept.what_it_is or '(not yet mapped)'}"
        )
        category = ", ".join(concept.domain_tags) if concept.domain_tags else "(not yet mapped)"
        lines.append(f"**CATEGORY**: {category}")
        year = str(concept.introduced_year) if concept.introduced_year else "(unknown)"
        lines.append(f"**YEAR_INTRODUCED**: {year}")
        lines.append("")

        # Lineage
        lines.append("---")
        lines.append("")
        lines.append(f"## LINEAGE ({len(lineage)} steps)")
        lines.append("")
        if lineage:
            for i, step in enumerate(lineage, 1):
                lines.append(f" {i}. **{step.name}**")
                if step.label:
                    lines.append(f"    → {step.label}")
                elif step.what_it_is:
                    lines.append(f"    → {step.what_it_is}")
                lines.append("")
            # Final step: the concept itself
            lines.append(f" {len(lineage) + 1}. **{concept.name}** ← YOU ARE HERE")
        else:
            lines.append(" *(No lineage data — this may be a foundational concept)*")
        lines.append("")

        # Key innovations (from innovation_chain)
        lines.append("---")
        lines.append("")
        lines.append("## KEY_INNOVATIONS")
        lines.append("")
        if concept.innovation_chain:
            for item in concept.innovation_chain:
                if isinstance(item, dict):
                    step_text = item.get("step", "")
                    why = item.get("why_needed", item.get("why", ""))
                    if step_text:
                        lines.append(f" · {step_text}")
                        if why:
                            lines.append(f"   ({why})")
                    elif why:
                        lines.append(f" · {why}")
                else:
                    lines.append(f" · {item}")
        else:
            lines.append(" *(not yet mapped)*")
        lines.append("")

        # Limitations
        lines.append("---")
        lines.append("")
        lines.append("## LIMITATIONS")
        lines.append("")
        if concept.limitations:
            for lim in concept.limitations:
                lines.append(f" · {lim}")
        else:
            lines.append(" *(not yet mapped)*")
        lines.append("")

        # Alternatives
        lines.append("---")
        lines.append("")
        lines.append("## ALTERNATIVES")
        lines.append("")
        if alternatives:
            for alt in alternatives:
                lines.append(f" · {alt}")
        else:
            lines.append(" *(none mapped yet)*")
        lines.append("")

        # Content angles
        lines.append("---")
        lines.append("")
        lines.append("## CONTENT_ANGLES")
        lines.append("")
        if concept.content_angles:
            for i, angle in enumerate(concept.content_angles, 1):
                lines.append(f" {i}. {angle}")
        else:
            lines.append(" *(not yet mapped)*")
        lines.append("")

        # Related
        lines.append("---")
        lines.append("")
        lines.append("## RELATED")
        lines.append("")
        if related:
            for name, rel_type in related:
                lines.append(f" · {name}  [{rel_type}]")
        else:
            lines.append(" *(none mapped yet)*")
        lines.append("")

        # Sources
        lines.append("---")
        lines.append("")
        lines.append("## SOURCES")
        lines.append("")
        if concept.source_refs:
            for ref in concept.source_refs:
                lines.append(f" - {ref}")
        else:
            lines.append(f" - Source: {concept.source}")
        lines.append("")

        return "\n".join(lines)

    def _build_json_payload(
        self,
        concept: Concept,
        lineage: list[LineageStep],
        alternatives: list[str],
        related: list[tuple[str, str]],
    ) -> dict:
        return {
            "concept": concept.name,
            "definition": concept.what_it_is or "",
            "category": ", ".join(concept.domain_tags) if concept.domain_tags else "",
            "year_introduced": concept.introduced_year,
            "lineage": [
                {"step": i + 1, "name": step.name, "why": step.label or step.what_it_is}
                for i, step in enumerate(lineage)
            ],
            "key_innovations": [
                (item if isinstance(item, str) else item.get("step", ""))
                for item in concept.innovation_chain
            ],
            "limitations": concept.limitations,
            "alternatives": alternatives,
            "content_angles": concept.content_angles,
            "related": [{"name": n, "type": t} for n, t in related],
            "sources": concept.source_refs or [concept.source],
        }
