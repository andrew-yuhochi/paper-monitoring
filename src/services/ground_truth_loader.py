# Service for loading hand-crafted ground truth concept notes into the GraphStore.
# Reads Markdown files with YAML frontmatter (python-frontmatter) from a directory,
# maps frontmatter fields to Concept and ConceptRelationship models, and upserts
# them via GraphStore. Re-runnable (idempotent). No LLM calls — pure file → DB.

import logging
from pathlib import Path

import frontmatter

from src.models.concepts import Concept, ConceptRelationship
from src.store.graph_store import GraphStore

logger = logging.getLogger(__name__)


def load_ground_truth(
    directory: Path,
    store: GraphStore,
    user_id: str = "default",
) -> tuple[int, int]:
    """Load hand-crafted concept notes from *directory* into *store*.

    Each ``.md`` file is expected to have YAML frontmatter containing at minimum
    a ``name`` field.  All other fields are optional and fall back to model
    defaults when absent.

    Returns:
        (concepts_loaded, relationships_loaded) — counts for the current run.
        Upsert semantics: re-running against the same directory is safe.
    """
    md_files = sorted(directory.glob("*.md"))
    if not md_files:
        logger.warning("No .md files found in ground truth directory: %s", directory)
        return 0, 0

    # First pass: load all concepts and build a name → canonical_name lookup so
    # that relationship target resolution can work case-insensitively.
    concepts_loaded = 0
    known_names: dict[str, str] = {}  # lower-case → canonical name

    for md_file in md_files:
        try:
            post = frontmatter.load(str(md_file))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to parse frontmatter in %s: %s", md_file.name, exc)
            continue

        meta = post.metadata
        name = meta.get("name")
        if not name:
            logger.warning("Skipping %s — no 'name' field in frontmatter", md_file.name)
            continue

        concept = Concept(
            name=name,
            concept_type=meta.get("concept_type", "Concept"),
            what_it_is=meta.get("what_it_is", ""),
            what_problem_it_solves=meta.get("what_problem_it_solves", ""),
            innovation_chain=meta.get("innovation_chain") or [],
            limitations=meta.get("limitations") or [],
            introduced_year=meta.get("introduced_year"),
            domain_tags=meta.get("domain_tags") or [],
            source="manual",
            source_refs=meta.get("source_refs") or [],
            content_angles=meta.get("content_angles") or [],
            user_id=user_id,
        )

        store.upsert_concept(concept)
        known_names[name.lower()] = name
        concepts_loaded += 1
        logger.debug("Upserted concept: %s", name)

    # Second pass: load relationships, resolving targets against known_names.
    relationships_loaded = 0

    for md_file in md_files:
        try:
            post = frontmatter.load(str(md_file))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to re-parse %s: %s", md_file.name, exc)
            continue

        meta = post.metadata
        from_name = meta.get("name")
        if not from_name:
            continue

        raw_rels = meta.get("relationships") or []
        for rel in raw_rels:
            target = rel.get("target", "")
            rel_type = rel.get("type", "")
            label = rel.get("label", "")

            # Fuzzy target resolution — simple case-insensitive dict lookup.
            # All 15 concepts are known; no heavyweight library needed.
            canonical_target = known_names.get(target.lower())
            if canonical_target is None:
                logger.warning(
                    "Relationship from '%s' skipped — unresolved target '%s'",
                    from_name,
                    target,
                )
                continue

            # Skip self-referential edges — the schema enforces
            # source_concept_id != target_concept_id.
            if canonical_target.lower() == from_name.lower():
                logger.warning(
                    "Relationship from '%s' skipped — self-referential edge not allowed",
                    from_name,
                )
                continue

            relationship = ConceptRelationship(
                from_concept=from_name,
                to_concept=canonical_target,
                relationship_type=rel_type,
                label=label,
                user_id=user_id,
            )
            store.upsert_concept_relationship(relationship)
            relationships_loaded += 1
            logger.debug(
                "Upserted relationship: %s -[%s]-> %s", from_name, rel_type, canonical_target
            )

    logger.info(
        "Ground truth load complete: %d concepts, %d relationships",
        concepts_loaded,
        relationships_loaded,
    )
    return concepts_loaded, relationships_loaded
