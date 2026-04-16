"""Seeder service: orchestrates the knowledge bank seeding pipeline.

Used by seed.py (CLI) for:
  - Full seeding: landmark papers + survey papers + textbook chapters
  - Single-paper seeding: python -m src.seed --arxiv-id <id>

Architecture: two-pass concept writing
  Pass 1 — for each source, create concept nodes + INTRODUCES edges
  Pass 2 — create PREREQUISITE_OF edges (all concept targets now exist)
"""

import logging
from pathlib import Path

from src.integrations.arxiv_client import ArxivFetcher
from src.integrations.pdf_extractor import PdfExtractor
from src.models.arxiv import ArxivPaper
from src.models.classification import ExtractedConcept
from src.models.seeding import ChapterText
from src.services.classifier import OllamaClassifier
from src.store.graph_store import GraphStore
from src.utils.normalize import normalize_concept_name

logger = logging.getLogger(__name__)

# Type alias for textbook config entries
TextbookConfig = tuple[Path, list[tuple[int, int, str]]]


class Seeder:
    """Orchestrates the knowledge bank seeding pipeline."""

    def __init__(
        self,
        store: GraphStore,
        arxiv_fetcher: ArxivFetcher,
        pdf_extractor: PdfExtractor,
        classifier: OllamaClassifier,
    ) -> None:
        self._store = store
        self._arxiv = arxiv_fetcher
        self._pdf = pdf_extractor
        self._classifier = classifier
        # Buffer for two-pass prerequisite writing:
        # list of (concept_node_id, prerequisite_concept_names)
        self._prerequisite_buffer: list[tuple[str, list[str]]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed_paper(self, paper: ArxivPaper, source_label: str) -> int:
        """Seed a single ArxivPaper: upsert paper node, extract+store concepts.

        Returns number of concepts extracted.
        Concept nodes and INTRODUCES edges are written immediately.
        PREREQUISITE_OF edges are buffered for the two-pass flush.
        """
        paper_node_id = f"paper:{paper.arxiv_id}"

        # Upsert the paper node
        paper_props = {
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "published_date": paper.published_date,
            "arxiv_url": paper.arxiv_url,
            "pdf_url": paper.pdf_url,
            "seeded_from": source_label,
        }
        self._store.upsert_node(paper_node_id, "paper", paper.title, paper_props)
        logger.info("Seeding paper %s: %s", paper.arxiv_id, paper.title)

        # Extract concepts from title + abstract
        text = f"{paper.title}\n\n{paper.abstract}"
        concepts = self._classifier.extract_concepts(
            text, source_id=paper_node_id, source_type=source_label,
        )
        logger.info("  -> %d concepts extracted", len(concepts))

        self._store_concepts(concepts, paper_node_id, source_label)
        return len(concepts)

    def seed_chapter(self, chapter: ChapterText, chunk_size: int = 3000) -> int:
        """Seed a textbook chapter: extract+store concepts.

        Long chapters are split into chunks of ``chunk_size`` words to avoid
        overwhelming the LLM with too much input (which causes schema errors).
        Concepts from all chunks are merged; duplicates are handled by upsert.

        Returns total number of concepts extracted across all chunks.
        """
        logger.info("Seeding chapter: %s", chapter.source_description)
        words = chapter.text.split()
        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]
        logger.info("  Split into %d chunks (%d words total)", len(chunks), len(words))

        all_concepts: list[ExtractedConcept] = []
        for idx, chunk in enumerate(chunks, start=1):
            logger.info("  Extracting from chunk %d of %d...", idx, len(chunks))
            concepts = self._classifier.extract_concepts(
                chunk,
                source_id=f"{chapter.source_description} (chunk {idx}/{len(chunks)})",
                source_type="textbook_chapter",
            )
            all_concepts.extend(concepts)
            logger.info("    -> %d concepts from chunk %d", len(concepts), idx)

        logger.info("  -> %d total concepts extracted", len(all_concepts))
        self._store_concepts(all_concepts, source_node_id=None, source_label=chapter.source_description)
        return len(all_concepts)

    def flush_prerequisites(self) -> int:
        """Pass 2: write all buffered PREREQUISITE_OF edges.

        Must be called after all seed_paper / seed_chapter calls.
        Returns number of edges written.
        """
        edges_written = 0
        for concept_node_id, prereq_names in self._prerequisite_buffer:
            for prereq_name in prereq_names:
                norm = normalize_concept_name(prereq_name)
                if not norm:
                    continue
                target_id = f"concept:{norm}"
                # Only create the edge if the target node exists
                if self._store.get_node(target_id) is not None:
                    self._store.upsert_edge(
                        concept_node_id, target_id, "PREREQUISITE_OF"
                    )
                    edges_written += 1
                else:
                    logger.debug(
                        "Skipping PREREQUISITE_OF edge — target not in graph: %s",
                        target_id,
                    )
        logger.info("flush_prerequisites: wrote %d PREREQUISITE_OF edges", edges_written)
        self._prerequisite_buffer.clear()
        return edges_written

    def seed_all(
        self,
        landmark_ids: list[str],
        survey_ids: list[str],
        textbook_configs: list[TextbookConfig],
        seed_papers: bool = True,
        seed_textbooks: bool = True,
    ) -> dict:
        """Full seeding run (two-pass).

        Pass 1: landmark papers -> survey papers -> textbook chapters
                (writes concept nodes + INTRODUCES edges as it goes)
        Pass 2: flush_prerequisites() writes all PREREQUISITE_OF edges

        Returns a summary dict with counts.
        """
        total_papers = 0
        total_concepts_from_papers = 0
        total_concepts_from_textbooks = 0

        if seed_papers:
            # --- Landmark papers ---
            logger.info("Fetching %d landmark papers...", len(landmark_ids))
            try:
                landmark_papers = self._arxiv.fetch_batch(landmark_ids)
            except Exception as exc:
                logger.error("Failed to fetch landmark papers: %s", exc)
                landmark_papers = []

            for paper in landmark_papers:
                try:
                    n = self.seed_paper(paper, source_label="landmark_paper")
                    total_concepts_from_papers += n
                    total_papers += 1
                except Exception as exc:
                    logger.error("Error seeding landmark paper %s: %s", paper.arxiv_id, exc)

            # --- Survey papers ---
            logger.info("Fetching %d survey papers...", len(survey_ids))
            try:
                survey_papers = self._arxiv.fetch_batch(survey_ids)
            except Exception as exc:
                logger.error("Failed to fetch survey papers: %s", exc)
                survey_papers = []

            for paper in survey_papers:
                try:
                    n = self.seed_paper(paper, source_label="survey_paper")
                    total_concepts_from_papers += n
                    total_papers += 1
                except Exception as exc:
                    logger.error("Error seeding survey paper %s: %s", paper.arxiv_id, exc)

        if seed_textbooks:
            # --- Textbook chapters ---
            for pdf_path, chapter_ranges in textbook_configs:
                if not pdf_path.exists():
                    logger.warning("Textbook PDF not found, skipping: %s", pdf_path)
                    continue
                chapters = self._pdf.extract_chapters(pdf_path, chapter_ranges)
                for chapter in chapters:
                    try:
                        n = self.seed_chapter(chapter)
                        total_concepts_from_textbooks += n
                    except Exception as exc:
                        logger.error(
                            "Error seeding chapter %s: %s", chapter.source_description, exc
                        )

        # Pass 2: flush prerequisites
        prereq_edges = self.flush_prerequisites()
        concept_count = len(self._store.get_concept_index())

        summary = {
            "papers_seeded": total_papers,
            "concepts_from_papers": total_concepts_from_papers,
            "concepts_from_textbooks": total_concepts_from_textbooks,
            "prerequisite_edges": prereq_edges,
            "total_concepts_in_store": concept_count,
        }
        logger.info("Seeding complete: %s", summary)
        return summary

    def seed_paper_by_id(self, arxiv_id: str) -> int:
        """Seed a single paper by arXiv ID (for --arxiv-id CLI flag).

        Fetches the paper, extracts concepts, flushes prerequisites.
        Returns number of concepts extracted.
        """
        logger.info("Fetching paper %s...", arxiv_id)
        try:
            paper = self._arxiv.fetch_by_id(arxiv_id)
        except Exception as exc:
            logger.error("Failed to fetch paper %s: %s", arxiv_id, exc)
            return 0

        n = self.seed_paper(paper, source_label="manual_seed")
        self.flush_prerequisites()
        logger.info("Seeded paper %s: %d concepts", arxiv_id, n)
        return n

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _store_concepts(
        self,
        concepts: list[ExtractedConcept],
        source_node_id: str | None,
        source_label: str,
    ) -> None:
        """Upsert concept nodes and INTRODUCES edges; buffer prerequisite pairs.

        Called identically from seed_paper and seed_chapter.
        """
        for concept in concepts:
            norm = normalize_concept_name(concept.name)
            if not norm:
                logger.warning("Skipping concept with empty normalised name: %r", concept.name)
                continue

            concept_node_id = f"concept:{norm}"
            concept_props = {
                "description": concept.description,
                "domain_tags": concept.domain_tags,
                "seeded_from": source_label,
            }
            self._store.upsert_node(
                concept_node_id, "concept", concept.name, concept_props
            )

            # INTRODUCES edge: paper -> concept (only when seeding from a paper node)
            if source_node_id is not None:
                self._store.upsert_edge(source_node_id, concept_node_id, "INTRODUCES")

            # Buffer prerequisites for Pass 2
            if concept.prerequisite_concept_names:
                self._prerequisite_buffer.append(
                    (concept_node_id, concept.prerequisite_concept_names)
                )
