"""Pre-filter scoring: ranks and trims the raw weekly paper list.

Scoring formula:
    score = (hf_upvotes * UPVOTE_WEIGHT) + category_priority_score

All weights and priorities are read from config.py. Papers absent from the
HuggingFace daily data receive upvotes=0 but remain eligible via category score.
"""

import logging
from src.config import Settings, settings as default_settings
from src.models.arxiv import ArxivPaper
from src.models.graph import ScoredPaper
from src.models.huggingface import HFPaper

logger = logging.getLogger(__name__)


class PreFilter:
    """Scores and filters the raw weekly arXiv paper list."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or default_settings

    def score_and_filter(
        self,
        papers: list[ArxivPaper],
        hf_data: dict[str, HFPaper],
        top_n: int | None = None,
    ) -> list[ScoredPaper]:
        """Score all papers and return the top N by score descending.

        Args:
            papers:  Raw arXiv paper list (may be 800-1,200 papers).
            hf_data: Dict keyed by arXiv ID from HuggingFaceFetcher.fetch_week().
                     Papers absent from this dict receive upvotes=0.
            top_n:   Maximum papers to return. Defaults to config.prefilter_top_n.

        Returns:
            List of ScoredPaper sorted by score descending, length <= top_n.
        """
        if top_n is None:
            top_n = self._cfg.prefilter_top_n

        scored: list[ScoredPaper] = []
        for paper in papers:
            hf_paper = hf_data.get(paper.arxiv_id)
            upvotes = hf_paper.upvotes if hf_paper is not None else 0
            category_score = self._cfg.prefilter_category_priorities.get(
                paper.primary_category, 0
            )
            score = (upvotes * self._cfg.prefilter_upvote_weight) + category_score
            scored.append(ScoredPaper(paper=paper, hf_data=hf_paper, score=score))

        scored.sort(key=lambda sp: sp.score, reverse=True)
        result = scored[:top_n]

        logger.info(
            "PreFilter: %d papers in → %d papers out (top_n=%d)",
            len(papers), len(result), top_n,
        )
        return result
