"""HuggingFace Daily Papers API client.

Fetches paper metadata from the HuggingFace Daily Papers endpoint:
  GET https://huggingface.co/api/daily_papers?date=YYYY-MM-DD
  GET https://huggingface.co/api/papers/{arxiv_id}

No authentication required. Rate limit: 1-second delay between sequential
day-by-day requests in fetch_week (configurable via config.hf_fetch_delay).
All HTTP and parsing errors are caught and logged as warnings; methods never
raise to callers.
"""

import logging
import time
from datetime import date, timedelta

import requests

from src.config import Settings, settings as default_settings
from src.models.huggingface import HFPaper

logger = logging.getLogger(__name__)

_HF_BASE_URL = "https://huggingface.co/api"


def _parse_entry(entry: dict) -> HFPaper | None:
    """Parse a single dict from the HF API JSON into an HFPaper.

    Returns None (and logs a warning) if the entry is malformed or missing
    a required paper.id field.
    """
    try:
        paper = entry.get("paper", {})
        arxiv_id = paper.get("id", "")
        if not arxiv_id:
            logger.warning("HF entry missing paper.id; skipping entry")
            return None

        title = paper.get("title", "")
        upvotes = paper.get("upvotes", 0)
        ai_keywords = paper.get("ai_keywords") or []
        ai_summary = paper.get("ai_summary") or None
        num_comments = entry.get("numComments", 0)

        return HFPaper(
            arxiv_id=arxiv_id,
            title=title,
            upvotes=upvotes,
            ai_keywords=ai_keywords,
            ai_summary=ai_summary,
            num_comments=num_comments,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse HF entry: %s", exc)
        return None


class HuggingFaceFetcher:
    """Fetches paper metadata from the HuggingFace Daily Papers API."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or default_settings

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_daily_papers(self, date_str: str) -> list[HFPaper]:
        """Fetch all HF daily papers for a given date (YYYY-MM-DD format).

        Returns an empty list on any HTTP error, connection error, or
        malformed JSON. Individual malformed entries are skipped.
        """
        url = f"{_HF_BASE_URL}/daily_papers"
        params = {"date": date_str}

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(
                "HF daily papers request failed for date=%s: %s", date_str, exc
            )
            return []

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "HF daily papers returned invalid JSON for date=%s: %s", date_str, exc
            )
            return []

        if not isinstance(data, list):
            logger.warning(
                "HF daily papers unexpected response type for date=%s: %s",
                date_str, type(data).__name__,
            )
            return []

        papers: list[HFPaper] = []
        for entry in data:
            paper = _parse_entry(entry)
            if paper is not None:
                papers.append(paper)

        logger.info("Fetched %d papers from HF for date=%s", len(papers), date_str)
        return papers

    def fetch_week(self, end_date: str) -> dict[str, HFPaper]:
        """Fetch 7 days of HF daily papers ending on end_date (inclusive).

        Dates are fetched from end_date backwards (end_date, end_date-1, ...,
        end_date-6). Sleeps config.hf_fetch_delay between each day's call.
        Returns a dict keyed by arXiv ID; first occurrence of a duplicate ID wins.
        """
        end = date.fromisoformat(end_date)
        result: dict[str, HFPaper] = {}

        for i in range(7):
            day = end - timedelta(days=i)
            day_str = day.isoformat()

            papers = self.fetch_daily_papers(day_str)
            for paper in papers:
                if paper.arxiv_id not in result:
                    result[paper.arxiv_id] = paper

            # Sleep between calls, but not after the final one
            if i < 6:
                time.sleep(self._cfg.hf_fetch_delay)

        logger.info(
            "fetch_week complete for end_date=%s. Total unique papers: %d",
            end_date, len(result),
        )
        return result

    def lookup_paper(self, arxiv_id: str) -> HFPaper | None:
        """Look up a single paper by arXiv ID.

        Returns None if the paper is not found or on any error.
        """
        url = f"{_HF_BASE_URL}/papers/{arxiv_id}"

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(
                "HF paper lookup failed for arxiv_id=%s: %s", arxiv_id, exc
            )
            return None

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "HF paper lookup returned invalid JSON for arxiv_id=%s: %s",
                arxiv_id, exc,
            )
            return None

        # The single-paper endpoint returns the same shape as one array element
        return _parse_entry(data)
