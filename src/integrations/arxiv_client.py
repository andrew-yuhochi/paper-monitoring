"""arXiv API client for fetching paper metadata.

Uses the arXiv Atom API (https://export.arxiv.org/api/query).
No authentication required. Rate limit: 3-second delay between requests.

Date-filter workaround: arXiv's submittedDate range query is broken (returns 0
results). Instead we sort by submittedDate descending and post-filter in Python.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone

import requests

from src.config import Settings, settings as default_settings
from src.models.arxiv import ArxivPaper

logger = logging.getLogger(__name__)

# XML namespace map for the arXiv Atom feed
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

_ARXIV_API_URL = "https://export.arxiv.org/api/query"

# Regex to extract canonical arXiv ID from a URL like:
# http://arxiv.org/abs/2604.11805v1  ->  2604.11805
_ID_RE = re.compile(r"abs/([^v]+?)(?:v\d+)?$")


def _extract_arxiv_id(id_url: str) -> str:
    """Extract the canonical arXiv ID (no version suffix) from the <id> URL."""
    match = _ID_RE.search(id_url.strip())
    if match:
        return match.group(1)
    # Fallback: return everything after the last slash, stripped of version
    raw = id_url.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", raw)


def _parse_date(date_str: str) -> str:
    """Parse an ISO 8601 timestamp (e.g. '2026-04-13T17:59:40Z') to 'YYYY-MM-DD'."""
    return date_str[:10]


def _parse_entry(entry: ET.Element) -> ArxivPaper | None:
    """Parse a single Atom <entry> element into an ArxivPaper.

    Returns None (and logs a warning) if the entry is malformed or missing
    required fields.
    """
    try:
        id_url = (entry.findtext("atom:id", namespaces=_NS) or "").strip()
        if not id_url:
            logger.warning("arXiv entry missing <id>; skipping")
            return None

        arxiv_id = _extract_arxiv_id(id_url)

        title_raw = entry.findtext("atom:title", namespaces=_NS) or ""
        title = " ".join(title_raw.split())  # normalise whitespace / newlines

        abstract_raw = entry.findtext("atom:summary", namespaces=_NS) or ""
        abstract = " ".join(abstract_raw.split())

        if not abstract:
            logger.warning("arXiv paper %s has empty abstract; skipping", arxiv_id)
            return None

        authors = [
            author.findtext("atom:name", namespaces=_NS) or ""
            for author in entry.findall("atom:author", namespaces=_NS)
        ]
        authors = [a for a in authors if a]

        categories = entry.findall("atom:category", namespaces=_NS)
        primary_category = categories[0].get("term", "") if categories else ""
        all_categories = [c.get("term", "") for c in categories if c.get("term")]

        published_raw = entry.findtext("atom:published", namespaces=_NS) or ""
        updated_raw = entry.findtext("atom:updated", namespaces=_NS) or ""

        if not published_raw:
            logger.warning("arXiv paper %s missing <published>; skipping", arxiv_id)
            return None

        published_date = _parse_date(published_raw)
        updated_date = _parse_date(updated_raw) if updated_raw else None

        comment = entry.findtext("arxiv:comment", namespaces=_NS)
        if comment:
            comment = " ".join(comment.split())

        return ArxivPaper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            primary_category=primary_category,
            all_categories=all_categories,
            published_date=published_date,
            updated_date=updated_date,
            arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            comment=comment,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse arXiv entry: %s", exc)
        return None


class ArxivFetcher:
    """Fetches paper metadata from the arXiv Atom API."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or default_settings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_with_retry(self, params: dict) -> requests.Response:
        """Execute a GET request to the arXiv API with exponential-backoff retry.

        Retries up to 3 times on HTTP 5xx or connection errors (delays: 3s, 9s, 27s).
        Raises immediately on HTTP 4xx.
        Raises RuntimeError after all retries are exhausted.
        """
        delays = [3, 9, 27]
        last_exc: Exception | None = None

        for attempt, delay in enumerate(delays):
            try:
                resp = requests.get(_ARXIV_API_URL, params=params, timeout=30)
            except requests.RequestException as exc:
                logger.warning(
                    "arXiv request error (attempt %d/3): %s — retrying in %ds",
                    attempt + 1, exc, delay,
                )
                last_exc = exc
                time.sleep(delay)
                continue

            if resp.status_code == 200:
                return resp

            if resp.status_code == 429:
                # Rate limited — retry with longer backoff
                retry_after = int(resp.headers.get("Retry-After", delay * 10))
                logger.warning(
                    "arXiv rate-limited (429, attempt %d/3) — waiting %ds",
                    attempt + 1, retry_after,
                )
                time.sleep(retry_after)
                last_exc = requests.HTTPError(response=resp)
                continue

            if 400 <= resp.status_code < 500:
                logger.error(
                    "arXiv returned HTTP %d (client error); aborting. params=%s",
                    resp.status_code, params,
                )
                resp.raise_for_status()

            # 5xx
            logger.warning(
                "arXiv returned HTTP %d (attempt %d/3) — retrying in %ds",
                resp.status_code, attempt + 1, delay,
            )
            last_exc = requests.HTTPError(response=resp)
            time.sleep(delay)

        raise RuntimeError(
            f"arXiv API unavailable after 3 attempts. Last error: {last_exc}"
        )

    def _parse_feed(self, xml_text: str) -> list[ArxivPaper]:
        """Parse an Atom XML feed string and return a list of ArxivPaper objects."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.error("Failed to parse arXiv XML feed: %s", exc)
            return []

        papers: list[ArxivPaper] = []
        for entry in root.findall("atom:entry", namespaces=_NS):
            paper = _parse_entry(entry)
            if paper is not None:
                papers.append(paper)
        return papers

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_recent(
        self,
        categories: list[str] | None = None,
        max_results: int | None = None,
    ) -> list[ArxivPaper]:
        """Fetch recent papers across the given categories, sorted by submission date.

        Implements the date-filter workaround: queries with sortBy=submittedDate
        descending, paginates until papers older than lookback_days are found,
        then post-filters in Python. Deduplicates by arxiv_id across categories.

        Args:
            categories: List of arXiv category codes (e.g. ['cs.LG', 'cs.AI']).
                        Defaults to config.arxiv_categories.
            max_results: Max results per page per category request.
                         Defaults to config.arxiv_max_results_per_category.

        Returns:
            Deduplicated list of ArxivPaper objects published within the lookback window.
        """
        if categories is None:
            categories = self._cfg.arxiv_categories
        if max_results is None:
            max_results = self._cfg.arxiv_max_results_per_category

        cutoff: date = date.today() - timedelta(days=self._cfg.arxiv_lookback_days)
        seen_ids: set[str] = set()
        all_papers: list[ArxivPaper] = []

        for category in categories:
            logger.info("Fetching arXiv papers for category: %s", category)
            start = 0
            stop_category = False

            while not stop_category:
                params = {
                    "search_query": f"cat:{category}",
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "start": start,
                    "max_results": max_results,
                }

                if start > 0:
                    logger.debug(
                        "Sleeping %.1fs between arXiv requests (rate limit)",
                        self._cfg.arxiv_fetch_delay,
                    )
                    time.sleep(self._cfg.arxiv_fetch_delay)

                resp = self._get_with_retry(params)
                page_papers = self._parse_feed(resp.text)

                if not page_papers:
                    logger.info(
                        "No more results for %s at start=%d", category, start
                    )
                    break

                for paper in page_papers:
                    paper_date = date.fromisoformat(paper.published_date)

                    if paper_date < cutoff:
                        logger.info(
                            "Reached papers older than lookback window for %s "
                            "(published %s < cutoff %s); stopping pagination",
                            category, paper_date, cutoff,
                        )
                        stop_category = True
                        break

                    if paper.arxiv_id not in seen_ids:
                        seen_ids.add(paper.arxiv_id)
                        all_papers.append(paper)

                if len(page_papers) < max_results:
                    # Fewer results than requested — no more pages
                    break

                start += max_results

            logger.info(
                "Category %s done. Total unique papers so far: %d", category, len(all_papers)
            )

            # Delay before the next category fetch
            if category != categories[-1]:
                time.sleep(self._cfg.arxiv_fetch_delay)

        logger.info(
            "fetch_recent complete. Total papers after dedup: %d", len(all_papers)
        )
        return all_papers

    def fetch_by_id(self, arxiv_id: str) -> ArxivPaper:
        """Fetch a single paper by arXiv ID.

        Used by seed.py --arxiv-id for manually adding papers to the knowledge bank.

        Args:
            arxiv_id: Canonical arXiv ID (e.g. '1706.03762').

        Returns:
            ArxivPaper for the requested ID.

        Raises:
            ValueError: If the paper is not found or the response is malformed.
        """
        params = {"id_list": arxiv_id}
        resp = self._get_with_retry(params)
        papers = self._parse_feed(resp.text)

        if not papers:
            raise ValueError(f"arXiv returned no results for id={arxiv_id!r}")

        return papers[0]

    def fetch_batch(self, arxiv_ids: list[str]) -> list[ArxivPaper]:
        """Fetch multiple papers by arXiv ID in a single batch request.

        Used during seeding to fetch landmark and survey papers. Respects rate
        limiting if the batch spans multiple pages (unlikely for typical seed sizes).

        Args:
            arxiv_ids: List of canonical arXiv IDs.

        Returns:
            List of ArxivPaper objects (order may differ from input).
        """
        if not arxiv_ids:
            return []

        # arXiv accepts comma-separated id_list; a batch of ≤50 fits in one request
        id_list = ",".join(arxiv_ids)
        params = {
            "id_list": id_list,
            "max_results": len(arxiv_ids),
        }

        resp = self._get_with_retry(params)
        papers = self._parse_feed(resp.text)

        if len(papers) < len(arxiv_ids):
            found_ids = {p.arxiv_id for p in papers}
            missing = [aid for aid in arxiv_ids if aid not in found_ids]
            logger.warning(
                "fetch_batch: requested %d papers, got %d. Missing: %s",
                len(arxiv_ids), len(papers), missing,
            )

        return papers
