"""Unit tests for HuggingFaceFetcher.

All tests use canned JSON responses — no live API calls are made.
HTTP is mocked via unittest.mock.patch("requests.get").
"""

import json
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from src.integrations.hf_client import HuggingFaceFetcher
from src.models.huggingface import HFPaper


# ---------------------------------------------------------------------------
# Helpers — canned data builders
# ---------------------------------------------------------------------------

def _make_paper_entry(
    arxiv_id: str = "2604.11521",
    title: str = "Continuous Adversarial Flow Models",
    upvotes: int = 1,
    ai_keywords: list[str] | None = None,
    ai_summary: str | None = "Continuous adversarial flow models improve image generation.",
    num_comments: int = 1,
) -> dict:
    """Build one array element as returned by the HF daily_papers endpoint."""
    if ai_keywords is None:
        ai_keywords = ["continuous-time flow models", "adversarial objective"]
    return {
        "paper": {
            "id": arxiv_id,
            "title": title,
            "upvotes": upvotes,
            "ai_keywords": ai_keywords,
            "ai_summary": ai_summary,
            "publishedAt": "2026-04-13T00:00:00.000Z",
        },
        "numComments": num_comments,
        "submittedBy": {},
    }


def _mock_json_response(data, status_code: int = 200) -> MagicMock:
    """Create a mock requests.Response that returns *data* from .json()."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = data
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=resp
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _mock_text_response(body: str, status_code: int = 200) -> MagicMock:
    """Create a mock requests.Response whose .json() raises ValueError (bad JSON)."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.side_effect = ValueError("No JSON object could be decoded")
    resp.raise_for_status = MagicMock()
    return resp


def _fetcher_no_sleep() -> HuggingFaceFetcher:
    """Return a HuggingFaceFetcher with hf_fetch_delay=0 for speed in most tests."""
    cfg = MagicMock()
    cfg.hf_fetch_delay = 0.0
    return HuggingFaceFetcher(cfg=cfg)


# ---------------------------------------------------------------------------
# fetch_daily_papers
# ---------------------------------------------------------------------------

class TestFetchDailyPapers:

    @patch("src.integrations.hf_client.requests.get")
    def test_fetch_daily_papers_success(self, mock_get):
        """Returns 2 HFPaper objects with correct field values."""
        entry1 = _make_paper_entry(
            arxiv_id="2604.11521",
            title="Continuous Adversarial Flow Models",
            upvotes=5,
            ai_keywords=["flow models"],
            ai_summary="A summary.",
            num_comments=2,
        )
        entry2 = _make_paper_entry(
            arxiv_id="2604.99999",
            title="Another Paper",
            upvotes=10,
            ai_keywords=["transformers"],
            ai_summary=None,
            num_comments=0,
        )
        mock_get.return_value = _mock_json_response([entry1, entry2])

        fetcher = _fetcher_no_sleep()
        papers = fetcher.fetch_daily_papers("2026-04-13")

        assert len(papers) == 2
        assert all(isinstance(p, HFPaper) for p in papers)

        p1 = papers[0]
        assert p1.arxiv_id == "2604.11521"
        assert p1.title == "Continuous Adversarial Flow Models"
        assert p1.upvotes == 5
        assert p1.ai_keywords == ["flow models"]
        assert p1.ai_summary == "A summary."
        assert p1.num_comments == 2

        p2 = papers[1]
        assert p2.arxiv_id == "2604.99999"
        assert p2.title == "Another Paper"
        assert p2.upvotes == 10
        assert p2.ai_summary is None
        assert p2.num_comments == 0

    @patch("src.integrations.hf_client.requests.get")
    def test_fetch_daily_papers_http_error(self, mock_get):
        """RequestException → returns empty list, does not raise."""
        mock_get.side_effect = requests.RequestException("connection refused")

        fetcher = _fetcher_no_sleep()
        result = fetcher.fetch_daily_papers("2026-04-13")

        assert result == []

    @patch("src.integrations.hf_client.requests.get")
    def test_fetch_daily_papers_4xx(self, mock_get):
        """HTTP 404 → returns empty list, does not raise."""
        mock_get.return_value = _mock_json_response({}, status_code=404)

        fetcher = _fetcher_no_sleep()
        result = fetcher.fetch_daily_papers("2026-04-13")

        assert result == []

    @patch("src.integrations.hf_client.requests.get")
    def test_fetch_daily_papers_5xx(self, mock_get):
        """HTTP 500 → returns empty list, does not raise."""
        mock_get.return_value = _mock_json_response({}, status_code=500)

        fetcher = _fetcher_no_sleep()
        result = fetcher.fetch_daily_papers("2026-04-13")

        assert result == []

    @patch("src.integrations.hf_client.requests.get")
    def test_fetch_daily_papers_invalid_json(self, mock_get):
        """200 with malformed JSON body → returns empty list, does not raise."""
        mock_get.return_value = _mock_text_response("not json {{{", status_code=200)

        fetcher = _fetcher_no_sleep()
        result = fetcher.fetch_daily_papers("2026-04-13")

        assert result == []

    @patch("src.integrations.hf_client.requests.get")
    def test_fetch_daily_papers_skips_malformed_entry(self, mock_get):
        """Entry missing paper.id is skipped; valid entry is returned."""
        good = _make_paper_entry(arxiv_id="2604.11521", title="Good Paper")
        bad = {
            "paper": {
                # no "id" key
                "title": "Bad Paper",
                "upvotes": 0,
            },
            "numComments": 0,
            "submittedBy": {},
        }
        mock_get.return_value = _mock_json_response([good, bad])

        fetcher = _fetcher_no_sleep()
        papers = fetcher.fetch_daily_papers("2026-04-13")

        assert len(papers) == 1
        assert papers[0].arxiv_id == "2604.11521"


# ---------------------------------------------------------------------------
# fetch_week
# ---------------------------------------------------------------------------

class TestFetchWeek:

    @patch("src.integrations.hf_client.time.sleep")
    @patch("src.integrations.hf_client.requests.get")
    def test_fetch_week_calls_7_days(self, mock_get, mock_sleep):
        """fetch_week('2026-04-14') calls the daily endpoint for 7 dates ending on 2026-04-14."""
        mock_get.return_value = _mock_json_response([])  # empty for all days

        fetcher = _fetcher_no_sleep()
        fetcher.fetch_week("2026-04-14")

        assert mock_get.call_count == 7
        called_dates = [
            call_args[1]["params"]["date"]
            for call_args in mock_get.call_args_list
        ]
        expected_dates = [
            "2026-04-14",
            "2026-04-13",
            "2026-04-12",
            "2026-04-11",
            "2026-04-10",
            "2026-04-09",
            "2026-04-08",
        ]
        assert called_dates == expected_dates

    @patch("src.integrations.hf_client.time.sleep")
    @patch("src.integrations.hf_client.requests.get")
    def test_fetch_week_deduplicates(self, mock_get, mock_sleep):
        """Same arXiv ID on two different days → appears once; first occurrence wins."""
        paper_day1 = _make_paper_entry(arxiv_id="2604.11521", upvotes=10)
        paper_day2 = _make_paper_entry(arxiv_id="2604.11521", upvotes=99)

        # Day 1 returns the duplicate; all other days return empty
        mock_get.side_effect = [
            _mock_json_response([paper_day1]),  # 2026-04-14
            _mock_json_response([paper_day2]),  # 2026-04-13 — same ID
            _mock_json_response([]),
            _mock_json_response([]),
            _mock_json_response([]),
            _mock_json_response([]),
            _mock_json_response([]),
        ]

        fetcher = _fetcher_no_sleep()
        result = fetcher.fetch_week("2026-04-14")

        assert len(result) == 1
        assert "2604.11521" in result
        # First occurrence wins → upvotes=10
        assert result["2604.11521"].upvotes == 10

    @patch("src.integrations.hf_client.time.sleep")
    @patch("src.integrations.hf_client.requests.get")
    def test_fetch_week_sleeps_between_calls(self, mock_get, mock_sleep):
        """Sleep is called exactly 6 times (between 7 day calls) with hf_fetch_delay."""
        mock_get.return_value = _mock_json_response([])

        cfg = MagicMock()
        cfg.hf_fetch_delay = 1.0
        fetcher = HuggingFaceFetcher(cfg=cfg)
        fetcher.fetch_week("2026-04-14")

        assert mock_sleep.call_count == 6
        for c in mock_sleep.call_args_list:
            assert c == call(1.0)

    @patch("src.integrations.hf_client.time.sleep")
    @patch("src.integrations.hf_client.requests.get")
    def test_fetch_week_returns_empty_on_total_failure(self, mock_get, mock_sleep):
        """All 7 daily fetches return HTTP errors → result is an empty dict."""
        mock_get.return_value = _mock_json_response({}, status_code=503)

        fetcher = _fetcher_no_sleep()
        result = fetcher.fetch_week("2026-04-14")

        assert result == {}


# ---------------------------------------------------------------------------
# lookup_paper
# ---------------------------------------------------------------------------

class TestLookupPaper:

    @patch("src.integrations.hf_client.requests.get")
    def test_lookup_paper_success(self, mock_get):
        """Valid single-paper JSON → returns correct HFPaper."""
        entry = _make_paper_entry(
            arxiv_id="2604.11521",
            title="Continuous Adversarial Flow Models",
            upvotes=3,
            ai_keywords=["flow models", "adversarial"],
            ai_summary="Great paper.",
            num_comments=5,
        )
        mock_get.return_value = _mock_json_response(entry)

        fetcher = _fetcher_no_sleep()
        paper = fetcher.lookup_paper("2604.11521")

        assert paper is not None
        assert isinstance(paper, HFPaper)
        assert paper.arxiv_id == "2604.11521"
        assert paper.title == "Continuous Adversarial Flow Models"
        assert paper.upvotes == 3
        assert paper.ai_keywords == ["flow models", "adversarial"]
        assert paper.ai_summary == "Great paper."
        assert paper.num_comments == 5

        # Verify the correct URL was called
        called_url = mock_get.call_args[0][0]
        assert "2604.11521" in called_url

    @patch("src.integrations.hf_client.requests.get")
    def test_lookup_paper_not_found(self, mock_get):
        """HTTP 404 → returns None, does not raise."""
        mock_get.return_value = _mock_json_response({}, status_code=404)

        fetcher = _fetcher_no_sleep()
        result = fetcher.lookup_paper("9999.99999")

        assert result is None

    @patch("src.integrations.hf_client.requests.get")
    def test_lookup_paper_connection_error(self, mock_get):
        """ConnectionError → returns None, does not raise."""
        mock_get.side_effect = requests.ConnectionError("connection refused")

        fetcher = _fetcher_no_sleep()
        result = fetcher.lookup_paper("2604.11521")

        assert result is None
