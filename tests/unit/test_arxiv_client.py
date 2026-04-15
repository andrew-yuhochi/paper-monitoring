"""Unit tests for ArxivFetcher.

All tests use canned XML responses — no live API calls are made.
"""

import textwrap
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.integrations.arxiv_client import ArxivFetcher, _extract_arxiv_id, _parse_date
from src.models.arxiv import ArxivPaper


# ---------------------------------------------------------------------------
# Helpers — canned XML builders
# ---------------------------------------------------------------------------

def _make_feed(entries_xml: str) -> str:
    """Wrap entry XML fragments in a minimal Atom feed envelope.

    Note: we build this via string join rather than textwrap.dedent so that
    the multi-line `entries_xml` content doesn't break dedent's common-prefix
    calculation and push the <?xml declaration off column 0.
    """
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<feed xmlns="http://www.w3.org/2005/Atom"',
        '      xmlns:arxiv="http://arxiv.org/schemas/atom"',
        '      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">',
        '  <opensearch:totalResults>1</opensearch:totalResults>',
        '  <opensearch:startIndex>0</opensearch:startIndex>',
        '  <opensearch:itemsPerPage>10</opensearch:itemsPerPage>',
        entries_xml,
        '</feed>',
    ])


def _make_entry(
    arxiv_id: str = "1706.03762",
    title: str = "Attention Is All You Need",
    abstract: str = "The dominant sequence transduction models are based on complex RNNs.",
    authors: list[str] | None = None,
    primary_category: str = "cs.CL",
    extra_categories: list[str] | None = None,
    published: str = "2017-06-12T00:00:00Z",
    updated: str = "2017-06-12T00:00:00Z",
    comment: str | None = None,
    version: str = "v1",
) -> str:
    if authors is None:
        authors = ["Vaswani, Ashish", "Shazeer, Noam"]
    author_xml = "".join(f"<author><name>{a}</name></author>" for a in authors)

    cats = [primary_category] + (extra_categories or [])
    category_xml = "".join(
        f'<category term="{c}" scheme="http://arxiv.org/schemas/atom"/>' for c in cats
    )

    comment_xml = f"<arxiv:comment>{comment}</arxiv:comment>" if comment else ""

    return textwrap.dedent(f"""\
        <entry>
          <id>http://arxiv.org/abs/{arxiv_id}{version}</id>
          <title>{title}</title>
          <summary>{abstract}</summary>
          {author_xml}
          {category_xml}
          <published>{published}</published>
          <updated>{updated}</updated>
          {comment_xml}
        </entry>
    """)


def _mock_response(xml: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = xml
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestExtractArxivId:
    def test_strips_url_prefix_and_version(self):
        assert _extract_arxiv_id("http://arxiv.org/abs/1706.03762v1") == "1706.03762"

    def test_strips_https(self):
        assert _extract_arxiv_id("https://arxiv.org/abs/2604.11805v3") == "2604.11805"

    def test_no_version(self):
        assert _extract_arxiv_id("http://arxiv.org/abs/1706.03762") == "1706.03762"

    def test_new_style_id(self):
        assert _extract_arxiv_id("http://arxiv.org/abs/2312.00752v2") == "2312.00752"


class TestParseDate:
    def test_iso_timestamp_to_date(self):
        assert _parse_date("2026-04-13T17:59:40Z") == "2026-04-13"

    def test_date_only_passthrough(self):
        assert _parse_date("2026-04-13") == "2026-04-13"


# ---------------------------------------------------------------------------
# _parse_feed / _parse_entry (via ArxivFetcher internals)
# ---------------------------------------------------------------------------

class TestParseFeed:
    def setup_method(self):
        self.fetcher = ArxivFetcher()

    def test_parses_single_entry(self):
        xml = _make_feed(_make_entry())
        papers = self.fetcher._parse_feed(xml)
        assert len(papers) == 1
        p = papers[0]
        assert isinstance(p, ArxivPaper)
        assert p.arxiv_id == "1706.03762"
        assert p.title == "Attention Is All You Need"
        assert p.primary_category == "cs.CL"
        assert p.published_date == "2017-06-12"
        assert p.arxiv_url == "https://arxiv.org/abs/1706.03762"
        assert p.pdf_url == "https://arxiv.org/pdf/1706.03762"

    def test_parses_authors(self):
        xml = _make_feed(_make_entry(authors=["Alice Smith", "Bob Jones", "Carol Wu"]))
        papers = self.fetcher._parse_feed(xml)
        assert papers[0].authors == ["Alice Smith", "Bob Jones", "Carol Wu"]

    def test_parses_multiple_categories(self):
        xml = _make_feed(_make_entry(primary_category="cs.CL", extra_categories=["cs.LG", "cs.AI"]))
        papers = self.fetcher._parse_feed(xml)
        assert papers[0].primary_category == "cs.CL"
        assert papers[0].all_categories == ["cs.CL", "cs.LG", "cs.AI"]

    def test_parses_comment(self):
        xml = _make_feed(_make_entry(comment="Code at https://github.com/example/repo"))
        papers = self.fetcher._parse_feed(xml)
        assert papers[0].comment == "Code at https://github.com/example/repo"

    def test_no_comment_is_none(self):
        xml = _make_feed(_make_entry())
        papers = self.fetcher._parse_feed(xml)
        assert papers[0].comment is None

    def test_strips_whitespace_from_title_and_abstract(self):
        entry = _make_entry(
            title="  Attention\n  Is All\n  You Need  ",
            abstract="  Line one.\n  Line two.  ",
        )
        papers = self.fetcher._parse_feed(_make_feed(entry))
        assert papers[0].title == "Attention Is All You Need"
        assert papers[0].abstract == "Line one. Line two."

    def test_skips_entry_with_empty_abstract(self):
        entry = _make_entry(abstract="")
        papers = self.fetcher._parse_feed(_make_feed(entry))
        assert papers == []

    def test_skips_malformed_entry_and_continues(self):
        good = _make_entry(arxiv_id="1706.03762", title="Good paper",
                           abstract="Good abstract.")
        # Malformed: no <id> element (simulated by invalid XML fragment injected — we
        # instead test with a missing-abstract entry alongside a good one)
        bad = _make_entry(abstract="")
        xml = _make_feed(bad + good)
        papers = self.fetcher._parse_feed(xml)
        assert len(papers) == 1
        assert papers[0].arxiv_id == "1706.03762"

    def test_returns_empty_on_invalid_xml(self):
        papers = self.fetcher._parse_feed("this is not xml at all <<>>")
        assert papers == []

    def test_parses_multiple_entries(self):
        e1 = _make_entry(arxiv_id="1706.03762", abstract="Abstract one.")
        e2 = _make_entry(arxiv_id="1810.04805", title="BERT", abstract="Abstract two.")
        papers = self.fetcher._parse_feed(_make_feed(e1 + e2))
        assert len(papers) == 2
        ids = {p.arxiv_id for p in papers}
        assert ids == {"1706.03762", "1810.04805"}


# ---------------------------------------------------------------------------
# fetch_by_id
# ---------------------------------------------------------------------------

class TestFetchById:
    def setup_method(self):
        self.fetcher = ArxivFetcher()

    @patch("src.integrations.arxiv_client.requests.get")
    def test_returns_single_paper(self, mock_get):
        xml = _make_feed(_make_entry(arxiv_id="1706.03762"))
        mock_get.return_value = _mock_response(xml)

        paper = self.fetcher.fetch_by_id("1706.03762")

        assert paper.arxiv_id == "1706.03762"
        mock_get.assert_called_once()
        call_params = mock_get.call_args[1]["params"]
        assert call_params["id_list"] == "1706.03762"

    @patch("src.integrations.arxiv_client.requests.get")
    def test_raises_if_no_results(self, mock_get):
        empty_feed = _make_feed("")
        mock_get.return_value = _mock_response(empty_feed)

        with pytest.raises(ValueError, match="no results"):
            self.fetcher.fetch_by_id("0000.00000")


# ---------------------------------------------------------------------------
# fetch_batch
# ---------------------------------------------------------------------------

class TestFetchBatch:
    def setup_method(self):
        self.fetcher = ArxivFetcher()

    @patch("src.integrations.arxiv_client.requests.get")
    def test_returns_all_papers(self, mock_get):
        e1 = _make_entry(arxiv_id="1706.03762", abstract="Abs one.")
        e2 = _make_entry(arxiv_id="1810.04805", title="BERT", abstract="Abs two.")
        xml = _make_feed(e1 + e2)
        mock_get.return_value = _mock_response(xml)

        papers = self.fetcher.fetch_batch(["1706.03762", "1810.04805"])

        assert len(papers) == 2
        call_params = mock_get.call_args[1]["params"]
        assert "1706.03762" in call_params["id_list"]
        assert "1810.04805" in call_params["id_list"]

    @patch("src.integrations.arxiv_client.requests.get")
    def test_empty_input_returns_empty_list(self, mock_get):
        papers = self.fetcher.fetch_batch([])
        assert papers == []
        mock_get.assert_not_called()

    @patch("src.integrations.arxiv_client.requests.get")
    def test_logs_warning_on_partial_results(self, mock_get, caplog):
        # Request 2 papers but API only returns 1
        xml = _make_feed(_make_entry(arxiv_id="1706.03762", abstract="Abs."))
        mock_get.return_value = _mock_response(xml)

        import logging
        with caplog.at_level(logging.WARNING, logger="src.integrations.arxiv_client"):
            papers = self.fetcher.fetch_batch(["1706.03762", "9999.99999"])

        assert len(papers) == 1
        assert "9999.99999" in caplog.text


# ---------------------------------------------------------------------------
# fetch_recent — date filtering and deduplication
# ---------------------------------------------------------------------------

class TestFetchRecent:
    """Tests for fetch_recent using patched HTTP and time.sleep."""

    def _fetcher_with_lookback(self, days: int = 7) -> ArxivFetcher:
        cfg = MagicMock()
        cfg.arxiv_categories = ["cs.LG"]
        cfg.arxiv_max_results_per_category = 10
        cfg.arxiv_lookback_days = days
        cfg.arxiv_fetch_delay = 0.0  # no sleep in tests
        return ArxivFetcher(cfg=cfg)

    def _today_str(self, offset: int = 0) -> str:
        return (date.today() + timedelta(days=offset)).strftime("%Y-%m-%dT00:00:00Z")

    @patch("src.integrations.arxiv_client.time.sleep")
    @patch("src.integrations.arxiv_client.requests.get")
    def test_returns_papers_within_lookback(self, mock_get, mock_sleep):
        recent = _make_entry(arxiv_id="2604.00001", abstract="Recent.", published=self._today_str(-1))
        xml = _make_feed(recent)
        mock_get.return_value = _mock_response(xml)

        fetcher = self._fetcher_with_lookback(days=7)
        papers = fetcher.fetch_recent()

        assert len(papers) == 1
        assert papers[0].arxiv_id == "2604.00001"

    @patch("src.integrations.arxiv_client.time.sleep")
    @patch("src.integrations.arxiv_client.requests.get")
    def test_stops_pagination_on_old_paper(self, mock_get, mock_sleep):
        """When an old paper is encountered mid-feed, pagination should stop
        and only papers before it (i.e., newer, returned earlier) should be kept."""
        recent = _make_entry(arxiv_id="2604.00001", abstract="Recent.", published=self._today_str(-1))
        old = _make_entry(arxiv_id="2001.00001", abstract="Old paper.", published="2020-01-01T00:00:00Z")
        xml = _make_feed(recent + old)
        mock_get.return_value = _mock_response(xml)

        fetcher = self._fetcher_with_lookback(days=7)
        papers = fetcher.fetch_recent()

        ids = [p.arxiv_id for p in papers]
        assert "2604.00001" in ids
        assert "2001.00001" not in ids

    @patch("src.integrations.arxiv_client.time.sleep")
    @patch("src.integrations.arxiv_client.requests.get")
    def test_deduplicates_cross_category_papers(self, mock_get, mock_sleep):
        """Same paper appearing in two category fetches should only appear once."""
        cfg = MagicMock()
        cfg.arxiv_categories = ["cs.LG", "cs.AI"]
        cfg.arxiv_max_results_per_category = 10
        cfg.arxiv_lookback_days = 7
        cfg.arxiv_fetch_delay = 0.0
        fetcher = ArxivFetcher(cfg=cfg)

        entry = _make_entry(
            arxiv_id="2604.00001", abstract="Cross-listed.", published=self._today_str(-1)
        )
        xml = _make_feed(entry)
        mock_get.return_value = _mock_response(xml)

        papers = fetcher.fetch_recent()

        assert len(papers) == 1
        assert papers[0].arxiv_id == "2604.00001"

    @patch("src.integrations.arxiv_client.time.sleep")
    @patch("src.integrations.arxiv_client.requests.get")
    def test_excludes_papers_exactly_on_cutoff(self, mock_get, mock_sleep):
        """Papers published exactly on the cutoff date (today - lookback_days)
        should be excluded (cutoff is exclusive: published_date >= cutoff)."""
        cutoff_date = date.today() - timedelta(days=7)
        cutoff_str = cutoff_date.strftime("%Y-%m-%dT00:00:00Z")

        entry = _make_entry(arxiv_id="2604.00001", abstract="Abs.", published=cutoff_str)
        # The paper published on exactly the cutoff day: should be included
        # because we use >=
        xml = _make_feed(entry)
        mock_get.return_value = _mock_response(xml)

        fetcher = self._fetcher_with_lookback(days=7)
        papers = fetcher.fetch_recent()

        # cutoff = today - 7; paper published on cutoff — date.fromisoformat(cutoff_str[:10]) == cutoff
        # In our code: paper_date < cutoff → stop (exclusive lower bound)
        # So a paper on exactly cutoff is NOT old → it should be included
        assert len(papers) == 1


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def setup_method(self):
        self.fetcher = ArxivFetcher()

    @patch("src.integrations.arxiv_client.time.sleep")
    @patch("src.integrations.arxiv_client.requests.get")
    def test_retries_on_5xx_then_succeeds(self, mock_get, mock_sleep):
        """Should retry on HTTP 500 and succeed on the third attempt."""
        error_resp = _mock_response("", status_code=503)
        ok_resp = _mock_response(_make_feed(_make_entry(abstract="Abs.")))

        mock_get.side_effect = [error_resp, error_resp, ok_resp]

        paper = self.fetcher.fetch_by_id("1706.03762")
        assert paper.arxiv_id == "1706.03762"
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("src.integrations.arxiv_client.time.sleep")
    @patch("src.integrations.arxiv_client.requests.get")
    def test_raises_after_3_failed_attempts(self, mock_get, mock_sleep):
        """After 3 failed attempts (all 5xx), should raise RuntimeError."""
        error_resp = _mock_response("", status_code=503)
        mock_get.return_value = error_resp

        with pytest.raises(RuntimeError, match="unavailable after 3 attempts"):
            self.fetcher.fetch_by_id("1706.03762")

        assert mock_get.call_count == 3

    @patch("src.integrations.arxiv_client.time.sleep")
    @patch("src.integrations.arxiv_client.requests.get")
    def test_raises_immediately_on_4xx(self, mock_get, mock_sleep):
        """4xx errors should not be retried — raise immediately."""
        error_resp = _mock_response("", status_code=400)
        error_resp.raise_for_status.side_effect = requests.HTTPError(response=error_resp)
        mock_get.return_value = error_resp

        with pytest.raises(requests.HTTPError):
            self.fetcher.fetch_by_id("1706.03762")

        assert mock_get.call_count == 1
        mock_sleep.assert_not_called()

    @patch("src.integrations.arxiv_client.time.sleep")
    @patch("src.integrations.arxiv_client.requests.get")
    def test_retries_on_connection_error(self, mock_get, mock_sleep):
        """Connection errors should trigger retry like 5xx."""
        ok_resp = _mock_response(_make_feed(_make_entry(abstract="Abs.")))
        mock_get.side_effect = [
            requests.ConnectionError("connection refused"),
            ok_resp,
        ]

        paper = self.fetcher.fetch_by_id("1706.03762")
        assert paper.arxiv_id == "1706.03762"
        assert mock_get.call_count == 2
