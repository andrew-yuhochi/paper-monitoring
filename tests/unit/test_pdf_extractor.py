"""Unit tests for PdfExtractor.

All tests create a synthetic PDF in-memory via PyMuPDF — no external files needed.
"""

from pathlib import Path

import fitz
import pytest

from src.integrations.pdf_extractor import PdfExtractor
from src.models.seeding import ChapterText


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_pdf(tmp_path: Path, num_pages: int = 4) -> Path:
    """Create a minimal PDF with `num_pages` pages, each containing unique text."""
    doc = fitz.open()  # new empty document
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i} content. Unique text for page {i}.")
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPdfExtractor:
    """Tests for PdfExtractor.extract_chapters."""

    def test_extract_single_range(self, tmp_path: Path) -> None:
        """Extracting pages 0–1 returns one ChapterText with content from both pages."""
        pdf_path = _make_test_pdf(tmp_path)
        extractor = PdfExtractor()

        results = extractor.extract_chapters(
            pdf_path,
            [(0, 1, "Test Book, Ch. 1")],
        )

        assert len(results) == 1
        chapter = results[0]
        assert isinstance(chapter, ChapterText)
        assert "Page 0" in chapter.text
        assert "Page 1" in chapter.text
        assert chapter.source_description == "Test Book, Ch. 1"

    def test_extract_multiple_ranges(self, tmp_path: Path) -> None:
        """Two ranges return two ChapterText objects with the correct page content each."""
        pdf_path = _make_test_pdf(tmp_path)
        extractor = PdfExtractor()

        results = extractor.extract_chapters(
            pdf_path,
            [
                (0, 1, "Test Book, Ch. 1"),
                (2, 3, "Test Book, Ch. 2"),
            ],
        )

        assert len(results) == 2

        ch1, ch2 = results
        assert "Page 0" in ch1.text
        assert "Page 1" in ch1.text
        assert ch1.source_description == "Test Book, Ch. 1"

        assert "Page 2" in ch2.text
        assert "Page 3" in ch2.text
        assert ch2.source_description == "Test Book, Ch. 2"

    def test_single_page_range(self, tmp_path: Path) -> None:
        """A range of (1, 1) extracts exactly page 1 and nothing else."""
        pdf_path = _make_test_pdf(tmp_path)
        extractor = PdfExtractor()

        results = extractor.extract_chapters(
            pdf_path,
            [(1, 1, "Single Page")],
        )

        assert len(results) == 1
        text = results[0].text
        assert "Page 1 content" in text
        assert "Page 0" not in text
        assert "Page 2" not in text

    def test_missing_pdf_returns_empty(self, tmp_path: Path) -> None:
        """A path that does not exist returns [] without raising."""
        missing = tmp_path / "does_not_exist.pdf"
        extractor = PdfExtractor()

        results = extractor.extract_chapters(missing, [(0, 1, "Any Source")])

        assert results == []

    def test_out_of_bounds_start_skips_range(self, tmp_path: Path) -> None:
        """start_page >= total_pages causes that range to be skipped entirely."""
        pdf_path = _make_test_pdf(tmp_path, num_pages=4)  # pages 0-3
        extractor = PdfExtractor()

        results = extractor.extract_chapters(
            pdf_path,
            [(10, 12, "Out of Bounds")],
        )

        assert results == []

    def test_out_of_bounds_end_clamped(self, tmp_path: Path) -> None:
        """end_page beyond the last page is clamped; result includes valid pages without raising."""
        pdf_path = _make_test_pdf(tmp_path, num_pages=4)  # pages 0-3
        extractor = PdfExtractor()

        results = extractor.extract_chapters(
            pdf_path,
            [(2, 99, "Clamped Range")],
        )

        assert len(results) == 1
        text = results[0].text
        # Pages 2 and 3 should be present (clamped to last page = 3)
        assert "Page 2" in text
        assert "Page 3" in text

    def test_source_description_preserved(self, tmp_path: Path) -> None:
        """The source_description field on the returned ChapterText exactly matches the input."""
        pdf_path = _make_test_pdf(tmp_path)
        source = "Goodfellow et al. Deep Learning, Ch. 3"
        extractor = PdfExtractor()

        results = extractor.extract_chapters(pdf_path, [(0, 0, source)])

        assert len(results) == 1
        assert results[0].source_description == source

    def test_empty_chapter_ranges(self, tmp_path: Path) -> None:
        """Passing an empty chapter_ranges list returns []."""
        pdf_path = _make_test_pdf(tmp_path)
        extractor = PdfExtractor()

        results = extractor.extract_chapters(pdf_path, [])

        assert results == []
