"""PDF text extraction integration using PyMuPDF (fitz).

Used exclusively during the seeding phase to extract text from textbook chapters
by page range. Each call to extract_chapters returns one ChapterText per range.
Pages are 0-indexed and end_page is inclusive.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF

from src.models.seeding import ChapterText

logger = logging.getLogger(__name__)


class PdfExtractor:
    """Extracts text from specified page ranges of a PDF file.

    Uses PyMuPDF (fitz) for extraction. Designed for the seeding phase where
    textbook chapters are read by page range to populate the knowledge bank.
    """

    def extract_chapters(
        self,
        pdf_path: Path,
        chapter_ranges: list[tuple[int, int, str]],
    ) -> list[ChapterText]:
        """Extract text from specified page ranges of a PDF.

        Args:
            pdf_path: Path to the PDF file.
            chapter_ranges: List of (start_page, end_page, source_description) tuples.
                            Pages are 0-indexed, end_page is inclusive.
                            The source_description (3rd element) is stored verbatim on
                            the returned ChapterText, e.g.
                            "Goodfellow et al. Deep Learning, Ch. 3". This extends the
                            2-tuple interface shown in the TDD to avoid a parallel list.

        Returns:
            List of ChapterText objects, one per chapter range.
            Returns empty list if the PDF file does not exist or cannot be opened,
            or if chapter_ranges is empty.
        """
        if not chapter_ranges:
            return []

        if not pdf_path.exists():
            logger.error(
                "PDF file not found",
                extra={"pdf_path": str(pdf_path)},
            )
            return []

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.error(
                "Failed to open PDF",
                extra={"pdf_path": str(pdf_path), "error": str(exc)},
            )
            return []

        total_pages = doc.page_count
        results: list[ChapterText] = []

        try:
            for start_page, end_page, source_description in chapter_ranges:
                if start_page >= total_pages:
                    logger.warning(
                        "start_page out of range — skipping chapter range",
                        extra={
                            "start_page": start_page,
                            "total_pages": total_pages,
                            "source_description": source_description,
                        },
                    )
                    continue

                clamped_end = end_page
                if end_page >= total_pages:
                    clamped_end = total_pages - 1
                    logger.warning(
                        "end_page out of range — clamped to last page",
                        extra={
                            "requested_end_page": end_page,
                            "clamped_end_page": clamped_end,
                            "source_description": source_description,
                        },
                    )

                page_texts: list[str] = []
                for page_num in range(start_page, clamped_end + 1):
                    page = doc.load_page(page_num)
                    page_texts.append(page.get_text("text"))

                extracted_text = "\n".join(page_texts)
                pages_extracted = clamped_end - start_page + 1

                logger.info(
                    "Extracted chapter text",
                    extra={
                        "source_description": source_description,
                        "pages_extracted": pages_extracted,
                        "start_page": start_page,
                        "end_page": clamped_end,
                    },
                )

                results.append(
                    ChapterText(
                        text=extracted_text,
                        source_description=source_description,
                    )
                )
        finally:
            doc.close()

        return results
