"""
document_processor.py - Document Text Extraction & Chunking
=============================================================
Supports: PDF, DOCX, TXT, PPTX, XLSX, CSV, MP4/Video, MP3/Audio

For video & audio files, OpenAI Whisper API is used for transcription.
Whisper limit: 25 MB per file. Larger files are skipped with a warning.
"""

import re
import csv
import logging
from pathlib import Path
from typing import Optional

import PyPDF2
from docx import Document as DocxDocument
from pptx import Presentation as PptxPresentation
from pptx.util import Pt
import openpyxl

from config import settings

logger = logging.getLogger(__name__)

# File types that use OpenAI Whisper for transcription
AUDIO_VIDEO_EXTENSIONS = {".mp4", ".mp3", ".m4a", ".wav", ".avi", ".webm", ".mpeg", ".wmv", ".mov"}

# Whisper API file size limit (25 MB)
WHISPER_MAX_BYTES = 25 * 1024 * 1024


class DocumentChunk:
    """
    Represents a single chunk (piece) of text from a document.

    Attributes:
        text: The actual text content
        source_file: Which file this came from
        chunk_index: Which chunk number this is (0, 1, 2...)
        char_start: Character position where this chunk starts in the original doc
        token_count: Approximate number of tokens (words) in this chunk
    """

    def __init__(
        self,
        text: str,
        source_file: str,
        chunk_index: int,
        char_start: int = 0,
    ):
        self.text = text
        self.source_file = source_file
        self.chunk_index = chunk_index
        self.char_start = char_start
        self.token_count = self._estimate_tokens(text)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count without calling OpenAI API.
        Rule of thumb: 1 token ≈ 4 characters in English text.
        """
        return len(text) // 4

    def __repr__(self) -> str:
        return (
            f"DocumentChunk(file='{self.source_file}', "
            f"index={self.chunk_index}, tokens={self.token_count})"
        )


class DocumentProcessor:
    """
    Processes documents into chunks ready for embedding and search.

    Usage:
        processor = DocumentProcessor()
        chunks = processor.process_file("/path/to/document.pdf")
    """

    def __init__(self):
        """Initialize with settings from config."""
        self.chunk_size = settings.chunk_size      # Max characters per chunk
        self.chunk_overlap = settings.chunk_overlap  # Overlap between chunks
        logger.info(
            f"DocumentProcessor ready "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def process_file(self, file_path: str) -> list[DocumentChunk]:
        """
        Process a single file: extract text, clean it, split into chunks.

        Args:
            file_path: Full path to the file (PDF, DOCX, or TXT)

        Returns:
            List of DocumentChunk objects ready for embedding.
            Returns empty list if the file can't be processed.
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        logger.info(f"Processing: {path.name}")

        # Step 1: Extract text based on file type
        raw_text = self._extract_text(path)

        if not raw_text:
            logger.warning(f"No text extracted from {path.name}")
            return []

        logger.info(f"  Extracted {len(raw_text):,} characters from {path.name}")

        # Step 2: Clean the text
        clean_text = self._clean_text(raw_text)
        logger.info(f"  After cleaning: {len(clean_text):,} characters")

        # Step 3: Split into chunks
        chunks = self._split_into_chunks(clean_text, source_file=path.name)
        logger.info(f"  Created {len(chunks)} chunks")

        return chunks

    def process_multiple_files(self, file_paths: list[str]) -> list[DocumentChunk]:
        """
        Process multiple files and combine all chunks.

        Args:
            file_paths: List of file paths to process

        Returns:
            All chunks from all files combined.
        """
        all_chunks = []
        total_files = len(file_paths)

        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing file {i}/{total_files}: {Path(file_path).name}")
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)

        logger.info(
            f"Processed {total_files} file(s), "
            f"created {len(all_chunks)} total chunks"
        )
        return all_chunks

    # =========================================================================
    # TEXT EXTRACTION - Different methods for different file types
    # =========================================================================

    def _extract_text(self, path: Path) -> str:
        """Route to the correct extractor based on file extension."""
        suffix = path.suffix.lower()

        extractors = {
            # Documents
            ".pdf":  self._extract_from_pdf,
            ".docx": self._extract_from_docx,
            ".doc":  self._extract_from_docx,
            ".txt":  self._extract_from_txt,
            # Presentations
            ".pptx": self._extract_from_pptx,
            ".ppt":  self._extract_from_pptx,
            # Spreadsheets
            ".xlsx": self._extract_from_xlsx,
            ".xls":  self._extract_from_xlsx,
            ".csv":  self._extract_from_csv,
            # Video & Audio (Whisper transcription)
            ".mp4":  self._extract_from_audio_video,
            ".mp3":  self._extract_from_audio_video,
            ".m4a":  self._extract_from_audio_video,
            ".wav":  self._extract_from_audio_video,
            ".avi":  self._extract_from_audio_video,
            ".webm": self._extract_from_audio_video,
            ".mpeg": self._extract_from_audio_video,
            ".wmv":  self._extract_from_audio_video,
            ".mov":  self._extract_from_audio_video,
        }

        extractor = extractors.get(suffix)
        if not extractor:
            logger.warning(f"Unsupported file type: {suffix} — skipping {path.name}")
            return ""

        try:
            return extractor(path)
        except Exception as e:
            logger.error(f"Error extracting text from {path.name}: {e}")
            return ""

    def _extract_from_pdf(self, path: Path) -> str:
        """
        Extract all text from a PDF file page by page.

        Some PDFs are scanned images - those won't have text and
        will return empty strings. You'd need OCR for those.
        """
        logger.debug(f"  Extracting text from PDF: {path.name}")
        text_parts = []

        with open(path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(reader.pages)
            logger.debug(f"  PDF has {total_pages} page(s)")

            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        # Add a page marker so we know where text came from
                        text_parts.append(f"\n[Page {page_num}]\n{page_text}")
                    else:
                        logger.debug(f"  Page {page_num}: no text (might be a scanned image)")
                except Exception as e:
                    logger.warning(f"  Could not read page {page_num}: {e}")

        return "\n".join(text_parts)

    def _extract_from_docx(self, path: Path) -> str:
        """
        Extract all text from a DOCX file, including tables.

        DOCX files have paragraphs AND tables. We extract both
        to make sure we don't miss anything.
        """
        logger.debug(f"  Extracting text from DOCX: {path.name}")

        doc = DocxDocument(str(path))
        text_parts = []

        # Extract regular paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                # Keep heading styles for context
                if para.style.name.startswith("Heading"):
                    text_parts.append(f"\n## {para.text.strip()}")
                else:
                    text_parts.append(para.text.strip())

        # Extract text from tables (common in HR policy docs!)
        for table_num, table in enumerate(doc.tables, 1):
            text_parts.append(f"\n[Table {table_num}]")
            for row in table.rows:
                # Join cells with | to make it readable
                row_text = " | ".join(
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                )
                if row_text:
                    text_parts.append(row_text)

        return "\n".join(text_parts)

    def _extract_from_txt(self, path: Path) -> str:
        """Read a plain text file (UTF-8, fallback Latin-1)."""
        logger.debug(f"  Reading TXT file: {path.name}")
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")

    def _extract_from_pptx(self, path: Path) -> str:
        """
        Extract all text from a PowerPoint file (PPTX).

        Extracts text from:
          - Every slide's text boxes and shapes
          - Tables inside slides
          - Speaker notes (often contain key context!)
        """
        logger.debug(f"  Extracting text from PPTX: {path.name}")
        prs = PptxPresentation(str(path))
        text_parts = []

        for slide_num, slide in enumerate(prs.slides, 1):
            slide_texts = []
            text_parts.append(f"\n[Slide {slide_num}]")

            for shape in slide.shapes:
                # Text frames (title, content boxes)
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        line = para.text.strip()
                        if line:
                            slide_texts.append(line)

                # Tables inside slides
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = " | ".join(
                            cell.text.strip()
                            for cell in row.cells
                            if cell.text.strip()
                        )
                        if row_text:
                            slide_texts.append(row_text)

            text_parts.extend(slide_texts)

            # Speaker notes — often contain detailed explanations
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    text_parts.append(f"[Speaker Notes: {notes}]")

        result = "\n".join(text_parts)
        logger.debug(f"  Extracted {len(result):,} chars from {len(prs.slides)} slides")
        return result

    def _extract_from_xlsx(self, path: Path) -> str:
        """
        Extract all text from an Excel file (XLSX).

        Reads every sheet, every row, every cell.
        Column headers are preserved so the data makes sense in context.
        """
        logger.debug(f"  Extracting text from XLSX: {path.name}")
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        text_parts = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            text_parts.append(f"\n[Sheet: {sheet_name}]")
            rows_with_data = 0

            for row in ws.iter_rows(values_only=True):
                # Convert all cells to strings, skip empty rows
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    text_parts.append(" | ".join(cells))
                    rows_with_data += 1

                # Stop after 2000 rows to avoid huge spreadsheets
                if rows_with_data >= 2000:
                    text_parts.append("[...truncated after 2000 rows...]")
                    break

        wb.close()
        return "\n".join(text_parts)

    def _extract_from_csv(self, path: Path) -> str:
        """Extract text from a CSV file, preserving headers."""
        logger.debug(f"  Extracting text from CSV: {path.name}")
        text_parts = []

        try:
            with open(path, encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if row:
                        text_parts.append(" | ".join(cell.strip() for cell in row))
                    if i >= 2000:
                        text_parts.append("[...truncated...]")
                        break
        except Exception as e:
            logger.error(f"CSV read error: {e}")

        return "\n".join(text_parts)

    def _extract_from_audio_video(self, path: Path) -> str:
        """
        Transcribe audio or video files using OpenAI Whisper API.

        HOW IT WORKS:
          1. The file (MP4, MP3, WAV, etc.) is sent to OpenAI's Whisper model
          2. Whisper converts speech to text (supports 99 languages)
          3. The transcript is returned and chunked like any other document

        LIMITS:
          - Max file size: 25 MB (OpenAI Whisper API limit)
          - Larger files are skipped with a warning
          - Supported: mp4, mp3, m4a, wav, avi, webm, mpeg, wmv, mov

        COST: ~$0.006 per minute of audio (very cheap)
        """
        logger.info(f"  Transcribing audio/video with Whisper: {path.name}")

        # Check file size first
        file_size = path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        if file_size > WHISPER_MAX_BYTES:
            logger.warning(
                f"  {path.name} is {file_size_mb:.1f} MB — exceeds Whisper's 25 MB limit. "
                f"Skipping. To process large videos, compress them to under 25 MB first."
            )
            return f"[Video/Audio file '{path.name}' ({file_size_mb:.1f} MB) was too large to transcribe. Max: 25 MB]"

        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)

            logger.info(f"  Uploading {file_size_mb:.1f} MB to Whisper API...")

            with open(path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                )

            # transcript is a plain string
            text = str(transcript).strip()
            logger.info(f"  Transcription complete: {len(text):,} characters")
            return f"[Transcript of {path.name}]\n{text}"

        except Exception as e:
            logger.error(f"  Whisper transcription failed for {path.name}: {e}")
            return f"[Transcription failed for {path.name}: {e}]"

    # =========================================================================
    # TEXT CLEANING
    # =========================================================================

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text to improve search quality.

        Removes:
          - Multiple blank lines (keeps single blank lines)
          - Multiple spaces (keeps single spaces)
          - Garbage characters from PDF extraction
          - Leading/trailing whitespace

        Preserves:
          - Paragraph breaks (important for context)
          - Punctuation
          - Numbers
        """
        if not text:
            return ""

        # Remove non-printable characters except newlines and tabs
        text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)

        # Replace tab characters with spaces
        text = text.replace("\t", " ")

        # Collapse multiple spaces into one
        text = re.sub(r"[ ]{2,}", " ", text)

        # Collapse more than 2 consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove lines that are just whitespace or single characters
        lines = text.split("\n")
        lines = [line for line in lines if len(line.strip()) > 1]
        text = "\n".join(lines)

        # Final trim
        return text.strip()

    # =========================================================================
    # TEXT CHUNKING
    # =========================================================================

    def _split_into_chunks(
        self,
        text: str,
        source_file: str,
    ) -> list[DocumentChunk]:
        """
        Split text into overlapping chunks for embedding.

        WHY OVERLAPPING?
          If a sentence spans the boundary between two chunks,
          the overlap ensures both chunks contain it.
          This prevents losing information at chunk boundaries.

        Example with chunk_size=20, overlap=5:
          Text: "The quick brown fox jumps over the lazy dog"
          Chunk 1: "The quick brown fox " (chars 0-19)
          Chunk 2: "n fox jumps over the" (chars 15-34)  <- 5 char overlap
          Chunk 3: "r the lazy dog"       (chars 30-end) <- 5 char overlap

        Args:
            text: The cleaned text to split
            source_file: Filename to attach to each chunk

        Returns:
            List of DocumentChunk objects.
        """
        if not text:
            return []

        chunks = []
        chunk_index = 0
        start = 0
        text_length = len(text)

        while start < text_length:
            # Define the end of this chunk
            end = start + self.chunk_size

            if end >= text_length:
                # Last chunk - take everything that's left
                chunk_text = text[start:]
            else:
                # Try to end at a sentence boundary (period, !, ?)
                # Look backwards from `end` for a punctuation mark
                boundary = self._find_sentence_boundary(text, end)
                chunk_text = text[start:boundary]
                end = boundary

            # Only create chunk if it has meaningful content
            chunk_text = chunk_text.strip()
            if len(chunk_text) > 20:  # Skip tiny chunks
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        char_start=start,
                    )
                )
                chunk_index += 1

            # Move forward, but back up by `overlap` characters
            # so the next chunk overlaps with this one
            if end >= text_length:
                break
            start = end - self.chunk_overlap

        return chunks

    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """
        Find the nearest sentence boundary at or before `position`.

        Looks backwards up to 100 characters for a '.', '!', or '?'
        followed by a space or newline. Falls back to the original
        position if no boundary is found.

        Args:
            text: The full text
            position: The target split position

        Returns:
            The best position to split at.
        """
        # Search backwards up to 100 characters for sentence end
        search_start = max(0, position - 100)
        search_text = text[search_start:position]

        # Find the last sentence-ending punctuation
        # Regex: period/!/? followed by space or newline
        matches = list(re.finditer(r"[.!?]\s", search_text))

        if matches:
            # Use the last match (closest to our target position)
            last_match = matches[-1]
            return search_start + last_match.end()

        # No sentence boundary found - just split at the target position
        return position

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def count_tokens_estimate(self, text: str) -> int:
        """
        Estimate token count for a text string.
        Uses the 4-chars-per-token rule of thumb.
        For exact counts, you'd need tiktoken library.
        """
        return len(text) // 4

    def get_processing_stats(self, chunks: list[DocumentChunk]) -> dict:
        """
        Get statistics about a batch of processed chunks.

        Useful for logging and API responses.
        """
        if not chunks:
            return {"total_chunks": 0, "total_tokens": 0, "files": []}

        files = list({c.source_file for c in chunks})
        total_tokens = sum(c.token_count for c in chunks)

        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "total_files": len(files),
            "files": files,
            "avg_tokens_per_chunk": total_tokens // len(chunks) if chunks else 0,
        }


# =============================================================================
# QUICK TEST - Run this file directly to test document processing
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        processor = DocumentProcessor()
        chunks = processor.process_file(test_file)
        stats = processor.get_processing_stats(chunks)
        print(f"\nResults: {stats}")
        if chunks:
            print(f"\nFirst chunk preview:\n{chunks[0].text[:200]}...")
    else:
        print("Usage: python document_processor.py <path-to-document>")
        print("Example: python document_processor.py my_policy.pdf")
