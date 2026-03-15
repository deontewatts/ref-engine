"""
REF File Parser
---------------
Extracts structured text content from PDF and plain-text files.
Produces a ParsedDocument with:
  - full_text: concatenated string
  - pages: list of per-page text
  - metadata: title, page count, file size, etc.
  - structure: detected headings, sections, equations
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path


@dataclass
class DocumentSection:
    """A structural unit of the document — a heading + its body text."""
    heading: str
    body: str
    page_index: int
    char_offset: int

    @property
    def full_text(self) -> str:
        return f"{self.heading}\n{self.body}"

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


@dataclass
class ParsedDocument:
    """
    The complete parsed representation of a file.
    This is the input object to the REF scoring pipeline.
    """
    filepath: str
    filename: str
    file_size_bytes: int
    full_text: str
    pages: List[str] = field(default_factory=list)
    sections: List[DocumentSection] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    parse_errors: List[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def equation_count(self) -> int:
        """Count mathematical expressions as a proxy for technical depth."""
        patterns = [
            r"[A-Za-z]\s*=\s*[^=\n]{3,}",   # variable = expression
            r"\b[A-Z]\([A-Za-z|,\s]+\)",       # function notation F(x|y)
            r"∑|∫|∂|∇|∈|∀|∃|→|⟨|⟩|≤|≥|≠",   # math symbols
            r"\d+\.\d+e[+-]\d+",               # scientific notation
        ]
        return sum(len(re.findall(p, self.full_text)) for p in patterns)

    @property
    def heading_count(self) -> int:
        return len(self.sections)

    def get_chunk(self, chunk_size: int = 500) -> List[str]:
        """Split full text into overlapping chunks for per-chunk analysis."""
        words = self.full_text.split()
        if not words:
            return []
        step = max(1, chunk_size - 50)  # 50-word overlap
        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks


class FileParser:
    """
    Agentic file parser that auto-detects format and dispatches
    to the appropriate extraction strategy.
    """

    def parse(self, filepath: str) -> ParsedDocument:
        path = Path(filepath)
        if not path.exists():
            return self._error_doc(filepath, f"File not found: {filepath}")

        file_size = path.stat().st_size
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._parse_pdf(filepath, file_size)
        elif suffix in {".txt", ".md", ".rst"}:
            return self._parse_text(filepath, file_size)
        else:
            # Try as text regardless
            return self._parse_text(filepath, file_size)

    def _parse_pdf(self, filepath: str, file_size: int) -> ParsedDocument:
        """Extract text from PDF using pypdf."""
        try:
            import pypdf
        except ImportError:
            return self._error_doc(filepath, "pypdf not installed")

        pages_text = []
        parse_errors = []

        try:
            reader = pypdf.PdfReader(filepath)
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    # Clean up hyphenation artifacts
                    text = re.sub(r"-\n", "", text)
                    # Normalise whitespace
                    text = re.sub(r"\n{3,}", "\n\n", text)
                    pages_text.append(text)
                except Exception as e:
                    parse_errors.append(f"Page {i}: {str(e)[:60]}")
                    pages_text.append("")

            full_text = "\n\n".join(pages_text)

            # If PDF is image-only (no extractable text), try corpus injection
            if len(full_text.strip()) < 100:
                try:
                    from ref_engine.corpus_injector import get_corpus_text
                    injected = get_corpus_text(filepath)
                    if injected.strip():
                        full_text = injected
                        pages_text = [injected]
                        parse_errors.append("IMAGE_PDF: text injected from corpus")
                except ImportError:
                    pass

            metadata = {"page_count": len(reader.pages)}

            # Try to get PDF metadata
            info = reader.metadata
            if info:
                metadata["title"]  = getattr(info, "title",  "") or ""
                metadata["author"] = getattr(info, "author", "") or ""

        except Exception as e:
            return self._error_doc(filepath,
                                   f"PDF parse failed: {str(e)[:100]}")

        sections = self._extract_sections(full_text, pages_text)

        return ParsedDocument(
            filepath=filepath,
            filename=Path(filepath).name,
            file_size_bytes=file_size,
            full_text=full_text,
            pages=pages_text,
            sections=sections,
            metadata=metadata,
            parse_errors=parse_errors,
        )

    def _parse_text(self, filepath: str, file_size: int) -> ParsedDocument:
        """Parse plain text file."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                full_text = f.read()
        except Exception as e:
            return self._error_doc(filepath, str(e))

        pages_text = [full_text]  # single-page for text files
        sections = self._extract_sections(full_text, pages_text)

        return ParsedDocument(
            filepath=filepath,
            filename=Path(filepath).name,
            file_size_bytes=file_size,
            full_text=full_text,
            pages=pages_text,
            sections=sections,
            metadata={},
        )

    def _extract_sections(self, full_text: str,
                          pages: List[str]) -> List[DocumentSection]:
        """
        Heuristically detect sections from heading patterns.
        Works for PDFs where headings are rendered as large text runs,
        markdown-style headings, or ALL-CAPS headings.
        """
        sections = []
        # Patterns: markdown headers, ALL-CAPS lines, title-cased short lines
        heading_patterns = [
            re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE),
            re.compile(r"^([A-Z][A-Z\s\-:]{4,60})$", re.MULTILINE),
            re.compile(r"^([A-Z][a-z].{5,60})\n", re.MULTILINE),
        ]

        # Find all potential headings with positions
        heading_spans = []
        for pattern in heading_patterns:
            for m in pattern.finditer(full_text):
                heading_spans.append((m.start(), m.group(1).strip()))

        # Sort by position and deduplicate nearby headings
        heading_spans.sort(key=lambda x: x[0])
        deduped = []
        last_pos = -200
        for pos, heading in heading_spans:
            if pos - last_pos > 150 and len(heading.split()) <= 15:
                deduped.append((pos, heading))
                last_pos = pos

        # Build sections from spans
        for idx, (start, heading) in enumerate(deduped):
            end = deduped[idx + 1][0] if idx + 1 < len(deduped) else len(full_text)
            body = full_text[start + len(heading):end].strip()
            if len(body) < 20:
                continue

            # Approximate page index
            chars_before = start
            cumulative = 0
            page_idx = 0
            for i, page in enumerate(pages):
                cumulative += len(page)
                if chars_before < cumulative:
                    page_idx = i
                    break

            sections.append(DocumentSection(
                heading=heading,
                body=body[:2000],  # cap body length per section
                page_index=page_idx,
                char_offset=start,
            ))

        return sections

    def _error_doc(self, filepath: str, error: str) -> ParsedDocument:
        return ParsedDocument(
            filepath=filepath,
            filename=Path(filepath).name,
            file_size_bytes=0,
            full_text="",
            parse_errors=[error],
        )
