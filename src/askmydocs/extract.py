# src/askmydocs/extract.py
"""
PDF -> clean, section-tagged text sections.

This module has ONE job: extraction. It knows about PyMuPDF and PDF quirks,
and nothing else. Downstream code (chunking, embedding) never touches a PDF
directly — it only ever sees the normalized `Section` objects this produces.
That seam is deliberate: swap the extractor later (OCR, a different library)
and nothing downstream changes.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf4llm


@dataclass(frozen=True)
class Section:
    """A logical section of a document: its heading, its text, and where it
    lives. `heading` becomes the citation anchor (e.g. '2.2 Retriever: DPR').
    `page` is best-effort — the page the section's text begins on."""
    doc_id: str          # stable id for the source doc, e.g. "RAG"
    heading: str         # section title, or "" if text precedes any heading
    text: str            # cleaned body text of this section
    page: int            # 1-based page number where this section starts


# --- cleaning helpers -------------------------------------------------------
# Each fixes a specific artifact we SAW in the pymupdf4llm probe output.

_PICTURE_MARKER = re.compile(r"\*\*==>.*?<==\*\*")          # "**==> picture ... <==**"
_HEADING_LINE = re.compile(r"^#{1,6}\s+(.*)$")             # markdown headings "## **2.1 Models**"
_MD_EMPHASIS = re.compile(r"[*_]{1,3}")                     # stray markdown * and _ around math
_DEHYPHENATE = re.compile(r"(\w)-\n(\w)")                   # "knowl-\nedge" -> "knowledge"
_MULTISPACE = re.compile(r"[ \t]{2,}")


def _normalize(text: str) -> str:
    """Undo the PDF extraction artifacts we identified in the probe."""
    # 1. Unicode normalize: ligatures like "ﬁ" -> "fi", "ﬂ" -> "fl"
    text = unicodedata.normalize("NFKC", text)
    # 2. Join words broken across a line by a hyphen
    text = _DEHYPHENATE.sub(r"\1\2", text)
    # 3. Drop the "picture intentionally omitted" markers
    text = _PICTURE_MARKER.sub(" ", text)
    # 4. Strip stray markdown emphasis chars left around italic math vars
    text = _MD_EMPHASIS.sub("", text)
    # 5. Collapse runs of spaces, tidy whitespace
    text = _MULTISPACE.sub(" ", text)
    return text.strip()


def _clean_heading(raw: str) -> str:
    """A heading line still has markdown noise: '## **2.1 Models**' -> '2.1 Models'."""
    return _MD_EMPHASIS.sub("", raw).strip()


# --- the public function ----------------------------------------------------

def extract_sections(pdf_path: Path) -> list[Section]:
    """
    Extract a PDF into cleaned, heading-delimited Sections.

    Strategy: pymupdf4llm gives us page-by-page Markdown with correct reading
    order and `#`-style headings (we confirmed this on RAG.pdf). We walk that
    Markdown line by line, starting a new Section every time we hit a heading,
    and tag each Section with the page it started on.
    """
    doc_id = pdf_path.stem  # "RAG.pdf" -> "RAG"

    # page_chunks=True returns one dict per page, each with its markdown text.
    # This is how we recover page numbers for citations.
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    sections: list[Section] = []
    cur_heading = ""
    cur_lines: list[str] = []
    cur_page = 1

    def flush(page: int):
        nonlocal cur_lines, cur_heading
        body = _normalize("\n".join(cur_lines))
        if body:  # skip empty sections (e.g. a heading with no body yet)
            sections.append(Section(doc_id=doc_id, heading=cur_heading,
                                    text=body, page=page))
        cur_lines = []

    for page_no, page in enumerate(pages, start=1):
        md = page.get("text", "")
        for line in md.splitlines():
            m = _HEADING_LINE.match(line.strip())
            if m:
                # New heading => close out the previous section, start fresh.
                flush(cur_page)
                cur_heading = _clean_heading(m.group(1))
                cur_page = page_no
            else:
                if not cur_lines:           # first body line of a new section
                    cur_page = page_no       # remember where this section starts
                cur_lines.append(line)
    flush(cur_page)  # don't forget the final section

    return sections