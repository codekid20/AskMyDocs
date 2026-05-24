"""Stage 1 of ingestion: PDFs in data/raw/ -> one JSON object per SECTION in
sections.jsonl. Extraction + cleaning lives in extract.py; this orchestrates
and persists. Chunking is the next slice.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from askmydocs.config import settings
from askmydocs.extract import extract_sections


class SectionRecord(BaseModel):
    doc_id: str
    heading: str
    page: int
    text: str


def ingest_all(raw_dir: Path | None = None, out_path: Path | None = None) -> int:
    raw_dir = raw_dir or settings.raw_dir
    out_path = out_path or settings.sections_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in {}", raw_dir)
        return 0

    total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for pdf in pdfs:
            logger.info("Parsing {}", pdf.name)
            sections = extract_sections(pdf)
            for s in sections:
                rec = SectionRecord(
                    doc_id=s.doc_id, heading=s.heading, page=s.page, text=s.text
                )
                f.write(rec.model_dump_json() + "\n")
            logger.info("  -> {} sections", len(sections))
            total += len(sections)

    logger.success("Wrote {} sections to {}", total, out_path)
    return total


if __name__ == "__main__":
    ingest_all()