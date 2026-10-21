from dataclasses import dataclass
import fitz

@dataclass
class Passage:
    text: str
    page_start: int
    page_end: int
    chunk_id: str

def extract_pdf_pages(path: str) -> list[dict]:
    """Extract PDF text page-by-page to preserve original PDF page numbers."""
    pages = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": index + 1, "text": text})
    if not pages:
        raise ValueError("No extractable text found. If the PDF is scanned, OCR it first.")
    return pages

def chunk_pages(pages: list[dict], chunk_size: int = 1200, overlap: int = 180) -> list[Passage]:
    """Chunk within page boundaries and keep page attribution."""
    chunks = []
    for page in pages:
        text = " ".join(page["text"].split())
        start = 0
        part = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            piece = text[start:end].strip()
            if piece:
                chunks.append(Passage(piece, page["page"], page["page"], f"p{page['page']}_c{part}"))
            if end >= len(text):
                break
            start = max(0, end - overlap)
            part += 1
    return chunks
