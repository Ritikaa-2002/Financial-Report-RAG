"""Turn PDF filings into searchable chunks that remember their page number."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf


@dataclass
class Chunk:
    chunk_id: int
    doc: str
    page: int  # 0-indexed, matching the PDF page order
    text: str


def extract_pages(pdf_path):
    """Return the plain text of every page in the PDF."""
    with pymupdf.open(pdf_path) as pdf:
        return [_clean(page.get_text()) for page in pdf]


def _clean(text):
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def chunk_pages(doc, pages, words_per_chunk=200, overlap=50):
    """Split each page into overlapping word windows.

    Chunks never cross a page boundary, so every chunk can be cited to an exact
    page. words_per_chunk=None keeps one chunk per page.
    """
    chunks = []
    for page_num, text in enumerate(pages):
        words = text.split()
        if not words:
            continue
        if words_per_chunk is None or len(words) <= words_per_chunk:
            windows = [words]
        else:
            step = words_per_chunk - overlap
            windows = [words[i : i + words_per_chunk] for i in range(0, len(words) - overlap, step)]
        for w in windows:
            chunks.append(Chunk(len(chunks), doc, page_num, " ".join(w)))
    return chunks


def build_corpus(pdf_dir, words_per_chunk=200, overlap=50, docs=None, cache_dir=None):
    """Chunk every PDF in pdf_dir (or only the named docs). Page text is cached as JSON."""
    pdf_dir = Path(pdf_dir)
    cache_dir = Path(cache_dir or pdf_dir.parent / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(docs) if docs else sorted(p.stem for p in pdf_dir.glob("*.pdf"))
    corpus = []
    for name in names:
        cache = cache_dir / f"{name}.pages.json"
        if cache.exists():
            pages = json.loads(cache.read_text())
        else:
            pages = extract_pages(pdf_dir / f"{name}.pdf")
            cache.write_text(json.dumps(pages))
        for c in chunk_pages(name, pages, words_per_chunk, overlap):
            c.chunk_id = len(corpus)
            corpus.append(c)
    return corpus


def save_chunks(chunks, path):
    Path(path).write_text("\n".join(json.dumps(asdict(c)) for c in chunks))
