"""The OFFLINE program: PDF textbook -> vision transcription -> vector DB.

Run it like:
    python ingest.py "data/textbooks/math_ch3.pdf" --subject math

Pipeline:
  1. render each PDF page to an image (PyMuPDF)
  2. show the image to the VISION model -> clean Markdown (formulas
     become LaTeX, Arabic survives -- classic OCR would shred both)
  3. save the transcript to data/transcriptions/{subject}/ (cache: a
     re-run reuses it instead of paying credits again)
  4. chunk the text PAGE BY PAGE so every chunk knows its page number
  5. embed + upsert into the subject's Chroma collection

The agent graph (program 2) is untouched by all of this: it will simply
find the chunks waiting in Chroma.
"""
import argparse
import base64
import random
import re
import time
from pathlib import Path

import pymupdf  # PDF rendering (package "pymupdf", formerly imported as `fitz`)
from chromadb.api.types import Metadatas
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    TRANSCRIPT_DIR, VISION_MODEL, NVIDIA_API_KEY,
    CHUNK_SIZE, CHUNK_OVERLAP, INGEST_DPI, INGEST_PAGE_SLEEP,
)
from agent.rag import get_collection, embed_passages

PAGE_SECTION_RE = re.compile(
    r"^## Page (?P<page>\d+)\s*$\n?(?P<body>.*?)(?=^## Page \d+\s*$|\Z)",
    re.MULTILINE | re.DOTALL,
)

TRANSCRIBE_PROMPT = """Transcris cette page de manuel scolaire en Markdown.

Règles :
- Garde la structure : titres en ##/###, listes, tableaux en Markdown.
- TOUTES les formules mathématiques en LaTeX entre $...$ ou $$...$$.
- Pour un angle (ex. angle en B formé par les points A, B, C), utilise UN
  SEUL \\widehat englobant les trois lettres : $\\widehat{ABC}$.
  N'IMBRIQUE JAMAIS un \\widehat à l'intérieur d'un autre : ne jamais
  écrire $\\widehat{A\\widehat{B}C}$, c'est incorrect.
- Le texte arabe est recopié tel quel.
- Ignore les décorations, numéros de page, en-têtes/pieds de page.
- Pour chaque image/figure, écris simplement [figure].
- Réponds UNIQUEMENT avec la transcription Markdown, rien d'autre."""

# Safety net: even with the instruction above, vision models occasionally
# still emit a nested \widehat (e.g. \widehat{A\widehat{B}C} instead of
# \widehat{ABC}). This collapses any such nesting, looping until stable so
# it also catches double-nesting. Applied to every page right after
# transcription, so both the cached .md file and what lands in Chroma stay
# clean.
NESTED_WIDEHAT_RE = re.compile(r"\\widehat\{([A-Za-zÀ-ÿ]*)\\widehat\{([A-Za-zÀ-ÿ]*)\}([A-Za-zÀ-ÿ]*)\}")


def fix_nested_widehat(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = NESTED_WIDEHAT_RE.sub(r"\\widehat{\1\2\3}", text)
    return text


def transcribe_page(llm: ChatNVIDIA, png_bytes: bytes, max_retries: int = 5) -> str:
    """Show one page IMAGE to the vision model, get Markdown back.

    Images travel in chat APIs as base64 "data URIs" embedded in the
    message content, next to the text instructions.

    NVIDIA's hosted endpoint occasionally fails mid-run with a transient
    504 (Gateway Timeout) or 500 ("Inference connection error") that has
    nothing to do with this specific page -- it's the shared inference
    infra hiccuping. We retry with exponential backoff instead of letting
    one bad call kill an hour of progress.
    """
    b64 = base64.b64encode(png_bytes).decode("ascii")
    message = HumanMessage(content=[
        {"type": "text", "text": TRANSCRIBE_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ])
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            content = llm.invoke([message]).content
            if not isinstance(content, str):
                raise TypeError("The vision model returned structured content instead of Markdown text.")
            return content
        except Exception as e:  # noqa: BLE001 -- langchain-nvidia raises plain
                                 # Exception for both 500s and 504s, so we
                                 # can't narrow the type; we narrow by message.
            last_err = e
            msg = str(e)
            transient = any(s in msg for s in (
                "500", "502", "503", "504", "timeout", "Timeout",
                "connection error", "Internal Server Error",
            ))
            if not transient or attempt == max_retries:
                raise
            wait = min(90, 3 * 2 ** attempt) + random.uniform(0, 2)
            print(f"    ⚠️ erreur transitoire NVIDIA (tentative {attempt}/{max_retries}): "
                  f"{msg[:150]} — retry dans {wait:.0f}s")
            time.sleep(wait)
    # unreachable, but keeps type checkers happy
    raise last_err  # type: ignore[misc]


def chunk_and_store(transcript: str, subject: str, source_name: str,
                    skip_pages: int = 0, page_offset: int = 0) -> int:
    """Chunk PAGE BY PAGE, embed, upsert into Chroma. Returns chunk count."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )
    texts: list[str] = []
    metas: Metadatas = []
    ids: list[str] = []
    # The transcript is built as "## Page N" sections. Match headers at the
    # start of a line so the first page is included too.
    for match in PAGE_SECTION_RE.finditer(transcript):
        page_number = int(match.group("page"))
        if page_number <= skip_pages:
            continue  # skip again at chunk time: works even with a cached
                      # transcript that contains those pages
        body = match.group("body").strip()
        for j, chunk in enumerate(splitter.split_text(body)):
            texts.append(chunk)
            # Citations must match the BOOK's printed page numbers, not the
            # PDF's (which includes cover/TOC pages before page 1).
            metas.append({"page": page_number - page_offset, "source": source_name})
            ids.append(f"{source_name}-p{page_number}-c{j}")

    if not texts:
        return 0

    get_collection(subject).upsert(
        ids=ids,                                   # deterministic IDs
        documents=texts,
        metadatas=metas,
        embeddings=embed_passages(texts),          # computed locally, free
    )
    return len(texts)


def _load_partial_sections(transcript_path: Path) -> dict[int, str]:
    """Read whatever transcript already exists on disk (possibly a partial
    file left by an earlier crashed run) and return {page_number: section}.
    Returns {} if nothing is there yet."""
    if not transcript_path.exists():
        return {}
    text = transcript_path.read_text(encoding="utf-8")
    sections = {}
    for match in PAGE_SECTION_RE.finditer(text):
        page_number = int(match.group("page"))
        body = fix_nested_widehat(match.group("body").strip())
        sections[page_number] = f"## Page {page_number}\n\n{body}"
    return sections


def ingest_pdf(pdf_path: Path, subject: str, skip_pages: int = 0,
               page_offset: int = 0) -> None:
    transcript_path = TRANSCRIPT_DIR / subject / f"{pdf_path.stem}.md"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Steps 1-3: transcription (the expensive part, so it's cached) --
    # We cache PER PAGE, not just per whole-PDF: NVIDIA's hosted endpoint
    # occasionally errors out mid-run (500/504), and re-transcribing 70
    # already-done pages just to retry page 71 would waste time and
    # credits. Every page is written to disk the moment it's done, and a
    # re-run skips any page already present in the file.
    sections = _load_partial_sections(transcript_path)
    with pymupdf.open(pdf_path) as doc:
        total_pages = len(doc)
        already_done = sum(1 for p in sections if p > skip_pages)
        if already_done:
            print(f"[cache] {already_done} page(s) déjà transcrites dans "
                  f"{transcript_path}, reprise à partir de là.")
        pending = [i for i in range(skip_pages, total_pages) if (i + 1) not in sections]

        if pending:
            llm = ChatNVIDIA(model=VISION_MODEL, api_key=NVIDIA_API_KEY, timeout=3000)
            print(f"[1/3] transcription de {len(pending)} page(s) restante(s) "
                  f"sur {skip_pages + 1}..{total_pages} avec {VISION_MODEL}...")
            for i in pending:
                page = doc.load_page(i)
                png = page.get_pixmap(dpi=INGEST_DPI).tobytes("png")
                markdown = fix_nested_widehat(transcribe_page(llm, png))
                sections[i + 1] = f"## Page {i + 1}\n\n{markdown}"
                print(f"    page {i + 1}/{total_pages} ok")
                # Save progress after EVERY page so a crash never costs
                # more than the one page currently in flight.
                ordered = "\n\n".join(sections[p] for p in sorted(sections))
                transcript_path.write_text(ordered, encoding="utf-8")
                time.sleep(INGEST_PAGE_SLEEP)  # be kind to the rate limiter
            print(f"[2/3] transcription complète sauvegardée -> {transcript_path}")
        else:
            print(f"[cache] réutilisation complète de {transcript_path}")
            # Rewrite the file too, in case cached pages still have the
            # unfixed nested-\widehat bug from before this fix existed.
            ordered = "\n\n".join(sections[p] for p in sorted(sections))
            transcript_path.write_text(ordered, encoding="utf-8")

    transcript = "\n\n".join(sections[p] for p in sorted(sections))

    # --- Steps 4-5: chunk + store (local, free, instant) ----------------
    print("[3/3] découpage + embeddings + insertion dans Chroma...")
    n_chunks = chunk_and_store(transcript, subject, pdf_path.stem,
                               skip_pages, page_offset)
    print(f"✅ {n_chunks} chunks ajoutés à la collection '{subject}'. "
          f"Total: {get_collection(subject).count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a textbook PDF into RAG")
    parser.add_argument("pdf", help="path to the PDF, e.g. data/textbooks/math_ch3.pdf")
    parser.add_argument("--subject", required=True, help="math, physique, svt...")
    parser.add_argument("--skip-pages", type=int, default=0,
                        help="skip the N first PDF pages (cover, TOC, etc.)")
    parser.add_argument("--page-offset", type=int, default=0,
                        help="shift so stored page = printed book page "
                             "(page_offset = PDF page of book page 1, minus 1)")
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.is_file():
        raise SystemExit(f"Fichier introuvable: {pdf}")
    ingest_pdf(pdf, args.subject, skip_pages=args.skip_pages,
               page_offset=args.page_offset)