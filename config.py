"""Central configuration: paths, model names, tunables.

Every other file imports its settings from here. If you later swap the
NVIDIA model or change chunk size, you edit THIS file only.
"""
import os
from pathlib import Path

# python-dotenv reads the .env file and injects its content as environment
# variables. The API key never appears in the code itself.
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# --- Folders -----------------------------------------------------------
BASE_DIR = Path(__file__).parent
TEXTBOOK_DIR = BASE_DIR / "data" / "textbooks"           # drop PDFs here
TRANSCRIPT_DIR = BASE_DIR / "data" / "transcriptions"    # PDF -> markdown output
SYLLABUS_DIR = BASE_DIR / "data" / "syllabus"            # chapter list per subject
CHROMA_DIR = BASE_DIR / "db" / "chroma"                  # the vector store lives here
CHECKPOINT_PATH = BASE_DIR / "db" / "checkpoints.sqlite" # conversation memory

# --- Models (NVIDIA NIM — verify exact IDs on build.nvidia.com) --------
# Main brain: chat, tutor, quiz, planner, router.
CHAT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
# Reads PDF page IMAGES during ingestion (math formulas + Arabic survive).
VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"
# Local embedding model: runs on YOUR pc, no API, free, good at FR + AR.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# --- API key -----------------------------------------------------------
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# --- Retrieval tunables (we tune these together when testing RAG) ------
CHUNK_SIZE = 900        # characters per chunk: ~2-3 paragraphs
CHUNK_OVERLAP = 120     # chunks share text so ideas cut at a border stay findable
RETRIEVE_K = 4          # how many chunks we fetch per question

# --- PDF ingestion -----------------------------------------------------
INGEST_DPI = 100         # image quality when rendering PDF pages
INGEST_PAGE_SLEEP = 2.0  # pause between pages: respects free credits/rate limits

# Create folders on first run so nothing crashes on a fresh machine.
for _d in (TEXTBOOK_DIR, TRANSCRIPT_DIR, SYLLABUS_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)
