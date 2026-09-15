"""RAG mechanics: turn text into vectors, store in Chroma, retrieve.

Pipeline we implement here:
  STORE (offline):  textbook chunk -> embed("passage: ...") -> Chroma row
  SEARCH (runtime): student question -> embed("query: ...") -> nearest rows

Two ideas everything rests on:

1. An EMBEDDING maps text to a vector of ~768 numbers. Texts with similar
   meaning get vectors pointing in similar directions, in ANY language --
   that is why a French question can find a French lesson.

2. Chroma is a vector database: it stores (vector, text, metadata) and can
   answer "which stored vectors are closest to this query vector?" using
   cosine similarity. We compute the vectors ourselves (locally, free) and
   hand them to Chroma explicitly.
"""
import chromadb
from chromadb.api.types import Embedding, Embeddings
from config import CHROMA_DIR, EMBEDDING_MODEL

_model = None  # will hold the loaded model; loaded lazily on first use


def _get_model():
    """Load the embedding model once, on first call, then reuse it."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_passages(texts: list[str]) -> Embeddings:
    """Embed texts that will be STORED in the vector DB."""
    vectors = _get_model().encode(
        [f"passage: {t}" for t in texts], normalize_embeddings=True
    )
    # Chroma's native Embedding type is a NumPy float array. Keeping the
    # encoder output in that form also avoids a lossy list conversion.
    return [vector for vector in vectors]


def embed_query(text: str) -> Embedding:
    """Embed a SEARCH query. Different prefix than storage -- see lesson."""
    return _get_model().encode(
        [f"query: {text}"], normalize_embeddings=True
    )[0]


def get_collection(subject: str):
    """One Chroma collection (= one table) per subject: math, physique..."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=subject,
        metadata={"hnsw:space": "cosine"},  # rank by cosine similarity
    )


def collection_count(subject: str) -> int:
    """How many chunks are stored for this subject (0 = nothing ingested)."""
    return get_collection(subject).count()


def retrieve(subject: str, query: str, k: int = 4) -> list[dict]:
    """Return the k chunks most similar to the query, with their metadata.

    Returns [{"text": ..., "page": ..., "source": ...}, ...]
    Returns [] when nothing was ingested for this subject yet.
    """
    col = get_collection(subject)
    if col.count() == 0:
        return []
    res = col.query(
        query_embeddings=[embed_query(query)],
        n_results=min(k, col.count()),
        include=["documents", "metadatas"],
    )
    # Chroma's result type marks these fields as optional even though we
    # requested both explicitly. Guarding them also keeps retrieval safe if a
    # backend returns an incomplete result.
    document_groups = res["documents"]
    metadata_groups = res["metadatas"]
    if not document_groups or not metadata_groups:
        return []

    docs, metas = document_groups[0], metadata_groups[0]
    return [
        {"text": doc, "page": meta.get("page"), "source": meta.get("source")}
        for doc, meta in zip(docs, metas)
    ]


def format_context(docs: list[dict]) -> str:
    """Render retrieved chunks as the CONTEXT block we paste into prompts."""
    if not docs:
        return "(no course material ingested for this subject yet)"
    blocks = []
    for d in docs:
        blocks.append(f"[page {d['page']} | {d['source']}]\n{d['text']}")
    return "\n\n---\n\n".join(blocks)
