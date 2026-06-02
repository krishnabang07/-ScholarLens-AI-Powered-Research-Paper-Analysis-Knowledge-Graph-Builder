"""
pipeline/embedder.py — dense embedding index using sentence-transformers + FAISS.

Supports:
  - Adding paper embeddings (abstract + title)
  - Semantic search (top-k nearest neighbors)
  - Persistence (save/load index to disk)
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

INDEX_PATH = Path("data/faiss.index")
META_PATH  = Path("data/faiss_meta.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast & good


class EmbeddingIndex:
    """
    Wraps a FAISS flat inner-product index.
    Metadata (paper_id, title, snippet) stored in a parallel JSON list.
    """

    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: list[dict] = []
        self.dim = 384

    async def initialize(self):
        """Load model and existing index (if any)."""
        logger.info("Loading sentence-transformer model: %s", MODEL_NAME)
        self.model = SentenceTransformer(MODEL_NAME)

        if INDEX_PATH.exists() and META_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            self.metadata = json.loads(META_PATH.read_text())
            logger.info("Loaded FAISS index: %d vectors", self.index.ntotal)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            logger.info("Created new FAISS index (dim=%d)", self.dim)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts and L2-normalize for cosine similarity via inner product."""
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(vecs)
        return vecs.astype("float32")

    def add(self, paper_id: str, title: str, abstract: str, year: int, authors: list[str]):
        """Add a single paper to the index."""
        text = f"{title}. {abstract}"
        vec = self._embed([text])
        self.index.add(vec)
        self.metadata.append({
            "paper_id": paper_id,
            "title": title,
            "snippet": abstract[:300],
            "year": year,
            "authors": authors,
        })
        self._save()
        logger.debug("Added paper to index: '%s' (total: %d)", title[:40], self.index.ntotal)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return top-k semantically similar papers."""
        if self.index.ntotal == 0:
            return []
        
        q_vec = self._embed([query])
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        
        return results

    def _save(self):
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        META_PATH.write_text(json.dumps(self.metadata, indent=2))
