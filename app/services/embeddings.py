import json
import os
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.utils.helpers import setup_logger

log = setup_logger(__name__)


class EmbeddingEngine:
    def __init__(self):
        self.model = None
        self.index = None
        self.metadata: List[dict] = []

    def _load_model(self):
        if self.model is None:
            log.info(f"Loading SentenceTransformer model: {settings.embedding_model}")
            self.model = SentenceTransformer(settings.embedding_model)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        self._load_model()
        embeddings = self.model.encode(texts)
        return np.array(embeddings, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        if not text:
            text = " "
        return self.encode([text])[0]

    def build_index(self, texts: List[str], metadata: Optional[List[dict]] = None):
        if not texts:
            log.warning("No texts provided to build index.")
            return

        embeddings = self.encode(texts)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.metadata = metadata if metadata else [{} for _ in texts]
        log.info(f"Built FAISS index with {self.index.ntotal} vectors.")

    def save_index(self, path: Optional[str] = None):
        if path is None:
            path = settings.faiss_index_path

        if self.index is None:
            log.warning("No index to save.")
            return

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        faiss.write_index(self.index, path)

        meta_path = path + "_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        log.info(f"Saved FAISS index with {self.index.ntotal} vectors to {path}")

    def load_index(self, path: Optional[str] = None):
        if path is None:
            path = settings.faiss_index_path

        if not os.path.exists(path):
            log.info(f"FAISS index file not found at {path}, starting with empty index.")
            self.index = None
            self.metadata = []
            return

        try:
            self.index = faiss.read_index(path)
            meta_path = path + "_meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = [{} for _ in range(self.index.ntotal)]
            log.info(f"Loaded FAISS index with {self.index.ntotal} vectors from {path}")
        except Exception as e:
            log.warning(f"Failed to load index from {path}: {e}")
            self.index = None
            self.metadata = []

    def search(self, query: str, top_k: int = 5) -> List[Tuple[dict, float]]:
        """Search similar knowledge chunks. Safely handles empty query or empty index."""
        if not query or not query.strip():
            return []

        if self.index is None:
            self.load_index()

        if self.index is None or self.index.ntotal == 0:
            return []

        actual_k = min(top_k, self.index.ntotal)
        if actual_k <= 0:
            return []

        query_vector = self.encode_single(query).reshape(1, -1)
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, actual_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(distances[0][i])))

        return results

    def add_to_index(self, texts: List[str], new_metadata: List[dict]):
        """Add vectors to index, filtering out duplicates based on id or content."""
        if not texts:
            return

        if self.index is None:
            self.build_index(texts, new_metadata)
            return

        # Deduplicate against existing metadata
        existing_ids = {m.get("id") for m in self.metadata if "id" in m}
        filtered_texts = []
        filtered_meta = []

        for t, m in zip(texts, new_metadata):
            m_id = m.get("id")
            if m_id is not None and m_id in existing_ids:
                continue
            filtered_texts.append(t)
            filtered_meta.append(m)

        if not filtered_texts:
            log.info("All documents to add are already indexed.")
            return

        embeddings = self.encode(filtered_texts)
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.metadata.extend(filtered_meta)
        log.info(f"Added {len(filtered_texts)} vectors. New index size: {self.index.ntotal}")

    @property
    def index_size(self) -> int:
        return self.index.ntotal if self.index else 0


embedding_engine = EmbeddingEngine()
