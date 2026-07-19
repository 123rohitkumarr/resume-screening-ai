import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.utils.helpers import setup_logger

log = setup_logger(__name__)

class EmbeddingEngine:
    def __init__(self):
        self.model = None
        self.index = None
        self.metadata = []

    def _load_model(self):
        if self.model is None:
            log.info(f"Loading SentenceTransformer model: {settings.embedding_model}")
            self.model = SentenceTransformer(settings.embedding_model)

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        embeddings = self.model.encode(texts)
        return np.array(embeddings, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def build_index(self, texts: list[str], metadata: list[dict] = None):
        embeddings = self.encode(texts)
        faiss.normalize_L2(embeddings)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
        self.metadata = metadata if metadata else [{} for _ in texts]
        log.info(f"Built FAISS index with {self.index.ntotal} vectors.")

    def save_index(self, path: str = None):
        if path is None:
            path = settings.faiss_index_path
        
        if self.index is None:
            log.warning("No index to save.")
            return

        faiss.write_index(self.index, path)
        
        meta_path = path + '_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f)
        log.info(f"Saved FAISS index to {path}")

    def load_index(self, path: str = None):
        if path is None:
            path = settings.faiss_index_path
            
        try:
            self.index = faiss.read_index(path)
            meta_path = path + '_meta.json'
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
            log.info(f"Loaded FAISS index with {self.index.ntotal} vectors from {path}")
        except Exception as e:
            log.warning(f"Failed to load index from {path}: {e}")
            self.index = None
            self.metadata = []

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        if self.index is None:
            return []
            
        query_vector = self.encode_single(query).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(distances[0][i])))
                
        return results

    def add_to_index(self, texts: list[str], new_metadata: list[dict]):
        if self.index is None:
            self.build_index(texts, new_metadata)
            return
            
        embeddings = self.encode(texts)
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.metadata.extend(new_metadata)
        log.info(f"Added {len(texts)} vectors. New index size: {self.index.ntotal}")

    @property
    def index_size(self) -> int:
        return self.index.ntotal if self.index else 0

embedding_engine = EmbeddingEngine()
