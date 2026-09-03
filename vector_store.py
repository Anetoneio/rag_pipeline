"""
Vector store: indexes chunk embeddings and answers "give me the k
vectors closest to this query vector" -- this is the retrieval core
of RAG.

WHY FAISS (specifically IndexFlatIP):
 FAISS is a library, not a server/database -- it runs in-process like
  any other Python import, which keeps this project self-contained and
  easy for anyone to `pip install` and run (no Docker, no external DB
  to stand up). Good fit for a single-document / few-thousand-chunk demo.
 IndexFlatIP does EXACT brute-force search (compares the query against
  every stored vector, no approximation). At our scale (a handful of
  documents -> thousands of chunks) exact search is still fast (a few
  milliseconds) and guarantees we never miss the true nearest neighbor,
  which an approximate index (e.g. IVF, HNSW) could. Those approximate
  indexes only start paying off at millions of vectors.
  "IP" = inner product. Since Embedder L2-normalizes every vector,
  inner product is mathematically identical to cosine similarity here,
  so this gives us cosine-similarity search "for free" without a
  separate normalization step inside FAISS.

file is roughly analogous to a hand-rolled hash map /
nearest-neighbor structure might write in C++: `texts` and
`metadatas` are just parallel arrays, and `index` is the structure that
maps a vector -> the row index into those arrays (like storing an
index into an array instead of the object itself, to keep memory
layout compact).
"""

import pickle
from typing import List, Dict, Tuple
import numpy as np
import faiss


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []

    def add(self, embeddings: np.ndarray, texts: List[str], metadatas: List[Dict]):
        assert len(texts) == len(metadatas) == embeddings.shape[0]
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def search(self, query_embedding: np.ndarray, k: int = 4) -> List[Tuple[str, Dict, float]]:
        query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 if k > number of stored vectors
                continue
            results.append((self.texts[idx], self.metadatas[idx], float(score)))
        return results

    def save(self, path_prefix: str):
        faiss.write_index(self.index, f"{path_prefix}.index")
        with open(f"{path_prefix}.meta.pkl", "wb") as f:
            pickle.dump(
                {"texts": self.texts, "metadatas": self.metadatas, "dim": self.dim}, f
            )

    @classmethod
    def load(cls, path_prefix: str) -> "VectorStore":
        with open(f"{path_prefix}.meta.pkl", "rb") as f:
            data = pickle.load(f)
        store = cls(dim=data["dim"])
        store.index = faiss.read_index(f"{path_prefix}.index")
        store.texts = data["texts"]
        store.metadatas = data["metadatas"]
        return store
