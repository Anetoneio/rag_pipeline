"""
this is where we use all of the code at one place 

"""
from typing import List
from .loader import load_document
from .chunker import RecursiveChunker
from .embedder import Embedder
from .vector_store import VectorStore
from .generator import generate_answer


class RAGPipeline:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.chunker = RecursiveChunker(chunk_size, chunk_overlap)
        self.embedder = Embedder(embedding_model)
        self.store = VectorStore(dim=self.embedder.dim)

    def ingest(self, path: str):
        text, metadata = load_document(path)
        chunks = self.chunker.chunk(text, base_metadata=metadata)
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.store.add(embeddings, texts, metadatas)
        print(f"Ingested {path}: {len(texts)} chunks added.")

    def ingest_many(self, paths: List[str]):
        for p in paths:
            self.ingest(p)

    def query(self, question: str, k: int = 4):
        query_emb = self.embedder.embed_one(question)
        retrieved = self.store.search(query_emb, k=k)
        answer = generate_answer(question, retrieved)
        return answer, retrieved

    def save(self, path_prefix: str = "index"):
        self.store.save(path_prefix)

    def load(self, path_prefix: str = "index"):
        self.store = VectorStore.load(path_prefix)

