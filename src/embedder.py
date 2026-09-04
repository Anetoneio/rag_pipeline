"""
This is the embedding step , we need floating point precision vectors of fixed length. 
 For relevant data the vector would be geometrically close what is meant by search by meaning 
 Model choice :  sentence-transformers/all-MiniLM-L6-v2
 this model is free open source and has no rate limiting easy on memory which is the sort we need for this project
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit vectors -> dot product = cosine sim
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
