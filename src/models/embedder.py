import numpy as np
import ollama
import time
import math

"""
BM25 + embeddings + reranker
"""

"""
Uses Ollama for embeddings — no PyTorch or sentence-transformers encoding.
-HTTP call to ollama per embed request as a process
-calls once per text in a loop. 
https://docs.ollama.com/capabilities/embeddings#python-2

On sentence transformers
-requires pytorch (2gb+ install)
-S.T batch all texts in one pass (faster for larger batches)
-Uses more RAM
https://medium.com/@farissyariati/ask-your-codebase-anything-using-ollama-embeddings-and-rag-c65081a5ef20
"""


class Embedder:
    def __init__(self, config):
        self.emb_model = config.embedding_model
        self.keep_alive = config.embedding_keep_alive  # e.g. "10m", "30m", "0" to unload immediately
        if config.embedding_dim == 0:
            probe = ollama.embed(model=self.emb_model, input="probe", keep_alive=self.keep_alive)
            self.dim = len(probe["embeddings"][0])
            print(f"[embedder] dimension=auto detected {self.dim} from {self.emb_model}", flush=True)
        else:
            self.dim = config.embedding_dim

    def embed(self, text: str) -> np.ndarray:
        #Embed a single string. Returns normalized float32 numpy array.
        response = ollama.embed(model=self.emb_model, input=text, keep_alive=self.keep_alive)
        #model, embeddings, total_duration, load_duration, propmt_eval_count
        vec = np.array(response["embeddings"][0], dtype=np.float32)
        norm = np.linalg.norm(vec) #L2-normalizes embedding vector
        if norm > 0: 
            vec = vec / norm
        return vec
    #2D matrix (N,384), 1 row per input
    def embed_batch(self, texts: list) -> np.ndarray:
        #Single HTTP call to Ollama for all texts
        response = ollama.embed(model=self.emb_model, input=texts, keep_alive=self.keep_alive)
        vecs = np.array(response["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms