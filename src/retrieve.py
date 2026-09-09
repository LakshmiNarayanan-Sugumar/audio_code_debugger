from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

def load_index(index_path="corpus_index.faiss", meta_path="corpus_meta.pkl"):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve(query_text, index, chunks, k=4):
    model = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({**chunks[idx], "distance": float(dist)})
    return results