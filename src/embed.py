from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

MODEL_NAME = "all-MiniLM-L6-v2"

def build_index(chunks, index_path="corpus_index.faiss", meta_path="corpus_meta.pkl"):
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["code"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks