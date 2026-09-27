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


def retrieve(query_text, index, chunks, k=4, exclude_file=None, exclude_names=None):
    model = SentenceTransformer(MODEL_NAME)

    query_vec = model.encode(
        [query_text],
        convert_to_numpy=True
    ).astype(np.float32)

    # Search more chunks than needed so that we can
    # skip the excluded file if necessary.
    search_k = min(index.ntotal, max(k * 3, k))

    distances, indices = index.search(query_vec, search_k)

    results = []

    for idx, dist in zip(indices[0], distances[0]):

        # Skip invalid FAISS indices
        if idx < 0:
            continue

        chunk = chunks[idx]

        # Exclude the entire buggy source file when requested
        # (used for files with no function-level structure, where
        # whole-file exclusion is the smallest meaningful unit).
        if exclude_file is not None:
            if chunk.get("file") == exclude_file:
                continue

        # Exclude specific named chunks (e.g. one function) rather than
        # a whole file — used when the file has real function boundaries,
        # so unrelated chunks from the same file (module-level setup,
        # other functions) stay available as legitimate context.
        if exclude_names is not None:
            if chunk.get("name") in exclude_names:
                continue

        results.append({
            **chunk,
            "distance": float(dist)
        })

        if len(results) >= k:
            break

    return results