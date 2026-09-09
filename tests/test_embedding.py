from src.chunk import chunk_corpus
from src.embed import build_index

chunks = chunk_corpus("corpus")
index, stored_chunks = build_index(chunks)

print(f"Indexed {index.ntotal} vectors")
print(f"Vector dimension: {index.d}")