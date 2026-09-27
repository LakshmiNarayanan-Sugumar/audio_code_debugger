from src.chunk import chunk_corpus
from src.embed import build_index

chunks = chunk_corpus("corpus")
build_index(chunks)
print(f"Rebuilt index with {len(chunks)} chunks")