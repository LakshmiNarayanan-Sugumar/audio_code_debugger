from src.chunk import chunk_corpus

chunks = chunk_corpus("corpus")

print(f"Total chunks: {len(chunks)}\n")
for c in chunks:
    print(f"[{c['file']}] {c['type']}: {c['name']} ({len(c['code'].splitlines())} lines)")