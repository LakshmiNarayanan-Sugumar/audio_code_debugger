from src.retrieve import load_index, retrieve
from src.chunk import chunk_corpus

index, chunks = load_index()

query = "BiLSTM model test metrics look too good, suspicious scaling or data splitting issue"
results = retrieve(query, index, chunks, k=14)  # k=14 = everything, to see full ranking

print("Full ranking:")
for r in results:
    print(f"  [{r['distance']:.3f}] {r['file']} :: {r['name']}")

print("\n--- Contents of data_preprocessing.py chunks ---")
raw_chunks = chunk_corpus("corpus")
for c in raw_chunks:
    if c["file"] == "data_preprocessing.py":
        print(f"\n[{c['name']}]")
        print(c["code"])