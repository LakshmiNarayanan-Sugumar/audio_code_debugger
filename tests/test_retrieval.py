from src.retrieve import load_index, retrieve

index, chunks = load_index()

test_queries = {
    "sample_1_shape_bug": "Tata Motors BiLSTM model shape mismatch error on input layer during prediction",
    "sample_2_serving_hang": "FastAPI predict endpoint hangs and times out, works fine in notebook",
    "sample_3_scaler_leakage": "BiLSTM model test metrics look too good, suspicious scaling or data splitting issue",
}

for sample, query in test_queries.items():
    print(f"--- {sample} ---")
    print(f"Query: {query}")
    results = retrieve(query, index, chunks, k=4)
    for r in results:
        print(f"  [{r['distance']:.3f}] {r['file']} :: {r['name']}")
    print()