from src.pipeline import run_pipeline

samples = ["sample_1_shape_bug", "sample_2_serving_hang", "sample_3_scaler_leakage"]

for sample in samples:
    base = f"debugger_test_data/{sample}"
    result = run_pipeline(
        audio_path=f"{base}/bug_report_audio.mp3",
        buggy_code_path=f"{base}/buggy_code.py",
    )
    print(f"\n{'='*20} {sample} {'='*20}")
    print(f"Transcript: {result['transcript']}\n")
    print(f"Retrieved from: {result['retrieved_files']}\n")
    print(result["output"])
    print()