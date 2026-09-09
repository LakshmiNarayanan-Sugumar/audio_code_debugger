from src.transcribe import transcribe_audio

samples = ["sample_1_shape_bug", "sample_2_serving_hang", "sample_3_scaler_leakage"]

for sample in samples:
    base = f"debugger_test_data/{sample}"
    audio_path = f"{base}/bug_report_audio.mp3"
    ground_truth_path = f"{base}/bug_report_transcript.txt"

    predicted = transcribe_audio(audio_path)
    with open(ground_truth_path) as f:
        ground_truth = f.read().strip()

    print(f"--- {sample} ---")
    print(f"Ground truth: {ground_truth}")
    print(f"Whisper output: {predicted}")
    print()