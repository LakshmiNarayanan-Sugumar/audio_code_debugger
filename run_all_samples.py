from src.pipeline import run_pipeline


samples = {
    "sample_1_shape_bug": {
        "audio": "debugger_test_data/sample_1_shape_bug/bug_report_audio.mp3",
        "code": "debugger_test_data/sample_1_shape_bug/buggy_code.py",
        "exclude_file": "model.py",
    },

    "sample_2_serving_hang": {
        "audio": "debugger_test_data/sample_2_serving_hang/bug_report_audio.mp3",
        "code": "debugger_test_data/sample_2_serving_hang/buggy_code.py",
        # predict_api.py has real function boundaries, so we exclude only
        # the `predict` function chunk (the literal answer) instead of the
        # whole file. This keeps the module-level setup chunk (df_cache
        # caching, tf.config.threading calls) available as legitimate
        # retrieval context, without leaking the reconstructed function body.
        # k=1: buggy_code.py already contains the full PredictResponse class,
        # both decorators, and both endpoint functions in the prompt, so no
        # other chunk adds information — it only adds contamination risk.
        "exclude_names": ["predict"],
        "k": 1,
    },

    "sample_3_scaler_leakage": {
        "audio": "debugger_test_data/sample_3_scaler_leakage/bug_report_audio.mp3",
        "code": "debugger_test_data/sample_3_scaler_leakage/buggy_code.py",
        "exclude_file": "data_preprocessing.py",
    },
}


if __name__ == "__main__":

    for sample_id, sample in samples.items():

        print(f"\n{'=' * 20} {sample_id} {'=' * 20}")

        result = run_pipeline(
            sample["audio"],
            sample["code"],
            k=sample.get("k", 4),
            exclude_file=sample.get("exclude_file"),
            exclude_names=sample.get("exclude_names")
        )

        print("Transcript:", result["transcript"])

        print("Retrieved from:")
        for file, name in zip(result["retrieved_files"], result["retrieved_names"]):
            print(f"  - {file} :: {name}")

        print("\nOutput:\n")
        print(result["output"])