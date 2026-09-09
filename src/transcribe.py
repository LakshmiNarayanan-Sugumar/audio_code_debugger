from faster_whisper import WhisperModel

VOCAB_HINT = "BiLSTM, FastAPI, yfinance, scaler, StandardScaler, MinMaxScaler, endpoint, input_shape, Tata Motors"

def transcribe_audio(audio_path: str, model_size: str = "small") -> str:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, initial_prompt=VOCAB_HINT)
    transcript = " ".join(segment.text.strip() for segment in segments)
    return transcript