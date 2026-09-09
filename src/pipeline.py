from src.transcribe import transcribe_audio
from src.retrieve import load_index, retrieve
from src.prompt import build_prompt
from src.generate import generate_fix

def run_pipeline(audio_path, buggy_code_path, k=4):
    transcript = transcribe_audio(audio_path)

    with open(buggy_code_path) as f:
        buggy_code = f.read()

    index, chunks = load_index()
    query = f"{transcript}\n\n{buggy_code}"
    context_chunks = retrieve(query, index, chunks, k=k)
    prompt = build_prompt(context_chunks, transcript, buggy_code)
    result = generate_fix(prompt)

    return {
        "transcript": transcript,
        "retrieved_files": [c["file"] for c in context_chunks],
        "output": result,
    }