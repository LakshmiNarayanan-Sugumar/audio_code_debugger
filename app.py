import gradio as gr
from src.transcribe import transcribe_audio
from src.retrieve import load_index, retrieve
from src.prompt import build_prompt
from src.generate import generate_fix

def debug_from_audio(audio_path, code_file):
    if audio_path is None or code_file is None:
        return "Please provide both an audio bug report and a code file.", "", ""

    transcript = transcribe_audio(audio_path)

    with open(code_file.name) as f:
        buggy_code = f.read()

    index, chunks = load_index()
    query = f"{transcript}\n\n{buggy_code}"
    context_chunks = retrieve(query, index, chunks, k=4)
    prompt = build_prompt(context_chunks, transcript, buggy_code)
    result = generate_fix(prompt)

    retrieved_files = ", ".join(sorted(set(c["file"] for c in context_chunks)))

    return transcript, retrieved_files, result


demo = gr.Interface(
    fn=debug_from_audio,
    inputs=[
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Describe your bug (audio)"),
        gr.File(label="Upload your buggy code file (.py)"),
    ],
    outputs=[
        gr.Textbox(label="Transcript"),
        gr.Textbox(label="Retrieved Context From"),
        gr.Markdown(label="Diagnosis & Fix"),
    ],
    title="Fix My Code — Audio Debugger",
    description="Describe a bug out loud and upload the buggy file. The system transcribes your report, retrieves relevant context from the codebase, and generates a diagnosis + fix.",
)

if __name__ == "__main__":
    demo.launch()