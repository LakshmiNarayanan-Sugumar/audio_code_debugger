# Fix My Code — Audio Debugger

A RAG-based debugging assistant that takes a *spoken* bug report, retrieves relevant context from a real codebase, and generates a diagnosis and fix.

---

## How It Works

1. **Transcription** — Whisper (via `faster-whisper`) converts the audio bug report to text, primed with a domain vocabulary hint for technical terms.
2. **Retrieval** — The transcript and buggy code are embedded together and searched against a FAISS index of the target codebase, chunked at the function/class level using Python's `ast` module.
3. **Generation** — Retrieved context, the transcript, and the buggy code are combined into a structured prompt sent to Qwen2.5-Coder-7B-Instruct, which returns a diagnosis and a fix.

    Audio bug report → Whisper → Transcript
                                     ↓
    Buggy code ──────────────→ Retrieval query
                                     ↓
                        FAISS search over indexed codebase
                                     ↓
                      Retrieved context + transcript + buggy code
                                     ↓
                         Qwen2.5-Coder-7B-Instruct
                                     ↓
                           Diagnosis + Fixed code

---

## Test Corpus

The retrieval corpus is built from a real project — a Tata Motors stock price predictor (BiLSTM model, FastAPI serving layer, data preprocessing pipeline). Three real bugs from that project's development history serve as test cases.

| Sample | Bug Type | Root Cause |
|---|---|---|
| `sample_1_shape_bug` | Shape mismatch at inference | `input_shape` mixed `X_train`/`X_test` dimensions |
| `sample_2_serving_hang` | FastAPI endpoint hangs | Blocking I/O (`yf.download()` called synchronously inside the request handler) |
| `sample_3_scaler_leakage` | Inflated test metrics | Scaler fit before train/test split |

Each sample includes an audio bug report, ground-truth transcript, buggy code, and the actual fix — used to validate every pipeline stage against real outcomes, not just "does it run."

**Retrieval design note:** each sample's own source file is excluded from retrieval to prevent the pipeline from just retrieving its own answer verbatim. For `predict_api.py` — the only corpus file with real function-level structure — this exclusion is applied at the chunk level (excluding only the `predict` function, the literal answer) rather than the whole file, so legitimate supporting context (module-level setup: caching pattern, TensorFlow threading config) stays available. `model.py` and `data_preprocessing.py` are monolithic scripts with no function boundaries, so whole-file exclusion is the smallest meaningful unit there.

---

## Results

| Sample | Retrieval | Diagnosis | Fix |
|---|---|---|---|
| Shape bug | N/A (relevant file excluded by design) | Correct | Correct |
| Serving hang | Correct chunk (`predict_api.py :: module_level_1`) | Correct | Correct (varies across runs) |
| Scaler leakage | N/A (relevant file excluded by design) | Correct | Correct |

Samples 1 and 3 have their source file excluded from retrieval to prevent answer leakage, and the remaining corpus has no related code, so retrieval adds no useful context there and the model diagnoses them from general knowledge. Sample 2 is the case where retrieval contributes: its supporting context lives in a file that can be excluded at the chunk level.

Full verified run: [`results_log.txt`](results_log.txt).

**Known limitation — generation variance:** even with `temperature=0.1`, the serving-hang sample's *diagnosis* converges reliably across runs (always correctly identifies the blocking `yf.download()` call), but the generated *fix* varies — different runs produced `asyncio.to_thread`, a `ThreadPoolExecutor`, and a broken `BackgroundTasks` pattern that read cached data before it was ever written. This is a genuine, documented finding, not glossed over: diagnosis is a short, constrained output and stabilizes easily; code generation has many valid-looking implementation paths and doesn't fully stabilize at low temperature alone. In practice this means LLM-generated fixes need a verification step (tests, execution, or human review) before being trusted — a general limitation of LLM code generation, not specific to this pipeline.

An earlier version of this pipeline (before the retrieval fix above) also surfaced a distinct **context-contamination** failure mode: when irrelevant chunks were retrieved alongside the relevant one, the model would blend variable names and structure from the irrelevant context into the fix (e.g. renaming a correct variable to match unrelated code, or dropping a decorator that existed correctly in the original file). This was mitigated with explicit prompt constraints (`src/prompt.py`, rules 10-11) requiring the buggy code's existing names/structure to take precedence over anything seen only in retrieved context, and by narrowing retrieval to the single most relevant chunk for this sample.

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Hugging Face token (needs **Inference Providers** permission enabled):

    HF_TOKEN=your_token_here

Build the retrieval index:

```bash
python rebuild_index.py
```

Run this again any time a file under `corpus/` changes — the index is a snapshot, not rebuilt automatically.

---

## Run

```bash
python app.py
```

Opens a Gradio interface at `http://127.0.0.1:7860`. Upload an audio bug report and the corresponding code file to get a diagnosis and fix.

---

## Tech Stack

- **`faster-whisper`** — speech-to-text
- **`sentence-transformers`** (`all-MiniLM-L6-v2`) — embeddings
- **`faiss-cpu`** — vector search
- **`langchain-huggingface`** + Qwen2.5-Coder-7B-Instruct — generation
- **`gradio`** — interface

---

## Project Structure

```
├── app.py                    # Gradio interface
├── run_all_samples.py        # Runs the pipeline end-to-end on all 3 test samples
├── rebuild_index.py          # Re-chunks + re-embeds corpus/, rebuilds the FAISS index
├── results_log.txt           # Saved output of the verified run_all_samples.py run
├── src/
│   ├── transcribe.py         # Whisper transcription
│   ├── chunk.py               # AST-based code chunking
│   ├── embed.py                # Embedding + FAISS index building
│   ├── retrieve.py              # Retrieval logic
│   ├── prompt.py                 # Prompt template
│   ├── generate.py                 # LLM generation
│   └── pipeline.py                   # End-to-end orchestration
├── corpus/                    # Indexed codebase (Tata Motors project)
├── debugger_test_data/        # 3 validated test samples
└── tests/                     # Validation scripts for each pipeline stage
```

---

## Example Output

**Interface:**

![Gradio interface for the audio debugger](screenshots/gradio_interface.png)

**Sample result** — audio bug report → transcript → retrieved context → diagnosis + fix:

![Diagnosis and fix generated from an audio bug report](screenshots/sample_output.png)