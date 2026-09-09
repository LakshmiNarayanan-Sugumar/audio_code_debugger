PROMPT_TEMPLATE = """You are a debugging assistant reviewing a machine learning codebase.

RELEVANT CODE CONTEXT FROM THE PROJECT (for reference only — do not include this in your fix):
{context}

BUG REPORT (transcribed from audio):
{transcript}

BUGGY CODE:
{buggy_code}

Based on the bug report and the relevant context above, respond in exactly this format:

DIAGNOSIS: Explain what is causing the bug and why, referencing specific variable/function names.

FIX: Provide the complete corrected version of the BUGGY CODE shown above, with the bug fixed. Do not include or repeat the RELEVANT CODE CONTEXT shown above — that is reference material only, not part of the file being fixed."""


def build_prompt(context_chunks, transcript, buggy_code):
    context = "\n\n".join(
        f"[{c['file']} :: {c['name']}]\n{c['code']}" for c in context_chunks
    )
    return PROMPT_TEMPLATE.format(
        context=context, transcript=transcript, buggy_code=buggy_code
    )