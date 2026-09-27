PROMPT_TEMPLATE = """You are a debugging assistant reviewing a machine learning codebase.

Your job is to diagnose the BUGGY CODE using the BUG REPORT and the relevant project context.

RELEVANT CODE CONTEXT FROM THE PROJECT:
{context}

BUG REPORT (transcribed from audio):
{transcript}

BUGGY CODE:
{buggy_code}

DEBUGGING RULES:
1. Treat the BUGGY CODE as the primary code being diagnosed.
2. Use the RELEVANT CODE CONTEXT only as supporting information about how the project works.
3. Identify a root cause that is actually supported by the code shown above.
4. The proposed root cause must explain the specific symptom described in the BUG REPORT.
5. Do not invent runtime behavior that cannot occur from the shown code.
6. For runtime, hanging, timeout, or performance problems, inspect blocking I/O, synchronous operations, external calls, model inference, and resource/threading behavior before assuming a data-validation problem.
7. If the code explicitly raises an exception for a condition, do not describe that condition as an infinite loop or silent hang unless there is additional code that can cause such behavior.
8. Reference specific variables, functions, or operations from the BUGGY CODE when explaining the diagnosis.
9. The FIX must address the diagnosed root cause, not merely add error handling around an unrelated condition.
10. Never change a variable name, decorator, or function signature that already exists correctly in the BUGGY CODE to match something seen only in the RELEVANT CODE CONTEXT. The BUGGY CODE's existing names and structure are authoritative; the context is for understanding behavior, not for copying syntax.
11. If the FIX uses any name, module, or function call that requires an import (e.g. copied from RELEVANT CODE CONTEXT), the corresponding import statement must also be present in the FIX. Check every import against every name actually used before finalizing.
12. Return the complete corrected version of the BUGGY CODE.

Respond in exactly this format:

DIAGNOSIS:
Explain the root cause and why it produces the reported symptom. Reference specific variable/function names or operations from the BUGGY CODE.

FIX:
Provide the complete corrected version of the BUGGY CODE with the root cause fixed. Do not include the RELEVANT CODE CONTEXT in the fix.
"""

def build_prompt(context_chunks, transcript, buggy_code):
    context = "\n\n".join(
        f"[{c['file']} :: {c['name']}]\n{c['code']}" for c in context_chunks
    )
    return PROMPT_TEMPLATE.format(
        context=context, transcript=transcript, buggy_code=buggy_code
    )