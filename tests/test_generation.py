from src.retrieve import load_index, retrieve
from src.prompt import build_prompt
from src.generate import generate_fix

index, chunks = load_index()

transcript = "Hey, I'm working on my Tata Motors BiLSTM model. It builds fine, model.summary() looks totally normal, and training even runs without any errors. But when I try to run predictions on my test set afterward, I get a shape mismatch error from the input layer. I don't get why the model trained just fine, so I'm confused where the mismatch is even coming from."

with open("debugger_test_data/sample_1_shape_bug/buggy_code.py") as f:
    buggy_code = f.read()

context_chunks = retrieve(transcript, index, chunks, k=4)
prompt = build_prompt(context_chunks, transcript, buggy_code)

result = generate_fix(prompt)
print(result)