import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        provider="auto",
        task="conversational",
        max_new_tokens=1024,
        temperature=0.1,
    )
    return ChatHuggingFace(llm=llm)

def generate_fix(prompt_text):
    chat_model = get_llm()
    response = chat_model.invoke(prompt_text)
    return response.content