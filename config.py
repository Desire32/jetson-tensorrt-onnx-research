# config.py

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    DATA_DIR: str = "data"
    TOP_K: int = 3
    CHUNK_SIZE: int = 350
    CHUNK_OVERLAP: int = 60
    MAX_CONTEXT_CHARS: int = 2500
    MAX_NEW_TOKENS: int = 250
    LOG_FILE: str = "rag_query_logs.csv"
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    MODEL_TINY_LLAMA: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    API: str = "mlc"
    MODEL_SHEARED_LLAMA: str = "princeton-nlp/Sheared-LLaMA-1.3B"
    MODEL_EMBED: str = "sentence-transformers/all-MiniLM-L6-v2"
    SYSTEM_PROMPT: str = (
        "You are a helpful assistant.\n"
        "Answer using ONLY the provided context.\n"
        'If the answer is not in the context, say: "I don\'t know based on the provided context."'
    )
    TEST_PROMPTS = [
        "How does the public authorities protect and preserve the monument of Panagia Aggeloktisti from deterioration?",
        "Who is the owner of the Panagia aggeloktisti moument?",
        "What are some of the conservation works that have been taking place in Panagia aggeloktisti?",
        "",
    ]
