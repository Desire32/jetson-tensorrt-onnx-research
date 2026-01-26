 # config.py
import os
from dataclasses import dataclass, field

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
    MODEL_SHEARED_LLAMA: str = "princeton-nlp/Sheared-LLaMA-1.3B"
    MODEL_EMBED: str = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_MXBAI: str = "mixedbread-ai/mxbai-edge-colbert-v0-17m"
    API: str = "mlc"
    SYSTEM_PROMPT: str = (
        "You are an expert cultural heritage assistant focused exclusively on the Byzantine church"
        "Panagia tis Angeloktistis (Panagia Aggeloktisti), Kiti, Cyprus.\n"
        "Answer concisely.\n"
        "Use only the provided context.\n"
        "If the answer is not in the context, say you don't know based on the provided context."
        "Do not repeat you self"
    )
