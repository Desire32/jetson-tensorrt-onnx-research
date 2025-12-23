# config.py

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    DATA_DIR: str = "data"
    TOP_K: int = 3
    CHUNK_SIZE: int = 350
    CHUNK_OVERLAP: int = 60
    MAX_CONTEXT_CHARS: int = 1500
    MAX_NEW_TOKENS: int = 96
    LOG_FILE: str = "rag_query_logs.csv"
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    SYSTEM_PROMPT: str = (
        "You are a helpful assistant.\n"
        "Answer using ONLY the provided context.\n"
        'If the answer is not in the context, say: "I don\'t know based on the provided context."'
    )
