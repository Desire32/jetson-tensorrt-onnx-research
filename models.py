# models.py


import time
from typing import Tuple

from nano_llm import ChatHistory, NanoLLM

from config import Config


def load_nano_llm(
    config: Config, model_path: str, api: str
) -> Tuple[NanoLLM, ChatHistory]:
    """

    Args:
        config
        model_path
        api:  API (mlc, awq, hf)

    Returns:
        tuple: (model, chat_history)
    """
    try:
        model = NanoLLM.from_pretrained(
            model=model_path,
            api=api,
            api_token=config.HF_TOKEN,
            quantization="q4f16_ft",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load NanoLLM model: {e}")

    chat_history = ChatHistory(model, system_prompt=config.SYSTEM_PROMPT)

    return model, chat_history


def load_embed(model: str, device: str, directory: str, top_k: int):
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    embed_model = HuggingFaceEmbedding(model_name=model, device=device)

    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(
        chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
    )
    r0 = time.time()
    documents = SimpleDirectoryReader(directory).load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=top_k)
    r1 = time.time()
    print(f"Retrieval start: {r0}")
    print(f"Retrieval end: {r1}")
    print(f"Retrieval latency: {time.time() - r0 * 1000.0}")
    return retriever
