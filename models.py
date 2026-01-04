# models.py
# ------------------------------------------
import time
from typing import Tuple

import wandb
from nano_llm import ChatHistory, NanoLLM

from config import Config
from parsers import args


def load_nano_llm(
    config: Config, model_path: str, api: str
) -> Tuple[NanoLLM, ChatHistory]:
    """
    Load NanoLLM model and initialize chat history.
    Logs model loading metrics to the active wandb run.
    """

    t0 = time.time()
    try:
        model = NanoLLM.from_pretrained(
            model=model_path,
            api=args.api,
            api_token=config.HF_TOKEN,
            quantization=args.quantization,
            use_safetensors=True,
        )
    except Exception as e:
        wandb.log({"model_load_error": str(e)})
        raise RuntimeError(f"Failed to load NanoLLM model: {e}")

    chat_history = ChatHistory(model, system_prompt=config.SYSTEM_PROMPT)
    num_params = sum(p.numel() for p in model.parameters())
    load_time_s = time.time() - t0
    wandb.log(
        {
            "model/load_time_s": load_time_s,
            "model/path": model_path,
            "model/api": args.api,
            "model/num_params": num_params / 1e6,
            "model/quantization": args.quantization,
            "model/system_prompt_length": len(config.SYSTEM_PROMPT),
        }
    )

    return model, chat_history


def load_embed(model: str, device: str, directory: str, top_k: int):
    """
    Build embedding index and retriever.
    Logs embedding + indexing metrics.
    """

    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    t0 = time.time()

    # -------- embedding model init
    embed_init_t0 = time.time()
    embed_model = HuggingFaceEmbedding(model_name=model, device=device)
    embed_init_s = time.time() - embed_init_t0

    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )

    doc_load_t0 = time.time()
    documents = SimpleDirectoryReader(directory).load_data()
    doc_load_s = time.time() - doc_load_t0

    num_documents = len(documents)

    index_build_t0 = time.time()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index_build_s = time.time() - index_build_t0

    retriever = index.as_retriever(similarity_top_k=top_k)
    total_init_s = time.time() - t0

    wandb.log(
        {
            "embed/init_time_s": embed_init_s,
            "embed/doc_load_time_s": doc_load_s,
            "embed/index_build_time_s": index_build_s,
            "embed/total_init_time_s": total_init_s,
            "embed/model": args.model,
            "embed/device": device,
            "embed/top_k": top_k,
            "embed/chunk_size": Config.CHUNK_SIZE,
            "embed/chunk_overlap": Config.CHUNK_OVERLAP,
            "embed/num_documents": num_documents,
        }
    )

    return retriever
