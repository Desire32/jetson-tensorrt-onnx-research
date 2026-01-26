# parsers.py

import argparse
from config import Config
# [https://github.com/dusty-nv/NanoLLM/blob/main/nano_llm/chat/example.py]
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--model",
    type=str,
    choices=[
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "princeton-nlp/Sheared-LLaMA-1.3B",
    ],
    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    help="path to the model",  # HuggingFace repo/model name, or path to HF model checkpoint
)

# quantization
parser.add_argument(
    "--quantization",
    type=str,
    choices=[
        "q4f16_ft",
        "q8f16_ft",
        "q4f16_0",
        "q4f16_1",
        "q0f16",
        "q0f32",
        "q3f16_0",
        "q3f16_1",
    ],
    default="q4f16_ft",
    help="Choose quantization type",
)

"""
build.py: error: argument --quantization: invalid choice: 'none' (choose from 'autogptq_llama_q4f16_0', 'autogptq_llama_q4f16_1', 'q0f16', 'q0f32', 'q3f16_0', 'q3f16_1', 'q4f16_0', 'q4f16_1', 'q4f16_2', 'q4f16_ft',
    'q4f16_ft_group', 'q4f32_0', 'q4f32_1', 'q8f16_ft', 'q8f16_ft_group', 'q8f16_1')
"""

# max new tokens
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=256,
    help="Maximum response length for each bot reply",
)

# embed model
parser.add_argument(
    "--embed",
    type=str,
    choices=[
        "sentence-transformers/all-MiniLM-L6-v2",
        "mixedbread-ai/mxbai-edge-colbert-v0-17m",
    ],
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Choose embedding model",
)

# api (mlc / awq / hf)
parser.add_argument(
    "--api",
    type=str,
    default="mlc",
    choices=["mlc", "awq", "hf"],
    help="API format for model",
)

parser.add_argument("--chunk_size", type=int, default=Config.CHUNK_SIZE)
parser.add_argument("--chunk_overlap", type=int, default=Config.CHUNK_OVERLAP)
parser.add_argument("--top_k", type=int, default=Config.TOP_K)
#parser.add_argument("--max_context_len", type=int, default=256) #added by LN - desparate attempt

args = parser.parse_args()
