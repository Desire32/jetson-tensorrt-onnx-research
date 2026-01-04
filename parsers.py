# parsers.py

import argparse

# [https://github.com/dusty-nv/NanoLLM/blob/main/nano_llm/chat/example.py]
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--model",
    type=str,
    choices=[
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "ShearedLlama/ShearedLlama-1.1B-Chat-v1.0",
    ],
    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    help="path to the model",  # HuggingFace repo/model name, or path to HF model checkpoint
)

# quantization
parser.add_argument(
    "--quantization",
    type=str,
    choices=["q4f16_ft", "q4f16", "fp16"],
    default="q4f16_ft",
    help="Choose quantization type",
)

# max new tokens
parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=256,
    help="Maximum response length for each bot reply",
)

# embed model
parser.add_argument(
    "--embed",
    type=str,
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

args = parser.parse_args()
