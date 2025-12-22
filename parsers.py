# parsers.py

import argparse

# [https://github.com/dusty-nv/NanoLLM/blob/main/nano_llm/chat/example.py]
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--model",
    type=str,
    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    help="path to the model",  # HuggingFace repo/model name, or path to HF model checkpoint
)
parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=256,
    help="the maximum response length for each bot reply",
)
parser.add_argument(
    "--api", type=str, default="mlc", help="api format for a model"
)  # mlc, awq, hf
parser.add_argument(
    "--embed",
    type=str,
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="choose embed model",
)
args = parser.parse_args()
