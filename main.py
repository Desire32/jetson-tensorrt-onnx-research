# main.py
# ------------------------------------------
import argparse
import os
import subprocess
from dataclasses import dataclass

from termcolor import cprint

from config import Config
from generate import generate_chat
from models import load_embed, load_nano_llm
from parsers import args
from utils import SystemSampler, build_context, build_prompt, clean_text, log_to_csv


# ----------------------------------------------------------
# init clean from cuda memory (root rights must have)
def clear_memory():
    subprocess.run("sync && echo 3 > /proc/sys/vm/drop_caches", shell=True)


clear_memory()

# working_slm_models = ['TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'princeton-nlp/Sheared-LLaMA-1.3B']


def main():
    config = Config()
    model, chat_history = load_nano_llm(config, args.model, args.api)
    generate_chat(model, chat_history)


if __name__ == "__main__":
    main()
