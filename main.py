# main.py
# ------------------------------------------

import subprocess

from config import Config
from generate import generate_chat
from models import load_nano_llm
from parsers import args


# ----------------------------------------------------------
# init clean from cuda memory (root rights must have)
def clear_memory():
    subprocess.run("sync && echo 3 > /proc/sys/vm/drop_caches", shell=True)


clear_memory()


def main():
    config = Config()
    model, chat_history = load_nano_llm(config, args.model, args.api)
    generate_chat(model, chat_history, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
