# main.py
# ------------------------------------------
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from termcolor import cprint

from config import Config
from models import load_embed, load_nano_llm
from utils import log_to_csv


def main():
    config = Config()
    model, chat_history = load_nano_llm(config, config.MODEL_TINY_LLAMA, config.API)
    retriever = load_embed(config.MODEL_EMBED, "cpu", "data", 3)

    while True:
        t0 = time.time()  # _0
        start_time = datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S")
        print(">> ", end="", flush=True)
        prompt = input().strip()

        nodes = retriever.retrieve(prompt)

        context = "\n\n".join([node.text for node in nodes])
        chat_history.append("user", f"Context: {context}\n\nQuestion: {prompt}")
        embedding, _ = chat_history.embed_chat()

        text = ""
        reply = model.generate(
            embedding,
            streaming=True,
            kv_cache=chat_history.kv_cache,
            stop_tokens=chat_history.template.stop,
            max_new_tokens=config.MAX_NEW_TOKENS,
        )

        for token in reply:
            text += token
            cprint(token, "blue", end="\n\n" if reply.eos else "", flush=True)

        print("\n")
        tokens_generated = max(len(text.split()), 1)

        # -------------------------------- bot communication
        chat_history.append("bot", text)

        if len(chat_history) > 3:
            chat_history.reset()
        # --------------------------------

        t1 = time.time()  # _1
        end_time = datetime.fromtimestamp(t1).strftime("%Y-%m-%d %H:%M:%S")
        latency_s = t1 - t0
        print(f"Start: {start_time}")
        print(f"End: {end_time}")
        print(f"Generation latency: {latency_s * 1000:.2f} ms")
        print(f"Tokens generated: {tokens_generated}")
        print(
            f"Tokens per second: {tokens_generated / latency_s if latency_s > 0 else 0:.2f}"
        )

        log_to_csv(
            {
                "start_time": start_time,
                "end_time": end_time,
                "latency_ms": latency_s * 1000.0,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_generated / latency_s
                if latency_s > 0
                else 0.0,
                "prompt": prompt,
                "reply": text,
            }
        )


if __name__ == "__main__":
    main()
