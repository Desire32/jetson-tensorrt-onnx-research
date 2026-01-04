# main.py
# ------------------------------------------
import os
import threading
import time
from datetime import datetime

import wandb
from termcolor import cprint

from config import Config
from models import load_embed, load_nano_llm
from utils import log_system_metrics


def main():
    config = Config()

    wandb.init(
        project="jetson-llm-bench",
        name=f"tinyllama-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "model": config.MODEL_SHEARED_LLAMA,
            "embed_model": config.MODEL_EMBED,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "device": "cpu",
        },
    )
    thread = threading.Thread(target=log_system_metrics, daemon=True)
    thread.start()

    model, chat_history = load_nano_llm(
        config, config.MODEL_SHEARED_LLAMA, config.API
    )
    retriever = load_embed(config.MODEL_EMBED, "cpu", "data", 3)

    step = 0

    while True:
        t0 = time.time()
        print(">> ", end="", flush=True)
        prompt = input().strip()

        nodes = retriever.retrieve(prompt)
        context = "\n\n".join([node.text for node in nodes])

        chat_history.append(
            "user", f"Context: {context}\n\nQuestion: {prompt}"
        )
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

        # -------- bot communication
        chat_history.append("bot", text)
        if len(chat_history) > 3:
            chat_history.reset()

        t1 = time.time()
        latency_s = t1 - t0

        tokens_per_second = (
            tokens_generated / latency_s if latency_s > 0 else 0.0
        )

        wandb.log(
            {
                "latency_ms": latency_s * 1000.0,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_per_second,
                "prompt_length": len(prompt.split()),
            },
            step=step,
        )

        step += 1

if __name__ == "__main__":
    main()
