# generate.py


"""
to change
"""

import datetime
import time

from config import Config
from models import load_embed
from parsers import args
from utils import SystemSampler, build_context, build_prompt, clean_text, log_to_csv


def generate_with_metrics(model, prompt: str):
    """
    Generation with latency, tokens, CPU/GPU.
    """
    sampler = SystemSampler()
    sampler.start()
    t0 = time.time()

    output_text = ""
    for tok in model.generate(
        prompt,
        streaming=True,
        kv_cache=None,
        stop_tokens=[2],  # hard code
        max_new_tokens=Config.MAX_NEW_TOKENS,
    ):
        output_text += tok

    t1 = time.time()
    sampler.stop()

    latency_s = t1 - t0
    tokens_generated = max(len(output_text.split()), 1)
    util = sampler.summary()

    return {
        "text": output_text.strip(),
        "generation_latency_ms": latency_s * 1000.0,
        "tokens_generated": tokens_generated,
        "tokens_per_sec": tokens_generated / latency_s if latency_s > 0 else 0.0,
        "util": util,
    }


def generate_chat(model, chat_history):
    """
    Interactive chat with RAG and metrics integrated
    """
    while True:
        try:
            prompt = input(">> ").strip()
            if not prompt:
                continue

            chat_history.append("user", prompt)

            t_retrieval_start = time.time()
            nodes = load_embed(args.embed, "cpu", "data", 3).retrieve(prompt)
            context = build_context(nodes, Config.MAX_CONTEXT_CHARS)
            retrieval_latency_ms = (time.time() - t_retrieval_start) * 1000.0

            dlg = build_prompt(context, prompt, Config.SYSTEM_PROMPT)
            gen = generate_with_metrics(model, dlg)

            from termcolor import cprint  # change to stream

            cprint(gen["text"], "blue")
            print()

            chat_history.append("bot", gen["text"])

            end_to_end_ms = retrieval_latency_ms + gen["generation_latency_ms"]
            log_row = {
                "timestamp": time.time(),
                "question": prompt,
                "answer": gen["text"],
                "retrieval_latency_ms": f"{retrieval_latency_ms:.2f}",
                "generation_latency_ms": f"{gen['generation_latency_ms']:.2f}",
                "end_to_end_ms": f"{end_to_end_ms:.2f}",
                "tokens_generated": gen["tokens_generated"],
                "tokens_per_sec": f"{gen['tokens_per_sec']:.2f}",
            }
            log_to_csv(log_row, csv_path=Config.LOG_FILE)

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"Error: {e}")
