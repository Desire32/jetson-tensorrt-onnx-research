# main.py
#
import os
import threading
import time
from datetime import datetime
from typing import Iterable, Optional, Tuple

import wandb
from config import Config
from models import load_embed, load_nano_llm
from parsers import args
from utils import build_context, build_prompt, clean_text, log_system_metrics

if args.api == "hf":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

STOP_TOKENS = [2]  # TinyLlama / Llama-family EOS token id
DEFAULT_MAX_CONTEXT_CHARS = 1500


def generate_stream(model, prompt: str) -> Iterable[str]:
    generation_kwargs = {
        "streaming": True,
        "kv_cache": None,
        "stop_tokens": STOP_TOKENS,
        "max_new_tokens": args.max_new_tokens,
    }

    try:
        return model.generate(prompt, **generation_kwargs)
    except TypeError:
        pass

    for embed_method in ("embed_text", "embed"):
        if hasattr(model, embed_method):
            emb = getattr(model, embed_method)(prompt)
            return model.generate(emb, **generation_kwargs)

    raise RuntimeError(
        "NanoLLM generate interface not supported (no prompt or embed_* path)."
    )


def consume_stream_collect(
    stream: Iterable[str],
    t_gen_start: Optional[float] = None,
) -> Tuple[str, Optional[float]]:
    """
    Consume the stream silently, collect the full answer, and compute TTFT.
    No per-chunk printing. Returns full answer and TTFT (seconds).
    """
    cut_markers = ["\nUSER QUESTION:", "\nCONTEXT:", "\nSYSTEM:"]
    HOLD = 64
    buf = ""
    out = ""

    ttft_s: Optional[float] = None
    first_tok_seen = False

    # mild repetition guard
    words = []
    ngram_counts = {}
    N = 12
    MAX_HITS = 3

    for tok in stream:
        if (not first_tok_seen) and (t_gen_start is not None) and tok and tok.strip():
            first_tok_seen = True
            ttft_s = time.perf_counter() - t_gen_start

        buf += tok

        # repetition guard
        tw = tok.strip().split()
        if tw:
            words.extend(tw)
            if len(words) >= N:
                ngram = " ".join(words[-N:]).lower()
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
                if ngram_counts[ngram] >= MAX_HITS:
                    break

        # cut markers
        hit_pos = None
        for m in cut_markers:
            p = buf.find(m)
            if p != -1:
                hit_pos = p if hit_pos is None else min(hit_pos, p)

        if hit_pos is not None:
            out += buf[:hit_pos]
            return out, ttft_s

        # holdback flush
        if len(buf) > HOLD:
            out += buf[:-HOLD]
            buf = buf[-HOLD:]

    if buf.strip():
        out += buf

    return out, ttft_s

def main():
    config = Config()

    latency_store = 0
    tokens_gen_store = 0
    tokens_per_second_store = 0
    ttft_store = 0
    count = 0

    run_name = (
        f"{args.model}__{args.embed}__{args.api}__{args.quantization}__"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    wandb.init(
        project="SECTION_5. metrics",
        config={
            "model": args.model,
            "embed_model": args.embed,
            "max_new_tokens": args.max_new_tokens,
            "quantization": args.quantization,
            "api": args.api,
            "top_k": getattr(config, "TOP_K", None),
        },
        name=run_name,
        reinit=True,
    )

    thread = threading.Thread(target=log_system_metrics, daemon=True)
    thread.start()

    model, _chat_history = load_nano_llm(config, args.model, config.API)
    retriever = load_embed(args.embed, "cpu", "data", config.TOP_K)

    warmup_remaining = 1

    while True:
        print(">> ", end="", flush=True)
        try:
            raw_q = input()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        question = clean_text(raw_q.strip())
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        t0 = time.perf_counter()

        print(f"[QUESTION]: {question}")

        nodes = retriever.retrieve(question) or []

        max_chars = getattr(config, "MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS)
        context = build_context(nodes, max_chars)
        prompt = build_prompt(
            system_prompt=config.SYSTEM_PROMPT,
            context=context,
            question=question
        )

        t_gen_start = time.perf_counter()
        stream = generate_stream(model, prompt)
        answer_text, ttft_s = consume_stream_collect(stream, t_gen_start=t_gen_start)
        t_gen_end = time.perf_counter()

        latency_s = t_gen_end - t0

        answer_text = clean_text(answer_text).strip()

        print(f"[ASSISTANT] {answer_text}\n", flush=True)

        if warmup_remaining > 0:
            warmup_remaining -= 1
            continue

        tokens_generated = max(len(answer_text.split()), 1)
        tokens_per_second = tokens_generated / latency_s if latency_s > 0 else 0.0

        count += 1
        latency_store += latency_s
        tokens_gen_store += tokens_generated
        tokens_per_second_store += tokens_per_second
        if ttft_s is not None:
            ttft_store += ttft_s * 1000

        wandb.log(
            {
                "latency_ms": (latency_store / count) * 1000.0 + (ttft_store/ count) if ttft_s is not None else None, # full process (latency + ttft)
                "ttft_ms": (ttft_store / count) if ttft_s is not None else None,
                "tokens_generated": tokens_gen_store / count,
                "tokens_per_second": tokens_per_second_store / count,
            }
        )


if __name__ == "__main__":
    main()
