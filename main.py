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
from parsers import args
from utils import log_system_metrics

if args.api == "hf":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # really does help


"""
to clean memory before launch:
sudo sync echo 3 | sudo tee /proc/sys/vm/drop_caches
"""

"""
NOTE:
    the reason why 'hf' mode works so slowly, is because we turn off gpu help completely
"""

"""
hf mode for shearedllama:

    ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
    See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
"""


def main():
    config = Config()

    wandb.init(
        project="final-core",
        config={
            "model": args.model,
            "embed_model": args.embed,
            "max_new_tokens": args.max_new_tokens,
            "quantization": args.quantization,
        },
        name=f"{args.model}--{args.embed}quantization-{args.api}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )

    # for memory tracking
    thread = threading.Thread(target=log_system_metrics, daemon=True)
    thread.start()

    model, chat_history = load_nano_llm(config, args.model, config.API)
    retriever = load_embed(args.embed, "cpu", "data", config.TOP_K)

    step = 0

    while True:
        t0 = time.time()
        print(">> ", end="", flush=True)
        prompt = input().strip()
        print(f"[USER] {prompt}")

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
            max_new_tokens=args.max_new_tokens,
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

        tokens_per_second = tokens_generated / latency_s if latency_s > 0 else 0.0

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
