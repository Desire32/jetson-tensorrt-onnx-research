# utils.py
import csv
import os
import wandb
import time


def log_system_metrics(interval=2):
    """
    free -h analog.
    NOTE: SWAP hasn't been used in this implementation, so we turned it off.
    """
    while True:
        with open("/proc/meminfo") as f:
            meminfo = {line.split(":")[0]: int(line.split()[1]) for line in f}

        mem_total_mb = round(meminfo["MemTotal"] / 1024, 1)
        mem_free_mb = round(meminfo["MemFree"] / 1024, 1)
        mem_available_mb = round(meminfo["MemAvailable"] / 1024, 1)

        wandb.log(
            {
                "mem/total_MB": mem_total_mb,
                "mem/free_MB": mem_free_mb,
                "mem/available_MB": mem_available_mb,
            }
        )
        time.sleep(interval)


def clean_text(s: str) -> str:
    s = (s or "").replace("</s>", "").replace("\x00", "").strip()
    return s.replace("{", "{{").replace("}", "}}")


def build_context(nodes, max_chars: int) -> str:
    parts = []
    total = 0
    for n in nodes:
        chunk = clean_text(get_node_text(n))
        if not chunk:
            continue
        block = chunk + "\n\n"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)
    return "".join(parts).strip()


def build_prompt(system_prompt: str, context: str, question: str) -> str:
    # Keep the same plain role-structured prompt that worked well for you
    # and add a minimal anti-repetition instruction.
    return clean_text(
        f"SYSTEM:\n{system_prompt}\n"
        "Answer using ONLY the provided context.\n"
        "If the answer is not in the context, say: \"I don't know based on the provided context.\".\n"
        "Do not repeat the same sentence.\n\n"
        f"CONTEXT:\n{context if context else '(none retrieved)'}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"ASSISTANT ANSWER:"
    )
