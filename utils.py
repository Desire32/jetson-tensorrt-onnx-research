# utils.py

import csv
import os
import time

import wandb


def clean_text(s: str) -> str:
    s = (s or "").replace("</s>", "").replace("\x00", "").strip()
    return s.replace("{", "{{").replace("}", "}}")


def build_prompt(context: str, question: str, system_prompt: str) -> str:
    return clean_text(
        f"SYSTEM:\n{system_prompt}\n\n"
        f"CONTEXT:\n{context if context else '(none retrieved)'}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"ASSISTANT ANSWER:"
    )


def log_system_metrics(interval=5):
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
        # swap_total_mb = round(meminfo["SwapTotal"] / 1024, 1)
        # swap_free_mb = round(meminfo["SwapFree"] / 1024, 1)

        wandb.log(
            {
                "mem/total_MB": mem_total_mb,
                "mem/free_MB": mem_free_mb,
                "mem/available_MB": mem_available_mb,
                # "swap/total_MB": swap_total_mb,
                # "swap/free_MB": swap_free_mb,
            }
        )

        time.sleep(interval)
