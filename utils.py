# utils.py

import csv
import os
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

import psutil


def clean_text(s: str) -> str:
    s = (s or "").replace("</s>", "").replace("\x00", "").strip()
    return s.replace("{", "{{").replace("}", "}}")


def get_node_text(node) -> str:
    t = getattr(node, "text", None)
    if t:
        return t
    inner = getattr(node, "node", None)
    if inner is not None:
        t2 = getattr(inner, "text", None)
        if t2:
            return t2
        try:
            return inner.get_content()
        except Exception:
            pass
    try:
        return node.get_content()
    except Exception:
        return ""


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


def build_prompt(context: str, question: str, system_prompt: str) -> str:
    return clean_text(
        f"SYSTEM:\n{system_prompt}\n\n"
        f"CONTEXT:\n{context if context else '(none retrieved)'}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"ASSISTANT ANSWER:"
    )


def log_to_csv(row: dict, csv_path: str):
    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_gpu_util():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


class SystemSampler:
    def __init__(self, interval_sec: float = 0.1):
        """
        CPU / GPU Monitoring
        """
        self.interval: float = interval_sec
        self.cpu_samples: List[float] = []
        self.gpu_samples: List[float] = []
        self._stop: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self):
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            gpu = get_gpu_util()
            if gpu is not None:
                self.gpu_samples.append(gpu)
            time.sleep(self.interval)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)

    def summary(self):
        def stats(x):
            if not x:
                return None, None, None
            return min(x), sum(x) / len(x), max(x)

        cpu_min, cpu_avg, cpu_max = stats(self.cpu_samples)
        gpu_min, gpu_avg, gpu_max = stats(self.gpu_samples)

        return {
            "cpu_min": cpu_min,
            "cpu_avg": cpu_avg,
            "cpu_max": cpu_max,
            "gpu_min": gpu_min,
            "gpu_avg": gpu_avg,
            "gpu_max": gpu_max,
        }
