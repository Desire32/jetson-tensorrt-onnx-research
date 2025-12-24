# utils.py

import csv
import os


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


def log_to_csv(row: dict, csv_path: str):
    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
