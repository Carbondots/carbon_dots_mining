#!/usr/bin/env python3
"""Folder Step 03 / Pipeline Step 07: classify retained chunks as synthesis-relevant with LLM."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from textwrap import dedent

import pandas as pd
from lmstudio import llm

from synthesis_unit import (
    DEFAULT_LLM_MODEL_NAME,
    display_path,
    extract_document_id,
    list_document_dirs,
    strip_think_block,
)


LLM_MODEL = llm(DEFAULT_LLM_MODEL_NAME)
TEMPERATURE = 0.1
MAX_TOKENS = 120



def build_prompt(text_block: str) -> str:
    prompt = dedent(
        """
        You are a materials scientist assistant.

        Your task is to determine whether the following paragraph describes the synthesis method
        or preparation process of carbon dots (CDs), including:

        If the paragraph contains any of the following synthesis-related information, answer "YES":
        - Any synthesis step (e.g., heating, hydrothermal, solvothermal, carbonization, microwave, pyrolysis, etc.)
        - Reaction conditions (e.g., temperature, time, precursor, precursor amount, solvent, solvent volume, microwave power, pressure)
        - Purification or post-treatment (e.g., centrifugation, dialysis, filtration, neutralization)
        - Synthesis Method
        - Temperature
        - Time
        - Microwave Power
        - Pressure
        - Cooling Method
        - Precursor
        - Precursor Amount
        - Solvent
        - Solvent Volume
        - Purification
        - CDs Naming in Paper
        - Is Multi CDs

        # -------------- Additional decision rule --------------
        Additionally, answer "YES" if the paragraph, in any wording, implies that detailed synthesis
        information is provided elsewhere (e.g. "Supporting Information", "Supplementary Materials",
        "Methods section", "Appendix", "ESI", "Table S1", etc.), regardless of the exact phrase used.
        # ------------------------------------------------------

        Please only answer "YES" or "NO".
        Do not explain your answer.NO REASEN! ONLY YES OR NO!

        Paragraph:
        """
    ).strip()
    return f"{prompt}\n{text_block.strip()}\n\nAnswer:"



def extract_yes_no(raw_response: str) -> str:
    clean = strip_think_block(raw_response).upper()
    if "YES" in clean:
        return "YES"
    if "NO" in clean:
        return "NO"
    return "NO"



def query_llm(text: str, max_retries: int, retry_delay: int) -> str:
    prompt = build_prompt(text)
    for attempt in range(1, max_retries + 1):
        try:
            result = LLM_MODEL.respond(prompt, config={"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS})
            raw = (result.content or "").strip()
            return extract_yes_no(raw)
        except Exception as exc:
            if attempt >= max_retries:
                print(f"LLM call failed after retries: {exc}")
                return "NO"
            time.sleep(retry_delay)
    return "NO"



def process_document(document_dir: Path, max_retries: int, retry_delay: int, overwrite: bool) -> None:
    document_id = extract_document_id(document_dir)
    cos_csv = document_dir / "Synthesis" / "cos" / f"{document_id}.csv"
    if not cos_csv.exists():
        print(f"Skip missing cosine CSV: {display_path(cos_csv)}")
        return

    out_dir = document_dir / "Synthesis" / "LLM_decision_32b"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"{document_id}.csv"
    out_txt = out_dir / f"{document_id}.txt"
    out_md = out_dir / f"{document_id}.md"
    if out_csv.exists() and out_txt.exists() and out_md.exists() and not overwrite:
        print(f"Skip already processed: {document_id}")
        return

    df = pd.read_csv(cos_csv, encoding="utf-8")
    if "text" not in df.columns or "retain" not in df.columns:
        print(f"Skip invalid input columns: {display_path(cos_csv)}")
        return

    df["LLM_decision"] = "NO"
    yes_index = df.index[df["retain"] == "YES"].tolist()

    for idx in yes_index:
        df.at[idx, "LLM_decision"] = query_llm(str(df.at[idx, "text"]), max_retries=max_retries, retry_delay=retry_delay)

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with out_txt.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            handle.write(
                f"[Para {row.get('para_id', 'N/A')}, Tokens: {row.get('token_count', 'N/A')}] "
                f"(retain={row['retain']}, LLM_decision={row['LLM_decision']}):\n"
                f"{str(row['text']).strip()}\n\n\n"
            )

    with out_md.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            handle.write(
                f"[Para {row.get('para_id', 'N/A')}, Tokens: {row.get('token_count', 'N/A')}]  \n"
                f"**retain**: {row['retain']} | **LLM_decision**: {row['LLM_decision']}\n\n"
                f"{str(row['text']).strip()}\n\n---\n---\n\n"
            )

    print(f"Completed: {document_id}")



def run(pipeline_root: Path, max_retries: int, retry_delay: int, overwrite: bool) -> None:
    for document_dir in list_document_dirs(pipeline_root):
        process_document(document_dir, max_retries=max_retries, retry_delay=retry_delay, overwrite=overwrite)



def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM YES/NO classification on retained chunks.")
    parser.add_argument("--mining-root", dest="pipeline_root", type=Path, default=Path("data") / "mining")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    run(args.pipeline_root, max_retries=args.max_retries, retry_delay=args.retry_delay, overwrite=args.overwrite)
