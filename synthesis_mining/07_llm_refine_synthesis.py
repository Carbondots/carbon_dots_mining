#!/usr/bin/env python3
"""Step 07: Refine YES paragraphs into synthesis-only summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import pandas as pd
from lmstudio import llm

from synthesis_unit import (
    DEFAULT_LLM_MODEL_NAME,
    collect_yes_paragraph_texts,
    extract_document_id,
    list_document_dirs,
    strip_code_fence_block,
    strip_think_block,
)


LLM_MODEL = llm(DEFAULT_LLM_MODEL_NAME)
TEMPERATURE = 0.1
MAX_TOKENS = 10000



def build_prompt(text_block: str) -> str:
    prompt = dedent(
        """
        You are a senior synthesis expert specializing in carbon dots (CDs).
        You have extensive expertise in experimental synthesis methods for CDs
        and are tasked with carefully analyzing scientific text.

        Your mission is to extract ONLY the synthesis details of CDs that were
        actually performed in THIS PAPER'S experimental work.

        1. KEEP ONLY sentences that explicitly describe the synthesis of CDs performed in THIS PAPER, including:
           - Carbonization methods (hydrothermal, solvothermal, microwave, pyrolysis, electrochemical, etc.)
           - Reaction conditions (temperature, time, pressure)
           - Precursors and solvents used
           - Purification methods (dialysis, centrifugation, filtration)

        2. DISCARD ALL sentences about:
           - Composite or hybrid formation with other materials
           - Post-synthetic modifications, characterizations, applications, reviews
           - Synthesis details referenced from OTHER studies

        3. OUTPUT:
           - First, list each extracted synthesis sentence verbatim in original order.
           - Then, provide one aggregation paragraph that summarizes all details.

        STRICT BOUNDARY:
        Only process text between the markers:
        ===== TEXT START =====
        and
        ===== TEXT END =====
        Do NOT include any other content or example text in your output.
        """
    ).strip()
    return f"{prompt}\n===== TEXT START =====\n{text_block.strip()}\n===== TEXT END ====="



def call_llm_refine(text_block: str, log_path: Path) -> str:
    prompt = build_prompt(text_block)

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n=== PROMPT (truncated 4k) ===\n")
        handle.write(prompt[:4000] + "\n")

    response = LLM_MODEL.respond(prompt, config={"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS})
    raw = (response.content or "").strip()
    if not raw:
        raise RuntimeError("Empty LLM response")

    cleaned = strip_code_fence_block(strip_think_block(raw))
    if not cleaned.strip():
        raise RuntimeError("LLM response became empty after cleaning")

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n=== OUTPUT ===\n")
        handle.write(cleaned + "\n")

    return cleaned



def process_document(document_dir: Path, overwrite: bool) -> None:
    document_id = extract_document_id(document_dir)
    decision_csv = document_dir / "Synthesis" / "LLM_decision_32b" / f"{document_id}.csv"
    out_dir = document_dir / "Synthesis" / "LLM_abstract_qwen2.5vl"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_md = out_dir / f"{document_id}.md"
    err_md = out_dir / f"{document_id}_error.md"
    log_txt = out_dir / "in_output.txt"

    if out_md.exists() and not overwrite:
        print(f"Skip already processed: {document_id}")
        return
    if not decision_csv.exists():
        return

    df = pd.read_csv(decision_csv, encoding="utf-8")
    if not {"LLM_decision", "text", "para_id"}.issubset(df.columns):
        return

    yes_texts = collect_yes_paragraph_texts(df)
    if not yes_texts:
        return

    combined_text = "\n".join(yes_texts)

    try:
        refined = call_llm_refine(combined_text, log_txt)
        out_md.write_text(refined + "\n", encoding="utf-8")
        if err_md.exists():
            err_md.unlink()
        print(f"Completed: {document_id}")
    except Exception as exc:
        err_md.write_text(f"# Error\n\n{exc}\n", encoding="utf-8")
        print(f"Failed: {document_id} ({exc})")


def run(pipeline_root: Path, overwrite: bool) -> None:
    for document_dir in list_document_dirs(pipeline_root):
        process_document(document_dir, overwrite)



def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine YES chunks into synthesis-only summaries.")
    parser.add_argument("--mining-root", dest="pipeline_root", type=Path, default=Path("data") / "mining")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    run(args.pipeline_root, overwrite=args.overwrite)
