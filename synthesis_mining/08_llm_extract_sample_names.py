#!/usr/bin/env python3
"""Step 08: Extract per-sample names and synthesis descriptions with LLM."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from textwrap import dedent

import pandas as pd
from lmstudio import llm

from synthesis_unit import (
    DEFAULT_LLM_MODEL_NAME,
    collect_yes_paragraph_texts,
    extract_document_id,
    list_document_dirs,
    strip_think_block,
)


LLM_MODEL = llm(DEFAULT_LLM_MODEL_NAME)
TEMPERATURE = 0.0
MAX_TOKENS = 30000
MAX_RETRIES = 3
RETRY_DELAY = 15



def build_prompt(original_text: str, refined_text: str) -> str:
    return dedent(
        f"""
        You are a synthesis researcher in carbon dots.

        Your task is to extract synthesis details for carbon dots (CDs) explicitly synthesized in this paper (exclude CDs cited from other works).

        For each CDs synthesized in this paper, provide clearly:

        - CDs Name: (Provide the given name; if none is provided explicitly, assign a meaningful and distinct name based on synthesis characteristics (e.g., N-CDs (Urea), Microwave-CDs (Citric Acid)).)
        - Synthesis Method: (hydrothermal, microwave, pyrolysis, etc.)
        - Precursors: (list explicitly)
        - Reaction Solvents: (specify explicitly)
        - Reaction Conditions:
            - Temperature (°C)
            - Time (hours/minutes)
            - Precursor quantity (mass or volume), explicitly stating which precursor each quantity refers to.
            - Solvent volume, explicitly stating which solvent it refers to.
            - Microwave power (if mentioned)
            - Pressure conditions (if mentioned)
        - Purification Methods: (dialysis, filtration, centrifugation, explicitly stating solvents used if mentioned)

        STRICT Exclusion:
        - Do NOT include CDs from referenced studies or comparative analyses.
        - Do NOT include CDs composited or hybridized with other materials.
        - Do NOT include post-synthesis modifications.
        - Exclude any CD names or synthesis descriptions matching the regex:
        r'(?i)(@|/|composite|hybrid|doped|modified)'
        - Tables containing citations (e.g., [12], superscripts) must be fully excluded.

        Special Instruction for Multiple CDs:
        If multiple CDs are synthesized and the text states their method is the same, similar,
        or equivalent to another, you must still explicitly and quantitatively repeat the
        full synthesis details for each CD separately in the
        "Detailed Synthesis Methods for Each Type of CDs" section.
        Also, list control samples or baseline CDs (e.g., same method except without a precursor) as separate entries with full synthesis details.

        Output Format (plain text only, no tables):

        Number of synthesized CDs: X

        Detailed Synthesis Methods for Each Type of CDs:

        1. CDs_name: [Name]
        Synthesis Method: [Method], using [precursors] dissolved in [reaction solvent], at [temperature]°C for [time]. Precursor quantity: [quantity clearly matched to precursor]. Solvent volume: [volume clearly matched to solvent]. Purified by [methods, explicitly state solvents if mentioned].

        (Repeat for each CD synthesized.)

        ---

        Original Text:
        {original_text}

        Refined Text:
        {refined_text}
        """
    ).strip()



def query_llm(original_text: str, refined_text: str, log_file_path: Path) -> str | None:
    prompt = build_prompt(original_text, refined_text)

    with log_file_path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n=== INPUT TO LLM ===\n")
        handle.write(prompt + "\n")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = LLM_MODEL.respond(prompt, config={"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS})
            raw = (result.content or "").strip()

            with log_file_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n=== OUTPUT (ATTEMPT {attempt}) ===\n")
                handle.write(raw + "\n")

            return strip_think_block(raw)
        except Exception as exc:
            with log_file_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\nLLM error on attempt {attempt}: {exc}\n")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                return None
    return None



def process_document(document_dir: Path, overwrite: bool) -> None:
    document_id = extract_document_id(document_dir)

    decision_csv = document_dir / "Synthesis" / "LLM_decision_32b" / f"{document_id}.csv"
    abstract_md = document_dir / "Synthesis" / "LLM_abstract_qwen2.5vl" / f"{document_id}.md"
    out_dir = document_dir / "Synthesis" / "LLM_name_qwen2.5vl"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_md = out_dir / f"{document_id}.md"
    err_md = out_dir / f"{document_id}_error.md"
    log_txt = out_dir / f"{document_id}_in_output.txt"

    if out_md.exists() and not overwrite:
        print(f"Skip already processed: {document_id}")
        return
    if not abstract_md.exists() or not decision_csv.exists():
        return

    try:
        refined_text = abstract_md.read_text(encoding="utf-8").strip()
    except Exception as exc:
        err_md.write_text(f"# Error\n\nCannot read refined markdown: {exc}\n", encoding="utf-8")
        return

    df = pd.read_csv(decision_csv, encoding="utf-8")
    if "LLM_decision" not in df.columns or "text" not in df.columns:
        err_md.write_text("# Error\n\nMissing required columns in decision CSV.\n", encoding="utf-8")
        return

    yes_texts = collect_yes_paragraph_texts(df)
    if not yes_texts:
        err_md.write_text("# Error\n\nNo YES paragraphs found in decision CSV.\n", encoding="utf-8")
        return

    original_text = "\n".join(yes_texts).strip()
    if not original_text:
        err_md.write_text("# Error\n\nMerged YES text is empty.\n", encoding="utf-8")
        return

    output_text = query_llm(original_text=original_text, refined_text=refined_text, log_file_path=log_txt)
    if output_text is None:
        err_md.write_text("# Error\n\nLLM call failed after retries.\n", encoding="utf-8")
        return

    final_text = refined_text
    if output_text.strip() and "No additional details required" not in output_text:
        final_text += "\n\n" + output_text

    final_text = "# Carbon Dots Name and Method Extraction\n\n" + final_text
    out_md.write_text(final_text, encoding="utf-8")
    if err_md.exists():
        err_md.unlink()
    print(f"Completed: {document_id}")


def run(pipeline_root: Path, overwrite: bool) -> None:
    for document_dir in list_document_dirs(pipeline_root):
        process_document(document_dir, overwrite)



def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CD names and methods from refined synthesis text.")
    parser.add_argument("--mining-root", dest="pipeline_root", type=Path, default=Path("data") / "mining")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    run(args.pipeline_root, overwrite=args.overwrite)
