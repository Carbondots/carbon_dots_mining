#!/usr/bin/env python3
"""Step 12: fill review-target fields with evidence-based LLM re-check."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from lmstudio import llm

from synthesis_unit import (
    DEFAULT_LLM_MODEL_NAME,
    QUANTITY_VALUE_COLUMNS,
    SYNTHESIS_TABLE_COLUMNS,
    append_text,
    build_cd_description_from_row,
    display_path,
    extract_document_id,
    list_document_dirs,
    load_decision_context_text,
    parse_json_object_text,
    read_csv_safely,
    replace_path_subfolder_and_suffix,
    strip_code_fence_block,
    strip_think_block,
)


LLM_MODEL = llm(DEFAULT_LLM_MODEL_NAME)
TEMPERATURE = 0.1
MAX_TOKENS = 8000

IN_SUBFOLDER = "LLM_table_qwen2.5vl"
OUT_SUBFOLDER = "letter_table"
REVIEW_JSON_SUBFOLDER = "LLM_table_qwen2.5vl"
DECISION_CSV_SUBFOLDER = "LLM_decision_32b"

def build_single_field_prompt(
    cd_name: str,
    column: str,
    cd_desc: str,
    context_text: str,
    raw_values: list,
    current_value: str,
) -> str:
    field_rule = make_field_rule(column)
    prompt = f"""You are a careful research assistant for carbon dot (CD) synthesis.

# CD to locate (read this FIRST)
- CD name: {cd_name}
- CD description (use ONLY to find the right paragraph; do NOT copy values from it):
  {cd_desc}

# Paper context (the ONLY source of truth)
====================
{context_text}
====================

# TASK (ONE FIELD ONLY)
Extract **ONLY** the value for column **"{column}"** for the CD named **{cd_name}** from the context above.
- If the answer is not clearly stated in the context, return "N/A".
- Do **NOT** infer or invent numbers/units.
- Answer should be **one concise line**.

# Field-specific rule
{field_rule}

# Prior attempts (for awareness; DO NOT trust them blindly)
- Current merged value: {current_value}
- Three extractions: {raw_values}

# STRICT OUTPUT (JSON ONLY)
Return **valid JSON** with exactly ONE key and ONE value.
The key must be **exactly**: "{column}"

Example (shape only):
{{
  "{column}": "<your answer or N/A>"
}}

No extra text. No markdown fences. No additional keys.
"""
    return prompt



def call_llm_json_single(prompt: str, expected_key: str, max_retries: int = 3, wait_seconds: int = 15):
    last_raw = ""
    for _ in range(max_retries):
        try:
            result = LLM_MODEL.respond(prompt, config={"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS})
            raw = (result.content or "").strip()
            last_raw = raw

            cleaned = strip_code_fence_block(strip_think_block(raw))
            parsed = parse_json_object_text(cleaned)
            if isinstance(parsed, dict) and expected_key in parsed:
                val = parsed.get(expected_key, "N/A")
                if not isinstance(val, str):
                    val = json.dumps(val, ensure_ascii=False)
                return val, raw, parsed
        except Exception:
            time.sleep(wait_seconds)

    return "N/A", last_raw, {}



def io_log_path(final_csv_path: Path) -> Path:
    return final_csv_path.parent / "in_output.txt"


def make_field_rule(column: str) -> str:
    if column in QUANTITY_VALUE_COLUMNS:
        return f'- For "{column}": Do NOT invent numbers; tie every number to its subject (e.g., "lysine (3 g)"); if multiple entries, separate with ", "; if unclear, return "N/A".'
    if column == "Solvent":
        return '- For "Solvent": reaction solvent only; washing/dialysis liquids belong to Purification; if unclear, return "N/A".'
    if column == "Purification":
        return '- For "Purification": purification steps only (e.g., filtration/centrifugation/dialysis) in one concise line; if unclear, return "N/A".'
    if column in {"Temperature", "Time", "Microwave_Power"}:
        return f'- For "{column}": one line; if multiple steps exist, list concisely; if unclear, return "N/A".'
    return f'- For "{column}": one concise line; if unclear, return "N/A".'

def run(pipeline_root: Path, overwrite: bool) -> None:
    document_dirs = list_document_dirs(pipeline_root)
    print(f"Found {len(document_dirs)} document folders.")

    for document_dir in document_dirs:
        document_id = extract_document_id(document_dir)

        merged_csv_path = document_dir / "Synthesis" / IN_SUBFOLDER / f"{document_id}.csv"
        final_csv_path = document_dir / "Synthesis" / OUT_SUBFOLDER / f"{document_id}.csv"

        if final_csv_path.exists() and not overwrite:
            print(f"Skip existing final CSV: {display_path(final_csv_path, pipeline_root)}")
            continue
        if not merged_csv_path.exists():
            continue

        final_csv_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = io_log_path(final_csv_path)

        review_json_path = replace_path_subfolder_and_suffix(
            merged_csv_path,
            IN_SUBFOLDER,
            REVIEW_JSON_SUBFOLDER,
            ".json",
        )
        review_json_path = review_json_path.with_name(review_json_path.stem + "_review_todo.json")

        decision_csv_path = replace_path_subfolder_and_suffix(
            merged_csv_path,
            IN_SUBFOLDER,
            DECISION_CSV_SUBFOLDER,
            ".csv",
        )

        merged_df = read_csv_safely(merged_csv_path)
        if merged_df is None:
            print(f"Skip unreadable merged CSV: {display_path(merged_csv_path, pipeline_root)}")
            continue

        if not review_json_path.exists():
            print(f"Skip missing review JSON: {display_path(review_json_path, pipeline_root)}")
            continue

        try:
            review_payload = json.loads(review_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Skip invalid review JSON: {display_path(review_json_path, pipeline_root)} ({exc})")
            continue

        items = review_payload.get("items", []) if isinstance(review_payload, dict) else []
        if not items:
            print(f"Skip empty review items: {display_path(review_json_path, pipeline_root)}")
            continue

        context_text = load_decision_context_text(decision_csv_path)
        if not context_text.strip():
            print(f"Skip empty decision context: {display_path(decision_csv_path, pipeline_root)}")
            continue

        todo_by_cd: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            cd_name = item.get("cd_name", "")
            if cd_name:
                todo_by_cd[cd_name].append(item)

        total_updates = 0

        for row_index in merged_df.index:
            row = merged_df.loc[row_index].to_dict()
            cd_name = row.get("CDs_Naming_in_Paper", "")
            reqs = todo_by_cd.get(cd_name, [])
            if not reqs:
                continue

            request_columns = []
            for req in reqs:
                col = req.get("column")
                if col and col in SYNTHESIS_TABLE_COLUMNS and col not in request_columns:
                    request_columns.append(col)

            updates_this_cd = 0
            for col in request_columns:
                raw_vals = []
                for req in reqs:
                    if req.get("column") == col:
                        raw_vals = req.get("raw_values", [])
                        break

                current_val = row.get(col, "N/A")
                cd_desc = build_cd_description_from_row(row, exclude_columns={col})

                prompt = build_single_field_prompt(
                    cd_name=cd_name,
                    column=col,
                    cd_desc=cd_desc,
                    context_text=context_text,
                    raw_values=raw_vals,
                    current_value=current_val,
                )

                append_text(log_path, f"\n\n=== DOC: {final_csv_path.name} | CD: {cd_name} | COL: {col} ===")
                append_text(log_path, "--- PROMPT BEGIN ---")
                append_text(log_path, prompt)
                append_text(log_path, "--- PROMPT END ---")

                value, raw_reply, parsed = call_llm_json_single(prompt, expected_key=col)

                append_text(log_path, "--- RAW REPLY BEGIN ---")
                append_text(log_path, raw_reply or "[EMPTY]")
                append_text(log_path, "--- RAW REPLY END ---")
                append_text(log_path, "--- PARSED JSON ---")
                append_text(log_path, json.dumps(parsed if parsed else {col: value}, ensure_ascii=False, indent=2))

                before = str(merged_df.at[row_index, col]) if col in merged_df.columns else "N/A"
                merged_df.at[row_index, col] = value
                after = value

                if before != after:
                    updates_this_cd += 1
                append_text(log_path, f"[WRITE] {cd_name} | {col}: {before} => {after}")

            total_updates += updates_this_cd
            if updates_this_cd == 0:
                append_text(log_path, f"[NOTE] No field updated for CD: {cd_name}")

        merged_df.to_csv(final_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Saved final CSV: {display_path(final_csv_path, pipeline_root)} (updated cells: {total_updates})")

    print("All documents processed.")



def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM re-check review-target fields and write final CSVs.")
    parser.add_argument("--mining-root", dest="pipeline_root", type=Path, default=Path("data") / "mining")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    run(args.pipeline_root, overwrite=args.overwrite)
