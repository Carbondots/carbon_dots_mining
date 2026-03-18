#!/usr/bin/env python3
"""Folder Step 07 / Pipeline Step 11: merge multi-extraction markdown tables into one CSV per document."""

from __future__ import annotations

import argparse
import json
import re
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from lmstudio import llm

from synthesis_unit import (
    DEFAULT_LLM_MODEL_NAME,
    QUANTITY_COLUMN_PAIRS,
    QUANTITY_VALUE_COLUMNS,
    SYNTHESIS_TABLE_COLUMNS,
    display_path,
    extract_document_id,
    list_document_dirs,
    parse_markdown_table,
    split_top_level_items,
    strip_think_block,
)


LLM_MODEL = llm(DEFAULT_LLM_MODEL_NAME)
TEMPERATURE = 0.0
MAX_TOKENS = 1200
REASON_NA = "NA"
REASON_UNCERTAIN = "UNCERTAIN"

DEC_RE = re.compile(r"<decision>\s*(.*?)\s*</decision>", re.I | re.S)
VAL_RE = re.compile(r"<value>\s*(.*?)\s*</value>", re.I | re.S)
PAIR_RE = re.compile(r"^(.+?)\s*\(([^()]*\d[^()]*)\)\s*$")



def norm_case_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()



def parse_xml(raw: str) -> tuple[str, str]:
    dec_match = DEC_RE.search(raw or "")
    val_match = VAL_RE.search(raw or "")
    dec = dec_match.group(1).strip().upper() if dec_match else ""
    val = val_match.group(1).strip() if val_match else ""
    return dec, val



def postcheck_value(column: str, value: str) -> str:
    if not value:
        return "N/A"
    v = value.splitlines()[0].strip().strip('"').strip("'").strip("`")
    if len(v) > 300:
        return "N/A"
    if column in QUANTITY_VALUE_COLUMNS and v != "N/A":
        segments = split_top_level_items(v)
        if not segments or not all(PAIR_RE.match(seg) for seg in segments):
            return "N/A"
    return v or "N/A"



def build_merge_prompt(column: str, values: list[str]) -> str:
    cleaned = [str(v or "N/A") for v in values][:3]
    while len(cleaned) < 3:
        cleaned.append(cleaned[-1] if cleaned else "N/A")

    quantity = column in QUANTITY_VALUE_COLUMNS
    head = "quantity column" if quantity else "table column"
    rules = (
        "- Do NOT invent numbers.\n"
        "- Every number MUST bind to its subject, e.g., \"lysine (3 g)\".\n"
        "- If subjects/numbers conflict -> decision=CONFLICT, value=N/A."
        if quantity
        else
        "- If core meaning conflicts -> decision=CONFLICT, value=N/A.\n"
        "- If same meaning pick most specific; if compatible but incomplete MERGE concisely."
    )
    return f"""You will consolidate three extractions for the SAME {head}.

Column: {column}
Extraction1: {cleaned[0]}
Extraction2: {cleaned[1]}
Extraction3: {cleaned[2]}

Rules:
{rules}
Output ONLY this XML:
<decision>SAME|MERGE|CONFLICT</decision>
<value>FINAL_SINGLE_LINE_VALUE_OR_N/A</value>""".strip()



def llm_merge(column: str, values: list[str]) -> tuple[str, str]:
    non_empty = [str(v).strip() for v in values if str(v or "").strip() and str(v).strip() != "N/A"]
    if not non_empty:
        return "CONFLICT", "N/A"

    prompt = build_merge_prompt(column, values)
    try:
        result = LLM_MODEL.respond(prompt, config={"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS})
        raw = strip_think_block(getattr(result, "content", "") or "")
        dec, val = parse_xml(raw)
        if dec not in {"SAME", "MERGE", "CONFLICT"}:
            return "CONFLICT", "N/A"
        return dec, val or "N/A"
    except Exception:
        return "CONFLICT", "N/A"



def parse_all_extractions(path: Path) -> list[pd.DataFrame]:
    text = path.read_text(encoding="utf-8")
    chunks = re.findall(r"(### Extraction\s+\d+)(.*?)(?=### Extraction\s+\d+|$)", text, flags=re.S)
    out: list[pd.DataFrame] = []
    for _, block in chunks:
        df = parse_markdown_table(block)
        if df is not None:
            out.append(df)
    return out



def quantity_segments_are_valid(cell: str) -> bool:
    text = str(cell or "").strip()
    if not text or text == "N/A":
        return False
    segments = split_top_level_items(text)
    if not segments:
        return False
    return all(PAIR_RE.match(seg or "") for seg in segments)



def extract_pairs_multi(cell: str) -> list[tuple[str | None, str]]:
    if not cell or cell == "N/A":
        return []
    pairs: list[tuple[str | None, str]] = []
    for seg in split_top_level_items(cell):
        match = PAIR_RE.match(seg)
        if not match:
            continue
        subj = match.group(1).strip() or None
        num = match.group(2).strip()
        pairs.append((subj, num))
    return pairs



def merge_series(column_values: pd.Series, column: str, review_items: list[dict], document_id: str, cd_name: str) -> str:
    values = ["N/A" if v is None else str(v) for v in column_values.tolist()]
    non_na = [v for v in values if v.strip() and v != "N/A"]

    if not non_na:
        review_items.append(
            {
                "doc_id": document_id,
                "cd_name": cd_name,
                "column": column,
                "reason": REASON_NA,
                "current_value": "N/A",
                "raw_values": values,
            }
        )
        return "N/A"

    if len({norm_case_space(v) for v in non_na}) == 1:
        for v in values:
            if v != "N/A":
                return v

    decision, value = llm_merge(column, values)
    if decision == "CONFLICT":
        review_items.append(
            {
                "doc_id": document_id,
                "cd_name": cd_name,
                "column": column,
                "reason": REASON_NA,
                "current_value": "N/A",
                "raw_values": values,
            }
        )
        return "N/A"

    cleaned = postcheck_value(column, value)
    if cleaned == "N/A":
        review_items.append(
            {
                "doc_id": document_id,
                "cd_name": cd_name,
                "column": column,
                "reason": REASON_NA,
                "current_value": "N/A",
                "raw_values": values,
            }
        )
        return "N/A"

    if decision == "MERGE":
        review_items.append(
            {
                "doc_id": document_id,
                "cd_name": cd_name,
                "column": column,
                "reason": REASON_UNCERTAIN,
                "current_value": cleaned,
                "raw_values": values,
            }
        )

    return cleaned



def unify_extractions(df_list: list[pd.DataFrame], document_id: str, review_json_path: Path) -> pd.DataFrame:
    if not df_list:
        return pd.DataFrame(columns=SYNTHESIS_TABLE_COLUMNS)

    data = pd.concat(df_list, ignore_index=True)
    review_items: list[dict] = []
    merged_rows: list[dict] = []

    subject_cols = ["Synthesis_Method", "Temperature", "Time", "Microwave_Power", "Precursor", "Solvent", "Purification"]

    for cd_name, group in data.groupby("CDs_Naming_in_Paper", dropna=False):
        row = {"CDs_Naming_in_Paper": cd_name}

        for col in subject_cols:
            row[col] = merge_series(group[col], col, review_items, document_id, str(cd_name))

        for col, subj_col in QUANTITY_COLUMN_PAIRS:
            values = group[col].fillna("N/A").tolist()
            invalid = [v for v in values if str(v).strip() not in {"", "N/A"} and not quantity_segments_are_valid(str(v))]
            if invalid:
                row[col] = "N/A"
                review_items.append(
                    {
                        "doc_id": document_id,
                        "cd_name": str(cd_name),
                        "column": col,
                        "reason": "INVALID_FORMAT",
                        "current_value": "N/A",
                        "raw_values": values,
                    }
                )
                continue

            pair_counts: dict[tuple[str | None, str], int] = {}
            subj_first: dict[str | None, int] = {}
            subj_repr: dict[str | None, str] = {}
            tick = 0

            hint_subj = row.get(subj_col, "N/A")
            hint_key = norm_case_space(hint_subj) if hint_subj and hint_subj != "N/A" else None

            for cell in values:
                for subj_raw, num in extract_pairs_multi(str(cell or "")):
                    key = norm_case_space(subj_raw) if subj_raw else hint_key
                    if key not in subj_first:
                        subj_first[key] = tick
                        tick += 1
                    if key not in subj_repr:
                        subj_repr[key] = subj_raw if subj_raw else hint_subj
                    pair_counts[(key, num)] = pair_counts.get((key, num), 0) + 1

            if pair_counts:
                by_subj: dict[str | None, list[tuple[str, int]]] = {}
                for (subj_key, num), cnt in pair_counts.items():
                    by_subj.setdefault(subj_key, []).append((num, cnt))

                out_parts: list[str] = []
                for subj_key in sorted(by_subj.keys(), key=lambda x: (x is None, subj_first.get(x, 10**9))):
                    winner = sorted(by_subj[subj_key], key=lambda t: (-t[1], t[0]))[0][0]
                    if subj_key is None:
                        out_parts.append(f"({winner})")
                    else:
                        out_parts.append(f"{subj_repr.get(subj_key, hint_subj)} ({winner})")
                row[col] = ", ".join(dict.fromkeys(out_parts)) if out_parts else "N/A"
            else:
                decision, val = llm_merge(col, values)
                if decision == "CONFLICT":
                    row[col] = "N/A"
                    review_items.append(
                        {
                            "doc_id": document_id,
                            "cd_name": str(cd_name),
                            "column": col,
                            "reason": REASON_NA,
                            "current_value": "N/A",
                            "raw_values": values,
                        }
                    )
                else:
                    cleaned = postcheck_value(col, val)
                    row[col] = cleaned
                    if cleaned == "N/A" or decision == "MERGE":
                        review_items.append(
                            {
                                "doc_id": document_id,
                                "cd_name": str(cd_name),
                                "column": col,
                                "reason": REASON_UNCERTAIN if decision == "MERGE" else REASON_NA,
                                "current_value": cleaned,
                                "raw_values": values,
                            }
                        )

        merged_rows.append(row)

    review_json_path.write_text(
        json.dumps({"doc_id": document_id, "items": review_items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out_df = pd.DataFrame(merged_rows)
    for col in SYNTHESIS_TABLE_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = "N/A"
    return out_df[SYNTHESIS_TABLE_COLUMNS].fillna("N/A")



def run(pipeline_root: Path, overwrite: bool) -> None:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_log = pipeline_root / f"step10_run_{run_stamp}.log"
    global_json = pipeline_root / f"step10_run_{run_stamp}.json"

    done = 0
    skipped_missing = 0
    skipped_no_table = 0
    failed: list[dict] = []

    for document_dir in list_document_dirs(pipeline_root):
        document_id = extract_document_id(document_dir)
        input_md = document_dir / "Synthesis" / "LLM_table_qwen2.5vl" / f"{document_id}_all_extractions.md"
        output_csv = document_dir / "Synthesis" / "LLM_table_qwen2.5vl" / f"{document_id}.csv"
        review_json = document_dir / "Synthesis" / "LLM_table_qwen2.5vl" / f"{document_id}_review_todo.json"

        if output_csv.exists() and not overwrite:
            continue

        if not input_md.exists():
            skipped_missing += 1
            failed.append({"doc_id": document_id, "status": "skip_missing_input", "path": str(input_md)})
            continue

        try:
            dfs = parse_all_extractions(input_md)
        except Exception as exc:
            skipped_no_table += 1
            failed.append(
                {
                    "doc_id": document_id,
                    "status": "skip_parse_error",
                    "path": str(input_md),
                    "exception": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            continue

        if not dfs:
            skipped_no_table += 1
            failed.append({"doc_id": document_id, "status": "skip_no_table", "path": str(input_md)})
            continue

        try:
            out_df = unify_extractions(dfs, document_id=document_id, review_json_path=review_json)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            done += 1
            print(f"Merged CSV saved: {display_path(output_csv, pipeline_root)}")
        except Exception as exc:
            failed.append(
                {
                    "doc_id": document_id,
                    "status": "error_merge_or_write",
                    "path": str(output_csv),
                    "exception": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )

    summary = {
        "run_stamp": run_stamp,
        "pipeline_root": str(pipeline_root),
        "done": done,
        "skipped_missing_input": skipped_missing,
        "skipped_no_table_or_parse_error": skipped_no_table,
        "failed_count": len(failed),
        "failed_items": failed,
    }
    global_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"Run: {run_stamp}",
        f"Pipeline root: {pipeline_root}",
        f"Written: {done}",
        f"Skipped (missing input): {skipped_missing}",
        f"Skipped (no table/parse): {skipped_no_table}",
        f"Failed: {len(failed)}",
        "",
    ]
    for item in failed:
        lines.append(f"- {item.get('status')} | {item.get('doc_id')} | {item.get('path')}")
    global_log.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"Completed. Written={done}, skipped_missing={skipped_missing}, skipped_no_table={skipped_no_table}")
    print(f"Run log: {display_path(global_log, pipeline_root)}")
    print(f"Run json: {display_path(global_json, pipeline_root)}")



def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge step-09 extraction markdown into one CSV per document.")
    parser.add_argument("--mining-root", dest="pipeline_root", type=Path, default=Path("data") / "mining")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    run(args.pipeline_root, overwrite=args.overwrite)
