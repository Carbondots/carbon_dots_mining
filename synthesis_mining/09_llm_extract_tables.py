#!/usr/bin/env python3
"""Step 09: extract markdown tables from step-08 outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lmstudio import llm

from synthesis_unit import (
    DEFAULT_LLM_MODEL_NAME,
    SYNTHESIS_TABLE_COLUMNS,
    display_path,
    extract_document_id,
    list_document_dirs,
    parse_json_object_text,
    split_top_level_items,
    strip_think_block,
)


LLM_MODEL = llm(DEFAULT_LLM_MODEL_NAME)
TEMPERATURE = 0.1
MAX_TOKENS = 8000

PAIR_SEG_RE = re.compile(r"^(.+?)\s*\(([^()]*\d[^()]*)\)\s*$")
CANON_EXTRACT_RE = re.compile(r"[*_`]*\s*CDs\s*[_ ]?\s*Name\s*[*_`]*\s*[:\-]\s*(.+?)\s*$", re.I)



def write_error(error_path: Path, title: str, content: str) -> None:
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with error_path.open("a", encoding="utf-8") as f:
        f.write("\n\n---\n")
        f.write(f"### {title}\n\n{content}\n")



def append_name_log(pipeline_root: Path, document_id: str, text: str) -> None:
    log_path = pipeline_root / "extractions_name.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n=== DOC {document_id} ===\n{text}\n")



def append_letter_log(pipeline_root: Path, document_id: str, labels: set[str]) -> None:
    if not labels:
        return
    log_path = pipeline_root / "synthesis_letter.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("---\n")
        for label in sorted(labels):
            f.write(f"{document_id}: {label}\n")



def strip_wrapping(value: str) -> str:
    s = (value or "").strip().strip("\u200b\u2009\u202f")
    wraps = ("**", "__", "~~", "``", "`", "*", "_")
    for _ in range(2):
        changed = False
        for wrap in wraps:
            if s.startswith(wrap) and s.endswith(wrap) and len(s) >= 2 * len(wrap):
                s = s[len(wrap) : -len(wrap)].strip()
                changed = True
                break
        if not changed:
            break
    return s



def sanitize_sample_name(raw: str) -> str:
    s = strip_wrapping(raw).replace("*", "").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip().strip("'\"“”‘’")
    return s



def parse_canonical_names(text: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if "cds" not in line.lower() or "name" not in line.lower():
            continue
        m = CANON_EXTRACT_RE.search(line)
        raw = (m.group(1) if m else "").strip()
        if not raw:
            parts = re.split(r"[:\-]", line, maxsplit=1)
            if len(parts) > 1:
                raw = parts[1].strip()
        if not raw:
            continue
        name = sanitize_sample_name(raw)
        key = re.sub(r"[^0-9A-Za-z]+", "", name).lower()
        if key and key not in seen:
            seen.add(key)
            names.append(name)
    return names



def split_md_row(row: str) -> list[str]:
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [x.strip() for x in s.split("|")]



def parse_table_rows(table_md: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in table_md.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        if re.search(r"\|\s*:?-{2,}:?\s*\|", s):
            continue
        cols = [c.strip() for c in s.split("|")[1:-1]]
        if cols:
            rows.append(cols)
    return rows



def extract_first_valid_table(raw: str) -> str:
    sep_pat = re.compile(r"^\s*\|?(\s*[:\-]+?\s*\|)+\s*[:\-]+?\s*\|?\s*$")
    blocks: list[str] = []
    cur: list[str] = []
    for line in raw.splitlines():
        if line.strip().startswith("|"):
            cur.append(line.rstrip())
        elif cur:
            blocks.append("\n".join(cur))
            cur = []
    if cur:
        blocks.append("\n".join(cur))

    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        header_idx = next((i for i, ln in enumerate(lines) if not sep_pat.search(ln)), -1)
        if header_idx < 0:
            continue
        if len(split_md_row(lines[header_idx])) != len(SYNTHESIS_TABLE_COLUMNS):
            continue
        good = True
        for i, ln in enumerate(lines):
            if i == header_idx or sep_pat.search(ln):
                continue
            if len(split_md_row(ln)) != len(SYNTHESIS_TABLE_COLUMNS):
                good = False
                break
        if good:
            return block
    return ""



def validate_quantity(cell: str) -> bool:
    value = (cell or "").strip()
    if value == "N/A":
        return True
    parts = split_top_level_items(value, separators=",")
    if not parts:
        return False
    return all(PAIR_SEG_RE.match(part or "") for part in parts)



def validate_table(table_md: str, canonical_names: list[str]) -> tuple[bool, str, dict | None]:
    rows = parse_table_rows(table_md)
    if not rows:
        return False, "empty_rows", {
            "code": "empty_rows",
            "column": "",
            "row_name": "",
            "row_index": 0,
            "bad_value": "",
            "reason": "empty rows",
            "subject_value": "",
        }

    header = rows[0]
    if header != SYNTHESIS_TABLE_COLUMNS:
        return False, "header_check", {
            "code": "header_check",
            "column": "",
            "row_name": "",
            "row_index": 0,
            "bad_value": str(header),
            "reason": "header mismatch",
            "subject_value": "",
        }

    data = rows[1:]
    if len(data) != len(canonical_names):
        return False, "row_count_mismatch", {
            "code": "row_count_mismatch",
            "column": "",
            "row_name": "",
            "row_index": 0,
            "bad_value": str(len(data)),
            "reason": "row count mismatch",
            "subject_value": "",
        }

    expected = [strip_wrapping(x) for x in canonical_names]
    got = [strip_wrapping(row[0]) for row in data]
    if got != expected:
        return False, "name_order_check_strip", {
            "code": "name_order_check_strip",
            "column": "CDs_Naming_in_Paper",
            "row_name": "",
            "row_index": 0,
            "bad_value": str(got),
            "reason": "name/order mismatch",
            "subject_value": "",
        }

    idx_map = {name: i for i, name in enumerate(header)}
    for ridx, row in enumerate(data, start=1):
        if not validate_quantity(row[idx_map["Precursor_Amount"]]):
            return False, "amount_format", {
                "code": "amount_format",
                "column": "Precursor_Amount",
                "row_name": row[0],
                "row_index": ridx,
                "bad_value": row[idx_map["Precursor_Amount"]],
                "reason": "invalid quantity format",
                "subject_value": row[idx_map["Precursor"]],
            }
        if not validate_quantity(row[idx_map["Solvent_Volume"]]):
            return False, "volume_format", {
                "code": "volume_format",
                "column": "Solvent_Volume",
                "row_name": row[0],
                "row_index": ridx,
                "bad_value": row[idx_map["Solvent_Volume"]],
                "reason": "invalid quantity format",
                "subject_value": row[idx_map["Solvent"]],
            }

    return True, "ok", None



def rows_to_markdown(rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(SYNTHESIS_TABLE_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(SYNTHESIS_TABLE_COLUMNS)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join((c or "").replace("\n", " ").strip() for c in row) + " |")
    return "\n".join(lines)



def build_prompt(content: str, canonical_names: list[str], retry_hint: str = "") -> str:
    canonical_json = json.dumps(canonical_names, ensure_ascii=False)
    hint_block = ""
    if (retry_hint or "").strip():
        hint_block = f"""
# Previous Failure Hint (must fix in this attempt)
{retry_hint}
"""
    return f"""
# ROLE
You are a precision extraction model for carbon-dot synthesis tables.

# HARD CONSTRAINTS (non-negotiable)
- Canonical sample names and order are FIXED:
  CANONICAL_SAMPLE_NAMES = {canonical_json}
- Output exactly ONE row per canonical name, IN THIS ORDER.
- Column 1 (CDs_Naming_in_Paper) must match canonical names EXACTLY
  (identical spelling/casing), NO renaming/reordering.
- Output only ONE markdown table with EXACTLY 10 columns and exact header order.
- Do NOT invent numbers or units.
- "N/A" is the last resort only:
  if a value can be inferred from same-procedure relation in name content, do NOT output N/A.
- Columns Precursor_Amount and Solvent_Volume should use:
  single: SUBJECT (NUMBER UNIT)
  multi: SUBJECT_A (NUMBER UNIT), SUBJECT_B (NUMBER UNIT)

# Anti-error examples (generic placeholders)
- Bad: 10 mg (PRECURSOR_A) -> Good: PRECURSOR_A (10 mg)
- Bad: 4 mL (SOLVENT_A) -> Good: SOLVENT_A (4 mL)
- Bad: 20 mg (mg) -> Good: PRECURSOR_A (20 mg)
- Bad: PRECURSOR_A: 10 mg -> Good: PRECURSOR_A (10 mg)

# CSV Table Requirements (Markdown)
- Columns (in order, 10 columns):
  a. CDs_Naming_in_Paper
  b. Synthesis_Method
  c. Temperature
  d. Time
  e. Microwave_Power
  f. Precursor
  g. Precursor_Amount
  h. Solvent
  i. Solvent_Volume
  j. Purification

# Extraction Rules
- Solvent: Reaction solvent only. Use post-reaction solvents in Purification.
- If truly not derivable, use "N/A".
{hint_block}
# Silent self-check before output (do not print checklist)
1) exactly one markdown table
2) exact 10 headers in order
3) row count == len(CANONICAL_SAMPLE_NAMES)
4) first column matches canonical names in order
5) quantity/volume format is valid or N/A

# Name Content ({{id}}.md from name folder)
{content}

# Output
{{Markdown table only; no prose; no code fence}}
""".strip()



def _build_retry_hint(fail_info: dict | None) -> str:
    if not fail_info:
        return ""
    code = fail_info.get("code", "")
    col = fail_info.get("column", "")
    bad = fail_info.get("bad_value", "")
    if code not in {"amount_format", "volume_format"}:
        return f"Previous failure: {code}. Keep all hard constraints."
    return (
        f"Previous failure at {col}. Bad value: {bad}\n"
        "Use format SUBJECT (NUMBER UNIT).\n"
        "Examples:\n"
        "- Bad: 10 mg (PRECURSOR_A) -> Good: PRECURSOR_A (10 mg)\n"
        "- Bad: 4 mL (SOLVENT_A) -> Good: SOLVENT_A (4 mL)\n"
        "- Bad: 20 mg (mg) -> Good: PRECURSOR_A (20 mg)\n"
        "N/A is last resort only. If derivable from same-procedure relation in name content, do not use N/A."
    )



def build_repair_patch_prompt(
    name_content: str,
    canonical_names: list[str],
    table_md: str,
    fail_info: dict,
) -> str:
    canonical_json = json.dumps(canonical_names, ensure_ascii=False)
    fail_json = json.dumps(fail_info or {}, ensure_ascii=False)
    return f"""
# ROLE
You are a strict table patch generator.

# Goal
Fix quantity/volume format errors with MINIMAL edits, using name content ({{id}}.md) as the primary evidence.

# Hard constraints
- Canonical sample names and order are fixed: {canonical_json}
- Do not add/delete/reorder rows.
- Do not modify sample names.
- You may patch ONLY these columns:
  1) Precursor_Amount
  2) Solvent_Volume
- Do not invent numbers/units.
- N/A is last resort only:
  if value can be inferred from same-procedure relation in name content, do NOT output N/A.
- Preferred format:
  SUBJECT (NUMBER UNIT)
  SUBJECT_A (NUMBER UNIT), SUBJECT_B (NUMBER UNIT)

# Generic examples
- Bad: 10 mg (PRECURSOR_A) -> Good: PRECURSOR_A (10 mg)
- Bad: 4 mL (SOLVENT_A) -> Good: SOLVENT_A (4 mL)
- Bad: 20 mg (mg) -> Good: PRECURSOR_A (20 mg)

# Current fail_info
{fail_json}

# Current table
{table_md}

# Name content ({{id}}.md)
{name_content}

# Output format (JSON ONLY)
{{
  "patches": [
    {{"row_name": "SAMPLE_A", "column": "Precursor_Amount", "value": "PRECURSOR_X (10 mg)"}},
    {{"row_name": "SAMPLE_A", "column": "Solvent_Volume", "value": "SOLVENT_Y (4 mL)"}}
  ]
}}
""".strip()



def _validate_patch_payload(payload: dict, canonical_names: list[str]):
    if not isinstance(payload, dict):
        return False, "payload is not object", []
    patches = payload.get("patches")
    if not isinstance(patches, list) or not patches:
        return False, "patches must be non-empty list", []

    allowed_cols = {"Precursor_Amount", "Solvent_Volume"}
    canonical_set = set(canonical_names)
    canonical_strip = {strip_wrapping(x) for x in canonical_names}

    cleaned = []
    for patch in patches:
        if not isinstance(patch, dict):
            return False, "patch item must be object", []
        row_name = str(patch.get("row_name", "")).strip()
        column = str(patch.get("column", "")).strip()
        value = str(patch.get("value", "")).strip()
        if not row_name or not column:
            return False, "row_name/column required", []
        if column not in allowed_cols:
            return False, f"column not allowed: {column}", []
        if row_name not in canonical_set and strip_wrapping(row_name) not in canonical_strip:
            return False, f"row_name not in canonical: {row_name}", []
        cleaned.append({"row_name": row_name, "column": column, "value": value})
    return True, "", cleaned



def _apply_cell_patches(table_md: str, patches: list[dict], canonical_names: list[str]):
    rows = parse_table_rows(table_md)
    if not rows:
        return False, "", "cannot parse table rows"

    header = rows[0]
    if header != SYNTHESIS_TABLE_COLUMNS:
        return False, "", "header mismatch before patch"

    data = rows[1:]
    col_index = {name: idx for idx, name in enumerate(header)}
    row_index_exact = {row[0]: i for i, row in enumerate(data)}
    row_index_strip = {}
    for i, row in enumerate(data):
        row_index_strip.setdefault(strip_wrapping(row[0]), i)

    for patch in patches:
        row_name = patch["row_name"]
        column = patch["column"]
        value = patch["value"]
        ridx = row_index_exact.get(row_name)
        if ridx is None:
            ridx = row_index_strip.get(strip_wrapping(row_name))
        if ridx is None:
            return False, "", f"row not found: {row_name}"
        data[ridx][col_index[column]] = value

    return True, rows_to_markdown(data), ""



def call_table_llm(
    content: str,
    canonical_names: list[str],
    retry_hint: str = "",
    max_repair_tries_per_flow: int = 3,
) -> tuple[bool, str, str, str, dict | None]:
    prompt = build_prompt(content, canonical_names, retry_hint=retry_hint)
    result = LLM_MODEL.respond(prompt, config={"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS})
    raw = strip_think_block(getattr(result, "content", None) or "")
    last_raw = raw

    table = extract_first_valid_table(raw)
    if not table:
        fail_info = {
            "code": "no_markdown_table",
            "column": "",
            "row_name": "",
            "row_index": 0,
            "bad_value": "",
            "reason": "no markdown table",
            "subject_value": "",
        }
        return False, "", "no_markdown_table", last_raw, fail_info

    ok, code, fail_info = validate_table(table, canonical_names)
    if ok:
        return True, table, "ok", last_raw, None

    if code not in {"amount_format", "volume_format"}:
        return False, "", code, last_raw, fail_info

    current_table = table
    current_fail = fail_info
    for repair_try in range(1, max_repair_tries_per_flow + 1):
        repair_prompt = build_repair_patch_prompt(
            name_content=content,
            canonical_names=canonical_names,
            table_md=current_table,
            fail_info=current_fail or {},
        )
        repair_result = LLM_MODEL.respond(
            repair_prompt,
            config={"temperature": 0.0, "maxTokens": MAX_TOKENS},
        )
        repair_raw = strip_think_block(getattr(repair_result, "content", None) or "")
        last_raw = repair_raw

        payload = parse_json_object_text(repair_raw)
        if not payload:
            last_raw = f"[PatchJSONParseError] repair_try={repair_try}\n\n{repair_raw}"
            continue

        payload_ok, payload_reason, patches = _validate_patch_payload(payload, canonical_names)
        if not payload_ok:
            last_raw = (
                f"[PatchPayloadInvalid] repair_try={repair_try}\n"
                f"reason={payload_reason}\n"
                f"payload={payload}"
            )
            continue

        apply_ok, patched_table, apply_reason = _apply_cell_patches(current_table, patches, canonical_names)
        if not apply_ok:
            last_raw = (
                f"[PatchApplyError] repair_try={repair_try}\n"
                f"reason={apply_reason}\n"
                f"patches={patches}"
            )
            continue

        ok2, code2, fail2 = validate_table(patched_table, canonical_names)
        if ok2:
            return True, patched_table, "ok", last_raw, None

        current_table = patched_table
        current_fail = fail2 or current_fail
        if code2 not in {"amount_format", "volume_format"}:
            return False, "", code2, last_raw, current_fail

    final_code = (current_fail or {}).get("code", "patch_repair_failed")
    return False, "", final_code, last_raw, current_fail



def run(pipeline_root: Path, start_from_id: str, max_attempts: int, required_success: int, overwrite: bool) -> None:
    document_dirs = list_document_dirs(pipeline_root, start_from_id=start_from_id)
    print(f"Found {len(document_dirs)} candidate document folders.")

    for document_dir in document_dirs:
        document_id = extract_document_id(document_dir)
        input_md = document_dir / "Synthesis" / "LLM_name_qwen2.5vl" / f"{document_id}.md"
        out_dir = document_dir / "Synthesis" / "LLM_table_qwen2.5vl"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_md = out_dir / f"{document_id}_all_extractions.md"
        error_md = out_dir / f"{document_id}_error.md"

        labels: set[str] = set()

        if output_md.exists() and not overwrite:
            print(f"Skip existing output: {display_path(output_md, pipeline_root)}")
            continue

        if error_md.exists():
            try:
                error_md.unlink()
            except Exception:
                pass

        if not input_md.exists():
            labels.add("input missing")
            msg = f"Missing input markdown: {input_md}"
            write_error(error_md, "InputMissing", msg)
            append_name_log(pipeline_root, document_id, msg)
            append_letter_log(pipeline_root, document_id, labels)
            continue

        try:
            content = input_md.read_text(encoding="utf-8")
        except Exception as exc:
            labels.add("input read failed")
            msg = f"Cannot read input markdown: {exc}"
            write_error(error_md, "ReadError", msg)
            append_name_log(pipeline_root, document_id, msg)
            append_letter_log(pipeline_root, document_id, labels)
            continue

        canonical_names = parse_canonical_names(content)
        if not canonical_names:
            labels.add("no CDs_name parsed")
            msg = "No CDs_name entries found in input markdown."
            write_error(error_md, "NoCanonicalName", msg)
            append_name_log(pipeline_root, document_id, msg)
            append_letter_log(pipeline_root, document_id, labels)
            continue

        print(f"Processing {document_id} with {len(canonical_names)} canonical names.")
        row_records = {name: [] for name in canonical_names}
        retry_hint = ""

        for attempt in range(1, max_attempts + 1):
            print(f"Attempt {attempt}/{max_attempts}")
            try:
                ok, table, code, raw, fail_info = call_table_llm(
                    content,
                    canonical_names,
                    retry_hint=retry_hint,
                    max_repair_tries_per_flow=3,
                )
            except Exception as exc:
                ok, table, code, raw, fail_info = False, "", "exception", str(exc), None

            if not ok:
                labels.add(code)
                retry_hint = _build_retry_hint(fail_info)
                write_error(error_md, f"Attempt{attempt}_{code}", raw[:12000])
                append_name_log(pipeline_root, document_id, f"Attempt {attempt} failed: {code}")
                continue

            rows = parse_table_rows(table)[1:]
            for idx, row in enumerate(rows):
                name = canonical_names[idx]
                if len(row_records[name]) < required_success:
                    row_copy = row[:]
                    row_copy[0] = name
                    row_records[name].append(row_copy)

            if all(len(row_records[name]) >= required_success for name in canonical_names):
                break

        reached = [n for n in canonical_names if len(row_records[n]) >= required_success]
        dropped = [n for n in canonical_names if len(row_records[n]) < required_success]

        if not reached:
            labels.add("no row reached threshold")
            detail = ", ".join(f"{n}:{len(row_records[n])}/{required_success}" for n in canonical_names)
            append_name_log(pipeline_root, document_id, f"No sample reached threshold {required_success}. {detail}")
            append_letter_log(pipeline_root, document_id, labels)
            continue

        if dropped:
            labels.add("row below threshold")
            append_name_log(
                pipeline_root,
                document_id,
                f"Dropped samples below threshold {required_success}: {', '.join(dropped)}",
            )

        sections = []
        for k in range(required_success):
            rows_k = [row_records[name][k] for name in reached]
            sections.append(f"### Extraction {k + 1}\n\n{rows_to_markdown(rows_k)}\n")

        output_md.write_text("\n".join(sections), encoding="utf-8")
        print(f"Saved extraction tables: {display_path(output_md, pipeline_root)}")
        append_letter_log(pipeline_root, document_id, labels)



def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract markdown tables from LLM_name outputs.")
    parser.add_argument("--mining-root", dest="pipeline_root", type=Path, default=Path("data") / "mining")
    parser.add_argument("--start-from-id", type=str, default="")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--required-success", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    run(
        pipeline_root=args.pipeline_root,
        start_from_id=args.start_from_id,
        max_attempts=args.max_attempts,
        required_success=args.required_success,
        overwrite=args.overwrite,
    )
