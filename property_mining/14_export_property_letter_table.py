#!/usr/bin/env python3

"""Step 14: export the Step 13 structured property markdown to property letter-table CSV."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from property_unit import (
    append_log_line as append_log,
    ensure_dir,
    ensure_root_exists,
    iter_paper_dirs,
    normalize_property_sample_name,
    paper_id_from_dir,
    read_text,
    relative_to_paper,
    relative_to_root,
    stage_markdown_path,
    timestamp_now,
)


INPUT_STAGE_DIR = "change_resolved_properties"
OUTPUT_DIRNAME = "letter_table"
SAMPLE_HEADER_RE = re.compile(r"^(#+)\s+(.*\S)\s*$")
PAPER_HEADER_RE = re.compile(r"^#\s*Paper\b", re.IGNORECASE)
ENTRY_START_RE = re.compile(r"^\s*(\d+)\.\s+.*\S\s*$")
EVIDENCE_LINE_RE = re.compile(r"^\s*evidence\s*:\s*$", re.IGNORECASE)
PROPERTY_ABSTRACT_RE = re.compile(r"^\s*Property abstract\s*:\s*$", re.IGNORECASE)
STRUCTURED_RE = re.compile(r"^\s*Structured\s*:\s*$", re.IGNORECASE)
TAG_LINE_RE = re.compile(
    r"^\s*(?P<prefix>\[(?P<prefix_inner>[^\]]+)\])?\s*(?P<tag>[A-Za-z][A-Za-z0-9_]*)\s*:\s*(?P<sent>.+?)\s*$"
)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
KNOWN_SUFFIXES = {"(APP)", "(VS)", "(VS-refined)", "(VS-REFINED)", "(MAIN)"}

CSV_COLS = [
    "CDs_Naming_in_Paper",
    "Ex",
    "Em",
    "QY",
    "lifetime",
    "ExDep",
    "Chiral",
    "CPL",
]
TARGET_COLS = set(CSV_COLS[1:])
NUMERIC_COLS = {"Ex", "Em", "QY", "lifetime"}

TAG_MAP = {
    "EX": "Ex",
    "EM": "Em",
    "QY": "QY",
    "LIFETIME": "lifetime",
    "EXDEP": "ExDep",
    "EXDEPENDENT": "ExDep",
    "EXCITATIONDEPENDENT": "ExDep",
    "EXCITATIONDEPENDENCE": "ExDep",
    "CHIRAL": "Chiral",
    "CPL": "CPL",
}

VARY_BY_MAP = {
    "ph": "pH",
    "solvent": "solvent",
    "medium": "solvent",
    "state": "solvent",
    "phase": "solvent",
    "ex": "Ex",
    "excitation": "Ex",
    "excitationwavelength": "Ex",
    "em": "Em",
    "emission": "Em",
    "emissionwavelength": "Em",
    "component": "component",
    "components": "component",
    "tau": "component",
}

LIFETIME_UNIT_MAP = {
    "fs": "fs",
    "ps": "ps",
    "ns": "ns",
    "us": "us",
    "ms": "ms",
    "s": "s",
    "\u03bcs": "us",
    "microsecond": "us",
    "microseconds": "us",
    "nanosecond": "ns",
    "nanoseconds": "ns",
    "picosecond": "ps",
    "picoseconds": "ps",
    "femtosecond": "fs",
    "femtoseconds": "fs",
    "millisecond": "ms",
    "milliseconds": "ms",
    "second": "s",
    "seconds": "s",
}


@dataclass
class ParseFail:
    md_path: str
    sample_name: str
    struct_line: str
    reason: str


@dataclass
class Step14Result:
    paper_id: str
    paper_dir: str
    status: str
    input_md: str
    output_csv: str
    note: str = ""


def normalize_sample_header(text: str) -> str:
    sample = normalize_property_sample_name(
        text,
        blank_series_level=True,
        extra_suffixes=tuple(KNOWN_SUFFIXES),
    )
    return re.sub(r"\s+", " ", str(sample or "")).strip()


def normalize_tag_name(tag: str) -> str:
    if not isinstance(tag, str):
        return ""
    compact = re.sub(r"\s+", "", tag).upper()
    return TAG_MAP.get(compact, tag.strip())


def normalize_vary_by(vary_by: Any) -> str:
    text = str(vary_by or "").strip()
    if not text:
        return ""
    key = re.sub(r"\s+", "", text).lower()
    return VARY_BY_MAP.get(key, text)


def normalize_lifetime_unit_token(unit: str) -> str:
    return LIFETIME_UNIT_MAP.get(str(unit or "").strip().lower(), "")


def normalize_unit_for_tag(tag: str, unit: Any) -> str:
    raw = str(unit or "").strip()
    if not raw:
        return ""
    if tag in {"Ex", "Em"}:
        return "nm"
    if tag == "QY":
        return "%"
    if tag == "lifetime":
        return normalize_lifetime_unit_token(raw) or raw
    return raw


def fmt_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return format_number(value)
    return str(value).strip()


def to_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        match = NUM_RE.search(str(value).strip())
        if not match:
            return None
        return float(match.group(0))
    except Exception:
        return None


def format_number(value: Any) -> str:
    numeric = to_float(value)
    if numeric is None:
        return str(value).strip()
    if abs(numeric - round(numeric)) < 1e-12:
        return str(int(round(numeric)))
    return ("%0.10f" % numeric).rstrip("0").rstrip(".")


def attach_unit(value_text: str, unit: str) -> str:
    value = str(value_text or "").strip()
    final_unit = str(unit or "").strip()
    if not value:
        return ""
    if not final_unit:
        return value
    if final_unit == "%":
        return f"{value}%"
    return f"{value} {final_unit}"


def as_num_text_list(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, list):
        for item in value:
            out.extend(as_num_text_list(item))
        return out
    if isinstance(value, bool) or value is None:
        return out
    if isinstance(value, (int, float)):
        return [format_number(value)]
    text = str(value).strip()
    if not text:
        return out
    matches = NUM_RE.findall(text)
    if matches:
        return [format_number(match) for match in matches]
    return [text]


def as_label_list(value: Any) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            text = fmt_scalar(item)
            if text:
                out.append(text)
        return out
    text = fmt_scalar(value)
    return [text] if text else []


def format_condition_values(vary_by: str, value: Any) -> List[str]:
    normalized = normalize_vary_by(vary_by)
    if not normalized:
        return []
    if normalized in {"Ex", "Em"}:
        return [attach_unit(item, "nm") for item in as_num_text_list(value)]
    if normalized == "pH":
        numbers = as_num_text_list(value)
        return numbers if numbers else as_label_list(value)
    return as_label_list(value)


def format_numeric_item(tag: str, item: Dict[str, Any]) -> List[str]:
    raw_values = item.get("values", item.get("value", None))
    values = [attach_unit(value, normalize_unit_for_tag(tag, item.get("unit", ""))) for value in as_num_text_list(raw_values)]
    values = [value for value in values if str(value).strip()]
    if not values:
        return []
    vary_by = normalize_vary_by(item.get("vary_by", ""))
    if not vary_by:
        return values
    vary_values = format_condition_values(vary_by, item.get("vary_values", item.get("vary_value", None)))
    if not vary_values:
        return [f"{value}({vary_by})" for value in values]
    if len(vary_values) == 1:
        return [f"{value}({vary_by}:{vary_values[0]})" for value in values]
    if len(vary_values) == len(values):
        return [f"{values[index]}({vary_by}:{vary_values[index]})" for index in range(len(values))]
    if len(values) == 1:
        return [f"{values[0]}({vary_by}:{','.join(vary_values)})"]
    return [f"{value}({vary_by}:{vary_values[0]})" for value in values]


def json_strip_trailing_commas(text: str) -> str:
    previous = None
    current = str(text or "")
    while current != previous:
        previous = current
        current = TRAILING_COMMA_RE.sub(r"\1", current)
    return current


def py_literal_from_json_tokens(text: str) -> str:
    output = re.sub(r"\bnull\b", "None", text, flags=re.IGNORECASE)
    output = re.sub(r"\btrue\b", "True", output, flags=re.IGNORECASE)
    output = re.sub(r"\bfalse\b", "False", output, flags=re.IGNORECASE)
    return output


def try_parse_jsonish(text: str) -> Optional[Any]:
    source = str(text or "").strip()
    if not source:
        return None
    candidates = [source, json_strip_trailing_commas(source)]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    for candidate in candidates:
        try:
            return ast.literal_eval(candidate)
        except Exception:
            pass
    for candidate in candidates:
        try:
            return ast.literal_eval(py_literal_from_json_tokens(candidate))
        except Exception:
            pass
    return None


def normalize_struct_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if "tag" in payload or "values" in payload or "label" in payload:
            return [payload]
        for key in ("items", "data", "structured", "result", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def is_structured_boundary_line(line: str) -> bool:
    text = str(line or "")
    return bool(ENTRY_START_RE.match(text) or EVIDENCE_LINE_RE.match(text) or PROPERTY_ABSTRACT_RE.match(text))


def parse_structured_sentence(line_tag: str, sentence: str) -> Tuple[Dict[str, List[str]], bool, str]:
    line_tag_norm = normalize_tag_name(line_tag)
    payload = try_parse_jsonish(sentence)
    if payload is None:
        return {}, False, "json_parse_failed"

    items = normalize_struct_items(payload)
    if not items:
        return {}, False, "no_dict_item"

    out: Dict[str, List[str]] = {}
    item_ok = 0
    for item in items:
        tag = normalize_tag_name(fmt_scalar(item.get("tag", ""))) or line_tag_norm
        if tag not in TARGET_COLS:
            continue
        if tag in NUMERIC_COLS:
            values = format_numeric_item(tag, item)
        else:
            values = as_label_list(item.get("label", item.get("values", "")))
        values = [value for value in values if str(value).strip()]
        if not values:
            continue
        out.setdefault(tag, []).extend(values)
        item_ok += 1

    if item_ok == 0:
        return {}, False, "no_target_value"
    return out, True, ""


def parse_markdown_to_rows(md_path: str) -> Tuple[List[Dict[str, str]], int, List[ParseFail]]:
    lines = read_text(md_path).splitlines()
    rows: List[Dict[str, Any]] = []
    current_row: Optional[Dict[str, Any]] = None
    current_sample = ""
    in_structured = False
    parsed_sentence_count = 0
    fails: List[ParseFail] = []

    def ensure_row(sample_name: str) -> Dict[str, Any]:
        nonlocal current_row
        if current_row is not None and current_row.get("CDs_Naming_in_Paper", "") == sample_name:
            return current_row
        row = {key: [] for key in CSV_COLS}
        row["CDs_Naming_in_Paper"] = sample_name
        rows.append(row)
        current_row = row
        return row

    for line in lines:
        header_match = SAMPLE_HEADER_RE.match(line)
        if header_match:
            level = header_match.group(1)
            title = header_match.group(2).strip()
            if len(level) == 1 and PAPER_HEADER_RE.match(line):
                pass
            else:
                current_sample = normalize_sample_header(title)
                ensure_row(current_sample or "(NO_SAMPLE_HEADER)")
            in_structured = False
            continue

        if STRUCTURED_RE.match(line):
            in_structured = True
            continue

        if not in_structured:
            continue
        if not str(line).strip():
            continue

        match = TAG_LINE_RE.match(line)
        if not match:
            if is_structured_boundary_line(line):
                in_structured = False
                continue
            row = ensure_row(current_sample or "(NO_SAMPLE_HEADER)")
            fails.append(
                ParseFail(
                    md_path=md_path,
                    sample_name=str(row["CDs_Naming_in_Paper"]),
                    struct_line=line.strip(),
                    reason="structured_line_not_tagged",
                )
            )
            in_structured = False
            continue

        line_tag = normalize_tag_name(str(match.group("tag") or "").strip())
        sentence = str(match.group("sent") or "").strip()
        parsed_sentence_count += 1
        row = ensure_row(current_sample or "(NO_SAMPLE_HEADER)")

        try:
            parsed_map, ok, reason = parse_structured_sentence(line_tag, sentence)
        except Exception as exc:
            fails.append(
                ParseFail(
                    md_path=md_path,
                    sample_name=str(row["CDs_Naming_in_Paper"]),
                    struct_line=line.strip(),
                    reason=f"structured_parse_exception:{exc.__class__.__name__}",
                )
            )
            continue
        if not ok:
            fails.append(
                ParseFail(
                    md_path=md_path,
                    sample_name=str(row["CDs_Naming_in_Paper"]),
                    struct_line=line.strip(),
                    reason=reason,
                )
            )
            continue
        for column, values in parsed_map.items():
            if column in TARGET_COLS:
                row[column].extend(values)

    output_rows: List[Dict[str, str]] = []
    for row in rows:
        out: Dict[str, str] = {}
        for key in CSV_COLS:
            if key == "CDs_Naming_in_Paper":
                out[key] = str(row.get(key, "")).strip()
            else:
                values = [str(value).strip() for value in row.get(key, []) if str(value).strip()]
                out[key] = ";".join(values)
        output_rows.append(out)
    return output_rows, parsed_sentence_count, fails


def write_csv(rows: Sequence[Dict[str, str]], out_csv: str) -> None:
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_log_paths(output_dir: str, paper_id: str) -> Dict[str, str]:
    return {
        "io": os.path.join(output_dir, f"{paper_id}_step14.io.log"),
        "error": os.path.join(output_dir, f"{paper_id}_step14_error.trace.log"),
    }


def write_io_log(result: Step14Result, log_path: str) -> None:
    append_log(log_path, f"[{timestamp_now()}] paper={result.paper_id} status={result.status}")
    if result.input_md:
        append_log(log_path, f"input_md={relative_to_paper(result.paper_dir, result.input_md)}")
    if result.output_csv:
        append_log(log_path, f"output_csv={relative_to_paper(result.paper_dir, result.output_csv)}")
    if result.note:
        append_log(log_path, f"note={result.note}")
    append_log(log_path, "")


def process_one_paper(paper_dir: str, *, skip_existing: bool) -> Step14Result:
    paper_id = paper_id_from_dir(paper_dir)
    if paper_id is None:
        return Step14Result("", paper_dir, "SKIP_INVALID_DIR", "", "", "Directory name does not start with a paper id.")

    input_md = stage_markdown_path(paper_dir, INPUT_STAGE_DIR, paper_id, kind="main")
    output_dir = os.path.join(paper_dir, "property", OUTPUT_DIRNAME)
    output_csv = os.path.join(output_dir, f"{paper_id}.csv")
    ensure_dir(output_dir)
    log_paths = build_log_paths(output_dir, paper_id)

    if skip_existing and os.path.exists(output_csv):
        result = Step14Result(paper_id, paper_dir, "SKIP_EXISTS", input_md, output_csv, "Step 14 output already exists.")
        write_io_log(result, log_paths["io"])
        return result

    if not os.path.exists(input_md):
        result = Step14Result(paper_id, paper_dir, "SKIP_NO_INPUT_MD", input_md, output_csv, "Missing Step 13 markdown.")
        write_io_log(result, log_paths["io"])
        return result

    with open(log_paths["error"], "w", encoding="utf-8", newline="\n") as handle:
        handle.write("")

    write_ok = False
    try:
        rows, struct_count, fails = parse_markdown_to_rows(input_md)
        write_csv(rows, output_csv)
        write_ok = True
    except Exception as exc:
        append_log(log_paths["error"], f"[{timestamp_now()}] paper={paper_id} stage=EXPORT status=ERROR detail={exc!r}")
        result = Step14Result(paper_id, paper_dir, "SKIP_EXPORT_ERROR", input_md, output_csv, repr(exc))
        write_io_log(result, log_paths["io"])
        return result

    for fail in fails:
        append_log(
            log_paths["error"],
            f"[{timestamp_now()}] paper={paper_id} sample={fail.sample_name} status={fail.reason} line={fail.struct_line}",
        )

    note = f"rows={len(rows)}; structured_sent={struct_count}; failed_struct={len(fails)}"
    result = Step14Result(
        paper_id,
        paper_dir,
        "PROCESSED" if write_ok and rows else "PROCESSED_EMPTY",
        input_md,
        output_csv,
        note,
    )
    write_io_log(result, log_paths["io"])
    return result


def write_root_logs(mining_root: str, results: Sequence[Step14Result]) -> None:
    main_log_path = os.path.join(mining_root, "step14_export_property_letter_table.log")
    error_log_path = os.path.join(mining_root, "step14_export_property_letter_table_error.log")
    main_statuses = {"SKIP_EXISTS", "PROCESSED", "PROCESSED_EMPTY"}

    with open(main_log_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Step 14 export property letter table\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status not in main_statuses:
                continue
            handle.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_md:
                handle.write(f"  input_md={relative_to_root(mining_root, result.input_md)}\n")
            if result.output_csv:
                handle.write(f"  output_csv={relative_to_root(mining_root, result.output_csv)}\n")
            if result.note:
                handle.write(f"  note={result.note}\n")

    with open(error_log_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Step 14 export property letter table issues\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status in main_statuses:
                continue
            handle.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_md:
                handle.write(f"  input_md={relative_to_root(mining_root, result.input_md)}\n")
            if result.output_csv:
                handle.write(f"  output_csv={relative_to_root(mining_root, result.output_csv)}\n")
            if result.note:
                handle.write(f"  note={result.note}\n")


def process_all_papers(mining_root: str, paper_ids: Optional[Sequence[str]] = None, *, skip_existing: bool = True) -> None:
    root = ensure_root_exists(mining_root)
    results: List[Step14Result] = []
    for paper_dir in tqdm(iter_paper_dirs(root, paper_ids=paper_ids), desc="Step14: export-letter-table"):
        try:
            results.append(process_one_paper(paper_dir, skip_existing=skip_existing))
        except Exception as exc:
            paper_id = paper_id_from_dir(paper_dir) or ""
            results.append(
                Step14Result(
                    paper_id,
                    paper_dir,
                    "SKIP_FATAL",
                    stage_markdown_path(paper_dir, INPUT_STAGE_DIR, paper_id, kind="main") if paper_id else "",
                    os.path.join(paper_dir, "property", OUTPUT_DIRNAME, f"{paper_id}.csv") if paper_id else "",
                    repr(exc),
                )
            )
    write_root_logs(root, results)

    processed = sum(1 for result in results if result.status == "PROCESSED")
    processed_empty = sum(1 for result in results if result.status == "PROCESSED_EMPTY")
    skipped_existing = sum(1 for result in results if result.status == "SKIP_EXISTS")
    issues = sum(1 for result in results if result.status not in {"PROCESSED", "PROCESSED_EMPTY", "SKIP_EXISTS"})
    print(
        "[DONE] papers=%d processed=%d processed_empty=%d skipped_existing=%d issues=%d"
        % (len(results), processed, processed_empty, skipped_existing, issues)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Property preprocessing Step 14: export the property letter-table CSV.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Re-run even if Step 14 output already exists.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_all_papers(
        mining_root=args.root,
        paper_ids=args.paper_ids,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
