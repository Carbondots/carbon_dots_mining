#!/usr/bin/env python3
"""Shared helpers for the clean synthesis-mining pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_LLM_MODEL_NAME = "qwen.qwen2.5-vl-32b-instruct"

SYNTHESIS_TABLE_COLUMNS = [
    "CDs_Naming_in_Paper",
    "Synthesis_Method",
    "Temperature",
    "Time",
    "Microwave_Power",
    "Precursor",
    "Precursor_Amount",
    "Solvent",
    "Solvent_Volume",
    "Purification",
]

QUANTITY_COLUMN_PAIRS = [
    ("Precursor_Amount", "Precursor"),
    ("Solvent_Volume", "Solvent"),
]
QUANTITY_VALUE_COLUMNS = {column for column, _ in QUANTITY_COLUMN_PAIRS}

__all__ = [
    "DEFAULT_LLM_MODEL_NAME",
    "QUANTITY_COLUMN_PAIRS",
    "QUANTITY_VALUE_COLUMNS",
    "SYNTHESIS_TABLE_COLUMNS",
    "append_text",
    "build_cd_description_from_row",
    "collect_yes_paragraph_texts",
    "display_path",
    "ensure_directory",
    "extract_first_markdown_table",
    "extract_document_id",
    "list_document_dirs",
    "load_decision_context_text",
    "load_document_context_text",
    "parse_json_object_text",
    "parse_markdown_table",
    "read_csv_safely",
    "replace_path_subfolder_and_suffix",
    "split_top_level_items",
    "strip_code_fence_block",
    "strip_think_block",
]


def extract_document_id(document_dir: Path) -> str:
    return document_dir.name.split("_")[0]


def list_document_dirs(pipeline_root: Path, start_from_id: str = "") -> list[Path]:
    document_dirs: list[Path] = []

    for entry in sorted(pipeline_root.iterdir(), key=lambda path: path.name):
        if not entry.is_dir():
            continue

        document_id = extract_document_id(entry)
        if not document_id.isdigit():
            continue

        if start_from_id and _document_id_before(document_id, start_from_id):
            continue

        document_dirs.append(entry)

    return document_dirs


def _document_id_before(document_id: str, start_from_id: str) -> bool:
    if document_id.isdigit() and start_from_id.isdigit():
        return int(document_id) < int(start_from_id)
    return document_id < start_from_id


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_text(log_path: Path, text: str) -> None:
    ensure_directory(log_path.parent)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write((text or "").rstrip() + "\n")


def display_path(path: Path, base_path: Path | None = None) -> str:
    target = Path(path)
    if not target.is_absolute():
        return target.as_posix()

    candidates = []
    if base_path is not None:
        candidates.append(Path(base_path))
    candidates.append(Path.cwd())

    resolved_target = target.resolve()
    for candidate in candidates:
        try:
            return resolved_target.relative_to(candidate.resolve()).as_posix()
        except Exception:
            continue

    return target.name or "<path>"


def strip_think_block(raw_text: str) -> str:
    raw_text = raw_text or ""
    if "</think>" in raw_text:
        return raw_text.split("</think>")[-1].strip()
    return raw_text.strip()


def strip_code_fence_block(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned

    cleaned = re.sub(r"^```.*?\n", "", cleaned, count=1, flags=re.DOTALL).rstrip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].rstrip()
    return cleaned


def _extract_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def parse_json_object_text(text: str) -> dict[str, Any] | None:
    cleaned = strip_code_fence_block(strip_think_block(text))
    if not cleaned:
        return None

    candidates = [cleaned, _extract_balanced_json_object(cleaned)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def split_top_level_items(value: str, separators: str = ",;，；") -> list[str]:
    text = (value or "").strip()
    if not text:
        return []

    parts: list[str] = []
    buffer: list[str] = []
    depth = 0
    opening_pairs = {"(": ")", "[": "]", "（": "）"}
    closing_chars = set(opening_pairs.values())

    for char in text:
        if char in opening_pairs:
            depth += 1
            buffer.append(char)
        elif char in closing_chars:
            depth = max(0, depth - 1)
            buffer.append(char)
        elif char in separators and depth == 0:
            item = "".join(buffer).strip()
            if item:
                parts.append(item)
            buffer = []
        else:
            buffer.append(char)

    tail = "".join(buffer).strip()
    if tail:
        parts.append(tail)

    return parts


def replace_path_subfolder_and_suffix(
    base_path: Path,
    source_subfolder: str,
    target_subfolder: str,
    new_suffix: str,
) -> Path:
    replaced = str(base_path).replace(source_subfolder, target_subfolder)
    return Path(replaced).with_suffix(new_suffix)


def read_csv_safely(csv_path: Path):
    import pandas as pd

    try:
        return pd.read_csv(csv_path, keep_default_na=False)
    except Exception:
        return None


def collect_yes_paragraph_texts(decision_df) -> list[str]:
    required_columns = {"LLM_decision", "text"}
    if not required_columns.issubset(getattr(decision_df, "columns", [])):
        return []

    yes_df = decision_df[decision_df["LLM_decision"] == "YES"].copy()
    if "para_id" in yes_df.columns:
        yes_df = yes_df.sort_values("para_id")

    return [str(text).strip() for text in yes_df["text"].tolist() if str(text).strip()]


def read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except Exception:
        return ""


def load_decision_context_text(decision_csv_path: Path) -> str:
    decision_df = read_csv_safely(decision_csv_path)
    if decision_df is None:
        return ""
    return "\n".join(collect_yes_paragraph_texts(decision_df)).strip()


def _cell_ok(value: Any, column: str, exclude_columns: set[str]) -> bool:
    if column in exclude_columns or value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.upper() != "N/A"


def build_cd_description_from_row(
    row: Mapping[str, Any],
    exclude_columns: set[str] | None = None,
) -> str:
    exclude_columns = set(exclude_columns or ())

    name = row.get("CDs_Naming_in_Paper", "")
    method = row.get("Synthesis_Method", "")
    precursor = row.get("Precursor", "")
    precursor_amount = row.get("Precursor_Amount", "")
    solvent = row.get("Solvent", "")
    solvent_volume = row.get("Solvent_Volume", "")
    temperature = row.get("Temperature", "")
    time_value = row.get("Time", "")

    if _cell_ok(name, "CDs_Naming_in_Paper", exclude_columns):
        parts = [f"{name} was synthesized"]
    else:
        parts = ["The carbon dots were synthesized"]

    if _cell_ok(method, "Synthesis_Method", exclude_columns):
        parts.append(f"via {method}")

    chemistry_bits: list[str] = []
    if _cell_ok(precursor, "Precursor", exclude_columns):
        chemistry_bits.append(f"using {precursor}")
    if _cell_ok(precursor_amount, "Precursor_Amount", exclude_columns):
        chemistry_bits.append(f"with precursor amount {precursor_amount}")
    if _cell_ok(solvent, "Solvent", exclude_columns):
        chemistry_bits.append(f"in {solvent}")
    if _cell_ok(solvent_volume, "Solvent_Volume", exclude_columns):
        chemistry_bits.append(f"solvent volume {solvent_volume}")
    if chemistry_bits:
        parts.append(", ".join(chemistry_bits))

    condition_bits: list[str] = []
    if _cell_ok(temperature, "Temperature", exclude_columns):
        condition_bits.append(f"at {temperature}")
    if _cell_ok(time_value, "Time", exclude_columns):
        condition_bits.append(f"for {time_value}")
    if condition_bits:
        parts.append(" ".join(condition_bits))

    sentence = " ".join(part.strip() for part in parts if part.strip())
    return sentence + ("" if sentence.endswith(".") else ".")


def extract_first_markdown_table(md_text: str) -> str | None:
    blocks: list[str] = []
    current: list[str] = []
    for line in (md_text or "").splitlines():
        if line.strip().startswith("|"):
            current.append(line.rstrip())
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks[0] if blocks else None


def parse_markdown_table(
    md_text: str,
    expected_columns: Sequence[str] = SYNTHESIS_TABLE_COLUMNS,
):
    import pandas as pd

    table = extract_first_markdown_table(md_text)
    if not table:
        return None

    lines = [line.strip() for line in table.splitlines() if line.strip()]
    rows = [line for line in lines if not re.match(r"^\|\s*[:-]{2,}.*\|$", line)]
    if len(rows) < 2:
        return None

    headers = [header.strip() for header in rows[0].strip("|").split("|")]
    data_rows: list[list[str]] = []
    for line in rows[1:]:
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) < len(headers):
            columns += [""] * (len(headers) - len(columns))
        data_rows.append(columns[: len(headers)])

    frame = pd.DataFrame(data_rows, columns=headers)
    for column in expected_columns:
        if column not in frame.columns:
            frame[column] = "N/A"
    return frame[list(expected_columns)].replace(r"^\s*$", "N/A", regex=True)


def load_document_context_text(document_dir: Path) -> str:
    document_id = extract_document_id(document_dir)
    parts: list[str] = []
    seen: set[str] = set()

    candidate_paths = [
        document_dir / "preprocess" / "cut" / f"{document_id}_cut.md",
        document_dir / "Synthesis" / "cos_tokenized" / f"{document_id}.md",
        document_dir / "Synthesis" / "LLM_abstract_qwen2.5vl" / f"{document_id}.md",
        document_dir / "Synthesis" / "LLM_name_qwen2.5vl" / f"{document_id}.md",
    ]

    for path in candidate_paths:
        text = read_text_if_exists(path).strip()
        if text and text not in seen:
            seen.add(text)
            parts.append(text)

    decision_csv = document_dir / "Synthesis" / "LLM_decision_32b" / f"{document_id}.csv"
    yes_text = load_decision_context_text(decision_csv)
    if yes_text and yes_text not in seen:
        parts.append(yes_text)

    return "\n\n".join(parts)
