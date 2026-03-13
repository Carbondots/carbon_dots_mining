#!/usr/bin/env python3

"""Step 13: resolve Step 12 change entries and write sanitized final property markdown."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from property_unit import (
    append_log_line as append_log,
    ensure_dir,
    ensure_root_exists,
    iter_paper_dirs,
    paper_id_from_dir,
    read_letter_table_samples,
    read_text,
    relative_to_paper,
    relative_to_root,
    remove_think_blocks,
    stage_markdown_path,
    strip_code_fences,
    timestamp_now,
    write_text,
)

try:
    from lmstudio import llm as lmstudio_llm
except Exception:
    lmstudio_llm = None


INPUT_STAGE_DIR = "conflict_resolved_properties"
OUTPUT_STAGE_DIR = "change_resolved_properties"
DEFAULT_MODEL = "qwen.qwen2.5-vl-32b-instruct"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1800
DEFAULT_RETRIES = 5
DEFAULT_VOTES = 3
END_SENTINEL = "<END_OF_JSON>"

CHANGE_TAGS = {"QY", "lifetime"}
KNOWN_SUFFIXES = {"(APP)", "(VS)", "(VS-refined)", "(VS-REFINED)", "(MAIN)"}
QY_PERCENT_RE = re.compile(r"(?i)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:%|percent(?:age)?)")
LIFETIME_VALUE_RE = re.compile("(?i)\\b\\d+(?:\\.\\d+)?\\s*(fs|ps|ns|us|\u03bcs|ms|s)\\b")
LIFETIME_UNIT_RE = re.compile("(?i)\\b(fs|ps|ns|us|\u03bcs|microseconds?|nanoseconds?|picoseconds?|milliseconds?|seconds?|s)\\b")
SAMPLE_HEADER_RE = re.compile(r"^(#+)\s+(.*\S)\s*$")
PAPER_HEADER_RE = re.compile(r"^#\s*Paper\b", re.IGNORECASE)
ENTRY_CHANGE_START_RE = re.compile(r"^\s*(\d+)\.\s+\[change\]\s*$", re.IGNORECASE)
ENTRY_PARA_START_RE = re.compile(r"^\s*(\d+)\.\s+\[\s*para\s*=.+\]\s*$", re.IGNORECASE)
PROPERTY_ABSTRACT_RE = re.compile(r"^\s*Property abstract\s*:\s*$", re.IGNORECASE)
STRUCTURED_RE = re.compile(r"^\s*Structured\s*:\s*$", re.IGNORECASE)
TAG_LINE_RE = re.compile(
    r"^\s*(?P<prefix>\[(?P<prefix_inner>[^\]]+)\])?\s*(?P<tag>[A-Za-z][A-Za-z0-9_]*)\s*:\s*(?P<sent>.+?)\s*$"
)
ENTRY_HEAD_META_RE = re.compile(
    r"\[\s*para\s*=\s*(?P<para>\d*)\s*;\s*(?:win|window)\s*=\s*(?P<win>[^;\]]*)\s*;\s*sids\s*=\s*(?P<sids>[^\]]*)\s*\]",
    re.IGNORECASE,
)
ENTRY_NUM_PREFIX_RE = re.compile(r"^\s*\d+\.\s+")
CHANGE_MARK_RE = re.compile(r"^\s*\[change\]\s*$", re.IGNORECASE)
CHANGE_META_LINE_RE = re.compile(r"^\s*([AB])\.\s*(\[[^\]]*\])\s*$", re.IGNORECASE)
EVIDENCE_LINE_RE = re.compile(r"^\s*evidence\s*:\s*$", re.IGNORECASE)
INT_RE = re.compile(r"^\s*-?\d+\s*$")
CONDITION_PATTERNS = {
    "Em": re.compile(r"(?i)\bEm\s*[=:]\s*(-?\d+(?:\.\d+)?)\s*nm\b"),
    "Ex": re.compile(r"(?i)\bEx\s*[=:]\s*(-?\d+(?:\.\d+)?)\s*nm\b"),
    "pH": re.compile(r"(?i)\bpH\s*[=:]\s*(-?\d+(?:\.\d+)?)\b"),
    "solvent": re.compile(r"(?i)\bsolvent\s*[=:]\s*([^\)\]\n;,.]+)"),
}


@dataclass
class TagLine:
    entry_idx: int
    order_in_entry: int
    prefix: str
    tag_raw: str
    tag: str
    sentence: str
    keep: bool = True
    structured: Optional[Any] = None
    structured_raw: str = ""


@dataclass
class Entry:
    entry_idx: int
    preprop_lines: List[str] = field(default_factory=list)
    prop_header_line: str = "Property abstract:"
    tag_lines: List[TagLine] = field(default_factory=list)
    tail_lines: List[str] = field(default_factory=list)
    suppress_structured: bool = False
    force_structured: bool = False


@dataclass
class SampleBlock:
    header_line: str
    sample_name: str
    entries: List[Entry] = field(default_factory=list)


@dataclass
class ChangePair:
    tag: str
    tag_a: TagLine
    tag_b: TagLine
    has_struct_a: bool
    has_struct_b: bool


@dataclass
class Step13Result:
    paper_id: str
    paper_dir: str
    status: str
    input_md: str
    output_merge_md: str
    output_struct_md: str
    output_main_md: str
    note: str = ""


def normalize_tag_name(tag: str) -> str:
    text = str(tag or "").strip()
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text).upper()
    mapping = {
        "EX": "Ex",
        "EM": "Em",
        "QY": "QY",
        "LIFETIME": "lifetime",
        "EXDEP": "ExDep",
        "CHIRAL": "Chiral",
        "CPL": "CPL",
    }
    return mapping.get(compact, text)


def normalize_sample_header(text: str) -> str:
    sample = str(text or "").strip()
    lowered = sample.lower()
    if lowered.startswith("step "):
        return ""
    while True:
        trimmed = sample.rstrip()
        removed = False
        for suffix in KNOWN_SUFFIXES:
            if trimmed.endswith(suffix):
                sample = trimmed[: -len(suffix)].rstrip()
                removed = True
                break
        if not removed:
            break
    return re.sub(r"\s+", " ", sample).strip()


def normalize_model_output(raw: Any) -> str:
    return strip_code_fences(remove_think_blocks(raw)).strip()


def strip_entry_number_prefix(line: str) -> str:
    return ENTRY_NUM_PREFIX_RE.sub("", str(line or ""), count=1)


def parse_entry_head_meta(value: Any) -> Dict[str, str]:
    if isinstance(value, (list, tuple)):
        blob = "\n".join(str(item or "") for item in value)
    else:
        blob = str(value or "")
    match = ENTRY_HEAD_META_RE.search(blob)
    if not match:
        return {"para": "", "win": "", "sids": ""}
    return {
        "para": str(match.group("para") or "").strip(),
        "win": str(match.group("win") or "").strip(),
        "sids": re.sub(r"\s+", "", str(match.group("sids") or "").strip()),
    }


def format_meta(para: str, win: str, sids: str) -> str:
    return f"[para={para}; window={win}; sids={sids}]"


def parse_int_maybe(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if text and INT_RE.match(text):
        try:
            return int(text)
        except Exception:
            return None
    return None


def parse_sids_text(raw: str) -> List[int]:
    return [int(value) for value in re.split(r"[,\s]+", str(raw or "")) if value.strip().isdigit()]


def match_entry_start(line: str) -> Optional[re.Match[str]]:
    text = str(line or "")
    return ENTRY_CHANGE_START_RE.match(text) or ENTRY_PARA_START_RE.match(text)


def to_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value).strip())
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


def unique_text(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = str(value or "")
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def unique_float(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    seen: Set[str] = set()
    for value in values:
        key = "%0.12g" % float(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(value))
    return out


def normalize_sentence_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def first_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        return ""
    parts = [part.strip() for part in re.split(r"(?<=[\.\!\?])\s+", cleaned) if part.strip()]
    return parts[0] if parts else cleaned


def one_line_json_text(value: Any) -> str:
    try:
        return json.dumps("" if value is None else str(value), ensure_ascii=False)
    except Exception:
        return repr(value)


def parse_num_list_any(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, list):
        out = [to_float(item) for item in value]
        return [float(item) for item in out if item is not None]
    numeric = to_float(value)
    return [float(numeric)] if numeric is not None else []


def normalize_qy_percent_values(values: Sequence[float], refine_sentence: str, evidence_text: str) -> List[float]:
    if "%" in str(refine_sentence or ""):
        return [float(value) for value in values]
    evidence_percents: List[float] = []
    for match in QY_PERCENT_RE.finditer(str(evidence_text or "")):
        try:
            evidence_percents.append(float(match.group(1)))
        except Exception:
            continue
    out: List[float] = []
    for value in values:
        current = float(value)
        if current >= 1:
            out.append(current)
            continue
        if any(abs(evidence - current) <= 1e-6 for evidence in evidence_percents):
            out.append(current)
            continue
        scaled = current * 100.0
        if any(abs(evidence - scaled) <= 1e-4 for evidence in evidence_percents):
            out.append(scaled)
            continue
        out.append(scaled)
    return out


def condition_value_key(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "%0.12g" % float(value)
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def prefix_label(prefix: str) -> str:
    text = str(prefix or "").strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip().upper()
        if inner in {"A", "B"}:
            return inner
    return ""


def entry_has_change_marker(preprop_lines: Sequence[str]) -> bool:
    for line in preprop_lines:
        stripped = strip_entry_number_prefix(str(line)).strip()
        if not stripped:
            continue
        if CHANGE_MARK_RE.match(stripped) or CHANGE_META_LINE_RE.match(stripped):
            return True
    return False


def json_strip_trailing_commas(text: str) -> str:
    current = str(text or "")
    previous = None
    while previous != current:
        previous = current
        current = re.sub(r",\s*([}\]])", r"\1", current)
    return current


def try_parse_jsonish(text: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    meta = {"ok": False, "mode": "", "error": ""}
    if not isinstance(text, str):
        meta["error"] = "NOT_STR"
        return None, meta
    cleaned = text.strip()
    if not cleaned:
        meta["error"] = "EMPTY"
        return None, meta
    for parser, mode in (
        (json.loads, "json.loads"),
        (lambda value: json.loads(json_strip_trailing_commas(value)), "json.loads+trailing_comma_fix"),
        (ast.literal_eval, "ast.literal_eval"),
    ):
        try:
            return parser(cleaned), {"ok": True, "mode": mode, "error": ""}
        except Exception:
            continue
    meta["error"] = "PARSE_FAIL"
    return None, meta


def extract_json_fragments(text: str) -> List[str]:
    source = str(text or "").strip()
    if not source:
        return []

    def balanced_from(start: int) -> Optional[str]:
        stack: List[str] = []
        in_string = False
        escaped = False
        for index in range(start, len(source)):
            char = source[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack:
                    continue
                top = stack[-1]
                if (top == "{" and char == "}") or (top == "[" and char == "]"):
                    stack.pop()
                    if not stack:
                        return source[start : index + 1]
                else:
                    return None
        return None

    out: List[str] = []
    for opener in ("{", "["):
        start = source.find(opener)
        if start >= 0:
            fragment = balanced_from(start)
            if fragment and fragment not in out:
                out.append(fragment)
    return out


def extract_llm_body_before_end_mark(
    raw: Any,
    *,
    end_mark: str = END_SENTINEL,
    allow_missing_end_mark: bool = False,
) -> Tuple[Optional[str], Dict[str, Any]]:
    meta = {"ok": False, "reason": "", "cleaned": ""}
    cleaned = normalize_model_output(raw)
    meta["cleaned"] = cleaned
    if not cleaned:
        meta["reason"] = "EMPTY"
        return None, meta
    position = cleaned.find(end_mark)
    if position >= 0:
        body = cleaned[:position].strip()
    elif allow_missing_end_mark:
        body = cleaned
    else:
        meta["reason"] = "NO_END_MARK"
        return None, meta
    if not body:
        meta["reason"] = "EMPTY_BODY"
        return None, meta
    meta["ok"] = True
    meta["reason"] = "OK"
    return body, meta


STRUCTURED_ALLOWED_KEYS = {"tag", "values", "unit", "vary_by", "vary_values", "label"}


def keep_supported_structured_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key).strip()
            if key_text not in STRUCTURED_ALLOWED_KEYS:
                continue
            out[key_text] = keep_supported_structured_fields(child)
        return out
    if isinstance(value, list):
        return [keep_supported_structured_fields(child) for child in value]
    return value


def sanitize_structured_payload(tag: str, structured: Any, structured_raw: str) -> Tuple[Optional[Any], str]:
    payload = structured
    if payload is None and str(structured_raw or "").strip():
        payload, _ = try_parse_jsonish(str(structured_raw))
    if payload is None:
        return None, str(structured_raw or "")
    payload = keep_supported_structured_fields(payload)
    return payload, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def split_entry(entry_lines: Sequence[str], entry_idx_hint: Optional[int]) -> Optional[Entry]:
    if not entry_lines or not any(str(line or "").strip() for line in entry_lines):
        return None

    prop_index = None
    for index, line in enumerate(entry_lines):
        if PROPERTY_ABSTRACT_RE.match(str(line)):
            prop_index = index
            break

    entry_idx = entry_idx_hint if entry_idx_hint is not None else 0
    if prop_index is None:
        return Entry(entry_idx=entry_idx, preprop_lines=list(entry_lines), tag_lines=[], tail_lines=[])

    preprop_lines = list(entry_lines[:prop_index])
    prop_header_line = str(entry_lines[prop_index])
    after = list(entry_lines[prop_index + 1 :])

    struct_index = None
    for index, line in enumerate(after):
        if STRUCTURED_RE.match(str(line)):
            struct_index = index
            break
    property_lines = after if struct_index is None else after[:struct_index]
    structured_lines = [] if struct_index is None else after[struct_index + 1 :]

    tag_lines: List[TagLine] = []
    for order, line in enumerate(property_lines):
        match = TAG_LINE_RE.match(str(line))
        if not match:
            continue
        tag_raw = str(match.group("tag") or "").strip()
        tag_lines.append(
            TagLine(
                entry_idx=entry_idx,
                order_in_entry=order,
                prefix=str(match.group("prefix") or "").strip(),
                tag_raw=tag_raw,
                tag=normalize_tag_name(tag_raw),
                sentence=str(match.group("sent") or "").strip(),
            )
        )

    parsed_struct_rows: List[Tuple[str, str, Optional[Any], str]] = []
    tail_lines: List[str] = []
    for line in structured_lines:
        match = TAG_LINE_RE.match(str(line))
        if not match:
            if str(line).strip():
                tail_lines.append(str(line))
            continue
        prefix = str(match.group("prefix") or "").strip()
        tag = normalize_tag_name(str(match.group("tag") or "").strip())
        payload_raw = str(match.group("sent") or "").strip()
        payload, _ = try_parse_jsonish(payload_raw)
        payload, payload_text = sanitize_structured_payload(tag, payload, payload_raw)
        parsed_struct_rows.append((prefix, tag, payload, payload_text or payload_raw))

    used = [False] * len(parsed_struct_rows)
    for tag_line in tag_lines:
        hit = -1
        for index, row in enumerate(parsed_struct_rows):
            if used[index]:
                continue
            prefix, tag, _, _ = row
            if prefix == tag_line.prefix and tag == tag_line.tag:
                hit = index
                break
        if hit == -1:
            for index, row in enumerate(parsed_struct_rows):
                if used[index]:
                    continue
                _, tag, _, _ = row
                if tag == tag_line.tag:
                    hit = index
                    break
        if hit >= 0:
            used[hit] = True
            tag_line.structured = parsed_struct_rows[hit][2]
            tag_line.structured_raw = parsed_struct_rows[hit][3]

    return Entry(
        entry_idx=entry_idx,
        preprop_lines=preprop_lines,
        prop_header_line=prop_header_line,
        tag_lines=tag_lines,
        tail_lines=tail_lines,
    )


def parse_markdown(path: str) -> Tuple[List[str], List[SampleBlock]]:
    lines = read_text(path).splitlines()
    paper_header: List[str] = []
    samples: List[SampleBlock] = []
    current_sample: Optional[SampleBlock] = None
    current_entry_lines: List[str] = []
    current_entry_idx_hint: Optional[int] = None
    pending_pre_entry: List[str] = []
    in_entry = False

    def flush_current_entry() -> None:
        nonlocal current_entry_lines, current_entry_idx_hint, pending_pre_entry, in_entry
        if current_sample is None:
            current_entry_lines = []
            current_entry_idx_hint = None
            pending_pre_entry = []
            in_entry = False
            return
        entry = split_entry(current_entry_lines, current_entry_idx_hint)
        if entry is not None:
            current_sample.entries.append(entry)
        current_entry_lines = []
        current_entry_idx_hint = None
        pending_pre_entry = []
        in_entry = False

    for line in lines:
        header_match = SAMPLE_HEADER_RE.match(str(line))
        if header_match:
            level = header_match.group(1)
            title = header_match.group(2).strip()
            if len(level) == 1 and PAPER_HEADER_RE.match(str(line)):
                if current_sample is None and not in_entry:
                    paper_header.append(str(line))
                elif in_entry:
                    current_entry_lines.append(str(line))
                else:
                    pending_pre_entry.append(str(line))
                continue

            sample_name = normalize_sample_header(title)
            if not sample_name:
                if current_sample is None and not in_entry:
                    paper_header.append(str(line))
                elif in_entry:
                    current_entry_lines.append(str(line))
                else:
                    pending_pre_entry.append(str(line))
                continue

            if current_sample is not None and in_entry:
                flush_current_entry()

            current_sample = SampleBlock(header_line=str(line), sample_name=sample_name, entries=[])
            samples.append(current_sample)
            current_entry_lines = []
            current_entry_idx_hint = None
            pending_pre_entry = []
            in_entry = False
            continue

        if current_sample is None:
            paper_header.append(str(line))
            continue

        entry_match = match_entry_start(str(line))
        if entry_match:
            if in_entry:
                flush_current_entry()
            try:
                current_entry_idx_hint = int(entry_match.group(1))
            except Exception:
                current_entry_idx_hint = None
            current_entry_lines = pending_pre_entry + [str(line)]
            pending_pre_entry = []
            in_entry = True
            continue

        if in_entry:
            current_entry_lines.append(str(line))
        elif str(line).strip():
            pending_pre_entry.append(str(line))

    if current_sample is not None and in_entry:
        flush_current_entry()
    return paper_header, samples


def apply_entry_number_to_preprop(preprop_lines: Sequence[str], out_entry_no: int) -> List[str]:
    if not preprop_lines:
        return [f"{out_entry_no}."]
    out = list(preprop_lines)
    head_index = None
    for index, line in enumerate(out):
        if str(line).strip():
            head_index = index
            break
    if head_index is None:
        return [f"{out_entry_no}."]
    out[head_index] = f"{out_entry_no}. {strip_entry_number_prefix(out[head_index])}".rstrip()
    return out


def structured_to_line(prefix: str, tag: str, structured: Any, structured_raw: str = "") -> str:
    left = f"{prefix}{tag}" if prefix else tag
    if structured is not None:
        payload = json.dumps(keep_supported_structured_fields(structured), ensure_ascii=False, separators=(",", ":"))
    else:
        payload = str(structured_raw or "").strip() or "{}"
    return f"{left}: {payload}"


def should_write_structured(entry: Entry) -> bool:
    if entry.suppress_structured:
        return False
    if entry.force_structured:
        return True
    for tag_line in entry.tag_lines:
        if not tag_line.keep:
            continue
        if tag_line.structured is not None or str(tag_line.structured_raw or "").strip():
            return True
    return False


def build_entry_lines(entry: Entry, out_entry_no: int) -> List[str]:
    out = apply_entry_number_to_preprop(entry.preprop_lines, out_entry_no)
    kept = [line for line in sorted(entry.tag_lines, key=lambda item: item.order_in_entry) if line.keep]
    if not kept:
        return out
    if out and out[-1].strip():
        out.append("")
    out.append(entry.prop_header_line)
    for tag_line in kept:
        left = f"{tag_line.prefix}{normalize_tag_name(tag_line.tag)}" if tag_line.prefix else normalize_tag_name(tag_line.tag)
        out.append(f"{left}: {tag_line.sentence.strip()}")
    if should_write_structured(entry):
        out.append("")
        out.append("Structured:")
        for tag_line in kept:
            structured, structured_text = sanitize_structured_payload(tag_line.tag, tag_line.structured, tag_line.structured_raw)
            tag_line.structured = structured
            tag_line.structured_raw = structured_text
            out.append(
                structured_to_line(
                    tag_line.prefix.strip(),
                    normalize_tag_name(tag_line.tag),
                    tag_line.structured,
                    tag_line.structured_raw,
                )
            )
    if entry.tail_lines:
        out.append("")
        out.extend(entry.tail_lines)
    return out


def write_markdown(out_path: str, paper_header: Sequence[str], samples: Sequence[SampleBlock]) -> None:
    lines: List[str] = list(paper_header)
    if lines and lines[-1].strip():
        lines.append("")
    for sample in samples:
        lines.append(sample.header_line)
        lines.append("")
        for entry_no, entry in enumerate(sample.entries, start=1):
            lines.extend(build_entry_lines(entry, entry_no))
            lines.append("")
            lines.append("")
    while lines and not lines[-1].strip():
        lines.pop()
    write_text(out_path, "\n".join(lines) + ("\n" if lines else ""))


def inspect_change_pair(entry: Entry) -> Tuple[Optional[ChangePair], str]:
    if not entry_has_change_marker(entry.preprop_lines):
        return None, "NO_CHANGE_MARKER"
    by_letter: Dict[str, TagLine] = {}
    for tag_line in entry.tag_lines:
        letter = prefix_label(tag_line.prefix)
        if letter in {"A", "B"}:
            by_letter[letter] = tag_line
    if "A" not in by_letter or "B" not in by_letter:
        return None, "MISSING_A_OR_B"
    tag_a = by_letter["A"]
    tag_b = by_letter["B"]
    norm_a = normalize_tag_name(tag_a.tag)
    norm_b = normalize_tag_name(tag_b.tag)
    if norm_a != norm_b or norm_a not in CHANGE_TAGS:
        return None, "UNSUPPORTED_TAG_OR_MISMATCH"
    has_struct_a = tag_a.structured is not None or bool(str(tag_a.structured_raw or "").strip())
    has_struct_b = tag_b.structured is not None or bool(str(tag_b.structured_raw or "").strip())
    return ChangePair(norm_a, tag_a, tag_b, has_struct_a, has_struct_b), "OK"


def parse_change_evidence_blocks(preprop_lines: Sequence[str]) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    index = 0
    total = len(preprop_lines)
    while index < total:
        match = CHANGE_META_LINE_RE.match(str(preprop_lines[index] or ""))
        if not match:
            index += 1
            continue
        letter = match.group(1).upper()
        meta = str(match.group(2) or "").strip()
        entry_meta = parse_entry_head_meta(meta)
        index += 1
        if index < total and EVIDENCE_LINE_RE.match(str(preprop_lines[index] or "")):
            index += 1
        evidence_lines: List[str] = []
        while index < total and not CHANGE_META_LINE_RE.match(str(preprop_lines[index] or "")):
            evidence_lines.append(str(preprop_lines[index] or ""))
            index += 1
        blocks.append(
            {
                "letter": letter,
                "meta": meta or format_meta(entry_meta.get("para", ""), entry_meta.get("win", ""), entry_meta.get("sids", "")),
                "text": "\n".join(evidence_lines).strip(),
            }
        )
    return blocks


def extract_values_unit(structured: Any, structured_raw: str) -> Tuple[List[float], str]:
    payload = structured
    if payload is None and str(structured_raw or "").strip():
        payload, _ = try_parse_jsonish(str(structured_raw))
    values: List[float] = []
    unit = ""

    def pull(item: Dict[str, Any]) -> None:
        nonlocal unit
        current_unit = str(item.get("unit", "")).strip()
        if current_unit and not unit:
            unit = current_unit
        values.extend(parse_num_list_any(item.get("values", item.get("value", None))))

    if isinstance(payload, dict):
        pull(payload)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                pull(item)
    return unique_float(values), unit


def value_text_from_tag_line(tag_line: TagLine) -> Tuple[str, str, Optional[float]]:
    values, unit = extract_values_unit(tag_line.structured, tag_line.structured_raw)
    if not values:
        return "unknown", unit, None
    first_value = float(values[0])
    return attach_unit(format_number(first_value), unit), unit, first_value


def normalize_lifetime_unit_token(unit: str) -> str:
    mapping = {
        "μs": "us",
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
    final_unit = mapping.get(str(unit or "").strip().lower(), str(unit or "").strip().lower())
    return final_unit if final_unit in {"fs", "ps", "ns", "us", "ms", "s"} else ""


def choose_lifetime_unit(refine_sentence: str, evidence_text: str, model_unit: str) -> str:
    counts: Dict[str, int] = {}
    first_pos: Dict[str, int] = {}
    for pattern_text in (str(evidence_text or ""), str(refine_sentence or "")):
        for match in LIFETIME_VALUE_RE.finditer(pattern_text):
            unit = normalize_lifetime_unit_token(match.group(1))
            if not unit:
                continue
            counts[unit] = counts.get(unit, 0) + 1
            first_pos.setdefault(unit, len(first_pos))
    if counts:
        return sorted(counts.items(), key=lambda item: (-item[1], first_pos.get(item[0], 10**9)))[0][0]
    for match in LIFETIME_UNIT_RE.finditer(str(model_unit or "")):
        unit = normalize_lifetime_unit_token(match.group(1))
        if unit:
            return unit
    return normalize_lifetime_unit_token(model_unit)


def normalize_condition_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def extract_explicit_condition_values(text: str) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {"Em": [], "Ex": [], "pH": [], "solvent": []}
    source = str(text or "")
    for label, pattern in CONDITION_PATTERNS.items():
        for match in pattern.finditer(source):
            raw = str(match.group(1) or "").strip()
            if not raw:
                continue
            if label in {"Em", "Ex", "pH"}:
                numeric = to_float(raw)
                if numeric is None:
                    continue
                out[label].append(float(numeric))
            else:
                out[label].append(normalize_condition_text(raw))
    return {
        key: unique_float(values) if key in {"Em", "Ex", "pH"} else unique_text(values)
        for key, values in out.items()
    }


def pick_merge_condition_axis(sentence: str) -> Tuple[str, List[Any]]:
    condition_map = extract_explicit_condition_values(sentence)
    qualified: List[Tuple[str, List[Any]]] = []
    for label in ("Em", "Ex", "pH", "solvent"):
        values = condition_map.get(label, [])
        unique_values: List[Any] = []
        seen: Set[str] = set()
        for value in values:
            key = condition_value_key(value)
            if key in seen:
                continue
            seen.add(key)
            unique_values.append(value)
        if len(unique_values) >= 2:
            qualified.append((label, unique_values))
    if len(qualified) != 1:
        return "", []
    return qualified[0]


def normalize_vary_by(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").strip().lower())
    if text in {"em", "emission", "emissionwavelength"}:
        return "Em"
    if text in {"ex", "excitation", "excitationwavelength"}:
        return "Ex"
    if text == "ph":
        return "pH"
    if text in {"solvent", "medium", "state", "phase"}:
        return "solvent"
    if text in {"component", "components", "tau"}:
        return "component"
    return str(value or "").strip()


def normalize_vary_value(label: str, value: Any) -> Optional[Any]:
    if label in {"Em", "Ex", "pH"}:
        numeric = to_float(value)
        return float(numeric) if numeric is not None else None
    if label in {"solvent", "component"}:
        text = normalize_condition_text(value)
        return text or None
    return None


def extract_tag_values_from_sentence(tag: str, sentence: str) -> List[float]:
    normalized_tag = normalize_tag_name(tag)
    source = str(sentence or "")
    if normalized_tag == "lifetime":
        values: List[float] = []
        for match in LIFETIME_VALUE_RE.finditer(source):
            numeric = to_float(match.group(0))
            if numeric is not None:
                values.append(float(numeric))
        return unique_float(values)
    if normalized_tag == "QY":
        return unique_float(float(value) for value in re.findall(r"(?i)(-?\d+(?:\.\d+)?)\s*%", source))
    return []


def validate_merge_sentence(tag: str, sentence: str, value_a: Optional[float], value_b: Optional[float]) -> Tuple[str, List[Any], List[str]]:
    merge_axis, merge_values = pick_merge_condition_axis(sentence)
    tag_values = extract_tag_values_from_sentence(tag, sentence)
    invalid_reasons: List[str] = []
    if not merge_axis:
        invalid_reasons.append("NO_SINGLE_EXPLICIT_CONDITION_AXIS")
    if len(tag_values) < 2:
        invalid_reasons.append(f"VAL_LT2:{len(tag_values)}")
    elif len(unique_float(tag_values)) < 2:
        invalid_reasons.append("VAL_DISTINCT_LT2")
    if len(merge_values) < 2:
        invalid_reasons.append(f"COND_LT2:{len(merge_values)}")
    if value_a is not None and value_b is not None and abs(float(value_a) - float(value_b)) < 1e-12:
        invalid_reasons.append("A_B_ORIG_VALUES_EQUAL")
    return merge_axis, merge_values, invalid_reasons


def change_tag_spec(tag: str, sample_name: str) -> str:
    normalized_tag = normalize_tag_name(tag)
    if normalized_tag == "QY":
        return (
            f"Definition: QY is the photoluminescence quantum yield of {sample_name}.\n"
            "Keep two candidates together only when both values explicitly belong to the same sample and are mapped to different Em, Ex, pH, or solvent conditions.\n"
            "If this explicit mapping is missing, keep only one best candidate."
        )
    if normalized_tag == "lifetime":
        return (
            f"Definition: lifetime is the photoluminescence decay lifetime of {sample_name}.\n"
            "Keep two candidates together only when both values explicitly belong to the same sample and are mapped to different Em, Ex, pH, or solvent conditions.\n"
            "If this explicit mapping is missing, keep only one best candidate."
        )
    return f"Definition: {normalized_tag} is the target property of {sample_name}."


def build_change_prompt(
    sample_name: str,
    sample_desc: str,
    tag: str,
    value_a: str,
    value_b: str,
    sentence_a: str,
    sentence_b: str,
    blocks_a: Sequence[Dict[str, str]],
    blocks_b: Sequence[Dict[str, str]],
) -> str:
    desc = sample_desc.strip() if str(sample_desc or "").strip() else "(no synthesis description available)"

    def format_blocks(letter: str, blocks: Sequence[Dict[str, str]]) -> str:
        lines = [f"{letter} evidence:"]
        if not blocks:
            lines.append("Evidence 1: (empty)")
            return "\n".join(lines)
        for index, block in enumerate(blocks, start=1):
            lines.append(f"Evidence {index} {block.get('meta', '')}: {block.get('text', '') or '(empty)'}")
        return "\n".join(lines)

    return f"""
You are an expert curator for carbon-dot photoluminescence properties.

Task:
- This is one A/B change case for one target sample and one target tag.
- Candidate A gives {value_a}; Candidate B gives {value_b}.
- Keep two values together only when they explicitly map to different Em, Ex, pH, or solvent conditions for the same target sample.
- If both values can be kept, output one refined sentence that maps each value to one explicit condition on one consistent axis.
- Do not rewrite Ex as Em, and do not rewrite Em as Ex.
- Preserve explicit numeric units.
- If the evidence does not support a valid merged sentence, return NONE.

Target sample: {sample_name}
Target sample description:
{desc}
Target tag: {normalize_tag_name(tag)}
Tag definition:
{change_tag_spec(tag, sample_name)}

Candidate A refine sentence:
{sentence_a}
{format_blocks("A", blocks_a)}

Candidate B refine sentence:
{sentence_b}
{format_blocks("B", blocks_b)}

Output rules:
- Output only one JSON object, then immediately output {END_SENTINEL}
- Allowed schemas:
  {{"refine_sentence":"<one sentence>"}}{END_SENTINEL}
  {{"none":"YES"}}{END_SENTINEL}
- No markdown, no code fence, no extra explanation.
""".strip()


def build_struct_prompt(tag: str, sample_name: str, refine_sentence: str, unit_evidence_text: str) -> str:
    normalized_tag = normalize_tag_name(tag)
    unit_hint = "explicit_lifetime_unit" if normalized_tag == "lifetime" else "%"
    unit_block = ""
    if str(unit_evidence_text or "").strip():
        unit_block = "\n\nUnit evidence for disambiguating units only:\n" + unit_evidence_text
    return f"""
You are an expert curator for carbon-dot photoluminescence properties.

Task:
- Convert one refined sentence into one structured JSON array for tag={normalized_tag}.
- Each array element must represent one value.
- Required keys in each element: "tag", "values", "unit", "vary_by", "vary_values".
- Allowed vary_by values are "Em", "Ex", "pH", and "solvent".
- If the sentence does not explicitly support one consistent condition axis across the kept values, return {{"none":"YES"}}.

Target sample: {sample_name}
Target tag: {normalized_tag}
Tag definition:
{change_tag_spec(tag, sample_name)}
Refined sentence:
{refine_sentence}{unit_block}

Output format:
[{{"tag":"{normalized_tag}","values":V1,"unit":"{unit_hint}","vary_by":"Em_or_Ex_or_pH_or_solvent","vary_values":COND1}},{{"tag":"{normalized_tag}","values":V2,"unit":"{unit_hint}","vary_by":"same_as_first","vary_values":COND2}}]{END_SENTINEL}

Output rules:
- Output only one JSON array or one wrapper object, then immediately output {END_SENTINEL}
- Or output {{"none":"YES"}}{END_SENTINEL}
- No markdown, no code fence, no explanation.
""".strip()


def is_none_token(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) == 0.0
    return str(value).strip().lower() in {"none", "null", "no", "n", "false", "0", "skip", "cannot_merge"}


def parse_refine_output(raw: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    meta: Dict[str, Any] = {"ok": False, "reason": "", "mode": ""}
    body, body_meta = extract_llm_body_before_end_mark(raw, end_mark=END_SENTINEL, allow_missing_end_mark=False)
    if body is None:
        meta["reason"] = body_meta.get("reason", "NO_BODY")
        return None, meta
    if str(body).strip().lower() == "none":
        return {"kind": "NONE", "sentence": None}, {"ok": True, "reason": "NONE_PLAIN", "mode": "plain"}
    candidates = unique_text([body.strip()] + [item.strip() for item in extract_json_fragments(body) if str(item).strip()])
    for candidate in candidates:
        payload, parse_meta = try_parse_jsonish(candidate)
        if not isinstance(payload, dict):
            continue
        for key in ("none", "is_none", "merge", "status", "result", "decision"):
            if key in payload and is_none_token(payload.get(key)):
                return {"kind": "NONE", "sentence": None}, {"ok": True, "reason": "NONE_JSON", "mode": parse_meta.get("mode", "")}
        for key in ("refine_sentence", "sentence", "refine", "answer", "text"):
            if key in payload:
                sentence = first_sentence(str(payload.get(key) or "").strip())
                if sentence:
                    return {"kind": "SENTENCE", "sentence": sentence}, {"ok": True, "reason": "OK", "mode": parse_meta.get("mode", "")}
    plain = body.strip()
    if plain and not plain.startswith("{") and len(plain.split()) >= 3:
        return {"kind": "SENTENCE", "sentence": first_sentence(plain)}, {"ok": True, "reason": "PLAIN_BODY", "mode": "plain"}
    meta["reason"] = "PARSE_FAIL"
    return None, meta


def validate_struct_rows(tag: str, payload: Any, refine_sentence: str, unit_evidence_text: str) -> Optional[List[Dict[str, Any]]]:
    normalized_tag = normalize_tag_name(tag)
    array: Optional[List[Any]] = None
    if isinstance(payload, list):
        array = payload
    elif isinstance(payload, dict):
        for key in ("items", "data", "structured", "result", "rows", "values_list"):
            if isinstance(payload.get(key), list):
                array = payload.get(key)
                break
        if array is None and any(key in payload for key in ("tag", "values", "value", "vary_by", "vary_values")):
            array = [payload]
        elif any(key in payload for key in ("none", "is_none")) and is_none_token(payload.get("none", payload.get("is_none"))):
            return []
    if not isinstance(array, list):
        return None

    out: List[Dict[str, Any]] = []
    for item in array:
        if not isinstance(item, dict):
            return None
        tag_value = normalize_tag_name(str(item.get("tag", "")).strip()) or normalized_tag
        if tag_value != normalized_tag:
            return None
        vary_by = normalize_vary_by(item.get("vary_by", ""))
        if vary_by not in {"Em", "Ex", "pH", "solvent"}:
            return None
        raw_values = parse_num_list_any(item.get("values", item.get("value", None)))
        if not raw_values:
            return None
        if normalized_tag == "QY":
            values = normalize_qy_percent_values(raw_values, refine_sentence, unit_evidence_text)
            unit = "%"
        else:
            values = [float(value) for value in raw_values]
            unit = choose_lifetime_unit(refine_sentence, unit_evidence_text, str(item.get("unit", "")))
        if not unit:
            return None
        vary_values = item.get("vary_values", item.get("vary_value", None))
        vary_value_list = vary_values if isinstance(vary_values, list) else [vary_values]
        if len(vary_value_list) not in {1, len(values)}:
            return None
        if len(vary_value_list) == 1 and len(values) > 1:
            vary_value_list = vary_value_list * len(values)
        for value, vary_value in zip(values, vary_value_list):
            normalized_value = normalize_vary_value(vary_by, vary_value)
            if normalized_value is None:
                return None
            out.append(
                {
                    "tag": normalized_tag,
                    "values": float(value),
                    "unit": unit,
                    "vary_by": vary_by,
                    "vary_values": normalized_value,
                }
            )
    if len(out) < 2:
        return None
    if len({str(item.get("vary_by", "")) for item in out}) != 1:
        return None
    if len({condition_value_key(item.get("vary_values")) for item in out}) < 2:
        return None
    if len({condition_value_key(item.get("values")) for item in out}) < 2:
        return None
    return out


def parse_struct_output(raw: str, tag: str, refine_sentence: str, unit_evidence_text: str) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    meta: Dict[str, Any] = {"ok": False, "reason": "", "mode": ""}
    body, body_meta = extract_llm_body_before_end_mark(raw, end_mark=END_SENTINEL, allow_missing_end_mark=False)
    if body is None:
        meta["reason"] = body_meta.get("reason", "NO_BODY")
        return None, meta
    if body.strip().lower() == "none":
        return [], {"ok": True, "reason": "NONE_PLAIN", "mode": "plain"}
    candidates = unique_text([body.strip()] + [item.strip() for item in extract_json_fragments(body) if str(item).strip()])
    for candidate in candidates:
        payload, parse_meta = try_parse_jsonish(candidate)
        if isinstance(payload, dict) and any(key in payload for key in ("none", "is_none")):
            if is_none_token(payload.get("none", payload.get("is_none"))):
                return [], {"ok": True, "reason": "NONE_JSON", "mode": parse_meta.get("mode", "")}
        validated = validate_struct_rows(tag, payload, refine_sentence, unit_evidence_text)
        if validated is not None:
            return validated, {"ok": True, "reason": "OK", "mode": parse_meta.get("mode", "")}
    meta["reason"] = "PARSE_OR_VALIDATE_FAIL"
    return None, meta


class LLMClient:
    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        self.available = callable(lmstudio_llm)
        self.model = lmstudio_llm(model_name) if self.available else None
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def respond_raw(self, prompt: str) -> str:
        if not self.available or self.model is None:
            raise RuntimeError("lmstudio_not_available")
        response = self.model.respond(prompt, config={"temperature": self.temperature, "maxTokens": self.max_tokens})
        if response is None:
            return ""
        if hasattr(response, "content"):
            try:
                return "" if response.content is None else str(response.content)
            except Exception:
                pass
        return str(response)

    def call_refine(self, prompt: str, max_retry: int, trace: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not self.available:
            trace.append({"attempt": 0, "status": "FAIL", "raw": "LLM_UNAVAILABLE", "meta": {"reason": "LLM_UNAVAILABLE"}})
            return None
        for attempt in range(1, max_retry + 1):
            try:
                raw = self.respond_raw(prompt)
                value, meta = parse_refine_output(raw)
            except Exception as exc:
                raw = f"LLM_CALL_ERROR: {exc}"
                value = None
                meta = {"reason": "LLM_CALL_ERROR"}
            ok = bool(value and value.get("kind") in {"SENTENCE", "NONE"})
            trace.append({"attempt": attempt, "status": "OK" if ok else "FAIL", "raw": raw, "meta": meta, "value": value})
            if ok:
                return value
        return None

    def call_struct(
        self,
        prompt: str,
        tag: str,
        max_retry: int,
        trace: List[Dict[str, Any]],
        *,
        refine_sentence: str,
        unit_evidence_text: str,
    ) -> Optional[List[Dict[str, Any]]]:
        if not self.available:
            trace.append({"attempt": 0, "status": "FAIL", "raw": "LLM_UNAVAILABLE", "meta": {"reason": "LLM_UNAVAILABLE"}})
            return None
        for attempt in range(1, max_retry + 1):
            try:
                raw = self.respond_raw(prompt)
                value, meta = parse_struct_output(raw, tag, refine_sentence, unit_evidence_text)
            except Exception as exc:
                raw = f"LLM_CALL_ERROR: {exc}"
                value = None
                meta = {"reason": "LLM_CALL_ERROR"}
            trace.append({"attempt": attempt, "status": "OK" if value is not None else "FAIL", "raw": raw, "meta": meta, "value": value})
            if value is not None:
                return value
        return None


def choose_vote_sentence(votes: Sequence[Optional[Dict[str, Any]]]) -> Tuple[str, str, bool]:
    none_count = 0
    sentences: List[str] = []
    for vote in votes:
        if not isinstance(vote, dict):
            continue
        kind = str(vote.get("kind", "")).upper()
        if kind == "NONE":
            none_count += 1
            continue
        if kind == "SENTENCE":
            sentence = first_sentence(str(vote.get("sentence", "")).strip())
            if sentence:
                sentences.append(sentence)
    if sentences:
        counts = Counter(normalize_sentence_key(sentence) for sentence in sentences)
        best_key, best_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        if none_count > 0 and none_count >= best_count:
            return "", ("NONE_MAJORITY_OR_TIE" if none_count > 1 else "NONE_SINGLE"), True
        chosen = next(sentence for sentence in sentences if normalize_sentence_key(sentence) == best_key)
        return chosen, ("LLM_MAJORITY" if best_count >= 2 else "LLM_PLURALITY"), False
    if none_count > 0:
        return "", ("NONE_MAJORITY_OR_TIE" if none_count > 1 else "NONE_SINGLE"), True
    return "", "NO_VALID_SENTENCE", True


def struct_vote_key(rows: Sequence[Dict[str, Any]]) -> str:
    normalized_rows: List[Tuple[str, str, str, str, str]] = []
    for row in rows:
        normalized_rows.append(
            (
                str(row.get("tag", "")).strip(),
                format_number(row.get("values", "")),
                str(row.get("unit", "")).strip(),
                str(row.get("vary_by", "")).strip(),
                format_number(row.get("vary_values", "")) if isinstance(row.get("vary_values"), (int, float)) else str(row.get("vary_values", "")).strip(),
            )
        )
    normalized_rows.sort()
    return json.dumps(normalized_rows, ensure_ascii=False, separators=(",", ":"))


def choose_vote_struct(votes: Sequence[Optional[List[Dict[str, Any]]]]) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    none_count = len([1 for vote in votes if isinstance(vote, list) and len(vote) == 0])
    valid = [vote for vote in votes if isinstance(vote, list) and len(vote) >= 2]
    if valid:
        counts = Counter(struct_vote_key(vote) for vote in valid)
        best_key, best_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        if none_count > 0 and none_count >= best_count:
            return None, ("NONE_MAJORITY_OR_TIE" if none_count > 1 else "NONE_SINGLE")
        chosen = next(vote for vote in valid if struct_vote_key(vote) == best_key)
        return chosen, ("LLM_MAJORITY" if best_count >= 2 else "LLM_PLURALITY")
    if none_count > 0:
        return None, ("NONE_MAJORITY_OR_TIE" if none_count > 1 else "NONE_SINGLE")
    return None, "NO_VALID_STRUCT"


def make_log_writer(log_path: str):
    def write(message: Any) -> None:
        append_log(log_path, "" if message is None else str(message))
    return write


def next_case_no(counter: Dict[str, int]) -> int:
    counter["n"] = int(counter.get("n", 0)) + 1
    return int(counter["n"])


def pick_fallback_candidate(
    pair: ChangePair,
    blocks_a: Sequence[Dict[str, str]],
    blocks_b: Sequence[Dict[str, str]],
) -> Tuple[str, TagLine, Sequence[Dict[str, str]]]:
    if pair.has_struct_a and not pair.has_struct_b:
        return "A", pair.tag_a, blocks_a
    if pair.has_struct_b and not pair.has_struct_a:
        return "B", pair.tag_b, blocks_b
    if len(blocks_b) > len(blocks_a):
        return "B", pair.tag_b, blocks_b
    return "A", pair.tag_a, blocks_a


def process_step1_sample(
    sample: SampleBlock,
    llm_client: LLMClient,
    sample_desc: str,
    votes: int,
    retries: int,
    log_writer,
    case_counter: Dict[str, int],
    error_log_writer,
) -> int:
    handled = 0
    for entry in sample.entries:
        pair, reason = inspect_change_pair(entry)
        if pair is None:
            if reason != "NO_CHANGE_MARKER":
                error_log_writer(
                    f"paper_sample={sample.sample_name}\tstage=STEP13_MERGE\tstatus={reason}\tentry_idx={entry.entry_idx}"
                )
            continue
        handled += 1
        blocks = parse_change_evidence_blocks(entry.preprop_lines)
        blocks_a = [block for block in blocks if block.get("letter") == "A"]
        blocks_b = [block for block in blocks if block.get("letter") == "B"]
        value_a_text, _, value_a = value_text_from_tag_line(pair.tag_a)
        value_b_text, _, value_b = value_text_from_tag_line(pair.tag_b)

        merge_supported = pair.has_struct_a and pair.has_struct_b and llm_client.available
        vote_traces: List[List[Dict[str, Any]]] = []
        vote_results: List[Optional[Dict[str, Any]]] = []
        final_sentence = ""
        final_reason = "FALLBACK_SINGLE"
        is_none = True
        merge_axis = ""
        merge_values: List[Any] = []

        if merge_supported:
            prompt = build_change_prompt(
                sample.sample_name,
                sample_desc,
                pair.tag,
                value_a_text,
                value_b_text,
                pair.tag_a.sentence,
                pair.tag_b.sentence,
                blocks_a,
                blocks_b,
            )
            for _ in range(max(1, int(votes))):
                trace: List[Dict[str, Any]] = []
                vote_results.append(llm_client.call_refine(prompt, max_retry=max(1, int(retries)), trace=trace))
                vote_traces.append(trace)
            final_sentence, final_reason, is_none = choose_vote_sentence(vote_results)
            if not is_none:
                merge_axis, merge_values, invalid_reasons = validate_merge_sentence(pair.tag, final_sentence, value_a, value_b)
                if invalid_reasons:
                    final_sentence = ""
                    final_reason = f"{final_reason}|LOCAL_RULE_FORCE_NONE|" + "|".join(invalid_reasons)
                    is_none = True
                else:
                    final_reason = f"{final_reason}|MERGE_BY={merge_axis}"
        else:
            prompt = "(merge skipped: missing structured input or lmstudio unavailable)"
            if not pair.has_struct_a or not pair.has_struct_b:
                final_reason = "FALLBACK_SINGLE_MISSING_STRUCTURED"
            elif not llm_client.available:
                final_reason = "FALLBACK_SINGLE_LLM_UNAVAILABLE"
            else:
                final_reason = "FALLBACK_SINGLE"

        keep_letter, keep_tag_line, keep_blocks = pick_fallback_candidate(pair, blocks_a, blocks_b)
        if is_none:
            structured, structured_text = sanitize_structured_payload(
                pair.tag,
                keep_tag_line.structured,
                keep_tag_line.structured_raw,
            )
            entry.tag_lines = [
                TagLine(
                    entry_idx=entry.entry_idx,
                    order_in_entry=0,
                    prefix="",
                    tag_raw=pair.tag,
                    tag=pair.tag,
                    sentence=keep_tag_line.sentence.strip(),
                    keep=True,
                    structured=structured,
                    structured_raw=structured_text,
                )
            ]
            entry.suppress_structured = False
            entry.force_structured = structured is not None or bool(structured_text.strip())
            final_reason = f"{final_reason}|KEEP_{keep_letter}"
        else:
            entry.tag_lines = [
                TagLine(
                    entry_idx=entry.entry_idx,
                    order_in_entry=0,
                    prefix="",
                    tag_raw=pair.tag,
                    tag=pair.tag,
                    sentence=final_sentence,
                    keep=True,
                    structured=None,
                    structured_raw="",
                )
            ]
            entry.suppress_structured = True
            entry.force_structured = False
        entry.tail_lines = []

        case_no = next_case_no(case_counter)
        log_writer(f"====================== CASE {case_no} ======================")
        log_writer(f"Target sample name: {sample.sample_name}")
        log_writer(f"entry_idx={entry.entry_idx} tag={pair.tag}")
        log_writer(f"value_A={value_a_text} value_B={value_b_text}")
        log_writer(f"EVIDENCE_COUNT_A={len(blocks_a)} EVIDENCE_COUNT_B={len(blocks_b)}")
        log_writer("PROMPT:")
        log_writer(prompt)
        log_writer("")
        for vote_index, trace in enumerate(vote_traces, start=1):
            for record in trace:
                log_writer(f"vote={vote_index} attempt={record.get('attempt')} status={record.get('status')} OUT_RAW_ONE_LINE: {one_line_json_text(record.get('raw'))}")
                log_writer(f"vote={vote_index} attempt={record.get('attempt')} PARSE_REASON={(record.get('meta') or {}).get('reason', '')}")
            log_writer(f"vote={vote_index} FINAL_OUTPUT={one_line_json_text(trace[-1].get('value') if trace else None)}")
            log_writer("")
        if is_none:
            log_writer(f"KEEP={keep_letter}")
            log_writer(f"KEEP_REFINE={keep_tag_line.sentence}")
            log_writer(f"KEEP_STRUCT={json.dumps(entry.tag_lines[0].structured, ensure_ascii=False, separators=(',', ':')) if entry.tag_lines[0].structured is not None else 'null'}")
            log_writer(f"KEEP_EVIDENCE_COUNT={len(keep_blocks)}")
        else:
            log_writer(f"FINAL_REFINE_SENTENCE={final_sentence}")
            log_writer(f"FINAL_MERGE_CONDITION_AXIS={merge_axis}")
            log_writer(f"FINAL_MERGE_CONDITION_VALUES={json.dumps(merge_values, ensure_ascii=False, separators=(',', ':'))}")
        log_writer(f"FINAL_DECISION_REASON={final_reason}")
        log_writer("")
    return handled


def process_step2_sample(
    sample: SampleBlock,
    llm_client: LLMClient,
    votes: int,
    retries: int,
    log_writer,
    case_counter: Dict[str, int],
    error_log_writer,
) -> int:
    handled = 0
    for entry in sample.entries:
        if not entry_has_change_marker(entry.preprop_lines):
            continue
        if len(entry.tag_lines) != 1:
            error_log_writer(
                f"paper_sample={sample.sample_name}\tstage=STEP13_STRUCTURE\tstatus=UNEXPECTED_TAG_LINE_COUNT\tentry_idx={entry.entry_idx}\ttag_line_count={len(entry.tag_lines)}"
            )
            continue
        tag_line = entry.tag_lines[0]
        tag = normalize_tag_name(tag_line.tag)
        if tag not in CHANGE_TAGS:
            continue
        if should_write_structured(entry) and (tag_line.structured is not None or str(tag_line.structured_raw or "").strip()):
            continue
        handled += 1
        blocks = parse_change_evidence_blocks(entry.preprop_lines)
        evidence_text = "\n".join(str(block.get("text", "") or "").strip() for block in blocks if str(block.get("text", "") or "").strip()).strip()
        prompt = build_struct_prompt(tag, sample.sample_name, tag_line.sentence, evidence_text)
        vote_traces: List[List[Dict[str, Any]]] = []
        vote_results: List[Optional[List[Dict[str, Any]]]] = []
        if llm_client.available:
            for _ in range(max(1, int(votes))):
                trace: List[Dict[str, Any]] = []
                vote_results.append(
                    llm_client.call_struct(
                        prompt,
                        tag=tag,
                        max_retry=max(1, int(retries)),
                        trace=trace,
                        refine_sentence=tag_line.sentence,
                        unit_evidence_text=evidence_text,
                    )
                )
                vote_traces.append(trace)
        final_rows, reason = choose_vote_struct(vote_results)
        if isinstance(final_rows, list) and len(final_rows) >= 2:
            tag_line.structured = final_rows
            tag_line.structured_raw = json.dumps(final_rows, ensure_ascii=False, separators=(",", ":"))
            entry.suppress_structured = False
            entry.force_structured = True
        else:
            entry.suppress_structured = True
            entry.force_structured = False
            error_log_writer(
                f"paper_sample={sample.sample_name}\tstage=STEP13_STRUCTURE\tstatus={reason}\tentry_idx={entry.entry_idx}\ttag={tag}"
            )
        case_no = next_case_no(case_counter)
        log_writer(f"====================== CASE {case_no} ======================")
        log_writer(f"Target sample name: {sample.sample_name}")
        log_writer(f"entry_idx={entry.entry_idx} tag={tag}")
        log_writer(f"refine_sentence={tag_line.sentence}")
        log_writer("PROMPT:")
        log_writer(prompt)
        log_writer("")
        for vote_index, trace in enumerate(vote_traces, start=1):
            for record in trace:
                log_writer(f"vote={vote_index} attempt={record.get('attempt')} status={record.get('status')} OUT_RAW_ONE_LINE: {one_line_json_text(record.get('raw'))}")
                log_writer(f"vote={vote_index} attempt={record.get('attempt')} PARSE_REASON={(record.get('meta') or {}).get('reason', '')}")
            final_value = trace[-1].get("value") if trace else None
            log_writer(f"vote={vote_index} FINAL_STRUCT_ONE_LINE: {json.dumps(final_value, ensure_ascii=False, separators=(',', ':')) if isinstance(final_value, list) else 'null'}")
            log_writer("")
        log_writer(f"FINAL_STRUCT_ONE_LINE: {json.dumps(final_rows, ensure_ascii=False, separators=(',', ':')) if isinstance(final_rows, list) else 'null'}")
        log_writer(f"FINAL_DECISION_REASON={reason}")
        log_writer("")
    return handled


def sanitize_samples(samples: Sequence[SampleBlock]) -> None:
    for sample in samples:
        for entry in sample.entries:
            for tag_line in entry.tag_lines:
                structured, structured_text = sanitize_structured_payload(tag_line.tag, tag_line.structured, tag_line.structured_raw)
                tag_line.structured = structured
                tag_line.structured_raw = structured_text


def build_sample_desc_map(paper_dir: str, paper_id: str) -> Dict[str, str]:
    return {sample.name: sample.desc for sample in read_letter_table_samples(paper_dir, paper_id)}


def build_log_paths(output_dir: str, paper_id: str) -> Dict[str, str]:
    return {
        "io": os.path.join(output_dir, f"{paper_id}_step13.io.log"),
        "merge": os.path.join(output_dir, f"{paper_id}_step13_merge.trace.log"),
        "structure": os.path.join(output_dir, f"{paper_id}_step13_structure.trace.log"),
        "error": os.path.join(output_dir, f"{paper_id}_step13_error.trace.log"),
    }


def write_io_log(result: Step13Result, log_path: str) -> None:
    append_log(log_path, f"[{timestamp_now()}] paper={result.paper_id} status={result.status}")
    if result.input_md:
        append_log(log_path, f"input_md={relative_to_paper(result.paper_dir, result.input_md)}")
    if result.output_merge_md:
        append_log(log_path, f"output_merge_md={relative_to_paper(result.paper_dir, result.output_merge_md)}")
    if result.output_struct_md:
        append_log(log_path, f"output_struct_md={relative_to_paper(result.paper_dir, result.output_struct_md)}")
    if result.output_main_md:
        append_log(log_path, f"output_main_md={relative_to_paper(result.paper_dir, result.output_main_md)}")
    if result.note:
        append_log(log_path, f"note={result.note}")
    append_log(log_path, "")


def process_one_paper(
    paper_dir: str,
    *,
    llm_client: LLMClient,
    skip_existing: bool,
    votes: int,
    retries: int,
) -> Step13Result:
    paper_id = paper_id_from_dir(paper_dir)
    if paper_id is None:
        return Step13Result("", paper_dir, "SKIP_INVALID_DIR", "", "", "", "", "Directory name does not start with a paper id.")

    input_md = stage_markdown_path(paper_dir, INPUT_STAGE_DIR, paper_id, kind="main")
    output_dir = os.path.join(paper_dir, "property", OUTPUT_STAGE_DIR)
    output_merge_md = os.path.join(output_dir, f"{paper_id}_merge.md")
    output_struct_md = os.path.join(output_dir, f"{paper_id}_structured.md")
    output_main_md = stage_markdown_path(paper_dir, OUTPUT_STAGE_DIR, paper_id, kind="main")
    ensure_dir(output_dir)
    log_paths = build_log_paths(output_dir, paper_id)

    if skip_existing and os.path.exists(output_main_md):
        result = Step13Result(
            paper_id=paper_id,
            paper_dir=paper_dir,
            status="SKIP_EXISTS",
            input_md=input_md,
            output_merge_md=output_merge_md,
            output_struct_md=output_struct_md,
            output_main_md=output_main_md,
            note="Step 13 output already exists.",
        )
        write_io_log(result, log_paths["io"])
        return result

    if not os.path.exists(input_md):
        result = Step13Result(
            paper_id=paper_id,
            paper_dir=paper_dir,
            status="SKIP_NO_INPUT_MD",
            input_md=input_md,
            output_merge_md=output_merge_md,
            output_struct_md=output_struct_md,
            output_main_md=output_main_md,
            note="Missing Step 12 markdown.",
        )
        write_io_log(result, log_paths["io"])
        return result

    if not read_text(input_md).strip():
        write_text(output_main_md, "")
        result = Step13Result(
            paper_id=paper_id,
            paper_dir=paper_dir,
            status="PROCESSED_EMPTY",
            input_md=input_md,
            output_merge_md=output_merge_md,
            output_struct_md=output_struct_md,
            output_main_md=output_main_md,
            note="Input markdown was empty.",
        )
        write_io_log(result, log_paths["io"])
        return result

    write_text(log_paths["merge"], "")
    write_text(log_paths["structure"], "")
    write_text(log_paths["error"], "")
    merge_log_writer = make_log_writer(log_paths["merge"])
    structure_log_writer = make_log_writer(log_paths["structure"])
    error_log_writer = make_log_writer(log_paths["error"])

    try:
        paper_header, samples = parse_markdown(input_md)
    except Exception as exc:
        result = Step13Result(
            paper_id=paper_id,
            paper_dir=paper_dir,
            status="SKIP_PARSE_ERROR",
            input_md=input_md,
            output_merge_md=output_merge_md,
            output_struct_md=output_struct_md,
            output_main_md=output_main_md,
            note=repr(exc),
        )
        error_log_writer(f"paper={paper_id}\tstage=PARSE_MD\tstatus=ERROR\tdetail={exc!r}")
        write_io_log(result, log_paths["io"])
        return result

    sample_desc_map = build_sample_desc_map(paper_dir, paper_id)
    has_change = any(entry_has_change_marker(entry.preprop_lines) for sample in samples for entry in sample.entries)

    merge_case_counter = {"n": 0}
    merge_cases = 0
    if has_change:
        for sample in tqdm(samples, desc=f"Step13: merge-change {paper_id}", leave=False):
            merge_cases += process_step1_sample(
                sample,
                llm_client,
                sample_desc_map.get(sample.sample_name, "") or "",
                votes=votes,
                retries=retries,
                log_writer=merge_log_writer,
                case_counter=merge_case_counter,
                error_log_writer=error_log_writer,
            )
        sanitize_samples(samples)
        write_markdown(output_merge_md, paper_header, samples)
    else:
        sanitize_samples(samples)
        write_markdown(output_merge_md, paper_header, samples)

    structure_case_counter = {"n": 0}
    structure_cases = 0
    if has_change:
        for sample in tqdm(samples, desc=f"Step13: structure-change {paper_id}", leave=False):
            structure_cases += process_step2_sample(
                sample,
                llm_client,
                votes=votes,
                retries=retries,
                log_writer=structure_log_writer,
                case_counter=structure_case_counter,
                error_log_writer=error_log_writer,
            )
    sanitize_samples(samples)
    write_markdown(output_struct_md, paper_header, samples)
    write_markdown(output_main_md, paper_header, samples)

    status = "PROCESSED" if read_text(output_main_md).strip() else "PROCESSED_EMPTY"
    note_parts = [
        f"samples={len(samples)}",
        f"has_change={has_change}",
        f"merge_cases={merge_cases}",
        f"structure_cases={structure_cases}",
    ]
    if not llm_client.available:
        note_parts.append("lmstudio_unavailable")
    if not sample_desc_map:
        note_parts.append("synthesis_letter_table_missing_or_empty")
    result = Step13Result(
        paper_id=paper_id,
        paper_dir=paper_dir,
        status=status,
        input_md=input_md,
        output_merge_md=output_merge_md,
        output_struct_md=output_struct_md,
        output_main_md=output_main_md,
        note="; ".join(note_parts),
    )
    write_io_log(result, log_paths["io"])
    return result


def write_root_logs(mining_root: str, results: Sequence[Step13Result]) -> None:
    main_log_path = os.path.join(mining_root, "step13_resolve_change_entries.log")
    error_log_path = os.path.join(mining_root, "step13_resolve_change_entries_error.log")
    main_statuses = {"SKIP_EXISTS", "PROCESSED", "PROCESSED_EMPTY"}

    with open(main_log_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Step 13 resolve change entries\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status not in main_statuses:
                continue
            handle.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_md:
                handle.write(f"  input_md={relative_to_root(mining_root, result.input_md)}\n")
            if result.output_merge_md:
                handle.write(f"  output_merge_md={relative_to_root(mining_root, result.output_merge_md)}\n")
            if result.output_struct_md:
                handle.write(f"  output_struct_md={relative_to_root(mining_root, result.output_struct_md)}\n")
            if result.output_main_md:
                handle.write(f"  output_main_md={relative_to_root(mining_root, result.output_main_md)}\n")
            if result.note:
                handle.write(f"  note={result.note}\n")

    with open(error_log_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Step 13 resolve change entries issues\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status in main_statuses:
                continue
            handle.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_md:
                handle.write(f"  input_md={relative_to_root(mining_root, result.input_md)}\n")
            if result.output_main_md:
                handle.write(f"  output_main_md={relative_to_root(mining_root, result.output_main_md)}\n")
            if result.note:
                handle.write(f"  note={result.note}\n")


def process_all_papers(
    mining_root: str,
    paper_ids: Optional[Sequence[str]] = None,
    *,
    skip_existing: bool = True,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = DEFAULT_RETRIES,
    votes: int = DEFAULT_VOTES,
) -> None:
    root = ensure_root_exists(mining_root)
    llm_client = LLMClient(model_name, temperature, max_tokens)
    if not llm_client.available:
        print("[WARN] lmstudio is not installed. Step 13 will keep one candidate when merge or structure generation needs the model.")

    results: List[Step13Result] = []
    for paper_dir in tqdm(iter_paper_dirs(root, paper_ids=paper_ids), desc="Step13: resolve-change-entries"):
        try:
            results.append(
                process_one_paper(
                    paper_dir,
                    llm_client=llm_client,
                    skip_existing=skip_existing,
                    votes=max(1, int(votes)),
                    retries=max(1, int(retries)),
                )
            )
        except Exception as exc:
            paper_id = paper_id_from_dir(paper_dir) or ""
            results.append(
                Step13Result(
                    paper_id=paper_id,
                    paper_dir=paper_dir,
                    status="SKIP_FATAL",
                    input_md=stage_markdown_path(paper_dir, INPUT_STAGE_DIR, paper_id, kind="main") if paper_id else "",
                    output_merge_md=os.path.join(paper_dir, "property", OUTPUT_STAGE_DIR, f"{paper_id}_merge.md") if paper_id else "",
                    output_struct_md=os.path.join(paper_dir, "property", OUTPUT_STAGE_DIR, f"{paper_id}_structured.md") if paper_id else "",
                    output_main_md=stage_markdown_path(paper_dir, OUTPUT_STAGE_DIR, paper_id, kind="main") if paper_id else "",
                    note=repr(exc),
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
    parser = argparse.ArgumentParser(description="Property preprocessing Step 13: resolve Step 12 change entries.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Re-run even if Step 13 output already exists.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model name for Step 13.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature for Step 13.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="LLM max tokens for Step 13.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retry count per Step 13 vote.")
    parser.add_argument("--votes", type=int, default=DEFAULT_VOTES, help="Vote count for Step 13 merge and structure calls.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_all_papers(
        mining_root=args.root,
        paper_ids=args.paper_ids,
        skip_existing=not args.force,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        votes=args.votes,
    )


if __name__ == "__main__":
    main()
