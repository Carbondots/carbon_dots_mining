#!/usr/bin/env python3

"""Step 12: resolve same-sample property conflicts after Step 11 structuring."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from property_unit import (
    append_log_line as append_log,
    build_decision_sid_maps,
    ensure_dir,
    ensure_root_exists,
    iter_paper_dirs,
    normalize_sample_header as shared_normalize_sample_header,
    paper_id_from_dir,
    parse_sids_text as shared_parse_sids_text,
    read_letter_table_samples,
    read_text,
    relative_to_paper,
    relative_to_root,
    remove_think_blocks,
    resolve_decision_csv_path,
    stage_markdown_path,
    strip_code_fences,
    timestamp_now,
    write_text,
)

try:
    from lmstudio import llm as lmstudio_llm
except Exception:
    lmstudio_llm = None


INPUT_STAGE_DIR = "final_structured_properties"
OUTPUT_STAGE_DIR = "conflict_resolved_properties"
DEFAULT_MODEL = "qwen.qwen2.5-vl-32b-instruct"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1800
DEFAULT_RETRIES = 5
DEFAULT_VOTES = 3
END_SENTINEL = "<END_OF_JSON>"

NUMERIC_TAGS = {"Ex", "Em", "QY", "lifetime"}
CATEGORICAL_TAGS = {"ExDep", "Chiral", "CPL"}
SUPPORTED_TAGS = NUMERIC_TAGS | CATEGORICAL_TAGS
KNOWN_SUFFIXES = {"(APP)", "(VS)", "(VS-refined)", "(VS-REFINED)", "(MAIN)"}
IMAGE_BANNED_SUBSTRS = ("![](images", "images/", ".jpg")
IMAGE_ALLOW_TOKEN = "TABLE_FROM_IMAGE"
CHIRAL_SAMPLE_MARKER_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:L|D)\s*-\s*[A-Za-z0-9]|(?<![A-Za-z0-9])(?:L|D)\s*-?\s*/\s*(?:D|L)\s*-?"
)
CHIRAL_NEG_PATTERNS = (
    r"\bachiral\b",
    r"\bnon[-\s]?chiral\b",
    r"\bnot\s+chiral\b",
    r"\bwithout\s+chirality\b",
    r"\bno\s+chirality\b",
    r"\bchirality\s+(?:is\s+)?absent\b",
    r"\bnot\s+optically\s+active\b",
)
CHIRAL_POS_PATTERNS = (
    r"\bchiral\b",
    r"\bchirality\b",
    r"\boptically\s+active\b",
    r"\bshows?\s+chirality\b",
    r"\bwith\s+chirality\b",
)

SAMPLE_HEADER_RE = re.compile(r"^(#+)\s+(.*\S)\s*$")
ENTRY_START_RE = re.compile(r"^\s*(\d+)\.\s+.*\S\s*$")
PROPERTY_ABSTRACT_RE = re.compile(r"^\s*Property abstract\s*:\s*$", re.IGNORECASE)
STRUCTURED_RE = re.compile(r"^\s*Structured\s*:\s*$", re.IGNORECASE)
TAG_LINE_RE = re.compile(
    r"^\s*(?P<prefix>\[(?P<prefix_inner>[^\]]+)\])?\s*(?P<tag>[A-Za-z][A-Za-z0-9_]*)\s*:\s*(?P<sent>.+?)\s*$"
)
ENTRY_HEAD_META_RE = re.compile(
    r"(?:[A-Z]\.)?\[\s*para\s*=\s*(?P<para>[^;\]]*)\s*;\s*(?:win|window)\s*=\s*(?P<win>[^;\]]*)\s*;\s*sids\s*=\s*(?P<sids>[^\]]*)\s*\]",
    re.IGNORECASE,
)
ENTRY_NUM_PREFIX_RE = re.compile(r"^\s*\d+\.\s+")
INT_RE = re.compile(r"^\s*-?\d+\s*$")


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


def normalize_sample_header(header_text: str) -> Optional[str]:
    text = str(header_text or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("step "):
        return None
    while True:
        trimmed = text.rstrip()
        removed = False
        for suffix in KNOWN_SUFFIXES:
            if trimmed.endswith(suffix):
                text = trimmed[: -len(suffix)].rstrip()
                removed = True
                break
        if not removed:
            break
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_model_output(raw: Any) -> str:
    return strip_code_fences(remove_think_blocks(raw)).strip()


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


def parse_yes_no_maybe(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return True if value == 1 else (False if value == 0 else None)
    if isinstance(value, float):
        return True if value == 1.0 else (False if value == 0.0 else None)
    text = str(value).strip().upper()
    if text in {"YES", "Y", "TRUE", "T", "1"}:
        return True
    if text in {"NO", "N", "FALSE", "F", "0"}:
        return False
    return None


def unique_keep_order(values: Sequence[int]) -> List[int]:
    out: List[int] = []
    seen: Set[int] = set()
    for value in values:
        current = int(value)
        if current in seen:
            continue
        seen.add(current)
        out.append(current)
    return out


def format_meta(para: str, window: str, sids: str) -> str:
    return f"[para={para}; window={window}; sids={sids}]"


def normalize_sentence_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def strip_entry_number_prefix(line: str) -> str:
    return ENTRY_NUM_PREFIX_RE.sub("", str(line or ""), count=1)


def make_log_writer(log_path: str):
    def _write(message: Any) -> None:
        append_log(log_path, "" if message is None else str(message))
    return _write


def next_case_no(case_counter: Dict[str, int]) -> int:
    case_counter["n"] = int(case_counter.get("n", 0)) + 1
    return int(case_counter["n"])


def one_line_json_str(value: Any) -> str:
    try:
        return json.dumps("" if value is None else str(value), ensure_ascii=False)
    except Exception:
        return repr(value)


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
    allow_missing_end_mark: bool = True,
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
        body = cleaned.strip()
    else:
        meta["reason"] = "NO_END_MARK"
        return None, meta
    if not body:
        meta["reason"] = "EMPTY_BODY"
        return None, meta
    meta["ok"] = True
    meta["reason"] = "OK"
    return body, meta


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


@dataclass
class SampleBlock:
    header_line: str
    sample_name: str
    entries: List[Entry] = field(default_factory=list)


@dataclass
class EvidenceBlock:
    para: str
    window: str
    sids: str
    sids_list: List[int]
    meta: str
    evidence_text: str
    evidence_source: str = ""
    evidence_status: str = ""
    evidence_detail: str = ""


@dataclass
class CandidateGroup:
    candidate_id: int
    refine_sentence: str
    value_signature: str
    tag_lines: List[TagLine]
    entry_indices: List[int]
    evidence_blocks: List[EvidenceBlock]


@dataclass
class ConflictCaseRecord:
    sample_name: str
    sample: SampleBlock
    tag: str
    allow_two: bool
    candidates: List[CandidateGroup]
    selected_ids: List[int]
    reason: str
    multi_vote_pairs: List[Tuple[int, int]] = field(default_factory=list)
    promoted_by_paper_level_multi: bool = False


@dataclass
class Step12Result:
    paper_id: str
    paper_dir: str
    status: str
    input_md: str
    input_decision_csv: str
    input_letter_csv: str
    output_md: str
    note: str = ""


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
        header_match = SAMPLE_HEADER_RE.match(line)
        if header_match:
            sample_name = normalize_sample_header(header_match.group(2))
            if sample_name is None:
                if current_sample is None and not in_entry:
                    paper_header.append(line)
                elif in_entry:
                    current_entry_lines.append(line)
                else:
                    pending_pre_entry.append(line)
                continue

            if current_sample is not None and in_entry:
                flush_current_entry()
            current_sample = SampleBlock(header_line=line, sample_name=sample_name, entries=[])
            samples.append(current_sample)
            current_entry_lines = []
            current_entry_idx_hint = None
            pending_pre_entry = []
            in_entry = False
            continue

        if current_sample is None:
            paper_header.append(line)
            continue

        entry_match = ENTRY_START_RE.match(line)
        if entry_match:
            if in_entry:
                flush_current_entry()
            try:
                current_entry_idx_hint = int(entry_match.group(1))
            except Exception:
                current_entry_idx_hint = None
            current_entry_lines = pending_pre_entry + [line]
            pending_pre_entry = []
            in_entry = True
            continue

        if in_entry:
            current_entry_lines.append(line)
        elif line.strip():
            pending_pre_entry.append(line)

    if current_sample is not None and in_entry:
        flush_current_entry()
    return paper_header, samples


def split_entry(entry_lines: Sequence[str], entry_idx_hint: Optional[int]) -> Optional[Entry]:
    if not entry_lines or not any(str(line or "").strip() for line in entry_lines):
        return None

    prop_index = None
    for index, line in enumerate(entry_lines):
        if PROPERTY_ABSTRACT_RE.match(line):
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
        if STRUCTURED_RE.match(line):
            struct_index = index
            break

    property_lines = after if struct_index is None else after[:struct_index]
    structured_lines = [] if struct_index is None else after[struct_index + 1 :]

    tag_lines: List[TagLine] = []
    for order, line in enumerate(property_lines):
        match = TAG_LINE_RE.match(line)
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
        match = TAG_LINE_RE.match(line)
        if not match:
            if str(line).strip():
                tail_lines.append(str(line))
            continue
        prefix = str(match.group("prefix") or "").strip()
        tag = normalize_tag_name(str(match.group("tag") or "").strip())
        payload_raw = str(match.group("sent") or "").strip()
        payload, _ = try_parse_jsonish(payload_raw)
        parsed_struct_rows.append((prefix, tag, payload, payload_raw))

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
        payload = json.dumps(structured, ensure_ascii=False, separators=(",", ":"))
    else:
        payload = structured_raw.strip() or "{}"
    return f"{left}: {payload}"


def build_entry_lines(entry: Entry, out_entry_no: int) -> List[str]:
    out = apply_entry_number_to_preprop(entry.preprop_lines, out_entry_no)
    kept = [tag_line for tag_line in sorted(entry.tag_lines, key=lambda item: item.order_in_entry) if tag_line.keep]
    if not kept:
        return out
    if out and out[-1].strip():
        out.append("")
    out.append(entry.prop_header_line)
    for tag_line in kept:
        left = f"{tag_line.prefix}{normalize_tag_name(tag_line.tag)}" if tag_line.prefix else normalize_tag_name(tag_line.tag)
        out.append(f"{left}: {tag_line.sentence.strip()}")
    out.append("")
    out.append("Structured:")
    for tag_line in kept:
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
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(sample.header_line)
        lines.append("")
        out_no = 1
        for entry in sample.entries:
            lines.extend(build_entry_lines(entry, out_no))
            out_no += 1
            lines.append("")
    while lines and not lines[-1].strip():
        lines.pop()
    write_text(out_path, "\n".join(lines) + ("\n" if lines else ""))


def extract_meta_blocks_from_preprop(preprop_lines: Sequence[str]) -> List[Dict[str, Any]]:
    starts: List[Tuple[int, re.Match[str]]] = []
    for index, line in enumerate(preprop_lines):
        match = ENTRY_HEAD_META_RE.search(str(line))
        if match:
            starts.append((index, match))

    if not starts:
        fallback_lines = [str(line) for line in preprop_lines if line is not None]
        if fallback_lines and ENTRY_START_RE.match(fallback_lines[0]):
            fallback_lines = fallback_lines[1:]
        while fallback_lines and re.match(r"^\s*evidence\s*:\s*$", fallback_lines[0], flags=re.IGNORECASE):
            fallback_lines = fallback_lines[1:]
        fallback_text = "\n".join(fallback_lines).strip()
        return [
            {
                "para": "",
                "window": "",
                "sids": "",
                "meta": format_meta("", "", ""),
                "sids_list": [],
                "fallback_text": fallback_text,
            }
        ]

    blocks: List[Dict[str, Any]] = []
    for index, (start, match) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(preprop_lines)
        segment = [str(line) for line in preprop_lines[start:end] if line is not None]
        para = str(match.group("para") or "").strip()
        window = str(match.group("win") or "").strip()
        sids = re.sub(r"\s+", "", str(match.group("sids") or "").strip())
        sids_list = parse_sids_text(sids)
        fallback_lines = segment[1:] if len(segment) >= 2 else []
        while fallback_lines and re.match(r"^\s*evidence\s*:\s*$", fallback_lines[0], flags=re.IGNORECASE):
            fallback_lines = fallback_lines[1:]
        blocks.append(
            {
                "para": para,
                "window": window,
                "sids": sids,
                "meta": format_meta(para, window, sids),
                "sids_list": sids_list,
                "fallback_text": "\n".join(fallback_lines).strip(),
            }
        )
    return blocks


def recover_evidence_text(
    block: Dict[str, Any],
    sid_to_text: Dict[int, str],
) -> Tuple[str, str, str, str]:
    sids_list = unique_keep_order(
        [int(value) for value in (block.get("sids_list") or []) if parse_int_maybe(value) is not None]
    )
    fallback_text = str(block.get("fallback_text") or "").strip()
    if sids_list:
        texts: List[str] = []
        missing: List[int] = []
        for sent_id in sids_list:
            current = str(sid_to_text.get(sent_id, "") or "").strip()
            if current:
                texts.append(current)
            else:
                missing.append(int(sent_id))
        if texts:
            detail = f"sids={','.join(str(value) for value in sids_list)}"
            if missing:
                detail += f"; missing_sids={','.join(str(value) for value in missing)}"
            return " ".join(texts).strip(), "decision_csv", "OK_CSV_SIDS", detail
    if fallback_text:
        return fallback_text, "markdown", "FALLBACK_MARKDOWN", "used markdown evidence"
    if sids_list:
        placeholder = f"(evidence unavailable; sids={','.join(str(value) for value in sids_list)})"
        return placeholder, "placeholder", "PLACEHOLDER_SIDS_RECOVER_FAILED", "decision csv lookup failed"
    return "(evidence unavailable)", "placeholder", "PLACEHOLDER_EMPTY", "no sids and no markdown evidence"


def structured_value_signature(tag: str, structured: Any, structured_raw: str) -> str:
    tag_name = normalize_tag_name(tag)
    if structured is None:
        return structured_raw.strip()[:300] if structured_raw else "NONE"
    if tag_name in NUMERIC_TAGS:
        values: List[float] = []
        unit = ""
        try:
            if isinstance(structured, dict):
                unit = str(structured.get("unit", "")).strip()
                raw_values = structured.get("values", [])
                if isinstance(raw_values, list):
                    for item in raw_values:
                        try:
                            values.append(float(item))
                        except Exception:
                            continue
            elif isinstance(structured, list):
                for item in structured:
                    if not isinstance(item, dict):
                        continue
                    if not unit:
                        unit = str(item.get("unit", "")).strip()
                    raw_values = item.get("values", None)
                    if isinstance(raw_values, list):
                        for current in raw_values:
                            try:
                                values.append(float(current))
                            except Exception:
                                continue
                    else:
                        try:
                            values.append(float(raw_values))
                        except Exception:
                            continue
        except Exception:
            pass
        unique_values = sorted(set(values))
        return f"values={unique_values if unique_values else '?'}; unit={unit}"
    if isinstance(structured, dict) and "label" in structured:
        return f"label={structured.get('label')}"
    return json.dumps(structured, ensure_ascii=False, separators=(",", ":"))[:300]


def build_candidates_for_tag(
    entries_and_lines: Sequence[Tuple[Entry, TagLine]],
    sid_to_text: Dict[int, str],
) -> List[CandidateGroup]:
    grouped: Dict[str, List[Tuple[Entry, TagLine]]] = {}
    order_keys: List[str] = []
    for entry, tag_line in entries_and_lines:
        key = normalize_sentence_key(tag_line.sentence)
        if key not in grouped:
            grouped[key] = []
            order_keys.append(key)
        grouped[key].append((entry, tag_line))

    order_keys.sort(
        key=lambda key: min(int(item[0].entry_idx) for item in grouped.get(key, [])) if grouped.get(key) else 10**9
    )

    out: List[CandidateGroup] = []
    candidate_id = 1
    for key in order_keys:
        members = grouped.get(key, [])
        if not members:
            continue
        representative_entry, representative_line = members[0]
        evidence_blocks: List[EvidenceBlock] = []
        for entry, _ in members:
            for block in extract_meta_blocks_from_preprop(entry.preprop_lines):
                evidence_text, source, status, detail = recover_evidence_text(block, sid_to_text)
                evidence_blocks.append(
                    EvidenceBlock(
                        para=str(block.get("para", "")),
                        window=str(block.get("window", "")),
                        sids=str(block.get("sids", "")),
                        sids_list=list(block.get("sids_list", [])),
                        meta=str(block.get("meta", "")),
                        evidence_text=evidence_text,
                        evidence_source=source,
                        evidence_status=status,
                        evidence_detail=detail,
                    )
                )
        out.append(
            CandidateGroup(
                candidate_id=candidate_id,
                refine_sentence=str(representative_line.sentence or "").strip(),
                value_signature=structured_value_signature(
                    representative_line.tag,
                    representative_line.structured,
                    representative_line.structured_raw,
                ),
                tag_lines=[item[1] for item in members],
                entry_indices=sorted({int(item[0].entry_idx) for item in members}),
                evidence_blocks=evidence_blocks,
            )
        )
        candidate_id += 1
    return out


def short_text_for_log(text: Any, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip() + "...<truncated>"


def dominant_evidence_pick(
    candidates: Sequence[CandidateGroup],
    evidence_blocks_by_candidate: Optional[Dict[int, List[EvidenceBlock]]] = None,
) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    scored = sorted(
        [
            (
                candidate.candidate_id,
                len(list((evidence_blocks_by_candidate or {}).get(candidate.candidate_id, candidate.evidence_blocks) or [])),
            )
            for candidate in candidates
        ],
        key=lambda item: (-item[1], item[0]),
    )
    meta: Dict[str, Any] = {
        "rule": "choose_top_only_if_not_tied_and_top_minus_second_gt_1",
        "scored": [{"candidate_id": int(candidate_id), "evidence_count": int(count)} for candidate_id, count in scored],
        "triggered": False,
        "reason": "",
    }
    if not scored:
        meta["reason"] = "NO_CANDIDATE"
        return None, meta
    if len(scored) == 1:
        meta["triggered"] = True
        meta["reason"] = "ONLY_ONE_CANDIDATE"
        return [int(scored[0][0])], meta
    top_id, top_count = scored[0]
    second_count = scored[1][1]
    top_tied = sum(1 for _, count in scored if count == top_count) > 1
    if (not top_tied) and (top_count - second_count > 1):
        meta["triggered"] = True
        meta["reason"] = "DOMINANCE_TRIGGERED"
        return [int(top_id)], meta
    meta["reason"] = "NO_DOMINANCE"
    return None, meta


def inspect_image_evidence(text: Any) -> Dict[str, Any]:
    lowered = str(text or "").lower()
    has_image_marker = any(marker in lowered for marker in IMAGE_BANNED_SUBSTRS)
    has_allow_token = IMAGE_ALLOW_TOKEN.lower() in lowered
    should_filter = bool(has_image_marker and not has_allow_token)
    if not has_image_marker:
        decision = "KEEP_NO_IMAGE_MARKER"
    elif has_allow_token:
        decision = "KEEP_TABLE_FROM_IMAGE_OVERRIDE"
    else:
        decision = "REMOVE_IMAGE_MARKER_NO_TABLE_TOKEN"
    return {
        "has_image_marker": has_image_marker,
        "has_allow_token": has_allow_token,
        "should_filter": should_filter,
        "decision": decision,
    }


def prefilter_candidates_for_llm_image_markers(
    candidates: Sequence[CandidateGroup],
) -> Tuple[List[CandidateGroup], Dict[int, List[EvidenceBlock]], Dict[str, Any]]:
    llm_candidates: List[CandidateGroup] = []
    llm_evidence_map: Dict[int, List[EvidenceBlock]] = {}
    meta: Dict[str, Any] = {
        "excluded_single_image_ids": [],
        "excluded_all_image_ids": [],
        "trimmed_candidate_ids": [],
    }
    for candidate in candidates:
        source_blocks = list(candidate.evidence_blocks or [])
        kept_blocks: List[EvidenceBlock] = []
        removed_count = 0
        removed_image_count = 0
        for block in source_blocks:
            inspection = inspect_image_evidence(block.evidence_text)
            if inspection["should_filter"]:
                removed_count += 1
                removed_image_count += 1
                continue
            kept_blocks.append(block)
        if len(source_blocks) == 1 and removed_count == 1 and removed_image_count == 1:
            meta["excluded_single_image_ids"].append(candidate.candidate_id)
            continue
        if removed_count > 0 and not kept_blocks and removed_image_count == len(source_blocks):
            meta["excluded_all_image_ids"].append(candidate.candidate_id)
            continue
        if removed_count > 0:
            meta["trimmed_candidate_ids"].append(candidate.candidate_id)
        llm_candidates.append(candidate)
        llm_evidence_map[candidate.candidate_id] = kept_blocks if removed_count > 0 else source_blocks
    meta["excluded_single_image_ids"] = sorted(set(meta["excluded_single_image_ids"]))
    meta["excluded_all_image_ids"] = sorted(set(meta["excluded_all_image_ids"]))
    meta["trimmed_candidate_ids"] = sorted(set(meta["trimmed_candidate_ids"]))
    return llm_candidates, llm_evidence_map, meta


def image_prefilter_reason_suffix(meta: Dict[str, Any]) -> str:
    parts: List[str] = []
    if meta.get("excluded_single_image_ids"):
        parts.append(f"IMG_EXCLUDED_SINGLE={meta['excluded_single_image_ids']}")
    if meta.get("excluded_all_image_ids"):
        parts.append(f"IMG_EXCLUDED_ALL={meta['excluded_all_image_ids']}")
    if meta.get("trimmed_candidate_ids"):
        parts.append(f"IMG_TRIMMED={meta['trimmed_candidate_ids']}")
    return "|".join(parts)


def chiral_bool_from_text(text: Any) -> Optional[bool]:
    source = str(text or "")
    if not source.strip():
        return None
    for pattern in CHIRAL_NEG_PATTERNS:
        if re.search(pattern, source, flags=re.IGNORECASE):
            return False
    for pattern in CHIRAL_POS_PATTERNS:
        if re.search(pattern, source, flags=re.IGNORECASE):
            return True
    return None


def chiral_bool_from_obj(obj: Any) -> Optional[bool]:
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if float(obj) == 1.0:
            return True
        if float(obj) == 0.0:
            return False
    if isinstance(obj, str):
        return chiral_bool_from_text(obj)
    if isinstance(obj, dict):
        for key in ("label", "value", "status", "class", "result", "chiral", "is_chiral"):
            if key in obj:
                hit = chiral_bool_from_obj(obj.get(key))
                if hit is not None:
                    return hit
        try:
            return chiral_bool_from_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            return None
    if isinstance(obj, list):
        values = [hit for hit in (chiral_bool_from_obj(item) for item in obj) if hit is not None]
        if not values:
            return None
        return values[0] if len(set(values)) == 1 else None
    return None


def infer_candidate_chiral_value(candidate: CandidateGroup) -> Tuple[Optional[bool], str]:
    for index, tag_line in enumerate(candidate.tag_lines, start=1):
        hit = chiral_bool_from_obj(tag_line.structured)
        if hit is not None:
            return hit, f"structured_obj_tagline_{index}"
    for index, tag_line in enumerate(candidate.tag_lines, start=1):
        hit = chiral_bool_from_text(tag_line.structured_raw)
        if hit is not None:
            return hit, f"structured_raw_tagline_{index}"
    hit = chiral_bool_from_text(candidate.refine_sentence)
    if hit is not None:
        return hit, "refine_sentence"
    for index, tag_line in enumerate(candidate.tag_lines, start=1):
        hit = chiral_bool_from_text(tag_line.sentence)
        if hit is not None:
            return hit, f"tag_sentence_{index}"
    return None, "UNPARSEABLE"


def sample_name_has_chiral_marker(sample_name: str) -> bool:
    return CHIRAL_SAMPLE_MARKER_RE.search(str(sample_name or "")) is not None


def apply_chiral_sample_name_hard_filter(
    sample_name: str,
    candidates: Sequence[CandidateGroup],
) -> Tuple[Optional[List[int]], str, Dict[str, Any]]:
    expected = sample_name_has_chiral_marker(sample_name)
    expected_label = "CHIRAL" if expected else "ACHIRAL"
    matched: List[CandidateGroup] = []
    mismatch_ids: List[int] = []
    unknown_ids: List[int] = []
    meta: Dict[str, Any] = {"expected_label": expected_label}

    for candidate in candidates:
        value, source = infer_candidate_chiral_value(candidate)
        meta.setdefault("candidate_inference", []).append(
            {
                "candidate_id": int(candidate.candidate_id),
                "inferred_value": value,
                "source": source,
            }
        )
        if value is None:
            unknown_ids.append(candidate.candidate_id)
        elif bool(value) == bool(expected):
            matched.append(candidate)
        else:
            mismatch_ids.append(candidate.candidate_id)

    if matched:
        scored = sorted(
            [(candidate.candidate_id, len(candidate.evidence_blocks)) for candidate in matched],
            key=lambda item: (-item[1], item[0]),
        )
        keep_id = int(scored[0][0]) if scored else int(matched[0].candidate_id)
        reason = (
            f"RULE_CHIRAL_SAMPLE_NAME_EXPECT_{expected_label}"
            f"|matched={sorted(candidate.candidate_id for candidate in matched)}"
            f"|mismatch={sorted(mismatch_ids)}"
            f"|unknown={sorted(unknown_ids)}"
        )
        return [keep_id], reason, meta
    if not unknown_ids:
        reason = f"RULE_CHIRAL_SAMPLE_NAME_EXPECT_{expected_label}_DROP_ALL|mismatch={sorted(mismatch_ids)}"
        return [], reason, meta
    reason = (
        f"RULE_CHIRAL_SAMPLE_NAME_EXPECT_{expected_label}_INCONCLUSIVE"
        f"|mismatch={sorted(mismatch_ids)}|unknown={sorted(unknown_ids)}"
    )
    return None, reason, meta


def property_conflict_spec(tag: str, sample_name: str = "the sample") -> str:
    tag_name = normalize_tag_name(tag)
    if tag_name == "Ex":
        return (
            f"Definition: Ex is the excitation wavelength used to excite {sample_name} in PL or PLE measurement.\n"
            "- Prefer explicit best or optimal excitation over generic test settings.\n"
            "- Do not use UV-vis absorption maxima as Ex."
        )
    if tag_name == "Em":
        return (
            f"Definition: Em is the emission peak wavelength of {sample_name} in PL or PLE measurement.\n"
            "- Prefer explicit emission peaks over generic readout settings.\n"
            "- Do not substitute other wavelength concepts."
        )
    if tag_name == "QY":
        return (
            f"Definition: QY is the photoluminescence quantum yield of {sample_name}.\n"
            "- Keep two candidates only when the evidence explicitly links different QY values to different Em, Ex, pH, or solvent conditions for the same sample."
        )
    if tag_name == "lifetime":
        return (
            f"Definition: lifetime is the photoluminescence decay lifetime of {sample_name}.\n"
            "- Keep two candidates only when the evidence explicitly links different lifetime values to different Em, Ex, pH, or solvent conditions for the same sample."
        )
    if tag_name == "ExDep":
        return (
            f"Definition: ExDep indicates whether emission of {sample_name} is excitation-dependent.\n"
            "- Prefer explicit excitation-dependent or excitation-independent conclusions over indirect wording."
        )
    if tag_name == "Chiral":
        return f"Definition: Chiral indicates whether {sample_name} is chiral."
    if tag_name == "CPL":
        return f"Definition: CPL indicates whether {sample_name} shows circularly polarized luminescence."
    return f"Definition: unsupported tag '{tag_name}'."


def build_conflict_prompt(
    sample_name: str,
    sample_desc: str,
    tag: str,
    candidates: Sequence[CandidateGroup],
    evidence_blocks_by_candidate: Optional[Dict[int, List[EvidenceBlock]]] = None,
) -> str:
    tag_name = normalize_tag_name(tag)
    allow_two = tag_name in {"QY", "lifetime"}
    sample_desc_text = sample_desc.strip() if sample_desc and sample_desc.strip() else "(no synthesis description available)"

    candidate_lines: List[str] = []
    for candidate in candidates:
        candidate_lines.append(f"Candidate {candidate.candidate_id}:")
        candidate_lines.append(f"refine_sentence: {candidate.refine_sentence}")
        candidate_lines.append("evidence:")
        blocks = list((evidence_blocks_by_candidate or {}).get(candidate.candidate_id, candidate.evidence_blocks) or [])
        if not blocks:
            candidate_lines.append("Evidence 1: (empty)")
        else:
            for index, block in enumerate(blocks, start=1):
                if block.meta:
                    candidate_lines.append(f"Evidence {index} {block.meta}: {block.evidence_text or '(empty)'}")
                else:
                    candidate_lines.append(f"Evidence {index}: {block.evidence_text or '(empty)'}")
        candidate_lines.append("")

    output_rule = (
        '- Return exactly one JSON object {"best_index": <int>} and then <END_OF_JSON>.'
        if not allow_two
        else (
            '- Default to one best index.\n'
            '- Return two indices only when the same sample has explicitly different QY or lifetime values under different Em, Ex, pH, or solvent conditions.\n'
            '- Use {"best_index": <int>, "multi": "NO"} for one candidate.\n'
            '- Use {"best_index": [<int>, <int>], "multi": "YES"} for two candidates.\n'
            "- Then output <END_OF_JSON>."
        )
    )

    return f"""
You are an expert in carbon-dot photoluminescence property curation.

Select the best candidate for one sample and one property tag.

Target sample: {sample_name}
Target sample description:
{sample_desc_text}
Target tag: {tag_name}
Tag definition:
{property_conflict_spec(tag_name, sample_name)}

Rules:
- Use only evidence about the target sample.
- Ignore pure measurement settings when they do not report the property itself.
- Prefer explicit conclusions over weak inferences.
- Prefer exact property values over vague ranges or generic wording.
- For QY and lifetime, keep two candidates only under the strict multi-condition rule above.
- Output only JSON followed immediately by {END_SENTINEL}.

{output_rule}

Candidates:
{chr(10).join(candidate_lines).strip()}
""".strip()


def to_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[int] = []
        for item in value:
            current = parse_int_maybe(item)
            if current is not None:
                out.append(int(current))
        return out
    current = parse_int_maybe(value)
    if current is not None:
        return [current]
    if isinstance(value, str):
        return [int(item) for item in re.findall(r"\d+", value)]
    return []


def normalize_keep_candidate_ids(obj: Any, allowed_ids: Set[int]) -> List[int]:
    out: List[int] = []
    if isinstance(obj, dict):
        for key in (
            "keep_candidate_ids",
            "candidate_ids",
            "selected_candidate_ids",
            "selected_ids",
            "keep_ids",
            "best_candidate_ids",
            "best_index",
            "best_id",
            "candidate_id",
        ):
            if key in obj:
                out = to_int_list(obj.get(key))
                if out:
                    break
    elif isinstance(obj, list):
        out = to_int_list(obj)
    else:
        out = to_int_list(obj)
    out = [value for value in unique_keep_order(out) if value in allowed_ids]
    return sorted(out)


def parse_keep_candidate_ids_output(
    raw: Any,
    *,
    candidate_ids: Sequence[int],
    allow_two: bool = False,
    end_mark: str = END_SENTINEL,
    allow_missing_end_mark: bool = True,
) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    allowed = {int(value) for value in candidate_ids}
    meta: Dict[str, Any] = {"ok": False, "reason": "", "mode": "", "multi": None}

    def validate_cardinality(ids: List[int], multi_flag: Optional[bool]) -> Tuple[bool, str]:
        if not ids:
            return False, "EMPTY_IDS"
        if not allow_two:
            if multi_flag is True:
                return False, "MULTI_YES_NOT_ALLOWED_FOR_TAG"
            if len(ids) > 1:
                return False, "MULTI_COUNT_NOT_ALLOWED_FOR_TAG"
            return True, "OK"
        if len(ids) > 2:
            return False, "TOO_MANY_IDS"
        if len(ids) == 2:
            if multi_flag is None:
                return False, "MISSING_MULTI_FOR_TWO_IDS"
            if not multi_flag:
                return False, "MULTI_FLAG_CONFLICT_FOR_TWO_IDS"
        if len(ids) == 1 and multi_flag is True:
            return False, "MULTI_YES_BUT_SINGLE_ID"
        return True, "OK"

    def extract_multi_flag(obj: Any, text_fallback: str = "") -> Optional[bool]:
        if isinstance(obj, dict):
            for key in ("multi", "is_multi", "multiple", "multi_keep"):
                if key in obj:
                    return parse_yes_no_maybe(obj.get(key))
        if text_fallback:
            match = re.search(r'"?multi"?\s*[:=]\s*"?([A-Za-z0-9_]+)"?', text_fallback, flags=re.IGNORECASE)
            if match:
                return parse_yes_no_maybe(match.group(1))
        return None

    body, body_meta = extract_llm_body_before_end_mark(
        raw,
        end_mark=end_mark,
        allow_missing_end_mark=allow_missing_end_mark,
    )
    if body is None:
        meta["reason"] = body_meta.get("reason", "NO_BODY")
        return None, meta

    candidates = [body.strip()] + [fragment.strip() for fragment in extract_json_fragments(body) if fragment.strip()]
    unique_candidates: List[str] = []
    seen: Set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    last_invalid_reason = ""
    for candidate in unique_candidates:
        payload, parse_meta = try_parse_jsonish(candidate)
        if payload is None:
            continue
        ids = normalize_keep_candidate_ids(payload, allowed)
        if not ids:
            continue
        multi_flag = extract_multi_flag(payload, candidate)
        ok, reason = validate_cardinality(ids, multi_flag)
        if ok:
            meta["ok"] = True
            meta["reason"] = "OK"
            meta["mode"] = parse_meta.get("mode", "")
            meta["multi"] = multi_flag
            return ids, meta
        last_invalid_reason = reason

    meta["reason"] = last_invalid_reason or "PARSE_FAIL"
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
        response = self.model.respond(
            prompt,
            config={"temperature": self.temperature, "maxTokens": self.max_tokens},
        )
        return response.content if hasattr(response, "content") else str(response)

    def call_keep_candidate_ids(
        self,
        prompt: str,
        candidate_ids: Sequence[int],
        *,
        allow_two: bool = False,
        max_retry: int = 5,
        trace: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[List[int]], str]:
        last_clean = ""
        for attempt in range(1, max_retry + 1):
            try:
                raw = self.respond_raw(prompt)
                clean = normalize_model_output(raw)
                ids, meta = parse_keep_candidate_ids_output(
                    raw,
                    candidate_ids=candidate_ids,
                    allow_two=allow_two,
                    end_mark=END_SENTINEL,
                    allow_missing_end_mark=False,
                )
            except Exception as exc:
                raw = f"LLM_CALL_ERROR: {exc}"
                clean = str(raw)
                ids = None
                meta = {"ok": False, "reason": "LLM_CALL_ERROR", "mode": ""}
            last_clean = clean
            if trace is not None:
                trace.append(
                    {
                        "attempt": attempt,
                        "status": "OK" if ids else "FAIL",
                        "raw": raw,
                        "clean": clean,
                        "ids": ids,
                        "meta": meta,
                    }
                )
            if ids:
                return ids, clean
        return None, last_clean


def choose_from_vote_results(
    vote_results: Sequence[Optional[List[int]]],
    candidates: Sequence[CandidateGroup],
    *,
    allow_two: bool = False,
    max_n: int = 2,
) -> Tuple[List[int], str]:
    cleaned_votes: List[Tuple[int, ...]] = []
    for vote in vote_results:
        if not isinstance(vote, list) or not vote:
            continue
        current = unique_keep_order([int(value) for value in vote if parse_int_maybe(value) is not None])
        if not current:
            continue
        if not allow_two:
            current = current[:1]
        else:
            if len(current) > max_n:
                continue
            current = current[:max_n]
        cleaned_votes.append(tuple(current))

    if cleaned_votes:
        ranked = sorted(
            Counter(cleaned_votes).items(),
            key=lambda item: (-item[1], 0 if len(item[0]) == 1 else 1, item[0]),
        )
        best_ids, best_count = ranked[0]
        return list(best_ids), "LLM_MAJORITY" if best_count >= 2 else "LLM_PLURALITY"

    scored = sorted(
        [(candidate.candidate_id, len(candidate.evidence_blocks)) for candidate in candidates],
        key=lambda item: (-item[1], item[0]),
    )
    if scored:
        return [int(scored[0][0])], "FALLBACK_TOP_EVIDENCE_COUNT"
    return [1], "FALLBACK_DEFAULT_1"


def make_merged_entry_for_two_candidates(tag: str, cand_a: CandidateGroup, cand_b: CandidateGroup) -> Entry:
    preprop_lines: List[str] = ["[change]"]

    def append_candidate_blocks(letter: str, candidate: CandidateGroup) -> None:
        blocks = list(candidate.evidence_blocks or [])
        if not blocks:
            preprop_lines.append(f"{letter}.{format_meta('', '', '')}")
            preprop_lines.append("Evidence:")
            preprop_lines.append("(empty)")
            preprop_lines.append("")
            return
        for block in blocks:
            preprop_lines.append(f"{letter}.{block.meta or format_meta('', '', '')}")
            preprop_lines.append("Evidence:")
            preprop_lines.append((block.evidence_text or "").strip() or "(empty)")
            preprop_lines.append("")

    append_candidate_blocks("A", cand_a)
    append_candidate_blocks("B", cand_b)
    while preprop_lines and not preprop_lines[-1].strip():
        preprop_lines.pop()

    tag_name = normalize_tag_name(tag)
    tag_lines = [
        TagLine(
            entry_idx=0,
            order_in_entry=0,
            prefix="[A]",
            tag_raw=tag_name,
            tag=tag_name,
            sentence=str(cand_a.refine_sentence or "").strip(),
            structured=cand_a.tag_lines[0].structured if cand_a.tag_lines else None,
            structured_raw=cand_a.tag_lines[0].structured_raw if cand_a.tag_lines else "",
        ),
        TagLine(
            entry_idx=0,
            order_in_entry=1,
            prefix="[B]",
            tag_raw=tag_name,
            tag=tag_name,
            sentence=str(cand_b.refine_sentence or "").strip(),
            structured=cand_b.tag_lines[0].structured if cand_b.tag_lines else None,
            structured_raw=cand_b.tag_lines[0].structured_raw if cand_b.tag_lines else "",
        ),
    ]
    return Entry(entry_idx=0, preprop_lines=preprop_lines, prop_header_line="Property abstract:", tag_lines=tag_lines)


def collect_multi_vote_pairs_from_traces(vote_traces: Sequence[Sequence[Dict[str, Any]]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for trace in vote_traces:
        for record in trace:
            ids = record.get("ids")
            if not isinstance(ids, list):
                continue
            current = unique_keep_order([int(value) for value in ids if parse_int_maybe(value) is not None])
            if len(current) == 2:
                out.append(tuple(sorted(current)))
    return out


def select_best_multi_pair(multi_vote_pairs: Sequence[Tuple[int, int]]) -> List[int]:
    if not multi_vote_pairs:
        return []
    pair, _ = sorted(Counter(multi_vote_pairs).items(), key=lambda item: (-item[1], item[0]))[0]
    return [int(pair[0]), int(pair[1])]


def apply_keep_flags_for_candidates(candidates: Sequence[CandidateGroup], selected_ids: Sequence[int]) -> None:
    selected = {int(value) for value in selected_ids if parse_int_maybe(value) is not None}
    for candidate in candidates:
        keep = int(candidate.candidate_id) in selected
        for tag_line in candidate.tag_lines:
            tag_line.keep = keep


def merge_two_selected_candidates(
    sample: SampleBlock,
    tag: str,
    candidates: Sequence[CandidateGroup],
    selected_ids: Sequence[int],
) -> bool:
    ids = unique_keep_order([int(value) for value in selected_ids if parse_int_maybe(value) is not None])
    if len(ids) != 2:
        return False
    cand_a = next((candidate for candidate in candidates if int(candidate.candidate_id) == ids[0]), None)
    cand_b = next((candidate for candidate in candidates if int(candidate.candidate_id) == ids[1]), None)
    if cand_a is None or cand_b is None:
        return False

    merged_entry = make_merged_entry_for_two_candidates(tag, cand_a, cand_b)
    entry_positions = {int(entry.entry_idx): index for index, entry in enumerate(sample.entries)}
    involved_positions: List[int] = []
    for entry_id in list(cand_a.entry_indices or []) + list(cand_b.entry_indices or []):
        if int(entry_id) in entry_positions:
            involved_positions.append(entry_positions[int(entry_id)])
    insert_pos = min(involved_positions) if involved_positions else len(sample.entries)
    sample.entries.insert(insert_pos, merged_entry)
    for tag_line in cand_a.tag_lines:
        tag_line.keep = False
    for tag_line in cand_b.tag_lines:
        tag_line.keep = False
    return True


def prune_sample_entries(sample: SampleBlock) -> None:
    new_entries: List[Entry] = []
    for entry in sample.entries:
        entry.tag_lines = [tag_line for tag_line in entry.tag_lines if tag_line.keep]
        if entry.tag_lines:
            new_entries.append(entry)
    sample.entries = new_entries


def apply_paper_level_multi_promotion(case_records: Sequence[ConflictCaseRecord], log_writer) -> None:
    active_tags = sorted(
        {
            normalize_tag_name(record.tag)
            for record in case_records
            if record.allow_two and len(record.selected_ids) == 2
        }
    )
    if not active_tags:
        return

    log_writer("==== PAPER_LEVEL_MULTI_PROMOTION ====")
    log_writer(f"ACTIVE_MULTI_TAGS={active_tags}")
    promoted = 0
    for record in case_records:
        tag_name = normalize_tag_name(record.tag)
        if (not record.allow_two) or tag_name not in active_tags or len(record.selected_ids) == 2:
            continue
        pair = select_best_multi_pair(record.multi_vote_pairs)
        if len(pair) != 2:
            continue
        allowed_ids = {int(candidate.candidate_id) for candidate in record.candidates}
        pair = unique_keep_order([value for value in pair if value in allowed_ids])
        if len(pair) != 2:
            continue
        apply_keep_flags_for_candidates(record.candidates, pair)
        if not merge_two_selected_candidates(record.sample, record.tag, record.candidates, pair):
            continue
        record.selected_ids = list(pair)
        record.promoted_by_paper_level_multi = True
        record.reason = f"{record.reason}|PAPER_LEVEL_MULTI_PROMOTE" if record.reason else "PAPER_LEVEL_MULTI_PROMOTE"
        promoted += 1
        log_writer(
            f"PAPER_LEVEL_MULTI_PROMOTE sample={record.sample_name} tag={record.tag} "
            f"selected_ids={record.selected_ids} from_multi_pairs={sorted(set(record.multi_vote_pairs))}"
        )
    log_writer(f"PAPER_LEVEL_MULTI_PROMOTION_COUNT={promoted}")
    log_writer("")


def emit_conflict_case_log(
    log_writer,
    case_counter: Dict[str, int],
    *,
    sample_name: str,
    tag: str,
    candidates: Sequence[CandidateGroup],
    reason: str,
    selected_ids: Sequence[int],
    prompt: str,
    vote_traces: Sequence[Sequence[Dict[str, Any]]],
    evidence_decisions: Optional[Dict[int, List[str]]] = None,
    multi_vote_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> None:
    case_no = next_case_no(case_counter)
    log_writer(f"====================== CASE {case_no} ======================")
    log_writer(f"Target sample name: {sample_name}")
    log_writer("DEDUP_KIND: SAME_SAMPLE_TAG_CONFLICT")
    log_writer(f"tag={tag}")
    log_writer(f"CANDIDATE_COUNT={len(list(candidates))}")
    log_writer("")

    for candidate in candidates:
        log_writer(f"CAND{candidate.candidate_id} entry_idxs={candidate.entry_indices}")
        log_writer(f"Refined sentence: {candidate.refine_sentence}")
        log_writer(f"STRUCTURED_VALUES: {candidate.value_signature}")
        log_writer(f"EVIDENCE_BLOCK_COUNT={len(candidate.evidence_blocks)}")
        decisions = (evidence_decisions or {}).get(candidate.candidate_id, [])
        for block_index, block in enumerate(candidate.evidence_blocks, start=1):
            decision = decisions[block_index - 1] if block_index - 1 < len(decisions) else ""
            log_writer(f"  BLOCK{block_index} {block.meta}")
            log_writer(f"  cand{candidate.candidate_id}_evidence_{block_index}_decision={decision}")
            log_writer(f"  cand{candidate.candidate_id}_evidence_{block_index}_source={block.evidence_source or 'unknown'}")
            log_writer(f"  cand{candidate.candidate_id}_evidence_{block_index}_status={block.evidence_status or 'unknown'}")
            if block.evidence_detail:
                log_writer(f"  cand{candidate.candidate_id}_evidence_{block_index}_detail={block.evidence_detail}")
            log_writer(f"  cand{candidate.candidate_id}_evidence_{block_index}_text={one_line_json_str(block.evidence_text)}")
        log_writer("")

    if prompt:
        log_writer("PROMPT:")
        log_writer(prompt.rstrip("\n"))
        log_writer("")

    for vote_index, trace in enumerate(vote_traces, start=1):
        if not trace:
            log_writer(f"vote={vote_index} status=NO_TRACE")
            continue
        final_ids: List[int] = []
        for record in trace:
            if record.get("ids"):
                final_ids = list(record.get("ids"))
            log_writer(
                f"vote={vote_index} attempt={record.get('attempt')} status={record.get('status')} "
                f"OUT_RAW_ONE_LINE: {one_line_json_str(record.get('raw'))}"
            )
            log_writer(
                f"vote={vote_index} attempt={record.get('attempt')} PARSE_REASON="
                f"{(record.get('meta') or {}).get('reason', '')}"
            )
        log_writer(f"vote={vote_index} FINAL_PARSED_IDS={final_ids}")
        log_writer("")

    if multi_vote_pairs:
        log_writer(f"MULTI_YES_VOTE_PAIRS={sorted(set(multi_vote_pairs))}")
        log_writer("")

    log_writer(f"FINAL_KEEP_CANDIDATE_IDS={list(selected_ids)}")
    log_writer(f"FINAL_DECISION_REASON={reason}")
    log_writer("")


def resolve_conflicts_in_one_sample(
    sample: SampleBlock,
    sid_to_text: Dict[int, str],
    llm_client: LLMClient,
    *,
    sample_desc: str,
    vote_n: int,
    retries: int,
    detail_log_writer,
    error_log_writer,
    case_counter: Dict[str, int],
    case_records: Optional[List[ConflictCaseRecord]] = None,
) -> None:
    tag_map: Dict[str, List[Tuple[Entry, TagLine]]] = {}
    for entry in sample.entries:
        for tag_line in entry.tag_lines:
            tag_name = normalize_tag_name(tag_line.tag)
            tag_line.tag = tag_name
            if tag_name in SUPPORTED_TAGS and tag_line.keep:
                tag_map.setdefault(tag_name, []).append((entry, tag_line))

    for tag, members in tag_map.items():
        if len(members) <= 1:
            continue

        candidates = build_candidates_for_tag(members, sid_to_text)
        if len(candidates) <= 1:
            continue

        for candidate in candidates:
            for block_index, block in enumerate(candidate.evidence_blocks, start=1):
                status = str(block.evidence_status or "").strip()
                if status and status != "OK_CSV_SIDS":
                    error_log_writer(
                        f"sample={sample.sample_name}\ttag={tag}\tcandidate={candidate.candidate_id}\t"
                        f"block={block_index}\tstatus={status}\tdetail={block.evidence_detail}"
                    )

        allow_two = tag in {"QY", "lifetime"}
        prompt = ""
        vote_traces: List[List[Dict[str, Any]]] = []
        evidence_decisions: Dict[int, List[str]] = {}
        allow_empty_selected = False
        selected_ids: List[int] = []
        reason = ""

        chiral_selected: Optional[List[int]] = None
        if tag == "Chiral":
            chiral_selected, reason, _ = apply_chiral_sample_name_hard_filter(sample.sample_name, candidates)
        if chiral_selected is not None:
            selected_ids = list(chiral_selected)
            if not selected_ids and "_DROP_ALL" in reason:
                allow_empty_selected = True
            for candidate in candidates:
                evidence_decisions[candidate.candidate_id] = ["CHIRAL_SKIP"] * len(candidate.evidence_blocks)
        else:
            llm_candidates, llm_evidence_map, image_meta = prefilter_candidates_for_llm_image_markers(candidates)
            image_reason_suffix = image_prefilter_reason_suffix(image_meta)
            for candidate in candidates:
                decisions: List[str] = []
                for block in candidate.evidence_blocks:
                    decisions.append("IMAGE_SKIP" if inspect_image_evidence(block.evidence_text)["should_filter"] else "LLM")
                evidence_decisions[candidate.candidate_id] = decisions

            if not llm_candidates:
                selected_ids = []
                allow_empty_selected = True
                reason = "RULE_IMAGE_PREFILTER_EMPTY_DROP_TAG"
            elif len(llm_candidates) == 1:
                selected_ids = [int(llm_candidates[0].candidate_id)]
                reason = "RULE_IMAGE_PREFILTER_SINGLE_REMAIN"
            else:
                direct_ids, _ = dominant_evidence_pick(llm_candidates, evidence_blocks_by_candidate=llm_evidence_map)
                if direct_ids:
                    selected_ids = list(direct_ids)
                    reason = "RULE_EVIDENCE_COUNT_DOMINANCE_AFTER_IMAGE_PREFILTER"
                    for candidate in llm_candidates:
                        evidence_decisions[candidate.candidate_id] = [
                            "IMAGE_SKIP" if value == "IMAGE_SKIP" else "COUNT_SKIP"
                            for value in evidence_decisions.get(candidate.candidate_id, [])
                        ]
                else:
                    prompt = build_conflict_prompt(
                        sample.sample_name,
                        sample_desc,
                        tag,
                        llm_candidates,
                        evidence_blocks_by_candidate=llm_evidence_map,
                    )
                    if llm_client.available:
                        vote_results: List[Optional[List[int]]] = []
                        for _ in range(max(1, int(vote_n))):
                            trace: List[Dict[str, Any]] = []
                            ids, _ = llm_client.call_keep_candidate_ids(
                                prompt,
                                candidate_ids=[candidate.candidate_id for candidate in llm_candidates],
                                allow_two=allow_two,
                                max_retry=max(1, int(retries)),
                                trace=trace,
                            )
                            vote_results.append(ids)
                            vote_traces.append(trace)
                        selected_ids, reason = choose_from_vote_results(
                            vote_results,
                            llm_candidates,
                            allow_two=allow_two,
                            max_n=2,
                        )
                    else:
                        selected_ids, fallback_reason = choose_from_vote_results(
                            [],
                            llm_candidates,
                            allow_two=allow_two,
                            max_n=2,
                        )
                        reason = f"{fallback_reason}|LLM_UNAVAILABLE"
            if image_reason_suffix:
                reason = f"{reason}|{image_reason_suffix}" if reason else image_reason_suffix

        multi_vote_pairs = collect_multi_vote_pairs_from_traces(vote_traces) if allow_two else []
        selected_ids = unique_keep_order([int(value) for value in selected_ids if parse_int_maybe(value) is not None])
        if not allow_two and len(selected_ids) > 1:
            selected_ids = selected_ids[:1]
            reason = f"{reason}|FORCE_SINGLE_TAG" if reason else "FORCE_SINGLE_TAG"
        if allow_two and len(selected_ids) > 2:
            selected_ids = selected_ids[:2]
            reason = f"{reason}|TRIM_TO_TWO" if reason else "TRIM_TO_TWO"
        if not selected_ids and not allow_empty_selected:
            selected_ids = [candidates[0].candidate_id]
            reason = f"{reason}|EMPTY_GUARD" if reason else "EMPTY_GUARD"

        apply_keep_flags_for_candidates(candidates, selected_ids)
        if allow_two and len(selected_ids) == 2 and merge_two_selected_candidates(sample, tag, candidates, selected_ids):
            reason = f"{reason}|MERGE_TWO_TO_ONE_ENTRY" if reason else "MERGE_TWO_TO_ONE_ENTRY"

        emit_conflict_case_log(
            detail_log_writer,
            case_counter,
            sample_name=sample.sample_name,
            tag=tag,
            candidates=candidates,
            reason=reason,
            selected_ids=selected_ids,
            prompt=prompt,
            vote_traces=vote_traces,
            evidence_decisions=evidence_decisions,
            multi_vote_pairs=multi_vote_pairs,
        )
        if case_records is not None:
            case_records.append(
                ConflictCaseRecord(
                    sample_name=sample.sample_name,
                    sample=sample,
                    tag=tag,
                    allow_two=allow_two,
                    candidates=list(candidates),
                    selected_ids=list(selected_ids),
                    reason=str(reason),
                    multi_vote_pairs=list(multi_vote_pairs),
                )
            )


def build_sample_desc_map(paper_dir: str, paper_id: str) -> Dict[str, str]:
    return {sample.name: sample.desc for sample in read_letter_table_samples(paper_dir, paper_id)}


def build_log_paths(output_dir: str, paper_id: str) -> Dict[str, str]:
    return {
        "io": os.path.join(output_dir, f"{paper_id}_step12_conflicts.io.log"),
        "same_sample": os.path.join(output_dir, f"{paper_id}_step12_same_sample.trace.log"),
        "process": os.path.join(output_dir, f"{paper_id}_step12_process.trace.log"),
        "error": os.path.join(output_dir, f"{paper_id}_step12_error.trace.log"),
    }


def write_process_trace(log_path: str, message: str) -> None:
    append_log(log_path, f"[{timestamp_now()}] {message}")


def write_io_log(result: Step12Result, log_path: str) -> None:
    append_log(log_path, f"[{timestamp_now()}] paper={result.paper_id} status={result.status}")
    if result.input_md:
        append_log(log_path, f"input_md={relative_to_paper(result.paper_dir, result.input_md)}")
    if result.input_decision_csv:
        append_log(log_path, f"input_decision_csv={relative_to_paper(result.paper_dir, result.input_decision_csv)}")
    if result.input_letter_csv:
        append_log(log_path, f"input_letter_csv={relative_to_paper(result.paper_dir, result.input_letter_csv)}")
    if result.output_md:
        append_log(log_path, f"output_md={relative_to_paper(result.paper_dir, result.output_md)}")
    if result.note:
        append_log(log_path, f"note={result.note}")
    append_log(log_path, "")


def process_one_paper(
    paper_dir: str,
    *,
    llm_client: LLMClient,
    skip_existing: bool,
    vote_n: int,
    retries: int,
) -> Step12Result:
    paper_id = paper_id_from_dir(paper_dir)
    if paper_id is None:
        return Step12Result("", paper_dir, "SKIP_INVALID_DIR", "", "", "", "", "Directory name does not start with a paper id.")

    input_md = stage_markdown_path(paper_dir, INPUT_STAGE_DIR, paper_id, kind="main")
    decision_csv = resolve_decision_csv_path(paper_dir, paper_id)
    letter_csv = os.path.join(paper_dir, "Synthesis", "letter_table", f"{paper_id}.csv")
    output_md = stage_markdown_path(paper_dir, OUTPUT_STAGE_DIR, paper_id, kind="main")
    output_dir = os.path.dirname(output_md)
    ensure_dir(output_dir)
    log_paths = build_log_paths(output_dir, paper_id)

    if skip_existing and os.path.exists(output_md):
        result = Step12Result(
            paper_id=paper_id,
            paper_dir=paper_dir,
            status="SKIP_EXISTS",
            input_md=input_md,
            input_decision_csv=decision_csv,
            input_letter_csv=letter_csv,
            output_md=output_md,
            note="Step 12 output already exists.",
        )
        write_io_log(result, log_paths["io"])
        return result

    if not os.path.exists(input_md):
        result = Step12Result(
            paper_id=paper_id,
            paper_dir=paper_dir,
            status="SKIP_NO_INPUT_MD",
            input_md=input_md,
            input_decision_csv=decision_csv,
            input_letter_csv=letter_csv,
            output_md=output_md,
            note="Missing Step 11 structured markdown.",
        )
        write_io_log(result, log_paths["io"])
        return result

    if not read_text(input_md).strip():
        write_text(output_md, "")
        result = Step12Result(
            paper_id=paper_id,
            paper_dir=paper_dir,
            status="PROCESSED_EMPTY",
            input_md=input_md,
            input_decision_csv=decision_csv,
            input_letter_csv=letter_csv,
            output_md=output_md,
            note="Input markdown was empty.",
        )
        write_io_log(result, log_paths["io"])
        return result

    write_text(log_paths["same_sample"], "")
    write_text(log_paths["process"], "")
    write_text(log_paths["error"], "")
    detail_log_writer = make_log_writer(log_paths["same_sample"])
    error_log_writer = make_log_writer(log_paths["error"])

    write_process_trace(log_paths["process"], f"paper={paper_id} stage=START input_md={relative_to_paper(paper_dir, input_md)}")

    sid_to_text: Dict[int, str] = {}
    decision_note = "decision_csv_missing"
    if os.path.exists(decision_csv):
        try:
            _, sid_to_text = build_decision_sid_maps(decision_csv)
            decision_note = f"decision_csv_loaded rows={len(sid_to_text)}"
        except Exception as exc:
            decision_note = f"decision_csv_error={exc!r}"
            error_log_writer(f"paper={paper_id}\tstage=LOAD_DECISION_CSV\tstatus=ERROR\tdetail={exc!r}")
    else:
        error_log_writer(f"paper={paper_id}\tstage=LOAD_DECISION_CSV\tstatus=MISSING")
    write_process_trace(log_paths["process"], f"paper={paper_id} stage=DECISION_CSV note={decision_note}")

    sample_desc_map = build_sample_desc_map(paper_dir, paper_id)
    if not sample_desc_map:
        error_log_writer(f"paper={paper_id}\tstage=LOAD_LETTER_TABLE\tstatus=EMPTY_OR_MISSING")
    else:
        write_process_trace(log_paths["process"], f"paper={paper_id} stage=LETTER_TABLE samples={len(sample_desc_map)}")

    try:
        paper_header, samples = parse_markdown(input_md)
    except Exception as exc:
        result = Step12Result(
            paper_id=paper_id,
            paper_dir=paper_dir,
            status="SKIP_PARSE_ERROR",
            input_md=input_md,
            input_decision_csv=decision_csv,
            input_letter_csv=letter_csv,
            output_md=output_md,
            note=repr(exc),
        )
        error_log_writer(f"paper={paper_id}\tstage=PARSE_MD\tstatus=ERROR\tdetail={exc!r}")
        write_io_log(result, log_paths["io"])
        return result

    case_counter = {"n": 0}
    case_records: List[ConflictCaseRecord] = []
    for sample in tqdm(samples, desc=f"Step12: resolve-conflicts {paper_id}", leave=False):
        resolve_conflicts_in_one_sample(
            sample,
            sid_to_text,
            llm_client,
            sample_desc=sample_desc_map.get(sample.sample_name, "") or "",
            vote_n=max(1, int(vote_n)),
            retries=max(1, int(retries)),
            detail_log_writer=detail_log_writer,
            error_log_writer=error_log_writer,
            case_counter=case_counter,
            case_records=case_records,
        )

    apply_paper_level_multi_promotion(case_records, detail_log_writer)
    for sample in samples:
        prune_sample_entries(sample)

    write_markdown(output_md, paper_header, samples)
    non_empty_output = bool(read_text(output_md).strip())
    status = "PROCESSED" if non_empty_output else "PROCESSED_EMPTY"
    note_parts = [
        f"samples={len(samples)}",
        f"conflict_cases={case_counter['n']}",
        decision_note,
    ]
    if not os.path.exists(letter_csv):
        note_parts.append("letter_table_missing")
    elif not sample_desc_map:
        note_parts.append("letter_table_empty")
    result = Step12Result(
        paper_id=paper_id,
        paper_dir=paper_dir,
        status=status,
        input_md=input_md,
        input_decision_csv=decision_csv,
        input_letter_csv=letter_csv,
        output_md=output_md,
        note="; ".join(note_parts),
    )
    write_process_trace(
        log_paths["process"],
        f"paper={paper_id} stage=WRITE_OUTPUT status={status} output_md={relative_to_paper(paper_dir, output_md)}",
    )
    write_io_log(result, log_paths["io"])
    return result


def write_root_logs(mining_root: str, results: Sequence[Step12Result]) -> None:
    main_log_path = os.path.join(mining_root, "step12_resolve_property_conflicts.log")
    error_log_path = os.path.join(mining_root, "step12_resolve_property_conflicts_error.log")
    main_statuses = {"SKIP_EXISTS", "PROCESSED", "PROCESSED_EMPTY"}

    with open(main_log_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Step 12 property conflict resolution\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status not in main_statuses:
                continue
            handle.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_md:
                handle.write(f"  input_md={relative_to_root(mining_root, result.input_md)}\n")
            if result.input_decision_csv:
                handle.write(f"  input_decision_csv={relative_to_root(mining_root, result.input_decision_csv)}\n")
            if result.input_letter_csv:
                handle.write(f"  input_letter_csv={relative_to_root(mining_root, result.input_letter_csv)}\n")
            if result.output_md:
                handle.write(f"  output_md={relative_to_root(mining_root, result.output_md)}\n")
            if result.note:
                handle.write(f"  note={result.note}\n")

    with open(error_log_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Step 12 property conflict resolution issues\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status in main_statuses:
                continue
            handle.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_md:
                handle.write(f"  input_md={relative_to_root(mining_root, result.input_md)}\n")
            if result.input_decision_csv:
                handle.write(f"  input_decision_csv={relative_to_root(mining_root, result.input_decision_csv)}\n")
            if result.input_letter_csv:
                handle.write(f"  input_letter_csv={relative_to_root(mining_root, result.input_letter_csv)}\n")
            if result.output_md:
                handle.write(f"  output_md={relative_to_root(mining_root, result.output_md)}\n")
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
        print("[WARN] lmstudio is not installed. Step 12 will use rule-based fallbacks when needed.")

    results: List[Step12Result] = []
    for paper_dir in tqdm(iter_paper_dirs(root, paper_ids=paper_ids), desc="Step12: resolve-property-conflicts"):
        try:
            results.append(
                process_one_paper(
                    paper_dir,
                    llm_client=llm_client,
                    skip_existing=skip_existing,
                    vote_n=votes,
                    retries=retries,
                )
            )
        except Exception as exc:
            paper_id = paper_id_from_dir(paper_dir) or ""
            results.append(
                Step12Result(
                    paper_id=paper_id,
                    paper_dir=paper_dir,
                    status="SKIP_FATAL",
                    input_md=stage_markdown_path(paper_dir, INPUT_STAGE_DIR, paper_id, kind="main") if paper_id else "",
                    input_decision_csv=resolve_decision_csv_path(paper_dir, paper_id) if paper_id else "",
                    input_letter_csv=os.path.join(paper_dir, "Synthesis", "letter_table", f"{paper_id}.csv") if paper_id else "",
                    output_md=stage_markdown_path(paper_dir, OUTPUT_STAGE_DIR, paper_id, kind="main") if paper_id else "",
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
    parser = argparse.ArgumentParser(description="Property preprocessing Step 12: resolve same-sample property conflicts.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Re-run even if Step 12 output already exists.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model name for Step 12.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature for Step 12.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="LLM max tokens for Step 12.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retry count per Step 12 vote.")
    parser.add_argument("--votes", type=int, default=DEFAULT_VOTES, help="Vote count for Step 12 candidate selection.")
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
