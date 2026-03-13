#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step 11: structure and deduplicate reviewed property markdown."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from property_unit import (
    append_log_line as append_log,
    build_decision_sid_maps,
    build_evidence_lines_from_sids,
    ensure_dir,
    ensure_root_exists,
    iter_paper_dirs,
    paper_id_from_dir,
    parse_property_markdown,
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


INPUT_STAGE_DIR = "reviewed_final_properties"
OUTPUT_STAGE_DIR = "final_structured_properties"
DEFAULT_MODEL = "qwen.qwen2.5-vl-32b-instruct"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 650
DEFAULT_RETRIES = 5
DEFAULT_VOTES = 3
END_SENTINEL = "<END_OF_JSON>"
PROPERTY_TAG_ORDER = ("Ex", "Em", "QY", "lifetime", "ExDep", "Chiral", "CPL")
NUMERIC_TAGS = {"Ex", "Em", "QY", "lifetime"}
CATEGORICAL_TAGS = {"ExDep", "Chiral", "CPL"}
SUPPORTED_TAGS = set(PROPERTY_TAG_ORDER)
TAGS_NEED_UNIT_EVIDENCE = {"QY", "lifetime"}
MAIN_LOG_STATUSES = {"SKIP_EXISTS", "PROCESSED", "PROCESSED_EMPTY"}
QY_PERCENT_RE = re.compile(r"(?i)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:%|percent(?:age)?)")
LIFETIME_UNIT_VALUE_RE = re.compile(r"(?i)\b\d+(?:\.\d+)?\s*(fs|ps|ns|us|\u00b5s|\u03bcs|ms|s)\b")
LIFETIME_UNIT_TOKEN_RE = re.compile(r"(?i)\b(fs|ps|ns|us|\u00b5s|\u03bcs|ms|s)\b")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

TAG_MEANINGS = {
    "Ex": "The excitation wavelength used to excite the target carbon-dot sample.",
    "Em": "The emission peak wavelength of the target carbon-dot sample.",
    "QY": "The photoluminescence quantum yield of the target carbon-dot sample.",
    "lifetime": "The photoluminescence decay lifetime of the target carbon-dot sample.",
    "ExDep": "Whether the target sample shows excitation-dependent emission behavior.",
    "Chiral": "Whether the target sample itself is described as chiral or optically active.",
    "CPL": "Whether the target sample itself is described as showing circularly polarized luminescence.",
}


@dataclass
class EvidenceBlock:
    para_id: int
    win_level: str
    window_sids: List[int] = field(default_factory=list)
    evidence_lines: List[str] = field(default_factory=list)


@dataclass
class CandidateItem:
    entry_id: int
    sample: str
    tag: str
    sentence: str
    evidence_blocks: List[EvidenceBlock] = field(default_factory=list)
    structured: Optional[Any] = None
    keep: bool = True


@dataclass
class Step11Result:
    paper_id: str
    paper_dir: str
    status: str
    input_md: str
    input_decision_csv: str
    output_md: str
    output_snapshot_md: str
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


def normalize_refined_sentence(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def safe_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_model_output(raw: Any) -> str:
    return strip_code_fences(remove_think_blocks(str(raw or ""))).strip()


def llm_call_raw(model, prompt: str, *, temperature: float, max_tokens: int) -> str:
    response = model.respond(prompt, config={"temperature": temperature, "maxTokens": max_tokens})
    return response.content if hasattr(response, "content") else str(response)


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
        value = float(value)
        if value >= 1:
            out.append(value)
            continue
        scaled = value * 100.0
        if any(abs(item - scaled) <= 1e-4 for item in evidence_percents):
            out.append(scaled)
            continue
        out.append(value if any(abs(item - value) <= 1e-6 for item in evidence_percents) else scaled)
    return out


def normalize_lifetime_unit_token(unit: str) -> str:
    text = str(unit or "").strip().lower()
    mapping = {
        "\u00b5s": "us",
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
    return mapping.get(text, text)


def choose_lifetime_unit(refine_sentence: str, evidence_text: str, model_unit: str) -> str:
    counts: Dict[str, int] = {}
    order: Dict[str, int] = {}
    for source_text in (str(evidence_text or ""), str(refine_sentence or "")):
        for match in LIFETIME_UNIT_VALUE_RE.finditer(source_text):
            unit = normalize_lifetime_unit_token(match.group(1))
            counts[unit] = counts.get(unit, 0) + 1
            order.setdefault(unit, len(order))
    if counts:
        return sorted(counts.items(), key=lambda item: (-item[1], order[item[0]]))[0][0]
    for match in LIFETIME_UNIT_TOKEN_RE.finditer(str(model_unit or "")):
        unit = normalize_lifetime_unit_token(match.group(1))
        if unit:
            return unit
    return normalize_lifetime_unit_token(model_unit)


def extract_first_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = NUMBER_RE.search(str(value).strip())
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def extract_fragment(text: str, open_char: str, close_char: str) -> str:
    source = str(text or "")
    start = source.find(open_char)
    if start < 0:
        return ""
    depth = 0
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
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    return ""

def parse_jsonish_value(raw: Any) -> Tuple[Optional[Any], str]:
    cleaned = normalize_model_output(raw)
    if END_SENTINEL in cleaned:
        cleaned = cleaned.split(END_SENTINEL, 1)[0].strip()
    if not cleaned:
        return None, "EMPTY"
    if cleaned.lower() == "none":
        return None, "NONE"
    candidates = [cleaned, extract_fragment(cleaned, "{", "}"), extract_fragment(cleaned, "[", "]")]
    seen: Set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        for parser in (json.loads, ast.literal_eval):
            try:
                payload = parser(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload, "OBJECT"
            if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
                return payload, "ARRAY"
    return None, "FAIL"


def canon_vary_by(value: Any) -> str:
    text = str(value or "").strip()
    compact = re.sub(r"\s+", "", text).lower()
    mapping = {
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
    return mapping.get(compact, text)


def normalize_numeric_unit_and_values(tag: str, unit: Any, values: Sequence[float], refine_sentence: str, evidence_text: str) -> Optional[Tuple[str, List[float]]]:
    tag_name = normalize_tag_name(tag)
    if tag_name in {"Ex", "Em"}:
        return "nm", [float(value) for value in values]
    if tag_name == "QY":
        return "%", normalize_qy_percent_values(values, refine_sentence, evidence_text)
    if tag_name == "lifetime":
        chosen_unit = choose_lifetime_unit(refine_sentence, evidence_text, str(unit or ""))
        return (chosen_unit, [float(value) for value in values]) if chosen_unit else None
    return str(unit or "").strip(), [float(value) for value in values]


def validate_structured(tag: str, payload: Any, refine_sentence: str, evidence_text: str) -> Optional[Any]:
    tag_name = normalize_tag_name(tag)
    if tag_name in CATEGORICAL_TAGS:
        if isinstance(payload, list) and len(payload) == 1:
            payload = payload[0]
        if not isinstance(payload, dict):
            return None
        if normalize_tag_name(payload.get("tag", "")) != tag_name:
            return None
        label = str(payload.get("label", "")).strip().upper()
        return {"tag": tag_name, "label": label} if label in {"YES", "NO"} else None

    allowed_vary = {"", "pH", "solvent", "Ex", "Em"}
    if tag_name == "lifetime":
        allowed_vary.add("component")

    def normalize_one(item: Dict[str, Any], *, require_condition: bool) -> Optional[Dict[str, Any]]:
        if normalize_tag_name(item.get("tag", "")) != tag_name:
            return None
        values_raw = item.get("values", [])
        values_list = values_raw if isinstance(values_raw, list) else [values_raw]
        values: List[float] = []
        for value in values_list:
            number = extract_first_number(value)
            if number is None:
                return None
            values.append(float(number))
        if not values:
            return None
        vary_by = canon_vary_by(item.get("vary_by", ""))
        if vary_by not in allowed_vary:
            return None
        vary_values_raw = item.get("vary_values", [])
        vary_values = vary_values_raw if isinstance(vary_values_raw, list) else ([vary_values_raw] if vary_values_raw not in (None, "") else [])
        if require_condition and not vary_by:
            return None
        if not vary_by:
            vary_values = []
        normalized = normalize_numeric_unit_and_values(tag_name, item.get("unit", ""), values, refine_sentence, evidence_text)
        if normalized is None:
            return None
        unit_out, values_out = normalized
        if vary_by in {"Ex", "Em", "pH"}:
            vary_values = [float(extract_first_number(value)) for value in vary_values if extract_first_number(value) is not None]
        else:
            vary_values = [str(value).strip() for value in vary_values if str(value).strip()]
        return {"tag": tag_name, "values": values_out, "unit": unit_out, "vary_by": vary_by, "vary_values": vary_values}

    if isinstance(payload, dict):
        return normalize_one(payload, require_condition=False)
    if isinstance(payload, list) and payload:
        out = [normalize_one(item, require_condition=True) for item in payload if isinstance(item, dict)]
        out = [item for item in out if item is not None]
        return out or None
    return None


def flatten_values_unit_ignore_conditions(structured: Any, tag: str) -> Optional[Tuple[str, List[float]]]:
    tag_name = normalize_tag_name(tag)
    if isinstance(structured, dict):
        if normalize_tag_name(structured.get("tag", "")) != tag_name:
            return None
        values = structured.get("values", [])
        if not isinstance(values, list):
            return None
        numbers = [extract_first_number(value) for value in values]
        if any(value is None for value in numbers):
            return None
        return str(structured.get("unit", "")).strip().lower(), sorted(set(float(value) for value in numbers if value is not None))
    if isinstance(structured, list) and structured:
        unit_text = None
        values: List[float] = []
        for item in structured:
            if not isinstance(item, dict) or normalize_tag_name(item.get("tag", "")) != tag_name:
                return None
            current_unit = str(item.get("unit", "")).strip().lower()
            if unit_text is None:
                unit_text = current_unit
            elif unit_text != current_unit:
                return None
            raw_value = item.get("values", None)
            if isinstance(raw_value, list):
                if len(raw_value) != 1:
                    return None
                raw_value = raw_value[0]
            number = extract_first_number(raw_value)
            if number is None:
                return None
            values.append(float(number))
        return (unit_text or "", sorted(set(values))) if values else None
    return None


def values_unit_key(structured: Any, tag: str) -> Tuple[Any, ...]:
    tag_name = normalize_tag_name(tag)
    if tag_name in NUMERIC_TAGS:
        flattened = flatten_values_unit_ignore_conditions(structured, tag_name)
        return ("NONE",) if flattened is None else ("NUM", tag_name, flattened[0], tuple(flattened[1]))
    if isinstance(structured, dict):
        return ("CAT", tag_name, str(structured.get("label", "")).strip().upper())
    return ("NONE",)


def struct_preference_key(structured: Any, tag: str) -> Tuple[int, int, int]:
    if structured is None:
        return (0, 0, 0)
    payload_len = len(safe_json_text(structured))
    if isinstance(structured, dict):
        values = structured.get("values", []) if isinstance(structured.get("values", []), list) else []
        vary = 1 if str(structured.get("vary_by", "")).strip() else 0
        return (vary, len(values), payload_len)
    if isinstance(structured, list):
        vary = 1 if any(str(item.get("vary_by", "")).strip() for item in structured if isinstance(item, dict)) else 0
        return (vary, len(structured), payload_len)
    return (0, 0, payload_len)


def build_struct_prompt(tag: str, sample_name: str, refine_sentence: str, evidence_text: str) -> str:
    tag_name = normalize_tag_name(tag)
    meaning = TAG_MEANINGS.get(tag_name, tag_name)
    if tag_name in CATEGORICAL_TAGS:
        return f"""Extract one categorical property statement.
Target sample: {sample_name}
Target tag: {tag_name}
Meaning: {meaning}
Sentence: {refine_sentence}
Return exactly one JSON object with schema {{"tag":"{tag_name}","label":"YES|NO"}} or the word none, then append {END_SENTINEL}."""
    unit_block = ""
    if tag_name in TAGS_NEED_UNIT_EVIDENCE:
        unit_block = f"\nUnit evidence for disambiguating the unit only: {evidence_text}"
    return f"""Extract one structured numeric property statement.
Target sample: {sample_name}
Target tag: {tag_name}
Meaning: {meaning}
Sentence: {refine_sentence}{unit_block}
Return exactly one JSON object, one JSON array, or the word none, then append {END_SENTINEL}.
Use tag={tag_name}. For Ex and Em use unit nm. For QY use %. For lifetime use an explicit time unit.
Allowed vary_by values are "", "pH", "solvent", "Ex", "Em" and also "component" only for lifetime."""


def build_choose_best_prompt(tag: str, sample_name: str, sentences: Sequence[str]) -> str:
    numbered = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(sentences, start=1))
    return f"""Choose the best sentence for the same property.
Target sample: {sample_name}
Target tag: {normalize_tag_name(tag)}
Prefer the clearest sentence with the same core value but richer valid experimental context.
Candidates:\n{numbered}
Return exactly one JSON object {{"best_index": <1-based integer>}} and append {END_SENTINEL}."""

def build_log_paths(output_dir: str, paper_id: str) -> Dict[str, str]:
    return {
        "io": os.path.join(output_dir, f"{paper_id}_step11.io.log"),
        "structure": os.path.join(output_dir, f"{paper_id}_step11_structure.trace.log"),
        "dedup": os.path.join(output_dir, f"{paper_id}_step11_dedup.trace.log"),
    }


def write_io_log(result: Step11Result, log_path: str) -> None:
    append_log(log_path, f"[{timestamp_now()}] paper={result.paper_id} status={result.status}")
    if result.input_md:
        append_log(log_path, f"input_md={relative_to_paper(result.paper_dir, result.input_md)}")
    if result.input_decision_csv:
        append_log(log_path, f"input_decision_csv={relative_to_paper(result.paper_dir, result.input_decision_csv)}")
    if result.output_md:
        append_log(log_path, f"output_md={relative_to_paper(result.paper_dir, result.output_md)}")
    if result.output_snapshot_md:
        append_log(log_path, f"output_snapshot_md={relative_to_paper(result.paper_dir, result.output_snapshot_md)}")
    if result.note:
        append_log(log_path, f"note={result.note}")
    append_log(log_path, "")


def clone_blocks(blocks: Sequence[EvidenceBlock]) -> List[EvidenceBlock]:
    return [EvidenceBlock(block.para_id, block.win_level, list(block.window_sids), list(block.evidence_lines)) for block in blocks]


def block_key(block: EvidenceBlock) -> Tuple[int, str, Tuple[int, ...], Tuple[str, ...]]:
    return (block.para_id, block.win_level, tuple(block.window_sids), tuple(block.evidence_lines))


def block_header(block: EvidenceBlock, entry_no: Optional[int] = None) -> str:
    prefix = f"{entry_no}. " if entry_no is not None else ""
    return f"{prefix}[para={block.para_id}; window={block.win_level}; sids={','.join(str(value) for value in block.window_sids)}]"


def item_location(item: CandidateItem) -> str:
    return block_header(item.evidence_blocks[0]) if item.evidence_blocks else "[para=0; window=; sids=]"


def evidence_text_from_blocks(blocks: Sequence[EvidenceBlock]) -> str:
    return " ".join(str(line).strip() for block in blocks for line in block.evidence_lines if str(line).strip()).strip()


def dedupe_evidence_blocks(blocks: Sequence[EvidenceBlock]) -> List[EvidenceBlock]:
    out: List[EvidenceBlock] = []
    seen: Set[Tuple[int, str, Tuple[int, ...], Tuple[str, ...]]] = set()
    for block in blocks:
        key = block_key(block)
        if key in seen:
            continue
        seen.add(key)
        out.append(EvidenceBlock(block.para_id, block.win_level, list(block.window_sids), list(block.evidence_lines)))
    return out


def candidate_sort_key(item: CandidateItem) -> Tuple[Any, ...]:
    block = item.evidence_blocks[0] if item.evidence_blocks else EvidenceBlock(0, "", [], [])
    try:
        tag_order = PROPERTY_TAG_ORDER.index(item.tag)
    except ValueError:
        tag_order = len(PROPERTY_TAG_ORDER)
    return (str(item.sample).lower(), block.para_id, min(block.window_sids) if block.window_sids else 0, tag_order, item.entry_id)


def group_items_by_sample(items: Sequence[CandidateItem]) -> Dict[str, List[CandidateItem]]:
    grouped: Dict[str, List[CandidateItem]] = defaultdict(list)
    for item in items:
        grouped[item.sample].append(item)
    for sample_items in grouped.values():
        sample_items.sort(key=candidate_sort_key)
    return dict(sorted(grouped.items(), key=lambda item: item[0].lower()))


def render_structured_markdown(items: Sequence[CandidateItem]) -> str:
    grouped = group_items_by_sample(items)
    if not grouped:
        return ""
    lines: List[str] = []
    first_sample = True
    for sample_name, sample_items in grouped.items():
        if not first_sample:
            lines.append("")
        first_sample = False
        lines.append(f"# {sample_name}")
        lines.append("")
        for entry_no, item in enumerate(sample_items, start=1):
            blocks = dedupe_evidence_blocks(item.evidence_blocks) or [EvidenceBlock(0, "", [], [])]
            head, *others = blocks
            lines.append(block_header(head, entry_no=entry_no))
            lines.append("Evidence:")
            lines.extend(head.evidence_lines or [""])
            for other in others:
                lines.append("")
                lines.append(block_header(other))
                lines.append("Evidence:")
                lines.extend(other.evidence_lines or [""])
            lines.append("Property abstract:")
            lines.append(f"{item.tag}: {item.sentence}")
            lines.append("")
            lines.append("Structured:")
            lines.append(f"{item.tag}: {safe_json_text(item.structured)}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_input_candidates(path: str, sid_to_text: Dict[int, str]) -> Tuple[List[CandidateItem], int]:
    candidates: List[CandidateItem] = []
    skipped_unsupported = 0
    entry_id = 1
    for entry in parse_property_markdown(path):
        sample = str(entry.get("sample", "")).strip()
        if not sample:
            continue
        window_sids = [int(value) for value in entry.get("window_sids", []) or [] if str(value).isdigit()]
        evidence_lines = build_evidence_lines_from_sids(window_sids, sid_to_text, fallback_lines=entry.get("evidence_lines", []) or [])
        block = EvidenceBlock(int(entry.get("para_id", 0) or 0), str(entry.get("win_level", "") or "").strip(), window_sids, [str(line) for line in evidence_lines])
        for tag, sentence in entry.get("tag_sentences", []) or []:
            tag_name = normalize_tag_name(tag)
            cleaned_sentence = str(sentence or "").strip()
            if not tag_name or not cleaned_sentence:
                continue
            if tag_name not in SUPPORTED_TAGS:
                skipped_unsupported += 1
                continue
            candidates.append(CandidateItem(entry_id, sample, tag_name, cleaned_sentence, [copy.deepcopy(block)]))
            entry_id += 1
    return candidates, skipped_unsupported


def call_structured_with_votes(model, item: CandidateItem, temperature: float, max_tokens: int, retries: int, votes: int, log_path: str) -> Optional[Any]:
    evidence_text = evidence_text_from_blocks(item.evidence_blocks)
    prompt = build_struct_prompt(item.tag, item.sample, item.sentence, evidence_text)
    append_log(log_path, f"=== STRUCTURE entry={item.entry_id} sample={item.sample} tag={item.tag} time={timestamp_now()} ===")
    append_log(log_path, prompt)
    results: List[Any] = []
    for vote_index in range(1, votes + 1):
        result: Optional[Any] = None
        for attempt in range(1, retries + 1):
            try:
                raw = llm_call_raw(model, prompt, temperature=temperature, max_tokens=max_tokens)
            except Exception as exc:
                append_log(log_path, f"vote={vote_index} attempt={attempt} status=ERROR raw={safe_json_text(repr(exc))}")
                continue
            payload, state = parse_jsonish_value(raw)
            if state == "NONE":
                append_log(log_path, f"vote={vote_index} attempt={attempt} status=NONE raw={safe_json_text(raw)}")
                break
            result = validate_structured(item.tag, payload, item.sentence, evidence_text) if state in {"OBJECT", "ARRAY"} else None
            append_log(log_path, f"vote={vote_index} attempt={attempt} status={'OK' if result is not None else 'FAIL'} raw={safe_json_text(raw)}")
            if result is not None:
                append_log(log_path, f"vote={vote_index} parsed={safe_json_text(result)}")
                break
        results.append(result)
    counts: Dict[Tuple[Any, ...], int] = Counter(("NONE",) if value is None else values_unit_key(value, item.tag) for value in results)
    best = None
    if counts:
        best_key, best_count = sorted(counts.items(), key=lambda item: (-item[1], 0 if item[0] != ("NONE",) else 1, str(item[0])))[0]
        if best_count >= 2 and best_key != ("NONE",):
            matches = [value for value in results if value is not None and values_unit_key(value, item.tag) == best_key]
            best = sorted(matches, key=lambda value: struct_preference_key(value, item.tag), reverse=True)[0] if matches else None
        elif best_count < 2 and normalize_tag_name(item.tag) in NUMERIC_TAGS:
            matches = [value for value in results if value is not None]
            best = sorted(matches, key=lambda value: struct_preference_key(value, item.tag), reverse=True)[0] if matches else None
    append_log(log_path, f"RESULT={safe_json_text(best) if best is not None else 'null'}")
    append_log(log_path, "")
    return best


def pick_best_sentence_index(model, sample_name: str, tag: str, members: Sequence[CandidateItem], temperature: float, max_tokens: int, retries: int, votes: int, log_path: str) -> int:
    prompt = build_choose_best_prompt(tag, sample_name, [item.sentence for item in members])
    append_log(log_path, f"=== STRUCT_EQUAL sample={sample_name} tag={tag} time={timestamp_now()} ===")
    append_log(log_path, prompt)
    vote_indexes: List[Optional[int]] = []
    for vote_index in range(1, votes + 1):
        chosen: Optional[int] = None
        for attempt in range(1, retries + 1):
            try:
                raw = llm_call_raw(model, prompt, temperature=temperature, max_tokens=max_tokens)
            except Exception as exc:
                append_log(log_path, f"vote={vote_index} attempt={attempt} status=ERROR raw={safe_json_text(repr(exc))}")
                continue
            payload, state = parse_jsonish_value(raw)
            if state != "OBJECT" or not isinstance(payload, dict):
                append_log(log_path, f"vote={vote_index} attempt={attempt} status=FAIL raw={safe_json_text(raw)}")
                continue
            try:
                index_value = int(payload.get("best_index", 0))
            except Exception:
                index_value = 0
            if 1 <= index_value <= len(members):
                chosen = index_value
                append_log(log_path, f"vote={vote_index} attempt={attempt} status=OK best_index={chosen} raw={safe_json_text(raw)}")
                break
            append_log(log_path, f"vote={vote_index} attempt={attempt} status=FAIL raw={safe_json_text(raw)}")
        vote_indexes.append(chosen)
    counts = Counter(index for index in vote_indexes if index is not None)
    if counts:
        best_index, best_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        if best_count >= 2:
            return best_index
    scored = []
    for index, item in enumerate(members, start=1):
        scored.append((index, len(item.sentence.strip()), *struct_preference_key(item.structured, tag)))
    scored.sort(key=lambda item: (item[1], item[2], item[3], item[4], -item[0]), reverse=True)
    return scored[0][0] if scored else 1


def resolve_final_destinations(entry_id: int, kept_ids: Set[int], merge_targets: Dict[int, Set[int]], seen: Optional[Set[int]] = None) -> List[int]:
    if entry_id in kept_ids:
        return [entry_id]
    seen = set(seen or set())
    if entry_id in seen:
        return []
    seen.add(entry_id)
    out: Set[int] = set()
    for target in merge_targets.get(entry_id, set()):
        if target in kept_ids:
            out.add(target)
        else:
            out.update(resolve_final_destinations(target, kept_ids, merge_targets, seen))
    return sorted(out)


def deduplicate_one_sample(sample_name: str, items: List[CandidateItem], model, temperature: float, max_tokens: int, retries: int, votes: int, structure_log_path: str, dedup_log_path: str) -> Tuple[List[CandidateItem], List[CandidateItem]]:
    merge_targets: Dict[int, Set[int]] = defaultdict(set)
    seen_exact: Dict[Tuple[str, str, str], CandidateItem] = {}
    for item in items:
        evidence_key = evidence_text_from_blocks(item.evidence_blocks) if item.tag in TAGS_NEED_UNIT_EVIDENCE else ""
        dedup_key = (item.tag, normalize_refined_sentence(item.sentence), normalize_refined_sentence(evidence_key) or str(item.entry_id))
        if dedup_key in seen_exact:
            base = seen_exact[dedup_key]
            item.keep = False
            merge_targets[item.entry_id].add(base.entry_id)
            append_log(dedup_log_path, f"SAME_SENTENCE sample={sample_name} tag={item.tag} keep={base.entry_id} drop={item.entry_id}")
        else:
            seen_exact[dedup_key] = item

    struct_cache: Dict[Tuple[str, str, str], Optional[Any]] = {}
    for item in items:
        if not item.keep:
            continue
        evidence_key = evidence_text_from_blocks(item.evidence_blocks) if item.tag in TAGS_NEED_UNIT_EVIDENCE else ""
        cache_key = (item.tag, normalize_refined_sentence(item.sentence), normalize_refined_sentence(evidence_key) or str(item.entry_id))
        if cache_key not in struct_cache:
            struct_cache[cache_key] = call_structured_with_votes(model, item, temperature, max_tokens, retries, votes, structure_log_path)
        item.structured = copy.deepcopy(struct_cache[cache_key])
        if item.structured is None:
            item.keep = False
            append_log(dedup_log_path, f"DROP_NO_STRUCT sample={sample_name} tag={item.tag} entry={item.entry_id}")

    snapshot_items = [copy.deepcopy(item) for item in items if item.keep and item.structured is not None]
    by_tag: Dict[str, List[CandidateItem]] = defaultdict(list)
    for item in items:
        if item.keep and item.structured is not None:
            by_tag[item.tag].append(item)

    for tag, members in by_tag.items():
        groups: Dict[Tuple[Any, ...], List[CandidateItem]] = defaultdict(list)
        for item in members:
            if item.keep and item.structured is not None:
                groups[values_unit_key(item.structured, tag)].append(item)
        for group in groups.values():
            alive = [item for item in group if item.keep and item.structured is not None]
            if len(alive) <= 1:
                continue
            if len({normalize_refined_sentence(item.sentence) for item in alive}) == 1:
                continue
            alive = sorted(alive, key=lambda item: item.entry_id)
            chosen_index = pick_best_sentence_index(model, sample_name, tag, alive, temperature, max_tokens, retries, votes, dedup_log_path)
            base = alive[0]
            chosen = alive[chosen_index - 1]
            for member in alive[1:]:
                member.keep = False
                merge_targets[member.entry_id].add(base.entry_id)
            base.sentence = chosen.sentence
            base.structured = copy.deepcopy(chosen.structured)
            append_log(dedup_log_path, f"STRUCT_EQUAL sample={sample_name} tag={tag} keep_base={base.entry_id} chosen_sentence_from={chosen.entry_id}")

    for tag, members in by_tag.items():
        if tag not in NUMERIC_TAGS:
            continue
        alive = [item for item in members if item.keep and item.structured is not None]
        flattened = [flatten_values_unit_ignore_conditions(item.structured, tag) for item in alive]
        dropped: Set[int] = set()
        for left in range(len(alive)):
            if left in dropped or flattened[left] is None:
                continue
            left_unit, left_values = flattened[left][0], set(flattened[left][1])
            for right in range(len(alive)):
                if left == right or right in dropped or flattened[right] is None:
                    continue
                right_unit, right_values = flattened[right][0], set(flattened[right][1])
                if left_unit == right_unit and left_values.issubset(right_values) and len(left_values) < len(right_values):
                    alive[left].keep = False
                    merge_targets[alive[left].entry_id].add(alive[right].entry_id)
                    dropped.add(left)
                    append_log(dedup_log_path, f"NUMERIC_SUBSET sample={sample_name} tag={tag} keep={alive[right].entry_id} drop={alive[left].entry_id}")
                    break

    kept_items = [item for item in items if item.keep and item.structured is not None]
    kept_ids = {item.entry_id for item in kept_items}
    dest_lookup = {item.entry_id: item for item in kept_items}
    for item in items:
        if item.entry_id in kept_ids:
            continue
        destinations = resolve_final_destinations(item.entry_id, kept_ids, merge_targets)
        for destination_id in destinations:
            dest_lookup[destination_id].evidence_blocks.extend(clone_blocks(item.evidence_blocks))
    for item in kept_items:
        item.evidence_blocks = dedupe_evidence_blocks(item.evidence_blocks)
    kept_items.sort(key=candidate_sort_key)
    snapshot_items.sort(key=candidate_sort_key)
    return kept_items, snapshot_items

def process_one_paper(paper_dir: str, model, skip_existing: bool, temperature: float, max_tokens: int, retries: int, votes: int) -> Step11Result:
    paper_id = paper_id_from_dir(paper_dir)
    if paper_id is None:
        return Step11Result("", paper_dir, "SKIP_INVALID_DIR", "", "", "", "", "Directory name does not start with a paper id.")

    input_md = stage_markdown_path(paper_dir, INPUT_STAGE_DIR, paper_id, kind="main")
    output_md = stage_markdown_path(paper_dir, OUTPUT_STAGE_DIR, paper_id, kind="main")
    output_dir = os.path.dirname(output_md)
    output_snapshot_md = os.path.join(output_dir, f"{paper_id}_structured_snapshot.md")
    decision_csv = resolve_decision_csv_path(paper_dir, paper_id)
    ensure_dir(output_dir)
    log_paths = build_log_paths(output_dir, paper_id)

    if skip_existing and os.path.exists(output_md):
        result = Step11Result(paper_id, paper_dir, "SKIP_EXISTS", input_md, decision_csv if os.path.exists(decision_csv) else "", output_md, output_snapshot_md, "Step 11 output already exists.")
        write_io_log(result, log_paths["io"])
        return result
    if not os.path.exists(input_md):
        result = Step11Result(paper_id, paper_dir, "SKIP_NO_INPUT_MD", input_md, decision_csv if os.path.exists(decision_csv) else "", output_md, output_snapshot_md, "Missing Step 10 reviewed markdown.")
        write_io_log(result, log_paths["io"])
        return result
    if not read_text(input_md).strip():
        write_text(output_md, "")
        write_text(output_snapshot_md, "")
        result = Step11Result(paper_id, paper_dir, "PROCESSED_EMPTY", input_md, decision_csv if os.path.exists(decision_csv) else "", output_md, output_snapshot_md, "Input markdown was empty.")
        write_io_log(result, log_paths["io"])
        return result

    sid_to_text: Dict[int, str] = {}
    decision_note = "markdown_evidence_only"
    if os.path.exists(decision_csv):
        try:
            _, sid_to_text = build_decision_sid_maps(decision_csv)
            decision_note = f"decision_csv={relative_to_paper(paper_dir, decision_csv)}"
        except Exception as exc:
            decision_note = f"decision_csv_error={exc!r}; fallback=markdown_evidence_only"
    else:
        decision_csv = ""

    candidates, skipped_unsupported = load_input_candidates(input_md, sid_to_text)
    if not candidates:
        write_text(output_md, "")
        write_text(output_snapshot_md, "")
        result = Step11Result(paper_id, paper_dir, "PROCESSED_EMPTY", input_md, decision_csv, output_md, output_snapshot_md, f"No supported property items were found; skipped_unsupported={skipped_unsupported}; {decision_note}")
        write_io_log(result, log_paths["io"])
        return result

    by_sample: Dict[str, List[CandidateItem]] = defaultdict(list)
    for item in candidates:
        by_sample[item.sample].append(item)

    final_items: List[CandidateItem] = []
    snapshot_items: List[CandidateItem] = []
    for sample_name in tqdm(sorted(by_sample.keys(), key=str.lower), desc=f"Step11: structure-dedup {paper_id}", leave=False):
        sample_final, sample_snapshot = deduplicate_one_sample(sample_name, sorted(by_sample[sample_name], key=candidate_sort_key), model, temperature, max_tokens, retries, votes, log_paths["structure"], log_paths["dedup"])
        final_items.extend(sample_final)
        snapshot_items.extend(sample_snapshot)

    final_items.sort(key=candidate_sort_key)
    snapshot_items.sort(key=candidate_sort_key)
    write_text(output_snapshot_md, render_structured_markdown(snapshot_items))
    write_text(output_md, render_structured_markdown(final_items))

    status = "PROCESSED_EMPTY" if not final_items else "PROCESSED"
    note = f"input_items={len(candidates)}; snapshot_items={len(snapshot_items)}; final_items={len(final_items)}; skipped_unsupported={skipped_unsupported}; {decision_note}"
    result = Step11Result(paper_id, paper_dir, status, input_md, decision_csv, output_md, output_snapshot_md, note)
    write_io_log(result, log_paths["io"])
    return result


def write_root_logs(mining_root: str, results: Sequence[Step11Result]) -> None:
    main_log_path = os.path.join(mining_root, "step11_structure_and_dedup.log")
    error_log_path = os.path.join(mining_root, "step11_structure_and_dedup_error.log")
    with open(main_log_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Step 11 structure and deduplication\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status not in MAIN_LOG_STATUSES:
                continue
            handle.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_md:
                handle.write(f"  input_md={relative_to_root(mining_root, result.input_md)}\n")
            if result.input_decision_csv:
                handle.write(f"  input_decision_csv={relative_to_root(mining_root, result.input_decision_csv)}\n")
            if result.output_snapshot_md:
                handle.write(f"  output_snapshot_md={relative_to_root(mining_root, result.output_snapshot_md)}\n")
            if result.output_md:
                handle.write(f"  output_md={relative_to_root(mining_root, result.output_md)}\n")
            if result.note:
                handle.write(f"  note={result.note}\n")
    with open(error_log_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Step 11 structure and deduplication issues\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status in MAIN_LOG_STATUSES:
                continue
            handle.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_md:
                handle.write(f"  input_md={relative_to_root(mining_root, result.input_md)}\n")
            if result.output_snapshot_md:
                handle.write(f"  output_snapshot_md={relative_to_root(mining_root, result.output_snapshot_md)}\n")
            if result.output_md:
                handle.write(f"  output_md={relative_to_root(mining_root, result.output_md)}\n")
            if result.note:
                handle.write(f"  note={result.note}\n")


def process_all_papers(mining_root: str, paper_ids: Optional[Sequence[str]] = None, model_name: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS, retries: int = DEFAULT_RETRIES, votes: int = DEFAULT_VOTES, skip_existing: bool = True) -> None:
    if lmstudio_llm is None:
        raise RuntimeError("lmstudio is not installed. Install or configure LM Studio before running Step 11.")
    root = ensure_root_exists(mining_root)
    model = lmstudio_llm(model_name)
    results: List[Step11Result] = []
    for paper_dir in tqdm(iter_paper_dirs(root, paper_ids=paper_ids), desc="Step11: structure-and-dedup"):
        try:
            results.append(process_one_paper(paper_dir, model, skip_existing, temperature, max_tokens, retries, votes))
        except Exception as exc:
            paper_id = paper_id_from_dir(paper_dir) or ""
            results.append(Step11Result(paper_id, paper_dir, "SKIP_FATAL", stage_markdown_path(paper_dir, INPUT_STAGE_DIR, paper_id, kind="main") if paper_id else "", resolve_decision_csv_path(paper_dir, paper_id) if paper_id else "", stage_markdown_path(paper_dir, OUTPUT_STAGE_DIR, paper_id, kind="main") if paper_id else "", os.path.join(paper_dir, "property", OUTPUT_STAGE_DIR, f"{paper_id}_structured_snapshot.md") if paper_id else "", repr(exc)))
    write_root_logs(root, results)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Property preprocessing Step 11: structure and deduplicate reviewed properties.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Re-run even if Step 11 outputs already exist.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model name for Step 11.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature for Step 11.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="LLM max tokens for Step 11.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Max retries per Step 11 vote.")
    parser.add_argument("--votes", type=int, default=DEFAULT_VOTES, help="Vote count used for structure extraction and sentence selection.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_all_papers(args.root, paper_ids=args.paper_ids, model_name=args.model, temperature=args.temperature, max_tokens=args.max_tokens, retries=args.retries, votes=args.votes, skip_existing=not args.force)


if __name__ == "__main__":
    main()
