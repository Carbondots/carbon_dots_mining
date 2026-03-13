#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from pipeline_utils import dedupe_preserve_order


DEFAULT_PROPERTY_WINDOW = 1
NUMERIC_PROPERTY_TAGS = ("Ex", "Em", "QY", "lifetime")
PROPERTY_TAG_ORDER = ("Ex", "Em", "QY", "lifetime", "ExDep", "Chiral", "CPL")
_NUMERIC_PROPERTY_TAG_SET = set(NUMERIC_PROPERTY_TAGS)
_SUPPORTED_PROPERTY_TAG_SET = set(PROPERTY_TAG_ORDER)
_PROPERTY_ENTRY_RE = re.compile(
    r"^\s*(\d+)\.\s*\[para=(\d+);\s*(?:win|window)=([^;]+);\s*sids=([^\]]*)\]\s*$",
    re.IGNORECASE,
)
_BRACKET_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]\s*")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def iter_paper_dirs(mining_root: str, paper_ids: Optional[Sequence[str]] = None) -> List[str]:
    wanted = {str(pid) for pid in paper_ids or []}
    paper_dirs: List[str] = []
    for name in sorted(os.listdir(mining_root)):
        paper_dir = os.path.join(mining_root, name)
        if not os.path.isdir(paper_dir):
            continue
        folder_id = paper_id_from_dir(paper_dir)
        if folder_id is None:
            continue
        if wanted and folder_id not in wanted:
            continue
        paper_dirs.append(paper_dir)
    return paper_dirs


def paper_id_from_dir(paper_dir: str) -> Optional[str]:
    name = os.path.basename(paper_dir)
    head = name.split("_", 1)[0]
    return head if head.isdigit() else None


def ensure_root_exists(mining_root: str) -> str:
    root = os.path.normpath(mining_root)
    if not os.path.isdir(root):
        raise FileNotFoundError("MINING_ROOT does not exist.")
    return root


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_relpath(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def relative_path(path: str, start: str) -> str:
    try:
        return normalize_relpath(os.path.relpath(path, start=start))
    except Exception:
        return normalize_relpath(path)


def relative_to_root(mining_root: str, path: str) -> str:
    return relative_path(path, ensure_root_exists(mining_root))


def relative_to_paper(paper_dir: str, path: str) -> str:
    return relative_path(path, os.path.normpath(paper_dir))


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def write_text(path: str, text: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def append_text(path: str, text: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log_line(path: Optional[str], line: str) -> None:
    if not path:
        return
    append_text(path, str(line).rstrip() + "\n")


def sort_key(value: Any) -> Tuple[int, Any]:
    try:
        return (0, int(str(value)))
    except Exception:
        return (1, str(value))


def safe_read_csv(path: str, *, dtype: Any = None) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=dtype)
        except Exception:
            continue
    return None


def remove_think_blocks(raw: str) -> str:
    text = str(raw or "")
    return text.split("</think>", 1)[-1].strip() if "</think>" in text else text.strip()


def strip_code_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned, count=1)
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()
    return cleaned


def strip_llm_wrappers(text: str, *, end_sentinel: str = "") -> str:
    cleaned = strip_code_fences(remove_think_blocks(text))
    if end_sentinel and end_sentinel in cleaned:
        cleaned = cleaned.split(end_sentinel, 1)[0].strip()
    return cleaned.strip()


def _extract_balanced_snippet(text: str, open_char: str, close_char: str) -> Optional[str]:
    start = text.find(open_char)
    if start < 0:
        return None

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
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_json_object_text(text: str, *, end_sentinel: str = "") -> Optional[Dict[str, Any]]:
    cleaned = strip_llm_wrappers(text, end_sentinel=end_sentinel)
    if not cleaned:
        return None

    candidates = [cleaned, _extract_balanced_snippet(cleaned, "{", "}")]
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


def parse_json_array_text(
    text: str,
    *,
    end_sentinel: str = "",
    allow_single_object: bool = True,
) -> Optional[List[Any]]:
    cleaned = strip_llm_wrappers(text, end_sentinel=end_sentinel)
    if not cleaned:
        return None

    candidates = [
        cleaned,
        _extract_balanced_snippet(cleaned, "[", "]"),
        _extract_balanced_snippet(cleaned, "{", "}"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, list):
            return payload
        if allow_single_object and isinstance(payload, dict):
            return [payload]
    return None


def parse_boolean_answer(text: str, *, end_sentinel: str = "", default: bool = True) -> bool:
    cleaned = strip_llm_wrappers(text, end_sentinel=end_sentinel)
    if not cleaned:
        return default

    patterns = (
        r"\{(true|false)\}",
        r'"\w+"\s*:\s*(true|false)',
        r"\b(true|false)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
    return default


def replace_path_tokens(text: str, path: str, placeholder: str) -> str:
    if not path:
        return str(text or "")
    sanitized = str(text or "").replace(path, placeholder)
    return sanitized.replace(path.replace("\\", "/"), placeholder)


def sanitize_text(
    text: str,
    *,
    replacements: Sequence[Tuple[str, str]] = (),
    strip_ansi: bool = False,
    normalize_newlines: bool = True,
    line_filter: Optional[Callable[[str], bool]] = None,
    strip_result: bool = True,
) -> str:
    sanitized = str(text or "")
    if strip_ansi:
        sanitized = _ANSI_ESCAPE_RE.sub("", sanitized)
    if normalize_newlines:
        sanitized = sanitized.replace("\r\n", "\n").replace("\r", "\n")
    for source, placeholder in replacements:
        sanitized = replace_path_tokens(sanitized, source, placeholder)

    lines = [line.rstrip() for line in sanitized.split("\n")]
    if line_filter is not None:
        lines = [line for line in lines if line_filter(line)]
    result = "\n".join(lines)
    return result.strip() if strip_result else result


def sanitize_temp_log_text(
    text: str,
    *,
    temp_paper_dir: str,
    temp_root: str = "",
    extra_replacements: Sequence[Tuple[str, str]] = (),
    strip_ansi: bool = False,
    line_filter: Optional[Callable[[str], bool]] = None,
    strip_result: bool = True,
) -> str:
    replacements: List[Tuple[str, str]] = []
    if temp_paper_dir:
        replacements.append((temp_paper_dir, "<paper_dir>"))
    resolved_temp_root = temp_root or os.path.dirname(temp_paper_dir.rstrip("/\\"))
    if resolved_temp_root:
        replacements.append((resolved_temp_root, "<temp_root>"))
    replacements.extend(extra_replacements)
    return sanitize_text(
        text,
        replacements=replacements,
        strip_ansi=strip_ansi,
        line_filter=line_filter,
        strip_result=strip_result,
    )


def build_temp_log_transform(
    *,
    temp_paper_dir: str,
    temp_root: str = "",
    extra_replacements: Sequence[Tuple[str, str]] = (),
    strip_ansi: bool = False,
    line_filter: Optional[Callable[[str], bool]] = None,
    strip_result: bool = True,
) -> Callable[[str], str]:
    def transform(text: str) -> str:
        return sanitize_temp_log_text(
            text,
            temp_paper_dir=temp_paper_dir,
            temp_root=temp_root,
            extra_replacements=extra_replacements,
            strip_ansi=strip_ansi,
            line_filter=line_filter,
            strip_result=strip_result,
        )

    return transform


def copy_sanitized_text_file(
    src_path: str,
    dst_path: str,
    *,
    replacements: Sequence[Tuple[str, str]] = (),
    strip_ansi: bool = False,
    line_filter: Optional[Callable[[str], bool]] = None,
    transform: Optional[Callable[[str], str]] = None,
    strip_result: bool = True,
) -> bool:
    if not os.path.exists(src_path):
        return False
    text = read_text(src_path)
    if transform is not None:
        output = transform(text)
    else:
        output = sanitize_text(
            text,
            replacements=replacements,
            strip_ansi=strip_ansi,
            line_filter=line_filter,
            strip_result=strip_result,
        )
    write_text(dst_path, output)
    return True


def copy_temp_log_file(
    src_path: str,
    dst_path: str,
    *,
    temp_paper_dir: str,
    temp_root: str = "",
    extra_replacements: Sequence[Tuple[str, str]] = (),
    strip_ansi: bool = False,
    line_filter: Optional[Callable[[str], bool]] = None,
    strip_result: bool = True,
) -> bool:
    return copy_sanitized_text_file(
        src_path,
        dst_path,
        transform=build_temp_log_transform(
            temp_paper_dir=temp_paper_dir,
            temp_root=temp_root,
            extra_replacements=extra_replacements,
            strip_ansi=strip_ansi,
            line_filter=line_filter,
            strip_result=strip_result,
        ),
    )


def write_completed_process_log(
    dst_path: str,
    completed: Any,
    *,
    transform: Optional[Callable[[str], str]] = None,
) -> None:
    stdout_text = str(getattr(completed, "stdout", "") or "")
    stderr_text = str(getattr(completed, "stderr", "") or "")
    if transform is not None:
        stdout_text = transform(stdout_text)
        stderr_text = transform(stderr_text)

    sections = [
        f"returncode={getattr(completed, 'returncode', '')}",
        "",
        "STDOUT:",
        stdout_text.rstrip(),
        "",
        "STDERR:",
        stderr_text.rstrip(),
        "",
    ]
    write_text(dst_path, "\n".join(sections))


def build_temp_paper_dir_from_map(
    temp_root: str,
    paper_dir: str,
    file_map: Mapping[str, str],
) -> str:
    temp_paper_dir = os.path.join(temp_root, os.path.basename(paper_dir))
    ensure_dir(temp_paper_dir)
    for relative_dst, src_path in file_map.items():
        if not src_path or not os.path.exists(src_path):
            continue
        dst_path = os.path.join(temp_paper_dir, *normalize_relpath(relative_dst).split("/"))
        ensure_dir(os.path.dirname(dst_path))
        shutil.copy2(src_path, dst_path)
    return temp_paper_dir


def move_paper_dir_to_sibling_root(
    paper_dir: str,
    *,
    sibling_suffix: str = "_no_property",
    log_path: Optional[str] = None,
    reason: str = "",
) -> Optional[str]:
    if not paper_dir or not os.path.isdir(paper_dir):
        append_log_line(log_path, "[MOVE_NO_PROPERTY_SKIP] reason=paper_dir_missing")
        return None

    mining_root = os.path.dirname(paper_dir.rstrip("/\\"))
    mining_parent = os.path.dirname(mining_root.rstrip("/\\"))
    mining_name = os.path.basename(mining_root.rstrip("/\\"))
    dest_root = os.path.join(mining_parent, f"{mining_name}{sibling_suffix}")
    ensure_dir(dest_root)

    folder_name = os.path.basename(paper_dir.rstrip("/\\"))
    dest_dir = os.path.join(dest_root, folder_name)
    if os.path.exists(dest_dir):
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir = os.path.join(dest_root, f"{folder_name}__moved_{suffix}")

    shutil.move(paper_dir, dest_dir)
    append_log_line(
        log_path,
        f"[MOVE_TO_NO_PROPERTY] src={relative_path(paper_dir, mining_root)} "
        f"dst={relative_path(dest_dir, mining_root)} reason={reason}",
    )
    return dest_dir


def stage_markdown_path(paper_dir: str, stage_dir: str, paper_id: str, kind: str = "main") -> str:
    directory = os.path.join(paper_dir, "property", stage_dir)
    if kind == "main":
        file_name = f"{paper_id}.md"
    else:
        file_name = f"{paper_id}_{kind}.md"
    return os.path.join(directory, file_name)


def stage_markdown_paths(
    paper_dir: str,
    stage_dir: str,
    paper_id: str,
    kinds: Sequence[str] = ("main", "vs", "app"),
) -> Dict[str, str]:
    return {
        kind: stage_markdown_path(paper_dir, stage_dir, paper_id, kind=kind)
        for kind in kinds
    }


def parse_sids_text(raw: str) -> List[int]:
    if not str(raw).strip():
        return []
    return [int(value) for value in re.split(r"[,\s]+", str(raw)) if value.strip().isdigit()]


def normalize_property_sample_name(
    header: str,
    *,
    blank_series_level: bool = False,
    extra_suffixes: Sequence[str] = (),
) -> Optional[str]:
    sample = str(header or "").strip()
    if not sample:
        return None
    lowered = sample.lower()
    if lowered.startswith("step "):
        return None
    if blank_series_level and sample == "(series-level)":
        return ""

    suffixes = (
        "(APP)",
        "(MAIN)",
        "(VS)",
        "(VS-REFINED)",
        "(VS-UNRESOLVED)",
        *tuple(extra_suffixes),
    )
    while sample:
        removed = False
        upper_sample = sample.upper()
        for suffix in suffixes:
            if not suffix:
                continue
            if upper_sample.endswith(str(suffix).upper()):
                sample = sample[: -len(str(suffix))].rstrip()
                removed = True
                break
        if not removed:
            break
    return sample or None


def normalize_sample_header(header: str) -> Optional[str]:
    return normalize_property_sample_name(header)


def strip_property_tag_prefix(tag: str) -> str:
    return _BRACKET_PREFIX_RE.sub("", str(tag or "")).strip()


def is_supported_property_tag(tag: str) -> bool:
    return strip_property_tag_prefix(tag) in _SUPPORTED_PROPERTY_TAG_SET


def parse_property_markdown(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    lines = read_text(path).splitlines()
    entries: List[Dict[str, Any]] = []
    current_sample: Optional[str] = None
    current_entry: Optional[Dict[str, Any]] = None
    state = "idle"

    def flush_entry() -> None:
        nonlocal current_entry
        if current_entry is None:
            return
        if current_entry["tag_sentences"]:
            entries.append(current_entry)
        current_entry = None

    for raw_line in lines:
        stripped = raw_line.strip()

        if stripped.startswith("# "):
            flush_entry()
            sample = normalize_sample_header(stripped[2:])
            if sample is not None:
                current_sample = sample
            state = "idle"
            continue

        if stripped.startswith("## "):
            flush_entry()
            sample = normalize_sample_header(stripped[3:])
            if sample is not None:
                current_sample = sample
            state = "idle"
            continue

        match = _PROPERTY_ENTRY_RE.match(stripped)
        if match:
            flush_entry()
            current_entry = {
                "sample": current_sample or "",
                "para_id": int(match.group(2)),
                "win_level": match.group(3).strip(),
                "window_sids": parse_sids_text(match.group(4)),
                "evidence_lines": [],
                "tag_sentences": [],
            }
            state = "after_header"
            continue

        if current_entry is None:
            continue

        lowered = stripped.lower()
        if lowered.startswith("evidence:"):
            state = "evidence"
            continue
        if lowered.startswith("cleaned properties:") or lowered.startswith("property abstract:") or lowered.startswith("refined properties:"):
            state = "properties"
            continue
        if not stripped:
            if state == "properties":
                flush_entry()
                state = "idle"
            continue

        if state == "evidence":
            current_entry["evidence_lines"].append(raw_line.rstrip("\n"))
            continue
        if state == "properties" and ":" in raw_line:
            tag, sentence = raw_line.split(":", 1)
            tag = strip_property_tag_prefix(tag)
            sentence = sentence.strip()
            if tag and sentence:
                current_entry["tag_sentences"].append((tag, sentence))

    flush_entry()
    return [entry for entry in entries if entry.get("sample")]


def property_tag_sort_key(tag: str) -> Tuple[int, str]:
    try:
        return (PROPERTY_TAG_ORDER.index(tag), tag)
    except ValueError:
        return (len(PROPERTY_TAG_ORDER), tag)


def build_property_item(
    *,
    sample: str,
    para_id: int,
    win_level: str,
    window_sids: Sequence[int],
    evidence_lines: Sequence[str],
    tag: str,
    sentence: str,
    tag_order: int = 0,
) -> Dict[str, Any]:
    return {
        "sample": sample,
        "para_id": int(para_id),
        "win_level": str(win_level).strip(),
        "window_sids": [int(value) for value in window_sids],
        "evidence_lines": [str(value) for value in evidence_lines],
        "tag": str(tag).strip(),
        "sentence": str(sentence).strip(),
        "tag_order": int(tag_order),
    }


def flatten_property_entries(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for entry in entries:
        sample = str(entry.get("sample", "")).strip()
        para_id = int(entry.get("para_id", 0) or 0)
        win_level = str(entry.get("win_level", "") or "").strip()
        window_sids = [int(value) for value in entry.get("window_sids", []) if str(value).isdigit()]
        evidence_lines = [str(value) for value in entry.get("evidence_lines", [])]
        for tag_order, (tag, sentence) in enumerate(entry.get("tag_sentences", []) or []):
            tag = str(tag).strip()
            sentence = str(sentence).strip()
            if not sample or not tag or not sentence:
                continue
            items.append(
                build_property_item(
                    sample=sample,
                    para_id=para_id,
                    win_level=win_level,
                    window_sids=window_sids,
                    evidence_lines=evidence_lines,
                    tag=tag,
                    sentence=sentence,
                    tag_order=tag_order,
                )
            )
    return items


def render_property_markdown(
    items_by_sample: Dict[str, List[Dict[str, Any]]],
    *,
    title: Optional[str] = None,
    property_label: str = "Property abstract",
    blank_sample_label: str = "(series-level)",
) -> str:
    if not items_by_sample:
        return ""

    lines: List[str] = []
    if title:
        lines.extend([f"# {title}", ""])

    first_sample = True
    for sample in sorted(items_by_sample.keys(), key=lambda value: str(value).lower()):
        items = items_by_sample.get(sample, []) or []
        if not items:
            continue
        if not first_sample:
            lines.append("")
        first_sample = False
        header_prefix = "##" if title else "#"
        lines.append(f"{header_prefix} {sample or blank_sample_label}")
        lines.append("")

        block_map: Dict[Tuple[int, str, Tuple[int, ...]], List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            block_map[
                (
                    int(item.get("para_id", 0) or 0),
                    str(item.get("win_level", "") or "").strip(),
                    tuple(item.get("window_sids", []) or []),
                )
            ].append(item)

        for block_index, key in enumerate(sorted(block_map.keys(), key=lambda value: (value[0], value[1], value[2])), start=1):
            para_id, win_level, sids_tuple = key
            block_items = sorted(
                block_map[key],
                key=lambda item: (
                    int(item.get("tag_order", 0) or 0),
                    property_tag_sort_key(str(item.get("tag", ""))),
                ),
            )
            sids_text = ",".join(str(value) for value in sids_tuple)
            lines.append(f"{block_index}. [para={para_id}; window={win_level}; sids={sids_text}]")
            lines.append("Evidence:")
            evidence_lines = block_items[0].get("evidence_lines", []) or []
            if evidence_lines:
                lines.extend(str(value) for value in evidence_lines)
            else:
                lines.append("")
            lines.append(f"{property_label}:")
            for item in block_items:
                tag = str(item.get("tag", "")).strip()
                sentence = str(item.get("sentence", "")).strip()
                if tag and sentence:
                    lines.append(f"{tag}: {sentence}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def normalize_property_markdown_file(
    src_path: str,
    dst_path: str,
    *,
    title: Optional[str] = None,
    property_label: str = "Property abstract",
    blank_sample_label: str = "(series-level)",
    supported_tags_only: bool = False,
    skip_blank_samples: bool = False,
) -> bool:
    if not os.path.exists(src_path):
        return False

    entries = parse_property_markdown(src_path)
    if not entries:
        raw_text = read_text(src_path)
        write_text(dst_path, raw_text if raw_text.strip() else "")
        return True

    items_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in flatten_property_entries(entries):
        tag = str(item.get("tag", "")).strip()
        if supported_tags_only and not is_supported_property_tag(tag):
            continue
        sample_name = str(item.get("sample", "")).strip()
        if skip_blank_samples and not sample_name:
            continue
        items_by_sample[sample_name].append(item)

    write_text(
        dst_path,
        render_property_markdown(
            items_by_sample,
            title=title,
            property_label=property_label,
            blank_sample_label=blank_sample_label,
        ),
    )
    return True


@dataclass(frozen=True)
class SampleInfo:
    name: str
    desc: str


def _cell_ok(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return False
    return text.upper() not in {"N/A", "NA", "NONE", "NULL", "-", "--"}


def _cell_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key, "")
    return str(value).strip() if _cell_ok(value) else ""


def build_cd_description_from_row(row: Mapping[str, Any]) -> str:
    name = _cell_text(row, "CDs_Naming_in_Paper")
    subject = f'Carbon dots labeled "{name}"' if name else "The carbon dots"

    method = _cell_text(row, "Synthesis_Method")
    temperature = _cell_text(row, "Temperature")
    duration = _cell_text(row, "Time")
    microwave_power = _cell_text(row, "Microwave_Power")
    precursor = _cell_text(row, "Precursor")
    precursor_amount = _cell_text(row, "Precursor_Amount")
    solvent = _cell_text(row, "Solvent")
    solvent_volume = _cell_text(row, "Solvent_Volume")
    ph_value = _cell_text(row, "pH")
    purification = _cell_text(row, "Purification")

    details = [
        method,
        temperature,
        duration,
        microwave_power,
        precursor,
        precursor_amount,
        solvent,
        solvent_volume,
        ph_value,
        purification,
    ]
    if not any(details):
        return (
            f'{subject} were prepared; the synthesis table did not provide usable preparation details '
            "for this sample."
        )

    parts = [f"{subject} were synthesized"]
    if method:
        if re.search(r"\b(method|route|process|approach)\b", method, flags=re.IGNORECASE):
            parts.append(f"using {method}")
        else:
            parts.append(f"by {method}")
    if precursor:
        parts.append(f"from {precursor}")
    if precursor_amount:
        parts.append(f"with precursor amount {precursor_amount}")
    if solvent:
        parts.append(f"in {solvent}")
    if solvent_volume:
        parts.append(f"(solvent volume {solvent_volume})")

    condition_bits: List[str] = []
    if temperature:
        condition_bits.append(f"at {temperature}")
    if duration:
        condition_bits.append(f"for {duration}")
    if microwave_power:
        condition_bits.append(f"under microwave power {microwave_power}")
    if ph_value:
        condition_bits.append(f"with pH {ph_value}")
    if condition_bits:
        parts.append(", ".join(condition_bits))

    if purification:
        parts.append(f"and purified by {purification}")

    sentence = " ".join(part for part in parts if part).strip()
    return sentence if sentence.endswith(".") else sentence + "."


def read_letter_table_samples(paper_dir: str, paper_id: str) -> List[SampleInfo]:
    csv_path = os.path.join(paper_dir, "Synthesis", "letter_table", f"{paper_id}.csv")
    df = safe_read_csv(csv_path)
    if df is None or df.empty or "CDs_Naming_in_Paper" not in df.columns:
        return []

    samples: List[SampleInfo] = []
    seen = set()
    for _, row in df.iterrows():
        name = _cell_text(row.to_dict(), "CDs_Naming_in_Paper")
        if not name or name in seen:
            continue
        seen.add(name)
        samples.append(SampleInfo(name=name, desc=build_cd_description_from_row(row.to_dict())))
    return samples


def normalize_name_signature(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def resolve_decision_csv_path(paper_dir: str, paper_id: str) -> str:
    candidates = (
        os.path.join(paper_dir, "property", "decision_LLM", f"{paper_id}.csv"),
        os.path.join(paper_dir, "property", "LLM_decision_separate", f"{paper_id}.csv"),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def build_decision_sid_maps(csv_path: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = read_csv_utf8_fallback(csv_path)
    if "sent_global_id" not in df.columns:
        raise ValueError("Missing column 'sent_global_id' in decision csv.")
    if "text" not in df.columns:
        raise ValueError("Missing column 'text' in decision csv.")

    if "merge_LLM_name" in df.columns:
        name_column = "merge_LLM_name"
    elif "LLM_name" in df.columns:
        name_column = "LLM_name"
    else:
        name_column = ""

    sid_to_names: Dict[int, str] = {}
    sid_to_text: Dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            sent_id = int(row["sent_global_id"])
        except Exception:
            continue
        text = "" if pd.isna(row["text"]) else str(row["text"]).strip()
        names = ""
        if name_column:
            raw_names = row.get(name_column, "")
            names = "" if pd.isna(raw_names) else str(raw_names).strip()
        sid_to_names[sent_id] = names
        sid_to_text[sent_id] = text
    return sid_to_names, sid_to_text


def split_names_cell(value: Any) -> List[str]:
    if value is None:
        return []
    parsed = loads_json_field(value, None)
    if isinstance(parsed, list):
        return dedupe_preserve_order(str(item).strip() for item in parsed if str(item).strip())
    text = str(value).strip().strip("[]")
    if not text:
        return []
    return dedupe_preserve_order(
        part.strip().strip("'\"")
        for part in re.split(r"[|;,/\n]+", text)
        if part.strip().strip("'\"")
    )


def candidates_from_sids(
    window_sids: Sequence[int],
    sid_to_names: Mapping[int, str],
    letter_samples: Sequence[SampleInfo],
) -> List[SampleInfo]:
    if not letter_samples:
        return []
    if not sid_to_names:
        return list(letter_samples)

    matched_signatures = set()
    for value in window_sids:
        try:
            sent_id = int(value)
        except Exception:
            continue
        for name in split_names_cell(sid_to_names.get(sent_id, "")):
            signature = normalize_name_signature(name)
            if signature:
                matched_signatures.add(signature)

    if not matched_signatures:
        return list(letter_samples)

    subset = [
        sample for sample in letter_samples
        if normalize_name_signature(sample.name) in matched_signatures
    ]
    return subset if len(subset) >= 2 else list(letter_samples)


def build_evidence_lines_from_sids(
    window_sids: Sequence[int],
    sid_to_text: Mapping[int, str],
    fallback_lines: Optional[Sequence[str]] = None,
) -> List[str]:
    lines: List[str] = []
    seen = set()
    for value in window_sids or []:
        try:
            sent_id = int(value)
        except Exception:
            continue
        if sent_id in seen:
            continue
        seen.add(sent_id)
        text = str(sid_to_text.get(sent_id, "") or "").strip()
        if text:
            lines.append(text)
    if lines:
        return lines
    return [str(line).strip() for line in (fallback_lines or []) if str(line).strip()]


def bundle_exists(out_dir: str, paper_id: str, require_md_txt: bool = True) -> bool:
    csv_path = os.path.join(out_dir, f"{paper_id}.csv")
    if not os.path.exists(csv_path):
        return False
    if not require_md_txt:
        return True
    return all(
        os.path.exists(os.path.join(out_dir, f"{paper_id}.{ext}"))
        for ext in ("md", "txt")
    )


def annotation_bundle_paths(paper_dir: str, subpath: str, paper_id: str) -> Dict[str, str]:
    out_dir = os.path.join(paper_dir, *subpath.split("/"))
    return {
        "out_dir": out_dir,
        "csv": os.path.join(out_dir, f"{paper_id}.csv"),
        "md": os.path.join(out_dir, f"{paper_id}.md"),
        "txt": os.path.join(out_dir, f"{paper_id}.txt"),
    }


def loads_json_field(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if pd.isna(value) if not isinstance(value, (list, dict, str)) else False:
        return default
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def read_csv_utf8_fallback(path: str) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    raise ValueError("Failed to read CSV with utf-8 fallback.")


def json_list_safe(value: Any) -> List[Any]:
    arr = loads_json_field(value, [])
    return arr if isinstance(arr, list) else []


def safe_paper_title(df: pd.DataFrame, fallback: str) -> str:
    if "pdf_name" not in df.columns or df.empty:
        return fallback
    title = str(df.iloc[0].get("pdf_name", fallback)).strip()
    return title or fallback


def write_annotation_bundle(
    df: pd.DataFrame,
    csv_path: str,
    md_path: str,
    txt_path: str,
    paper_title: str,
) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    with open(md_path, "w", encoding="utf-8") as f_md:
        f_md.write(f"# Annotations for {paper_title}\n\n")
        for _, row in df.iterrows():
            gid = row.get("sent_global_id", "")
            block = row.get("block_id", "")
            sec = row.get("main_section_norm", "")
            props_kw = json_list_safe(row.get("prop_kw_hits"))
            props_win = json_list_safe(row.get("prop_window_hits"))
            props_cand = json_list_safe(row.get("cand_props"))
            nums = loads_json_field(row.get("numbers_units"), [])
            nums_str = "; ".join(
                f"{entry.get('value_str', '')} {entry.get('unit', '')}".strip()
                for entry in nums
            ) if isinstance(nums, list) else ""
            text = str(row.get("text", "")).replace("\n", " ").strip()
            f_md.write(
                f"[Block {block}, Sec: {sec}, gid: {gid}, "
                f"prop_kw_hits: {', '.join(props_kw)}, "
                f"prop_window_hits: {', '.join(props_win)}, "
                f"cand_props: {', '.join(props_cand)}, numbers: {nums_str}]  \n"
            )
            f_md.write(f"{text}  \n\n---\n\n")

    with open(txt_path, "w", encoding="utf-8") as f_txt:
        for _, row in df.iterrows():
            gid = row.get("sent_global_id", "")
            block = row.get("block_id", "")
            sec = row.get("main_section_norm", "")
            props_kw = json_list_safe(row.get("prop_kw_hits"))
            props_win = json_list_safe(row.get("prop_window_hits"))
            props_cand = json_list_safe(row.get("cand_props"))
            nums = loads_json_field(row.get("numbers_units"), [])
            nums_str = "; ".join(
                f"{entry.get('value_str', '')} {entry.get('unit', '')}".strip()
                for entry in nums
            ) if isinstance(nums, list) else ""
            text = str(row.get("text", "")).replace("\n", " ").strip()
            f_txt.write(
                f"[Block {block}, Sec: {sec}, gid: {gid}, "
                f"prop_kw_hits: {', '.join(props_kw)}, "
                f"prop_window_hits: {', '.join(props_win)}, "
                f"cand_props: {', '.join(props_cand)}, numbers: {nums_str}]\n"
            )
            f_txt.write(f"{text}\n\n---\n\n")


def copy_annotation_bundle(
    src_dir: str,
    dst_dir: str,
    paper_id: str,
    overwrite: bool = False,
) -> bool:
    os.makedirs(dst_dir, exist_ok=True)
    copied = False
    for ext in ("csv", "md", "txt"):
        src = os.path.join(src_dir, f"{paper_id}.{ext}")
        dst = os.path.join(dst_dir, f"{paper_id}.{ext}")
        if not os.path.exists(src):
            continue
        if os.path.exists(dst) and not overwrite:
            continue
        shutil.copy2(src, dst)
        copied = True
    return copied


_WS = r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u3000]"
_ZERO_WIDTH = r"[\u200B\u200C\u200D\u2060\ufeff]"


def normalize_for_match(text: str) -> str:
    if not text:
        return text
    out = unicodedata.normalize("NFKC", text)
    out = re.sub(_ZERO_WIDTH, "", out)
    out = re.sub(_WS, " ", out)
    out = re.sub(r"[\u2010-\u2015\u2212]", "-", out)
    out = out.replace("％", "%")
    out = out.replace("\u00B5", "μ").replace("µ", "μ")
    out = re.sub(r"(?<=\d),(?=\d)", ".", out)
    out = re.sub(r"\bper[\s-]*cent\b", "percent", out, flags=re.I)
    out = re.sub(r"(?:λ|lambda)\s*[_\s-]*(ex|em)\b", r"λ_\1", out, flags=re.I)
    out = re.sub(r"\s+", " ", out).strip()
    return out


PAT_PERCENT_NUM = re.compile(r"(?P<val>\d{1,3}(?:\.\d+)?)\s*%", re.I | re.U)
PAT_PERCENT_WORD = re.compile(r"(?P<val>\d{1,3}(?:\.\d+)?)\s*percent", re.I | re.U)
PAT_NM_RANGE = re.compile(
    r"(?<![A-Za-z])(?P<val>\d{1,4}(?:\.\d+)?)\s*[-–—]\s*(?P<val2>\d{1,4}(?:\.\d+)?)\s*nm(?![A-Za-z])",
    re.I | re.U,
)
PAT_NM_SCALAR = re.compile(r"(?<![A-Za-z])(?P<val>\d{1,4}(?:\.\d+)?)\s*nm(?![A-Za-z])", re.I | re.U)
PAT_UM_SCALAR = re.compile(r"(?<![A-Za-z])(?P<val>\d{1,4}(?:\.\d+)?)\s*(?:μm|um)(?![A-Za-z])", re.I | re.U)
PAT_TIME_NSUSMS = re.compile(r"\b(?P<val>\d+(?:\.\d+)?)\s*(?P<u>ns|μs|us|ms)\b", re.I | re.U)
PAT_TIME_S = re.compile(r"\b(?P<val>\d+(?:\.\d+)?)\s*s(?:ec|econd)?s?\b", re.I | re.U)
NUM_TOKEN_RE = re.compile(r"\d[\d\.,]*[A-Za-z]?")

PAT_EM_WORDS = re.compile(
    r"("
    r"\bemission(?:\s+peak)?\b"
    r"|\bpeak\s+emission\b"
    r"|\bemission\s+wavelength\b"
    r"|\b(?:λ|lambda)\s*[_\s-]*em\b"
    r"|\bfluorescence\s+intensity\s+(?:was\s+observed\s+)?at\b"
    r"|\bmax(?:\.|imum)?\s*(?:pl|photoluminescence|emission)\s*(?:peak|wavelength)?\b"
    r"|\boptimal\s+(?:pl|photoluminescence|emission)\s*(?:peak|wavelength)\b"
    r")",
    re.I | re.U,
)
PAT_EX_WORDS = re.compile(
    r"("
    r"\bexcitation\b"
    r"|\bexcited\s+at\b"
    r"|\b(?:λ|lambda)\s*[_\s-]*ex\b"
    r"|\bexcitation\s+wavelengths?\s+of\b"
    r"|\bunder\s+(?:uv|laser)\s*\(?\d{2,4}\s*nm\)?"
    r"|\bmax(?:\.|imum)?\s*excitation\s*(?:wavelength|peak)?\b"
    r"|\boptimal\s+excitation\s+(?:wavelength|peak)\b"
    r")",
    re.I | re.U,
)
PAT_QY_WORDS = re.compile(
    r"(\bquantum\s+(yield|efficiency)\b|\bPLQY\b|\bQY\b|\bPhi\b|\bΦ\s*F\b)",
    re.I | re.U,
)
PAT_LIFETIME_WORDS = re.compile(
    r"(\blifetime\b|\bdecay(?:\s*time)?\b|\bfluorescence\s+lifetime\b|\bphosphorescence\s+lifetime\b|τ|\btau\b)",
    re.I | re.U,
)
PAT_EXDEP_POS = re.compile(
    r"(\bexcitation-?dependent\b|\bdependent\s+on\s+excitation\b|\bred-?shift(?:s|ed)?\b|\bblue-?shift(?:s|ed)?\b)",
    re.I | re.U,
)
PAT_EXDEP_NEG = re.compile(
    r"(\bexcitation-?independent\b|\bdoes\s+not\s+shift\b|\bremains\s+unchanged\s+with\s+excitation\b|\bindependent\s+of\s+excitation\b)",
    re.I | re.U,
)
PAT_CHIRAL = re.compile(
    r"(\bchiral(?:ity)?\b|\benantiomer(?:ic)?\b|\bL-[A-Za-z0-9\-]+?\s*CDs\b|\bD-[A-Za-z0-9\-]+?\s*CDs\b)",
    re.I | re.U,
)
PAT_CPL_CORE = re.compile(
    r"(\bCPL\b|\bcircularly\s+polarized\s+luminescence\b|\bg[\s_\-]?lum\b|\bglum\b|\bg-?factor\b)",
    re.I | re.U,
)
PAT_CPL_LUM = re.compile(r"(\bluminescen\w*\b|\bemission\b|\bPL\b)", re.I | re.U)


def count_tokens(sent: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", sent or ""))


def _append_pair(
    pairs: List[Dict[str, str]],
    seen_pairs: set,
    value_str: str,
    unit: str,
) -> None:
    key = (value_str, unit)
    if key in seen_pairs:
        return
    pairs.append({"value_str": value_str, "unit": unit})
    seen_pairs.add(key)


def extract_numbers_units(sent: str) -> Dict[str, Any]:
    units: List[str] = []
    seen_units: set = set()
    pairs: List[Dict[str, str]] = []
    seen_pairs: set = set()
    seen_raw_values: set = set()

    def mark_unit(unit: str) -> None:
        if unit not in seen_units:
            units.append(unit)
            seen_units.add(unit)

    for match in PAT_NM_RANGE.finditer(sent):
        value = f"{match.group('val')}-{match.group('val2')}"
        mark_unit("nm")
        _append_pair(pairs, seen_pairs, value, "nm")
        seen_raw_values.add(value)

    for match in PAT_NM_SCALAR.finditer(sent):
        value = match.group("val")
        mark_unit("nm")
        _append_pair(pairs, seen_pairs, value, "nm")
        seen_raw_values.add(value)

    for match in PAT_UM_SCALAR.finditer(sent):
        value = match.group("val")
        mark_unit("μm")
        _append_pair(pairs, seen_pairs, value, "μm")
        seen_raw_values.add(value)

    for match in PAT_PERCENT_NUM.finditer(sent):
        value = match.group("val")
        mark_unit("%")
        _append_pair(pairs, seen_pairs, value, "%")
        seen_raw_values.add(value)

    for match in PAT_PERCENT_WORD.finditer(sent):
        value = match.group("val")
        mark_unit("%")
        _append_pair(pairs, seen_pairs, value, "%")
        seen_raw_values.add(value)

    for match in PAT_TIME_NSUSMS.finditer(sent):
        value = match.group("val")
        unit = match.group("u").lower().replace("us", "μs")
        mark_unit(unit)
        _append_pair(pairs, seen_pairs, value, unit)
        seen_raw_values.add(value)

    for match in PAT_TIME_S.finditer(sent):
        value = match.group("val")
        mark_unit("s")
        _append_pair(pairs, seen_pairs, value, "s")
        seen_raw_values.add(value)

    low = sent.lower()
    if "nanometer" in low or "nanometre" in low or re.search(r"\bnm\b", low):
        mark_unit("nm")
    if re.search(r"\b(μm|um)\b", low):
        mark_unit("μm")
    if "%" in sent or "percent" in low or "percentage" in low:
        mark_unit("%")
    if re.search(r"\bns\b", low) or "nanosecond" in low:
        mark_unit("ns")
    if re.search(r"\b(μs|us)\b", low) or "microsecond" in low:
        mark_unit("μs")
    if re.search(r"\bms\b", low) or "millisecond" in low:
        mark_unit("ms")
    if re.search(r"\bs\b", low) or "second" in low:
        mark_unit("s")

    for match in NUM_TOKEN_RE.finditer(sent):
        value = match.group(0)
        if value in seen_raw_values:
            continue
        _append_pair(pairs, seen_pairs, value, "")
        seen_raw_values.add(value)

    return {"units_mask": units, "numbers_units": pairs}


def detect_keywords(sent: str) -> Dict[str, Any]:
    pos = bool(PAT_EXDEP_POS.search(sent))
    neg = bool(PAT_EXDEP_NEG.search(sent))
    return {
        "has_em": bool(PAT_EM_WORDS.search(sent)),
        "has_ex": bool(PAT_EX_WORDS.search(sent)),
        "has_qy": bool(PAT_QY_WORDS.search(sent)),
        "has_lifetime": bool(PAT_LIFETIME_WORDS.search(sent)),
        "has_exdep": "pos" if pos else ("neg" if neg else False),
        "has_chiral": bool(PAT_CHIRAL.search(sent)),
        "has_cpl_core": bool(PAT_CPL_CORE.search(sent)),
        "has_cpl_lum": bool(PAT_CPL_LUM.search(sent)),
    }


def annotate_sentences(
    sent_list: Sequence[str],
    para_keys: Optional[Sequence[str]] = None,
    window_size: int = DEFAULT_PROPERTY_WINDOW,
    return_full: bool = True,
) -> List[Dict[str, Any]]:
    norm_sents = [normalize_for_match(text) for text in sent_list]
    nums_seq = [extract_numbers_units(text) for text in norm_sents]
    kw_seq = [detect_keywords(text) for text in norm_sents]
    n = len(sent_list)

    def window_indices(i: int) -> List[int]:
        lo = max(0, i - window_size)
        hi = min(n - 1, i + window_size)
        indices = list(range(lo, hi + 1))
        if para_keys is None:
            return indices
        para_key = para_keys[i]
        return [j for j in indices if para_keys[j] == para_key]

    results: List[Dict[str, Any]] = []
    for i in range(n):
        norm = norm_sents[i]
        if count_tokens(norm) < 3:
            empty = {"prop_kw_hits": [], "prop_window_hits": [], "cand_props": []}
            if return_full:
                empty.update({"units_mask": [], "numbers_units": []})
            results.append(empty)
            continue

        hits: List[str] = []
        row_kw = kw_seq[i]
        if row_kw["has_em"]:
            hits.append("Em")
        if row_kw["has_ex"]:
            hits.append("Ex")
        if row_kw["has_qy"]:
            hits.append("QY")
        if row_kw["has_lifetime"]:
            hits.append("lifetime")
        if row_kw["has_exdep"]:
            hits.append("ExDep")
        if row_kw["has_chiral"]:
            hits.append("Chiral")
        if row_kw["has_cpl_core"]:
            hits.append("CPL")

        prop_kw_hits = sorted(set(hits))
        units_mask_sent = nums_seq[i]["units_mask"]
        numbers_units_sent = nums_seq[i]["numbers_units"]
        seed = bool(prop_kw_hits) or bool(units_mask_sent)

        cand_props: List[str] = []
        if seed:
            idxs = window_indices(i)
            units_mask_win: List[str] = []
            seen_units: set = set()
            for j in idxs:
                for unit in nums_seq[j]["units_mask"]:
                    if unit not in seen_units:
                        units_mask_win.append(unit)
                        seen_units.add(unit)

            has_em = any(kw_seq[j]["has_em"] for j in idxs)
            has_ex = any(kw_seq[j]["has_ex"] for j in idxs)
            has_qy = any(kw_seq[j]["has_qy"] for j in idxs)
            has_life = any(kw_seq[j]["has_lifetime"] for j in idxs)
            has_exdep = next((kw_seq[j]["has_exdep"] for j in idxs if kw_seq[j]["has_exdep"]), False)
            has_chiral = any(kw_seq[j]["has_chiral"] for j in idxs)
            has_cpl_core = any(kw_seq[j]["has_cpl_core"] for j in idxs)
            has_cpl_lum = any(kw_seq[j]["has_cpl_lum"] for j in idxs)

            if has_em and any(unit in units_mask_win for unit in ("nm", "μm")):
                cand_props.append("Em")
            if has_ex and any(unit in units_mask_win for unit in ("nm", "μm")):
                cand_props.append("Ex")
            if has_qy and "%" in units_mask_win:
                cand_props.append("QY")
            if has_life and any(unit in units_mask_win for unit in ("ns", "μs", "ms", "s")):
                cand_props.append("lifetime")
            if has_exdep:
                cand_props.append("ExDep")
            if has_chiral:
                cand_props.append("Chiral")
            if has_cpl_core and has_cpl_lum:
                cand_props.append("CPL")

        result = {
            "prop_kw_hits": prop_kw_hits,
            "prop_window_hits": sorted(set(cand_props)),
            "cand_props": sorted(set(cand_props)),
        }
        if return_full:
            result.update(
                {
                    "units_mask": sorted(set(units_mask_sent)),
                    "numbers_units": numbers_units_sent,
                }
            )
        results.append(result)

    return results


def derive_para_keys(df: pd.DataFrame) -> List[str]:
    if "block_id" in df.columns and "para_id_in_block" in df.columns:
        return (
            df["block_id"].astype(str) + ":" + df["para_id_in_block"].astype(str)
        ).tolist()
    if "para_global_id" in df.columns:
        return df["para_global_id"].astype(str).tolist()
    if "block_id" in df.columns:
        return df["block_id"].astype(str).tolist()
    return ["ALL"] * len(df)


def annotate_dataframe(
    df: pd.DataFrame,
    window_size: int = DEFAULT_PROPERTY_WINDOW,
    return_full: bool = True,
) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("Missing required column: text")

    out = df.copy()
    annotations = annotate_sentences(
        sent_list=out["text"].fillna("").astype(str).tolist(),
        para_keys=derive_para_keys(out),
        window_size=window_size,
        return_full=return_full,
    )
    out["prop_kw_hits"] = [
        json.dumps(entry["prop_kw_hits"], ensure_ascii=False) for entry in annotations
    ]
    out["prop_window_hits"] = [
        json.dumps(entry["prop_window_hits"], ensure_ascii=False) for entry in annotations
    ]
    out["cand_props"] = [
        json.dumps(entry["cand_props"], ensure_ascii=False) for entry in annotations
    ]
    if return_full:
        out["units_mask"] = [
            json.dumps(entry["units_mask"], ensure_ascii=False) for entry in annotations
        ]
        out["numbers_units"] = [
            json.dumps(entry["numbers_units"], ensure_ascii=False) for entry in annotations
        ]
    return out


def has_property_hits(row: pd.Series) -> bool:
    props = loads_json_field(row.get("prop_window_hits"), None)
    if props is None:
        props = loads_json_field(row.get("cand_props"), [])
    return bool(props)
