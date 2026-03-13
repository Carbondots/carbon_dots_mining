#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step 6: clean Step 5 refined property markdown with sentence-level LLM checks."""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from property_unit import (
    append_log_line as append_log,
    ensure_root_exists,
    iter_paper_dirs,
    move_paper_dir_to_sibling_root as move_paper_dir_to_no_property,
    normalize_property_sample_name,
    paper_id_from_dir,
    parse_boolean_answer,
    parse_sids_text as parse_sids,
    read_text,
    relative_path,
    remove_think_blocks as remove_think,
    strip_code_fences as strip_code_fence,
    write_text,
)

try:
    from lmstudio import llm as lmstudio_llm
except Exception:
    lmstudio_llm = None


DEFAULT_MODEL = "qwen.qwen2.5-vl-32b-instruct"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 200
END_SENTINEL = "<END_OF_JSON>"
ALLOWED_PROPERTY_TAGS = {"Ex", "Em", "QY", "lifetime", "ExDep", "Chiral", "CPL"}


@dataclass
class RefinedBlock:
    paper_id: str
    kind: str
    sample: str
    para_id: int
    win_level: str
    window_sids: List[int]
    evidence_text: str
    tags: List[str]
    props: Dict[str, str]


def parse_keep_from_llm(text: str) -> bool:
    return parse_boolean_answer(text, end_sentinel=END_SENTINEL)


def get_step6_log_dir(mining_root: str) -> str:
    log_dir = os.path.join(mining_root, "step6_logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_deleted_log_path(log_dir: str, kind: str) -> str:
    mapping = {
        "main": "cleaned_main_deleted.log",
        "vs": "cleaned_vs_deleted.log",
        "app": "cleaned_app_deleted.log",
    }
    return os.path.join(log_dir, mapping.get(kind, "cleaned_other_deleted.log"))


def get_empty_case_log_path(log_dir: str) -> str:
    return os.path.join(log_dir, "cleaned_empty_cases.log")


def write_run_headers(log_dir: str) -> None:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n=== RUN_AT\t{timestamp} ===\n"
    for kind in ("main", "vs", "app"):
        append_log(get_deleted_log_path(log_dir, kind), header.rstrip("\n"))
    append_log(get_empty_case_log_path(log_dir), header.rstrip("\n"))


def log_deleted_prop(log_dir: str, block: RefinedBlock, tag: str, sentence: str, reason: str) -> None:
    sids_text = ",".join(str(value) for value in block.window_sids)
    safe_sentence = sentence.replace("\t", " ").replace("\n", " ")
    line = (
        f"paper={block.paper_id}\tkind={block.kind}\tsample={block.sample}\t"
        f"para={block.para_id};window={block.win_level};sids={sids_text}\t"
        f"tag={tag}\treason={reason}\tsentence={safe_sentence}"
    )
    append_log(get_deleted_log_path(log_dir, block.kind), line)


def log_empty_case(
    log_dir: str,
    case_code: str,
    paper_dir: str,
    md_paths: Sequence[str],
    *,
    moved_to: Optional[str] = None,
    note: str = "",
) -> None:
    folder_name = os.path.basename(paper_dir.rstrip("/\\"))
    file_names = ",".join(os.path.basename(path) for path in md_paths)
    mining_root = os.path.dirname(paper_dir.rstrip("/\\"))
    parts = [
        f"case={case_code}",
        f"paper={folder_name}",
        f"files={file_names}",
    ]
    if moved_to:
        parts.append(f"moved_to={relative_path(moved_to, mining_root)}")
    if note:
        parts.append(f"note={note.replace(chr(9), ' ').replace(chr(10), ' ')}")
    append_log(get_empty_case_log_path(log_dir), "\t".join(parts))
def normalize_sample_header(text: str, kind: str) -> Optional[str]:
    suffixes = ()
    if kind == "vs":
        suffixes = ("(VS-unresolved)",)
    elif kind == "app":
        suffixes = ("(APP)",)
    return normalize_property_sample_name(
        text,
        blank_series_level=True,
        extra_suffixes=suffixes,
    )


def parse_refined_md(md_text: str, paper_id: str, kind: str) -> List[RefinedBlock]:
    blocks: List[RefinedBlock] = []
    lines = md_text.splitlines()
    total = len(lines)
    index = 0
    current_sample: Optional[str] = None
    entry_re = re.compile(
        r"^\s*\d+\.\s*\[para=(\d+);\s*(?:win|window)=([^;]+);\s*sids=([^\]]*)\]\s*$",
        re.IGNORECASE,
    )

    while index < total:
        stripped = lines[index].strip()

        if stripped.startswith("# "):
            current_sample = normalize_sample_header(stripped[2:], kind)
            index += 1
            continue

        if stripped.startswith("## "):
            current_sample = normalize_sample_header(stripped[3:], kind)
            index += 1
            continue

        match = entry_re.match(stripped)
        if not match or current_sample is None:
            index += 1
            continue

        para_id = int(match.group(1))
        win_level = match.group(2).strip()
        window_sids = parse_sids(match.group(3))
        index += 1

        evidence_text = ""
        if index < total and lines[index].strip().lower().startswith("evidence:"):
            index += 1
            evidence_lines: List[str] = []
            while index < total:
                probe = lines[index].strip()
                if not probe:
                    evidence_lines.append(lines[index].rstrip("\n"))
                    index += 1
                    continue
                if probe.lower().startswith("refined properties:") or probe.lower().startswith("property abstract:"):
                    break
                if probe.startswith("#") or entry_re.match(probe):
                    break
                evidence_lines.append(lines[index].rstrip("\n"))
                index += 1
            evidence_text = "\n".join(evidence_lines).strip()

        if index >= total:
            break

        if not (
            lines[index].strip().lower().startswith("refined properties:")
            or lines[index].strip().lower().startswith("property abstract:")
        ):
            index += 1
            continue
        index += 1

        tags: List[str] = []
        props: Dict[str, str] = {}
        while index < total:
            line = lines[index].rstrip("\n")
            stripped_prop = line.strip()
            if not stripped_prop:
                index += 1
                break
            if stripped_prop.startswith("#") or entry_re.match(stripped_prop):
                break
            match_prop = re.match(r"^\s*([A-Za-z]+)\s*:\s*(.*\S)\s*$", stripped_prop)
            if match_prop:
                tag = match_prop.group(1).strip()
                sentence = match_prop.group(2).strip()
                if tag not in props:
                    tags.append(tag)
                props[tag] = sentence
            index += 1

        if tags and props:
            blocks.append(
                RefinedBlock(
                    paper_id=paper_id,
                    kind=kind,
                    sample=current_sample or "",
                    para_id=para_id,
                    win_level=win_level,
                    window_sids=window_sids,
                    evidence_text=evidence_text,
                    tags=tags,
                    props=props,
                )
            )

    return blocks


def render_cleaned_md(blocks: Sequence[RefinedBlock], kind: str) -> str:
    if not blocks:
        return ""

    sample_map: Dict[str, List[RefinedBlock]] = {}
    for block in blocks:
        sample_map.setdefault(block.sample, []).append(block)

    parts = [f"# Step 6 Cleaned Properties ({kind})", ""]
    for sample_name in sorted(sample_map.keys(), key=lambda value: (value == "", str(value).lower())):
        label = sample_name if sample_name else "(series-level)"
        parts.append(f"## {label}")
        parts.append("")

        entries = sorted(
            sample_map[sample_name],
            key=lambda block: (block.para_id, min(block.window_sids) if block.window_sids else 0),
        )
        for entry_index, block in enumerate(entries, start=1):
            sids_text = ",".join(str(value) for value in block.window_sids)
            parts.append(f"{entry_index}. [para={block.para_id}; window={block.win_level}; sids={sids_text}]")
            parts.append("Evidence:")
            if block.evidence_text:
                parts.extend(block.evidence_text.splitlines())
            else:
                parts.append("")
            parts.append("Cleaned properties:")
            for tag in block.tags:
                sentence = block.props.get(tag, "").strip()
                if sentence:
                    parts.append(f"{tag}: {sentence}")
            parts.append("")

        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def property_spec(tag: str) -> str:
    specs = {
        "Ex": (
            "Ex means excitation wavelength used to excite the photoluminescence of the sample itself. "
            "The sentence should contain at least one explicit excitation wavelength value or a small set of "
            "discrete values. Do not accept pure UV-vis absorption peaks. Do not accept only scan ranges such as "
            "'300-600 nm' without a peak, maximum, or discrete excitation choices. Do not accept application-only "
            "instrument settings when they do not describe intrinsic photoluminescence characterization."
        ),
        "Em": (
            "Em means emission wavelength or emission band of the sample. The sentence should report at least one "
            "explicit emission wavelength value or several discrete emission values under different conditions. "
            "Do not accept UV-vis absorption information. Do not accept only scanned ranges without an emission peak."
        ),
        "QY": (
            "QY means photoluminescence quantum yield of the sample. The sentence should contain at least one "
            "explicit QY value, such as a percentage or a number between 0 and 1. Pure measurement settings are not "
            "enough. Do not accept only a broad possible range without clear measured values."
        ),
        "lifetime": (
            "lifetime means photoluminescence lifetime in ns, us, or ms. The sentence should contain at least one "
            "explicit lifetime value or several discrete lifetime components. Do not accept fitting windows or "
            "instrument settings without actual lifetime values."
        ),
        "ExDep": (
            "ExDep means excitation dependence of one sample. A valid sentence either directly states excitation "
            "dependence or excitation independence, or clearly says that emission wavelength or emission color shifts "
            "or remains unchanged as excitation wavelength changes. Do not accept changes caused by pH, concentration, "
            "temperature, reaction time, composition, dopant content, analytes, or comparisons between different samples."
        ),
    }
    return specs.get(tag, "Decide whether the sentence truly describes the requested property.")


def build_test_condition_prompt(sentence: str) -> str:
    return (
        "You are an expert in photoluminescence of carbon dots.\n"
        "Decide whether the sentence should be kept as a useful photoluminescence property sentence, or removed as a "
        "pure measurement or imaging setup sentence, or as a sentence that only describes composite host materials.\n\n"
        "Keep the sentence and return {true} when it reports at least one photoluminescence property of a carbon-dot "
        "sample, or when it mixes setup details with at least one real photoluminescence property.\n\n"
        "Remove the sentence and return {false} when all samples mentioned are composite host materials with no "
        "separate carbon-dot sample, or when the sentence is only setup information such as instruments, scan ranges, "
        "slit widths, microscope channels, or generic detection windows without a true property result.\n\n"
        "If uncertain, prefer keep only when the sentence contains an explicit photoluminescence parameter or behavior "
        "for a carbon-dot sample itself.\n\n"
        f"Sentence:\n{sentence}\n\n"
        "Output exactly {true} or {false} on one line, then on the next line output <END_OF_JSON>.\n"
    )


def build_tag_match_prompt(tag: str, sentence: str) -> str:
    return (
        "You are an expert in photoluminescence of carbon dots.\n"
        "Decide whether the sentence truly reports the requested property of the sample.\n\n"
        f"Property tag: {tag}\n"
        f"Definition:\n{property_spec(tag)}\n\n"
        f"Sentence:\n{sentence}\n\n"
        "Return {true} if the sentence clearly matches the property definition. Return {false} if it mainly describes "
        "a different property, only setup details, or does not really report this property.\n\n"
        "Output exactly {true} or {false} on one line, then on the next line output <END_OF_JSON>.\n"
    )


def call_llm_boolean(
    model,
    prompt: str,
    *,
    log_path: Optional[str],
    log_label: str,
    input_anchor: str,
    temperature: float,
    max_tokens: int,
) -> bool:
    try:
        response = model.respond(prompt, config={"temperature": temperature, "maxTokens": max_tokens})
    except Exception as exc:
        if log_path:
            append_log(log_path, f"=== {log_label} ERROR ===")
            append_log(log_path, f"ERROR: {repr(exc)}")
            append_log(log_path, "")
        raise

    raw = response.content if hasattr(response, "content") else str(response)
    cleaned = strip_code_fence(remove_think(raw))
    if not cleaned.strip():
        if log_path:
            append_log(log_path, f"=== {log_label} ERROR ===")
            append_log(log_path, "ERROR: empty output from LLM")
            append_log(log_path, "")
        return True

    if log_path:
        anchor_index = prompt.find(input_anchor)
        logged_input = prompt[anchor_index:] if anchor_index != -1 else prompt
        append_log(log_path, f"=== {log_label} ===")
        append_log(log_path, "INPUT:")
        append_log(log_path, logged_input)
        append_log(log_path, "OUTPUT:")
        append_log(log_path, cleaned.strip())
        append_log(log_path, "")

    return parse_keep_from_llm(cleaned)


def clean_block_with_llm(
    block: RefinedBlock,
    *,
    model,
    log_dir: str,
    test_log_path: Optional[str],
    tag_log_path: Optional[str],
    temperature: float,
    max_tokens: int,
) -> Optional[RefinedBlock]:
    if not block.tags or not block.props:
        return None

    final_tags: List[str] = []
    final_props: Dict[str, str] = {}

    for tag in block.tags:
        sentence = block.props.get(tag, "").strip()
        if not sentence or tag not in ALLOWED_PROPERTY_TAGS:
            continue

        if tag in {"Chiral", "CPL"}:
            final_tags.append(tag)
            final_props[tag] = sentence
            continue

        keep_after_condition = call_llm_boolean(
            model,
            build_test_condition_prompt(sentence),
            log_path=test_log_path,
            log_label="TEST_CONDITION",
            input_anchor="Sentence:\n",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not keep_after_condition:
            log_deleted_prop(log_dir, block, tag, sentence, reason="test_condition")
            continue

        keep_after_tag = call_llm_boolean(
            model,
            build_tag_match_prompt(tag, sentence),
            log_path=tag_log_path,
            log_label="TAG_MATCH",
            input_anchor=f"Property tag: {tag}\n",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not keep_after_tag:
            log_deleted_prop(log_dir, block, tag, sentence, reason="tag_mismatch")
            continue

        final_tags.append(tag)
        final_props[tag] = sentence

    if not final_tags:
        return None

    return RefinedBlock(
        paper_id=block.paper_id,
        kind=block.kind,
        sample=block.sample,
        para_id=block.para_id,
        win_level=block.win_level,
        window_sids=block.window_sids,
        evidence_text=block.evidence_text,
        tags=final_tags,
        props=final_props,
    )


def build_refined_paths(paper_dir: str, paper_id: str) -> Dict[str, Dict[str, str]]:
    in_dir = os.path.join(paper_dir, "property", "refined_properties")
    out_dir = os.path.join(paper_dir, "property", "abstract_clean")
    return {
        "main": {
            "input": os.path.join(in_dir, f"{paper_id}.md"),
            "output": os.path.join(out_dir, f"{paper_id}.md"),
        },
        "vs": {
            "input": os.path.join(in_dir, f"{paper_id}_vs.md"),
            "output": os.path.join(out_dir, f"{paper_id}_vs.md"),
        },
        "app": {
            "input": os.path.join(in_dir, f"{paper_id}_app.md"),
            "output": os.path.join(out_dir, f"{paper_id}_app.md"),
        },
    }


def process_one_paper(
    paper_dir: str,
    *,
    model,
    log_dir: str,
    temperature: float,
    max_tokens: int,
    skip_existing: bool,
) -> bool:
    paper_id = paper_id_from_dir(paper_dir)
    if paper_id is None:
        return False

    path_map = build_refined_paths(paper_dir, paper_id)
    out_dir = os.path.join(paper_dir, "property", "abstract_clean")

    tasks = []
    existing_outputs: List[str] = []
    for kind, paths in path_map.items():
        if os.path.exists(paths["output"]):
            existing_outputs.append(kind)
        if not os.path.exists(paths["input"]):
            continue
        if skip_existing and os.path.exists(paths["output"]):
            continue
        tasks.append((kind, paths["input"], paths["output"]))

    if not tasks:
        return False

    parsed_tasks = []
    parse_empty_files: List[str] = []
    for kind, in_md, out_md in tasks:
        blocks = parse_refined_md(read_text(in_md), paper_id=paper_id, kind=kind)
        if blocks:
            parsed_tasks.append((kind, in_md, out_md, blocks))
        else:
            parse_empty_files.append(in_md)

    if parse_empty_files:
        notes: List[str] = []
        if parsed_tasks:
            notes.append(
                "parsed_non_empty=" + ",".join(os.path.basename(in_md) for _, in_md, _, _ in parsed_tasks)
            )
        if existing_outputs:
            notes.append("existing_outputs=" + ",".join(existing_outputs))
        note = "; ".join(notes)
        if parsed_tasks or existing_outputs:
            log_empty_case(log_dir, "partial_parse_empty", paper_dir, parse_empty_files, note=note)
        else:
            moved_to = move_paper_dir_to_no_property(paper_dir)
            log_empty_case(log_dir, "all_parse_empty_moved", paper_dir, parse_empty_files, moved_to=moved_to)

    if not parsed_tasks:
        return True

    os.makedirs(out_dir, exist_ok=True)
    test_log_path = os.path.join(out_dir, f"{paper_id}_test_condition_llm.log")
    tag_log_path = os.path.join(out_dir, f"{paper_id}_tag_match_llm.log")

    rendered_results = []
    cleaned_empty_inputs: List[str] = []
    has_non_empty_output = False

    for kind, in_md, out_md, blocks in parsed_tasks:
        cleaned_blocks: List[RefinedBlock] = []
        for block in tqdm(blocks, desc=f"Step6: clean {paper_id} [{kind}]", leave=False):
            cleaned = clean_block_with_llm(
                block,
                model=model,
                log_dir=log_dir,
                test_log_path=test_log_path,
                tag_log_path=tag_log_path,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if cleaned is not None:
                cleaned_blocks.append(cleaned)

        rendered_results.append((out_md, render_cleaned_md(cleaned_blocks, kind)))
        if cleaned_blocks:
            has_non_empty_output = True
        else:
            cleaned_empty_inputs.append(in_md)

    if cleaned_empty_inputs and not has_non_empty_output and not parse_empty_files and not existing_outputs:
        moved_to = move_paper_dir_to_no_property(paper_dir)
        log_empty_case(
            log_dir,
            "all_cleaned_empty_moved",
            paper_dir,
            cleaned_empty_inputs,
            moved_to=moved_to,
        )
        return True

    for out_md, rendered_text in rendered_results:
        write_text(out_md, rendered_text)

    return True


def process_all_papers(
    mining_root: str,
    paper_ids: Optional[Sequence[str]] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    skip_existing: bool = True,
) -> None:
    if lmstudio_llm is None:
        raise RuntimeError("lmstudio is not installed. Install or configure LM Studio before running Step 6.")

    root = ensure_root_exists(mining_root)
    log_dir = get_step6_log_dir(root)
    write_run_headers(log_dir)

    model = lmstudio_llm(model_name)
    paper_dirs = iter_paper_dirs(root, paper_ids=paper_ids)
    for paper_dir in tqdm(paper_dirs, desc="Step6: clean-refined"):
        process_one_paper(
            paper_dir,
            model=model,
            log_dir=log_dir,
            temperature=temperature,
            max_tokens=max_tokens,
            skip_existing=skip_existing,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Property preprocessing step6: clean Step 5 refined markdown.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Re-run even if Step 6 outputs already exist.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model name used for Step 6.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="LLM max tokens per call.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_all_papers(
        mining_root=args.root,
        paper_ids=args.paper_ids,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
