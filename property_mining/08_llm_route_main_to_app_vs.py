#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step 8: route resolved main properties into final main, app, and vs markdown outputs."""

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from tqdm import tqdm

from property_unit import (
    append_log_line as append_log,
    build_decision_sid_maps,
    build_evidence_lines_from_sids,
    ensure_dir,
    ensure_root_exists,
    flatten_property_entries,
    iter_paper_dirs,
    paper_id_from_dir,
    parse_json_object_text,
    parse_property_markdown,
    read_text,
    relative_to_paper,
    relative_to_root,
    remove_think_blocks,
    render_property_markdown,
    resolve_decision_csv_path,
    strip_code_fences,
    timestamp_now,
    write_text,
)

try:
    from lmstudio import llm as lmstudio_llm
except Exception:
    lmstudio_llm = None


DEFAULT_MODEL = "qwen.qwen2.5-vl-32b-instruct"
DEFAULT_TEMPERATURE = 0.25
DEFAULT_MAX_TOKENS = 4000
DEFAULT_RETRIES = 3
END_SENTINEL = "<END_OF_JSON>"
CLASSIFY_TAGS = {"Ex", "Em", "QY", "lifetime"}
VALID_DECISIONS = {"main", "app", "vs", "drop"}
MAIN_LOG_STATUSES = {"SKIP_EXISTS", "PROCESSED"}
_SMART_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


@dataclass
class Step8Result:
    paper_id: str
    paper_dir: str
    status: str
    input_main: str
    output_dir: str
    note: str = ""


def build_log_paths(out_dir: str, paper_id: str) -> Dict[str, str]:
    return {
        "io": os.path.join(out_dir, f"{paper_id}_step8_route.io.log"),
        "llm": os.path.join(out_dir, f"{paper_id}_step8_route.llm.log"),
    }


def normalize_model_output(raw: Any) -> str:
    return strip_code_fences(remove_think_blocks(str(raw or ""))).translate(_SMART_QUOTES).strip()


def llm_call_raw(model, prompt: str, temperature: float, max_tokens: int) -> str:
    response = model.respond(prompt, config={"temperature": temperature, "maxTokens": max_tokens})
    return response.content if hasattr(response, "content") else str(response)


def write_llm_attempt(log_path: str, label: str, attempt: int, prompt: str, output: str, error: str = "") -> None:
    append_log(log_path, f"=== {label} attempt={attempt} at {timestamp_now()} ===")
    append_log(log_path, "INPUT:")
    append_log(log_path, prompt.rstrip())
    append_log(log_path, "OUTPUT:")
    append_log(log_path, output.rstrip() if output.strip() else "<EMPTY>")
    if error:
        append_log(log_path, f"ERROR: {error}")
    append_log(log_path, "")


def call_llm_json(
    model,
    *,
    prompt: str,
    log_path: str,
    label: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    validator: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
) -> Dict[str, Any]:
    last_error = "No LLM attempts were made."
    for attempt in range(1, retries + 1):
        prompt_to_send = prompt
        if attempt >= 2:
            prompt_to_send += (
                f"\n\nREMINDER: Return exactly one JSON object and end with {END_SENTINEL}. "
                "Do not add prose, markdown, or code fences."
            )

        try:
            raw = llm_call_raw(model, prompt_to_send, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            last_error = repr(exc)
            write_llm_attempt(log_path, label, attempt, prompt_to_send, "", last_error)
            continue

        cleaned = normalize_model_output(raw)
        parsed = parse_json_object_text(cleaned, end_sentinel=END_SENTINEL)
        error = ""
        if parsed is None:
            error = "Failed to parse a JSON object from the LLM output."
        elif validator is not None:
            error = validator(parsed) or ""

        write_llm_attempt(log_path, label, attempt, prompt_to_send, cleaned, error)
        if not error and parsed is not None:
            return parsed
        last_error = error or "Unknown parsing error."

    raise RuntimeError(f"{label} failed after {retries} attempts: {last_error}")


def validate_route_response(payload: Dict[str, Any]) -> Optional[str]:
    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in VALID_DECISIONS:
        return f"Invalid decision: {decision!r}"
    return None


def build_step1_recheck_prompt(
    *,
    sample_name: str,
    tag: str,
    final_sentence: str,
    evidence_text: str,
) -> str:
    return f"""You are an expert in classifying photoluminescence (PL) properties of carbon dots (CDs).

Your task is to classify ONE refined property sentence for a single target sample and a single tag into one of four categories: "main", "app", "vs", or "drop".

IMPORTANT: You are deciding the category ONLY for the target sample_name, even if the sentence mentions other samples or composite systems.

RESPONSE FORMAT:
- Output exactly ONE JSON object and then the string {END_SENTINEL}.
- Do NOT output any other text, comments, or markdown.

{{
  "decision": "<main or app or vs or drop>"
}}{END_SENTINEL}

INPUTS:
- sample_name = "{sample_name}"
- tag         = "{tag}"
- sentence:
  {final_sentence}

- evidence_text:
  {evidence_text}

CATEGORY DEFINITIONS:

1) "main"
   - The sentence describes an intrinsic PL property of the free CDs sample "{sample_name}" under normal characterization conditions.
   - It does NOT rely on an external analyte, detection target, imaging subject, anti-counterfeiting pattern, or any other functional application scenario.
   - It does NOT focus on a trend or comparison.
   - It is NOT exclusively describing a composite labeled with the character "@" as the only emitting material.

2) "app"
   - The sentence describes PL behavior of CDs being used in a concrete functional application, such as sensing, detection, imaging, anti-counterfeiting, encryption, or related optical applications.

3) "vs"
   - The sentence expresses a clear comparison or trend of the PL property with respect to another sample, composition, synthesis parameter, or state.

4) "drop"
   - The sentence should not be kept in any category for this project.

SIMPLIFIED RULE FOR '@' COMPOSITES:
- Any material label that contains "@" is treated as a composite or support system.
- If the sentence describes ONLY such composite(s) and there is NO clear free CDs system acting as the emitting material, then:
  * do NOT choose "main";
  * if it clearly belongs to a functional application, choose "app";
  * else if it clearly expresses a comparison or trend, choose "vs";
  * otherwise choose "drop".

DECISION PRIORITY:
1) app
2) vs
3) main
4) drop
"""


def call_step1_recheck_decision(
    model,
    *,
    sample_name: str,
    tag: str,
    final_sentence: str,
    evidence_text: str,
    log_path: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> str:
    payload = call_llm_json(
        model,
        prompt=build_step1_recheck_prompt(
            sample_name=sample_name,
            tag=tag,
            final_sentence=final_sentence,
            evidence_text=evidence_text,
        ),
        log_path=log_path,
        label="STEP8_ROUTE_DECISION",
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        validator=validate_route_response,
    )
    return str(payload["decision"]).strip().lower()


def outputs_exist(output_dir: str, paper_id: str) -> bool:
    expected = (
        os.path.join(output_dir, f"{paper_id}_main.md"),
        os.path.join(output_dir, f"{paper_id}_app.md"),
        os.path.join(output_dir, f"{paper_id}_vs.md"),
    )
    return all(os.path.exists(path) for path in expected)


def write_io_log(result: Step8Result, log_path: str) -> None:
    append_log(log_path, f"[{timestamp_now()}] paper={result.paper_id} status={result.status}")
    if result.input_main:
        append_log(log_path, f"input_main={relative_to_paper(result.paper_dir, result.input_main)}")
    append_log(log_path, f"output_dir={relative_to_paper(result.paper_dir, result.output_dir)}")
    if result.note:
        append_log(log_path, f"note={result.note}")
    append_log(log_path, "")


def process_one_paper(
    paper_dir: str,
    *,
    model,
    skip_existing: bool,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> Step8Result:
    paper_id = paper_id_from_dir(paper_dir)
    if paper_id is None:
        return Step8Result("", paper_dir, "SKIP_INVALID_DIR", "", "", "Directory name does not start with a paper id.")

    in_main = os.path.join(paper_dir, "property", "abstract_clean", f"{paper_id}_main.md")
    in_app = os.path.join(paper_dir, "property", "abstract_clean", f"{paper_id}_app.md")
    in_vs = os.path.join(paper_dir, "property", "abstract_clean", f"{paper_id}_vs.md")
    out_dir = os.path.join(paper_dir, "property", "clean_abstract")
    out_main = os.path.join(out_dir, f"{paper_id}_main.md")
    out_app = os.path.join(out_dir, f"{paper_id}_app.md")
    out_vs = os.path.join(out_dir, f"{paper_id}_vs.md")
    ensure_dir(out_dir)
    log_paths = build_log_paths(out_dir, paper_id)

    if skip_existing and outputs_exist(out_dir, paper_id):
        result = Step8Result(paper_id, paper_dir, "SKIP_EXISTS", in_main, out_dir, "All Step 8 outputs already exist.")
        write_io_log(result, log_paths["io"])
        return result

    if not any(os.path.exists(path) for path in (in_main, in_app, in_vs)):
        result = Step8Result(paper_id, paper_dir, "SKIP_NO_INPUTS", in_main, out_dir, "No Step 8 source markdown files were found.")
        write_io_log(result, log_paths["io"])
        return result

    decision_csv = resolve_decision_csv_path(paper_dir, paper_id)
    sid_to_text: Dict[int, str] = {}
    note_parts: List[str] = []
    if os.path.exists(decision_csv):
        try:
            _, sid_to_text = build_decision_sid_maps(decision_csv)
            note_parts.append(f"decision_csv={relative_to_paper(paper_dir, decision_csv)}")
        except Exception as exc:
            note_parts.append(f"decision_csv_error={exc!r}; fallback=markdown_evidence_only")
    else:
        note_parts.append("decision_csv_missing; fallback=markdown_evidence_only")

    seeded_app_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seeded_vs_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    routed_main_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    if os.path.exists(in_app):
        for item in flatten_property_entries(parse_property_markdown(in_app)):
            seeded_app_by_sample[str(item.get("sample", "")).strip()].append(dict(item))
    if os.path.exists(in_vs):
        for item in flatten_property_entries(parse_property_markdown(in_vs)):
            seeded_vs_by_sample[str(item.get("sample", "")).strip()].append(dict(item))

    if os.path.exists(in_main) and read_text(in_main).strip():
        main_items = flatten_property_entries(parse_property_markdown(in_main))
    else:
        main_items = []
        note_parts.append("missing_or_empty_main_input")

    dropped_count = 0
    llm_failures = 0

    for item in tqdm(main_items, desc=f"Step8: route {paper_id}", leave=False):
        sample = str(item.get("sample", "")).strip()
        tag = str(item.get("tag", "")).strip()
        sentence = str(item.get("sentence", "")).strip()
        if not sample or not tag or not sentence:
            continue

        routed_item = dict(item)
        evidence_lines = build_evidence_lines_from_sids(
            item.get("window_sids", []) or [],
            sid_to_text,
            fallback_lines=item.get("evidence_lines", []) or [],
        )
        routed_item["evidence_lines"] = evidence_lines

        if tag not in CLASSIFY_TAGS:
            routed_main_by_sample[sample].append(routed_item)
            continue

        evidence_text = " ".join(str(line).strip() for line in evidence_lines if str(line).strip()).strip()
        try:
            decision = call_step1_recheck_decision(
                model,
                sample_name=sample,
                tag=tag,
                final_sentence=sentence,
                evidence_text=evidence_text,
                log_path=log_paths["llm"],
                temperature=temperature,
                max_tokens=max_tokens,
                retries=retries,
            )
        except Exception:
            decision = "main"
            llm_failures += 1

        if decision == "main":
            routed_main_by_sample[sample].append(routed_item)
        elif decision == "app":
            seeded_app_by_sample[sample].append(routed_item)
        elif decision == "vs":
            seeded_vs_by_sample[sample].append(routed_item)
        else:
            dropped_count += 1

    write_text(out_main, render_property_markdown(routed_main_by_sample, property_label="Property abstract"))
    write_text(out_app, render_property_markdown(seeded_app_by_sample, property_label="Property abstract"))
    write_text(out_vs, render_property_markdown(seeded_vs_by_sample, property_label="Property abstract"))

    note_parts.extend(
        [
            f"main_written={sum(len(items) for items in routed_main_by_sample.values())}",
            f"app_written={sum(len(items) for items in seeded_app_by_sample.values())}",
            f"vs_written={sum(len(items) for items in seeded_vs_by_sample.values())}",
            f"drop_count={dropped_count}",
            f"llm_failures={llm_failures}",
        ]
    )
    result = Step8Result(
        paper_id=paper_id,
        paper_dir=paper_dir,
        status="PROCESSED",
        input_main=in_main,
        output_dir=out_dir,
        note="; ".join(note_parts),
    )
    write_io_log(result, log_paths["io"])
    return result


def write_root_logs(mining_root: str, results: Sequence[Step8Result]) -> None:
    main_log_path = os.path.join(mining_root, "step8_route_main_app_vs.log")
    error_log_path = os.path.join(mining_root, "step8_route_main_app_vs_error.log")

    with open(main_log_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# Step 8 main/app/vs routing\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status not in MAIN_LOG_STATUSES:
                continue
            fh.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_main:
                fh.write(f"  input_main={relative_to_root(mining_root, result.input_main)}\n")
            fh.write(f"  output_dir={relative_to_root(mining_root, result.output_dir)}\n")
            if result.note:
                fh.write(f"  note={result.note}\n")

    with open(error_log_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# Step 8 routing errors\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status in MAIN_LOG_STATUSES:
                continue
            fh.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_main:
                fh.write(f"  input_main={relative_to_root(mining_root, result.input_main)}\n")
            if result.output_dir:
                fh.write(f"  output_dir={relative_to_root(mining_root, result.output_dir)}\n")
            if result.note:
                fh.write(f"  note={result.note}\n")


def process_all_papers(
    mining_root: str,
    paper_ids: Optional[Sequence[str]] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = DEFAULT_RETRIES,
    skip_existing: bool = True,
) -> None:
    if lmstudio_llm is None:
        raise RuntimeError("lmstudio is not installed. Install or configure LM Studio before running Step 8.")

    root = ensure_root_exists(mining_root)
    model = lmstudio_llm(model_name)
    results: List[Step8Result] = []
    for paper_dir in tqdm(iter_paper_dirs(root, paper_ids=paper_ids), desc="Step8: route-main-app-vs"):
        try:
            results.append(
                process_one_paper(
                    paper_dir,
                    model=model,
                    skip_existing=skip_existing,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retries=retries,
                )
            )
        except Exception as exc:
            paper_id = paper_id_from_dir(paper_dir) or ""
            results.append(
                Step8Result(
                    paper_id=paper_id,
                    paper_dir=paper_dir,
                    status="SKIP_FATAL",
                    input_main=os.path.join(paper_dir, "property", "abstract_clean", f"{paper_id}_main.md") if paper_id else "",
                    output_dir=os.path.join(paper_dir, "property", "clean_abstract"),
                    note=repr(exc),
                )
            )

    write_root_logs(root, results)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Property preprocessing Step 8: route resolved main markdown into main/app/vs outputs.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Re-run even if Step 8 outputs already exist.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model name used for Step 8.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature for Step 8 routing.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="LLM max tokens per Step 8 routing call.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Max JSON-call retries per LLM request.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_all_papers(
        mining_root=args.root,
        paper_ids=args.paper_ids,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
