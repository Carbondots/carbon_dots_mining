#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step 9: bind Step 8 app/vs properties to concrete samples."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from property_unit import (
    SampleInfo,
    append_log_line as append_log,
    build_decision_sid_maps,
    build_property_item,
    candidates_from_sids,
    ensure_dir,
    ensure_root_exists,
    flatten_property_entries,
    iter_paper_dirs,
    normalize_name_signature,
    paper_id_from_dir,
    parse_json_object_text,
    parse_property_markdown,
    read_letter_table_samples,
    relative_to_paper,
    relative_to_root,
    remove_think_blocks,
    render_property_markdown,
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


DEFAULT_MODEL = "qwen.qwen2.5-vl-32b-instruct"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2000
DEFAULT_RETRIES = 3
END_SENTINEL = "<END_OF_JSON>"
SUPPORTED_TAGS = {"Ex", "Em", "QY", "lifetime", "ExDep", "Chiral", "CPL"}
MAIN_DEDUP_TAGS = {"Ex", "Em", "ExDep", "Chiral", "CPL"}
MAIN_LOG_STATUSES = {"SKIP_EXISTS", "PROCESSED", "PROCESSED_PARTIAL"}


@dataclass
class Step9Result:
    paper_id: str
    paper_dir: str
    status: str
    input_main: str
    input_app: str
    input_vs: str
    input_decision_csv: str
    input_letter_csv: str
    output_app: str
    output_vs: str
    note: str = ""


def normalize_model_output(raw: Any) -> str:
    return strip_code_fences(remove_think_blocks(str(raw or ""))).strip()


def llm_call_raw(model, prompt: str, *, temperature: float, max_tokens: int) -> str:
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
        payload = parse_json_object_text(cleaned, end_sentinel=END_SENTINEL)
        error = ""
        if payload is None:
            error = "Failed to parse a JSON object from the LLM output."
        elif validator is not None:
            error = validator(payload) or ""

        write_llm_attempt(log_path, label, attempt, prompt_to_send, cleaned, error)
        if not error and payload is not None:
            return payload
        last_error = error or "Unknown JSON validation error."

    raise RuntimeError(f"{label} failed after {retries} attempts: {last_error}")


def build_log_paths(output_dir: str, paper_id: str) -> Dict[str, str]:
    return {
        "io": os.path.join(output_dir, f"{paper_id}_step9_bind_app_vs.io.log"),
        "app": os.path.join(output_dir, f"{paper_id}_step9_bind_app.trace.log"),
        "vs": os.path.join(output_dir, f"{paper_id}_step9_bind_vs.trace.log"),
    }


def select_existing_path(candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else ""


def resolve_main_input_path(paper_dir: str, paper_id: str) -> str:
    return select_existing_path(
        (
            stage_markdown_path(paper_dir, "final_properties", paper_id, kind="main"),
            os.path.join(paper_dir, "property", "final_properties", f"{paper_id}_main.md"),
            os.path.join(paper_dir, "property", "clean_abstract", f"{paper_id}_main.md"),
        )
    )


def resolve_kind_input_path(paper_dir: str, paper_id: str, kind: str) -> str:
    return select_existing_path(
        (
            stage_markdown_path(paper_dir, "final_properties", paper_id, kind=kind),
            os.path.join(paper_dir, "property", "clean_abstract", f"{paper_id}_{kind}.md"),
        )
    )


def output_kind_path(paper_dir: str, paper_id: str, kind: str) -> str:
    return stage_markdown_path(paper_dir, "bound_app_vs_properties", paper_id, kind=kind)


def outputs_exist(paper_dir: str, paper_id: str) -> bool:
    return all(
        os.path.exists(output_kind_path(paper_dir, paper_id, kind))
        for kind in ("app", "vs")
    )


def extract_main_tag_map(main_path: str) -> Dict[Tuple[str, str], bool]:
    mapping: Dict[Tuple[str, str], bool] = {}
    if not os.path.exists(main_path):
        return mapping
    for item in flatten_property_entries(parse_property_markdown(main_path)):
        sample = str(item.get("sample", "")).strip()
        tag = str(item.get("tag", "")).strip()
        if sample and tag:
            mapping[(sample, tag)] = True
    return mapping


def build_candidate_block(candidates: Sequence[SampleInfo]) -> str:
    if not candidates:
        return "None."
    return "\n".join(
        f"{index}. {sample.name}: {sample.desc or '(no synthesis summary available)'}"
        for index, sample in enumerate(candidates, start=1)
    )


def build_sample_catalog(letter_samples: Sequence[SampleInfo]) -> Dict[str, SampleInfo]:
    return {sample.name: sample for sample in letter_samples}


def resolve_entry_candidates(
    *,
    entry_sample: str,
    window_sids: Sequence[int],
    sid_to_names: Dict[int, str],
    letter_samples: Sequence[SampleInfo],
) -> List[SampleInfo]:
    candidates = list(candidates_from_sids(window_sids, sid_to_names, letter_samples))
    if not candidates:
        return list(letter_samples)

    hint = str(entry_sample or "").strip()
    hint_signature = normalize_name_signature(hint)
    if not hint_signature:
        return candidates

    if any(normalize_name_signature(sample.name) == hint_signature for sample in candidates):
        return candidates

    for sample in letter_samples:
        if normalize_name_signature(sample.name) == hint_signature:
            return [sample, *candidates]
    return candidates


def build_binding_prompt(
    *,
    kind: str,
    tag: str,
    header_sample: str,
    current_sentence: str,
    evidence_text: str,
    candidates: Sequence[SampleInfo],
) -> str:
    sample_hint = str(header_sample or "").strip() or "(no reliable sample hint)"
    return f"""You are binding ONE property statement from a carbon-dot paper to the correct sample or samples from THIS paper only.

Output exactly ONE JSON object and then {END_SENTINEL}. Do not output any other text.

Allowed schema:
{{"scope":"none|single|multi|family","samples":["sampleA","sampleB"]}}{END_SENTINEL}

Definitions:
- "none": the evidence does not support any sample from this paper.
- "single": the evidence supports exactly one concrete candidate sample.
- "multi": the evidence supports multiple concrete candidate samples.
- "family": the evidence is about a series, group, or generic label and cannot be safely bound to concrete candidates.

Rules:
- Use ONLY sample names from the candidate list.
- If multiple concrete candidates are explicitly supported, return ALL of them in the listed order.
- Do not guess a specific candidate from a generic label such as "CDs" or "carbon dots".
- Prefer "family" over guessing when the evidence is series-level or ambiguous.

Inputs:
- source_kind = "{kind}"
- property_tag = "{tag}"
- header_sample_hint = "{sample_hint}"
- current_sentence = "{current_sentence}"

Evidence:
{evidence_text}

Candidate samples:
{build_candidate_block(candidates)}
""".strip()


def validate_binding_response(payload: Dict[str, Any]) -> Optional[str]:
    scope = str(payload.get("scope", "")).strip().lower()
    if scope not in {"none", "single", "multi", "family"}:
        return f"Invalid scope: {scope!r}"
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        return "Field 'samples' must be a list."
    return None


def normalize_bound_sample_names(
    *,
    payload: Dict[str, Any],
    candidates: Sequence[SampleInfo],
) -> Tuple[str, List[str]]:
    scope = str(payload.get("scope", "")).strip().lower()
    allowed_names = [sample.name for sample in candidates]
    allowed_set = set(allowed_names)
    selected: List[str] = []
    for raw_name in payload.get("samples", []) or []:
        name = str(raw_name or "").strip()
        if name and name in allowed_set and name not in selected:
            selected.append(name)

    if scope == "single":
        if len(selected) > 1:
            scope = "multi"
        elif not selected and len(allowed_names) == 1:
            selected = allowed_names[:1]
    elif scope == "multi" and len(selected) == 1:
        scope = "single"
    return scope, selected


def property_rewrite_rules(tag: str, kind: str) -> str:
    source_rule = {
        "app": (
            "Because source_kind=app, keep application context only when the evidence explicitly supports it. "
            "Do not turn an application sentence into a generic intrinsic property sentence."
        ),
        "vs": (
            "Because source_kind=vs, keep comparison or trend wording only when it is explicit in the evidence. "
            "Do not invent ranking language or comparative claims."
        ),
    }.get(kind, "Use only the source sentence type supported by the evidence.")

    shared = {
        "Ex": "Keep only an excitation wavelength explicitly supported by the evidence. Ignore absorption wavelengths and generic UV-light observation.",
        "Em": "Keep only an emission wavelength or emission peak explicitly supported by the evidence. Ignore readout channels unless explicitly stated as emission.",
        "QY": "Keep only an explicit quantum-yield value. Do not invent percentages or mix baseline and post-treatment values.",
        "lifetime": "Keep only an explicit lifetime value with a time unit. Do not invent averages across components.",
        "ExDep": "Keep only an explicit excitation-dependence or excitation-independence statement for the target sample.",
        "Chiral": "Keep only an explicit chirality statement for the target sample itself.",
        "CPL": "Keep only an explicit CPL statement for the target sample itself.",
    }
    return f"{shared.get(tag, 'Keep only evidence-supported content for the requested tag.')} {source_rule}"


def build_refine_prompt(
    *,
    kind: str,
    tag: str,
    target_sample: SampleInfo,
    evidence_text: str,
    current_sentence: str,
) -> str:
    desc = target_sample.desc.strip() if target_sample.desc.strip() else "(no synthesis summary available)"
    return f"""You are rewriting ONE property sentence for ONE exact carbon-dot sample.

Output exactly ONE JSON object and then {END_SENTINEL}. Do not output any other text.

Schema:
{{"result":"NONE|<one English sentence>"}}{END_SENTINEL}

Hard rules:
- If you output a sentence, it MUST contain the exact sample name "{target_sample.name}".
- Use ONLY information explicitly supported by the evidence.
- Do not invent numbers, units, comparison direction, analytes, or conditions.
- For numeric tags, the sentence must include an explicit numeric value and unit from the evidence.
- If the evidence is too ambiguous to rewrite a safe sentence for this exact sample, output NONE.

Inputs:
- source_kind = "{kind}"
- property_tag = "{tag}"
- target_sample_name = "{target_sample.name}"
- target_sample_description = "{desc}"

Tag and source rules:
{property_rewrite_rules(tag, kind)}

Evidence:
{evidence_text}

Current sentence:
{current_sentence}
""".strip()


def validate_refine_response(payload: Dict[str, Any]) -> Optional[str]:
    result = payload.get("result", None)
    if result is None:
        return "Missing field 'result'."
    if not isinstance(result, str):
        return "Field 'result' must be a string."
    return None


def rewrite_bound_sentence(
    model,
    *,
    kind: str,
    tag: str,
    target_sample: SampleInfo,
    evidence_text: str,
    current_sentence: str,
    log_path: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> str:
    payload = call_llm_json(
        model,
        prompt=build_refine_prompt(
            kind=kind,
            tag=tag,
            target_sample=target_sample,
            evidence_text=evidence_text,
            current_sentence=current_sentence,
        ),
        log_path=log_path,
        label=f"STEP9_REWRITE_{kind.upper()}_{tag}",
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        validator=validate_refine_response,
    )
    result = str(payload.get("result", "")).strip()
    if not result or result.upper() == "NONE":
        return ""
    if target_sample.name not in result:
        return ""
    return result


def append_item(
    items_by_sample: Dict[str, List[Dict[str, Any]]],
    seen_keys: set,
    *,
    sample: str,
    para_id: int,
    win_level: str,
    window_sids: Sequence[int],
    evidence_lines: Sequence[str],
    tag: str,
    sentence: str,
) -> None:
    key = (
        sample,
        int(para_id),
        str(win_level).strip(),
        tuple(int(value) for value in window_sids),
        tag,
        sentence,
    )
    if key in seen_keys:
        return False
    seen_keys.add(key)
    items_by_sample[sample].append(
        build_property_item(
            sample=sample,
            para_id=para_id,
            win_level=win_level,
            window_sids=window_sids,
            evidence_lines=evidence_lines,
            tag=tag,
            sentence=sentence,
        )
    )
    return True


def write_kind_output(out_path: str, items_by_sample: Dict[str, List[Dict[str, Any]]]) -> str:
    ensure_dir(os.path.dirname(out_path))
    write_text(out_path, render_property_markdown(items_by_sample, property_label="Property abstract"))
    return out_path


def process_one_kind(
    paper_dir: str,
    *,
    model,
    paper_id: str,
    kind: str,
    in_main: str,
    in_kind: str,
    log_path: str,
    sid_to_names: Dict[int, str],
    sid_to_text: Dict[int, str],
    letter_samples: Sequence[SampleInfo],
    main_tag_map: Dict[Tuple[str, str], bool],
    temperature: float,
    max_tokens: int,
    retries: int,
) -> Tuple[str, int]:
    entries = parse_property_markdown(in_kind)
    sample_catalog = build_sample_catalog(letter_samples)
    output_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_keys = set()
    written_count = 0

    append_log(log_path, f"# Step 9 binding trace for kind={kind}")
    append_log(log_path, f"paper={paper_id}")
    append_log(log_path, f"input_main={relative_to_paper(paper_dir, in_main) if in_main else '(missing)'}")
    append_log(log_path, f"input_kind={relative_to_paper(paper_dir, in_kind)}")
    append_log(log_path, "")

    for entry in tqdm(entries, desc=f"Step9: bind {paper_id} [{kind}]", leave=False):
        entry_sample = str(entry.get("sample", "")).strip()
        para_id = int(entry.get("para_id", 0) or 0)
        win_level = str(entry.get("win_level", "") or "").strip()
        window_sids = [int(value) for value in entry.get("window_sids", []) or [] if str(value).isdigit()]
        evidence_lines = [
            text
            for sent_id in window_sids
            for text in [str(sid_to_text.get(sent_id, "")).strip()]
            if text
        ]
        if not evidence_lines:
            evidence_lines = [str(line).strip() for line in entry.get("evidence_lines", []) or [] if str(line).strip()]
        evidence_text = " ".join(evidence_lines).strip()
        if not evidence_text:
            append_log(log_path, f"[SKIP] kind={kind} para={para_id} had empty evidence after reconstruction.")
            append_log(log_path, "")
            continue

        candidates = resolve_entry_candidates(
            entry_sample=entry_sample,
            window_sids=window_sids,
            sid_to_names=sid_to_names,
            letter_samples=letter_samples,
        )
        if not candidates:
            append_log(log_path, f"[SKIP] kind={kind} para={para_id} no candidate samples were available.")
            append_log(log_path, "")
            continue

        for item in flatten_property_entries([entry]):
            tag = str(item.get("tag", "")).strip()
            sentence = str(item.get("sentence", "")).strip()
            if tag not in SUPPORTED_TAGS or not sentence:
                continue

            if len(letter_samples) == 1:
                target_names = [letter_samples[0].name]
            else:
                payload = call_llm_json(
                    model,
                    prompt=build_binding_prompt(
                        kind=kind,
                        tag=tag,
                        header_sample=entry_sample,
                        current_sentence=sentence,
                        evidence_text=evidence_text,
                        candidates=candidates,
                    ),
                    log_path=log_path,
                    label=f"STEP9_BIND_{kind.upper()}_{tag}",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retries=retries,
                    validator=validate_binding_response,
                )
                scope, target_names = normalize_bound_sample_names(payload=payload, candidates=candidates)
                if scope not in {"single", "multi"} or not target_names:
                    append_log(
                        log_path,
                        f"[SKIP_BIND] kind={kind} para={para_id} tag={tag} scope={scope or 'invalid'} targets={target_names}",
                    )
                    append_log(log_path, "")
                    continue

            for target_name in target_names:
                if tag in MAIN_DEDUP_TAGS and main_tag_map.get((target_name, tag), False):
                    append_log(
                        log_path,
                        f"[SKIP_MAIN_DEDUP] kind={kind} para={para_id} sample={target_name} tag={tag}",
                    )
                    append_log(log_path, "")
                    continue

                target_sample = sample_catalog.get(target_name)
                if target_sample is None:
                    continue

                rewritten = rewrite_bound_sentence(
                    model,
                    kind=kind,
                    tag=tag,
                    target_sample=target_sample,
                    evidence_text=evidence_text,
                    current_sentence=sentence,
                    log_path=log_path,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retries=retries,
                )
                if not rewritten:
                    append_log(
                        log_path,
                        f"[SKIP_REWRITE_NONE] kind={kind} para={para_id} sample={target_name} tag={tag}",
                    )
                    append_log(log_path, "")
                    continue

                if append_item(
                    output_items,
                    seen_keys,
                    sample=target_name,
                    para_id=para_id,
                    win_level=win_level,
                    window_sids=window_sids,
                    evidence_lines=evidence_lines,
                    tag=tag,
                    sentence=rewritten,
                ):
                    written_count += 1

    out_path = output_kind_path(paper_dir, paper_id, kind)
    write_kind_output(out_path, output_items)
    return out_path, written_count


def write_io_log(result: Step9Result, log_path: str) -> None:
    append_log(log_path, f"[{timestamp_now()}] paper={result.paper_id} status={result.status}")
    if result.input_main:
        append_log(log_path, f"input_main={relative_to_paper(result.paper_dir, result.input_main)}")
    if result.input_app:
        append_log(log_path, f"input_app={relative_to_paper(result.paper_dir, result.input_app)}")
    if result.input_vs:
        append_log(log_path, f"input_vs={relative_to_paper(result.paper_dir, result.input_vs)}")
    if result.input_decision_csv:
        append_log(log_path, f"input_decision_csv={relative_to_paper(result.paper_dir, result.input_decision_csv)}")
    if result.input_letter_csv:
        append_log(log_path, f"input_letter_csv={relative_to_paper(result.paper_dir, result.input_letter_csv)}")
    if result.output_app:
        append_log(log_path, f"output_app={relative_to_paper(result.paper_dir, result.output_app)}")
    if result.output_vs:
        append_log(log_path, f"output_vs={relative_to_paper(result.paper_dir, result.output_vs)}")
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
) -> Step9Result:
    paper_id = paper_id_from_dir(paper_dir)
    if paper_id is None:
        return Step9Result("", paper_dir, "SKIP_INVALID_DIR", "", "", "", "", "", "", "", "Directory name does not start with a paper id.")

    in_main = resolve_main_input_path(paper_dir, paper_id)
    in_app = resolve_kind_input_path(paper_dir, paper_id, "app")
    in_vs = resolve_kind_input_path(paper_dir, paper_id, "vs")
    decision_csv = resolve_decision_csv_path(paper_dir, paper_id)
    letter_csv = os.path.join(paper_dir, "Synthesis", "letter_table", f"{paper_id}.csv")
    output_dir = os.path.join(paper_dir, "property", "bound_app_vs_properties")
    output_app = output_kind_path(paper_dir, paper_id, "app")
    output_vs = output_kind_path(paper_dir, paper_id, "vs")
    ensure_dir(output_dir)
    log_paths = build_log_paths(output_dir, paper_id)
    skip_ready = True
    if os.path.exists(in_app):
        skip_ready = skip_ready and os.path.exists(output_app)
    if os.path.exists(in_vs):
        skip_ready = skip_ready and os.path.exists(output_vs)

    if skip_existing and skip_ready and (os.path.exists(in_app) or os.path.exists(in_vs)):
        result = Step9Result(
            paper_id,
            paper_dir,
            "SKIP_EXISTS",
            in_main,
            in_app,
            in_vs,
            decision_csv,
            letter_csv,
            output_app,
            output_vs,
            "Step 9 outputs already exist.",
        )
        write_io_log(result, log_paths["io"])
        return result

    if not os.path.exists(in_app) and not os.path.exists(in_vs):
        result = Step9Result(
            paper_id,
            paper_dir,
            "SKIP_NO_APP_VS_INPUT",
            in_main,
            in_app,
            in_vs,
            decision_csv,
            letter_csv,
            output_app,
            output_vs,
            "Missing Step 8 app/vs inputs.",
        )
        write_io_log(result, log_paths["io"])
        return result

    if not os.path.exists(decision_csv):
        result = Step9Result(
            paper_id,
            paper_dir,
            "SKIP_NO_DECISION_CSV",
            in_main,
            in_app,
            in_vs,
            decision_csv,
            letter_csv,
            output_app,
            output_vs,
            "Missing decision csv.",
        )
        write_io_log(result, log_paths["io"])
        return result

    if not os.path.exists(letter_csv):
        result = Step9Result(
            paper_id,
            paper_dir,
            "SKIP_NO_SYNTHESIS_TABLE",
            in_main,
            in_app,
            in_vs,
            decision_csv,
            letter_csv,
            output_app,
            output_vs,
            "Missing synthesis letter table.",
        )
        write_io_log(result, log_paths["io"])
        return result

    letter_samples = read_letter_table_samples(paper_dir, paper_id)
    if not letter_samples:
        result = Step9Result(
            paper_id,
            paper_dir,
            "SKIP_EMPTY_SYNTHESIS_TABLE",
            in_main,
            in_app,
            in_vs,
            decision_csv,
            letter_csv,
            output_app,
            output_vs,
            "The synthesis letter table did not contain usable sample names.",
        )
        write_io_log(result, log_paths["io"])
        return result

    sid_to_names, sid_to_text = build_decision_sid_maps(decision_csv)
    main_tag_map = extract_main_tag_map(in_main)

    written_paths: List[str] = []
    note_parts: List[str] = []

    if os.path.exists(in_app):
        path, count = process_one_kind(
            paper_dir,
            model=model,
            paper_id=paper_id,
            kind="app",
            in_main=in_main,
            in_kind=in_app,
            log_path=log_paths["app"],
            sid_to_names=sid_to_names,
            sid_to_text=sid_to_text,
            letter_samples=letter_samples,
            main_tag_map=main_tag_map,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
        )
        written_paths.append(path)
        note_parts.append(f"app_items={count}")

    if os.path.exists(in_vs):
        path, count = process_one_kind(
            paper_dir,
            model=model,
            paper_id=paper_id,
            kind="vs",
            in_main=in_main,
            in_kind=in_vs,
            log_path=log_paths["vs"],
            sid_to_names=sid_to_names,
            sid_to_text=sid_to_text,
            letter_samples=letter_samples,
            main_tag_map=main_tag_map,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
        )
        written_paths.append(path)
        note_parts.append(f"vs_items={count}")

    available_outputs = [path for path in written_paths if os.path.exists(path)]
    if len(available_outputs) == 2:
        status = "PROCESSED"
    elif available_outputs:
        status = "PROCESSED_PARTIAL"
    else:
        status = "SKIP_NO_OUTPUT"
        note_parts.append("No Step 9 outputs were written.")

    if not os.path.exists(in_main):
        note_parts.append("Main input was missing, so main-tag deduplication was disabled.")

    result = Step9Result(
        paper_id=paper_id,
        paper_dir=paper_dir,
        status=status,
        input_main=in_main,
        input_app=in_app,
        input_vs=in_vs,
        input_decision_csv=decision_csv,
        input_letter_csv=letter_csv,
        output_app=output_app if os.path.exists(output_app) else "",
        output_vs=output_vs if os.path.exists(output_vs) else "",
        note="; ".join(note_parts),
    )
    write_io_log(result, log_paths["io"])
    return result


def write_root_logs(mining_root: str, results: Sequence[Step9Result]) -> None:
    main_log_path = os.path.join(mining_root, "step9_bind_app_vs_samples.log")
    error_log_path = os.path.join(mining_root, "step9_bind_app_vs_samples_error.log")

    with open(main_log_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Step 9 app/vs sample binding\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status not in MAIN_LOG_STATUSES:
                continue
            handle.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_main:
                handle.write(f"  input_main={relative_to_root(mining_root, result.input_main)}\n")
            if result.input_app:
                handle.write(f"  input_app={relative_to_root(mining_root, result.input_app)}\n")
            if result.input_vs:
                handle.write(f"  input_vs={relative_to_root(mining_root, result.input_vs)}\n")
            if result.output_app:
                handle.write(f"  output_app={relative_to_root(mining_root, result.output_app)}\n")
            if result.output_vs:
                handle.write(f"  output_vs={relative_to_root(mining_root, result.output_vs)}\n")
            if result.note:
                handle.write(f"  note={result.note}\n")

    with open(error_log_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Step 9 app/vs sample binding issues\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status in MAIN_LOG_STATUSES:
                continue
            handle.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_main:
                handle.write(f"  input_main={relative_to_root(mining_root, result.input_main)}\n")
            if result.input_app:
                handle.write(f"  input_app={relative_to_root(mining_root, result.input_app)}\n")
            if result.input_vs:
                handle.write(f"  input_vs={relative_to_root(mining_root, result.input_vs)}\n")
            if result.output_app:
                handle.write(f"  output_app={relative_to_root(mining_root, result.output_app)}\n")
            if result.output_vs:
                handle.write(f"  output_vs={relative_to_root(mining_root, result.output_vs)}\n")
            if result.note:
                handle.write(f"  note={result.note}\n")


def process_all_papers(
    mining_root: str,
    paper_ids: Optional[Sequence[str]] = None,
    skip_existing: bool = True,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = DEFAULT_RETRIES,
) -> None:
    if lmstudio_llm is None:
        raise RuntimeError("lmstudio is not installed. Install or configure LM Studio before running Step 9.")

    root = ensure_root_exists(mining_root)
    model = lmstudio_llm(model_name)
    results: List[Step9Result] = []
    for paper_dir in tqdm(iter_paper_dirs(root, paper_ids=paper_ids), desc="Step9: bind-app-vs"):
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
                Step9Result(
                    paper_id=paper_id,
                    paper_dir=paper_dir,
                    status="SKIP_FATAL",
                    input_main=resolve_main_input_path(paper_dir, paper_id) if paper_id else "",
                    input_app=resolve_kind_input_path(paper_dir, paper_id, "app") if paper_id else "",
                    input_vs=resolve_kind_input_path(paper_dir, paper_id, "vs") if paper_id else "",
                    input_decision_csv=resolve_decision_csv_path(paper_dir, paper_id) if paper_id else "",
                    input_letter_csv=os.path.join(paper_dir, "Synthesis", "letter_table", f"{paper_id}.csv") if paper_id else "",
                    output_app=output_kind_path(paper_dir, paper_id, "app") if paper_id else "",
                    output_vs=output_kind_path(paper_dir, paper_id, "vs") if paper_id else "",
                    note=repr(exc),
                )
            )
    write_root_logs(root, results)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Property preprocessing Step 9: bind app/vs properties to concrete samples.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Re-run even if Step 9 outputs already exist.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model name for Step 9.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature for Step 9.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="LLM max tokens for Step 9.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Max JSON-call retries per LLM request.")
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
    )


if __name__ == "__main__":
    main()
