#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step 10: review and merge main plus bound app/vs properties."""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from property_unit import (
    SampleInfo,
    append_log_line as append_log,
    build_decision_sid_maps,
    build_evidence_lines_from_sids,
    build_property_item,
    ensure_dir,
    ensure_root_exists,
    flatten_property_entries,
    iter_paper_dirs,
    paper_id_from_dir,
    parse_boolean_answer,
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


OUTPUT_STAGE_DIR = "reviewed_final_properties"
DEFAULT_MODEL = "qwen.qwen2.5-vl-32b-instruct"
DEFAULT_TEMPERATURE = 0.25
DEFAULT_MAX_TOKENS = 1000
DEFAULT_RETRIES = 3
DEFAULT_VOTES = 3
END_SENTINEL = "<END_OF_JSON>"
SUPPORTED_TAGS = {"Ex", "Em", "QY", "lifetime", "ExDep", "Chiral", "CPL"}
NUMERIC_TAGS = {"Ex", "Em", "QY", "lifetime"}
MAIN_LOG_STATUSES = {"SKIP_EXISTS", "PROCESSED", "PROCESSED_EMPTY"}


@dataclass
class Step10Result:
    paper_id: str
    paper_dir: str
    status: str
    input_main: str
    input_app: str
    input_vs: str
    input_decision_csv: str
    input_letter_csv: str
    output_md: str
    note: str = ""


@dataclass
class ReviewVote:
    verdict: str
    reason: str


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
        "io": os.path.join(output_dir, f"{paper_id}_step10_review.io.log"),
        "review": os.path.join(output_dir, f"{paper_id}_step10_stage1.trace.log"),
        "rewrite": os.path.join(output_dir, f"{paper_id}_step10_stage2.trace.log"),
        "recheck": os.path.join(output_dir, f"{paper_id}_step10_stage3_recheck.trace.log"),
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


def resolve_bound_input_path(paper_dir: str, paper_id: str, kind: str) -> str:
    return stage_markdown_path(paper_dir, "bound_app_vs_properties", paper_id, kind=kind)


def build_sample_catalog(letter_samples: Sequence[SampleInfo]) -> Dict[str, SampleInfo]:
    return {sample.name: sample for sample in letter_samples}


def build_other_samples_context(letter_samples: Sequence[SampleInfo]) -> str:
    if len(letter_samples) < 2:
        return ""
    order = ", ".join(sample.name for sample in letter_samples)
    lines = [f"candidate_sample_order = [{order}]", "Sample descriptions:"]
    for index, sample in enumerate(letter_samples, start=1):
        lines.append(f"{index}. {sample.name}: {sample.desc or '(no synthesis summary available)'}")
    return "\n".join(lines)


def source_kind_rule(source_kind: str) -> str:
    rules = {
        "main": "For source_kind=main, keep or rewrite only intrinsic sample-level properties of the standalone carbon-dot sample.",
        "app": "For source_kind=app, application context may remain only when the evidence explicitly supports it for the target sample.",
        "vs": "For source_kind=vs, comparative or trend wording may remain only when the evidence explicitly supports it for the target sample.",
    }
    return rules.get(source_kind, "Keep only evidence-supported content for the target sample.")


def tag_rule(tag: str) -> str:
    rules = {
        "Ex": "Keep only an excitation wavelength explicitly supported for the target sample. Do not use absorption wavelengths or generic UV-light observation.",
        "Em": "Keep only an emission wavelength or emission peak explicitly supported for the target sample. Do not use readout channels unless explicitly stated as emission.",
        "QY": "Keep only an explicit quantum-yield value for the target sample.",
        "lifetime": "Keep only an explicit lifetime value with a time unit for the target sample.",
        "ExDep": "Keep only an explicit excitation-dependence or excitation-independence statement for the target sample.",
        "Chiral": "Keep only an explicit chirality statement for the target sample itself.",
        "CPL": "Keep only an explicit CPL statement for the target sample itself.",
    }
    return rules.get(tag, "Keep only evidence-supported content for the requested tag.")


def build_review_prompt(
    *,
    source_kind: str,
    tag: str,
    target_sample: SampleInfo,
    evidence_text: str,
    current_sentence: str,
    other_samples_context: str,
) -> str:
    multi_block = f"\n[Other samples in this paper]\n{other_samples_context}\n" if other_samples_context else ""
    return f"""You are reviewing ONE carbon-dot property sentence against the evidence.

Output exactly ONE JSON object and then {END_SENTINEL}. Do not output any other text.

Schema:
{{"verdict":"KEEP|DROP|REWRITE","reason":"SUPPORTED|MEASUREMENT|WRONG_SAMPLE|GENERAL|NON_BASELINE|INCOMPLETE|COMPOSITE|OTHER"}}{END_SENTINEL}

Decision rules:
- KEEP: the current sentence is already correct and fully supported for the target sample.
- DROP: the sentence is unsupported, generic, a pure measurement setting, the wrong sample, or otherwise not salvageable.
- REWRITE: the evidence supports the target sample, but the current sentence needs correction, completion, or baseline extraction.

Hard rules:
- Use only the evidence.
- For numeric tags, explicit numeric value and unit are required.
- Do not guess a specific sample from a generic family label.
- Prefer DROP over KEEP when the evidence is ambiguous.

Inputs:
- source_kind = "{source_kind}"
- property_tag = "{tag}"
- target_sample_name = "{target_sample.name}"
- target_sample_description = "{target_sample.desc or '(no synthesis summary available)'}"

Tag rule:
{tag_rule(tag)}

Source-kind rule:
{source_kind_rule(source_kind)}{multi_block}

Evidence:
{evidence_text}

Current sentence:
{current_sentence}
""".strip()


def validate_review_response(payload: Dict[str, Any]) -> Optional[str]:
    verdict = str(payload.get("verdict", "")).strip().upper()
    if verdict not in {"KEEP", "DROP", "REWRITE"}:
        return f"Invalid verdict: {verdict!r}"
    reason = str(payload.get("reason", "")).strip().upper()
    if not reason:
        return "Field 'reason' must be present."
    return None


def aggregate_review_votes(votes: Sequence[ReviewVote]) -> ReviewVote:
    normalized = [vote for vote in votes if vote.verdict in {"KEEP", "DROP", "REWRITE"}]
    if not normalized:
        return ReviewVote("DROP", "OTHER")

    counts = Counter(vote.verdict for vote in normalized)
    for verdict, count in counts.most_common():
        if count >= 2:
            matching = [vote.reason for vote in normalized if vote.verdict == verdict]
            reason = Counter(matching).most_common(1)[0][0] if matching else "OTHER"
            return ReviewVote(verdict, reason)

    for fallback in ("REWRITE", "DROP", "KEEP"):
        for vote in normalized:
            if vote.verdict == fallback:
                return vote
    return ReviewVote("DROP", "OTHER")


def build_rewrite_prompt(
    *,
    source_kind: str,
    tag: str,
    target_sample: SampleInfo,
    evidence_text: str,
    current_sentence: str,
    review_reason: str,
    other_samples_context: str,
) -> str:
    multi_block = f"\n[Other samples in this paper]\n{other_samples_context}\n" if other_samples_context else ""
    return f"""You are rewriting ONE carbon-dot property sentence for ONE exact sample.

Output exactly ONE JSON object and then {END_SENTINEL}. Do not output any other text.

Schema:
{{"sentence":"NONE|<one English sentence>"}}{END_SENTINEL}

Hard rules:
- If you output a sentence, it MUST contain the exact sample name "{target_sample.name}".
- Use ONLY information explicitly supported by the evidence.
- Do not invent analytes, conditions, sample identity, numeric values, or units.
- For numeric tags, the sentence must contain an explicit numeric value and unit from the evidence.
- If the evidence cannot support a safe sentence for the target sample, output NONE.

Inputs:
- source_kind = "{source_kind}"
- property_tag = "{tag}"
- target_sample_name = "{target_sample.name}"
- target_sample_description = "{target_sample.desc or '(no synthesis summary available)'}"
- review_reason = "{review_reason}"

Tag rule:
{tag_rule(tag)}

Source-kind rule:
{source_kind_rule(source_kind)}{multi_block}

Evidence:
{evidence_text}

Current sentence:
{current_sentence}
""".strip()


def validate_rewrite_response(payload: Dict[str, Any]) -> Optional[str]:
    sentence = payload.get("sentence", None)
    if sentence is None:
        return "Missing field 'sentence'."
    if not isinstance(sentence, str):
        return "Field 'sentence' must be a string."
    return None


def build_recheck_prompt(
    *,
    source_kind: str,
    tag: str,
    target_sample: SampleInfo,
    evidence_text: str,
    rewritten_sentence: str,
    other_samples_context: str,
) -> str:
    multi_block = f"\n[Other samples in this paper]\n{other_samples_context}\n" if other_samples_context else ""
    return f"""You are verifying ONE rewritten carbon-dot property sentence.

Output exactly one JSON object with key "keep" and then {END_SENTINEL}. Do not output any other text.

Schema:
{{"keep": true | false}}{END_SENTINEL}

Keep the sentence only if it is fully supported for the target sample and obeys the source-kind and tag rules.

Inputs:
- source_kind = "{source_kind}"
- property_tag = "{tag}"
- target_sample_name = "{target_sample.name}"
- target_sample_description = "{target_sample.desc or '(no synthesis summary available)'}"

Tag rule:
{tag_rule(tag)}

Source-kind rule:
{source_kind_rule(source_kind)}{multi_block}

Evidence:
{evidence_text}

Rewritten sentence:
{rewritten_sentence}
""".strip()


def validate_recheck_response(payload: Dict[str, Any]) -> Optional[str]:
    keep = payload.get("keep", None)
    if keep is None:
        return "Missing field 'keep'."
    if isinstance(keep, bool):
        return None
    parsed = parse_boolean_answer(str(keep), end_sentinel=END_SENTINEL, default=False)
    if parsed in {True, False}:
        return None
    return "Field 'keep' must be boolean-like."


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
        return
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


def write_io_log(result: Step10Result, log_path: str) -> None:
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
    if result.output_md:
        append_log(log_path, f"output_md={relative_to_paper(result.paper_dir, result.output_md)}")
    if result.note:
        append_log(log_path, f"note={result.note}")
    append_log(log_path, "")


def collect_input_items(path: str, source_kind: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    for item in flatten_property_entries(parse_property_markdown(path)):
        sample = str(item.get("sample", "")).strip()
        tag = str(item.get("tag", "")).strip()
        sentence = str(item.get("sentence", "")).strip()
        if not sample or not tag or not sentence:
            continue
        item_copy = dict(item)
        item_copy["source_kind"] = source_kind
        out.append(item_copy)
    return out


def review_one_item(
    model,
    *,
    source_kind: str,
    tag: str,
    target_sample: SampleInfo,
    evidence_text: str,
    current_sentence: str,
    other_samples_context: str,
    log_path: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    votes: int,
) -> ReviewVote:
    vote_results: List[ReviewVote] = []
    for vote_index in range(1, votes + 1):
        payload = call_llm_json(
            model,
            prompt=build_review_prompt(
                source_kind=source_kind,
                tag=tag,
                target_sample=target_sample,
                evidence_text=evidence_text,
                current_sentence=current_sentence,
                other_samples_context=other_samples_context,
            ),
            log_path=log_path,
            label=f"STEP10_REVIEW_{source_kind.upper()}_{tag}_vote{vote_index}",
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            validator=validate_review_response,
        )
        vote_results.append(
            ReviewVote(
                verdict=str(payload.get("verdict", "")).strip().upper(),
                reason=str(payload.get("reason", "")).strip().upper() or "OTHER",
            )
        )
    return aggregate_review_votes(vote_results)


def rewrite_one_item(
    model,
    *,
    source_kind: str,
    tag: str,
    target_sample: SampleInfo,
    evidence_text: str,
    current_sentence: str,
    review_reason: str,
    other_samples_context: str,
    rewrite_log_path: str,
    recheck_log_path: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> str:
    payload = call_llm_json(
        model,
        prompt=build_rewrite_prompt(
            source_kind=source_kind,
            tag=tag,
            target_sample=target_sample,
            evidence_text=evidence_text,
            current_sentence=current_sentence,
            review_reason=review_reason,
            other_samples_context=other_samples_context,
        ),
        log_path=rewrite_log_path,
        label=f"STEP10_REWRITE_{source_kind.upper()}_{tag}",
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        validator=validate_rewrite_response,
    )
    sentence = str(payload.get("sentence", "")).strip()
    if not sentence or sentence.upper() == "NONE" or target_sample.name not in sentence:
        return ""

    recheck_payload = call_llm_json(
        model,
        prompt=build_recheck_prompt(
            source_kind=source_kind,
            tag=tag,
            target_sample=target_sample,
            evidence_text=evidence_text,
            rewritten_sentence=sentence,
            other_samples_context=other_samples_context,
        ),
        log_path=recheck_log_path,
        label=f"STEP10_RECHECK_{source_kind.upper()}_{tag}",
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        validator=validate_recheck_response,
    )
    keep_raw = recheck_payload.get("keep", False)
    keep = keep_raw if isinstance(keep_raw, bool) else parse_boolean_answer(str(keep_raw), default=False)
    return sentence if keep else ""


def process_one_paper(
    paper_dir: str,
    *,
    model,
    skip_existing: bool,
    temperature: float,
    max_tokens: int,
    retries: int,
    votes: int,
) -> Step10Result:
    paper_id = paper_id_from_dir(paper_dir)
    if paper_id is None:
        return Step10Result("", paper_dir, "SKIP_INVALID_DIR", "", "", "", "", "", "", "Directory name does not start with a paper id.")

    input_main = resolve_main_input_path(paper_dir, paper_id)
    input_app = resolve_bound_input_path(paper_dir, paper_id, "app")
    input_vs = resolve_bound_input_path(paper_dir, paper_id, "vs")
    decision_csv = resolve_decision_csv_path(paper_dir, paper_id)
    letter_csv = os.path.join(paper_dir, "Synthesis", "letter_table", f"{paper_id}.csv")
    output_md = stage_markdown_path(paper_dir, OUTPUT_STAGE_DIR, paper_id, kind="main")
    output_dir = os.path.dirname(output_md)
    ensure_dir(output_dir)
    log_paths = build_log_paths(output_dir, paper_id)

    if skip_existing and os.path.exists(output_md):
        result = Step10Result(
            paper_id,
            paper_dir,
            "SKIP_EXISTS",
            input_main,
            input_app,
            input_vs,
            decision_csv,
            letter_csv,
            output_md,
            "Step 10 output already exists.",
        )
        write_io_log(result, log_paths["io"])
        return result

    if not any(os.path.exists(path) for path in (input_main, input_app, input_vs)):
        result = Step10Result(
            paper_id,
            paper_dir,
            "SKIP_NO_INPUT_MD",
            input_main,
            input_app,
            input_vs,
            decision_csv,
            letter_csv,
            output_md,
            "Missing all Step 8 and Step 9 markdown inputs.",
        )
        write_io_log(result, log_paths["io"])
        return result

    if not os.path.exists(decision_csv):
        result = Step10Result(
            paper_id,
            paper_dir,
            "SKIP_NO_DECISION_CSV",
            input_main,
            input_app,
            input_vs,
            decision_csv,
            letter_csv,
            output_md,
            "Missing decision csv.",
        )
        write_io_log(result, log_paths["io"])
        return result

    if not os.path.exists(letter_csv):
        result = Step10Result(
            paper_id,
            paper_dir,
            "SKIP_NO_SYNTHESIS_TABLE",
            input_main,
            input_app,
            input_vs,
            decision_csv,
            letter_csv,
            output_md,
            "Missing synthesis letter table.",
        )
        write_io_log(result, log_paths["io"])
        return result

    letter_samples = read_letter_table_samples(paper_dir, paper_id)
    if not letter_samples:
        result = Step10Result(
            paper_id,
            paper_dir,
            "SKIP_EMPTY_SYNTHESIS_TABLE",
            input_main,
            input_app,
            input_vs,
            decision_csv,
            letter_csv,
            output_md,
            "The synthesis letter table did not contain usable sample names.",
        )
        write_io_log(result, log_paths["io"])
        return result

    sample_catalog = build_sample_catalog(letter_samples)
    other_samples_context = build_other_samples_context(letter_samples)
    _, sid_to_text = build_decision_sid_maps(decision_csv)

    input_items = (
        collect_input_items(input_main, "main")
        + collect_input_items(input_app, "app")
        + collect_input_items(input_vs, "vs")
    )

    merged_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_keys = set()
    kept_count = 0
    rewritten_count = 0
    dropped_count = 0

    for item in tqdm(input_items, desc=f"Step10: review-final {paper_id}", leave=False):
        sample_name = str(item.get("sample", "")).strip()
        tag = str(item.get("tag", "")).strip()
        sentence = str(item.get("sentence", "")).strip()
        source_kind = str(item.get("source_kind", "main")).strip()
        if sample_name not in sample_catalog or tag not in SUPPORTED_TAGS or not sentence:
            dropped_count += 1
            continue

        target_sample = sample_catalog[sample_name]
        window_sids = [int(value) for value in item.get("window_sids", []) or [] if str(value).isdigit()]
        evidence_lines = build_evidence_lines_from_sids(
            window_sids,
            sid_to_text,
            fallback_lines=item.get("evidence_lines", []) or [],
        )
        evidence_text = " ".join(str(line).strip() for line in evidence_lines if str(line).strip()).strip()
        if not evidence_text:
            dropped_count += 1
            continue

        review = review_one_item(
            model,
            source_kind=source_kind,
            tag=tag,
            target_sample=target_sample,
            evidence_text=evidence_text,
            current_sentence=sentence,
            other_samples_context=other_samples_context,
            log_path=log_paths["review"],
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            votes=votes,
        )

        final_sentence = ""
        if review.verdict == "KEEP":
            final_sentence = sentence
            kept_count += 1
        elif review.verdict == "REWRITE":
            final_sentence = rewrite_one_item(
                model,
                source_kind=source_kind,
                tag=tag,
                target_sample=target_sample,
                evidence_text=evidence_text,
                current_sentence=sentence,
                review_reason=review.reason,
                other_samples_context=other_samples_context,
                rewrite_log_path=log_paths["rewrite"],
                recheck_log_path=log_paths["recheck"],
                temperature=temperature,
                max_tokens=max_tokens,
                retries=retries,
            )
            if final_sentence:
                rewritten_count += 1
            else:
                dropped_count += 1
        else:
            dropped_count += 1

        if not final_sentence:
            continue

        append_item(
            merged_items,
            seen_keys,
            sample=sample_name,
            para_id=int(item.get("para_id", 0) or 0),
            win_level=str(item.get("win_level", "") or "").strip(),
            window_sids=window_sids,
            evidence_lines=evidence_lines,
            tag=tag,
            sentence=final_sentence,
        )

    write_text(output_md, render_property_markdown(merged_items, property_label="Property abstract"))
    status = "PROCESSED_EMPTY" if not merged_items else "PROCESSED"
    note = (
        f"input_items={len(input_items)}; kept={kept_count}; rewritten={rewritten_count}; "
        f"dropped={dropped_count}; final_items={sum(len(items) for items in merged_items.values())}"
    )
    result = Step10Result(
        paper_id=paper_id,
        paper_dir=paper_dir,
        status=status,
        input_main=input_main,
        input_app=input_app,
        input_vs=input_vs,
        input_decision_csv=decision_csv,
        input_letter_csv=letter_csv,
        output_md=output_md,
        note=note,
    )
    write_io_log(result, log_paths["io"])
    return result


def write_root_logs(mining_root: str, results: Sequence[Step10Result]) -> None:
    main_log_path = os.path.join(mining_root, "step10_review_final_properties.log")
    error_log_path = os.path.join(mining_root, "step10_review_final_properties_error.log")

    with open(main_log_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Step 10 final property review\n\nGenerated at {timestamp_now()}\n\n")
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
            if result.output_md:
                handle.write(f"  output_md={relative_to_root(mining_root, result.output_md)}\n")
            if result.note:
                handle.write(f"  note={result.note}\n")

    with open(error_log_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Step 10 final property review issues\n\nGenerated at {timestamp_now()}\n\n")
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
            if result.output_md:
                handle.write(f"  output_md={relative_to_root(mining_root, result.output_md)}\n")
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
    votes: int = DEFAULT_VOTES,
) -> None:
    if lmstudio_llm is None:
        raise RuntimeError("lmstudio is not installed. Install or configure LM Studio before running Step 10.")

    root = ensure_root_exists(mining_root)
    model = lmstudio_llm(model_name)
    results: List[Step10Result] = []
    for paper_dir in tqdm(iter_paper_dirs(root, paper_ids=paper_ids), desc="Step10: review-final-properties"):
        try:
            results.append(
                process_one_paper(
                    paper_dir,
                    model=model,
                    skip_existing=skip_existing,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retries=retries,
                    votes=votes,
                )
            )
        except Exception as exc:
            paper_id = paper_id_from_dir(paper_dir) or ""
            results.append(
                Step10Result(
                    paper_id=paper_id,
                    paper_dir=paper_dir,
                    status="SKIP_FATAL",
                    input_main=resolve_main_input_path(paper_dir, paper_id) if paper_id else "",
                    input_app=resolve_bound_input_path(paper_dir, paper_id, "app") if paper_id else "",
                    input_vs=resolve_bound_input_path(paper_dir, paper_id, "vs") if paper_id else "",
                    input_decision_csv=resolve_decision_csv_path(paper_dir, paper_id) if paper_id else "",
                    input_letter_csv=os.path.join(paper_dir, "Synthesis", "letter_table", f"{paper_id}.csv") if paper_id else "",
                    output_md=stage_markdown_path(paper_dir, OUTPUT_STAGE_DIR, paper_id, kind="main") if paper_id else "",
                    note=repr(exc),
                )
            )
    write_root_logs(root, results)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Property preprocessing Step 10: review and merge final properties.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Re-run even if Step 10 output already exists.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model name for Step 10.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature for Step 10.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="LLM max tokens for Step 10.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Max JSON-call retries per LLM request.")
    parser.add_argument("--votes", type=int, default=DEFAULT_VOTES, help="Number of review votes per input item.")
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
