#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step 7: resolve multi-sample attribution for Step 6 main property markdown."""

import argparse
import hashlib
import os
import shutil
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
    candidates_from_sids,
    ensure_dir,
    ensure_root_exists,
    iter_paper_dirs,
    paper_id_from_dir,
    parse_json_object_text,
    parse_property_markdown,
    read_letter_table_samples,
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
DEFAULT_STEP1_TEMPERATURE = 0.25
DEFAULT_STEP2_TEMPERATURE = 0.25
DEFAULT_VOTE_TEMPERATURE = 0.10
DEFAULT_STEP1_MAX_TOKENS = 1800
DEFAULT_STEP2_MAX_TOKENS = 2600
DEFAULT_VOTE_MAX_TOKENS = 800
DEFAULT_RETRIES = 5
DEFAULT_STEP2_MAX_LOOPS = 5
END_SENTINEL = "<END_OF_JSON>"
SUPPORTED_TAGS = {"Ex", "Em", "QY", "lifetime", "ExDep", "Chiral", "CPL"}
MAIN_LOG_STATUSES = {
    "COPIED_SINGLE",
    "SKIP_COPY_EXISTS",
    "PROCESSED_MULTI",
    "PROCESSED_MULTI_STEP2_MAXLOOP_REJECT",
}
_SMART_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


@dataclass
class Step2Outcome:
    sentence: str
    reason: str
    loops_used: int


@dataclass
class Step7Result:
    paper_id: str
    paper_dir: str
    status: str
    input_md: str
    output_md: str
    note: str = ""


def build_log_paths(paper_dir: str, paper_id: str) -> Dict[str, str]:
    out_dir = os.path.join(paper_dir, "property", "abstract_clean")
    return {
        "io": os.path.join(out_dir, f"{paper_id}_step7_multisample.io.log"),
        "llm": os.path.join(out_dir, f"{paper_id}_step7_multisample.llm.log"),
        "vote": os.path.join(out_dir, f"{paper_id}_step7_multisample.vote.log"),
        "reject": os.path.join(out_dir, f"{paper_id}_step7_multisample.reject.log"),
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


def validate_step1_response(payload: Dict[str, Any]) -> Optional[str]:
    action = str(payload.get("action", "")).strip().upper()
    if action not in {"KEEP", "REFINE"}:
        return f"Invalid action: {action!r}"
    if action == "REFINE" and not isinstance(payload.get("candidates", []), list):
        return "Field 'candidates' must be a list when action is REFINE."
    return None


def validate_step2_response(payload: Dict[str, Any]) -> Optional[str]:
    sentence = payload.get("sentence", None)
    if sentence is None:
        return "Missing field 'sentence'."
    if not isinstance(sentence, str):
        return "Field 'sentence' must be a string."
    return None


def build_step1_prompt(
    *,
    tag: str,
    current_assigned_sample: str,
    refined_sentence: str,
    evidence_text: str,
    candidate_samples: Sequence[SampleInfo],
) -> str:
    candidates_block = "\n".join(
        f"{index}. {sample.name}: {sample.desc}"
        for index, sample in enumerate(candidate_samples, start=1)
    )
    cand_order = ", ".join(sample.name for sample in candidate_samples) if candidate_samples else "(none)"

    return f"""You are an expert reviewer for extracting photoluminescence (PL) properties of carbon dots in a MULTI-SAMPLE context.

TASK:
Decide whether the refined sentence can be kept AS-IS for the currently assigned sample, or it must be refined again.

CRITICAL OUTPUT RULES:
- Output ONLY ONE JSON object. No explanation. No extra text. No code fences.
- The very last characters must be {END_SENTINEL}.

ALLOWED OUTPUT ONLY (choose exactly one):
(1) KEEP:
{{"action":"KEEP"}}{END_SENTINEL}

(2) REFINE (candidates must be non-empty and chosen from the provided list):
{{"action":"REFINE","candidates":["sampleA","sampleB"]}}{END_SENTINEL}

HARD CONDITIONS for action="KEEP" (ALL must be satisfied; otherwise REFINE):
1) The sentence explicitly names the currently assigned sample.
2) The sentence is unambiguously attributable to ONE sample in this multi-sample paper.
3) For numeric tags (Ex/Em/QY/lifetime): the sentence contains a concrete numeric value and does NOT mix multiple values unless each value is explicitly tied to stated measurement conditions.
4) No mismatch risk: if the evidence suggests a different prepared system or sample than the assigned one, you must REFINE.

HARD RULE - LIST / "respectively" ORDER ALIGNMENT:
- If the evidence contains an ordered list of values and uses cues like "respectively", then you MUST align values to samples by the given candidate sample order:
  candidate_sample_order = [{cand_order}]
  1st candidate -> 1st value, 2nd candidate -> 2nd value, and so on.
- KEEP is allowed ONLY if the numeric value or claim in refined_sentence matches the value aligned to the currently assigned sample.
- If the aligned value cannot be determined with high confidence, or refined_sentence uses the wrong aligned value, you MUST output REFINE.

How to choose candidates when action="REFINE":
- Output a NON-EMPTY list of names chosen ONLY from the provided candidate list below.
- Prefer names that appear to be discussed in the evidence.
- If unclear, output ALL candidates in the same order as provided.

INPUTS:
tag = "{tag}"
currently_assigned_sample = "{current_assigned_sample}"
refined_sentence = "{refined_sentence}"

evidence_text:
{evidence_text}

candidate_samples (ORDERED list; MUST preserve this order for alignment):
{candidates_block}
"""


def call_step1_keep_or_refine(
    model,
    *,
    tag: str,
    current_assigned_sample: str,
    refined_sentence: str,
    evidence_text: str,
    candidate_samples: Sequence[SampleInfo],
    log_path: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> Dict[str, Any]:
    fallback = {"action": "REFINE", "candidates": [sample.name for sample in candidate_samples]}
    try:
        payload = call_llm_json(
            model,
            prompt=build_step1_prompt(
                tag=tag,
                current_assigned_sample=current_assigned_sample,
                refined_sentence=refined_sentence,
                evidence_text=evidence_text,
                candidate_samples=candidate_samples,
            ),
            log_path=log_path,
            label="STEP1_KEEP_OR_REFINE",
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            validator=validate_step1_response,
        )
    except Exception:
        return fallback

    action = str(payload.get("action", "")).strip().upper()
    if action == "KEEP":
        return {"action": "KEEP"}

    allowed = {sample.name for sample in candidate_samples}
    names = payload.get("candidates", [])
    filtered = [str(name).strip() for name in names if str(name).strip() in allowed] if isinstance(names, list) else []
    return {"action": "REFINE", "candidates": filtered or fallback["candidates"]}


def refine_property_spec(tag: str, sample_name: str = "the sample") -> str:
    common_guard = (
        "- Standalone-only: extract properties ONLY for the standalone prepared sample named in [Sample name], not for any combined, mixed, complex, composite, or system state.\n"
        "- If the evidence describes the sample AFTER adding, binding, mixing, or introducing something, keep ONLY the baseline value for the standalone sample BEFORE or without that addition.\n"
        "  If the baseline value is not explicitly stated, output NONE.\n"
        "- If the evidence reports values under multiple synthesis or preparation conditions, treat them as different prepared batches.\n"
        f"  You MUST extract only the value or values that match the synthesis or preparation conditions in the [Sample description] for {sample_name}.\n"
        "  If you cannot uniquely match them, output NONE.\n"
        "- Ignore values that are clearly literature examples or previous reports unless explicitly stated as this work for this exact sample.\n"
    )
    common_numeric = (
        f"- Keep ONLY numeric value(s) explicitly stated in the evidence_text and assignable to {sample_name} for this tag.\n"
        "- NO INFERENCE: every number you output MUST appear in the evidence text.\n"
        "- You may output one or multiple discrete values ONLY when the evidence explicitly provides multiple values for the SAME standalone sample under different stated measurement conditions.\n"
        "- If a value is reported with conditions, include those stated conditions in the same sentence and do not invent missing conditions.\n"
        "- Do NOT output broad numeric ranges unless the evidence explicitly reports a range for this same standalone sample under a defined condition.\n"
    )

    if tag == "Em":
        return (
            "[Em - Emission wavelength]\n"
            + common_guard
            + common_numeric
            + "- The value(s) must refer to emission, fluorescence, or photoluminescence peak or maximum of the standalone sample, not excitation or absorption.\n"
            + "- If the evidence describes excitation-dependent emission peak shifts for the same sample, do NOT extract the endpoints as intrinsic Em; output NONE unless a fixed Em at a fixed excitation is explicitly stated.\n"
            + "- 'Monitored at XXX nm' in sensing or calibration is a readout channel, not an intrinsic Em peak, unless explicitly labeled as a peak or maximum.\n"
        )
    if tag == "Ex":
        return (
            "[Ex - Excitation wavelength]\n"
            + common_guard
            + common_numeric
            + "- The value(s) must refer to excitation wavelength(s) used or optimal for the sample's intrinsic photoluminescence, not absorption bands.\n"
            + "- If the evidence describes a shift of optimal or maximum excitation for the same sample as conditions change, do NOT extract the endpoints as intrinsic Ex; output NONE unless a fixed Ex under a fixed condition is explicitly stated.\n"
            + "- 'Under 365 nm UV light' for visual observation must NOT be treated as Ex.\n"
        )
    if tag == "QY":
        return (
            "[QY - Photoluminescence quantum yield]\n"
            + common_guard
            + common_numeric
            + "- QY is usually a percentage.\n"
            + "- If the evidence reports a change from baseline to after addition, keep ONLY the baseline value without the added species.\n"
            + "- If QY is only given for composites, analyte-bound, or mixed states and never for the standalone sample, output NONE.\n"
        )
    if tag == "lifetime":
        return (
            "[lifetime - PL lifetime]\n"
            + common_guard
            + common_numeric
            + "- Report lifetime value(s) exactly as stated with correct units; do NOT invent averages.\n"
            + "- If the evidence reports a change from baseline to after addition, keep ONLY the baseline value without the added species.\n"
            + "- If lifetime is only given for composites, analyte-bound, or mixed states and never for the standalone sample, output NONE.\n"
        )
    if tag == "ExDep":
        return (
            "[ExDep - Excitation dependence]\n"
            + common_guard
            + f"- Apply ONLY when the evidence explicitly states excitation-dependent or excitation-independent behavior for the standalone sample {sample_name}.\n"
            "- Multiple Ex-Em pairs alone are NOT sufficient; there must be an explicit dependence or independence statement.\n"
            "- Output a qualitative one-sentence statement and do not output numeric ranges.\n"
        )
    if tag == "Chiral":
        return (
            "[Chiral - Sample chirality]\n"
            + common_guard
            + f"- Report chirality only when {sample_name} itself is explicitly described as chiral.\n"
            "- Do NOT infer chirality from chiral precursors, ligands, or matrices alone.\n"
            "- If unclear or only described for a combined state, output NONE.\n"
        )
    if tag == "CPL":
        return (
            "[CPL - Circularly polarized luminescence]\n"
            + common_guard
            + f"- Report CPL only when the evidence explicitly states that {sample_name} shows circularly polarized luminescence or emission.\n"
            "- Do NOT infer CPL from ordinary PL or from chirality alone.\n"
            "- If unclear or only described for a combined state, output NONE.\n"
        )
    raise ValueError(f"Unsupported tag for Step 7 refinement: {tag}")


def build_step2_refine_prompt(
    *,
    tag: str,
    target_sample: SampleInfo,
    reference_sample_names: Sequence[str],
    evidence_text: str,
) -> str:
    reference_block = ", ".join(reference_sample_names) if reference_sample_names else "(none)"
    return f"""You are refining ONE photoluminescence (PL) property for ONE target sample in a MULTI-SAMPLE paper.

CRITICAL OUTPUT RULES:
- Output ONLY ONE JSON object. No explanation. No extra text. No code fences.
- The very last characters must be {END_SENTINEL}.
- The JSON must be exactly:
  {{"sentence":"<ONE single English sentence, or NONE>"}}{END_SENTINEL}

Hard constraints:
1) Output must be either one single complete English sentence or the exact string "NONE".
2) If output is a sentence, it MUST explicitly include the target sample name "{target_sample.name}".
3) Use ONLY information explicitly supported by evidence_text. NO INFERENCE.
4) Do NOT use placeholders like "(conditions)" or "under certain conditions" unless the exact condition text is explicitly present in evidence_text and you include that exact condition.
5) For numeric tags (Ex, Em, QY, lifetime): the output MUST contain exactly ONE target numeric value for this tag.
6) Multi-sample safety: if the evidence cannot uniquely support this tag for the target sample, output NONE.

ORDER-ALIGNMENT RULE:
- reference_samples_order = [{reference_block}]
- If evidence_text presents an ordered list of values for multiple samples and uses cues like "respectively" or an explicit aligned list or table-like mapping,
  then you MUST map the k-th value to the k-th sample in reference_samples_order.
- If such mapping cues are absent, or the list length does not match the sample list, or any ambiguity exists, output NONE.

Property template for this tag (MUST follow):
{refine_property_spec(tag, sample_name=target_sample.name)}

Context:
tag = "{tag}"
target_sample_name = "{target_sample.name}"
target_sample_description = "{target_sample.desc}"
reference_samples_order = [{reference_block}]

evidence_text:
{evidence_text}
"""


def append_step2_reject_log(
    reject_log_path: str,
    *,
    key: str,
    tag: str,
    target_sample: str,
    last_sentence: str,
    loops_used: int,
) -> None:
    append_log(
        reject_log_path,
        f"[{timestamp_now()}] STEP2_MAX_LOOP_REJECT key={key} tag={tag} sample={target_sample} loops={loops_used}",
    )
    if last_sentence:
        append_log(reject_log_path, f"last_sentence={last_sentence}")
    append_log(reject_log_path, "")


def call_step2_refine_one_sample(
    model,
    *,
    tag: str,
    target_sample: SampleInfo,
    reference_sample_names: Sequence[str],
    evidence_text: str,
    candidate_samples: Sequence[SampleInfo],
    log_path: str,
    reject_log_path: str,
    step1_temperature: float,
    step2_temperature: float,
    step1_max_tokens: int,
    step2_max_tokens: int,
    retries: int,
    max_refine_loops: int,
    key_for_log: str,
) -> Step2Outcome:
    last_sentence = ""
    saw_reject = False

    for loop_index in range(1, max_refine_loops + 1):
        try:
            payload = call_llm_json(
                model,
                prompt=build_step2_refine_prompt(
                    tag=tag,
                    target_sample=target_sample,
                    reference_sample_names=reference_sample_names,
                    evidence_text=evidence_text,
                ),
                log_path=log_path,
                label=f"STEP2_REFINE_ONE_SAMPLE(loop={loop_index})",
                temperature=step2_temperature,
                max_tokens=step2_max_tokens,
                retries=retries,
                validator=validate_step2_response,
            )
        except Exception:
            continue

        sentence = str(payload.get("sentence", "")).strip()
        if not sentence or sentence.upper() == "NONE":
            last_sentence = "NONE"
            continue

        last_sentence = sentence
        step1 = call_step1_keep_or_refine(
            model,
            tag=tag,
            current_assigned_sample=target_sample.name,
            refined_sentence=sentence,
            evidence_text=evidence_text,
            candidate_samples=candidate_samples,
            log_path=log_path,
            temperature=step1_temperature,
            max_tokens=step1_max_tokens,
            retries=retries,
        )
        if str(step1.get("action", "")).strip().upper() == "KEEP":
            return Step2Outcome(sentence=sentence, reason="OK", loops_used=loop_index)

        saw_reject = True

    if saw_reject:
        append_step2_reject_log(
            reject_log_path,
            key=key_for_log,
            tag=tag,
            target_sample=target_sample.name,
            last_sentence=last_sentence,
            loops_used=max_refine_loops,
        )
        return Step2Outcome(sentence="NONE", reason="MAX_LOOP_REJECT", loops_used=max_refine_loops)

    return Step2Outcome(sentence="NONE", reason="MODEL_NONE", loops_used=max_refine_loops)


def build_vote_prompt(
    *,
    tag: str,
    sample_name: str,
    evidence_text: str,
    sentences: Sequence[str],
    reference_samples_order: Sequence[str],
) -> str:
    items = "\n".join(f"{index}) {sentence}" for index, sentence in enumerate(sentences, start=1))
    refs = ", ".join(reference_samples_order) if reference_samples_order else "(none)"

    return f"""You are selecting the single BEST sentence to keep for ONE sample and ONE tag, based on the evidence.

CRITICAL OUTPUT RULES:
- Output ONLY ONE JSON object. No explanation. No extra text. No code fences.
- The very last characters must be {END_SENTINEL}.
- Output must be exactly:
  {{"pick": <INTEGER>}}{END_SENTINEL}

Selection rules:
1) The chosen sentence MUST be supported by the evidence_text. NO INFERENCE.
2) The chosen sentence MUST be about sample "{sample_name}" and must not attribute values to the wrong sample.
3) Reject vague placeholders such as "conditions" without explicit condition text.
4) ORDER ALIGNMENT:
   - reference_samples_order = [{refs}]
   - If evidence_text uses cues like "respectively", ordered lists, or aligned mapping, you MUST enforce the same order.
5) Prefer the sentence that is most precise and information-complete without adding unsupported details.

INPUT:
tag = "{tag}"
sample = "{sample_name}"
reference_samples_order = [{refs}]

evidence_text:
{evidence_text}

candidate_sentences:
{items}
"""


def append_vote_log(
    vote_log_path: str,
    *,
    key: str,
    tag: str,
    sample: str,
    picks: Sequence[int],
    final_pick: int,
    sentences: Sequence[str],
    chosen_sentence: str,
) -> None:
    append_log(vote_log_path, f"[{timestamp_now()}] VOTE_FINAL key={key} tag={tag} sample={sample}")
    append_log(vote_log_path, f"picks={list(picks)} final={final_pick}")
    for index, sentence in enumerate(sentences, start=1):
        append_log(vote_log_path, f"{index}) {sentence}")
    append_log(vote_log_path, f"CHOSEN({final_pick})={chosen_sentence}")
    append_log(vote_log_path, "")


def call_llm_vote_pick_once(
    model,
    *,
    tag: str,
    sample_name: str,
    evidence_text: str,
    sentences: Sequence[str],
    reference_samples_order: Sequence[str],
    log_path: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> Optional[int]:
    if not sentences:
        return None

    payload = call_llm_json(
        model,
        prompt=build_vote_prompt(
            tag=tag,
            sample_name=sample_name,
            evidence_text=evidence_text,
            sentences=sentences,
            reference_samples_order=reference_samples_order,
        ),
        log_path=log_path,
        label="STEP7_VOTE_PICK",
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
    )
    try:
        pick = int(payload.get("pick"))
    except Exception:
        return None
    return pick if 1 <= pick <= len(sentences) else None


def llm_vote_pick_majority(
    model,
    *,
    tag: str,
    sample_name: str,
    evidence_text: str,
    sentences: Sequence[str],
    reference_samples_order: Sequence[str],
    log_path: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> Tuple[int, List[int]]:
    picks: List[int] = []
    for _ in range(3):
        pick = call_llm_vote_pick_once(
            model,
            tag=tag,
            sample_name=sample_name,
            evidence_text=evidence_text,
            sentences=sentences,
            reference_samples_order=reference_samples_order,
            log_path=log_path,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
        )
        if pick is not None:
            picks.append(pick)

    if len(picks) < 2:
        return 1, picks

    winner, count = Counter(picks).most_common(1)[0]
    if count >= 2:
        return winner, picks

    pick_four = call_llm_vote_pick_once(
        model,
        tag=tag,
        sample_name=sample_name,
        evidence_text=evidence_text,
        sentences=sentences,
        reference_samples_order=reference_samples_order,
        log_path=log_path,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
    )
    if pick_four is not None:
        picks.append(pick_four)
        winner, count = Counter(picks).most_common(1)[0]
        if count >= 2:
            return winner, picks

    return 1, picks


def consolidate_items_by_sample_with_llm_vote(
    model,
    items_by_sample: Dict[str, List[Dict[str, Any]]],
    *,
    llm_log_path: str,
    vote_log_path: str,
    vote_temperature: float,
    vote_max_tokens: int,
    retries: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    vote_stats = {"votes": 0, "votes_used_4th": 0}

    for sample, items in (items_by_sample or {}).items():
        grouped: Dict[Tuple[int, str, Tuple[int, ...], str], List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            key = (
                int(item.get("para_id", 0) or 0),
                str(item.get("win_level", "") or "").strip(),
                tuple(int(value) for value in item.get("window_sids", []) or []),
                str(item.get("tag", "")).strip(),
            )
            if key[3]:
                grouped[key].append(item)

        for (para_id, win_level, sids_tuple, tag), group in grouped.items():
            if len(group) == 1:
                out[sample].append(group[0])
                continue

            refined = [item for item in group if str(item.get("source", "")).upper() == "REFINED"]
            candidates = refined if refined else group
            if len(candidates) == 1:
                out[sample].append(candidates[0])
                continue

            indexed_sentences = [
                (index, str(item.get("sentence", "")).strip())
                for index, item in enumerate(candidates)
                if str(item.get("sentence", "")).strip()
            ]
            if len(indexed_sentences) <= 1:
                out[sample].append(candidates[indexed_sentences[0][0]] if indexed_sentences else candidates[0])
                continue

            vote_stats["votes"] += 1
            evidence_text = " ".join(
                str(line).strip() for line in candidates[0].get("evidence_lines", []) or [] if str(line).strip()
            ).strip()
            reference_order = candidates[0].get("align_order", []) or []
            clean_sentences = [sentence for _, sentence in indexed_sentences]
            pick, picks = llm_vote_pick_majority(
                model,
                tag=tag,
                sample_name=sample,
                evidence_text=evidence_text,
                sentences=clean_sentences,
                reference_samples_order=reference_order,
                log_path=llm_log_path,
                temperature=vote_temperature,
                max_tokens=vote_max_tokens,
                retries=retries,
            )
            if len(picks) >= 4:
                vote_stats["votes_used_4th"] += 1

            pick_position = pick - 1 if 1 <= pick <= len(indexed_sentences) else 0
            chosen_index = indexed_sentences[pick_position][0]
            chosen_item = candidates[chosen_index]
            append_vote_log(
                vote_log_path,
                key=f"para={para_id};win={win_level};sids={','.join(str(value) for value in sids_tuple)}",
                tag=tag,
                sample=sample,
                picks=picks,
                final_pick=pick,
                sentences=clean_sentences,
                chosen_sentence=str(chosen_item.get("sentence", "")).strip(),
            )
            out[sample].append(chosen_item)

    return out, vote_stats


def make_property_item(
    *,
    sample: str,
    para_id: int,
    win_level: str,
    window_sids: Sequence[int],
    evidence_lines: Sequence[str],
    tag: str,
    sentence: str,
    tag_order: int,
    source: str,
    align_order: Sequence[str],
) -> Dict[str, Any]:
    item = build_property_item(
        sample=sample,
        para_id=para_id,
        win_level=win_level,
        window_sids=window_sids,
        evidence_lines=evidence_lines,
        tag=tag,
        sentence=sentence,
        tag_order=tag_order,
    )
    item["source"] = source
    item["align_order"] = list(align_order)
    return item


def write_io_log(result: Step7Result, log_path: str) -> None:
    append_log(log_path, f"[{timestamp_now()}] paper={result.paper_id} status={result.status}")
    if result.input_md:
        append_log(log_path, f"input={relative_to_paper(result.paper_dir, result.input_md)}")
    if result.output_md:
        append_log(log_path, f"output={relative_to_paper(result.paper_dir, result.output_md)}")
    if result.note:
        append_log(log_path, f"note={result.note}")
    append_log(log_path, "")


def process_one_paper(
    paper_dir: str,
    *,
    model,
    skip_existing: bool,
    step1_temperature: float,
    step2_temperature: float,
    vote_temperature: float,
    step1_max_tokens: int,
    step2_max_tokens: int,
    vote_max_tokens: int,
    retries: int,
    step2_max_loops: int,
) -> Step7Result:
    paper_id = paper_id_from_dir(paper_dir)
    if paper_id is None:
        return Step7Result("", paper_dir, "SKIP_INVALID_DIR", "", "", "Directory name does not start with a paper id.")

    in_md = os.path.join(paper_dir, "property", "abstract_clean", f"{paper_id}.md")
    out_md = os.path.join(paper_dir, "property", "abstract_clean", f"{paper_id}_main.md")
    log_paths = build_log_paths(paper_dir, paper_id)
    ensure_dir(os.path.dirname(out_md))

    if not os.path.exists(in_md):
        result = Step7Result(paper_id, paper_dir, "SKIP_NO_INPUT_MD", in_md, out_md, "Missing Step 6 main markdown.")
        write_io_log(result, log_paths["io"])
        return result

    if not read_text(in_md).strip():
        result = Step7Result(paper_id, paper_dir, "SKIP_EMPTY_INPUT_MD", in_md, out_md, "Input markdown is empty.")
        write_io_log(result, log_paths["io"])
        return result

    if skip_existing and os.path.exists(out_md):
        result = Step7Result(paper_id, paper_dir, "SKIP_COPY_EXISTS", in_md, out_md, "Step 7 output already exists.")
        write_io_log(result, log_paths["io"])
        return result

    letter_samples = read_letter_table_samples(paper_dir, paper_id)
    if not letter_samples:
        result = Step7Result(paper_id, paper_dir, "SKIP_NO_LETTER_TABLE", in_md, out_md, "Missing or empty letter table.")
        write_io_log(result, log_paths["io"])
        return result

    if len(letter_samples) < 2:
        shutil.copy2(in_md, out_md)
        result = Step7Result(paper_id, paper_dir, "COPIED_SINGLE", in_md, out_md, "Single-sample paper. Copied Step 6 main markdown.")
        write_io_log(result, log_paths["io"])
        return result

    decision_csv = resolve_decision_csv_path(paper_dir, paper_id)
    sid_to_names: Dict[int, str] = {}
    sid_to_text: Dict[int, str] = {}
    decision_note = ""
    if os.path.exists(decision_csv):
        try:
            sid_to_names, sid_to_text = build_decision_sid_maps(decision_csv)
            decision_note = f"decision_csv={relative_to_paper(paper_dir, decision_csv)}"
        except Exception as exc:
            decision_note = f"decision_csv_error={exc!r}; fallback=markdown_evidence_only"
    else:
        decision_note = "decision_csv_missing; fallback=markdown_evidence_only"

    entries = parse_property_markdown(in_md)
    if not entries:
        result = Step7Result(paper_id, paper_dir, "SKIP_NO_ENTRIES", in_md, out_md, "Parsed 0 property entries.")
        write_io_log(result, log_paths["io"])
        return result

    out_items_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    sample_map = {sample.name: sample for sample in letter_samples}
    step2_cache: Dict[Tuple[str, str, str], Step2Outcome] = {}
    step2_maxloop_reject_count = 0

    for entry in tqdm(entries, desc=f"Step7: resolve {paper_id}", leave=False):
        assigned_sample = str(entry.get("sample", "")).strip()
        if not assigned_sample:
            continue

        para_id = int(entry.get("para_id", 0) or 0)
        win_level = str(entry.get("win_level", "") or "").strip()
        window_sids = [int(value) for value in entry.get("window_sids", []) or [] if str(value).isdigit()]
        markdown_evidence = [str(line) for line in entry.get("evidence_lines", []) or []]
        evidence_lines = build_evidence_lines_from_sids(window_sids, sid_to_text, fallback_lines=markdown_evidence)
        evidence_text = " ".join(str(line).strip() for line in evidence_lines if str(line).strip()).strip()
        candidate_infos = candidates_from_sids(window_sids, sid_to_names, letter_samples) or list(letter_samples)
        candidate_order = [sample.name for sample in candidate_infos]
        tag_sentences = [
            (str(tag).strip(), str(sentence).strip())
            for tag, sentence in entry.get("tag_sentences", []) or []
            if str(tag).strip() and str(sentence).strip()
        ]

        for tag_order, (tag, sentence) in enumerate(tag_sentences):
            if tag not in SUPPORTED_TAGS:
                out_items_by_sample[assigned_sample].append(
                    make_property_item(
                        sample=assigned_sample,
                        para_id=para_id,
                        win_level=win_level,
                        window_sids=window_sids,
                        evidence_lines=evidence_lines,
                        tag=tag,
                        sentence=sentence,
                        tag_order=tag_order,
                        source="ORIG",
                        align_order=candidate_order,
                    )
                )
                continue

            step1 = call_step1_keep_or_refine(
                model,
                tag=tag,
                current_assigned_sample=assigned_sample,
                refined_sentence=sentence,
                evidence_text=evidence_text,
                candidate_samples=candidate_infos,
                log_path=log_paths["llm"],
                temperature=step1_temperature,
                max_tokens=step1_max_tokens,
                retries=retries,
            )
            if str(step1.get("action", "")).strip().upper() == "KEEP":
                out_items_by_sample[assigned_sample].append(
                    make_property_item(
                        sample=assigned_sample,
                        para_id=para_id,
                        win_level=win_level,
                        window_sids=window_sids,
                        evidence_lines=evidence_lines,
                        tag=tag,
                        sentence=sentence,
                        tag_order=tag_order,
                        source="ORIG",
                        align_order=candidate_order,
                    )
                )
                continue

            step2_candidates = [name for name in step1.get("candidates", []) or candidate_order if name in sample_map]
            if not step2_candidates:
                step2_candidates = [sample.name for sample in letter_samples if sample.name in sample_map]

            cache_key_base = hashlib.sha1(evidence_text.encode("utf-8", errors="ignore")).hexdigest()
            block_key = f"para={para_id};win={win_level};sids={','.join(str(value) for value in window_sids)}"
            for target_name in step2_candidates:
                target_sample = sample_map[target_name]
                cache_key = (tag, target_name, cache_key_base)
                if cache_key not in step2_cache:
                    outcome = call_step2_refine_one_sample(
                        model,
                        tag=tag,
                        target_sample=target_sample,
                        reference_sample_names=step2_candidates,
                        evidence_text=evidence_text,
                        candidate_samples=[target_sample],
                        log_path=log_paths["llm"],
                        reject_log_path=log_paths["reject"],
                        step1_temperature=step1_temperature,
                        step2_temperature=step2_temperature,
                        step1_max_tokens=step1_max_tokens,
                        step2_max_tokens=step2_max_tokens,
                        retries=retries,
                        max_refine_loops=step2_max_loops,
                        key_for_log=block_key,
                    )
                    step2_cache[cache_key] = outcome
                    if outcome.reason == "MAX_LOOP_REJECT":
                        step2_maxloop_reject_count += 1

                outcome = step2_cache[cache_key]
                if str(outcome.sentence).strip().upper() == "NONE":
                    continue

                out_items_by_sample[target_name].append(
                    make_property_item(
                        sample=target_name,
                        para_id=para_id,
                        win_level=win_level,
                        window_sids=window_sids,
                        evidence_lines=evidence_lines,
                        tag=tag,
                        sentence=outcome.sentence,
                        tag_order=tag_order,
                        source="REFINED",
                        align_order=step2_candidates,
                    )
                )

    out_items_by_sample, vote_stats = consolidate_items_by_sample_with_llm_vote(
        model,
        out_items_by_sample,
        llm_log_path=log_paths["llm"],
        vote_log_path=log_paths["vote"],
        vote_temperature=vote_temperature,
        vote_max_tokens=vote_max_tokens,
        retries=retries,
    )

    write_text(out_md, render_property_markdown(out_items_by_sample, property_label="Property abstract"))
    note_parts = [
        "multi-sample processed",
        f"samples_in_letter_table={len(letter_samples)}",
        f"entries={len(entries)}",
        f"wrote_samples={len(out_items_by_sample)}",
        f"step2_unique_runs={len(step2_cache)}",
        f"step2_maxloop_reject={step2_maxloop_reject_count}",
        f"votes={int(vote_stats.get('votes', 0) or 0)}",
        f"votes_used_4th={int(vote_stats.get('votes_used_4th', 0) or 0)}",
    ]
    if decision_note:
        note_parts.append(decision_note)

    result = Step7Result(
        paper_id=paper_id,
        paper_dir=paper_dir,
        status="PROCESSED_MULTI_STEP2_MAXLOOP_REJECT" if step2_maxloop_reject_count else "PROCESSED_MULTI",
        input_md=in_md,
        output_md=out_md,
        note="; ".join(note_parts),
    )
    write_io_log(result, log_paths["io"])
    return result


def write_root_logs(mining_root: str, results: Sequence[Step7Result]) -> None:
    main_log_path = os.path.join(mining_root, "step7_multisample_main.log")
    error_log_path = os.path.join(mining_root, "step7_multisample_error.log")

    with open(main_log_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# Step 7 multi-sample main resolution\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status not in MAIN_LOG_STATUSES:
                continue
            fh.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_md:
                fh.write(f"  input={relative_to_root(mining_root, result.input_md)}\n")
            if result.output_md:
                fh.write(f"  output={relative_to_root(mining_root, result.output_md)}\n")
            if result.note:
                fh.write(f"  note={result.note}\n")

    with open(error_log_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# Step 7 multi-sample errors\n\nGenerated at {timestamp_now()}\n\n")
        for result in results:
            if result.status in MAIN_LOG_STATUSES:
                continue
            fh.write(f"- paper={result.paper_id} status={result.status}\n")
            if result.input_md:
                fh.write(f"  input={relative_to_root(mining_root, result.input_md)}\n")
            if result.output_md:
                fh.write(f"  output={relative_to_root(mining_root, result.output_md)}\n")
            if result.note:
                fh.write(f"  note={result.note}\n")


def process_all_papers(
    mining_root: str,
    paper_ids: Optional[Sequence[str]] = None,
    model_name: str = DEFAULT_MODEL,
    step1_temperature: float = DEFAULT_STEP1_TEMPERATURE,
    step2_temperature: float = DEFAULT_STEP2_TEMPERATURE,
    vote_temperature: float = DEFAULT_VOTE_TEMPERATURE,
    step1_max_tokens: int = DEFAULT_STEP1_MAX_TOKENS,
    step2_max_tokens: int = DEFAULT_STEP2_MAX_TOKENS,
    vote_max_tokens: int = DEFAULT_VOTE_MAX_TOKENS,
    retries: int = DEFAULT_RETRIES,
    step2_max_loops: int = DEFAULT_STEP2_MAX_LOOPS,
    skip_existing: bool = True,
) -> None:
    if lmstudio_llm is None:
        raise RuntimeError("lmstudio is not installed. Install or configure LM Studio before running Step 7.")

    root = ensure_root_exists(mining_root)
    model = lmstudio_llm(model_name)
    results: List[Step7Result] = []
    for paper_dir in tqdm(iter_paper_dirs(root, paper_ids=paper_ids), desc="Step7: multisample-main"):
        try:
            results.append(
                process_one_paper(
                    paper_dir,
                    model=model,
                    skip_existing=skip_existing,
                    step1_temperature=step1_temperature,
                    step2_temperature=step2_temperature,
                    vote_temperature=vote_temperature,
                    step1_max_tokens=step1_max_tokens,
                    step2_max_tokens=step2_max_tokens,
                    vote_max_tokens=vote_max_tokens,
                    retries=retries,
                    step2_max_loops=step2_max_loops,
                )
            )
        except Exception as exc:
            paper_id = paper_id_from_dir(paper_dir) or ""
            results.append(
                Step7Result(
                    paper_id=paper_id,
                    paper_dir=paper_dir,
                    status="SKIP_FATAL",
                    input_md=os.path.join(paper_dir, "property", "abstract_clean", f"{paper_id}.md") if paper_id else "",
                    output_md=os.path.join(paper_dir, "property", "abstract_clean", f"{paper_id}_main.md") if paper_id else "",
                    note=repr(exc),
                )
            )

    write_root_logs(root, results)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Property preprocessing Step 7: resolve multi-sample main markdown.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Re-run even if Step 7 outputs already exist.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model name used for Step 7.")
    parser.add_argument("--step1-temperature", type=float, default=DEFAULT_STEP1_TEMPERATURE, help="LLM temperature for Step 7 keep/refine checks.")
    parser.add_argument("--step2-temperature", type=float, default=DEFAULT_STEP2_TEMPERATURE, help="LLM temperature for Step 7 sentence rewriting.")
    parser.add_argument("--vote-temperature", type=float, default=DEFAULT_VOTE_TEMPERATURE, help="LLM temperature for Step 7 duplicate voting.")
    parser.add_argument("--step1-max-tokens", type=int, default=DEFAULT_STEP1_MAX_TOKENS, help="Max tokens for Step 7 keep/refine calls.")
    parser.add_argument("--step2-max-tokens", type=int, default=DEFAULT_STEP2_MAX_TOKENS, help="Max tokens for Step 7 sentence rewriting calls.")
    parser.add_argument("--vote-max-tokens", type=int, default=DEFAULT_VOTE_MAX_TOKENS, help="Max tokens for Step 7 duplicate voting calls.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Max JSON-call retries per LLM request.")
    parser.add_argument("--step2-max-loops", type=int, default=DEFAULT_STEP2_MAX_LOOPS, help="Max rewrite loops for a single target sample.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_all_papers(
        mining_root=args.root,
        paper_ids=args.paper_ids,
        model_name=args.model,
        step1_temperature=args.step1_temperature,
        step2_temperature=args.step2_temperature,
        vote_temperature=args.vote_temperature,
        step1_max_tokens=args.step1_max_tokens,
        step2_max_tokens=args.step2_max_tokens,
        vote_max_tokens=args.vote_max_tokens,
        retries=args.retries,
        step2_max_loops=args.step2_max_loops,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
