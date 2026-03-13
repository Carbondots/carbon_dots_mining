#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step 5: refine sample-level property text from Step 4 decisions."""

import argparse
import ast
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from pipeline_utils import drop_no_digit_numeric_tags
from property_unit import (
    append_log_line as append_log,
    annotation_bundle_paths,
    dedupe_preserve_order,
    ensure_root_exists,
    iter_paper_dirs,
    move_paper_dir_to_sibling_root as move_paper_dir_to_no_property,
    paper_id_from_dir,
    parse_json_array_text as parse_llm_json_array,
    safe_read_csv,
    sort_key,
)

try:
    from lmstudio import llm as lmstudio_llm
except Exception:
    lmstudio_llm = None


DEFAULT_MODEL = "qwen.qwen2.5-vl-32b-instruct"
DEFAULT_TEMPERATURE = 0.35
DEFAULT_MAX_TOKENS = 2800
DEFAULT_RETRIES = 3
END_SENTINEL = "<END_OF_JSON>"
ALLOWED_SECTIONS = {"abstract", "methods", "results_discussion", "conclusion"}
PROPERTY_ORDER = ["Ex", "Em", "QY", "lifetime", "ExDep", "Chiral", "CPL"]
KIND_PRIORITY = {"app": 3, "vs": 2, "main": 1}
NUMERIC_TAGS = {"Ex", "Em", "QY", "lifetime"}
NUMERIC_PATTERNS = {
    "Ex": re.compile(r"(\d+(?:\.\d+)?)\s*nm\b", re.IGNORECASE),
    "Em": re.compile(r"(\d+(?:\.\d+)?)\s*nm\b", re.IGNORECASE),
    "QY": re.compile(r"(\d+(?:\.\d+)?)\s*%", re.IGNORECASE),
    "lifetime": re.compile(r"(\d+(?:\.\d+)?)\s*(?:fs|ps|ns|us|ms|s)\b", re.IGNORECASE),
}


def has_hits(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return text not in ("", "[]", "nan")


def normalize_tags(raw: Any) -> List[str]:
    if pd.isna(raw):
        return []
    text = str(raw).strip()
    if not text or text == "[]":
        return []

    parts: List[str]
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            parts = [str(item).strip() for item in parsed if str(item).strip()]
        else:
            parts = [str(parsed).strip()]
    except Exception:
        parts = [item.strip() for item in re.split(r"[;,\s]+", text) if item.strip()]

    return [tag for tag in PROPERTY_ORDER if tag in dedupe_preserve_order(parts)]


def parse_samples(raw: Any) -> List[str]:
    if pd.isna(raw):
        return []
    text = str(raw).strip()
    if not text:
        return []

    parts: List[str]
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            parts = [str(item).strip() for item in parsed if str(item).strip()]
        else:
            parts = [str(parsed).strip()]
    except Exception:
        parts = [item.strip() for item in re.split(r";+|\|\|", text) if item.strip()]

    return dedupe_preserve_order(parts)


def load_letter_table_name_set(paper_dir: str, paper_id: str, log_path: str) -> Optional[set]:
    letter_csv = os.path.join(paper_dir, "Synthesis", "letter_table", f"{paper_id}.csv")
    df = safe_read_csv(letter_csv)
    if df is None:
        append_log(log_path, f"[WARN] {paper_id}: letter_table csv not found, name cross-check skipped.")
        return None
    if "CDs_Naming_in_Paper" not in df.columns:
        append_log(log_path, f"[WARN] {paper_id}: column CDs_Naming_in_Paper not found, name cross-check skipped.")
        return None

    names = {
        str(value).strip()
        for value in df["CDs_Naming_in_Paper"].fillna("").astype(str).tolist()
        if str(value).strip()
    }
    return names


def build_window_evidence_text(df: pd.DataFrame, seed_sent_id: int, win_level: str) -> str:
    level = (win_level or "SMALL").strip().upper()
    if level not in {"SMALL", "EXTENDED"}:
        level = "SMALL"

    ordered = df.sort_values("sent_global_id").reset_index(drop=True)
    seed_rows = ordered[ordered["sent_global_id"] == seed_sent_id]
    if seed_rows.empty:
        raise ValueError(f"seed sentence {seed_sent_id} is missing from the decision dataframe")
    seed_para = int(seed_rows.iloc[0]["para_global_id"])

    def make_evidence_text(sent_ids: Sequence[int]) -> str:
        sent_id_list = sorted({int(value) for value in sent_ids})
        subset = ordered[ordered["sent_global_id"].isin(sent_id_list)].copy()
        subset = subset.sort_values("sent_global_id")
        whitelist = [
            {"sent_id": str(int(row["sent_global_id"])), "text": str(row["text"])}
            for _, row in subset.iterrows()
        ]
        window_text = " ".join(str(row["text"]) for _, row in subset.iterrows())
        return "WHITELIST:\n" + json.dumps(whitelist, ensure_ascii=False) + "\nWINDOW_TEXT:\n" + window_text

    if level == "EXTENDED":
        this_para = ordered[ordered["para_global_id"] == seed_para].sort_values("sent_global_id")
        this_sids = this_para["sent_global_id"].astype(int).tolist()
        if not this_sids:
            return make_evidence_text([seed_sent_id])

        if len(this_sids) > 7:
            return make_evidence_text(this_sids)

        para_ids = sorted(ordered["para_global_id"].astype(int).unique().tolist())
        para_pos = para_ids.index(seed_para)
        sent_ids: List[int] = []

        if para_pos > 0:
            prev_para = ordered[ordered["para_global_id"] == para_ids[para_pos - 1]].sort_values("sent_global_id")
            sent_ids.extend(prev_para["sent_global_id"].astype(int).tolist()[-5:])

        sent_ids.extend(this_sids)

        if para_pos + 1 < len(para_ids):
            next_para = ordered[ordered["para_global_id"] == para_ids[para_pos + 1]].sort_values("sent_global_id")
            sent_ids.extend(next_para["sent_global_id"].astype(int).tolist()[:5])

        return make_evidence_text(sent_ids)

    candidate_rows = ordered[
        ordered["main_section_norm"].isin(ALLOWED_SECTIONS)
        & ordered["prop_window_hits"].map(has_hits)
    ].copy()
    para_rows = candidate_rows[candidate_rows["para_global_id"] == seed_para].sort_values("sent_global_id")

    groups: List[List[int]] = []
    current_group: List[int] = []
    prev_sid: Optional[int] = None
    for _, row in para_rows.iterrows():
        sid = int(row["sent_global_id"])
        if not current_group:
            current_group = [sid]
        elif prev_sid is not None and sid - prev_sid <= 2:
            current_group.append(sid)
        else:
            groups.append(current_group)
            current_group = [sid]
        prev_sid = sid
    if current_group:
        groups.append(current_group)

    group_sids = next((group for group in groups if seed_sent_id in group), [seed_sent_id])
    group_sids = sorted(set(int(value) for value in group_sids))

    if len(group_sids) >= 2:
        para_sent_ids = (
            ordered[ordered["para_global_id"] == seed_para]
            .sort_values("sent_global_id")["sent_global_id"]
            .astype(int)
            .tolist()
        )
        if not para_sent_ids:
            return make_evidence_text([seed_sent_id])

        start_idx = para_sent_ids.index(min(group_sids))
        end_idx = para_sent_ids.index(max(group_sids))
        while (end_idx - start_idx + 1) < 5 and (start_idx > 0 or end_idx < len(para_sent_ids) - 1):
            if start_idx > 0:
                start_idx -= 1
            if (end_idx - start_idx + 1) >= 5:
                break
            if end_idx < len(para_sent_ids) - 1:
                end_idx += 1
        return make_evidence_text(para_sent_ids[start_idx : end_idx + 1])

    global_sent_ids = ordered["sent_global_id"].astype(int).tolist()
    if seed_sent_id not in global_sent_ids:
        return make_evidence_text([seed_sent_id])
    idx = global_sent_ids.index(seed_sent_id)
    return make_evidence_text(global_sent_ids[max(0, idx - 1) : min(len(global_sent_ids), idx + 2)])


def extract_whitelist_ids(evidence_text: str) -> List[int]:
    if "WHITELIST:\n" not in evidence_text or "\nWINDOW_TEXT:\n" not in evidence_text:
        return []
    try:
        _, tail = evidence_text.split("WHITELIST:\n", 1)
        json_part, _ = tail.split("\nWINDOW_TEXT:\n", 1)
        payload = json.loads(json_part)
    except Exception:
        return []

    sent_ids: List[int] = []
    for item in payload:
        try:
            sent_ids.append(int(item.get("sent_id")))
        except Exception:
            continue
    return sorted(set(sent_ids))


def collect_window_tags(df: pd.DataFrame, win_sids: Sequence[int]) -> List[str]:
    tags = set()
    for sent_id in win_sids:
        subset = df[df["sent_global_id"] == int(sent_id)]
        for _, row in subset.iterrows():
            for tag in normalize_tags(row.get("LLM_props", "")):
                tags.add(tag)
    return [tag for tag in PROPERTY_ORDER if tag in tags]


def property_spec(tag: str, sample_name: str) -> str:
    specs = {
        "Ex": (
            f"Only refine excitation wavelength (Ex) when the evidence explicitly assigns a numeric wavelength with a unit "
            f"to {sample_name}. Ignore scan ranges and omit the tag if the wavelength is not safely attributable."
        ),
        "Em": (
            f"Only refine emission wavelength (Em) when the evidence explicitly assigns a numeric emission value with a unit "
            f"to {sample_name}. Do not turn series boundaries or generic trend endpoints into a sample-specific value."
        ),
        "QY": (
            f"Only refine photoluminescence quantum yield (QY) when the evidence gives a clear numeric percentage for "
            f"{sample_name}. Do not mix reference dyes or other samples."
        ),
        "lifetime": (
            f"Only refine lifetime when the evidence gives an explicit lifetime value with a time unit for {sample_name}. "
            "Keep the reported units and do not average separate components."
        ),
        "ExDep": (
            f"Only refine excitation dependence (ExDep) when the evidence explicitly states that the emission of "
            f"{sample_name} changes with excitation or stays unchanged with excitation."
        ),
        "Chiral": (
            f"Only refine Chiral when the evidence explicitly describes {sample_name} itself as chiral or achiral."
        ),
        "CPL": (
            f"Only refine CPL when the evidence explicitly states that {sample_name} shows circularly polarized luminescence."
        ),
    }
    if tag not in specs:
        raise ValueError(f"Unknown property tag: {tag}")
    return specs[tag]


def build_refine_prompt(sample_name: str, tag: str, evidence_text: str) -> str:
    return f"""You are an expert in refining photoluminescence properties for carbon-dot materials.

Return exactly one JSON array followed immediately by {END_SENTINEL}. Do not output prose or markdown.

Task:
- Refine only the property tag "{tag}" for the target sample "{sample_name}".
- Use only evidence from the WHITELIST and WINDOW_TEXT blocks below.
- If you cannot refine a valid sentence for this tag, return exactly [{{"kind":"false"}}].

Kind rules:
- "main": intrinsic PL property of the exact target sample under normal characterization conditions.
- "vs": comparative or series-level behavior that cannot be safely written as one intrinsic property sentence for the target sample.
- "app": PL behavior in an application setting such as sensing, detection, imaging, anti-counterfeiting, or encryption.
- "false": no usable refinement for this tag.

Hard rules:
1. Use only wording, numbers, and units supported by the evidence.
2. For numeric tags (Ex, Em, QY, lifetime), output a sentence only when it includes an explicit numeric value and unit from the evidence.
3. The sentence must be concise, self-contained, and in English.
4. For "main", the sentence must clearly describe "{sample_name}" itself.
5. For "vs" and "app", "sample" may be "{sample_name}" or an empty string.
6. For every non-false object, "Property" must contain only the "{tag}" key.
7. "evidence" must list only sentence ids from WHITELIST.

Property rule:
{property_spec(tag, sample_name)}

Output schema:
[
  {{
    "kind": "<main|vs|app|false>",
    "sample": "<{sample_name} or empty string>",
    "evidence": [<int>, ...],
    "Property": {{
      "{tag}": "<one refined sentence>"
    }}
  }}
]
{END_SENTINEL}

{evidence_text}"""




def extract_property_sentence(prop: Dict[str, Any], requested_tag: str) -> str:
    if requested_tag in prop:
        return str(prop.get(requested_tag, "")).strip()
    for key, value in prop.items():
        if str(key).strip().lower() == requested_tag.lower():
            return str(value).strip()
    return ""


def validate_numeric_sentence(sentence: str, tag: str, evidence_lines: Sequence[str]) -> bool:
    if tag not in NUMERIC_TAGS:
        return True
    pattern = NUMERIC_PATTERNS[tag]
    matches = pattern.findall(sentence or "")
    if not matches:
        return False
    evidence_join = " ".join(str(line) for line in evidence_lines)
    return all(str(match).strip() in evidence_join for match in matches if str(match).strip())


def resolve_kind_conflicts(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(entries) < 2:
        return entries

    drop_indices = set()
    for left_idx, left in enumerate(entries):
        if left_idx in drop_indices:
            continue
        left_kind = left.get("kind")
        left_set = set(left.get("evidence", []))
        if left_kind not in KIND_PRIORITY or not left_set:
            continue
        for right_idx in range(left_idx + 1, len(entries)):
            if right_idx in drop_indices:
                continue
            right = entries[right_idx]
            right_kind = right.get("kind")
            right_set = set(right.get("evidence", []))
            if right_kind not in KIND_PRIORITY or not right_set or left_kind == right_kind:
                continue
            if left_set.issubset(right_set) or right_set.issubset(left_set):
                if KIND_PRIORITY[left_kind] >= KIND_PRIORITY[right_kind]:
                    drop_indices.add(right_idx)
                else:
                    drop_indices.add(left_idx)
                    break

    return [entry for idx, entry in enumerate(entries) if idx not in drop_indices]


def normalize_llm_objects(
    objects: Optional[List[Dict[str, Any]]],
    sample_name: str,
    requested_tag: str,
    whitelist_ids: Sequence[int],
    id_to_text: Dict[int, str],
) -> Optional[List[Dict[str, Any]]]:
    if objects is None:
        return None

    whitelist_list = [int(value) for value in whitelist_ids]
    whitelist_set = set(whitelist_list)
    normalized: List[Dict[str, Any]] = []

    for obj in objects:
        kind = str(obj.get("kind", "")).strip().lower()
        if kind not in {"main", "vs", "app", "false"}:
            continue
        if kind == "false":
            normalized.append({"kind": "false"})
            continue

        raw_sample = str(obj.get("sample", "") or "").strip()
        if kind == "main" and raw_sample not in ("", sample_name):
            continue
        if kind in {"vs", "app"} and raw_sample not in ("", sample_name):
            continue
        sample_label = sample_name if kind == "main" and not raw_sample else raw_sample

        prop = obj.get("Property")
        if not isinstance(prop, dict):
            continue
        sentence = extract_property_sentence(prop, requested_tag)
        if not sentence:
            continue

        evidence_raw = obj.get("evidence") or []
        evidence_ids: List[int] = []
        if isinstance(evidence_raw, (list, tuple)):
            for value in evidence_raw:
                try:
                    sent_id = int(value)
                except Exception:
                    continue
                if sent_id in whitelist_set:
                    evidence_ids.append(sent_id)
        if not evidence_ids:
            evidence_ids = whitelist_list[:]
        evidence_ids = [sent_id for sent_id in whitelist_list if sent_id in set(evidence_ids)] or whitelist_list[:]
        evidence_lines = [id_to_text.get(sent_id, "") for sent_id in evidence_ids]
        if not any(line.strip() for line in evidence_lines):
            continue
        if not validate_numeric_sentence(sentence, requested_tag, evidence_lines):
            continue

        normalized.append(
            {
                "kind": kind,
                "sample": sample_label,
                "evidence": evidence_ids,
                "Property": {requested_tag: sentence},
            }
        )

    useful = [item for item in normalized if item.get("kind") in {"main", "vs", "app"}]
    if useful:
        normalized = useful
    else:
        normalized = [item for item in normalized if item.get("kind") == "false"]

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in resolve_kind_conflicts(normalized):
        key = (
            item.get("kind"),
            item.get("sample", ""),
            tuple(item.get("evidence", [])),
            json.dumps(item.get("Property", {}), sort_keys=True, ensure_ascii=False),
        )
        if key in seen:
            continue
        deduped.append(item)
        seen.add(key)

    return deduped


def refine_tag_with_retries(
    model,
    sample_name: str,
    tag: str,
    evidence_text: str,
    whitelist_ids: Sequence[int],
    id_to_text: Dict[int, str],
    log_path: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> List[Dict[str, Any]]:
    prompt = build_refine_prompt(sample_name=sample_name, tag=tag, evidence_text=evidence_text)
    append_log(log_path, "")
    append_log(log_path, f"=== INPUT sample={sample_name} tag={tag} ===")
    append_log(log_path, evidence_text)

    for attempt in range(1, max(1, retries) + 1):
        try:
            response = model.respond(prompt, config={"temperature": temperature, "maxTokens": max_tokens})
        except Exception as exc:
            append_log(log_path, f"[ERROR] sample={sample_name} tag={tag} attempt={attempt}: {exc}")
            if attempt == retries:
                return []
            continue

        raw = "" if response is None else str(getattr(response, "content", "") or "")
        append_log(log_path, f"=== RAW sample={sample_name} tag={tag} attempt={attempt} ===")
        append_log(log_path, raw if raw else "(empty)")

        parsed_objects = parse_llm_json_array(raw, end_sentinel=END_SENTINEL)
        if parsed_objects is not None:
            parsed_objects = [item for item in parsed_objects if isinstance(item, dict)]
        normalized = normalize_llm_objects(
            parsed_objects,
            sample_name=sample_name,
            requested_tag=tag,
            whitelist_ids=whitelist_ids,
            id_to_text=id_to_text,
        )
        if normalized is None:
            append_log(log_path, f"[WARN] sample={sample_name} tag={tag} attempt={attempt}: invalid JSON format.")
            if attempt == retries:
                return []
            continue
        useful = [item for item in normalized if item.get("kind") in {"main", "vs", "app"}]
        if useful:
            return useful
        if attempt == retries:
            return []
    return []


def create_entry(
    paper_id: str,
    sample_name: str,
    kind: str,
    para_id: int,
    win_level: str,
    window_sids: Sequence[int],
    evidence_sids: Sequence[int],
    evidence_lines: Sequence[str],
    seed_sid: int,
    tag: str,
    sentence: str,
) -> Dict[str, Any]:
    return {
        "paper_id": paper_id,
        "sample": sample_name,
        "kind": kind,
        "para_id": int(para_id),
        "win_level": str(win_level),
        "window_sids": [int(value) for value in window_sids],
        "evidence_sids": [int(value) for value in evidence_sids],
        "evidence_lines": [str(value) for value in evidence_lines if str(value).strip()],
        "seed_sid": int(seed_sid),
        "tags": [tag],
        "tag_sentences": [(tag, sentence)],
        "refined_text": sentence,
    }


def merge_entry(entry: Dict[str, Any], tag: str, sentence: str) -> None:
    if tag in entry["tags"]:
        return
    entry["tags"].append(tag)
    entry["tag_sentences"].append((tag, sentence))
    order_map = {value: idx for idx, value in enumerate(PROPERTY_ORDER)}
    entry["tag_sentences"] = sorted(entry["tag_sentences"], key=lambda item: order_map.get(item[0], 999))
    entry["tags"] = [item[0] for item in entry["tag_sentences"]]
    entry["refined_text"] = " ".join(text for _, text in entry["tag_sentences"])


def apply_majority_kind_filter(
    main_entries: Dict[str, List[Dict[str, Any]]],
    vs_entries: List[Dict[str, Any]],
    app_entries: List[Dict[str, Any]],
    root_log_path: str,
    paper_id: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid_kinds = {"main", "vs", "app"}
    all_entries: List[Dict[str, Any]] = []
    for entries in main_entries.values():
        all_entries.extend(entries)
    all_entries.extend(vs_entries)
    all_entries.extend(app_entries)

    groups: Dict[frozenset, List[Dict[str, Any]]] = defaultdict(list)
    for entry in all_entries:
        kind = entry.get("kind")
        if kind not in valid_kinds:
            continue
        evidence_key = frozenset(int(value) for value in entry.get("evidence_sids", []) if str(value).strip())
        if evidence_key:
            groups[evidence_key].append(entry)

    to_drop = set()
    for evidence_set, entries in groups.items():
        kind_counter = Counter(entry.get("kind") for entry in entries if entry.get("kind") in valid_kinds)
        if len(kind_counter) <= 1:
            continue
        max_count = max(kind_counter.values())
        winners = [kind for kind, count in kind_counter.items() if count == max_count]
        if len(winners) != 1:
            append_log(root_log_path, f"[TIE_KEEP_ALL] {paper_id}: evidence={sorted(evidence_set)} counts={dict(kind_counter)}")
            continue
        winner = winners[0]
        append_log(root_log_path, f"[MAJORITY_KIND] {paper_id}: evidence={sorted(evidence_set)} keep={winner} counts={dict(kind_counter)}")
        for entry in entries:
            if entry.get("kind") != winner:
                to_drop.add(id(entry))

    if to_drop:
        for sample_name in list(main_entries.keys()):
            main_entries[sample_name] = [entry for entry in main_entries[sample_name] if id(entry) not in to_drop]
        vs_entries = [entry for entry in vs_entries if id(entry) not in to_drop]
        app_entries = [entry for entry in app_entries if id(entry) not in to_drop]
    return main_entries, vs_entries, app_entries


def write_grouped_md(md_path: str, sample_map: Dict[str, List[Dict[str, Any]]], kind_label: str) -> None:
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Step 5 Refined Properties ({kind_label})\n\n")
        available = {key: value for key, value in sample_map.items() if value}
        if not available:
            fh.write("(no entries)\n")
            return

        for sample_name in sorted(available.keys(), key=lambda value: (value == "", str(value).lower())):
            header = sample_name if sample_name else "(series-level)"
            fh.write(f"## {header}\n\n")
            entries = sorted(available[sample_name], key=lambda item: (item["para_id"], item["seed_sid"]))
            for index, entry in enumerate(entries, start=1):
                sids_text = ",".join(str(value) for value in entry["window_sids"])
                fh.write(f"{index}. [para={entry['para_id']}; window={entry['win_level']}; sids={sids_text}]\n")
                fh.write("Evidence:\n")
                fh.write(" ".join(entry["evidence_lines"]).strip() + "\n")
                fh.write("Refined properties:\n")
                for tag, sentence in entry["tag_sentences"]:
                    fh.write(f"{tag}: {sentence}\n")
                fh.write("\n")


def refined_outputs_exist(out_dir: str, paper_id: str) -> bool:
    required = [
        os.path.join(out_dir, f"{paper_id}.md"),
        os.path.join(out_dir, f"{paper_id}_vs.md"),
        os.path.join(out_dir, f"{paper_id}_app.md"),
    ]
    return all(os.path.exists(path) for path in required)


def process_one_paper(
    paper_dir: str,
    model,
    temperature: float,
    max_tokens: int,
    retries: int,
    skip_existing: bool,
    root_log_path: str,
) -> bool:
    paper_id = paper_id_from_dir(paper_dir)
    if paper_id is None:
        return False

    in_paths = annotation_bundle_paths(paper_dir, "property/decision_LLM", paper_id)
    out_dir = os.path.join(paper_dir, "property", "refined_properties")
    main_md_path = os.path.join(out_dir, f"{paper_id}.md")
    vs_md_path = os.path.join(out_dir, f"{paper_id}_vs.md")
    app_md_path = os.path.join(out_dir, f"{paper_id}_app.md")
    paper_log_path = os.path.join(out_dir, f"{paper_id}_step5.log")

    if not os.path.exists(in_paths["csv"]):
        append_log(root_log_path, f"[SKIP] {paper_id}: input decision_LLM csv not found.")
        return False

    if skip_existing and refined_outputs_exist(out_dir, paper_id):
        print(f"[SKIP] {paper_id}: Step 5 outputs already exist.")
        return False

    os.makedirs(out_dir, exist_ok=True)
    with open(paper_log_path, "w", encoding="utf-8") as fh:
        fh.write("")

    df = safe_read_csv(in_paths["csv"])
    if df is None:
        append_log(root_log_path, f"[SKIP] {paper_id}: failed to read Step 4 csv.")
        return False

    required_cols = {
        "sent_global_id",
        "para_global_id",
        "text",
        "main_section_norm",
        "prop_window_hits",
        "window_level",
        "LLM_props",
        "LLM_name",
        "merge_LLM_name",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"{paper_id}: missing required columns in Step 4 csv: {missing}")

    df = df.copy()
    df["sent_global_id"] = df["sent_global_id"].astype(int)
    df["para_global_id"] = df["para_global_id"].astype(int)
    for column in ("window_level", "LLM_props", "LLM_name", "merge_LLM_name", "prop_window_hits"):
        df[column] = df[column].fillna("").astype(str)

    id_to_text = {int(row["sent_global_id"]): str(row["text"]) for _, row in df.iterrows()}
    letter_name_set = load_letter_table_name_set(paper_dir, paper_id, root_log_path)

    seeds = df[
        df["main_section_norm"].isin(ALLOWED_SECTIONS)
        & df["prop_window_hits"].map(has_hits)
        & df["LLM_props"].map(lambda value: bool(normalize_tags(value)))
    ].copy()
    seeds = seeds.sort_values("sent_global_id")

    if seeds.empty:
        append_log(root_log_path, f"[SKIP_NO_SEEDS] {paper_id}: no Step 5 candidate windows after filters.")
        move_paper_dir_to_no_property(
            paper_dir,
            log_path=root_log_path,
            reason="no_candidate_windows_after_filters",
        )
        return True

    sample_entries: Dict[str, List[Dict[str, Any]]] = {}
    vs_entries: List[Dict[str, Any]] = []
    app_entries: List[Dict[str, Any]] = []
    main_index: Dict[Tuple[str, int, str, Tuple[int, ...]], Dict[str, Any]] = {}
    vs_index: Dict[Tuple[str, int, str, Tuple[int, ...]], Dict[str, Any]] = {}
    app_index: Dict[Tuple[str, int, str, Tuple[int, ...]], Dict[str, Any]] = {}
    processed_keys = set()

    for _, row in tqdm(seeds.iterrows(), total=len(seeds), desc=f"Step5: refine {paper_id}", leave=False):
        seed_sid = int(row["sent_global_id"])
        para_id = int(row["para_global_id"])
        win_level = str(row.get("window_level", "SMALL") or "SMALL").strip().upper() or "SMALL"

        raw_samples = row.get("merge_LLM_name", "") or row.get("LLM_name", "")
        samples = parse_samples(raw_samples)
        if not samples:
            append_log(root_log_path, f"[WARN] {paper_id}: no samples parsed for seed_sid={seed_sid}.")
            continue

        evidence_text = build_window_evidence_text(df, seed_sid, win_level)
        win_sids = extract_whitelist_ids(evidence_text) or [seed_sid]
        tags_all = collect_window_tags(df, win_sids)
        tags_all, removed_numeric_tags = drop_no_digit_numeric_tags(tags_all, evidence_text=evidence_text)

        if removed_numeric_tags:
            append_log(
                root_log_path,
                f"[INFO] {paper_id}: seed_sid={seed_sid} removed numeric tags without digits: {removed_numeric_tags}",
            )
        if not tags_all:
            continue

        for sample_name in samples:
            if letter_name_set is not None and sample_name not in letter_name_set:
                append_log(root_log_path, f"[WARN] {paper_id}: sample '{sample_name}' not found in letter_table.")

            for tag in tags_all:
                dedup_key = (win_level, tuple(win_sids), sample_name, tag)
                if dedup_key in processed_keys:
                    continue
                processed_keys.add(dedup_key)

                objects = refine_tag_with_retries(
                    model=model,
                    sample_name=sample_name,
                    tag=tag,
                    evidence_text=evidence_text,
                    whitelist_ids=win_sids,
                    id_to_text=id_to_text,
                    log_path=paper_log_path,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retries=retries,
                )
                if not objects:
                    continue

                for obj in objects:
                    kind = str(obj["kind"]).strip()
                    raw_output_sample = str(obj.get("sample", "") or "").strip()
                    sample_label = sample_name if kind == "main" and not raw_output_sample else raw_output_sample
                    used_ids = [int(value) for value in obj.get("evidence", [])] or win_sids
                    evidence_lines = [id_to_text.get(sent_id, "") for sent_id in used_ids]
                    sentence = str(obj["Property"].get(tag, "")).strip()
                    if not sentence:
                        continue

                    evidence_key = (sample_label, para_id, win_level, tuple(used_ids))
                    if kind == "main":
                        entry = main_index.get(evidence_key)
                        if entry is None:
                            entry = create_entry(
                                paper_id=paper_id,
                                sample_name=sample_label,
                                kind=kind,
                                para_id=para_id,
                                win_level=win_level,
                                window_sids=win_sids,
                                evidence_sids=used_ids,
                                evidence_lines=evidence_lines,
                                seed_sid=seed_sid,
                                tag=tag,
                                sentence=sentence,
                            )
                            main_index[evidence_key] = entry
                            sample_entries.setdefault(sample_label, []).append(entry)
                        else:
                            merge_entry(entry, tag, sentence)
                    elif kind == "vs":
                        entry = vs_index.get(evidence_key)
                        if entry is None:
                            entry = create_entry(
                                paper_id=paper_id,
                                sample_name=sample_label,
                                kind=kind,
                                para_id=para_id,
                                win_level=win_level,
                                window_sids=win_sids,
                                evidence_sids=used_ids,
                                evidence_lines=evidence_lines,
                                seed_sid=seed_sid,
                                tag=tag,
                                sentence=sentence,
                            )
                            vs_index[evidence_key] = entry
                            vs_entries.append(entry)
                        else:
                            merge_entry(entry, tag, sentence)
                    elif kind == "app":
                        entry = app_index.get(evidence_key)
                        if entry is None:
                            entry = create_entry(
                                paper_id=paper_id,
                                sample_name=sample_label,
                                kind=kind,
                                para_id=para_id,
                                win_level=win_level,
                                window_sids=win_sids,
                                evidence_sids=used_ids,
                                evidence_lines=evidence_lines,
                                seed_sid=seed_sid,
                                tag=tag,
                                sentence=sentence,
                            )
                            app_index[evidence_key] = entry
                            app_entries.append(entry)
                        else:
                            merge_entry(entry, tag, sentence)

    sample_entries, vs_entries, app_entries = apply_majority_kind_filter(
        main_entries=sample_entries,
        vs_entries=vs_entries,
        app_entries=app_entries,
        root_log_path=root_log_path,
        paper_id=paper_id,
    )

    vs_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in vs_entries:
        vs_by_sample[entry.get("sample", "")].append(entry)

    app_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in app_entries:
        app_by_sample[entry.get("sample", "")].append(entry)

    if not (sample_entries or vs_entries or app_entries):
        append_log(root_log_path, f"[NO_MD_OUTPUT_MOVE] {paper_id}: no main/vs/app outputs after refine.")
        move_paper_dir_to_no_property(
            paper_dir,
            log_path=root_log_path,
            reason="no_main_vs_app_outputs_after_refine",
        )
        return True

    write_grouped_md(main_md_path, sample_entries, "main")
    write_grouped_md(vs_md_path, dict(vs_by_sample), "vs")
    write_grouped_md(app_md_path, dict(app_by_sample), "app")
    append_log(root_log_path, f"[OK] {paper_id}: Step 5 outputs written.")
    print(f"[OK] {paper_id}: Step 5 refined outputs saved.")
    return True


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
        raise RuntimeError("lmstudio is not installed. Install or configure LM Studio before running Step 5.")

    root = ensure_root_exists(mining_root)
    model = lmstudio_llm(model_name)
    paper_dirs = iter_paper_dirs(root, paper_ids=paper_ids)
    root_log_path = os.path.join(root, "step5_llm_refine.log")
    with open(root_log_path, "a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 80 + "\n")
        fh.write(
            f"NEW RUN model={model_name} temperature={temperature} "
            f"max_tokens={max_tokens} retries={retries}\n"
        )
        fh.write("=" * 80 + "\n")

    for paper_dir in tqdm(paper_dirs, desc="Step5: llm-refine"):
        process_one_paper(
            paper_dir=paper_dir,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            skip_existing=skip_existing,
            root_log_path=root_log_path,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Property preprocessing step5: refine Step 4 property windows.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Re-run even if Step 5 outputs already exist.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model name used for Step 5.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="LLM max tokens per call.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retry count for invalid Step 5 outputs.")
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
