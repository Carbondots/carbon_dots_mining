#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step 4: decide sample names and property tags with an LLM."""

import argparse
import json
import os
import re
from textwrap import dedent
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from property_unit import (
    annotation_bundle_paths,
    ensure_dir,
    ensure_root_exists,
    iter_paper_dirs,
    loads_json_field,
    parse_json_object_text as parse_llm_json,
    paper_id_from_dir,
    relative_to_paper,
    safe_read_csv,
    safe_paper_title,
    sort_key,
    strip_llm_wrappers,
)

try:
    from lmstudio import llm as lmstudio_llm
except Exception:
    lmstudio_llm = None


DEFAULT_MODEL = "qwen.qwen2.5-vl-32b-instruct"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1200
DEFAULT_RETRIES = 3
DEFAULT_RAW_GAP_LIMIT = 2
END_SENTINEL = "<END_OF_JSON>"
ALLOWED_SECTIONS = {"abstract", "methods", "results_discussion", "conclusion"}
PROPERTY_ORDER = ["Ex", "Em", "QY", "lifetime", "ExDep", "Chiral", "CPL"]


def normalize_line(text: Any) -> str:
    return str(text or "").replace("\n", " ").strip()


def decision_outputs_exist(out_dir: str, paper_id: str) -> bool:
    return all(
        os.path.exists(os.path.join(out_dir, f"{paper_id}.{ext}"))
        for ext in ("csv", "md", "txt")
    )


def load_candidate_names(paper_dir: str, letter_csv: str, log_path: str) -> List[str]:
    rel_letter_csv = relative_to_paper(paper_dir, letter_csv)
    df = safe_read_csv(letter_csv)
    if df is None or "CDs_Naming_in_Paper" not in df.columns:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write("\n=== SAMPLE CANDIDATES ===\n")
            fh.write(f"letter_table_csv: {rel_letter_csv}\n")
            fh.write("warning: file missing or column 'CDs_Naming_in_Paper' not found.\n")
        return []

    series = df["CDs_Naming_in_Paper"].fillna("").astype(str).map(str.strip)
    candidates = [value for value in series.tolist() if value]
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("\n=== SAMPLE CANDIDATES ===\n")
        fh.write(f"letter_table_csv: {rel_letter_csv}\n")
        fh.write(f"candidate_count: {len(candidates)}\n")
        fh.write(json.dumps(candidates, ensure_ascii=False) + "\n")
    return candidates


def build_para_text_map(df: pd.DataFrame) -> Tuple[Dict[str, List[Tuple[str, str]]], List[str]]:
    required = {"para_global_id", "sent_global_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing paragraph mapping columns: {sorted(missing)}")

    para_map: Dict[str, List[Tuple[str, str]]] = {}
    for _, row in df.iterrows():
        para_id = str(row["para_global_id"])
        sent_id = str(row["sent_global_id"])
        text = "" if pd.isna(row["text"]) else str(row["text"])
        para_map.setdefault(para_id, []).append((sent_id, text))

    for para_id in para_map:
        para_map[para_id] = sorted(para_map[para_id], key=lambda item: sort_key(item[0]))

    return para_map, sorted(para_map.keys(), key=sort_key)


def get_para_sent_list(para_map: Dict[str, List[Tuple[str, str]]], para_id: str) -> List[Tuple[str, str]]:
    return [(str(sent_id), str(text)) for sent_id, text in para_map.get(para_id, [])]


def build_window_for_group(
    para_map: Dict[str, List[Tuple[str, str]]],
    para_order: Sequence[str],
    para_id: str,
    member_sent_ids: Sequence[str],
    mode: str = "SMALL",
    min_len_small: int = 3,
    para_len_threshold: int = 7,
    flank_limit: int = 5,
) -> Tuple[str, Dict[str, str], str]:
    para_sents = get_para_sent_list(para_map, para_id)
    whitelist_ids = {str(value) for value in member_sent_ids}
    whitelist = {sent_id: text for sent_id, text in para_sents if sent_id in whitelist_ids}
    seed_id = str(member_sent_ids[len(member_sent_ids) // 2])
    seed_text = whitelist.get(seed_id, "")

    if not para_sents:
        return seed_text, whitelist, " ".join(whitelist.get(sent_id, "") for sent_id in member_sent_ids).strip()

    para_index = {value: idx for idx, value in enumerate(para_order)}
    id_to_idx = {sent_id: idx for idx, (sent_id, _) in enumerate(para_sents)}

    if mode.upper() == "SMALL":
        if len(member_sent_ids) >= 2:
            idxs = [id_to_idx[sent_id] for sent_id in member_sent_ids if sent_id in id_to_idx]
            if not idxs:
                return seed_text, whitelist, " ".join(whitelist.values()).strip()
            lo, hi = min(idxs), max(idxs)
            while (hi - lo + 1) < min_len_small:
                if lo > 0:
                    lo -= 1
                elif hi < len(para_sents) - 1:
                    hi += 1
                else:
                    break
            window_text = " ".join(text.strip() for _, text in para_sents[lo : hi + 1] if text and text.strip())
            return seed_text, whitelist, window_text

        chunks: List[str] = []

        def push(pair: Tuple[str, str]) -> None:
            _, text = pair
            if text and text.strip():
                chunks.append(text.strip())

        if seed_id in id_to_idx:
            idx = id_to_idx[seed_id]
            push(para_sents[idx])
            if idx - 1 >= 0:
                push(para_sents[idx - 1])
            elif para_id in para_index and para_index[para_id] - 1 >= 0:
                prev_list = get_para_sent_list(para_map, para_order[para_index[para_id] - 1])
                if prev_list:
                    push(prev_list[-1])
            if idx + 1 < len(para_sents):
                push(para_sents[idx + 1])
            elif para_id in para_index and para_index[para_id] + 1 < len(para_order):
                next_list = get_para_sent_list(para_map, para_order[para_index[para_id] + 1])
                if next_list:
                    push(next_list[0])
        elif seed_text:
            chunks.append(seed_text)

        return seed_text, whitelist, " ".join(chunks)

    this_text = " ".join(text.strip() for _, text in para_sents if text and text.strip())
    if para_id not in para_index or len(para_sents) > para_len_threshold:
        return seed_text, whitelist, this_text

    parts: List[str] = []
    para_pos = para_index[para_id]
    if para_pos - 1 >= 0:
        prev_list = get_para_sent_list(para_map, para_order[para_pos - 1])
        prev_slice = prev_list[-flank_limit:] if len(prev_list) > flank_limit else prev_list
        prev_text = " ".join(text.strip() for _, text in prev_slice if text and text.strip())
        if prev_text:
            parts.append("--- PREV PARA ---\n" + prev_text)
    if this_text:
        parts.append("--- THIS PARA ---\n" + this_text)
    if para_pos + 1 < len(para_order):
        next_list = get_para_sent_list(para_map, para_order[para_pos + 1])
        next_slice = next_list[:flank_limit] if len(next_list) > flank_limit else next_list
        next_text = " ".join(text.strip() for _, text in next_slice if text and text.strip())
        if next_text:
            parts.append("--- NEXT PARA ---\n" + next_text)
    return seed_text, whitelist, "\n".join(parts) if parts else this_text


def compute_group_hints(base_df: pd.DataFrame, whitelist_ids: Sequence[str]) -> List[str]:
    subset = base_df[base_df["sent_global_id"].astype(str).isin({str(value) for value in whitelist_ids})]
    hits: List[str] = []
    seen = set()
    for raw in subset["prop_window_hits"].tolist():
        parsed = loads_json_field(raw, [])
        values = parsed if isinstance(parsed, list) else re.findall(r"[A-Za-z]+", str(raw))
        for value in values:
            value = str(value).strip()
            if value in PROPERTY_ORDER and value not in seen:
                hits.append(value)
                seen.add(value)
    return hits


def build_groups_by_paragraph(
    cand_df: pd.DataFrame,
    para_map: Dict[str, List[Tuple[str, str]]],
    raw_gap_limit: int,
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    tmp = cand_df[["para_global_id", "sent_global_id"]].copy()
    tmp["para_global_id"] = tmp["para_global_id"].astype(str)
    tmp["sent_global_id"] = tmp["sent_global_id"].astype(str)

    for para_id, subset in tmp.groupby("para_global_id", sort=True):
        sent_positions = {sent_id: idx for idx, (sent_id, _) in enumerate(para_map[str(para_id)])}
        sent_ids = sorted(subset["sent_global_id"].astype(str).tolist(), key=lambda value: sent_positions[value])
        if not sent_ids:
            continue
        current = [sent_ids[0]]
        for prev_sent_id, sent_id in zip(sent_ids, sent_ids[1:]):
            if sent_positions[sent_id] - sent_positions[prev_sent_id] - 1 <= raw_gap_limit:
                current.append(sent_id)
            else:
                groups.append({"para_id": str(para_id), "member_sent_ids": current[:]})
                current = [sent_id]
        groups.append({"para_id": str(para_id), "member_sent_ids": current[:]})

    groups.sort(key=lambda item: (sort_key(item["para_id"]), sort_key(item["member_sent_ids"][0])))
    return groups


def call_llm(model, prompt: str, temperature: float, max_tokens: int) -> str:
    response = model.respond(
        prompt,
        config={
            "temperature": temperature,
            "maxTokens": max_tokens,
            "stop": [END_SENTINEL],
            "response_format": {"type": "json_object"},
            "format": "json",
        },
    )
    if not response or not getattr(response, "content", "").strip():
        raise RuntimeError("Empty LLM response.")
    cleaned = strip_llm_wrappers(response.content)
    if not cleaned:
        raise RuntimeError("Empty LLM content after cleanup.")
    return cleaned


def dedupe_prop_hits(items: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    clean: List[Dict[str, str]] = []
    seen = set()
    for item in items:
        tag = str(item.get("tag", "")).strip()
        sent_id = str(item.get("sent_id", "")).strip()
        if not tag or not sent_id:
            continue
        key = (tag, sent_id)
        if key not in seen:
            clean.append({"tag": tag, "sent_id": sent_id})
            seen.add(key)
    return clean


def property_mini_spec(tag: str) -> str:
    specs = {
        "Ex": "[Ex] Excitation condition for photoluminescence, usually with wavelength evidence.",
        "Em": "[Em] Emission, fluorescence, or PL peak of the sample, usually with wavelength evidence.",
        "QY": "[QY] Photoluminescence quantum yield of the sample.",
        "lifetime": "[lifetime] Photoluminescence lifetime or decay time of the sample.",
        "ExDep": "[ExDep] Explicit statement that emission depends on or is independent of excitation.",
        "Chiral": "[Chiral] Explicit statement that the final sample is chiral.",
        "CPL": "[CPL] Explicit circularly polarized luminescence of the sample.",
    }
    if tag not in specs:
        raise ValueError(f"Unknown property tag: {tag}")
    return specs[tag]


def build_prop_prompt(window_text: str, whitelist: Dict[str, str], allowed_tags: Sequence[str]) -> str:
    return dedent(
        f"""
        You are an expert in extracting photoluminescence properties for carbon dots.

        Return exactly one JSON object and then append {END_SENTINEL}. Do not add prose or markdown.

        Task:
        - Decide which sentences in WHITELIST express which property tags.
        - Work only with the active tags for this call.
        - Use only sentence ids from WHITELIST.

        Active tags:
        {json.dumps(list(allowed_tags), ensure_ascii=False)}

        Property rules:
        {" ".join(property_mini_spec(tag) for tag in allowed_tags)}

        Output schema:
        {{"prop_hits":[{{"tag":"<active tag>","sent_id":"<whitelist id>"}}]}}

        WHITELIST:
        {json.dumps([{"sent_id": sent_id, "text": text} for sent_id, text in whitelist.items()], ensure_ascii=False)}

        ===== CONTEXT WINDOW =====
        {window_text}
        ===== END =====
        Output must end with {END_SENTINEL}
        """
    ).strip()


def build_name_prompt(candidates: Sequence[str], window_text: str, whitelist: Dict[str, str]) -> str:
    return dedent(
        f"""
        You are an expert in sample-name detection for carbon-dot materials.

        Return exactly one JSON object and then append {END_SENTINEL}. Do not add prose or markdown.

        Task:
        - Decide which WHITELIST sentences mention which items from SAMPLE_NAMES.
        - Match only names from SAMPLE_NAMES.
        - Case-insensitive matching is allowed.
        - If a sentence clearly refers to a family label that covers multiple candidate names,
          return all matching candidates.

        Output schema:
        {{"name_hits":[{{"name":"<one of SAMPLE_NAMES>","sent_id":"<whitelist id>"}}]}}

        SAMPLE_NAMES:
        {json.dumps(list(candidates), ensure_ascii=False)}

        WHITELIST:
        {json.dumps([{"sent_id": sent_id, "text": text} for sent_id, text in whitelist.items()], ensure_ascii=False)}

        ===== CONTEXT WINDOW =====
        {window_text}
        ===== END =====
        Output must end with {END_SENTINEL}
        """
    ).strip()


def run_with_retries_prop(
    model,
    prompt: str,
    whitelist_ids: Sequence[str],
    log_path: str,
    phase: str,
    paper_tag: str,
    allowed_tags: Sequence[str],
    temperature: float,
    max_tokens: int,
    retries: int,
) -> Dict[str, List[Dict[str, str]]]:
    whitelist = {str(value) for value in whitelist_ids}
    allowed = set(allowed_tags)

    for attempt in range(1, max(1, int(retries or 1)) + 1):
        try:
            raw = call_llm(model, prompt, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n=== RAW[{phase}][try {attempt}] paper={paper_tag} EXCEPTION ===\n{exc}\n")
            if attempt == retries:
                return {"prop_hits": []}
            continue

        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n=== RAW[{phase}][try {attempt}] paper={paper_tag} ===\n{raw}\n")

        obj = parse_llm_json(raw) or {}
        raw_hits = obj.get("prop_hits", [])
        if not isinstance(raw_hits, list) or not raw_hits:
            return {"prop_hits": []}

        clean_hits: List[Dict[str, str]] = []
        invalid = False
        seen = set()
        for item in raw_hits:
            if not isinstance(item, dict):
                invalid = True
                continue
            tag = str(item.get("tag", "")).strip()
            sent_id = str(item.get("sent_id", "")).strip()
            if tag not in allowed or sent_id not in whitelist:
                if tag or sent_id:
                    invalid = True
                continue
            key = (tag, sent_id)
            if key not in seen:
                clean_hits.append({"tag": tag, "sent_id": sent_id})
                seen.add(key)

        if clean_hits or not invalid:
            return {"prop_hits": clean_hits}

    return {"prop_hits": []}


def run_with_retries_name(
    model,
    prompt: str,
    candidates: Sequence[str],
    whitelist_ids: Sequence[str],
    log_path: str,
    phase: str,
    paper_tag: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> Dict[str, List[Dict[str, str]]]:
    whitelist = {str(value) for value in whitelist_ids}
    candidate_map = {str(name).lower(): str(name) for name in candidates}

    for attempt in range(1, max(1, int(retries or 1)) + 1):
        try:
            raw = call_llm(model, prompt, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n=== RAW[{phase}][try {attempt}] paper={paper_tag} EXCEPTION ===\n{exc}\n")
            if attempt == retries:
                return {"name_hits": []}
            continue

        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n=== RAW[{phase}][try {attempt}] paper={paper_tag} ===\n{raw}\n")

        obj = parse_llm_json(raw) or {}
        raw_hits = obj.get("name_hits", [])
        if not isinstance(raw_hits, list) or not raw_hits:
            return {"name_hits": []}

        clean_hits: List[Dict[str, str]] = []
        invalid = False
        seen = set()
        for item in raw_hits:
            if not isinstance(item, dict):
                invalid = True
                continue
            name = str(item.get("name", "")).strip()
            sent_id = str(item.get("sent_id", "")).strip()
            canonical = candidate_map.get(name.lower())
            if canonical is None or sent_id not in whitelist:
                if name or sent_id:
                    invalid = True
                continue
            key = (canonical, sent_id)
            if key not in seen:
                clean_hits.append({"name": canonical, "sent_id": sent_id})
                seen.add(key)

        if clean_hits or not invalid:
            return {"name_hits": clean_hits}

    return {"name_hits": []}


def log_prop_input(log_path: str, paper_tag: str, para_id: str, members: Sequence[str], window_text: str, whitelist: Dict[str, str], hints: Sequence[str], phase: str) -> None:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== INPUT[{phase}] paper={paper_tag} para={para_id} group={','.join(map(str, members))} ===\n")
        fh.write("PROP_TAGS:\n" + json.dumps(list(hints), ensure_ascii=False) + "\n")
        fh.write(
            "WHITELIST:\n"
            + json.dumps([{"sent_id": sent_id, "text": text} for sent_id, text in whitelist.items()], ensure_ascii=False)
            + "\n"
        )
        fh.write("WINDOW_TEXT:\n" + (window_text or "") + "\n")


def log_name_input(log_path: str, paper_tag: str, para_id: str, members: Sequence[str], window_text: str, whitelist: Dict[str, str], candidates: Sequence[str], phase: str) -> None:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== INPUT[{phase}] paper={paper_tag} para={para_id} group={','.join(map(str, members))} ===\n")
        fh.write("CANDIDATES:\n" + json.dumps(list(candidates), ensure_ascii=False) + "\n")
        fh.write(
            "WHITELIST:\n"
            + json.dumps([{"sent_id": sent_id, "text": text} for sent_id, text in whitelist.items()], ensure_ascii=False)
            + "\n"
        )
        fh.write("WINDOW_TEXT:\n" + (window_text or "") + "\n")


def log_prop_decision(log_path: str, para_id: str, members: Sequence[str], window_used: str, prop_hits: Sequence[Dict[str, str]]) -> None:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== DECISION[PROP] para={para_id} members={','.join(map(str, members))} window={window_used} ===\n")
        fh.write("prop_hits:\n")
        if not prop_hits:
            fh.write("  (none)\n")
            return
        for item in prop_hits:
            fh.write(f"  - sent_id={item.get('sent_id', '')}, tag={item.get('tag', '')}\n")


def log_name_decision(log_path: str, para_id: str, members: Sequence[str], window_used: str, name_hits: Sequence[Dict[str, str]]) -> None:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== DECISION[NAME] para={para_id} members={','.join(map(str, members))} window={window_used} ===\n")
        fh.write("name_hits:\n")
        if not name_hits:
            fh.write("  (none)\n")
            return
        for item in name_hits:
            fh.write(f"  - sent_id={item.get('sent_id', '')}, name={item.get('name', '')}\n")


def log_props_without_name(log_path: str, paper_tag: str, para_id: str, members: Sequence[str], prop_hits: Sequence[Dict[str, str]], candidates: Sequence[str], window_used: str) -> None:
    tags = []
    sent_ids = []
    for item in prop_hits:
        tag = str(item.get("tag", "")).strip()
        sent_id = str(item.get("sent_id", "")).strip()
        if tag and tag not in tags:
            tags.append(tag)
        if sent_id and sent_id not in sent_ids:
            sent_ids.append(sent_id)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("\n=== NOTE[PROP_NO_NAME] ===\n")
        fh.write(
            f"paper={paper_tag} para={para_id} members={list(members)} window={window_used} "
            f"property_hits={tags} evidence_sent_ids={sent_ids} candidate_samples={list(candidates)}\n"
        )


def write_decision_outputs(base_df: pd.DataFrame, csv_path: str, md_path: str, txt_path: str, paper_title: str) -> None:
    ensure_dir(os.path.dirname(csv_path))
    base_df.to_csv(csv_path, index=False, encoding="utf-8")

    filtered = base_df[base_df["prop_window_hits"].map(lambda value: str(value).strip() not in ("", "nan", "[]"))].copy()
    filtered["para_global_id"] = filtered["para_global_id"].astype(str)
    filtered["sent_global_id"] = filtered["sent_global_id"].astype(str)
    filtered = filtered.sort_values(by=["para_global_id", "sent_global_id"], key=lambda series: series.map(sort_key))

    with open(md_path, "w", encoding="utf-8") as md_fh:
        md_fh.write(f"# Step 4 Decisions for {paper_title}\n\n")
        for _, row in filtered.iterrows():
            sent_id = str(row.get("sent_global_id", "")).strip()
            window_level = str(row.get("window_level", "")).strip()
            llm_name = str(row.get("LLM_name", "")).strip()
            merged_name = str(row.get("merge_LLM_name", "")).strip()
            llm_props = str(row.get("LLM_props", "")).strip()
            text = normalize_line(row.get("text", ""))
            md_fh.write(
                f"[sid: {sent_id}, window: {window_level}, samples: {merged_name or llm_name}, "
                f"LLM_name: {llm_name}, merge_LLM_name: {merged_name}, LLM_props: {llm_props}]  \n"
            )
            md_fh.write(text + "\n\n---\n\n")

    with open(txt_path, "w", encoding="utf-8") as txt_fh:
        txt_fh.write(f"Step 4 Decisions for {paper_title}\n\n")
        for _, row in filtered.iterrows():
            sent_id = str(row.get("sent_global_id", "")).strip()
            window_level = str(row.get("window_level", "")).strip()
            llm_name = str(row.get("LLM_name", "")).strip()
            merged_name = str(row.get("merge_LLM_name", "")).strip()
            llm_props = str(row.get("LLM_props", "")).strip()
            text = normalize_line(row.get("text", ""))
            txt_fh.write(
                f"[sid: {sent_id}, window: {window_level}, samples: {merged_name or llm_name}, "
                f"LLM_name: {llm_name}, merge_LLM_name: {merged_name}, LLM_props: {llm_props}]\n"
            )
            txt_fh.write(text + "\n\n---\n\n")


def process_one_paper(
    paper_dir: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    raw_gap_limit: int,
    skip_existing: bool,
    root_log_fh,
) -> bool:
    paper_id = paper_id_from_dir(paper_dir)
    if paper_id is None:
        return False

    in_paths = annotation_bundle_paths(paper_dir, "property/label_LLM_vl", paper_id)
    out_paths = annotation_bundle_paths(paper_dir, "property/decision_LLM", paper_id)
    out_dir = out_paths["out_dir"]
    prop_log = os.path.join(out_dir, f"{paper_id}_property.log")
    name_log = os.path.join(out_dir, f"{paper_id}_name.log")
    error_log = os.path.join(out_dir, "ERROR.txt")

    if not os.path.exists(in_paths["csv"]):
        root_log_fh.write(f"[SKIP] {paper_id}: input label_LLM_vl csv not found.\n")
        root_log_fh.flush()
        return False

    if skip_existing and decision_outputs_exist(out_dir, paper_id):
        print(f"[SKIP] {paper_id}: decision_LLM already exists.")
        return False

    ensure_dir(out_dir)
    with open(prop_log, "w", encoding="utf-8") as fh:
        fh.write("")
    with open(name_log, "w", encoding="utf-8") as fh:
        fh.write("")

    letter_csv = os.path.join(paper_dir, "Synthesis", "letter_table", f"{paper_id}.csv")
    candidates = load_candidate_names(paper_dir, letter_csv, log_path=name_log)
    single_candidate = len(candidates) == 1

    base_df = safe_read_csv(in_paths["csv"])
    if base_df is None:
        root_log_fh.write(f"[SKIP] {paper_id}: failed to read input csv.\n")
        root_log_fh.flush()
        return False

    required = {"sent_global_id", "para_global_id", "main_section_norm", "text", "prop_window_hits"}
    missing = required - set(base_df.columns)
    if missing:
        raise ValueError(f"{paper_id}: missing required columns in label_LLM_vl csv: {sorted(missing)}")

    base_df = base_df.copy(deep=True)
    base_df["sent_global_id"] = base_df["sent_global_id"].astype(str)
    base_df["para_global_id"] = base_df["para_global_id"].astype(str)
    for column in ("window_level", "LLM_props", "LLM_name", "merge_LLM_name"):
        if column not in base_df.columns:
            base_df[column] = ""
        base_df[column] = base_df[column].fillna("").astype(str)

    cand_df = base_df[
        base_df["main_section_norm"].astype(str).isin(ALLOWED_SECTIONS)
        & base_df["prop_window_hits"].map(lambda value: str(value).strip() not in ("", "nan", "[]"))
    ].copy()

    if cand_df.empty:
        paper_title = safe_paper_title(base_df, paper_id)
        write_decision_outputs(base_df, out_paths["csv"], out_paths["md"], out_paths["txt"], paper_title)
        root_log_fh.write(f"[OK] {paper_id}: no candidate sentences, wrote empty decision columns.\n")
        root_log_fh.flush()
        return True

    para_map, para_order = build_para_text_map(base_df[["para_global_id", "sent_global_id", "text"]].copy())
    groups = build_groups_by_paragraph(cand_df, para_map, raw_gap_limit=raw_gap_limit)
    model = lmstudio_llm(model_name)
    order_index = {tag: idx for idx, tag in enumerate(PROPERTY_ORDER)}

    for group in tqdm(groups, desc=f"Step4: decision {paper_id}", leave=False):
        para_id = group["para_id"]
        members = [str(value) for value in group["member_sent_ids"]]
        try:
            _, whitelist_small, small_text = build_window_for_group(para_map, para_order, para_id, members, mode="SMALL")
            hint_tags = compute_group_hints(base_df, whitelist_small.keys())
            primary_tags = [tag for tag in hint_tags if tag in PROPERTY_ORDER]
            secondary_tags = [tag for tag in PROPERTY_ORDER if tag not in primary_tags]
            prop_hits: List[Dict[str, str]] = []
            prop_window = "SMALL"
            extended_text: Optional[str] = None

            if primary_tags:
                log_prop_input(prop_log, paper_id, para_id, members, small_text, whitelist_small, primary_tags, "SMALL:PROP_PRIMARY")
                prop_hits.extend(
                    run_with_retries_prop(
                        model,
                        build_prop_prompt(small_text, whitelist_small, primary_tags),
                        whitelist_small.keys(),
                        prop_log,
                        "SMALL:PROP_PRIMARY",
                        paper_id,
                        primary_tags,
                        temperature,
                        max_tokens,
                        retries,
                    ).get("prop_hits", [])
                )
                prop_hits = dedupe_prop_hits(prop_hits)
                if prop_hits and secondary_tags:
                    log_prop_input(prop_log, paper_id, para_id, members, small_text, whitelist_small, secondary_tags, "SMALL:PROP_SECONDARY")
                    prop_hits.extend(
                        run_with_retries_prop(
                            model,
                            build_prop_prompt(small_text, whitelist_small, secondary_tags),
                            whitelist_small.keys(),
                            prop_log,
                            "SMALL:PROP_SECONDARY",
                            paper_id,
                            secondary_tags,
                            temperature,
                            max_tokens,
                            retries,
                        ).get("prop_hits", [])
                    )
                    prop_hits = dedupe_prop_hits(prop_hits)
                if not prop_hits:
                    prop_window = "EXTENDED"
                    _, _, extended_text = build_window_for_group(para_map, para_order, para_id, members, mode="EXTENDED")
                    log_prop_input(prop_log, paper_id, para_id, members, extended_text, whitelist_small, primary_tags, "EXTENDED:PROP_PRIMARY")
                    prop_hits.extend(
                        run_with_retries_prop(
                            model,
                            build_prop_prompt(extended_text, whitelist_small, primary_tags),
                            whitelist_small.keys(),
                            prop_log,
                            "EXTENDED:PROP_PRIMARY",
                            paper_id,
                            primary_tags,
                            temperature,
                            max_tokens,
                            retries,
                        ).get("prop_hits", [])
                    )
                    prop_hits = dedupe_prop_hits(prop_hits)
                    if prop_hits and secondary_tags:
                        log_prop_input(prop_log, paper_id, para_id, members, extended_text, whitelist_small, secondary_tags, "EXTENDED:PROP_SECONDARY")
                        prop_hits.extend(
                            run_with_retries_prop(
                                model,
                                build_prop_prompt(extended_text, whitelist_small, secondary_tags),
                                whitelist_small.keys(),
                                prop_log,
                                "EXTENDED:PROP_SECONDARY",
                                paper_id,
                                secondary_tags,
                                temperature,
                                max_tokens,
                                retries,
                            ).get("prop_hits", [])
                        )
                        prop_hits = dedupe_prop_hits(prop_hits)
            else:
                log_prop_input(prop_log, paper_id, para_id, members, small_text, whitelist_small, PROPERTY_ORDER, "SMALL:PROP_ALL")
                prop_hits.extend(
                    run_with_retries_prop(
                        model,
                        build_prop_prompt(small_text, whitelist_small, PROPERTY_ORDER),
                        whitelist_small.keys(),
                        prop_log,
                        "SMALL:PROP_ALL",
                        paper_id,
                        PROPERTY_ORDER,
                        temperature,
                        max_tokens,
                        retries,
                    ).get("prop_hits", [])
                )
                prop_hits = dedupe_prop_hits(prop_hits)
                if not prop_hits:
                    prop_window = "EXTENDED"
                    _, _, extended_text = build_window_for_group(para_map, para_order, para_id, members, mode="EXTENDED")
                    log_prop_input(prop_log, paper_id, para_id, members, extended_text, whitelist_small, PROPERTY_ORDER, "EXTENDED:PROP_ALL")
                    prop_hits.extend(
                        run_with_retries_prop(
                            model,
                            build_prop_prompt(extended_text, whitelist_small, PROPERTY_ORDER),
                            whitelist_small.keys(),
                            prop_log,
                            "EXTENDED:PROP_ALL",
                            paper_id,
                            PROPERTY_ORDER,
                            temperature,
                            max_tokens,
                            retries,
                        ).get("prop_hits", [])
                    )
                    prop_hits = dedupe_prop_hits(prop_hits)

            log_prop_decision(prop_log, para_id, members, prop_window, prop_hits)
            if not prop_hits:
                continue

            name_hits: List[Dict[str, str]] = []
            name_window = "BYPASS"
            if not single_candidate and len(candidates) > 1:
                log_name_input(name_log, paper_id, para_id, members, small_text, whitelist_small, candidates, "SMALL:NAME")
                name_hits = run_with_retries_name(
                    model,
                    build_name_prompt(candidates, small_text, whitelist_small),
                    candidates,
                    whitelist_small.keys(),
                    name_log,
                    "SMALL:NAME",
                    paper_id,
                    temperature,
                    max_tokens,
                    retries,
                ).get("name_hits", [])
                name_window = "SMALL"
                if not name_hits:
                    if extended_text is None:
                        _, _, extended_text = build_window_for_group(para_map, para_order, para_id, members, mode="EXTENDED")
                    log_name_input(name_log, paper_id, para_id, members, extended_text, whitelist_small, candidates, "EXTENDED:NAME")
                    name_hits = run_with_retries_name(
                        model,
                        build_name_prompt(candidates, extended_text, whitelist_small),
                        candidates,
                        whitelist_small.keys(),
                        name_log,
                        "EXTENDED:NAME",
                        paper_id,
                        temperature,
                        max_tokens,
                        retries,
                    ).get("name_hits", [])
                    name_window = "EXTENDED"

            log_name_decision(name_log, para_id, members, name_window, name_hits)
            if not name_hits and not single_candidate:
                log_props_without_name(prop_log, paper_id, para_id, members, prop_hits, candidates, "EXTENDED" if "EXTENDED" in {prop_window, name_window} else "SMALL")
                continue

            tags_by_sent: Dict[str, List[str]] = {}
            for item in prop_hits:
                tags_by_sent.setdefault(str(item["sent_id"]), [])
                if item["tag"] not in tags_by_sent[str(item["sent_id"])]:
                    tags_by_sent[str(item["sent_id"])].append(item["tag"])

            names_union: List[str] = [candidates[0]] if single_candidate and candidates else []
            names_by_sent: Dict[str, List[str]] = {}
            if not single_candidate:
                for item in name_hits:
                    sent_id = str(item["sent_id"])
                    names_by_sent.setdefault(sent_id, [])
                    if item["name"] not in names_by_sent[sent_id]:
                        names_by_sent[sent_id].append(item["name"])
                    if item["name"] not in names_union:
                        names_union.append(item["name"])

            unique_group_name = names_union[0] if len(names_union) == 1 else None
            final_window = "EXTENDED" if "EXTENDED" in {prop_window, name_window} else "SMALL"

            for sent_id in members:
                mask = base_df["sent_global_id"] == str(sent_id)
                if single_candidate and candidates:
                    base_df.loc[mask, "LLM_name"] = candidates[0]
                elif sent_id in names_by_sent:
                    base_df.loc[mask, "LLM_name"] = ";".join(names_by_sent[sent_id])
                elif unique_group_name is not None and sent_id in tags_by_sent:
                    base_df.loc[mask, "LLM_name"] = unique_group_name

            group_prop_union: List[str] = []
            for sent_id, tags in tags_by_sent.items():
                for tag in sorted(set(tags), key=lambda value: order_index.get(value, 999)):
                    if tag not in group_prop_union:
                        group_prop_union.append(tag)

            for sent_id in members:
                if sent_id not in tags_by_sent:
                    continue
                mask = base_df["sent_global_id"] == str(sent_id)
                tags = sorted(set(tags_by_sent[sent_id]), key=lambda value: order_index.get(value, 999))
                base_df.loc[mask, "LLM_props"] = ";".join(tags)
                base_df.loc[mask, "merge_LLM_name"] = ";".join(names_union)
                base_df.loc[mask, "window_level"] = final_window

            seed_id = members[len(members) // 2]
            seed_mask = base_df["sent_global_id"] == seed_id
            if seed_mask.any():
                current_props = [value for value in str(base_df.loc[seed_mask, "LLM_props"].iloc[0]).split(";") if value]
                current_names = [value for value in str(base_df.loc[seed_mask, "merge_LLM_name"].iloc[0]).split(";") if value]
                for tag in group_prop_union:
                    if tag not in current_props:
                        current_props.append(tag)
                for name in names_union:
                    if name not in current_names:
                        current_names.append(name)
                base_df.loc[seed_mask, "LLM_props"] = ";".join(current_props)
                base_df.loc[seed_mask, "merge_LLM_name"] = ";".join(current_names)

        except Exception as exc:
            with open(error_log, "a", encoding="utf-8") as fh:
                fh.write(f"para={para_id}, members={members}, error={exc!r}\n")
            root_log_fh.write(f"[FAIL] {paper_id}: para={para_id}, members={members}, error={exc!r}\n")
            root_log_fh.flush()

    paper_title = safe_paper_title(base_df, paper_id)
    write_decision_outputs(base_df, out_paths["csv"], out_paths["md"], out_paths["txt"], paper_title)
    root_log_fh.write(f"[OK] {paper_id}: Step 4 outputs written.\n")
    root_log_fh.flush()
    return True


def process_all_papers(
    mining_root: str,
    paper_ids: Optional[Sequence[str]] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = DEFAULT_RETRIES,
    raw_gap_limit: int = DEFAULT_RAW_GAP_LIMIT,
    skip_existing: bool = True,
) -> None:
    if lmstudio_llm is None:
        raise RuntimeError("lmstudio is not installed. Install or configure LM Studio before running Step 4.")

    root = ensure_root_exists(mining_root)
    paper_dirs = iter_paper_dirs(root, paper_ids=paper_ids)
    root_log_path = os.path.join(root, "step4_llm_decision.log")
    with open(root_log_path, "a", encoding="utf-8") as root_log_fh:
        root_log_fh.write("\n" + "=" * 80 + "\n")
        root_log_fh.write(
            f"NEW RUN model={model_name} temperature={temperature} "
            f"max_tokens={max_tokens} retries={retries} raw_gap_limit={raw_gap_limit}\n"
        )
        root_log_fh.write("=" * 80 + "\n")
        for paper_dir in tqdm(paper_dirs, desc="Step4: llm-decision"):
            process_one_paper(
                paper_dir=paper_dir,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                retries=retries,
                raw_gap_limit=raw_gap_limit,
                skip_existing=skip_existing,
                root_log_fh=root_log_fh,
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Property preprocessing step4: LLM sample/property decision.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Re-run even if Step 4 outputs already exist.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model name used for Step 4.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="LLM max tokens per call.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retry count for schema-invalid LLM outputs.")
    parser.add_argument("--raw-gap-limit", type=int, default=DEFAULT_RAW_GAP_LIMIT, help="Maximum raw sentence gap used when building groups inside one paragraph.")
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
        raw_gap_limit=args.raw_gap_limit,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
