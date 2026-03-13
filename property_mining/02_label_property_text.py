#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step 2: run regex sentence labeling and optional LLM-based intro correction."""

import argparse
import os
from textwrap import dedent
from typing import List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

from property_unit import (
    annotate_dataframe,
    annotation_bundle_paths,
    bundle_exists,
    derive_para_keys,
    ensure_root_exists,
    has_property_hits,
    iter_paper_dirs,
    loads_json_field,
    parse_json_object_text as parse_json_object,
    paper_id_from_dir,
    safe_paper_title,
    write_annotation_bundle,
)

try:
    from lmstudio import llm as lmstudio_llm
except Exception:
    lmstudio_llm = None


DEFAULT_MODEL = "qwen.qwen2.5-vl-32b-instruct"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 8000
DEFAULT_PROMPT_TRUNC = 20000
RESULT_SECTION_NAME = "results_discussion"


def build_prompt_for_paragraph(
    paper_title: str,
    para_id: str,
    intro_rows_all: pd.DataFrame,
    intro_rows_candidates: pd.DataFrame,
    prompt_trunc: int,
) -> str:
    header = dedent(
        f"""
        You are an expert scientific editor for materials papers.

        TASK:
        From the INTRODUCTION sentences of ONE paragraph below, decide which sentences actually describe
        THIS PAPER's OWN work/results/findings and therefore should be promoted from 'intro' to 'results_discussion'.

        DECISION CRITERIA:
        - Promote ONLY sentences that clearly refer to the present work using cues like:
          "In this work", "Herein", "In this study", "We", "Our", "This paper", "This article",
          along with verbs such as "prepared/synthesized/fabricated/demonstrate/report/show/measure".
        - Sentences summarizing prior literature or others' work should remain 'intro'.
          Words like "previously", "has been reported", explicit citations such as "[12]" MAY appear in a mixed paragraph.
          DO NOT automatically classify a sentence as others' work solely due to these markers.
          Mixed paragraphs are common: if a sentence contains both reference markers and present-work cues,
          judge carefully sentence-by-sentence; promote only those clearly about THIS paper.

        INPUT SCOPE:
        - Only judge within the given paragraph context.
        - The context shows ALL intro sentences in the paragraph.
        - The candidates list shows the intro sentences that have property tags.

        STRICT OUTPUT (JSON ONLY):
        {{
          "promote_to_result": [<sent_global_id:int>, ...]
        }}

        PAPER_TITLE: {paper_title or "N/A"}
        PARAGRAPH_ID: {para_id}
        ===== PARAGRAPH CONTEXT (ALL INTRO SENTENCES, ordered) =====
        """
    ).strip()

    ctx_lines = []
    for _, row in intro_rows_all.sort_values("sent_global_id").iterrows():
        ctx_lines.append(f"[gid={row.get('sent_global_id')}] {str(row.get('text', '')).replace(chr(10), ' ').strip()}")

    cand_lines = []
    for _, row in intro_rows_candidates.sort_values("sent_global_id").iterrows():
        props = loads_json_field(row.get("prop_window_hits"), None)
        if props is None:
            props = loads_json_field(row.get("cand_props"), [])
        nums = loads_json_field(row.get("numbers_units"), [])
        nums_str = "; ".join(
            f"{item.get('value_str', '')} {item.get('unit', '')}".strip()
            for item in nums
        ) if isinstance(nums, list) else ""
        text = str(row.get("text", "")).replace("\n", " ").strip()
        cand_lines.append(f'[gid={row.get("sent_global_id")}] text="{text}" props={props} nums="{nums_str}"')

    prompt = (
        f"{header}\n"
        f"{os.linesep.join(ctx_lines)}\n"
        f"===== CANDIDATES (subset to judge) =====\n"
        f"{os.linesep.join(cand_lines) if cand_lines else '(none)'}\n"
        "===== END ====="
    )
    return prompt[:prompt_trunc] if len(prompt) > prompt_trunc else prompt


def llm_promote_ids(model, prompt: str, temperature: float, max_tokens: int) -> List[int]:
    response = model.respond(prompt, config={"temperature": temperature, "maxTokens": max_tokens})
    if not response or not getattr(response, "content", "").strip():
        raise RuntimeError("Empty LLM response.")
    data = parse_json_object(response.content)
    if data is None:
        raise RuntimeError("Failed to parse LLM JSON object.")
    ids = data.get("promote_to_result", [])
    if not isinstance(ids, list):
        raise RuntimeError("promote_to_result is not a list.")
    out: List[int] = []
    for item in ids:
        try:
            out.append(int(item))
        except Exception:
            continue
    return sorted(set(out))


def run_intro_llm_correction(
    df: pd.DataFrame,
    paper_title: str,
    log_path: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    prompt_trunc: int,
) -> pd.DataFrame:
    if lmstudio_llm is None:
        raise RuntimeError("lmstudio is not installed, cannot run LLM correction.")

    model = lmstudio_llm(model_name)
    out = df.copy()
    out["_para_key"] = derive_para_keys(out)
    out["__is_intro"] = out["main_section_norm"].astype(str).str.lower() == "intro"
    out["__has_props"] = out.apply(has_property_hits, axis=1)

    candidate_para_keys = sorted(
        out.loc[out["__is_intro"] & out["__has_props"], "_para_key"].astype(str).unique().tolist()
    )

    promoted_all = set()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# LLM Changes Log for {paper_title}\n\n")

    for para_key in candidate_para_keys:
        intro_rows_all = out[(out["_para_key"] == para_key) & (out["__is_intro"])].copy()
        intro_rows_candidates = intro_rows_all[intro_rows_all["__has_props"]].copy()
        if intro_rows_all.empty or intro_rows_candidates.empty:
            continue
        prompt = build_prompt_for_paragraph(
            paper_title=paper_title,
            para_id=str(para_key),
            intro_rows_all=intro_rows_all,
            intro_rows_candidates=intro_rows_candidates,
            prompt_trunc=prompt_trunc,
        )
        try:
            promote_ids = llm_promote_ids(model, prompt, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"## PARAGRAPH {para_key}\n- ERROR: {exc}\n\n")
            continue
        if not promote_ids:
            continue

        promoted_all.update(promote_ids)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"## PARAGRAPH {para_key}\n")
            fh.write(f"- promote_to_result: {promote_ids}\n")
            for _, row in intro_rows_all.sort_values("sent_global_id").iterrows():
                text = str(row.get("text", "")).replace("\n", " ").strip()
                fh.write(f"  [gid={row.get('sent_global_id')}] {text}\n")
            fh.write("\n")

    if promoted_all:
        mask = out["__is_intro"] & out["sent_global_id"].isin(promoted_all)
        out.loc[mask, "main_section_norm"] = RESULT_SECTION_NAME

    return out.drop(columns=["_para_key", "__is_intro", "__has_props"])


def process_one_paper(
    paper_dir: str,
    window_size: int,
    regex_only: bool,
    skip_existing: bool,
    model_name: str,
    temperature: float,
    max_tokens: int,
    prompt_trunc: int,
) -> bool:
    paper_id = paper_id_from_dir(paper_dir)
    if paper_id is None:
        return False

    token_csv = os.path.join(paper_dir, "property", "Tokenized", f"{paper_id}.csv")
    if not os.path.exists(token_csv):
        print(f"[SKIP] {paper_id}: Tokenized csv not found.")
        return False

    label_paths = annotation_bundle_paths(paper_dir, "property/label", paper_id)
    llm_paths = annotation_bundle_paths(paper_dir, "property/label_LLM", paper_id)
    llm_log = os.path.join(llm_paths["out_dir"], f"{paper_id}_llm_changes.log")

    if skip_existing and bundle_exists(llm_paths["out_dir"], paper_id):
        print(f"[SKIP] {paper_id}: label_LLM already exists.")
        return False

    df = pd.read_csv(token_csv)
    required = [
        "pdf_name",
        "block_id",
        "main_section_norm",
        "main_header_text",
        "para_id_in_block",
        "para_global_id",
        "sent_id_in_para",
        "sent_global_id",
        "text",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"{paper_id}: missing required columns in Tokenized csv: {missing}")

    regex_df = annotate_dataframe(df, window_size=window_size, return_full=True)
    paper_title = safe_paper_title(regex_df, paper_id)
    write_annotation_bundle(regex_df, label_paths["csv"], label_paths["md"], label_paths["txt"], paper_title)

    if regex_only:
        write_annotation_bundle(regex_df, llm_paths["csv"], llm_paths["md"], llm_paths["txt"], paper_title)
        with open(llm_log, "w", encoding="utf-8") as fh:
            fh.write("LLM correction skipped; label_LLM is a copy of regex label output.\n")
        print(f"[OK] {paper_id}: regex label saved, LLM stage skipped.")
        return True

    llm_df = run_intro_llm_correction(
        df=regex_df,
        paper_title=paper_title,
        log_path=llm_log,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_trunc=prompt_trunc,
    )
    write_annotation_bundle(llm_df, llm_paths["csv"], llm_paths["md"], llm_paths["txt"], paper_title)
    print(f"[OK] {paper_id}: regex label + LLM correction saved.")
    return True


def process_all_papers(
    mining_root: str,
    paper_ids: Optional[Sequence[str]] = None,
    window_size: int = 1,
    regex_only: bool = False,
    skip_existing: bool = True,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    prompt_trunc: int = DEFAULT_PROMPT_TRUNC,
) -> None:
    root = ensure_root_exists(mining_root)
    paper_dirs = iter_paper_dirs(root, paper_ids=paper_ids)
    for paper_dir in tqdm(paper_dirs, desc="Step2: label+llm"):
        process_one_paper(
            paper_dir=paper_dir,
            window_size=window_size,
            regex_only=regex_only,
            skip_existing=skip_existing,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_trunc=prompt_trunc,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Property preprocessing step2: regex labeling + LLM correction.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--window", type=int, default=1, help="Sentence window size for regex labeling.")
    parser.add_argument("--regex-only", action="store_true", help="Skip the LLM correction phase and copy regex label to label_LLM.")
    parser.add_argument("--force", action="store_true", help="Re-run even if label_LLM already exists.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LM Studio model name used for intro correction.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--prompt-trunc", type=int, default=DEFAULT_PROMPT_TRUNC)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.regex_only and lmstudio_llm is None:
        raise RuntimeError("lmstudio is not installed. Use --regex-only or install/configure LM Studio first.")
    process_all_papers(
        mining_root=args.root,
        paper_ids=args.paper_ids,
        window_size=args.window,
        regex_only=args.regex_only,
        skip_existing=not args.force,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        prompt_trunc=args.prompt_trunc,
    )


if __name__ == "__main__":
    main()
