#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run property preprocessing Steps 1-14 from a single entry point."""

import argparse
import importlib.util
import sys
from pathlib import Path


def load_process_all(file_name: str, module_name: str):
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.process_all_papers


run_step1 = load_process_all("01_tokenize_property_text.py", "step01_tokenize_property_text")
run_step2 = load_process_all("02_label_property_text.py", "step02_label_property_text")
run_step3 = load_process_all("03_ocr_tables_and_relabel.py", "step03_ocr_tables_and_relabel")
run_step4 = load_process_all("04_llm_decide_samples_and_props.py", "step04_llm_decide_samples_and_props")
run_step5 = load_process_all("05_llm_refine_property_text.py", "step05_llm_refine_property_text")
run_step6 = load_process_all("06_llm_clean_refined_properties.py", "step06_llm_clean_refined_properties")
run_step7 = load_process_all("07_llm_resolve_multisample_main.py", "step07_llm_resolve_multisample_main")
run_step8 = load_process_all("08_llm_route_main_to_app_vs.py", "step08_llm_route_main_to_app_vs")
run_step9 = load_process_all("09_llm_bind_app_vs_samples.py", "step09_llm_bind_app_vs_samples")
run_step10 = load_process_all("10_llm_review_final_properties.py", "step10_llm_review_final_properties")
run_step11 = load_process_all("11_llm_structure_and_deduplicate_properties.py", "step11_llm_structure_and_deduplicate_properties")
run_step12 = load_process_all("12_llm_resolve_property_conflicts.py", "step12_llm_resolve_property_conflicts")
run_step13 = load_process_all("13_llm_resolve_change_entries.py", "step13_llm_resolve_change_entries")
run_step14 = load_process_all("14_export_property_letter_table.py", "step14_export_property_letter_table")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run property preprocessing Steps 1-14.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--start-step", type=int, default=1, choices=range(1, 15), help="First step to run.")
    parser.add_argument("--end-step", type=int, default=14, choices=range(1, 15), help="Last step to run.")
    parser.add_argument("--force", action="store_true", help="Re-run even if stage outputs already exist.")
    parser.add_argument("--no-save-paragraphs", action="store_true", help="Do not save Step 1 Paragraphs intermediates.")
    parser.add_argument("--window", type=int, default=1, help="Sentence window size for Steps 2 and 3.")
    parser.add_argument("--regex-only", action="store_true", help="Skip Step 2 LLM correction and copy regex labels to label_LLM.")
    parser.add_argument(
        "--vl-mode",
        choices=["run", "skip", "prompt"],
        default="prompt",
        help="Step 3 mode: run VL OCR, skip it, or ask interactively.",
    )
    parser.add_argument("--step2-model", default="qwen.qwen2.5-vl-32b-instruct", help="LM Studio model name for Step 2 intro correction.")
    parser.add_argument("--step3-model", default="qwen2.5-vl-32b-instruct", help="LM Studio VL model name for Step 3 table OCR.")
    parser.add_argument("--step4-model", default="qwen.qwen2.5-vl-32b-instruct", help="LM Studio model name for Step 4 sample/property decisions.")
    parser.add_argument("--step4-temperature", type=float, default=0.0, help="LLM temperature for Step 4.")
    parser.add_argument("--step4-max-tokens", type=int, default=1200, help="LLM max tokens for Step 4.")
    parser.add_argument("--step4-retries", type=int, default=3, help="Retry count for schema-invalid Step 4 outputs.")
    parser.add_argument("--step4-raw-gap-limit", type=int, default=2, help="Maximum raw sentence gap used for Step 4 grouping.")
    parser.add_argument("--step5-model", default="qwen.qwen2.5-vl-32b-instruct", help="LM Studio model name for Step 5.")
    parser.add_argument("--step5-temperature", type=float, default=0.35, help="LLM temperature for Step 5.")
    parser.add_argument("--step5-max-tokens", type=int, default=2800, help="LLM max tokens for Step 5.")
    parser.add_argument("--step5-retries", type=int, default=3, help="Retry count for invalid Step 5 outputs.")
    parser.add_argument("--step6-model", default="qwen.qwen2.5-vl-32b-instruct", help="LM Studio model name for Step 6.")
    parser.add_argument("--step6-temperature", type=float, default=0.5, help="LLM temperature for Step 6.")
    parser.add_argument("--step6-max-tokens", type=int, default=200, help="LLM max tokens for Step 6.")
    parser.add_argument("--step9-model", default="qwen.qwen2.5-vl-32b-instruct", help="LM Studio model name for Step 9.")
    parser.add_argument("--step9-temperature", type=float, default=0.2, help="LLM temperature for Step 9.")
    parser.add_argument("--step9-max-tokens", type=int, default=2000, help="LLM max tokens for Step 9.")
    parser.add_argument("--step9-retries", type=int, default=3, help="Retry count for invalid Step 9 outputs.")
    parser.add_argument("--step10-model", default="qwen.qwen2.5-vl-32b-instruct", help="LM Studio model name for Step 10.")
    parser.add_argument("--step10-temperature", type=float, default=0.25, help="LLM temperature for Step 10.")
    parser.add_argument("--step10-max-tokens", type=int, default=1000, help="LLM max tokens for Step 10.")
    parser.add_argument("--step10-retries", type=int, default=3, help="Retry count for invalid Step 10 outputs.")
    parser.add_argument("--step10-votes", type=int, default=3, help="Review vote count for Step 10.")
    parser.add_argument("--step11-model", default="qwen.qwen2.5-vl-32b-instruct", help="LM Studio model name for Step 11.")
    parser.add_argument("--step11-temperature", type=float, default=0.3, help="LLM temperature for Step 11.")
    parser.add_argument("--step11-max-tokens", type=int, default=650, help="LLM max tokens for Step 11.")
    parser.add_argument("--step11-retries", type=int, default=5, help="Retry count per Step 11 vote.")
    parser.add_argument("--step11-votes", type=int, default=3, help="Vote count for Step 11 structure and sentence selection.")
    parser.add_argument("--step12-model", default="qwen.qwen2.5-vl-32b-instruct", help="LM Studio model name for Step 12.")
    parser.add_argument("--step12-temperature", type=float, default=0.2, help="LLM temperature for Step 12.")
    parser.add_argument("--step12-max-tokens", type=int, default=1800, help="LLM max tokens for Step 12.")
    parser.add_argument("--step12-retries", type=int, default=5, help="Retry count per Step 12 vote.")
    parser.add_argument("--step12-votes", type=int, default=3, help="Vote count for Step 12 candidate selection.")
    parser.add_argument("--step13-model", default="qwen.qwen2.5-vl-32b-instruct", help="LM Studio model name for Step 13.")
    parser.add_argument("--step13-temperature", type=float, default=0.2, help="LLM temperature for Step 13.")
    parser.add_argument("--step13-max-tokens", type=int, default=1800, help="LLM max tokens for Step 13.")
    parser.add_argument("--step13-retries", type=int, default=5, help="Retry count per Step 13 vote.")
    parser.add_argument("--step13-votes", type=int, default=3, help="Vote count for Step 13 merge and structure calls.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.start_step > args.end_step:
        raise SystemExit("--start-step must be less than or equal to --end-step.")

    step_calls = [
        (
            1,
            lambda: run_step1(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                save_paragraphs=not args.no_save_paragraphs,
                skip_existing=not args.force,
            ),
        ),
        (
            2,
            lambda: run_step2(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                window_size=args.window,
                regex_only=args.regex_only,
                skip_existing=not args.force,
                model_name=args.step2_model,
            ),
        ),
        (
            3,
            lambda: run_step3(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                window_size=args.window,
                vl_mode=args.vl_mode,
                skip_existing=not args.force,
                model_name=args.step3_model,
            ),
        ),
        (
            4,
            lambda: run_step4(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                model_name=args.step4_model,
                temperature=args.step4_temperature,
                max_tokens=args.step4_max_tokens,
                retries=args.step4_retries,
                raw_gap_limit=args.step4_raw_gap_limit,
                skip_existing=not args.force,
            ),
        ),
        (
            5,
            lambda: run_step5(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                model_name=args.step5_model,
                temperature=args.step5_temperature,
                max_tokens=args.step5_max_tokens,
                retries=args.step5_retries,
                skip_existing=not args.force,
            ),
        ),
        (
            6,
            lambda: run_step6(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                model_name=args.step6_model,
                temperature=args.step6_temperature,
                max_tokens=args.step6_max_tokens,
                skip_existing=not args.force,
            ),
        ),
        (
            7,
            lambda: run_step7(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                skip_existing=not args.force,
            ),
        ),
        (
            8,
            lambda: run_step8(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                skip_existing=not args.force,
            ),
        ),
        (
            9,
            lambda: run_step9(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                skip_existing=not args.force,
                model_name=args.step9_model,
                temperature=args.step9_temperature,
                max_tokens=args.step9_max_tokens,
                retries=args.step9_retries,
            ),
        ),
        (
            10,
            lambda: run_step10(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                skip_existing=not args.force,
                model_name=args.step10_model,
                temperature=args.step10_temperature,
                max_tokens=args.step10_max_tokens,
                retries=args.step10_retries,
                votes=args.step10_votes,
            ),
        ),
        (
            11,
            lambda: run_step11(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                skip_existing=not args.force,
                model_name=args.step11_model,
                temperature=args.step11_temperature,
                max_tokens=args.step11_max_tokens,
                retries=args.step11_retries,
                votes=args.step11_votes,
            ),
        ),
        (
            12,
            lambda: run_step12(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                skip_existing=not args.force,
                model_name=args.step12_model,
                temperature=args.step12_temperature,
                max_tokens=args.step12_max_tokens,
                retries=args.step12_retries,
                votes=args.step12_votes,
            ),
        ),
        (
            13,
            lambda: run_step13(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                skip_existing=not args.force,
                model_name=args.step13_model,
                temperature=args.step13_temperature,
                max_tokens=args.step13_max_tokens,
                retries=args.step13_retries,
                votes=args.step13_votes,
            ),
        ),
        (
            14,
            lambda: run_step14(
                mining_root=args.root,
                paper_ids=args.paper_ids,
                skip_existing=not args.force,
            ),
        ),
    ]

    for step_no, runner in step_calls:
        if args.start_step <= step_no <= args.end_step:
            runner()


if __name__ == "__main__":
    main()
