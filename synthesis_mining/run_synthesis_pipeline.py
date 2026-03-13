#!/usr/bin/env python3
"""Run synthesis preprocessing Steps 1-12 from a single entry point."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def load_callable(file_name: str, module_name: str, attr_name: str):
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


run_step1 = load_callable("01_pdf_to_markdown.py", "step01_pdf_to_markdown", "convert_pdf_root")
run_step2 = load_callable("02_latex_to_text.py", "step02_latex_to_text", "run")
run_step3 = load_callable("03_trim_markdown_tail.py", "step03_trim_markdown_tail", "run")
run_step4 = load_callable("04_tokenize_markdown.py", "step04_tokenize_markdown", "run")
run_step5 = load_callable("05_similarity_filter.py", "step05_similarity_filter", "run")
run_step6 = load_callable("06_llm_synthesis_decision.py", "step06_llm_synthesis_decision", "run")
run_step7 = load_callable("07_llm_refine_synthesis.py", "step07_llm_refine_synthesis", "run")
run_step8 = load_callable("08_llm_extract_sample_names.py", "step08_llm_extract_sample_names", "run")
run_step9 = load_callable("09_llm_extract_tables.py", "step09_llm_extract_tables", "run")
run_step10 = load_callable("10_merge_extractions_to_csv.py", "step10_merge_extractions_to_csv", "run")
run_step11 = load_callable("11_normalize_abbreviation_conflicts.py", "step11_normalize_abbreviation_conflicts", "run")
run_step12 = load_callable("12_llm_fill_review_fields.py", "step12_llm_fill_review_fields", "run")


def build_arg_parser() -> argparse.ArgumentParser:
    default_workers = min(8, max(1, os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description="Run synthesis preprocessing Steps 1-12.")
    parser.add_argument("--pdf-root", type=Path, default=Path("data") / "pdfs", help="Directory containing source PDF files.")
    parser.add_argument("--mining-root", dest="pipeline_root", type=Path, default=Path("data") / "mining", help="Pipeline root where per-document folders are written.")
    parser.add_argument("--templates-path", type=Path, default=Path(__file__).resolve().parent / "experiment_templates.txt", help="Template list for Step 5 similarity filtering.")
    parser.add_argument("--start-step", type=int, default=1, choices=range(1, 13), help="First step to run.")
    parser.add_argument("--end-step", type=int, default=12, choices=range(1, 13), help="Last step to run.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if stage outputs already exist.")
    parser.add_argument("--workers", type=int, default=default_workers, help="Worker count shared by Steps 2, 4, and 5.")
    parser.add_argument("--max-tokens", type=int, default=200, help="Chunk token limit for Step 4.")
    parser.add_argument("--similarity-model", default="all-MiniLM-L6-v2", help="Sentence-transformer model for Step 5.")
    parser.add_argument("--max-threshold", type=float, default=0.5, help="Step 5 maximum-similarity threshold.")
    parser.add_argument("--mean-threshold", type=float, default=0.2, help="Step 5 mean-similarity threshold.")
    parser.add_argument("--start-from-id", default="", help="Only process document ids greater than or equal to this value for Steps 5 and 9.")
    parser.add_argument("--decision-retries", type=int, default=3, help="Retry count for Step 6.")
    parser.add_argument("--decision-retry-delay", type=int, default=10, help="Retry delay in seconds for Step 6.")
    parser.add_argument("--table-max-attempts", type=int, default=5, help="Maximum extraction attempts for Step 9.")
    parser.add_argument("--table-required-success", type=int, default=3, help="Required successful tables for Step 9.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.start_step > args.end_step:
        raise SystemExit("--start-step must be less than or equal to --end-step.")

    step_calls = [
        (1, lambda: run_step1(args.pdf_root, args.pipeline_root)),
        (2, lambda: run_step2(args.pipeline_root, args.workers, args.overwrite)),
        (3, lambda: run_step3(args.pipeline_root, args.overwrite)),
        (4, lambda: run_step4(args.pipeline_root, args.workers, args.max_tokens, args.overwrite)),
        (
            5,
            lambda: run_step5(
                args.pipeline_root,
                args.templates_path,
                args.similarity_model,
                args.max_threshold,
                args.mean_threshold,
                args.workers,
                args.overwrite,
            ),
        ),
        (
            6,
            lambda: run_step6(
                args.pipeline_root,
                args.decision_retries,
                args.decision_retry_delay,
                args.overwrite,
            ),
        ),
        (7, lambda: run_step7(args.pipeline_root, args.overwrite)),
        (8, lambda: run_step8(args.pipeline_root, args.overwrite)),
        (
            9,
            lambda: run_step9(
                args.pipeline_root,
                args.start_from_id,
                args.table_max_attempts,
                args.table_required_success,
                args.overwrite,
            ),
        ),
        (10, lambda: run_step10(args.pipeline_root, args.overwrite)),
        (11, lambda: run_step11(args.pipeline_root)),
        (12, lambda: run_step12(args.pipeline_root, args.overwrite)),
    ]

    for step_no, runner in step_calls:
        if args.start_step <= step_no <= args.end_step:
            runner()


if __name__ == "__main__":
    main()
