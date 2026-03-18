#!/usr/bin/env python3
"""Step 02: Convert LaTeX-heavy Markdown to plain text safely."""

from __future__ import annotations

import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

from pylatexenc.latex2text import LatexNodes2Text

from synthesis_unit import display_path, extract_document_id, list_document_dirs


_LATEX_CONVERTER = LatexNodes2Text()


def pre_normalize_units(line: str) -> str:
    s = line
    s = re.sub(r"\\operatorname\*?\{\s*min\s*\}", "min", s)

    s = re.sub(r"(?P<num>\d+(?:\.\d+)?)\\upmu\\mathrm\{L\}", lambda m: f"{m.group('num')} uL", s)
    s = re.sub(r"\\upmu\\mathrm\{L\}", " uL", s)
    s = re.sub(r"(?P<num>\d+(?:\.\d+)?)\\upmu\s*L", lambda m: f"{m.group('num')} uL", s)
    s = re.sub(r"\\upmu\s*L", " uL", s)

    s = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\\mathrm\{(?P<unit>mL|mg|h|min|nm|mM|mg/mL|mL/min|cells/mL)\}",
        lambda m: f"{m.group('num')} {m.group('unit')}",
        s,
    )
    s = re.sub(r"\\mathrm\{(mL|mg|h|min|nm|mM|mg/mL|mL/min|cells/mL)\}", lambda m: f" {m.group(1)}", s)

    s = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\^\{\\circ\}\\mathrm\{C\}",
        lambda m: f"{m.group('num')} C",
        s,
    )
    s = re.sub(r"(?P<num>\d+(?:\.\d+)?)\^\{\\circ\}C", lambda m: f"{m.group('num')} C", s)

    return s



def safe_convert_line(line: str) -> str:
    normalized = pre_normalize_units(line)
    escaped = re.sub(r"(?<!\\)%", r"\\%", normalized)

    try:
        converted = _LATEX_CONVERTER.latex_to_text(escaped)
    except Exception:
        return line.rstrip("\n")

    original = line.strip()
    new_text = converted.strip()

    if original and not new_text:
        return line.rstrip("\n")
    if len(original) >= 20 and len(new_text) <= 0.6 * len(original):
        return line.rstrip("\n")

    return new_text



def convert_markdown_safely(md_text: str) -> str:
    return "\n".join(safe_convert_line(line) for line in md_text.splitlines())



def process_document(args: tuple[Path, bool]) -> str | None:
    document_dir, overwrite = args
    document_id = extract_document_id(document_dir)

    input_md = document_dir / "preprocess" / "PDF2md" / f"{document_id}.md"
    output_md = document_dir / "preprocess" / "latex" / f"{document_id}.md"
    output_md.parent.mkdir(parents=True, exist_ok=True)

    if not input_md.exists():
        return f"Missing input markdown: {display_path(input_md)}"

    if output_md.exists() and not overwrite:
        return None

    try:
        source = input_md.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Read failed for {display_path(input_md)}: {exc}"

    refined = convert_markdown_safely(source)
    if not refined.strip():
        return f"Converted content is empty: {display_path(input_md)}"

    try:
        output_md.write_text(refined + "\n", encoding="utf-8")
    except Exception as exc:
        return f"Write failed for {display_path(output_md)}: {exc}"

    return None



def run(pipeline_root: Path, workers: int, overwrite: bool) -> None:
    document_dirs = list_document_dirs(pipeline_root)
    print(f"Found {len(document_dirs)} document folders.")

    errors: list[str] = []
    with Pool(processes=workers) as pool:
        for result in pool.imap_unordered(process_document, [(document_dir, overwrite) for document_dir in document_dirs]):
            if result:
                errors.append(result)

    if errors:
        print("Completed with errors:")
        for msg in errors:
            print(f"- {msg}")
    else:
        print("Completed without errors.")



def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert markdown files from PDF2md to latex-clean text.")
    parser.add_argument("--mining-root", dest="pipeline_root", type=Path, default=Path("data") / "mining")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    run(args.pipeline_root, args.workers, args.overwrite)
