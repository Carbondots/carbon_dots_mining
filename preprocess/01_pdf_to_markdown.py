#!/usr/bin/env python3
"""Step 01: Convert PDF files to Markdown and intermediate artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze


REQUIRED_OUTPUT_SUFFIXES = (
    ".md",
    "_model.pdf",
    "_layout.pdf",
    "_spans.pdf",
    "_content_list.json",
    "_middle.json",
)


def is_already_converted(output_dir: Path, base_name: str) -> bool:
    return all((output_dir / f"{base_name}{suffix}").exists() for suffix in REQUIRED_OUTPUT_SUFFIXES)


def convert_pdf_root(pdf_root: Path, pipeline_root: Path) -> None:
    if not pdf_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {pdf_root}")

    pdf_files = sorted(p for p in pdf_root.iterdir() if p.suffix.lower() == ".pdf")
    reader = FileBasedDataReader("")

    for pdf_path in pdf_files:
        document_id = pdf_path.stem
        pdf_output_dir = pipeline_root / document_id / "preprocess" / "PDF2md"
        image_dir = pdf_output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        if is_already_converted(pdf_output_dir, document_id):
            print(f"Skip already converted file: {pdf_path.name}")
            continue

        print(f"Converting: {pdf_path.name}")

        image_writer = FileBasedDataWriter(str(image_dir))
        output_writer = FileBasedDataWriter(str(pdf_output_dir))

        pdf_bytes = reader.read(str(pdf_path))
        dataset = PymuDocDataset(pdf_bytes)

        if dataset.classify() == SupportedPdfParseMethod.OCR:
            infer_result = dataset.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = dataset.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        infer_result.draw_model(str(pdf_output_dir / f"{document_id}_model.pdf"))
        pipe_result.draw_layout(str(pdf_output_dir / f"{document_id}_layout.pdf"))
        pipe_result.draw_span(str(pdf_output_dir / f"{document_id}_spans.pdf"))

        pipe_result.dump_md(output_writer, f"{document_id}.md", "images")
        pipe_result.dump_content_list(output_writer, f"{document_id}_content_list.json", "images")
        pipe_result.dump_middle_json(output_writer, f"{document_id}_middle.json")

        print(f"Finished: {pdf_path.name}")



def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown pipeline artifacts.")
    parser.add_argument(
        "--pdf-root",
        "--input-pdf-dir",
        dest="pdf_root",
        type=Path,
        default=Path("data") / "pdfs",
        help="Directory containing source PDF files.",
    )
    parser.add_argument(
        "--mining-root",
        "--output-root-dir",
        dest="pipeline_root",
        type=Path,
        default=Path("data") / "mining",
        help="Pipeline root where per-document folders are written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    convert_pdf_root(args.pdf_root, args.pipeline_root)
