#!/usr/bin/env python3
"""Step 04: Split cleaned markdown into token-limited chunks."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import spacy

from synthesis_unit import display_path, extract_document_id, list_document_dirs


_NLP = None


def init_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        if "parser" not in _NLP.pipe_names and "senter" not in _NLP.pipe_names:
            _NLP.add_pipe("sentencizer")



def chunk_text_by_tokens(page_text: str, max_tokens: int = 200) -> list[tuple[str, int]]:
    init_nlp()
    doc = _NLP(page_text)

    chunks: list[tuple[str, int]] = []
    buffer_text = ""
    buffer_tokens = 0

    for sent in doc.sents:
        sent_tokens = len(list(sent))
        if buffer_tokens + sent_tokens > max_tokens and buffer_text:
            chunks.append((buffer_text.strip(), buffer_tokens))
            buffer_text = ""
            buffer_tokens = 0

        buffer_text += sent.text + " "
        buffer_tokens += sent_tokens

    if buffer_text:
        chunks.append((buffer_text.strip(), buffer_tokens))

    return chunks



def output_ready(tokenized_dir: Path, document_id: str) -> bool:
    return all((tokenized_dir / f"{document_id}{suffix}").exists() for suffix in (".csv", ".txt", ".md"))



def process_document(document_dir: Path, max_tokens: int, overwrite: bool) -> str:
    document_id = extract_document_id(document_dir)

    cut_md = document_dir / "preprocess" / "cut" / f"{document_id}_cut.md"
    tokenized_dir = document_dir / "Synthesis" / "Tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    if not cut_md.exists():
        return f"Skip missing file: {display_path(cut_md)}"
    if output_ready(tokenized_dir, document_id) and not overwrite:
        return f"Skip already tokenized: {document_id}"

    text = cut_md.read_text(encoding="utf-8")
    chunks = chunk_text_by_tokens(text, max_tokens=max_tokens)

    rows = [
        {
            "pdf_name": document_id,
            "para_id": idx + 1,
            "token_count": token_count,
            "text": chunk_text,
        }
        for idx, (chunk_text, token_count) in enumerate(chunks)
    ]
    df = pd.DataFrame(rows)

    out_csv = tokenized_dir / f"{document_id}.csv"
    out_txt = tokenized_dir / f"{document_id}.txt"
    out_md = tokenized_dir / f"{document_id}.md"

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with out_txt.open("w", encoding="utf-8") as txt_handle, out_md.open("w", encoding="utf-8") as md_handle:
        for _, row in df.iterrows():
            header = f"[Para {row['para_id']}, Tokens: {row['token_count']}]"
            body = row["text"]
            txt_handle.write(f"{header}:\n{body}\n\n\n")
            md_handle.write(f"{header}  \n\n{body}\n\n---\n---\n\n")

    return f"Tokenized: {document_id}"



def run(pipeline_root: Path, workers: int, max_tokens: int, overwrite: bool) -> None:
    document_dirs = list_document_dirs(pipeline_root)
    print(f"Found {len(document_dirs)} document folders.")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(process_document, document_dir, max_tokens, overwrite) for document_dir in document_dirs]
        for future in as_completed(futures):
            print(future.result())



def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize markdown files into paragraph chunks.")
    parser.add_argument("--mining-root", dest="pipeline_root", type=Path, default=Path("data") / "mining")
    parser.add_argument("--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)))
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    run(args.pipeline_root, args.workers, args.max_tokens, args.overwrite)
