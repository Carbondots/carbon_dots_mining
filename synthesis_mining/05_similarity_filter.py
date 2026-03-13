#!/usr/bin/env python3
"""Step 05: Score chunk similarity against synthesis templates."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from synthesis_unit import display_path, extract_document_id, list_document_dirs


_WORKER_MODEL = None
_WORKER_TEMPLATE_EMB = None
SCRIPT_DIR = Path(__file__).parent
DEFAULT_TEMPLATES_PATH = SCRIPT_DIR / "experiment_templates.txt"


def load_templates(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]



def parse_tokenized_md(md_path: Path, document_id: str) -> pd.DataFrame:
    md_text = md_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n---\n---\n+", md_text.strip())

    rows: list[dict] = []
    for block in blocks:
        lines = [line for line in block.strip().splitlines()]
        if not lines:
            continue

        match = re.match(r"\[Para\s+(\d+),\s*Tokens:\s*(\d+)\]", lines[0].strip())
        if not match:
            continue

        para_id = int(match.group(1))
        token_count = int(match.group(2))

        body_start = 1
        if len(lines) > 1 and lines[1].strip() == "":
            body_start = 2

        text = "\n".join(lines[body_start:]).strip()
        rows.append(
            {
                "pdf_name": document_id,
                "para_id": para_id,
                "token_count": token_count,
                "text": text,
            }
        )

    return pd.DataFrame(rows, columns=["pdf_name", "para_id", "token_count", "text"])



def _worker_init(model_name_or_path: str, templates: list[str]) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    global _WORKER_MODEL, _WORKER_TEMPLATE_EMB

    _WORKER_MODEL = SentenceTransformer(model_name_or_path, device="cpu")
    emb = _WORKER_MODEL.encode(templates, batch_size=32, convert_to_tensor=False)
    emb = np.asarray(emb, dtype=np.float32)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    _WORKER_TEMPLATE_EMB = emb



def score_similarity(df: pd.DataFrame, max_threshold: float, mean_threshold: float) -> pd.DataFrame:
    global _WORKER_MODEL, _WORKER_TEMPLATE_EMB
    if _WORKER_MODEL is None or _WORKER_TEMPLATE_EMB is None:
        raise RuntimeError("Worker model is not initialized.")

    text_embeddings = _WORKER_MODEL.encode(df["text"].astype(str).tolist(), batch_size=32, convert_to_tensor=False)
    text_embeddings = np.asarray(text_embeddings, dtype=np.float32)
    text_embeddings = text_embeddings / np.clip(np.linalg.norm(text_embeddings, axis=1, keepdims=True), 1e-12, None)

    sims = cosine_similarity(text_embeddings, _WORKER_TEMPLATE_EMB)
    max_sim = sims.max(axis=1)
    mean_sim = sims.mean(axis=1)

    df = df.copy()
    df["max_similarity"] = max_sim
    df["mean_similarity"] = mean_sim
    df["retain"] = np.where((max_sim > max_threshold) | (mean_sim > mean_threshold), "YES", "NO")
    return df



def process_document(args: tuple[Path, float, float, bool]) -> str:
    document_dir, max_threshold, mean_threshold, overwrite = args
    document_id = extract_document_id(document_dir)

    tokenized_md = document_dir / "Synthesis" / "Tokenized" / f"{document_id}.md"
    if not tokenized_md.exists():
        return f"Skip missing tokenized markdown: {safe_display_path(tokenized_md)}"

    cos_dir = document_dir / "Synthesis" / "cos"
    cos_dir.mkdir(parents=True, exist_ok=True)

    out_csv = cos_dir / f"{document_id}.csv"
    out_txt = cos_dir / f"{document_id}.txt"
    out_md = cos_dir / f"{document_id}.md"
    if out_csv.exists() and out_txt.exists() and out_md.exists() and not overwrite:
        return f"Skip already processed: {document_id}"

    df = parse_tokenized_md(tokenized_md, document_id)
    if df.empty:
        return f"Skip empty tokenized content: {document_id}"

    scored = score_similarity(df, max_threshold=max_threshold, mean_threshold=mean_threshold)
    scored.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with out_txt.open("w", encoding="utf-8") as handle:
        for _, row in scored.iterrows():
            handle.write(
                f"[Para {row['para_id']}, Tokens: {row['token_count']}] "
                f"(max_sim={row['max_similarity']:.4f}, mean_sim={row['mean_similarity']:.4f}, retain={row['retain']}):\n"
                f"{str(row['text']).strip()}\n\n\n"
            )

    with out_md.open("w", encoding="utf-8") as handle:
        for _, row in scored.iterrows():
            handle.write(
                f"[Para {row['para_id']}, Tokens: {row['token_count']}]  \n"
                f"**max_sim**: {row['max_similarity']:.4f} | "
                f"**mean_sim**: {row['mean_similarity']:.4f} | "
                f"**retain**: {row['retain']}\n\n"
                f"{str(row['text']).strip()}\n\n---\n---\n\n"
            )

    yes_count = int((scored["retain"] == "YES").sum())
    return f"Processed {document_id}: {len(scored)} chunks, {yes_count} retained"



def run(
    pipeline_root: Path,
    templates_path: Path,
    model_name_or_path: str,
    max_threshold: float,
    mean_threshold: float,
    workers: int,
    overwrite: bool,
) -> None:
    templates_path = Path(templates_path)
    if not templates_path.exists():
        raise FileNotFoundError(
            f"Synthesis template file not found: {safe_display_path(templates_path)}. "
            f"Expected default path: {safe_display_path(DEFAULT_TEMPLATES_PATH)}."
        )

    print(f"Using synthesis templates file: {display_path(templates_path)}")
    templates = load_templates(templates_path)
    document_dirs = list_document_dirs(pipeline_root)
    print(f"Found {len(document_dirs)} document folders.")

    try:
        ctx = multiprocessing.get_context("forkserver")
    except ValueError:
        ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(
        processes=max(1, workers),
        initializer=_worker_init,
        initargs=(model_name_or_path, templates),
        maxtasksperchild=10,
    ) as pool:
        for message in pool.imap_unordered(
            process_document,
            [(document_dir, max_threshold, mean_threshold, overwrite) for document_dir in document_dirs],
        ):
            print(message)



def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score chunk relevance for synthesis extraction.")
    parser.add_argument("--mining-root", dest="pipeline_root", type=Path, default=Path("data") / "mining")
    parser.add_argument(
        "--templates-path",
        type=Path,
        default=DEFAULT_TEMPLATES_PATH,
        help="Text file with one synthesis template sentence per line (default: clean/experiment_templates.txt).",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name or local relative path.",
    )
    parser.add_argument("--max-threshold", type=float, default=0.5)
    parser.add_argument("--mean-threshold", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 1) // 2)))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    run(
        pipeline_root=args.pipeline_root,
        templates_path=args.templates_path,
        model_name_or_path=args.model_name_or_path,
        max_threshold=args.max_threshold,
        mean_threshold=args.mean_threshold,
        workers=args.workers,
        overwrite=args.overwrite,
    )
