#!/usr/bin/env python3
"""Folder Step 02 / Pipeline Step 06: score cosine similarity and apply a recall patch."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from synthesis_unit import display_path, extract_document_id, list_document_dirs


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MINING_ROOT = Path("data") / "mining"
DEFAULT_TEMPLATES_PATH = Path("synthesis_mining") / "experiment_templates.txt"
INPUT_SUBDIR = Path("Synthesis") / "cos_tokenized"
OUTPUT_SUBDIR = Path("Synthesis") / "cos"

_WORKER_MODEL = None
_WORKER_TEMPLATE_EMB = None


ANCHOR_PAT = re.compile(
    r"\b(?:synthesis|synthesi[sz]ed|preparation|prepared|fabrication|fabricated)\b"
    r"|(?:experimental(?:\s+sections?)?)"
    r"|(?:materials?\s*(?:and|&)\s*methods?)"
    r"|(?:methodology)",
    re.I,
)
METHOD_CONTEXT_PAT = re.compile(
    r"\b(?:dissolved?|mixed|stirred?|heated?|hydrothermal|solvothermal|autoclave|microwave|"
    r"calcined?|carboni[sz]ed|pyrolysis|annealed?|reflux(?:ed)?|centrifuged?|dialy[sz]ed|"
    r"filtered?|filtration|washed?|sonicat(?:ed|ion)|dropwise|react(?:ed|ion)|solution|"
    r"precursor|ethanol|water|dmf|naoh|hcl|obtained?|yielded?|dispersed?|evaporated?|dried?)\b",
    re.I,
)
CONDITION_UNIT_PAT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:C|K|h|hr|hrs|hour|hours|min|mins|minutes|"
    r"mL|uL|ul|L|g|mg|kg|M|mM|wt%|mol%|rpm|W)\b",
    re.I,
)
METHODS_HEADER_PAT = re.compile(
    r"\b(?:experimental(?:\s+sections?)?|materials?\s*(?:and|&)\s*methods?|"
    r"methods?|methodology|synthesis|preparation|fabrication)\b",
    re.I,
)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _has_anchor_keyword(text: str) -> bool:
    return bool(ANCHOR_PAT.search(_normalize_space(text)))


def _has_method_context(text: str) -> bool:
    norm = _normalize_space(text)
    return bool(METHOD_CONTEXT_PAT.search(norm) or CONDITION_UNIT_PAT.search(norm))


def _header_looks_methods(text: str) -> bool:
    return bool(METHODS_HEADER_PAT.search(_normalize_space(text)))


def _append_reason(old_reason: str, new_reason: str) -> str:
    if not old_reason:
        return new_reason
    parts = [part.strip() for part in old_reason.split(";") if part.strip()]
    if new_reason not in parts:
        parts.append(new_reason)
    return "; ".join(parts)


def apply_recall_patch(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        for column in ["retain_cos", "recall_patch", "recall_patch_reason", "has_methods_section"]:
            if column not in out.columns:
                out[column] = []
        return out

    work = df.copy()
    if "para_id" in work.columns:
        work = work.sort_values("para_id").reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    retain_cos = work["retain"].astype(str).fillna("NO").tolist() if "retain" in work.columns else ["NO"] * len(work)
    texts = work["text"].astype(str).tolist()
    anchor_flags = [_has_anchor_keyword(text) for text in texts]
    context_flags = [_has_method_context(text) for text in texts]

    methods_flags: list[bool] = []
    for _, row in work.iterrows():
        sec = str(row.get("main_section_norm", "") or "").strip().lower()
        header = str(row.get("main_header_text", "") or "")
        methods_flags.append(sec == "methods" or _header_looks_methods(header))

    has_methods_section = any(methods_flags)
    relaxed_zone = [False] * len(work)
    if has_methods_section:
        for idx in range(len(work)):
            relaxed_zone[idx] = methods_flags[idx]
            if idx > 0 and methods_flags[idx - 1]:
                relaxed_zone[idx] = True
            if idx + 1 < len(work) and methods_flags[idx + 1]:
                relaxed_zone[idx] = True

    patch_mask = [False] * len(work)
    patch_reasons = [""] * len(work)

    def mark(idx: int, reason: str) -> None:
        if idx < 0 or idx >= len(work):
            return
        if retain_cos[idx] == "YES":
            return
        patch_mask[idx] = True
        patch_reasons[idx] = _append_reason(patch_reasons[idx], reason)

    for idx in range(len(work)):
        if has_methods_section:
            if relaxed_zone[idx] and anchor_flags[idx]:
                mark(idx, "anchor_in_methods_zone")
                continue
            if anchor_flags[idx] and context_flags[idx]:
                mark(idx, "keyword+context")
            continue

        if anchor_flags[idx] and context_flags[idx]:
            mark(idx, "keyword+context")

    work["retain_cos"] = retain_cos
    work["recall_patch"] = np.where(patch_mask, "YES", "NO")
    work["recall_patch_reason"] = patch_reasons
    work["has_methods_section"] = "YES" if has_methods_section else "NO"
    work["retain"] = np.where(
        (work["retain_cos"] == "YES") | (work["recall_patch"] == "YES"),
        "YES",
        "NO",
    )
    return work


def load_templates_from_file(file_path: Path) -> list[str]:
    with file_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _worker_init(model_name_or_path: str, templates_list: list[str]) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    try:
        import torch

        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.set_num_threads(1)
    except Exception:
        pass

    from sentence_transformers import SentenceTransformer

    global _WORKER_MODEL, _WORKER_TEMPLATE_EMB
    _WORKER_MODEL = SentenceTransformer(model_name_or_path, device="cpu")

    emb = _WORKER_MODEL.encode(templates_list, batch_size=32, convert_to_tensor=False)
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    _WORKER_TEMPLATE_EMB = emb / np.clip(norms, 1e-12, None)


def load_cos_tokenized_csv(csv_path: Path, document_id: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False)
    required = {"para_id", "token_count", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{document_id}: missing required columns {sorted(missing)}")

    if "pdf_name" not in df.columns:
        df["pdf_name"] = document_id

    for column in [
        "block_id",
        "main_section_norm",
        "main_header_text",
        "source_para_global_ids",
        "source_sent_global_ids",
    ]:
        if column not in df.columns:
            df[column] = ""

    df = df.sort_values("para_id").reset_index(drop=True)
    return df[
        [
            "pdf_name",
            "para_id",
            "token_count",
            "text",
            "block_id",
            "main_section_norm",
            "main_header_text",
            "source_para_global_ids",
            "source_sent_global_ids",
        ]
    ].copy()


def already_processed(cos_dir: Path, document_id: str) -> bool:
    return all((cos_dir / f"{document_id}.{ext}").exists() for ext in ("csv", "txt", "md"))


def score_similarity_worker(df: pd.DataFrame, max_threshold: float, mean_threshold: float) -> pd.DataFrame:
    global _WORKER_MODEL, _WORKER_TEMPLATE_EMB
    if _WORKER_MODEL is None or _WORKER_TEMPLATE_EMB is None:
        raise RuntimeError("Worker model is not initialized.")

    texts = df["text"].astype(str).tolist()
    text_embs = _WORKER_MODEL.encode(texts, batch_size=32, convert_to_tensor=False)
    text_embs = np.asarray(text_embs, dtype=np.float32)
    text_embs = text_embs / np.clip(np.linalg.norm(text_embs, axis=1, keepdims=True), 1e-12, None)

    sims_matrix = cosine_similarity(text_embs, _WORKER_TEMPLATE_EMB)
    max_sim = sims_matrix.max(axis=1)
    mean_sim = sims_matrix.mean(axis=1)

    scored = df.copy()
    scored["max_similarity"] = max_sim
    scored["mean_similarity"] = mean_sim
    scored["retain"] = np.where((max_sim > max_threshold) | (mean_sim > mean_threshold), "YES", "NO")
    return apply_recall_patch(scored)


def process_document(args: tuple[Path, float, float, bool]) -> str:
    document_dir, max_threshold, mean_threshold, overwrite = args
    document_dir = Path(document_dir)
    document_id = extract_document_id(document_dir)

    input_csv = document_dir / INPUT_SUBDIR / f"{document_id}.csv"
    if not input_csv.exists():
        return f"Skip {document_id}: missing cosine tokenized csv ({display_path(input_csv)})"

    output_dir = document_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / f"{document_id}.csv"
    out_txt = output_dir / f"{document_id}.txt"
    out_md = output_dir / f"{document_id}.md"
    if already_processed(output_dir, document_id) and not overwrite:
        return f"Skip {document_id}: already processed"

    df = load_cos_tokenized_csv(input_csv, document_id)
    if df.empty:
        return f"Skip {document_id}: empty cosine tokenized content"

    scored = score_similarity_worker(df, max_threshold=max_threshold, mean_threshold=mean_threshold)
    scored.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with out_txt.open("w", encoding="utf-8") as f_txt, out_md.open("w", encoding="utf-8") as f_md:
        for _, row in scored.iterrows():
            header = f"[Para {row['para_id']}, Tokens: {row['token_count']}]"
            meta = (
                f"block={row.get('block_id', '')}, "
                f"sec={row.get('main_section_norm', '')}, "
                f"has_methods_section={row.get('has_methods_section', 'NO')}"
            )
            body = str(row["text"]).strip()
            f_txt.write(
                f"{header} ({meta}, max_sim={row['max_similarity']:.4f}, "
                f"mean_sim={row['mean_similarity']:.4f}, retain_cos={row['retain_cos']}, "
                f"recall_patch={row['recall_patch']}, retain={row['retain']})\n"
                f"[patch_reason] {row.get('recall_patch_reason', '')}\n"
                f"[source_para_global_ids] {row.get('source_para_global_ids', '')}\n"
                f"[source_sent_global_ids] {row.get('source_sent_global_ids', '')}\n"
                f"{body}\n\n"
            )
            f_md.write(
                f"{header}  \n\n"
                f"**block_id**: {row.get('block_id', '')} | "
                f"**section**: {row.get('main_section_norm', '')} | "
                f"**has_methods_section**: {row.get('has_methods_section', 'NO')} | "
                f"**max_sim**: {row['max_similarity']:.4f} | "
                f"**mean_sim**: {row['mean_similarity']:.4f} | "
                f"**retain_cos**: {row['retain_cos']} | "
                f"**recall_patch**: {row['recall_patch']} | "
                f"**patch_reason**: {row.get('recall_patch_reason', '')} | "
                f"**retain**: {row['retain']}\n\n"
                f"**source_para_global_ids**: {row.get('source_para_global_ids', '')}  \n"
                f"**source_sent_global_ids**: {row.get('source_sent_global_ids', '')}\n\n"
                f"{body}\n\n---\n---\n\n"
            )

    cos_yes_count = int((scored["retain_cos"] == "YES").sum())
    patch_yes_count = int((scored["recall_patch"] == "YES").sum())
    final_yes_count = int((scored["retain"] == "YES").sum())
    return (
        f"[Done] {document_id}: chunks={len(scored)}, "
        f"cos_yes={cos_yes_count}, patch_yes={patch_yes_count}, final_yes={final_yes_count}"
    )


def run(
    pipeline_root: Path,
    templates_path: Path,
    model_name_or_path: str,
    max_threshold: float,
    mean_threshold: float,
    workers: int,
    overwrite: bool,
) -> None:
    pipeline_root = Path(pipeline_root)
    templates_path = Path(templates_path)
    if not templates_path.exists():
        raise FileNotFoundError(
            f"Synthesis template file not found: {display_path(templates_path)}. "
            f"Expected default path: {display_path(DEFAULT_TEMPLATES_PATH)}."
        )

    print(f"Using synthesis templates file: {display_path(templates_path)}")
    templates = load_templates_from_file(templates_path)
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
    parser = argparse.ArgumentParser(description="Score cosine similarity for synthesis chunks.")
    parser.add_argument("--mining-root", dest="pipeline_root", type=Path, default=DEFAULT_MINING_ROOT)
    parser.add_argument(
        "--templates-path",
        type=Path,
        default=DEFAULT_TEMPLATES_PATH,
        help="Text file with one synthesis template sentence per line.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=DEFAULT_MODEL_NAME,
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
