#!/usr/bin/env python3
"""Folder Step 01 / Pipeline Step 05: chunk structured sentence data for cosine filtering."""

import argparse
import os
import re
from multiprocessing.dummy import Pool as ThreadPool
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

try:
    import spacy
except Exception:
    spacy = None


DEFAULT_MAX_TOKENS = 150
INPUT_SUBDIR = os.path.join("preprocess", "Tokenized")
OUTPUT_SUBDIR = os.path.join("Synthesis", "cos_tokenized")

_nlp = None

DEFAULT_MINING_ROOT = os.path.join("data", "mining")
DEFAULT_WORKERS = min(8, max(1, os.cpu_count() or 1))


def _init_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp

    if spacy is None:
        _nlp = False
        return _nlp

    try:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
    except Exception:
        try:
            _nlp = spacy.blank("en")
        except Exception:
            _nlp = False
            return _nlp

    if _nlp is not False and "parser" not in _nlp.pipe_names and "senter" not in _nlp.pipe_names:
        if "sentencizer" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
    return _nlp


def _count_tokens(text: str) -> int:
    nlp = _init_nlp()
    if nlp is False:
        return len(re.findall(r"\S+", text or ""))
    return len(nlp.make_doc(text or ""))


def _join_ids(values: List[int]) -> str:
    ints = []
    for value in values:
        try:
            ints.append(int(value))
        except Exception:
            continue
    return ",".join(str(v) for v in sorted(set(ints)))


def _build_chunk(
    pdf_name: str,
    para_id: int,
    block_id: int,
    main_section_norm: str,
    main_header_text: str,
    texts: List[str],
    para_ids: List[int],
    sent_ids: List[int],
) -> Dict[str, object]:
    body = "\n\n".join(text for text in texts if text).strip()
    return {
        "pdf_name": pdf_name,
        "para_id": para_id,
        "token_count": _count_tokens(body),
        "text": body,
        "block_id": int(block_id),
        "main_section_norm": str(main_section_norm or ""),
        "main_header_text": str(main_header_text or ""),
        "source_para_global_ids": _join_ids(para_ids),
        "source_sent_global_ids": _join_ids(sent_ids),
    }


def _split_long_paragraph(paragraph: Dict[str, object], pdf_name: str, para_id_start: int, max_tokens: int):
    chunks = []
    current_texts: List[str] = []
    current_sent_ids: List[int] = []
    para_ids = [int(paragraph["para_global_id"])]
    next_para_id = para_id_start

    sentence_pairs = list(zip(paragraph["sentence_texts"], paragraph["sent_global_ids"]))

    for sent_text, sent_id in sentence_pairs:
        if not current_texts:
            current_texts = [sent_text]
            current_sent_ids = [int(sent_id)]
            continue

        trial = " ".join(current_texts + [sent_text]).strip()
        if _count_tokens(trial) <= max_tokens:
            current_texts.append(sent_text)
            current_sent_ids.append(int(sent_id))
        else:
            chunks.append(
                _build_chunk(
                    pdf_name=pdf_name,
                    para_id=next_para_id,
                    block_id=int(paragraph["block_id"]),
                    main_section_norm=str(paragraph["main_section_norm"]),
                    main_header_text=str(paragraph["main_header_text"]),
                    texts=[" ".join(current_texts).strip()],
                    para_ids=para_ids,
                    sent_ids=current_sent_ids,
                )
            )
            next_para_id += 1
            current_texts = [sent_text]
            current_sent_ids = [int(sent_id)]

    if current_texts:
        chunks.append(
            _build_chunk(
                pdf_name=pdf_name,
                para_id=next_para_id,
                block_id=int(paragraph["block_id"]),
                main_section_norm=str(paragraph["main_section_norm"]),
                main_header_text=str(paragraph["main_header_text"]),
                texts=[" ".join(current_texts).strip()],
                para_ids=para_ids,
                sent_ids=current_sent_ids,
            )
        )
        next_para_id += 1

    return chunks, next_para_id


def _paragraph_records_from_df(df: pd.DataFrame) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if df.empty:
        return records

    sorted_df = df.sort_values(["sent_global_id"]).reset_index(drop=True)
    for para_global_id, group in sorted_df.groupby("para_global_id", sort=False):
        group = group.sort_values("sent_global_id")
        first = group.iloc[0]
        sentence_texts = [str(x).strip() for x in group["text"].tolist() if str(x).strip()]
        paragraph_text = " ".join(sentence_texts).strip()
        records.append(
            {
                "pdf_name": str(first["pdf_name"]),
                "block_id": int(first["block_id"]),
                "main_section_norm": str(first.get("main_section_norm", "") or ""),
                "main_header_text": str(first.get("main_header_text", "") or ""),
                "para_global_id": int(para_global_id),
                "sent_global_ids": [int(x) for x in group["sent_global_id"].tolist()],
                "sentence_texts": sentence_texts,
                "text": paragraph_text,
            }
        )
    return records


def chunk_sentence_df(df: pd.DataFrame, max_tokens: int) -> pd.DataFrame:
    paragraphs = _paragraph_records_from_df(df)
    chunks: List[Dict[str, object]] = []
    next_para_id = 1

    buffer: List[Dict[str, object]] = []

    def flush_buffer():
        nonlocal buffer, next_para_id
        if not buffer:
            return
        chunks.append(
            _build_chunk(
                pdf_name=str(buffer[0]["pdf_name"]),
                para_id=next_para_id,
                block_id=int(buffer[0]["block_id"]),
                main_section_norm=str(buffer[0]["main_section_norm"]),
                main_header_text=str(buffer[0]["main_header_text"]),
                texts=[str(item["text"]) for item in buffer],
                para_ids=[int(item["para_global_id"]) for item in buffer],
                sent_ids=[sid for item in buffer for sid in item["sent_global_ids"]],
            )
        )
        next_para_id += 1
        buffer = []

    for paragraph in paragraphs:
        if buffer:
            same_block = int(buffer[0]["block_id"]) == int(paragraph["block_id"])
            same_section = str(buffer[0]["main_section_norm"]) == str(paragraph["main_section_norm"])
            if not (same_block and same_section):
                flush_buffer()

        para_text = str(paragraph["text"])
        para_tokens = _count_tokens(para_text)

        if para_tokens > max_tokens:
            flush_buffer()
            split_chunks, next_para_id = _split_long_paragraph(paragraph, str(paragraph["pdf_name"]), next_para_id, max_tokens)
            chunks.extend(split_chunks)
            continue

        if not buffer:
            buffer = [paragraph]
            continue

        trial_text = "\n\n".join([str(item["text"]) for item in buffer] + [para_text]).strip()
        if _count_tokens(trial_text) <= max_tokens:
            buffer.append(paragraph)
        else:
            flush_buffer()
            buffer = [paragraph]

    flush_buffer()

    columns = [
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
    return pd.DataFrame(chunks, columns=columns)


def save_chunk_df(df: pd.DataFrame, out_base: str) -> None:
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    df.to_csv(out_base + ".csv", index=False, encoding="utf-8-sig")

    with open(out_base + ".txt", "w", encoding="utf-8") as f_txt, open(
        out_base + ".md", "w", encoding="utf-8"
    ) as f_md:
        for _, row in df.iterrows():
            header = (
                f"[Para {row['para_id']}, Tokens: {row['token_count']}, "
                f"Block {row['block_id']}, Sec: {row['main_section_norm']}]"
            )
            source_para = row["source_para_global_ids"]
            source_sent = row["source_sent_global_ids"]
            body = str(row["text"]).strip()
            f_txt.write(
                f"{header}\n"
                f"[source_para_global_ids] {source_para}\n"
                f"[source_sent_global_ids] {source_sent}\n"
                f"{body}\n\n"
            )
            f_md.write(
                f"{header}  \n\n"
                f"**source_para_global_ids**: {source_para}  \n"
                f"**source_sent_global_ids**: {source_sent}\n\n"
                f"{body}\n\n---\n\n"
            )


def already_chunked(output_dir: str, doc_id: str) -> bool:
    needed = [f"{doc_id}.csv", f"{doc_id}.txt", f"{doc_id}.md"]
    return all(os.path.exists(os.path.join(output_dir, name)) for name in needed)


def process_single_folder(subdir: str, max_tokens: int) -> str:
    folder_name = os.path.basename(subdir)
    doc_id = folder_name.split("_")[0]
    input_csv = os.path.join(subdir, INPUT_SUBDIR, f"{doc_id}.csv")
    output_dir = os.path.join(subdir, OUTPUT_SUBDIR)
    out_base = os.path.join(output_dir, doc_id)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_csv):
        return f"[Skip] {doc_id}: missing preprocess/Tokenized csv"
    if already_chunked(output_dir, doc_id):
        return f"[Skip] {doc_id}: already chunked for cosine"

    df = pd.read_csv(input_csv, encoding="utf-8-sig", keep_default_na=False)
    required = {
        "pdf_name",
        "block_id",
        "main_section_norm",
        "main_header_text",
        "para_global_id",
        "sent_global_id",
        "text",
    }
    missing = required - set(df.columns)
    if missing:
        return f"[Error] {doc_id}: missing columns {sorted(missing)}"

    chunk_df = chunk_sentence_df(df, max_tokens=max_tokens)
    save_chunk_df(chunk_df, out_base)
    return f"[Done] {doc_id}: cosine chunks={len(chunk_df)}, max_tokens={max_tokens}"


def chunk_all(mining_root_path: str, max_tokens: int, workers: int) -> None:
    subdirs = [
        os.path.join(mining_root_path, name)
        for name in os.listdir(mining_root_path)
        if os.path.isdir(os.path.join(mining_root_path, name)) and name.split("_")[0].isdigit()
    ]
    _init_nlp()

    if workers <= 1:
        for subdir in tqdm(subdirs, desc="Synthesis Step 01 cosine chunking", dynamic_ncols=True):
            print(process_single_folder(subdir, max_tokens=max_tokens))
        return

    with ThreadPool(processes=workers) as pool:
        args = ((subdir, max_tokens) for subdir in subdirs)
        for message in tqdm(
            pool.starmap(process_single_folder, args),
            total=len(subdirs),
            desc=f"Synthesis Step 01 cosine chunking x{workers}",
            dynamic_ncols=True,
        ):
            if message:
                print(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk structured synthesis sentence data into cosine-ready text units."
    )
    parser.add_argument("--mining-root", default=os.environ.get("MINING_ROOT", DEFAULT_MINING_ROOT))
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chunk_all(args.mining_root, max_tokens=args.max_tokens, workers=args.workers)

