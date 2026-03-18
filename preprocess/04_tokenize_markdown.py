#!/usr/bin/env python3
"""Step 04: Build section-aware sentence data from trimmed markdown."""

import argparse
import os
import re
from dataclasses import dataclass
from multiprocessing.dummy import Pool as ThreadPool
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

try:
    import spacy
except Exception:
    spacy = None


OUTPUT_SUBDIR = os.path.join("preprocess", "Tokenized")
ABBREV_MARK = "<DOT_PROTECTED>"

DEFAULT_MINING_ROOT = os.path.join("data", "mining")
DEFAULT_WORKERS = min(8, max(1, os.cpu_count() or 1))

HDR_RE = re.compile(r"^\s*(#{1,6})\s+(.*\S)\s*$")
IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")

_nlp = None


def _fold_spaced_caps(text: str) -> str:
    if re.fullmatch(r"(?:[A-Za-z]\s+){2,}[A-Za-z]", (text or "").strip()):
        return text.replace(" ", "")
    return text


def build_section_patterns() -> Dict[str, List[re.Pattern]]:
    to_rx = lambda xs: [re.compile(x, re.IGNORECASE) for x in xs]
    return {
        "abstract": to_rx([
            r"\babstract\b",
            r"\bgraphical\s+abstract\b",
        ]),
        "intro": to_rx([
            r"\bintroduction\b",
            r"\bbackground\b",
            r"\boverview\b",
            r"\bmotivation\b",
            r"\brelated\s+work\b",
            r"\bliterature\s+review\b",
        ]),
        "methods": to_rx([
            r"\bmaterials?\s*(?:and|&)\s*methods\b",
            r"\bexperimental(?:\s+sections?)?\b",
            r"\bmethods?\b",
            r"\bmethodology\b",
            r"\bsynthesis\b",
            r"\bpreparation\b",
            r"\bfabrication\b",
        ]),
        "results_discussion": to_rx([
            r"\bresults?\s*(?:and|&)\s*discussion\b",
            r"\bresults?\b",
            r"\bfindings?\b",
            r"\bperformance\b",
            r"\b(optical|photophysical|emission|luminescence|pl)\s+properties\b",
            r"\bdiscussion\b",
            r"\banalysis\b",
        ]),
        "conclusion": to_rx([
            r"\bconclusions?\b",
            r"\bsummary\b",
            r"\boutlook\b",
            r"\bperspectives?\b",
            r"\bfuture\s+work\b",
        ]),
        "ignore": to_rx([
            r"\barticle\s*info(?:rmation)?\b|\barticleinfo\b",
            r"\bkeywords?\b",
            r"\bhighlights?\b",
            r"\bcorrespondence\b",
            r"\bjournal\s+pre-?proofs?\b",
            r"\baccess(?:\s+metrics(?:\s+more)?)?\b",
            r"\bsupporting\s+information\b",
            r"\bsupplementary\s+(?:information|materials?)\b",
            r"\bappendix\b",
            r"\breferences?\b",
            r"\bbibliography\b",
            r"\backnowledg(?:e)?ments?\b",
            r"\bconflicts?\s+of\s+interest\b",
            r"\bcompeting\s+interests?\b",
            r"\bdata\s+availability\b",
        ]),
    }


SECTION_PATTERNS = build_section_patterns()


def normalize_section_from_header(header_text: str) -> str:
    text = _fold_spaced_caps((header_text or "").strip())
    if not text:
        return "unknown"

    matched = set()
    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                matched.add(section)
                break

    main_order = ["intro", "methods", "results_discussion", "conclusion"]
    for section in reversed(main_order):
        if section in matched:
            return section
    if "abstract" in matched:
        return "abstract"
    if "ignore" in matched:
        return "ignore"
    return "unknown"


@dataclass
class Block:
    block_id: int
    raw: str
    main_header_text: str
    main_section_norm: str
    main_header_line_idx: Optional[int]
    start_line_idx: int
    end_line_idx: int


def _init_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp

    if spacy is None:
        _nlp = False
        return _nlp

    for model_name in ("en_core_web_sm",):
        try:
            _nlp = spacy.load(
                model_name,
                disable=["ner", "tagger", "lemmatizer"],
            )
            break
        except Exception:
            _nlp = None

    if _nlp is None:
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


def markdown_to_blocks(md_text: str) -> List[Block]:
    lines = (md_text or "").splitlines()
    blocks: List[Block] = []
    buf: List[str] = []
    block_id = 0
    main_header_text = ""
    main_section_norm = "unknown"
    main_header_line_idx: Optional[int] = None
    start_for_buf = 0
    current_top_level_root = ""

    def extract_section_root(title_text: str) -> str:
        match = re.match(r"^\s*(\d+)(?:\.\d+)*\.?(?:\s|$)", title_text or "")
        return match.group(1) if match else ""

    def is_numeric_subsection(title_text: str) -> bool:
        return bool(re.match(r"^\s*\d+\.\d+\.?(?:\s|$)", title_text or ""))

    def flush_block(start_line: int, end_line: int):
        nonlocal buf, block_id
        raw = "\n".join(buf).strip()
        if not raw:
            buf = []
            return
        blocks.append(
            Block(
                block_id=block_id,
                raw=raw,
                main_header_text=main_header_text,
                main_section_norm=main_section_norm,
                main_header_line_idx=main_header_line_idx,
                start_line_idx=start_line,
                end_line_idx=end_line,
            )
        )
        block_id += 1
        buf = []

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = HDR_RE.match(line)
        if match:
            level = len(match.group(1))
            title_text = match.group(2).strip()
            title_root = extract_section_root(title_text)
            if level == 1 and is_numeric_subsection(title_text) and current_top_level_root and title_root == current_top_level_root:
                title_section = main_section_norm
            else:
                title_section = normalize_section_from_header(title_text)
            if level == 1:
                if buf:
                    flush_block(start_for_buf, idx - 1)
                main_header_text = title_text
                main_section_norm = title_section
                main_header_line_idx = idx
                if not is_numeric_subsection(title_text) and title_root:
                    current_top_level_root = title_root
                start_for_buf = idx
                buf.extend(["", title_text, ""])
                idx += 1
                continue

            if not buf:
                start_for_buf = idx
            buf.extend(["", title_text, ""])
            idx += 1
            continue

        if not buf:
            start_for_buf = idx
        buf.append(line)
        idx += 1

    if buf:
        flush_block(start_for_buf, len(lines) - 1)

    return blocks


def _has_strong_terminal(line: str) -> bool:
    tail = (line or "").rstrip()
    return bool(
        re.search(r"[.?!;]\s*$", tail)
        or re.search(r"[.?!;][)\]\}\"']+\s*$", tail)
    )


def _looks_like_caption(line: str) -> bool:
    text = (line or "").strip()
    if not text:
        return False
    low = text.lower()
    if low.startswith(("fig.", "figure", "scheme", "eq.")):
        return True
    if re.match(r"^\(?[a-z]\)", text):
        return True
    if low.startswith("(for interpretation"):
        return True
    return False


def _consume_figure_block(lines: List[str], start_idx: int, emitted_lines: List[str]):
    figure_parts: List[str] = []

    while emitted_lines and not emitted_lines[-1].strip():
        emitted_lines.pop()

    if emitted_lines:
        prev_line = emitted_lines[-1].rstrip()
        if prev_line and not _has_strong_terminal(prev_line):
            figure_parts.append(prev_line.strip())
            emitted_lines.pop()
            while emitted_lines and not emitted_lines[-1].strip():
                emitted_lines.pop()

    idx = start_idx
    while idx < len(lines):
        current_line = lines[idx]
        if HDR_RE.match(current_line):
            break

        if IMG_RE.search(current_line.strip()):
            figure_parts.append(current_line.strip())
            idx += 1
            while idx < len(lines):
                next_line = lines[idx]
                stripped = next_line.strip()
                if HDR_RE.match(next_line) or IMG_RE.search(stripped):
                    break
                if not stripped:
                    following = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
                    if not _looks_like_caption(following):
                        idx += 1
                        break
                    idx += 1
                    continue
                figure_parts.append(stripped)
                idx += 1
            continue
        break

    if idx < len(lines):
        tail = lines[idx].strip()
        if tail and not HDR_RE.match(lines[idx]) and not IMG_RE.search(tail):
            if len(tail.split()) <= 12:
                figure_parts.append(tail)
                idx += 1

    body = " \n".join(part for part in figure_parts if part)
    return f"<<<FIG_BLOCK_START>>>\n{body}\n<<<FIG_BLOCK_END>>>", idx


def _collapse_images_with_figure_block(text: str) -> str:
    lines = (text or "").splitlines()
    out_lines: List[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if IMG_RE.search(line.strip()):
            figure_block, next_idx = _consume_figure_block(lines, idx, out_lines)
            if out_lines and out_lines[-1].strip():
                out_lines.append("")
            out_lines.append(figure_block)
            out_lines.append("")
            idx = next_idx
            continue
        out_lines.append(line)
        idx += 1
    return "\n".join(out_lines)


def block_to_paragraphs(block: Block) -> List[str]:
    protected_text = _collapse_images_with_figure_block(block.raw)
    return [part.strip() for part in re.split(r"\n\s*\n+", protected_text) if part.strip()]


def _is_figure_block(paragraph: str) -> bool:
    return paragraph.startswith("<<<FIG_BLOCK_START>>>") and paragraph.endswith("<<<FIG_BLOCK_END>>>")


def _clean_paragraph_text(paragraph: str) -> str:
    return (
        (paragraph or "")
        .replace("<<<FIG_BLOCK_START>>>", "")
        .replace("<<<FIG_BLOCK_END>>>", "")
        .strip()
    )


def _protect_abbrevs(text: str) -> str:
    text = re.sub(r"\bFig\.\s*(\d+[A-Za-z]?)\.", rf"Fig{ABBREV_MARK}\1{ABBREV_MARK}", text)
    text = re.sub(r"\bFigure\s*(\d+[A-Za-z]?)\.", rf"Figure{ABBREV_MARK}\1{ABBREV_MARK}", text)
    text = re.sub(r"\bScheme\.?\s*(\d+[A-Za-z]?)\.", rf"Scheme{ABBREV_MARK}\1{ABBREV_MARK}", text)
    text = re.sub(r"\bEq\.\s*\((\d+[A-Za-z]?)\)", rf"Eq{ABBREV_MARK}(\1)", text)
    text = re.sub(r"\betc\.", f"etc{ABBREV_MARK}", text, flags=re.IGNORECASE)
    text = re.sub(r"\bet\s+al\.", f"et{ABBREV_MARK}al{ABBREV_MARK}", text, flags=re.IGNORECASE)
    return text


def _unprotect_abbrevs(text: str) -> str:
    marker = re.escape(ABBREV_MARK)
    text = re.sub(rf"Fig{marker}(\d+[A-Za-z]?){marker}", r"Fig. \1.", text)
    text = re.sub(rf"Figure{marker}(\d+[A-Za-z]?){marker}", r"Figure \1.", text)
    text = re.sub(rf"Scheme{marker}(\d+[A-Za-z]?){marker}", r"Scheme \1.", text)
    text = re.sub(rf"Eq{marker}\((\d+[A-Za-z]?)\)", r"Eq. (\1)", text)
    text = text.replace(f"etc{ABBREV_MARK}", "etc.")
    text = text.replace(f"et{ABBREV_MARK}al{ABBREV_MARK}", "et al.")
    return text


def _spacy_split_to_sents(text: str) -> List[str]:
    nlp = _init_nlp()
    if nlp is False:
        return [part.strip() for part in re.split(r"(?<=[.?!;])\s+(?=[A-Z\[(])", text) if part.strip()]
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _second_pass_boundary_cut(sentence: str) -> List[str]:
    parts: List[str] = []
    start = 0
    length = len(sentence)

    while start < length:
        idx = start
        cut_done = False
        while idx < length:
            if sentence[idx] in ".?!;":
                after_punct = idx + 1
                while after_punct < length and sentence[after_punct] in ")]}\"'":
                    after_punct += 1
                next_start = after_punct
                while next_start < length and sentence[next_start].isspace():
                    next_start += 1
                if next_start < length:
                    next_char = sentence[next_start]
                    if next_char.isupper() or next_char in "[(":
                        chunk = sentence[start:after_punct].strip()
                        if chunk:
                            parts.append(chunk)
                        start = next_start
                        cut_done = True
                        break
            idx += 1

        if not cut_done:
            chunk = sentence[start:].strip()
            if chunk:
                parts.append(chunk)
            break

    return parts


def _starts_with_citation(text: str) -> bool:
    stripped = text.lstrip()
    if stripped.startswith("[") and re.match(r"^\[\s*[\dA-Za-z]", stripped):
        return True
    if stripped.startswith("(") and re.match(r"^\(\s*[A-Za-z0-9]", stripped):
        return True
    return False


def _ends_with_strong_stop(text: str) -> bool:
    tail = text.rstrip()
    if re.search(r"[.?!;]\s*$", tail):
        return True
    if re.search(r"[.?!;][)\]\}\"']+\s*$", tail):
        return True
    return False


def _is_bracket_only(text: str) -> bool:
    return bool(re.match(r"^\s*[\[(][^\]\)]+[\]\)]\s*\.?\s*$", text))


def _is_short_tail(text: str, max_tokens: int = 4) -> bool:
    return len(re.findall(r"\S+", text.strip())) <= max_tokens


def _merge_sentences_in_paragraph(sentences: List[str]) -> List[str]:
    out: List[str] = []
    for sentence in sentences:
        cleaned = sentence.strip()
        if not cleaned:
            continue
        if out:
            if _is_bracket_only(cleaned):
                out[-1] = (out[-1].rstrip() + " " + cleaned).strip()
                continue
            if _is_short_tail(cleaned, max_tokens=4):
                out[-1] = (out[-1].rstrip() + " " + cleaned).strip()
                continue
            if _starts_with_citation(cleaned) and not _ends_with_strong_stop(out[-1]):
                out[-1] = (out[-1].rstrip() + " " + cleaned).strip()
                continue
        out.append(cleaned)
    return out


def split_paragraph_into_sentences(paragraph: str) -> List[str]:
    if _is_figure_block(paragraph):
        cleaned = _clean_paragraph_text(paragraph)
        return [cleaned] if cleaned else []

    text = _clean_paragraph_text(paragraph)
    if not text:
        return []

    protected = _protect_abbrevs(text)
    rough_sents = _spacy_split_to_sents(protected)
    refined: List[str] = []
    for rough in rough_sents:
        refined.extend(_second_pass_boundary_cut(rough))
    refined = [_unprotect_abbrevs(sentence).strip() for sentence in refined if sentence.strip()]
    return _merge_sentences_in_paragraph(refined)


def blocks_to_sentence_df(blocks: List[Block], pdf_name: str) -> pd.DataFrame:
    rows = []
    para_global_id = 0
    sent_global_id = 0

    for block in blocks:
        paragraphs = block_to_paragraphs(block)
        for para_id_in_block, paragraph in enumerate(paragraphs, start=1):
            para_global_id += 1
            is_figure = _is_figure_block(paragraph)
            sentences = split_paragraph_into_sentences(paragraph)
            for sent_id_in_para, sentence in enumerate(sentences, start=1):
                sent_global_id += 1
                rows.append(
                    {
                        "pdf_name": pdf_name,
                        "block_id": block.block_id,
                        "main_section_norm": block.main_section_norm,
                        "main_header_text": block.main_header_text or "",
                        "para_id_in_block": para_id_in_block,
                        "para_global_id": para_global_id,
                        "sent_id_in_para": sent_id_in_para,
                        "sent_global_id": sent_global_id,
                        "is_figure": bool(is_figure),
                        "text": sentence.strip(),
                    }
                )

    columns = [
        "pdf_name",
        "block_id",
        "main_section_norm",
        "main_header_text",
        "para_id_in_block",
        "para_global_id",
        "sent_id_in_para",
        "sent_global_id",
        "is_figure",
        "text",
    ]
    return pd.DataFrame(rows, columns=columns)


def save_sentence_df(df: pd.DataFrame, out_base: str) -> None:
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    df.to_csv(out_base + ".csv", index=False, encoding="utf-8-sig")

    with open(out_base + ".txt", "w", encoding="utf-8") as f_txt, open(
        out_base + ".md", "w", encoding="utf-8"
    ) as f_md:
        for _, row in df.iterrows():
            tag = (
                f"[Block {row['block_id']}, Sec: {row['main_section_norm']}, "
                f"Para {row['para_id_in_block']}, Sent {row['sent_id_in_para']}]"
            )
            body = str(row["text"]).strip()
            f_txt.write(f"{tag}:\n{body}\n\n")
            f_md.write(f"{tag}  \n\n{body}\n\n---\n\n")


def already_tokenized(output_dir: str, doc_id: str) -> bool:
    needed = [f"{doc_id}.csv", f"{doc_id}.txt", f"{doc_id}.md"]
    return all(os.path.exists(os.path.join(output_dir, name)) for name in needed)


def process_single_folder(subdir: str) -> str:
    folder_name = os.path.basename(subdir)
    doc_id = folder_name.split("_")[0]

    cut_md = os.path.join(subdir, "preprocess", "cut", f"{doc_id}_cut.md")
    output_dir = os.path.join(subdir, OUTPUT_SUBDIR)
    out_base = os.path.join(output_dir, doc_id)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(cut_md):
        return f"[Skip] {doc_id}: missing cut markdown"
    if already_tokenized(output_dir, doc_id):
        return f"[Skip] {doc_id}: already tokenized"

    with open(cut_md, "r", encoding="utf-8") as f:
        md_text = f.read()

    blocks = markdown_to_blocks(md_text)
    df = blocks_to_sentence_df(blocks, pdf_name=doc_id)
    save_sentence_df(df, out_base)

    section_dist = dict(df["main_section_norm"].value_counts()) if not df.empty else {}
    return f"[Done] {doc_id}: sentences={len(df)}, sections={section_dist}"


def tokenize_all(mining_root_path: str, workers: int = DEFAULT_WORKERS) -> None:
    subdirs = [
        os.path.join(mining_root_path, name)
        for name in os.listdir(mining_root_path)
        if os.path.isdir(os.path.join(mining_root_path, name)) and name.split("_")[0].isdigit()
    ]
    _init_nlp()

    if workers <= 1:
        for subdir in tqdm(subdirs, desc="Step3 tokenizing", dynamic_ncols=True):
            print(process_single_folder(subdir))
        return

    with ThreadPool(processes=workers) as pool:
        for message in tqdm(
            pool.imap_unordered(process_single_folder, subdirs),
            total=len(subdirs),
            desc=f"Step3 tokenizing x{workers}",
            dynamic_ncols=True,
        ):
            if message:
                print(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build section-aware sentence-level synthesis outputs from trimmed markdown."
    )
    parser.add_argument("--mining-root", default=os.environ.get("MINING_ROOT", DEFAULT_MINING_ROOT))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenize_all(args.mining_root, workers=args.workers)

