#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step 1: tokenize property text from cut markdown into block- and sentence-level outputs."""

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from property_unit import ensure_root_exists, iter_paper_dirs, paper_id_from_dir


def _load_nlp():
    try:
        import spacy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Step1 requires spaCy. Install `spacy` and an English model such as "
            "`en_core_web_sm` or `en_core_sci_md` before running Step1."
        ) from exc

    try:
        nlp = spacy.load("en_core_sci_md")
        print("[INFO] Step1 using en_core_sci_md.")
        return nlp
    except Exception:
        nlp = spacy.load("en_core_web_sm")
        print("[INFO] Step1 fallback to en_core_web_sm.")
        return nlp


NLP = None


def get_nlp():
    global NLP
    if NLP is None:
        NLP = _load_nlp()
    return NLP


def _fold_spaced_caps(text: str) -> str:
    if re.fullmatch(r"(?:[A-Za-z]\s+){2,}[A-Za-z]", text.strip()):
        return text.replace(" ", "")
    return text


def build_section_patterns() -> Dict[str, List[re.Pattern]]:
    def compile_many(items: List[str]) -> List[re.Pattern]:
        return [re.compile(item, re.IGNORECASE) for item in items]

    return {
        "abstract": compile_many([r"\babstract\b", r"\bgraphical\s+abstract\b"]),
        "intro": compile_many(
            [
                r"\bintroduction\b",
                r"\bbackground\b",
                r"\boverview\b",
                r"\bmotivation\b",
                r"\brelated\s+work\b",
                r"\bliterature\s+review\b",
            ]
        ),
        "methods": compile_many(
            [
                r"\bmaterials?\s*(and|&)\s*methods\b",
                r"\bexperimental(\s+section)?\b",
                r"\bmethods?\b",
                r"\bmethodology\b",
                r"\bsynthesis\b",
                r"\bpreparation\b",
                r"\bfabrication\b",
                r"\bcharacterization\b",
                r"\bmaterials?\b",
                r"\bapparatus\b",
                r"\binstrumentation\b",
                r"\bchemicals?\b",
                r"\breagents?\b",
                r"\binstruments?\b",
            ]
        ),
        "results_discussion": compile_many(
            [
                r"\bresults?\s*(and|&)\s*discussion\b",
                r"\bresults?\b",
                r"\bfindings?\b",
                r"\bperformance\b",
                r"\b(optical|photophysical|emission|luminescence|pl)\s+properties\b",
                r"\bdiscussion\b",
                r"\banalysis\b",
            ]
        ),
        "conclusion": compile_many(
            [
                r"\bconclusions?\b",
                r"\bsummary\b",
                r"\boutlook\b",
                r"\bperspectives?\b",
                r"\bfuture\s+work\b",
            ]
        ),
        "ignore": compile_many(
            [
                r"\barticle\s*info(?:rmation)?\b|\barticleinfo\b",
                r"\bkeywords?\b",
                r"\bhighlights?\b",
                r"\bcorrespondence\b",
                r"\bjournal\s+pre-?proofs?\b",
                r"\baccess(\s+metrics(\s+more)?)?\b",
                r"\bsupporting\s+information\b",
                r"\bsupplementary\s+(information|materials?)\b",
                r"\bappendix\b",
                r"\breferences?\b",
                r"\bbibliography\b",
                r"\backnowledg(e)?ments?\b",
                r"\bconflicts?\s+of\s+interest\b",
                r"\bcompeting\s+interests?\b",
                r"\bdata\s+availability\b",
            ]
        ),
    }


SECTION_PATTERNS = build_section_patterns()
HEADER_RE = re.compile(r"^\s*(#{1,6})\s+(.*\S)\s*$")
IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
TABLE_BLOCK_RE = re.compile(r"<table\b.*?>.*?</table>", re.IGNORECASE | re.DOTALL)
CITE_RE = re.compile(
    r"\[(?:\s*\d+[A-Za-z]?(?:\s*[,;–-]\s*\d+[A-Za-z]?)*\s*)\]"
)
FIGREF_RE = re.compile(
    r"\bFig\.\s+((?:S\d+[A-Za-z]?|[0-9]+[A-Za-z]?))(?=[,.;\s\)])",
    re.IGNORECASE,
)
BOUNDARY_CANDIDATE_RE = re.compile(r"([.!?;][)\]\}»\"’']*)(\s+)(?=[A-Z\[])")


@dataclass
class Block:
    block_id: int
    raw: str
    main_header_text: Optional[str]
    main_section_norm: str
    main_header_line_idx: Optional[int]
    start_line_idx: int
    end_line_idx: int


def normalize_section_from_header(header_text: str) -> str:
    header = (header_text or "").strip()
    if not header:
        return "unknown"
    header = _fold_spaced_caps(header)

    matched = set()
    for section, patterns in SECTION_PATTERNS.items():
        if any(pattern.search(header) for pattern in patterns):
            matched.add(section)

    for section in reversed(["intro", "methods", "results_discussion", "conclusion"]):
        if section in matched:
            return section
    if "abstract" in matched:
        return "abstract"
    if "ignore" in matched:
        return "ignore"
    return "unknown"


def markdown_to_blocks(md_text: str, verbose: bool = False) -> List[Block]:
    lines = md_text.splitlines()
    blocks: List[Block] = []
    buf: List[str] = []
    block_id = 0
    main_header_text: Optional[str] = None
    main_section_norm = "unknown"
    main_header_line_idx: Optional[int] = None

    def flush(start_line: int, end_line: int) -> None:
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

    i = 0
    start_for_buf = 0
    while i < len(lines):
        line = lines[i]
        match = HEADER_RE.match(line)
        if match:
            level = len(match.group(1))
            title_text = match.group(2).strip()
            section = normalize_section_from_header(title_text)
            if level == 1:
                if buf:
                    flush(start_for_buf, i - 1)
                main_header_text = title_text
                main_section_norm = section
                main_header_line_idx = i
                start_for_buf = i
                buf.extend(["", title_text, ""])
                if verbose:
                    print(f"[H1] {title_text} -> {main_section_norm}")
                i += 1
                continue
            if not buf:
                start_for_buf = i
            buf.extend(["", title_text, ""])
            i += 1
            continue

        if not buf:
            start_for_buf = i
        buf.append(line)
        i += 1

    if buf:
        flush(start_for_buf, len(lines) - 1)

    if verbose:
        dist = Counter(block.main_section_norm for block in blocks)
        print(f"[CHECK-BLOCKS] blocks={len(blocks)} dist={dict(dist)}")
    return blocks


def trim_blocks_before_first_h1(blocks: List[Block]) -> List[Block]:
    first_idx = None
    for idx, block in enumerate(blocks):
        if block.main_header_text and str(block.main_header_text).strip():
            first_idx = idx
            break
    if first_idx is None:
        return blocks
    trimmed = blocks[first_idx:]
    for new_id, block in enumerate(trimmed):
        block.block_id = new_id
    return trimmed


def stabilize_block_sections(blocks: List[Block], verbose: bool = False) -> None:
    """
    仅执行区间规则（无全局单调）：
    Step-2:
      - intro→methods：区间内 unknown → intro
      - methods→results_discussion：区间内 全部标记（含 ignore，除 abstract）→ methods
    Step-3:
      - results_discussion→conclusion：区间内 全部标记（含 ignore，除 abstract）→ results_discussion
    Step-4:
      - 对任意正向跨级 l1→l2（ORDER[l1]+1 < ORDER[l2]），且“被跳过阶段在全文完全不存在”时，
        仅把区间内 unknown → l2（其他不动）
    Step-5:
      - 若全文所有 block 都是 unknown，则统一改为 intro
    """
    main5 = {"abstract", "intro", "methods", "results_discussion", "conclusion"}
    order = {"abstract": 0, "intro": 1, "methods": 2, "results_discussion": 3, "conclusion": 4}

    def build_anchors():
        anchors = []
        last = None
        for idx, block in enumerate(blocks):
            label = block.main_section_norm
            if label in main5 and label != last:
                anchors.append((idx, label))
                last = label
        return anchors

    def reassign_range(lo: int, hi: int, target: str, *, unknown_only: bool, include_ignore: bool) -> int:
        """
        改写开区间 (lo, hi) 内的块为 target。
        - unknown_only=True 仅改 unknown；否则改全部允许的标签
        - include_ignore=True 允许改写 ignore；abstract 永远不改
        """
        changed = 0
        for idx in range(lo + 1, hi):
            label = blocks[idx].main_section_norm
            if label == "abstract":
                continue
            if not include_ignore and label == "ignore":
                continue
            if unknown_only:
                if label == "unknown":
                    blocks[idx].main_section_norm = target
                    changed += 1
            else:
                if label != target:
                    blocks[idx].main_section_norm = target
                    changed += 1
        return changed

    # -------- Step-2：邻级稳定化（两段） --------
    # 2a) intro → methods：unknown → intro
    anchors = build_anchors()
    if anchors:
        for idx in range(len(anchors) - 1):
            left_idx, left_label = anchors[idx]
            right_idx, right_label = anchors[idx + 1]
            if left_label == "intro" and right_label == "methods":
                reassign_range(left_idx, right_idx, target="intro", unknown_only=True, include_ignore=False)

    # 2b) methods → results_discussion：全部（含 ignore，除 abstract）→ methods
    anchors = build_anchors()
    if anchors:
        for idx in range(len(anchors) - 1):
            left_idx, left_label = anchors[idx]
            right_idx, right_label = anchors[idx + 1]
            if left_label == "methods" and right_label == "results_discussion":
                reassign_range(left_idx, right_idx, target="methods", unknown_only=False, include_ignore=True)

    # -------- Step-3：results_discussion → conclusion：全部（含 ignore，除 abstract）→ results_discussion --------
    anchors = build_anchors()
    if anchors:
        for result_pos, (result_idx, result_label) in enumerate(anchors):
            if result_label != "results_discussion":
                continue
            conclusion_pos = None
            for next_pos in range(result_pos + 1, len(anchors)):
                if anchors[next_pos][1] == "conclusion":
                    conclusion_pos = next_pos
                    break
            if conclusion_pos is None:
                continue
            conclusion_idx, _ = anchors[conclusion_pos]
            reassign_range(result_idx, conclusion_idx, target="results_discussion", unknown_only=False, include_ignore=True)

    # -------- Step-4：跨级修补（仅 unknown，且“被跳过阶段全篇缺失”才触发）--------
    anchors = build_anchors()
    present = {block.main_section_norm for block in blocks if block.main_section_norm in main5}
    if anchors:
        order_seq = ["intro", "methods", "results_discussion", "conclusion"]
        for i in range(len(anchors) - 1):
            left_idx, left_label = anchors[i]
            for j in range(i + 1, len(anchors)):
                right_idx, right_label = anchors[j]
                if order[left_label] + 1 < order[right_label]:
                    skipped = [sec for sec in order_seq if order[left_label] < order[sec] < order[right_label]]
                    if all(section not in present for section in skipped):
                        reassign_range(left_idx, right_idx, target=right_label, unknown_only=True, include_ignore=False)

    # -------- Step-5：全文兜底 --------
    if blocks and all(block.main_section_norm == "unknown" for block in blocks):
        for block in blocks:
            block.main_section_norm = "intro"
        if verbose:
            print("[STABILIZE] all blocks were unknown -> reassigned all to intro")

    if verbose:
        sample = [
            (block.block_id, block.main_section_norm, block.main_header_text)
            for block in blocks
            if block.main_section_norm in main5 or block.main_section_norm == "unknown"
        ]
        print(f"[STABILIZE] done. sample={sample[:10]}")


def _doc_total_lines_from_blocks(blocks: List[Block]) -> int:
    return max((block.end_line_idx for block in blocks), default=-1) + 1


def _block_mid_quantile(block: Block, total_lines: int) -> float:
    if total_lines <= 1:
        return 0.0
    mid = 0.5 * (block.start_line_idx + block.end_line_idx)
    return max(0.0, min(1.0, mid / (total_lines - 1)))


def report_unknown_ratio(blocks: List[Block], pdf_name: str, log_path: str) -> None:
    main_sections = {"abstract", "intro", "methods", "results_discussion", "conclusion"}
    evaluable_sections = main_sections | {"unknown"}
    any_h1 = any(block.main_header_text and str(block.main_header_text).strip() for block in blocks)
    no_h1_flag = not any_h1

    if not blocks:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{pdf_name}\tblocks=0\tevaluable=0\tunknown_ratio=NA\tno_h1={no_h1_flag}\tH1_titles=[]\n")
        return

    total_lines = _doc_total_lines_from_blocks(blocks)
    candidates = [
        block
        for block in blocks
        if block.main_header_text
        and str(block.main_header_text).strip()
        and block.main_section_norm in evaluable_sections
    ]
    if not candidates:
        h1_titles = [block.main_header_text or "" for block in blocks if block.main_header_text][:5]
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(
                f"{pdf_name}\tblocks={len(blocks)}\tevaluable=0\tunknown_ratio=NA\t"
                f"no_h1={no_h1_flag}\tH1_titles={h1_titles}\n"
            )
        return

    present_main = {block.main_section_norm for block in candidates if block.main_section_norm in main_sections}

    def first_quantile(section: str) -> Optional[float]:
        values = [
            _block_mid_quantile(block, total_lines)
            for block in candidates
            if block.main_section_norm == section
        ]
        return min(values) if values else None

    first_q = {
        "intro": first_quantile("intro"),
        "methods": first_quantile("methods"),
        "results_discussion": first_quantile("results_discussion"),
        "conclusion": first_quantile("conclusion"),
    }

    def order_ok(margin: float = 0.10) -> bool:
        pairs = [
            ("intro", "methods"),
            ("methods", "results_discussion"),
            ("results_discussion", "conclusion"),
        ]
        for left, right in pairs:
            q_left = first_q[left]
            q_right = first_q[right]
            if q_left is None or q_right is None:
                continue
            if q_left - q_right > margin:
                return False
        return True

    if len(present_main) >= 4 or (len(present_main) >= 3 and order_ok()):
        return

    mid_unknown = 0
    mid_total = 0
    overall_unknown = 0
    for block in candidates:
        q = _block_mid_quantile(block, total_lines)
        zone = "head" if q <= 0.10 else "tail" if q >= 0.90 else "mid"
        if zone == "mid":
            mid_total += 1
        if block.main_section_norm == "unknown":
            overall_unknown += 1
            if zone == "mid":
                mid_unknown += 1

    mid_unknown_ratio = (mid_unknown / mid_total) if mid_total else 0.0
    overall_unknown_ratio = overall_unknown / len(candidates)
    need_log = (mid_total > 0 and mid_unknown_ratio > 0.50) or (
        overall_unknown_ratio > 0.70 and len(present_main) <= 2
    )
    if not need_log:
        return

    h1_titles = [block.main_header_text or "" for block in blocks if block.main_header_text][:5]
    ignore_cnt = sum(1 for block in blocks if block.main_section_norm == "ignore")
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(
            f"{pdf_name}\tblocks={len(blocks)}\tevaluable={len(candidates)}\tignore={ignore_cnt}\t"
            f"mid_unknown_ratio={mid_unknown_ratio:.2f}\toverall_unknown_ratio={overall_unknown_ratio:.2f}\t"
            f"present_main={sorted(present_main)}\tfirst_q={first_q}\tno_h1={no_h1_flag}\tH1_titles={h1_titles}\n"
        )


def build_paragraph_df(blocks: List[Block], pdf_name: str) -> pd.DataFrame:
    rows = [
        {
            "pdf_name": pdf_name,
            "block_id": block.block_id,
            "main_section_norm": block.main_section_norm,
            "main_header_text": block.main_header_text or "",
            "text": (block.raw or "").strip(),
        }
        for block in blocks
    ]
    return pd.DataFrame(
        rows,
        columns=["pdf_name", "block_id", "main_section_norm", "main_header_text", "text"],
    )


def save_paragraph_df(df: pd.DataFrame, out_base: str) -> None:
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    cols = ["pdf_name", "block_id", "main_section_norm", "main_header_text", "text"]
    df.to_csv(out_base + ".csv", index=False, encoding="utf-8-sig", columns=cols)
    with open(out_base + ".txt", "w", encoding="utf-8") as f_txt, open(
        out_base + ".md", "w", encoding="utf-8"
    ) as f_md:
        for _, row in df.iterrows():
            tag = f"[Block {row['block_id']}, Sec: {row['main_section_norm']}]"
            text = str(row["text"] or "")
            f_txt.write(f"{tag}:\n{text}\n\n")
            f_md.write(f"{tag}  \n\n{text}\n\n---\n\n")


def _protect_citations(text: str) -> str:
    return CITE_RE.sub(lambda m: f"CITEREF§{m.group(0)[1:-1]}§", text)


def _unprotect_citations(text: str) -> str:
    return re.sub(r"CITEREF§(.*?)§", r"[\1]", text)


def _protect_figrefs(text: str) -> str:
    return FIGREF_RE.sub(lambda m: f"FIGREF§{m.group(1)}§", text)


def _unprotect_figrefs(text: str) -> str:
    return re.sub(r"FIGREF§(.*?)§", r"Fig. \1", text)


def _resplit_boundary(sent: str) -> List[str]:
    parts: List[str] = []
    start_idx = 0
    for match in BOUNDARY_CANDIDATE_RE.finditer(sent):
        left_chunk = sent[start_idx : match.end(1)].strip()
        next_char_idx = match.end()
        skip_split = False
        if "FIGREF§" in left_chunk:
            after_ws = sent[next_char_idx:].lstrip()
            if after_ws.startswith("S"):
                skip_split = True
        if skip_split:
            continue
        if left_chunk:
            parts.append(left_chunk)
        start_idx = match.end()
    tail = sent[start_idx:].strip()
    if tail:
        parts.append(tail)
    return parts


def _block_to_segments_with_figures(block_text: str) -> List[Dict]:
    tables: List[str] = []

    def table_repl(match: re.Match) -> str:
        tables.append(match.group(0))
        return f"\nTABLEBLOCK§{len(tables) - 1}§\n"

    text = TABLE_BLOCK_RE.sub(table_repl, block_text)
    lines = text.splitlines()
    segments: List[Dict] = []
    current_lines: List[str] = []
    base_idx = 1
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("TABLEBLOCK§") and stripped.endswith("§"):
            if current_lines:
                joined = " ".join(x.strip() for x in current_lines if x.strip()).strip()
                if joined:
                    segments.append({"base_idx": base_idx, "is_figure": False, "text": joined})
                    base_idx += 1
                current_lines = []
            match = re.match(r"^TABLEBLOCK§(\d+)§$", stripped)
            if match:
                idx = int(match.group(1))
                if 0 <= idx < len(tables):
                    table_html = tables[idx].strip()
                    if table_html:
                        segments.append({"base_idx": base_idx, "is_figure": True, "text": table_html})
                        base_idx += 1
            i += 1
            continue

        if IMG_RE.search(stripped):
            if current_lines:
                joined = " ".join(x.strip() for x in current_lines if x.strip()).strip()
                if joined:
                    segments.append({"base_idx": base_idx, "is_figure": False, "text": joined})
                    base_idx += 1
                current_lines = []
            fig_lines = [stripped]
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line == "":
                    j += 1
                    break
                if IMG_RE.search(next_line) or (next_line.startswith("TABLEBLOCK§") and next_line.endswith("§")):
                    break
                fig_lines.append(next_line)
                j += 1
            fig_text = " ".join(x for x in fig_lines if x).strip()
            if fig_text:
                segments.append({"base_idx": base_idx, "is_figure": True, "text": fig_text})
                base_idx += 1
            i = j
            continue

        current_lines.append(lines[i])
        i += 1

    if current_lines:
        joined = " ".join(x.strip() for x in current_lines if x.strip()).strip()
        if joined:
            segments.append({"base_idx": base_idx, "is_figure": False, "text": joined})
    return segments


def split_paragraph_into_sentences(paragraph_text: str, is_figure_para: bool) -> List[str]:
    if not (paragraph_text or "").strip():
        return []
    if is_figure_para:
        return [paragraph_text.strip()]

    text = re.sub(r"\s*\n\s*", " ", paragraph_text)
    text = _protect_figrefs(_protect_citations(text))
    doc = get_nlp()(text)
    final_sents: List[str] = []
    for sent in [chunk.text.strip() for chunk in doc.sents if chunk.text.strip()]:
        for sub in _resplit_boundary(sent):
            restored = _unprotect_figrefs(_unprotect_citations(sub.strip()))
            if restored:
                final_sents.append(restored)
    return final_sents


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", (text or "").strip()))


def _is_title_sentence(entry: Dict) -> bool:
    text = (entry.get("text") or "").strip().lower()
    title = (entry.get("main_header_text") or "").strip().lower()
    return bool(text) and bool(title) and text == title


def merge_short_heads(entries: List[Dict], max_tokens: int = 5) -> List[Dict]:
    out: List[Dict] = []
    for entry in entries:
        if out:
            prev = out[-1]
            if (
                entry.get("block_id") == prev.get("block_id")
                and not prev.get("is_figure", False)
                and not entry.get("is_figure", False)
                and not _is_title_sentence(prev)
                and not _is_title_sentence(entry)
            ):
                prev_text = (prev.get("text") or "").strip()
                if prev_text and _word_count(prev_text) <= max_tokens:
                    last_ch = prev_text[-1:]
                    is_num_heading = bool(re.match(r"^\d+(?:\.\d+)*\.$", prev_text))
                    if last_ch not in ".?!;:" or is_num_heading:
                        merged = dict(entry)
                        merged["text"] = (prev_text.rstrip() + " " + (entry.get("text") or "").lstrip()).strip()
                        merged["base_para_idx"] = prev.get("base_para_idx", entry.get("base_para_idx"))
                        out[-1] = merged
                        continue
        out.append(entry)
    return out


def merge_short_heads_across_fig(entries: List[Dict]) -> List[Dict]:
    out = list(entries)
    i = 0
    while i < len(out):
        cur = out[i]
        if cur.get("is_figure", False):
            i += 1
            continue
        prev_text = (cur.get("text") or "").strip()
        if not prev_text:
            i += 1
            continue
        block_id = cur.get("block_id")
        j = i + 1
        seen_fig = False
        while j < len(out) and out[j].get("block_id") == block_id:
            if out[j].get("is_figure", False):
                seen_fig = True
                j += 1
                continue
            break
        if not seen_fig or j >= len(out) or out[j].get("block_id") != block_id or out[j].get("is_figure", False):
            i += 1
            continue
        tail = out[j]
        tail_text = (tail.get("text") or "").lstrip()
        if not tail_text:
            i += 1
            continue
        cond_no_terminator = prev_text[-1:] not in ".?!;:"
        cond_tail_lower = tail_text[:1].islower()
        if cond_no_terminator or cond_tail_lower:
            out[i]["text"] = (prev_text.rstrip() + " " + tail_text).strip()
            try:
                out[i]["base_para_idx"] = min(int(out[i].get("base_para_idx", 1)), int(tail.get("base_para_idx", 1)))
            except Exception:
                pass
            del out[j]
            continue
        i += 1
    return out


def merge_short_tails_around_fig(entries: List[Dict], max_tail_tokens: int = 5) -> List[Dict]:
    out: List[Dict] = []
    i = 0
    while i < len(entries):
        cur = entries[i]
        if cur.get("is_figure", False):
            out.append(cur)
            j = i + 1
            if j < len(entries):
                tail = entries[j]
                if not tail.get("is_figure", False) and _word_count(tail.get("text", "")) <= max_tail_tokens:
                    k = len(out) - 2
                    while k >= 0 and out[k].get("is_figure", False):
                        k -= 1
                    if k >= 0 and out[k].get("block_id") == cur.get("block_id") == tail.get("block_id"):
                        out[k]["text"] = ((out[k].get("text", "")).rstrip() + " " + (tail.get("text", "")).lstrip()).strip()
                        i = j + 1
                        continue
            i += 1
            continue
        out.append(cur)
        i += 1
    return out


def merge_orphan_short(entries: List[Dict], max_tokens: int = 5) -> List[Dict]:
    out: List[Dict] = []
    for entry in entries:
        if out:
            prev = out[-1]
            if (
                entry.get("block_id") == prev.get("block_id")
                and not _is_title_sentence(prev)
                and not _is_title_sentence(entry)
                and not entry.get("is_figure", False)
                and not prev.get("is_figure", False)
            ):
                cur_text = entry.get("text", "") or ""
                prev_text = prev.get("text", "") or ""
                cur_stripped = cur_text.lstrip()
                if not cur_stripped:
                    continue
                first_ch = cur_stripped[:1]
                prev_strip = prev_text.strip()
                prev_end = prev_strip[-1:] if prev_strip else ""
                prev_lower = prev_strip.lower()
                if prev_lower.endswith("et al.") and cur_stripped.startswith("["):
                    prev["text"] = (prev_strip + " " + cur_stripped).strip()
                    continue
                if prev_lower.endswith("i.e.") or prev_lower.endswith("e.g."):
                    prev["text"] = (prev_strip + " " + cur_stripped).strip()
                    continue
                if prev_end == "." and first_ch.islower():
                    prev["text"] = (prev_strip + " " + cur_stripped).strip()
                    continue
                if _word_count(cur_text) <= max_tokens:
                    tail_end = cur_text.strip()[-1:] if cur_text.strip() else ""
                    panel_like = bool(re.match(r"^[A-Za-z]?\d+[A-Za-z]?\.?$", cur_stripped))
                    looks_tail = (
                        first_ch.islower()
                        or first_ch in (")", "]", "}", "%", "±", "+", "-", "[", "(")
                        or first_ch.isdigit()
                        or tail_end not in ".?!;:"
                        or panel_like
                    )
                    if prev_end not in ".?!;" or looks_tail:
                        prev["text"] = (prev_text.rstrip() + " " + cur_stripped).strip()
                        continue
        out.append(entry)
    return out


def split_block_to_entries(block_row: pd.Series) -> List[Dict]:
    entries: List[Dict] = []
    segments = _block_to_segments_with_figures(block_row.get("text", "") or "")
    for seg in segments:
        base_idx = int(seg.get("base_idx", 1))
        is_figure = bool(seg.get("is_figure", False))
        text = seg.get("text", "") or ""
        for sent in split_paragraph_into_sentences(text, is_figure):
            sent = (sent or "").strip()
            if not sent:
                continue
            entries.append(
                {
                    "pdf_name": block_row["pdf_name"],
                    "block_id": int(block_row["block_id"]),
                    "main_section_norm": block_row["main_section_norm"],
                    "main_header_text": block_row.get("main_header_text", "") or "",
                    "base_para_idx": base_idx,
                    "is_figure": is_figure,
                    "text": sent,
                }
            )
    if not entries:
        return []
    entries = merge_short_heads(entries, max_tokens=5)
    entries = merge_short_heads_across_fig(entries)
    entries = merge_short_tails_around_fig(entries, max_tail_tokens=5)
    entries = merge_orphan_short(entries, max_tokens=5)
    return entries


def build_sentence_df_from_block_df(
    df_block: pd.DataFrame,
    pdf_name: str,
    max_sents_per_para: int = 8,
) -> pd.DataFrame:
    final_entries: List[Dict] = []
    global_sent_id = 0
    global_para_id = 0
    df_block = df_block.sort_values(["block_id"]).reset_index(drop=True)

    for _, block_row in df_block.iterrows():
        entries = split_block_to_entries(block_row)
        if not entries:
            continue

        para_id_in_block = 0
        cur_para_base_idx: Optional[int] = None
        cur_sent_in_para = 0
        last_was_figure = False
        for entry in entries:
            is_figure = bool(entry.get("is_figure", False))
            base_idx = int(entry.get("base_para_idx", 1))
            if is_figure:
                para_id_in_block += 1
                global_para_id += 1
                global_sent_id += 1
                final_entries.append(
                    {
                        "pdf_name": pdf_name,
                        "block_id": int(block_row["block_id"]),
                        "main_section_norm": block_row["main_section_norm"],
                        "main_header_text": block_row.get("main_header_text", "") or "",
                        "para_id_in_block": para_id_in_block,
                        "para_global_id": global_para_id,
                        "sent_id_in_para": 1,
                        "sent_global_id": global_sent_id,
                        "is_figure": True,
                        "text": entry.get("text", "").strip(),
                    }
                )
                cur_para_base_idx = None
                cur_sent_in_para = 0
                last_was_figure = True
                continue

            need_new_para = (
                last_was_figure
                or cur_para_base_idx is None
                or base_idx != cur_para_base_idx
                or cur_sent_in_para >= max_sents_per_para
            )
            if need_new_para:
                para_id_in_block += 1
                global_para_id += 1
                cur_para_base_idx = base_idx
                cur_sent_in_para = 0

            cur_sent_in_para += 1
            global_sent_id += 1
            final_entries.append(
                {
                    "pdf_name": pdf_name,
                    "block_id": int(block_row["block_id"]),
                    "main_section_norm": block_row["main_section_norm"],
                    "main_header_text": block_row.get("main_header_text", "") or "",
                    "para_id_in_block": para_id_in_block,
                    "para_global_id": global_para_id,
                    "sent_id_in_para": cur_sent_in_para,
                    "sent_global_id": global_sent_id,
                    "is_figure": False,
                    "text": entry.get("text", "").strip(),
                }
            )
            last_was_figure = False

    return pd.DataFrame(final_entries)


def save_sentence_df(df: pd.DataFrame, out_base: str) -> None:
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    cols = [
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
    df.to_csv(out_base + ".csv", index=False, encoding="utf-8-sig", columns=cols)
    with open(out_base + ".txt", "w", encoding="utf-8") as f_txt, open(
        out_base + ".md", "w", encoding="utf-8"
    ) as f_md:
        for _, row in df.iterrows():
            tag = (
                f"[Block {row['block_id']}, Sec: {row['main_section_norm']}, "
                f"Para {row['para_id_in_block']}, Sent {row['sent_id_in_para']}]"
            )
            text = str(row["text"])
            f_txt.write(f"{tag}:\n{text}\n\n")
            f_md.write(f"{tag}  \n\n{text}\n\n---\n\n")


def tokenized_bundle_exists(token_dir: str, paper_id: str) -> bool:
    return all(os.path.exists(os.path.join(token_dir, f"{paper_id}.{ext}")) for ext in ("csv", "md", "txt"))


def resolve_cut_markdown(paper_dir: str, paper_id: str) -> Optional[str]:
    candidates = [
        os.path.join(paper_dir, "preprocess", "cut_property", f"{paper_id}_cut.md"),
        os.path.join(paper_dir, "preprocess", "cut", f"{paper_id}_cut.md"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def process_one_paper(
    paper_dir: str,
    section_log_path: str,
    save_paragraphs: bool = True,
    skip_existing: bool = True,
) -> bool:
    paper_id = paper_id_from_dir(paper_dir)
    if paper_id is None:
        return False

    cut_md = resolve_cut_markdown(paper_dir, paper_id)
    if not cut_md:
        print(f"[SKIP] {paper_id}: cut markdown not found.")
        return False

    paragraph_dir = os.path.join(paper_dir, "property", "Paragraphs")
    token_dir = os.path.join(paper_dir, "preprocess", "Tokenized")
    paragraph_base = os.path.join(paragraph_dir, paper_id)
    token_base = os.path.join(token_dir, paper_id)

    if skip_existing and tokenized_bundle_exists(token_dir, paper_id):
        print(f"[SKIP] {paper_id}: Tokenized output already exists.")
        return False

    with open(cut_md, "r", encoding="utf-8") as fh:
        md_text = fh.read()

    blocks = markdown_to_blocks(md_text, verbose=False)
    blocks = trim_blocks_before_first_h1(blocks)
    stabilize_block_sections(blocks, verbose=False)
    report_unknown_ratio(blocks, paper_id, section_log_path)

    df_blocks = build_paragraph_df(blocks, pdf_name=paper_id)
    if save_paragraphs:
        save_paragraph_df(df_blocks, paragraph_base)

    df_sent = build_sentence_df_from_block_df(df_blocks, pdf_name=paper_id, max_sents_per_para=8)
    save_sentence_df(df_sent, token_base)
    print(f"[OK] {paper_id}: Step1 tokenized output saved.")
    return True


def process_all_papers(
    mining_root: str,
    paper_ids: Optional[List[str]] = None,
    save_paragraphs: bool = True,
    skip_existing: bool = True,
) -> None:
    root = ensure_root_exists(mining_root)
    section_log_path = os.path.join(root, "section_quality.log")
    paper_dirs = iter_paper_dirs(root, paper_ids=paper_ids)
    for paper_dir in tqdm(paper_dirs, desc="Step1: tokenize"):
        process_one_paper(
            paper_dir=paper_dir,
            section_log_path=section_log_path,
            save_paragraphs=save_paragraphs,
            skip_existing=skip_existing,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Property preprocessing step1: block splitting + sentence tokenization.")
    parser.add_argument("--root", required=True, help="Mining root directory.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids", help="Only process the given paper id. Repeatable.")
    parser.add_argument("--no-save-paragraphs", action="store_true", help="Do not write property/Paragraphs intermediate files.")
    parser.add_argument("--force", action="store_true", help="Re-run even if Tokenized output already exists.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_all_papers(
        mining_root=args.root,
        paper_ids=args.paper_ids,
        save_paragraphs=not args.no_save_paragraphs,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
