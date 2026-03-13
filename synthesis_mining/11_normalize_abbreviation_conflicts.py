#!/usr/bin/env python3
"""Step 11: normalize abbreviation/full-name conflicts in merged CSV values."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from lmstudio import llm

from synthesis_unit import (
    DEFAULT_LLM_MODEL_NAME,
    display_path,
    ensure_directory,
    extract_document_id,
    list_document_dirs,
    load_document_context_text,
    split_top_level_items,
    strip_think_block,
)


LLM_MODEL = llm(DEFAULT_LLM_MODEL_NAME)
TEMPERATURE = 0.0

SUBFOLDER = "LLM_table_qwen2.5vl"
IN_CSV_EXT = ".csv"
REVIEW_JSON_SUFFIX = "_review_todo.json"
CASCADE_JSON_SUFFIX = "_cascade_review.json"
CASCADE_MAP_JSON_SUFFIX = "_cascade_map.json"

ENTITY_VALUE_COLUMNS = {"Precursor", "Solvent"}
QUANTITY_SUBJECT_COLUMNS = {"Precursor_Amount": "Precursor", "Solvent_Volume": "Solvent"}

STATUS_WRITTEN = "written"
STATUS_SKIP = "skipped"
STATUS_ERROR = "error"

def norm_token(value: str) -> str:
    return " ".join((value or "").strip().lower().split()).strip(".")



def find_first_paren_span(text: str) -> tuple[int, int]:
    if not text:
        return -1, -1
    opens = {"(": ")", "（": "）"}
    start = -1
    start_char = ""
    for i, ch in enumerate(text):
        if ch in opens:
            start = i
            start_char = ch
            break
    if start < 0:
        return -1, -1

    close = opens[start_char]
    depth = 1
    for j in range(start + 1, len(text)):
        ch = text[j]
        if ch in opens:
            depth += 1
        elif ch in opens.values():
            depth -= 1
            if depth == 0:
                return start, j
    return -1, -1



def subject_before_paren(text: str) -> str:
    i, _ = find_first_paren_span(text)
    return text[:i].strip() if i >= 0 else (text or "").strip()



def has_digit(value: str) -> bool:
    return any(ch.isdigit() for ch in (value or ""))



def looks_like_acronym(value: str) -> bool:
    s = (value or "").strip()
    if not s or " " in s:
        return False
    s2 = s.replace(".", "").replace("-", "")
    if len(s2) < 2 or len(s2) > 12:
        return False
    return any(ch.isalpha() for ch in s2)



def cleanup_full_name(value: str) -> str:
    s = " ".join((value or "").strip().split())
    if not s:
        return ""
    s = s.rstrip(".,;:!?）]")
    parts = s.split()
    if len(parts) > 8:
        s = " ".join(parts[-8:])
    return norm_token(s)



def scan_abbrev_candidates(text: str, ctx_window: int = 60) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    if not text:
        return result

    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch in "(（":
            close = ")" if ch == "(" else "）"
            j = i + 1
            while j < n and text[j] != close:
                j += 1
            if j < n:
                inside = text[i + 1 : j].strip()
                left = text[max(0, i - 80) : i]
                left_piece = left.split("\n")[-1].strip()
                left_last = left_piece.split()[-1] if left_piece else ""

                if looks_like_acronym(inside) and left_piece:
                    acr = norm_token(inside)
                    full = cleanup_full_name(left_piece)
                    if acr and full:
                        ctx = text[max(0, i - ctx_window) : min(n, j + 1 + ctx_window)]
                        result.setdefault(acr, {}).setdefault(full, []).append(ctx)

                elif looks_like_acronym(left_last) and inside:
                    acr = norm_token(left_last)
                    full = cleanup_full_name(inside)
                    if acr and full:
                        ctx = text[max(0, i - ctx_window) : min(n, j + 1 + ctx_window)]
                        result.setdefault(acr, {}).setdefault(full, []).append(ctx)
                i = j
        i += 1

    return result



def call_llm(prompt: str, max_tokens: int = 1200) -> str:
    result = LLM_MODEL.respond(prompt, config={"temperature": TEMPERATURE, "maxTokens": max_tokens})
    return strip_think_block(getattr(result, "content", "") or "")



def llm_pick_candidate(acronym: str, candidates: list[dict]) -> dict:
    payload = json.dumps(candidates, ensure_ascii=False, indent=2)
    prompt = f"""
You must choose the single full chemical name used in THIS PAPER for the given token,
STRICTLY from the provided candidates by their "id".

Token: "{acronym}"

Candidates:
{payload}

Rules (critical):
- Choose exactly ONE candidate by "id". Do NOT invent names or use text outside candidate.name.
- Prefer well-formed chemical names that match the contexts.
- 'self' means the token itself is already a complete chemical name (e.g., "luminol", "citric acid").
  Choose 'self' ONLY IF the token is itself a full name AND no other candidate is clearly better.
- If multiple distinct chemicals truly share this token -> decision="ambiguous".
- If the token is not a reagent/solvent/precursor -> decision="rejected".
- If none is plausible -> return null (unknown).

Output EXACTLY one JSON object:
{{
  "acronym": "{acronym}",
  "choice_id": "<string or null>",
  "choice_name": "<string>",     // MUST EQUAL the chosen candidate's 'name'
  "confidence": <float 0..1>,
  "decision": "accepted" | "ambiguous" | "rejected",
  "reason": "<short one-line reason>"
}}
""".strip()

    text = call_llm(prompt, max_tokens=600)
    beg, end = text.find("{"), text.rfind("}")
    if beg < 0 or end <= beg:
        return {"acronym": acronym, "choice_id": None, "decision": "rejected", "reason": "no json"}
    try:
        obj = json.loads(text[beg : end + 1])
    except Exception:
        return {"acronym": acronym, "choice_id": None, "decision": "rejected", "reason": "parse error"}

    cid = obj.get("choice_id")
    valid_ids = {c["id"] for c in candidates}
    if obj.get("decision") == "accepted" and cid in valid_ids:
        return {"acronym": acronym, "choice_id": cid, "decision": "accepted", "reason": obj.get("reason", "")}
    if obj.get("decision") in {"ambiguous", "rejected"}:
        return {"acronym": acronym, "choice_id": None, "decision": obj.get("decision"), "reason": obj.get("reason", "")}
    return {"acronym": acronym, "choice_id": None, "decision": "rejected", "reason": "invalid decision"}



def values_unique_ignore_case(values: list[str]) -> bool:
    norm = {" ".join((v or "").strip().lower().split()) for v in values if (v or "").strip() and v != "N/A"}
    return len(norm) == 1



def common_tokens(raw_values: list[str], is_quantity: bool) -> set[str]:
    inter: set[str] | None = None
    for val in raw_values:
        s = str(val or "").strip()
        if not s or s == "N/A":
            continue
        cur: set[str] = set()
        for seg in split_top_level_items(s):
            tok = subject_before_paren(seg) if is_quantity else seg
            if tok:
                cur.add(norm_token(tok))
        if inter is None:
            inter = cur
        else:
            inter &= cur
    return inter or set()



def apply_map_entity(raw: str, mapping: dict[str, str], skip_tokens: set[str]) -> str:
    if not raw or raw == "N/A":
        return raw
    out = []
    for seg in split_top_level_items(raw):
        s = seg.strip()
        if not s:
            continue
        i, j = find_first_paren_span(s)
        subj = s[:i].strip() if i >= 0 else s
        tail = s[i:].strip() if i >= 0 else ""
        key = norm_token(subj)
        if key in skip_tokens:
            out.append(s)
            continue
        new_subj = mapping.get(key, subj)
        out.append((new_subj + (" " + tail if tail else "")).strip())
    return ", ".join(out) if out else "N/A"



def apply_map_quantity(raw: str, mapping: dict[str, str], skip_tokens: set[str]) -> str:
    if not raw or raw == "N/A":
        return "N/A"
    out = []
    for seg in split_top_level_items(raw):
        s = seg.strip()
        i, j = find_first_paren_span(s)
        if i < 0 or j < 0:
            return "N/A"
        subj = s[:i].strip()
        inner = s[i + 1 : j].strip()
        if not subj or not inner or not has_digit(inner):
            return "N/A"
        key = norm_token(subj)
        if key in skip_tokens:
            out.append(f"{subj} ({inner})")
            continue
        out.append(f"{mapping.get(key, subj)} ({inner})")
    return ", ".join(out) if out else "N/A"



def llm_equivalence_merge(column: str, raw_values: list[str], replaced_values: list[str], is_quantity: bool) -> tuple[str, str]:
    payload = {
        "column": column,
        "is_quantity": is_quantity,
        "raw_values": raw_values,
        "replaced_values": replaced_values,
    }
    prompt = f"""
You will judge if multiple table values refer to the SAME thing.

Input (JSON):
{json.dumps(payload, ensure_ascii=False, indent=2)}

Rules:
- Consider ONLY the "replaced_values" as the basis for normalization.
- For entity columns (chemicals): same set of chemicals, ignoring order and case -> SAME.
- For quantity columns: SAME requires the same chemical subject(s) AND the same numeric values/units.
- Do NOT invent new chemicals or numbers. If unsure -> CONFLICT.
- If SAME, return one single-line canonical value chosen from the replaced_values (or a minimal cleaned combination).
- Output EXACTLY two lines:
<decision>SAME|CONFLICT</decision>
<value>ONE_LINE_VALUE_OR_N/A</value>
""".strip()

    text = call_llm(prompt, max_tokens=1200)
    d1, d2 = text.find("<decision>"), text.find("</decision>")
    v1, v2 = text.find("<value>"), text.find("</value>")
    if d1 >= 0 and d2 > d1 and v1 >= 0 and v2 > v1:
        dec = text[d1 + 10 : d2].strip().upper()
        val = text[v1 + 7 : v2].strip() or "N/A"
        if dec in {"SAME", "CONFLICT"}:
            return dec, val
    return "CONFLICT", "N/A"



def read_csv_rows(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []



def atomic_write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_directory(path.parent)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8-sig", newline="") as tf:
        writer = csv.DictWriter(tf, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmp_path = Path(tf.name)
    tmp_path.replace(path)



def build_doc_map(document_id: str, work_folder: Path, doc_text: str, conflict_items: list[dict]) -> dict[str, str]:
    candidates_by_text = scan_abbrev_candidates(doc_text)

    acronyms: set[str] = set()
    non_acros: set[str] = set()
    for item in conflict_items:
        col = item.get("column", "")
        vals = item.get("raw_values", []) or []
        toks = set()
        for v in vals:
            if not v or v == "N/A":
                continue
            if col in ENTITY_VALUE_COLUMNS:
                toks.update(norm_token(seg) for seg in split_top_level_items(str(v)))
            else:
                toks.add(norm_token(subject_before_paren(str(v))))
        for t in toks:
            if looks_like_acronym(t):
                acronyms.add(t)
            elif t:
                non_acros.add(t)

    skip = set()
    for item in conflict_items:
        is_qty = item.get("column") in QUANTITY_SUBJECT_COLUMNS
        skip |= {t for t in common_tokens(item.get("raw_values", []) or [], is_qty) if looks_like_acronym(t)}

    decisions = []
    mapping: dict[str, str] = {}

    for acr in sorted(acronyms):
        if acr in skip:
            continue

        pool: dict[str, set[str]] = {}
        for full, ctxs in (candidates_by_text.get(acr, {}) or {}).items():
            pool.setdefault(full, set()).update(ctxs or [])
        for full in non_acros:
            pool.setdefault(full, set())

        candidates = [{"id": "self", "name": acr, "contexts": [], "is_self": True}]
        idx = 1
        for full, ctxs in pool.items():
            candidates.append({"id": f"c{idx}", "name": full, "contexts": list(ctxs)[:3], "is_self": False})
            idx += 1

        if len(candidates) == 1:
            continue

        decision = llm_pick_candidate(acr, candidates)
        decisions.append({"acronym": acr, "candidates": candidates, "decision": decision})

        if decision.get("decision") == "accepted":
            cid = decision.get("choice_id")
            cand_map = {c["id"]: c["name"] for c in candidates}
            if cid in cand_map:
                mapping[acr] = cand_map[cid]

    map_report = {
        "doc_id": document_id,
        "model": "qwen2.5-vl-32b-instruct",
        "temperature": TEMPERATURE,
        "skipped_common_acronyms": sorted(skip),
        "acronym_decisions": decisions,
    }
    (work_folder / f"{document_id}{CASCADE_MAP_JSON_SUFFIX}").write_text(
        json.dumps(map_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return mapping



def process_document(document_dir: Path) -> dict:
    document_id = extract_document_id(document_dir)
    work_folder = document_dir / "Synthesis" / SUBFOLDER
    csv_path = work_folder / f"{document_id}{IN_CSV_EXT}"
    review_path = work_folder / f"{document_id}{REVIEW_JSON_SUFFIX}"
    cascade_path = work_folder / f"{document_id}{CASCADE_JSON_SUFFIX}"

    if not work_folder.exists() or not csv_path.exists() or not review_path.exists():
        return {"doc_id": document_id, "status": STATUS_SKIP, "reason": "missing_input"}

    rows = read_csv_rows(csv_path)
    if not rows:
        return {"doc_id": document_id, "status": STATUS_SKIP, "reason": "empty_csv"}

    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"doc_id": document_id, "status": STATUS_ERROR, "reason": f"invalid_review_json: {exc}"}

    items = review.get("items", []) if isinstance(review, dict) else []
    if not isinstance(items, list):
        return {"doc_id": document_id, "status": STATUS_ERROR, "reason": "invalid_review_items"}

    conflict_items = [
        item
        for item in items
        if item.get("column") in ENTITY_VALUE_COLUMNS or item.get("column") in QUANTITY_SUBJECT_COLUMNS
    ]
    if not conflict_items:
        cascade_path.write_text(
            json.dumps(
                {
                    "doc_id": document_id,
                    "processed": 0,
                    "total_conflict_items": 0,
                    "skipped": True,
                    "reason": "no_conflict_items",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"doc_id": document_id, "status": STATUS_SKIP, "reason": "no_conflict_items"}

    doc_text = load_document_context_text(document_dir)
    doc_map = build_doc_map(document_id, work_folder, doc_text, conflict_items)

    if not doc_map:
        cascade_path.write_text(
            json.dumps(
                {
                    "doc_id": document_id,
                    "processed": 0,
                    "total_conflict_items": len(conflict_items),
                    "skipped": True,
                    "reason": "empty_map",
                    "items": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"doc_id": document_id, "status": STATUS_SKIP, "reason": "empty_map"}

    index_by_cd: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        cd = (row.get("CDs_Naming_in_Paper") or "").strip()
        if cd:
            index_by_cd.setdefault(cd, []).append(idx)

    processed = 0
    result_items = []

    for item in conflict_items:
        cd_name = (item.get("cd_name") or "").strip()
        col = item.get("column", "")
        raw_values = [str(v or "N/A") for v in (item.get("raw_values", []) or [])]
        if not cd_name or not col:
            continue

        is_quantity = col in QUANTITY_SUBJECT_COLUMNS
        skip_tokens = common_tokens(raw_values, is_quantity)

        replaced_values = []
        for raw in raw_values:
            if is_quantity:
                replaced_values.append(apply_map_quantity(raw, doc_map, skip_tokens))
            else:
                replaced_values.append(apply_map_entity(raw, doc_map, skip_tokens))

        if values_unique_ignore_case(replaced_values):
            status = "resolved"
            chosen = next(v for v in replaced_values if (v or "").strip() and v != "N/A")
            llm_after = None
        else:
            dec, val = llm_equivalence_merge(col, raw_values, replaced_values, is_quantity)
            if dec == "SAME" and val and val != "N/A":
                status = "resolved_via_llm"
                chosen = val
                llm_after = {"decision": dec, "value": val}
            else:
                status = "still_conflict"
                chosen = "N/A"
                llm_after = {"decision": dec, "value": "N/A"}

        result_items.append(
            {
                "doc_id": document_id,
                "cd_name": cd_name,
                "column": col,
                "status": status,
                "chosen_value": chosen,
                "raw_values": raw_values,
                "replaced_values": replaced_values,
                "used_map": doc_map,
                "llm_after_mapping": llm_after,
                "skip_tokens": sorted(skip_tokens),
            }
        )

        if status in {"resolved", "resolved_via_llm"} and chosen != "N/A":
            for ridx in index_by_cd.get(cd_name, []):
                rows[ridx][col] = chosen
            processed += 1

    if rows:
        fieldnames = list(rows[0].keys())
        atomic_write_csv(csv_path, rows, fieldnames)

    cascade_path.write_text(
        json.dumps(
            {
                "doc_id": document_id,
                "processed": processed,
                "total_conflict_items": len(conflict_items),
                "used_map": doc_map,
                "items": result_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "doc_id": document_id,
        "status": STATUS_WRITTEN,
        "reason": "ok",
        "processed": processed,
        "total_conflict_items": len(conflict_items),
    }

def run(pipeline_root: Path) -> None:
    jobs = list_document_dirs(pipeline_root)
    if not jobs:
        print("No numeric document folders found.")
        return

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_log = pipeline_root / "step10.5_global.log"
    global_jsonl = pipeline_root / "step10.5_global.jsonl"

    done = 0
    skipped = 0
    errors = 0
    results = []

    for document_dir in jobs:
        result = process_document(document_dir)
        results.append(result)
        print(f"{result.get('doc_id')}: {result.get('status')} ({result.get('reason')})")

        if result.get("status") == STATUS_WRITTEN:
            done += 1
        elif result.get("status") == STATUS_SKIP:
            skipped += 1
        else:
            errors += 1

        with global_jsonl.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "doc_result",
                        "run_stamp": run_stamp,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        **result,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    grouped: dict[str, list[str]] = {}
    for item in results:
        if item.get("status") == STATUS_WRITTEN:
            continue
        key = f"{item.get('status')}-{item.get('reason')}"
        grouped.setdefault(key, []).append(str(item.get("doc_id", "")))

    with global_log.open("a", encoding="utf-8") as f:
        f.write(f"\n=== Run {run_stamp} ===\n")
        f.write(f"done={done}, skipped={skipped}, errors={errors}\n")
        if grouped:
            for key in sorted(grouped):
                f.write(f"{key}\n")
                f.write("; ".join(sorted(set(grouped[key]))) + "\n")
        else:
            f.write("No non-written documents.\n")

    print(f"Completed. written={done}, skipped={skipped}, errors={errors}")
    print(f"Global log: {display_path(global_log, pipeline_root)}")
    print(f"Global jsonl: {display_path(global_jsonl, pipeline_root)}")



def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize abbreviation/full-name conflicts in merged CSV fields.")
    parser.add_argument("--mining-root", dest="pipeline_root", type=Path, default=Path("data") / "mining")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    run(args.pipeline_root)
