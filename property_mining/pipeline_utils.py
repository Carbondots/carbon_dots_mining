#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helper utilities for the cleaned property-mining pipeline."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple


NUMERIC_PROPERTY_TAGS = ("Ex", "Em", "QY", "lifetime")
_NUMERIC_PROPERTY_TAG_SET = set(NUMERIC_PROPERTY_TAGS)
_DIGIT_RE = re.compile(r"\d")


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def extract_window_text(evidence_text: str) -> str:
    text = "" if evidence_text is None else str(evidence_text)
    marker = "\nWINDOW_TEXT:\n"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    if text.startswith("WINDOW_TEXT:\n"):
        return text.split("WINDOW_TEXT:\n", 1)[1].strip()
    return text.strip()


def get_no_digit_numeric_tags(
    tags: Sequence[str],
    *,
    evidence_text: str = "",
    window_text: str = "",
) -> List[str]:
    ordered_tags = dedupe_preserve_order(tags)
    numeric_tags = [tag for tag in ordered_tags if tag in _NUMERIC_PROPERTY_TAG_SET]
    if not numeric_tags:
        return []

    text = str(window_text or "").strip()
    if not text:
        text = extract_window_text(evidence_text)

    if _DIGIT_RE.search(text):
        return []
    return numeric_tags


def drop_no_digit_numeric_tags(
    tags: Sequence[str],
    *,
    evidence_text: str = "",
    window_text: str = "",
) -> Tuple[List[str], List[str]]:
    ordered_tags = dedupe_preserve_order(tags)
    removed_tags = set(
        get_no_digit_numeric_tags(
            ordered_tags,
            evidence_text=evidence_text,
            window_text=window_text,
        )
    )
    kept_tags = [tag for tag in ordered_tags if tag not in removed_tags]
    removed_tags_ordered = [tag for tag in ordered_tags if tag in removed_tags]
    return kept_tags, removed_tags_ordered
