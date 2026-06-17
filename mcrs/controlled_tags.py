"""Controlled tag utilities for retrieval metadata."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WHITELIST_PATH = PROJECT_ROOT / "config" / "tag_whitelist.json"


def normalize_tag(tag: Any) -> str:
    tag = str(tag or "").strip().lower()
    tag = tag.replace("_", " ").replace("/", " / ")
    tag = re.sub(r"\s+", " ", tag)
    return tag.strip(" -_.,;:!?")


def iter_tags(value: Any):
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            yield item
        return
    for item in str(value).split(","):
        yield item


@lru_cache(maxsize=4)
def load_tag_whitelist(path: str | None = None) -> tuple[dict[str, str], dict[str, int]]:
    whitelist_path = Path(path) if path else DEFAULT_WHITELIST_PATH
    if not whitelist_path.exists():
        return {}, {}

    with whitelist_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    mapping: dict[str, str] = {}
    ranks: dict[str, int] = {}
    for rank, item in enumerate(data.get("tags", [])):
        canonical = normalize_tag(item.get("tag", ""))
        if not canonical:
            continue
        mapping[canonical] = canonical
        ranks[canonical] = min(rank, ranks.get(canonical, rank))
        for source in str(item.get("source_examples", "")).split(";"):
            source = normalize_tag(source)
            if source:
                mapping[source] = canonical
    return mapping, ranks


def controlled_tags(value: Any, max_tags: int = 8, whitelist_path: str | None = None) -> list[str]:
    whitelist, ranks = load_tag_whitelist(whitelist_path)
    if not whitelist:
        return []

    first_seen: dict[str, int] = {}
    seen = set()
    for index, raw_tag in enumerate(iter_tags(value)):
        tag = normalize_tag(raw_tag)
        canonical = whitelist.get(tag)
        if canonical and canonical not in seen:
            seen.add(canonical)
            first_seen[canonical] = index

    return sorted(
        seen,
        key=lambda tag: (ranks.get(tag, len(ranks)), first_seen[tag], tag),
    )[:max_tags]


def metadata_field(metadata: dict[str, Any], field: str) -> Any:
    if field == "controlled_tag_list":
        return controlled_tags(metadata.get("tag_list"))
    return metadata[field]
