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
def load_tag_whitelist(path: str | None = None) -> dict[str, str]:
    whitelist_path = Path(path) if path else DEFAULT_WHITELIST_PATH
    if not whitelist_path.exists():
        return {}

    with whitelist_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    mapping: dict[str, str] = {}
    for item in data.get("tags", []):
        canonical = normalize_tag(item.get("tag", ""))
        if not canonical:
            continue
        mapping[canonical] = canonical
        for source in str(item.get("source_examples", "")).split(";"):
            source = normalize_tag(source)
            if source:
                mapping[source] = canonical
    return mapping


def controlled_tags(value: Any, max_tags: int = 8, whitelist_path: str | None = None) -> list[str]:
    whitelist = load_tag_whitelist(whitelist_path)
    if not whitelist:
        return []

    tags = []
    seen = set()
    for raw_tag in iter_tags(value):
        tag = normalize_tag(raw_tag)
        canonical = whitelist.get(tag)
        if canonical and canonical not in seen:
            seen.add(canonical)
            tags.append(canonical)
        if len(tags) >= max_tags:
            break
    return tags


def metadata_field(metadata: dict[str, Any], field: str) -> Any:
    if field == "controlled_tag_list":
        return controlled_tags(metadata.get("tag_list"))
    return metadata[field]
