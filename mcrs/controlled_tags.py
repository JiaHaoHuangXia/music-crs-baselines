"""Controlled tag utilities for retrieval metadata."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WHITELIST_PATH = PROJECT_ROOT / "config" / "tag_whitelist.json"

TAG_SYNONYMS = {
    "1980s": "80s",
    "eighties": "80s",
    "80's": "80s",
    "1990s": "90s",
    "nineties": "90s",
    "90's": "90s",
    "2000s": "00s",
    "00's": "00s",
    "2010s": "10s",
    "10's": "10s",
    "hip-hop": "hip hop",
    "hiphop": "hip hop",
    "rap / hip hop": "hip hop",
    "rnb": "r&b",
    "r / b": "r&b",
    "rhythm and blues": "r&b",
    "synth-pop": "synthpop",
    "synth pop": "synthpop",
    "trip-hop": "trip hop",
    "lofi": "lo-fi",
    "lo fi": "lo-fi",
    "nu-metal": "nu metal",
    "post hardcore": "post-hardcore",
    "post punk": "post-punk",
    "science fiction": "sci-fi",
    "scifi": "sci-fi",
    "sci fi": "sci-fi",
}

TAG_PREFIX_SYNONYMS = (
    ("retro futuristic", "retro"),
    ("retro-futuristic", "retro"),
    ("retrofuturistic", "retro"),
)

TAG_NOISE_EXACT = {
    "",
    "favorite",
    "favorites",
    "favourite",
    "favourites",
    "fav",
    "favs",
    "seen live",
    "playlist",
    "my music",
}

TAG_NOISE_PATTERNS = [
    re.compile(r"^\d+\s+of\s+10\s+stars?$"),
    re.compile(r"^\d+\s*stars?$"),
    re.compile(r".*\bfav(e|s)?\b.*"),
    re.compile(r".*\bfavorites?\b.*"),
    re.compile(r".*\bfavourites?\b.*"),
    re.compile(r".*\bplaylist\b.*"),
]


def normalize_tag(tag: Any) -> str:
    tag = str(tag or "").strip().lower()
    tag = tag.replace("_", " ").replace("/", " / ")
    tag = re.sub(r"\s+", " ", tag)
    return tag.strip(" -_.,;:!?")


def normalize_music_tag(tag: Any) -> str:
    tag = normalize_tag(tag)
    tag = tag.replace("&amp;", "&")
    tag = re.sub(r"\s*-\s*", "-", tag)
    tag = re.sub(r"\s+", " ", tag)
    tag = TAG_SYNONYMS.get(tag, tag)
    for prefix, canonical in TAG_PREFIX_SYNONYMS:
        if tag.startswith(prefix):
            return canonical
    return tag


def is_tag_noise(tag: str) -> bool:
    if tag in TAG_NOISE_EXACT:
        return True
    if len(tag) < 2:
        return True
    return any(pattern.match(tag) for pattern in TAG_NOISE_PATTERNS)


def iter_tags(value: Any):
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            yield item
        return
    for item in str(value).split(","):
        yield item


def normalized_music_tags(value: Any, max_tags: int = 40) -> list[str]:
    tags = []
    seen = set()
    for raw_tag in iter_tags(value):
        tag = normalize_music_tag(raw_tag)
        if is_tag_noise(tag) or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
        if len(tags) >= max_tags:
            break
    return tags


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
