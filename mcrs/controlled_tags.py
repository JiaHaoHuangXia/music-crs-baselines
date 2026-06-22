"""Small tag-normalization helpers for retrieval and reranking."""

from __future__ import annotations

import re
from typing import Any


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
    "nu-metal": "nu metal",
    "post hardcore": "post-hardcore",
    "post punk": "post-punk",
}

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
    return TAG_SYNONYMS.get(tag, tag)


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


def metadata_field(metadata: dict[str, Any], field: str) -> Any:
    if field == "controlled_tag_list":
        return normalized_music_tags(metadata.get("tag_list"))
    if field in {"artist_style_profile", "release_decade"}:
        return metadata.get(field, [] if field == "artist_style_profile" else "")
    return metadata[field]
