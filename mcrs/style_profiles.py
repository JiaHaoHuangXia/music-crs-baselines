"""Metadata enrichment helpers for retrieval."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from mcrs.controlled_tags import metadata_field, normalize_tag, normalized_music_tags

ARTIST_STYLE_PROFILE_MAX_TAGS = 8
ARTIST_STYLE_PROFILE_TEXT_WEIGHT = 3
TITLE_CLEANING_VERSION = "clean_titles_v2"
RELEASE_DECADE_VERSION = "release_decade_v1"
FIELD_TEXT_WEIGHTS = {
    "track_name": 1,
    "artist_name": 1,
    "album_name": 1,
    "artist_style_profile": ARTIST_STYLE_PROFILE_TEXT_WEIGHT,
    "release_decade": 1,
}


def artist_key(value: Any) -> str:
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    return normalize_tag(value)


def corpus_cache_name(corpus_types: list[str]) -> str:
    """Return a cache key that changes when enriched/weighted text changes."""
    parts = []
    for field in corpus_types:
        if field == "artist_style_profile":
            parts.append(
                f"{field}_top{ARTIST_STYLE_PROFILE_MAX_TAGS}_w{ARTIST_STYLE_PROFILE_TEXT_WEIGHT}"
            )
        elif field in {"track_name", "album_name"}:
            parts.append(f"{field}_{TITLE_CLEANING_VERSION}")
        elif field == "release_decade":
            parts.append(f"{field}_{RELEASE_DECADE_VERSION}")
        elif FIELD_TEXT_WEIGHTS.get(field, 1) != 1:
            parts.append(f"{field}_w{FIELD_TEXT_WEIGHTS[field]}")
        else:
            parts.append(field)
    return "_".join(parts)


def metadata_field_weight(field: str) -> int:
    return FIELD_TEXT_WEIGHTS.get(field, 1)


TITLE_PARENTHESES_PATTERN = re.compile(r"\s*([\(\[])([^)\]]+)([\)\]])")
TITLE_SUFFIX_NOISE_PATTERN = re.compile(
    r"\s*[-\u2013\u2014]\s*[^-\u2013\u2014]*\b(remaster(?:ed)?|remix|radio edit|edit|explicit|clean|"
    r"deluxe|bonus track|mono|stereo|version|anniversary|expanded edition)\b.*$",
    re.IGNORECASE,
)
TITLE_EXTRA_SUFFIX_NOISE_PATTERN = re.compile(
    r"\s*[-\u2013\u2014]\s*[^-\u2013\u2014]*\b(mix|single edit|single version|album version|"
    r"original mix|karaoke|with lyrics|official video)\b.*$",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")

TITLE_NOISE_TERMS = (
    "remaster",
    "remastered",
    "remix",
    "mix",
    "radio edit",
    "single edit",
    "single version",
    "album version",
    "edit",
    "explicit",
    "clean",
    "deluxe",
    "bonus track",
    "mono",
    "stereo",
    "version",
    "anniversary",
    "expanded edition",
    "original mix",
    "karaoke",
    "with lyrics",
    "official video",
)
TITLE_KEEP_TERMS = (
    "acoustic",
    "live",
    "unplugged",
    "demo",
    "instrumental",
)


def _clean_parenthetical(match: re.Match[str]) -> str:
    content = match.group(2).lower()
    if any(term in content for term in TITLE_KEEP_TERMS):
        return match.group(0)
    if any(term in content for term in TITLE_NOISE_TERMS):
        return ""
    return match.group(0)


def clean_title_text(value: Any) -> str:
    text = str(value or "").strip()
    text = TITLE_PARENTHESES_PATTERN.sub(_clean_parenthetical, text)
    text = TITLE_SUFFIX_NOISE_PATTERN.sub("", text)
    text = TITLE_EXTRA_SUFFIX_NOISE_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -\u2013\u2014_.,;:")


def release_decade_text(value: Any) -> str:
    text = str(value or "").strip()
    match = YEAR_PATTERN.search(text)
    if not match:
        return ""
    year = int(match.group(0))
    decade = year - (year % 10)
    return f"{decade}s"


def metadata_field_text(field: str, entity: Any) -> str:
    if isinstance(entity, list):
        entity = ", ".join(str(item) for item in entity)
    if field in {"track_name", "album_name"}:
        entity = clean_title_text(entity)
    return f"{field}: {entity}"


def weighted_metadata_lines(metadata: dict[str, Any], corpus_types: list[str]) -> list[str]:
    """Format metadata fields, repeating weighted fields as separate lines."""
    lines = []
    for corpus_type in corpus_types:
        entity = metadata_field(metadata, corpus_type)
        text = metadata_field_text(corpus_type, entity)
        lines.extend(text for _ in range(metadata_field_weight(corpus_type)))
    return lines


def attach_artist_style_profiles(
    metadata_dict: dict[str, dict[str, Any]],
    max_tags: int = ARTIST_STYLE_PROFILE_MAX_TAGS,
) -> dict[str, dict[str, Any]]:
    """Attach artist-level style tags and compact release decade metadata."""
    artist_tag_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for metadata in metadata_dict.values():
        key = artist_key(metadata.get("artist_name"))
        if not key:
            continue
        artist_tag_counts[key].update(normalized_music_tags(metadata.get("tag_list"), max_tags=80))

    artist_profiles = {
        key: [tag for tag, _ in counts.most_common(max_tags)]
        for key, counts in artist_tag_counts.items()
    }

    for metadata in metadata_dict.values():
        key = artist_key(metadata.get("artist_name"))
        metadata["artist_style_profile"] = artist_profiles.get(key, [])
        metadata["release_decade"] = release_decade_text(metadata.get("release_date"))

    return metadata_dict
