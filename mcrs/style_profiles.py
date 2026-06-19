"""Artist style profile helpers for metadata retrieval."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from mcrs.controlled_tags import normalize_tag, normalized_music_tags


def artist_key(value: Any) -> str:
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    return normalize_tag(value)


def attach_artist_style_profiles(
    metadata_dict: dict[str, dict[str, Any]],
    max_tags: int = 8,
) -> dict[str, dict[str, Any]]:
    """Attach each track's artist-level style tags, derived from catalog tags."""
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

    return metadata_dict
