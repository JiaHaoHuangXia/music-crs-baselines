"""Export BM25-style explanation rows for the Streamlit dashboard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from mcrs.controlled_tags import normalize_music_tag, normalize_tag
from mcrs.style_profiles import release_decade_text


PROJECT_ROOT = Path(__file__).resolve().parent
TRACK_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
TRACK_SPLIT = "all_tracks"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "visualize_streamlit" / "bm25_explanation_table.csv"

KEYWORD_FIELDS = [
    "track_titles",
    "artists",
    "albums",
    "genres",
    "moods",
    "instruments",
    "themes",
    "era",
    "must_include_terms",
    "avoid_terms",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def field_to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def clean_terms(values: Any, normalize_music: bool = False) -> list[str]:
    if not isinstance(values, list):
        values = [] if values is None else [values]
    terms = []
    seen = set()
    for value in values:
        term = normalize_music_tag(value) if normalize_music else normalize_tag(value)
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def cache_path(cache_dir: Path, session_id: str, turn_number: int) -> Path:
    safe_session = str(session_id).replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe_session}_turn_{turn_number}_keywords.json"


def load_keywords(cache_dir: Path, session_id: str, turn_number: int) -> dict[str, list[str]]:
    path = cache_path(cache_dir, session_id, turn_number)
    if not path.exists():
        return {field: [] for field in KEYWORD_FIELDS}
    payload = read_json(path)
    return {
        field: [str(item).strip() for item in payload.get(field, []) if str(item).strip()]
        for field in KEYWORD_FIELDS
    }


def weighted_query_text(keywords: dict[str, list[str]]) -> str:
    weights = {
        "track_titles": 6,
        "artists": 6,
        "albums": 4,
        "genres": 1,
        "moods": 1,
        "instruments": 1,
        "themes": 1,
        "era": 1,
        "must_include_terms": 2,
    }
    lines = []
    for field, weight in weights.items():
        values = keywords.get(field, [])
        if values:
            lines.extend([f"{field}: {', '.join(values)}"] * weight)
    if keywords.get("avoid_terms"):
        lines.append(f"avoid_terms: {', '.join(keywords['avoid_terms'])}")
    return "\n".join(lines)


def contains_phrase(text: str, term: str) -> bool:
    term = normalize_tag(term)
    if not term:
        return False
    normalized_text = normalize_tag(text)
    if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized_text):
        return True
    music_text = normalize_music_tag(text)
    music_term = normalize_music_tag(term)
    return re.search(rf"(?<!\w){re.escape(music_term)}(?!\w)", music_text) is not None


def matched_terms_by_field(metadata: dict[str, Any], keywords: dict[str, list[str]]) -> dict[str, list[str]]:
    fields = {
        "track_name": field_to_string(metadata.get("track_name")),
        "artist_name": field_to_string(metadata.get("artist_name")),
        "album_name": field_to_string(metadata.get("album_name")),
        "tag_list": field_to_string(metadata.get("tag_list")),
        "release_decade": metadata.get("release_decade") or release_decade_text(metadata.get("release_date")),
    }
    terms = []
    for field in KEYWORD_FIELDS:
        normalize_music = field not in {"track_titles", "artists", "albums", "era"}
        terms.extend(clean_terms(keywords.get(field), normalize_music=normalize_music))
    terms = list(dict.fromkeys(terms))

    matches = {}
    for field, text in fields.items():
        matches[field] = [term for term in terms if contains_phrase(text, term)]
    return matches


def route_types(keywords: dict[str, list[str]], current_request: str) -> list[str]:
    text = normalize_tag(current_request)
    routes = []
    if keywords.get("track_titles") and any(contains_phrase(text, value) for value in keywords["track_titles"]):
        routes.append("title")
    if keywords.get("artists") and any(contains_phrase(text, value) for value in keywords["artists"]):
        routes.append("artist")
    if keywords.get("albums") and any(contains_phrase(text, value) for value in keywords["albums"]):
        routes.append("album")
    if keywords.get("era") and re.search(r"\b(19[5-9]0s|20[0-2]0s|50s|60s|70s|80s|90s|00s|2000s|2010s|10s)\b", text):
        routes.append("decade")
    if keywords.get("avoid_terms") and any(phrase in text for phrase in ["not ", "no ", "without ", "less ", "avoid "]):
        routes.append("negative")
    if any(phrase in text for phrase in ["same artist", "same band", "by them", "from them", "another one by", "more by"]):
        routes.append("same_artist")
    return routes


def current_user_request(conversations: list[dict[str, Any]], turn_number: int) -> str:
    for message in conversations:
        if int(message["turn_number"]) == turn_number and message["role"] == "user":
            return str(message["content"])
    return ""


def export_bm25_explanations(run_dir: Path, gemini_cache_dir: Path, output_path: Path) -> None:
    predictions = read_json(run_dir / "predictions.json")
    ground_truth = read_json(run_dir / "ground_truth.json")
    truth_lookup = {
        (row["session_id"], int(row["turn_number"])): row["ground_truth_track_id"]
        for row in ground_truth
    }

    track_dataset = load_dataset(TRACK_DATASET, split=TRACK_SPLIT)
    catalog = {row["track_id"]: row for row in track_dataset}
    for metadata in catalog.values():
        metadata["release_decade"] = release_decade_text(metadata.get("release_date"))

    conversations = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    conversation_lookup = {row["session_id"]: row["conversations"] for row in conversations}

    rows = []
    for prediction in predictions:
        session_id = prediction["session_id"]
        turn_number = int(prediction["turn_number"])
        key = (session_id, turn_number)
        ground_truth_id = truth_lookup.get(key)
        keywords = load_keywords(gemini_cache_dir, session_id, turn_number)
        request = current_user_request(conversation_lookup.get(session_id, []), turn_number)
        routes = route_types(keywords, request)
        query_text = weighted_query_text(keywords)

        for rank, track_id in enumerate(prediction["predicted_track_ids"], start=1):
            metadata = catalog.get(track_id, {})
            matches = matched_terms_by_field(metadata, keywords)
            matched_fields = [field for field, terms in matches.items() if terms]
            matched_terms = sorted({term for terms in matches.values() for term in terms})
            rows.append(
                {
                    "session_id": session_id,
                    "turn_number": turn_number,
                    "rank": rank,
                    "track_id": track_id,
                    "track_name": field_to_string(metadata.get("track_name")),
                    "artist_name": field_to_string(metadata.get("artist_name")),
                    "album_name": field_to_string(metadata.get("album_name")),
                    "tag_list": field_to_string(metadata.get("tag_list")),
                    "release_decade": metadata.get("release_decade", ""),
                    "is_ground_truth": track_id == ground_truth_id,
                    "ground_truth_track_id": ground_truth_id,
                    "current_user_request": request,
                    "route_types": ", ".join(routes) if routes else "none",
                    "keyword_json": json.dumps(keywords, ensure_ascii=False),
                    "final_bm25_query": query_text,
                    "matched_fields": ", ".join(matched_fields),
                    "matched_terms": ", ".join(matched_terms),
                    "matched_term_count": len(matched_terms),
                    "track_name_matches": ", ".join(matches["track_name"]),
                    "artist_name_matches": ", ".join(matches["artist_name"]),
                    "album_name_matches": ", ".join(matches["album_name"]),
                    "tag_list_matches": ", ".join(matches["tag_list"]),
                    "release_decade_matches": ", ".join(matches["release_decade"]),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Exported {len(rows)} BM25 explanation rows to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BM25 explanation rows for Streamlit.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--gemini_cache_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    cache_dir = (
        args.gemini_cache_dir
        if args.gemini_cache_dir.is_absolute()
        else PROJECT_ROOT / args.gemini_cache_dir
    )
    output_path = args.output_csv if args.output_csv.is_absolute() else PROJECT_ROOT / args.output_csv
    export_bm25_explanations(run_dir, cache_dir, output_path)


if __name__ == "__main__":
    main()
