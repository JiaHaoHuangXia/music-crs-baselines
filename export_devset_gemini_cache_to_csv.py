"""
Export cached Gemini reference tracks with their devset ground-truth tracks.

This script does not compute embeddings. It converts the raw Gemini cache files
created during the first-rows devset evaluation into one CSV row per Gemini
reference track, joined to the known target track for the same session and turn.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache" / "gemini_expansions_devset_first100"
DEFAULT_GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "exp" / "first_100" / "blindset_gemini_bert" / "ground_truth.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "visualize_streamlit" / "devset_gemini_ground_truth_table.csv"
)
TRACK_DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
TRACK_DATASET_SPLIT = "all_tracks"


def to_list_of_strings(value: Any, default: str = "") -> list[str]:
    """Normalize Gemini list-like fields into a clean list of strings."""
    if value is None:
        return [default] if default else []
    if isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
        return values or ([default] if default else [])
    value = str(value).strip()
    return [value] if value else ([default] if default else [])


def field_to_string(value: Any) -> str:
    """Format dataset or Gemini metadata consistently in the output CSV."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def normalize_reference_tracks(raw_data: Any) -> list[dict[str, Any]]:
    """Read supported Gemini JSON formats into a list of reference-track objects."""
    if isinstance(raw_data, dict):
        for key in [
            "recommendations",
            "songs",
            "tracks",
            "reference_songs",
            "reference_tracks",
            "items",
        ]:
            if isinstance(raw_data.get(key), list):
                raw_data = raw_data[key]
                break

    if not isinstance(raw_data, list):
        raise ValueError("Gemini cache content is not a list of tracks.")

    tracks = []
    for raw_track in raw_data:
        if not isinstance(raw_track, dict):
            continue
        tracks.append(
            {
                "track_name": to_list_of_strings(
                    raw_track.get("track_name"), "Unknown Track"
                )[:1],
                "artist_name": to_list_of_strings(
                    raw_track.get("artist_name"), "Unknown Artist"
                )[:1],
                "album_name": to_list_of_strings(
                    raw_track.get("album_name"), "Unknown Album"
                )[:1],
                "tag_list": to_list_of_strings(raw_track.get("tag_list")),
                "release_date": field_to_string(raw_track.get("release_date")),
            }
        )
    return tracks


def parse_cache_filename(cache_path: Path) -> tuple[str, int]:
    """Parse `<session_id>_turn_<number>.json` cache filenames."""
    if "_turn_" not in cache_path.stem:
        raise ValueError(f"Unexpected cache filename: {cache_path.name}")
    session_id, turn_text = cache_path.stem.rsplit("_turn_", 1)
    return session_id, int(turn_text)


def load_ground_truth(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load evaluated session/turn targets into a lookup table."""
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    return {
        (row["session_id"], int(row["turn_number"])): row
        for row in rows
    }


def load_track_metadata() -> dict[str, dict[str, Any]]:
    """Load real catalog metadata keyed by track ID."""
    dataset = load_dataset(TRACK_DATASET_NAME, split=TRACK_DATASET_SPLIT)
    return {track["track_id"]: track for track in dataset}


def get_tag_overlap(reference_tags: list[str], ground_truth_tags: Any) -> tuple[int, str]:
    """Return exact lowercased tag overlap between a reference and the target track."""
    reference_set = {tag.strip().lower() for tag in reference_tags if tag.strip()}
    ground_truth_set = {
        tag.strip().lower()
        for tag in to_list_of_strings(ground_truth_tags)
        if tag.strip()
    }
    overlap = sorted(reference_set.intersection(ground_truth_set))
    return len(overlap), ", ".join(overlap)


def export_cache(
    cache_dir: Path,
    ground_truth_path: Path,
    output_path: Path,
) -> None:
    """Export matching cache entries and ground-truth metadata to a CSV."""
    if not cache_dir.exists():
        raise FileNotFoundError(f"Gemini cache directory not found: {cache_dir}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground-truth JSON not found: {ground_truth_path}")

    ground_truth = load_ground_truth(ground_truth_path)
    catalog_metadata = load_track_metadata()
    rows = []
    cache_files_used = 0
    skipped_cache_files = 0

    for cache_path in sorted(cache_dir.glob("*.json")):
        session_id, turn_number = parse_cache_filename(cache_path)
        truth = ground_truth.get((session_id, turn_number))
        if truth is None:
            skipped_cache_files += 1
            continue

        ground_truth_track_id = truth["ground_truth_track_id"]
        ground_truth_track = catalog_metadata.get(ground_truth_track_id)
        if ground_truth_track is None:
            raise ValueError(
                f"Track metadata not found for ground truth ID: {ground_truth_track_id}"
            )

        with cache_path.open("r", encoding="utf-8") as file:
            reference_tracks = normalize_reference_tracks(json.load(file))

        cache_files_used += 1
        for reference_rank, reference_track in enumerate(reference_tracks, start=1):
            overlap_count, overlap_terms = get_tag_overlap(
                reference_track["tag_list"],
                ground_truth_track.get("tag_list"),
            )
            rows.append(
                {
                    "cache_file": cache_path.name,
                    "session_id": session_id,
                    "user_id": truth.get("user_id", ""),
                    "turn_number": turn_number,
                    "gemini_reference_rank": reference_rank,
                    "gemini_track_name": field_to_string(reference_track["track_name"]),
                    "gemini_artist_name": field_to_string(reference_track["artist_name"]),
                    "gemini_album_name": field_to_string(reference_track["album_name"]),
                    "gemini_tag_list": field_to_string(reference_track["tag_list"]),
                    "gemini_release_date": reference_track["release_date"],
                    "ground_truth_track_id": ground_truth_track_id,
                    "ground_truth_track_name": field_to_string(
                        ground_truth_track.get("track_name")
                    ),
                    "ground_truth_artist_name": field_to_string(
                        ground_truth_track.get("artist_name")
                    ),
                    "ground_truth_album_name": field_to_string(
                        ground_truth_track.get("album_name")
                    ),
                    "ground_truth_tag_list": field_to_string(
                        ground_truth_track.get("tag_list")
                    ),
                    "ground_truth_release_date": field_to_string(
                        ground_truth_track.get("release_date")
                    ),
                    "tag_overlap_count": overlap_count,
                    "tag_overlap_terms": overlap_terms,
                }
            )

    if not rows:
        raise ValueError(
            "No cache entries matched the provided ground-truth file. "
            "Check that the cache and ground truth came from the same evaluation run."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")

    print(f"Exported {len(rows)} Gemini reference rows to: {output_path}")
    print(f"Used {cache_files_used} cached conversation turns.")
    if skipped_cache_files:
        print(
            f"Skipped {skipped_cache_files} cache files not included in the "
            "provided ground-truth subset."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert devset Gemini cache files into a CSV joined with known "
            "ground-truth catalog tracks."
        )
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory containing `<session_id>_turn_<number>.json` cache files.",
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_PATH,
        help="Ground-truth JSON generated by the first-rows evaluation runner.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the exported CSV.",
    )
    args = parser.parse_args()

    export_cache(args.cache_dir, args.ground_truth, args.output_csv)


if __name__ == "__main__":
    main()
