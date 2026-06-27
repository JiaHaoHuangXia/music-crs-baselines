"""Export MiniLM reference-track retrieval details for the Streamlit dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mcrs.retrieval_modules import load_retrieval_module
from mcrs.style_profiles import release_decade_text, weighted_metadata_lines


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "visualize_streamlit" / "minilm_reference_retrieval_table.csv"


def read_json(path: Path) -> Any:
    """Load a JSON file using the project default encoding."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def field_to_string(value: Any) -> str:
    """Format scalar or list metadata fields for Streamlit tables."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def clean_list_field(value: Any) -> list[str]:
    """Normalize Gemini fields that may arrive as strings or lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    value = str(value).strip()
    return [value] if value else []


def parse_session_turn(path: Path) -> tuple[str, int | None]:
    """Parse `<session_id>_turn_<number>.json` cache filenames."""
    name = path.stem
    if "_turn_" not in name:
        return name, None
    session_id, turn_text = name.rsplit("_turn_", 1)
    try:
        return session_id, int(turn_text)
    except ValueError:
        return session_id, None


def load_gemini_cache_file(path: Path) -> list[dict[str, Any]]:
    """Load the five Gemini reference tracks cached for one conversation turn."""
    data = read_json(path)
    if isinstance(data, dict):
        for key in ["recommendations", "songs", "tracks", "reference_songs", "reference_tracks", "items"]:
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Invalid Gemini cache format: {path}")

    rows = []
    for item in data[:5]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "track_name": clean_list_field(item.get("track_name"))[:1] or ["Unknown Track"],
                "artist_name": clean_list_field(item.get("artist_name"))[:1] or ["Unknown Artist"],
                "album_name": clean_list_field(item.get("album_name"))[:1] or ["Unknown Album"],
                "tag_list": clean_list_field(item.get("tag_list")),
                "release_date": field_to_string(item.get("release_date")),
            }
        )
    return rows


def pseudo_track_to_text(track: dict[str, Any], corpus_types: list[str]) -> str:
    """Convert a Gemini reference track into the MiniLM metadata text format."""
    values = {
        "track_name": field_to_string(track.get("track_name")),
        "artist_name": field_to_string(track.get("artist_name")),
        "album_name": field_to_string(track.get("album_name")),
        "tag_list": field_to_string(track.get("tag_list")),
        "artist_style_profile": clean_list_field(track.get("tag_list"))[:8],
        "release_date": field_to_string(track.get("release_date")),
        "release_decade": release_decade_text(track.get("release_date")),
    }
    return "\n".join(weighted_metadata_lines(values, corpus_types))


def format_track(track_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the catalog fields shown for final, ground-truth, and retrieved tracks."""
    return {
        "track_id": track_id,
        "track_name": field_to_string(metadata.get("track_name")),
        "artist_name": field_to_string(metadata.get("artist_name")),
        "album_name": field_to_string(metadata.get("album_name")),
        "tag_list": field_to_string(metadata.get("tag_list")),
        "release_decade": metadata.get("release_decade") or release_decade_text(metadata.get("release_date")),
    }


def format_reference_track(track: dict[str, Any]) -> dict[str, Any]:
    """Return normalized Gemini reference-track fields for the dashboard."""
    return {
        "track_id": "",
        "track_name": field_to_string(track.get("track_name")),
        "artist_name": field_to_string(track.get("artist_name")),
        "album_name": field_to_string(track.get("album_name")),
        "tag_list": field_to_string(track.get("tag_list")),
        "release_decade": release_decade_text(track.get("release_date")),
    }


def export_minilm_explanations(
    run_dir: Path,
    gemini_cache_dir: Path,
    output_path: Path,
    corpus_types: list[str],
    topk_retrieved_per_reference: int,
    device: str | None,
) -> None:
    """Write one CSV with MiniLM final rankings, references, and per-reference retrievals."""
    predictions = read_json(run_dir / "predictions.json")
    ground_truth = read_json(run_dir / "ground_truth.json")
    truth_lookup = {
        (row["session_id"], int(row["turn_number"])): row["ground_truth_track_id"]
        for row in ground_truth
    }
    prediction_lookup = {
        (row["session_id"], int(row["turn_number"])): row["predicted_track_ids"]
        for row in predictions
    }

    retrieval = load_retrieval_module(
        "sentence_transformer",
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        ["all_tracks"],
        corpus_types,
        "./cache",
    )
    if device:
        retrieval.device = device

    rows = []
    for cache_path in sorted(gemini_cache_dir.glob("*.json")):
        if cache_path.stem.endswith("_keywords"):
            continue
        session_id, turn_number = parse_session_turn(cache_path)
        if turn_number is None:
            continue
        key = (session_id, int(turn_number))
        if key not in truth_lookup or key not in prediction_lookup:
            continue

        ground_truth_id = truth_lookup[key]
        ground_truth_metadata = retrieval.metadata_dict.get(ground_truth_id, {})
        final_track_ids = prediction_lookup[key]
        for rank, track_id in enumerate(final_track_ids, start=1):
            metadata = retrieval.metadata_dict.get(track_id, {})
            rows.append(
                {
                    "row_type": "final_recommendation",
                    "session_id": session_id,
                    "turn_number": turn_number,
                    "rank": rank,
                    "gemini_reference_rank": "",
                    "cosine_similarity": "",
                    "is_ground_truth": track_id == ground_truth_id,
                    **format_track(track_id, metadata),
                    "ground_truth_track_id": ground_truth_id,
                    "ground_truth_track_name": field_to_string(ground_truth_metadata.get("track_name")),
                }
            )

        rows.append(
            {
                "row_type": "ground_truth",
                "session_id": session_id,
                "turn_number": turn_number,
                "rank": "",
                "gemini_reference_rank": "",
                "cosine_similarity": "",
                "is_ground_truth": True,
                **format_track(ground_truth_id, ground_truth_metadata),
                "ground_truth_track_id": ground_truth_id,
                "ground_truth_track_name": field_to_string(ground_truth_metadata.get("track_name")),
            }
        )

        try:
            reference_tracks = load_gemini_cache_file(cache_path)
        except Exception as error:
            print(f"Skipping {cache_path.name}: {error}")
            continue

        for reference_rank, reference_track in enumerate(reference_tracks[:5], start=1):
            reference_row = format_reference_track(reference_track)
            rows.append(
                {
                    "row_type": "gemini_reference",
                    "session_id": session_id,
                    "turn_number": turn_number,
                    "rank": reference_rank,
                    "gemini_reference_rank": reference_rank,
                    "cosine_similarity": "",
                    "is_ground_truth": False,
                    **reference_row,
                    "ground_truth_track_id": ground_truth_id,
                    "ground_truth_track_name": field_to_string(ground_truth_metadata.get("track_name")),
                }
            )

            query = pseudo_track_to_text(reference_track, corpus_types)
            retrieved = retrieval.text_to_item_retrieval_with_scores(
                query,
                topk=topk_retrieved_per_reference,
            )
            for retrieved_rank, (track_id, score) in enumerate(retrieved, start=1):
                metadata = retrieval.metadata_dict.get(track_id, {})
                rows.append(
                    {
                        "row_type": "retrieved_from_reference",
                        "session_id": session_id,
                        "turn_number": turn_number,
                        "rank": retrieved_rank,
                        "gemini_reference_rank": reference_rank,
                        "cosine_similarity": score,
                        "is_ground_truth": track_id == ground_truth_id,
                        **format_track(track_id, metadata),
                        "ground_truth_track_id": ground_truth_id,
                        "ground_truth_track_name": field_to_string(ground_truth_metadata.get("track_name")),
                    }
                )

    if not rows:
        raise ValueError("No MiniLM explanation rows were exported. Check run_dir and Gemini cache.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    print(f"Exported {len(rows)} MiniLM explanation rows to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MiniLM explanation rows for Streamlit.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--gemini_cache_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--topk_retrieved_per_reference", type=int, default=20)
    parser.add_argument(
        "--corpus_types",
        nargs="+",
        default=["track_name", "artist_name", "album_name", "artist_style_profile", "release_decade"],
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    cache_dir = (
        args.gemini_cache_dir
        if args.gemini_cache_dir.is_absolute()
        else PROJECT_ROOT / args.gemini_cache_dir
    )
    output_path = args.output_csv if args.output_csv.is_absolute() else PROJECT_ROOT / args.output_csv
    export_minilm_explanations(
        run_dir=run_dir,
        gemini_cache_dir=cache_dir,
        output_path=output_path,
        corpus_types=args.corpus_types,
        topk_retrieved_per_reference=args.topk_retrieved_per_reference,
        device=args.device,
    )


if __name__ == "__main__":
    main()
