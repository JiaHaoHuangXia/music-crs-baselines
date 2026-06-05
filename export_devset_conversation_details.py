"""
Export devset conversation context and saved model responses for Streamlit.

This joins a completed first-rows evaluation run with the original public
conversation dataset. It does not run inference or call Gemini.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "exp" / "first_100" / "blindset_gemini_bert"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "visualize_streamlit" / "devset_conversation_details.json"
)
CONVERSATION_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"
CONVERSATION_SPLIT = "test"
TRACK_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
TRACK_SPLIT = "all_tracks"


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def field_to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def format_track(track_id: str, catalog: dict[str, dict[str, Any]]) -> dict[str, str]:
    metadata = catalog.get(track_id, {})
    track_name = field_to_string(metadata.get("track_name", ""))
    artist_name = field_to_string(metadata.get("artist_name", ""))
    album_name = field_to_string(metadata.get("album_name", ""))
    label = " - ".join(value for value in [track_name, artist_name] if value)
    return {
        "track_id": track_id,
        "track_name": track_name,
        "artist_name": artist_name,
        "album_name": album_name,
        "label": label or track_id,
    }


def history_before_turn(
    conversations: list[dict[str, Any]],
    target_turn_number: int,
    catalog: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """Return only messages that were available before the target prediction."""
    messages = []
    for message in conversations:
        if int(message["turn_number"]) >= target_turn_number:
            continue
        if message["role"] == "music":
            track = format_track(message["content"], catalog)
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Recommended track: {track['label']}",
                }
            )
        else:
            messages.append(
                {
                    "role": message["role"],
                    "content": str(message["content"]),
                }
            )
    return messages


def current_user_request(
    conversations: list[dict[str, Any]],
    target_turn_number: int,
) -> str:
    for message in conversations:
        if (
            int(message["turn_number"]) == target_turn_number
            and message["role"] == "user"
        ):
            return str(message["content"])
    raise ValueError(f"No user request found for turn {target_turn_number}.")


def export_details(run_dir: Path, output_path: Path) -> None:
    predictions_path = run_dir / "predictions.json"
    ground_truth_path = run_dir / "ground_truth.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {ground_truth_path}")

    predictions = read_json(predictions_path)
    ground_truth = read_json(ground_truth_path)
    evaluated_keys = {
        (row["session_id"], int(row["turn_number"])) for row in ground_truth
    }
    prediction_lookup = {
        (row["session_id"], int(row["turn_number"])): row for row in predictions
    }
    ground_truth_lookup = {
        (row["session_id"], int(row["turn_number"])): row for row in ground_truth
    }

    conversations = load_dataset(CONVERSATION_DATASET, split=CONVERSATION_SPLIT)
    conversation_lookup = {
        row["session_id"]: row
        for row in conversations
        if any(key[0] == row["session_id"] for key in evaluated_keys)
    }

    track_dataset = load_dataset(TRACK_DATASET, split=TRACK_SPLIT)
    catalog = {row["track_id"]: row for row in track_dataset}

    details = []
    for key in sorted(evaluated_keys):
        session_id, turn_number = key
        source_row = conversation_lookup.get(session_id)
        if source_row is None:
            raise ValueError(f"Conversation not found for evaluated session: {session_id}")
        prediction = prediction_lookup.get(key)
        if prediction is None:
            raise ValueError(f"Prediction not found for evaluated turn: {key}")

        truth = ground_truth_lookup[key]
        truth_track = format_track(truth["ground_truth_track_id"], catalog)
        predicted_tracks = [
            format_track(track_id, catalog)
            for track_id in prediction["predicted_track_ids"]
        ]

        details.append(
            {
                "session_id": session_id,
                "user_id": truth.get("user_id", source_row.get("user_id", "")),
                "turn_number": turn_number,
                "conversation_history": history_before_turn(
                    source_row["conversations"],
                    turn_number,
                    catalog,
                ),
                "current_user_request": current_user_request(
                    source_row["conversations"],
                    turn_number,
                ),
                "ground_truth_track": truth_track,
                "predicted_tracks": predicted_tracks,
                "predicted_response": prediction.get("predicted_response", ""),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(details, file, ensure_ascii=False, indent=2)

    print(f"Exported {len(details)} conversation-turn records to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export completed devset-run conversations and responses for Streamlit."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Directory containing predictions.json and ground_truth.json.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the exported Streamlit JSON artifact.",
    )
    args = parser.parse_args()
    export_details(args.run_dir, args.output_json)


if __name__ == "__main__":
    main()
