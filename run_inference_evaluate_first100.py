"""
Run the CRS model and evaluate it on the first 100 devset conversations.

The evaluation subset restricts conversation rows only. Retrieval still uses the
complete track catalog configured by the experiment YAML file.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from mcrs import load_crs_baseline


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATOR_ROOT = PROJECT_ROOT.parent / "music-crs-evaluator"
sys.path.insert(0, str(EVALUATOR_ROOT))

from metrics import (  # noqa: E402
    compute_catalog_diversity,
    compute_lexical_diversity,
    compute_recsys_metrics,
)


DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"
DATASET_SPLIT = "test"
TRACK_METADATA_NAME = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"


def chat_history_parser(
    conversations: List[Dict[str, Any]],
    music_crs: Any,
    target_turn_number: int,
) -> Tuple[List[Dict[str, str]], str]:
    """Return prior conversation messages and the current user request."""
    df_conversation = pd.DataFrame(conversations)
    df_history = df_conversation[df_conversation["turn_number"] < target_turn_number]
    chat_history = []

    for turn_data in df_history.to_dict(orient="records"):
        current_role = turn_data["role"]
        current_content = turn_data["content"]
        if current_role == "music":
            current_role = "assistant"
            current_content = music_crs.item_db.id_to_metadata(current_content)
        chat_history.append({"role": current_role, "content": current_content})

    current_turn = df_conversation[
        (df_conversation["turn_number"] == target_turn_number)
        & (df_conversation["role"] == "user")
    ]
    if current_turn.empty:
        raise ValueError(f"No user message found for turn {target_turn_number}.")
    return chat_history, current_turn.iloc[0]["content"]


def ground_truth_for_turn(
    conversations: List[Dict[str, Any]],
    target_turn_number: int,
) -> str:
    """Extract the ground-truth music track returned at one conversation turn."""
    matching_tracks = [
        message["content"]
        for message in conversations
        if message["turn_number"] == target_turn_number
        and message["role"] == "music"
    ]
    if not matching_tracks:
        raise ValueError(f"No ground-truth music item found for turn {target_turn_number}.")
    return matching_tracks[0]


def get_user_turn_numbers(conversations: List[Dict[str, Any]]) -> List[int]:
    """Return every turn for which the model must predict a recommendation."""
    return sorted(
        {
            message["turn_number"]
            for message in conversations
            if message["role"] == "user"
        }
    )


def load_model(
    config: Any,
    device_override: str | None,
    gemini_cache_dir_override: str | None,
) -> Any:
    """Instantiate the configured CRS model, allowing an optional CPU override."""
    device = device_override or config.device
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    gemini_cache_dir = (
        gemini_cache_dir_override
        or config.get("gemini_cache_dir", "./cache/gemini_expansions")
    )

    return load_crs_baseline(
        lm_type=config.lm_type,
        retrieval_type=config.retrieval_type,
        item_db_name=config.item_db_name,
        user_db_name=config.user_db_name,
        track_split_types=config.track_split_types,
        user_split_types=config.user_split_types,
        corpus_types=config.corpus_types,
        cache_dir=config.cache_dir,
        device=device,
        attn_implementation=config.attn_implementation,
        dtype=dtype,
        use_gemini_expansion=config.get("use_gemini_expansion", False),
        gemini_model_name=config.get(
            "gemini_model_name",
            "gemini-3.1-flash-lite-preview",
        ),
        gemini_cache_dir=gemini_cache_dir,
        gemini_expansion_mode=config.get("gemini_expansion_mode", "tag_query"),
        gemini_topk_per_reference=config.get("gemini_topk_per_reference", 50),
        gemini_rrf_k=config.get("gemini_rrf_k", 60),
        include_original_query_in_fusion=config.get(
            "include_original_query_in_fusion",
            False,
        ),
        original_query_weight=config.get("original_query_weight", 2.0),
        gemini_reference_weight=config.get("gemini_reference_weight", 1.0),
    )


def prepare_subset_data(
    subset: Any,
    music_crs: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build inference inputs, output metadata, and matching subset ground truth."""
    batch_data = []
    metadata = []
    ground_truth = []

    for item in subset:
        session_id = item["session_id"]
        user_id = item["user_id"]
        conversations = item["conversations"]

        for target_turn_number in get_user_turn_numbers(conversations):
            chat_history, user_query = chat_history_parser(
                conversations,
                music_crs,
                target_turn_number,
            )
            batch_data.append(
                {
                    "user_query": user_query,
                    "user_id": user_id,
                    "session_memory": chat_history,
                    "session_id": session_id,
                    "turn_number": target_turn_number,
                }
            )
            metadata.append(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "turn_number": target_turn_number,
                }
            )
            ground_truth.append(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "turn_number": target_turn_number,
                    "ground_truth_track_id": ground_truth_for_turn(
                        conversations,
                        target_turn_number,
                    ),
                }
            )

    return batch_data, metadata, ground_truth


def run_inference(
    music_crs: Any,
    batch_data: List[Dict[str, Any]],
    metadata: List[Dict[str, Any]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Generate track rankings and responses for each selected turn."""
    inference_results = []
    for start in tqdm(
        range(0, len(batch_data), batch_size),
        desc="Batch inference",
    ):
        batch = batch_data[start : start + batch_size]
        batch_metadata = metadata[start : start + batch_size]
        results = music_crs.batch_chat(batch)
        for result_metadata, result in zip(batch_metadata, results):
            inference_results.append(
                {
                    **result_metadata,
                    "predicted_track_ids": result["retrieval_items"],
                    "predicted_response": result["response"],
                }
            )
    return inference_results


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Evaluate subset predictions following the official local devset evaluator."""
    prediction_lookup = {
        (row["session_id"], row["turn_number"]): row
        for row in predictions
    }
    per_turn_results = []
    recommended_track_ids = []
    responses = []

    for truth in tqdm(ground_truth, desc="Evaluation"):
        key = (truth["session_id"], truth["turn_number"])
        prediction = prediction_lookup.get(key)
        if prediction is None:
            raise ValueError(f"Missing prediction for session/turn: {key}")

        track_ids = prediction["predicted_track_ids"]
        metrics = compute_recsys_metrics(
            track_ids,
            [truth["ground_truth_track_id"]],
            [1, 10, 20],
        )
        per_turn_results.append({"turn_number": truth["turn_number"], **metrics})
        recommended_track_ids.extend(track_ids)
        responses.append(prediction["predicted_response"])

    df_results = pd.DataFrame(per_turn_results)
    macro_scores = (
        df_results.groupby("turn_number").mean(numeric_only=True).mean(axis=0).to_dict()
    )

    music_catalog = load_dataset(TRACK_METADATA_NAME, split="all_tracks")
    total_catalog_size = len(music_catalog)
    macro_scores["catalog_diversity"] = compute_catalog_diversity(
        recommended_track_ids,
        total_catalog_size,
    )
    macro_scores["lexical_diversity"] = compute_lexical_diversity(responses)
    macro_scores["total_catalog_size"] = total_catalog_size
    macro_scores["subset_conversations"] = len(
        {row["session_id"] for row in ground_truth}
    )
    macro_scores["subset_turns"] = len(ground_truth)
    return macro_scores


def main(args: argparse.Namespace) -> None:
    config = OmegaConf.load(PROJECT_ROOT / "config" / f"{args.tid}.yaml")
    music_crs = load_model(config, args.device, args.gemini_cache_dir)

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    number_of_rows = min(args.num_rows, len(dataset))
    subset = dataset.select(range(number_of_rows))

    batch_data, metadata, ground_truth = prepare_subset_data(subset, music_crs)
    predictions = run_inference(
        music_crs,
        batch_data,
        metadata,
        args.batch_size,
    )
    scores = evaluate_predictions(predictions, ground_truth)

    output_dir = PROJECT_ROOT / args.output_dir / args.tid
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    with (output_dir / "ground_truth.json").open("w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    with (output_dir / "scores.json").open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"Evaluated {number_of_rows} conversations and {len(ground_truth)} turns.")
    print(f"Outputs saved to {output_dir}")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run per-turn inference and local evaluation on the first rows of "
            "TalkPlayData-Challenge-Dataset."
        )
    )
    parser.add_argument(
        "--tid",
        type=str,
        default="my_model",
        help="Configuration filename in config/ without the .yaml extension.",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=50,
        help="Number of conversation rows to evaluate from the start of the test split.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of turns passed to the model at once.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Optional override for the device in the model configuration.",
    )
    parser.add_argument(
        "--gemini_cache_dir",
        type=str,
        default="./cache/gemini_expansions_devset_first100",
        help=(
            "Cache folder for Gemini expansion calls. The default is separate "
            "from the Blind-A inference cache."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exp/first_100",
        help="Output folder relative to this repository.",
    )
    main(parser.parse_args())
