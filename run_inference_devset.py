"""
Batch inference script for Music CRS.
"""

import os
import json
import argparse
from mcrs import load_crs_baseline
from datasets import load_dataset
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import pandas as pd
from omegaconf import OmegaConf
import shutil

def format_context(value):
    if not value:
        return ""
    if isinstance(value, dict):
        return " ".join(
            f"{key}: {format_context(inner_value)}"
            for key, inner_value in sorted(value.items())
            if format_context(inner_value)
        )
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()

def chat_history_parser(conversations, music_crs, target_turn_number):
    """
    Parse conversation history up to a target turn.

    Args:
        conversations (List[Dict]): List of conversation turn dictionaries containing:
            - turn_number: Turn index (1-8)
            - role: Speaker role ('user', 'assistant', 'music')
            - content: Message content or track ID
        music_crs: CRS baseline instance (used to convert track IDs to metadata)
        target_turn_number (int): The turn to predict (history excludes this turn)

    Returns:
        Tuple[List[Dict], str]:
            - chat_history: List of previous messages formatted as [{"role": ..., "content": ...}]
            - user_query: The user query at the target turn
    """
    df_conversation = pd.DataFrame(conversations)
    df_history = df_conversation[df_conversation['turn_number'] < target_turn_number]
    chat_history = []
    for turn_data in df_history.to_dict(orient="records"):
        turn_number = turn_data['turn_number']
        current_role = turn_data['role']
        current_content = turn_data['content']
        if turn_data['role'] == "music":
            current_role = "assistant"
            current_content = music_crs.item_db.id_to_metadata(turn_data['content'])
        chat_history.append({
            "role": current_role,
            "content": current_content
        })
    df_current_turn = df_conversation[df_conversation['turn_number'] == target_turn_number]
    user_query = df_current_turn.iloc[0]['content']
    return chat_history, user_query

def main(args):
    """
    Run batch inference on TalkPlayData-2 test dataset.

    Args:
        args: Namespace object containing:
            - tid (str): Task/configuration identifier
            - batch_size (int): Batch size for inference
            - save_path (str): Output directory (currently unused)

    Returns:
        None. Results are saved to exp/inference/{tid}.json

    Processing:
        - Loads model configuration from config/{tid}.yaml
        - Processes all sessions × 8 turns in batches
        - Tracks progress with tqdm progress bar
        - Saves comprehensive results for evaluation
    """
    print("Preparing retrieval cache...")
    #os.system("rm -rf cache")
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    if config.get("clear_cache", False) and os.path.exists("cache"):
        shutil.rmtree("cache")
    music_crs = load_crs_baseline(
        lm_type=config.lm_type,
        retrieval_type=config.retrieval_type,
        item_db_name=config.item_db_name,
        user_db_name=config.user_db_name,
        track_split_types=config.track_split_types,
        user_split_types=config.user_split_types,
        corpus_types=config.corpus_types,
        cache_dir=config.cache_dir,
        device=config.device,
        attn_implementation=config.attn_implementation,
        dtype=torch.bfloat16,
        use_gemini_expansion=config.get("use_gemini_expansion", False),
        gemini_model_name=config.get("gemini_model_name", "gemini-3.1-flash-lite"),
        gemini_cache_dir=config.get("gemini_cache_dir", "./cache/gemini_expansions"),
        gemini_expansion_mode=config.get("gemini_expansion_mode", "tag_query"),
        gemini_topk_per_reference=config.get("gemini_topk_per_reference", 50),
        gemini_rrf_k=config.get("gemini_rrf_k", 60),
        include_original_query_in_fusion=config.get(
            "include_original_query_in_fusion",
            False,
        ),
        original_query_weight=config.get("original_query_weight", 2.0),
        gemini_reference_weight=config.get("gemini_reference_weight", 1.0),
        gemini_max_reference_tracks=config.get("gemini_max_reference_tracks", None),
        gemini_fusion_method=config.get("gemini_fusion_method", "rrf"),
    )
    db = load_dataset(config.test_dataset_name, split="test")
    # Prepare all batch data at once
    batch_data, metadata = [], []
    for item in db:
        user_id = item['user_id']
        session_id = item['session_id']
        for target_turn_number in range(1, 9):
            chat_history, user_query = chat_history_parser(item['conversations'], music_crs, target_turn_number)
            batch_data.append({
                'user_query': user_query,
                'user_id': user_id,
                'session_memory': chat_history,
                'session_id': session_id,
                'turn_number': target_turn_number,
                'conversation_goal': format_context(item.get('conversation_goal')),
                'user_profile': format_context(item.get('user_profile')),
            })
            metadata.append({
                'session_id': session_id,
                'user_id': user_id,
                'turn_number': target_turn_number
            })
    inference_results = []
    for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Batch inference"):
        batch = batch_data[i:i+args.batch_size]
        batch_metadata = metadata[i:i+args.batch_size]
        results = music_crs.batch_chat(batch)
        for j, result in enumerate(results):
            inference_results.append({
                "session_id": batch_metadata[j]['session_id'],
                "user_id": batch_metadata[j]['user_id'],
                "turn_number": batch_metadata[j]['turn_number'],
                "predicted_track_ids": result['retrieval_items'],
                "predicted_response": result["response"]
            })
    os.makedirs("exp/inference/devset", exist_ok=True)
    with open(f"exp/inference/devset/{args.tid}.json", "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch inference on TalkPlayData-2 test dataset for Music CRS evaluation."
    )
    parser.add_argument(
        "--tid",
        type=str,
        default="llama1b_bm25_testset",
        help="Task identifier matching a config file (e.g., 'llama1b_bm25' loads config/llama1b_bm25.yaml)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of queries to process in parallel. Reduce if encountering GPU memory issues."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./exp/inference",
        help="Base directory for saving results (currently not used, results saved to exp/inference/)"
    )
    args = parser.parse_args()
    main(args)
