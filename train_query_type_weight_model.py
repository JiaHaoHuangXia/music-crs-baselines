"""Train learned weights for the query-type BM25 reranker.

This keeps the existing retrieval pipeline intact:
conversation + Gemini keywords -> BM25 top-k candidates -> query-type features.
The trained logistic regression coefficients replace the hand-tuned reranker weights.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from run_inference_evaluate_first100 import (
    DATASET_NAME,
    DATASET_SPLIT,
    load_model,
    prepare_subset_data,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def build_retrieval_input(music_crs: Any, row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    session_memory = row["session_memory"].copy()
    session_memory.append({"role": "user", "content": row["user_query"]})
    conversation_text = "\n".join(
        f"{message['role']}: {message['content']}" for message in session_memory
    )

    retrieval_input = conversation_text
    if music_crs.use_gemini_expansion and music_crs.gemini_expander is not None:
        if music_crs.gemini_expansion_mode != "controlled_keyword_query":
            raise ValueError(
                "This trainer expects gemini_expansion_mode='controlled_keyword_query'."
            )
        gemini_query = music_crs.gemini_expander.expand_controlled_query(
            conversation_text,
            session_id=row.get("session_id"),
            turn_number=row.get("turn_number"),
        )
        retrieval_input = (
            conversation_text
            + "\n\nGemini-controlled BM25 search terms:\n"
            + "\n\n".join([gemini_query] * max(1, music_crs.gemini_keyword_block_weight))
        )

    return session_memory, retrieval_input


def collect_training_rows(
    music_crs: Any,
    batch_data: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    candidate_topk: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    truth_lookup = {
        (row["session_id"], row["turn_number"]): row["ground_truth_track_id"]
        for row in ground_truth
    }
    feature_names = music_crs.query_type_feature_names()
    x_rows = []
    y_rows = []
    positives = 0
    negatives = 0
    turns_with_positive_candidate = 0

    for row in tqdm(batch_data, desc="Collecting reranker training rows"):
        target_id = truth_lookup[(row["session_id"], row["turn_number"])]
        session_memory, retrieval_input = build_retrieval_input(music_crs, row)
        candidates = music_crs.retrieval.text_to_item_retrieval(
            retrieval_input,
            topk=candidate_topk,
        )
        query_features = music_crs._query_type_features(session_memory, retrieval_input)
        found_target = False

        for rank, track_id in enumerate(candidates, start=1):
            feature_values = music_crs.query_type_candidate_features(
                track_id,
                rank,
                query_features,
            )
            label = int(track_id == target_id)
            x_rows.append([feature_values[name] for name in feature_names])
            y_rows.append(label)
            positives += label
            negatives += 1 - label
            found_target = found_target or bool(label)

        turns_with_positive_candidate += int(found_target)

    diagnostics = {
        "candidate_topk": candidate_topk,
        "training_examples": len(y_rows),
        "positives": positives,
        "negatives": negatives,
        "turns": len(batch_data),
        "turns_with_positive_candidate": turns_with_positive_candidate,
        "candidate_recall": turns_with_positive_candidate / max(len(batch_data), 1),
    }
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=int), feature_names, diagnostics


def train_model(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    if int(y_train.sum()) == 0:
        raise ValueError("No positive candidates found; cannot train reranker weights.")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned query-type reranker weights.")
    parser.add_argument("--tid", default="devset_bm25_gemini_keywords_query_type_router")
    parser.add_argument("--num_rows", type=int, default=50)
    parser.add_argument("--candidate_topk", type=int, default=100)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument(
        "--gemini_cache_dir",
        default="./cache/gemini_keywords_devset_first100",
    )
    parser.add_argument(
        "--output",
        default="./models/query_type_logistic_weights.json",
    )
    args = parser.parse_args()

    config = OmegaConf.load(PROJECT_ROOT / "config" / f"{args.tid}.yaml")
    music_crs = load_model(config, args.device, args.gemini_cache_dir)
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    subset = dataset.select(range(min(args.num_rows, len(dataset))))
    batch_data, _, ground_truth = prepare_subset_data(subset, music_crs)

    x_train, y_train, feature_names, diagnostics = collect_training_rows(
        music_crs,
        batch_data,
        ground_truth,
        candidate_topk=args.candidate_topk,
    )
    model = train_model(x_train, y_train)
    scores = model.predict_proba(x_train)[:, 1]

    weights = {
        name: float(value)
        for name, value in zip(feature_names, model.coef_[0])
    }
    payload = {
        "model_type": "logistic_regression_query_type_reranker",
        "base_tid": args.tid,
        "num_rows": args.num_rows,
        "feature_names": feature_names,
        "intercept": float(model.intercept_[0]),
        "weights": weights,
        "training_diagnostics": diagnostics,
        "train_average_precision": float(average_precision_score(y_train, scores)),
    }
    if len(set(y_train.tolist())) == 2:
        payload["train_roc_auc"] = float(roc_auc_score(y_train, scores))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved learned query-type weights to: {output_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
