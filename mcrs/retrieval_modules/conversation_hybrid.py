"""Hybrid retrieval over labeled train conversations and track metadata.

The challenge objective is a single relevant track per turn.  This retriever
therefore treats the public train turns as supervised nearest-neighbor examples:
retrieve similar conversation states, vote for their ground-truth tracks, and
fuse that signal with catalog metadata retrieval.
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List

import bm25s
from datasets import concatenate_datasets, load_dataset

from .bm25 import BM25_MODEL

try:
    from .sentence_transformer import SENTENCE_TRANSFORMER_MODEL
except Exception:  # pragma: no cover - optional dependency/runtime guard
    SENTENCE_TRANSFORMER_MODEL = None


class CONVERSATION_HYBRID_MODEL:
    """Fuse train-turn label retrieval with metadata lexical/semantic retrieval."""

    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        train_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Dataset",
        train_split: str = "train",
        conversation_topk: int = 250,
        metadata_topk: int = 100,
        rrf_k: int = 60,
        use_semantic_metadata: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.corpus_types = corpus_types
        self.cache_dir = cache_dir
        self.train_dataset_name = train_dataset_name
        self.train_split = train_split
        self.conversation_topk = conversation_topk
        self.metadata_topk = metadata_topk
        self.rrf_k = rrf_k
        self.use_semantic_metadata = use_semantic_metadata
        self.index_dir = os.path.join(
            cache_dir,
            "conversation_hybrid",
            train_dataset_name.replace("/", "__"),
            train_split,
        )

        self.metadata_dict = self._load_metadata()
        self.bm25_metadata = BM25_MODEL(dataset_name, split_types, corpus_types, cache_dir)
        self.semantic_metadata = self._load_semantic_metadata()

        if self._has_conversation_index():
            self.conversation_index, self.train_track_ids = self._load_conversation_index()
        else:
            self.build_conversation_index()
            self.conversation_index, self.train_track_ids = self._load_conversation_index()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        metadata_dataset = load_dataset(self.dataset_name)
        metadata_concat_dataset = concatenate_datasets(
            [metadata_dataset[split_type] for split_type in self.split_types]
        )
        return {item["track_id"]: item for item in metadata_concat_dataset}

    def _load_semantic_metadata(self):
        if not self.use_semantic_metadata:
            return None
        if SENTENCE_TRANSFORMER_MODEL is None:
            return None
        try:
            return SENTENCE_TRANSFORMER_MODEL(
                self.dataset_name,
                self.split_types,
                self.corpus_types,
                self.cache_dir,
            )
        except Exception as exc:
            print(f"Sentence-transformer metadata retrieval unavailable: {exc}", flush=True)
            return None

    def _has_conversation_index(self) -> bool:
        return (
            os.path.exists(os.path.join(self.index_dir, "params.index.json"))
            and os.path.exists(os.path.join(self.index_dir, "train_track_ids.json"))
        )

    def _load_conversation_index(self):
        index = bm25s.BM25.load(self.index_dir, load_corpus=True)
        with open(os.path.join(self.index_dir, "train_track_ids.json"), "r", encoding="utf-8") as file:
            train_track_ids = json.load(file)
        return index, train_track_ids

    def _clean_join(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, dict):
            return " ".join(
                f"{key}: {self._clean_join(inner_value)}"
                for key, inner_value in sorted(value.items())
                if self._clean_join(inner_value)
            )
        return str(value).strip()

    def _metadata_to_text(self, track_id: str) -> str:
        metadata = self.metadata_dict.get(track_id)
        if not metadata:
            return f"track_id: {track_id}"
        fields = ["track_name", "artist_name", "album_name", "tag_list", "release_date"]
        return "\n".join(
            f"{field}: {self._clean_join(metadata.get(field))}"
            for field in fields
            if self._clean_join(metadata.get(field))
        )

    def _ground_truth_for_turn(self, conversations: List[Dict[str, Any]], turn_number: int) -> str | None:
        for message in conversations:
            if message.get("turn_number") == turn_number and message.get("role") == "music":
                return message.get("content")
        return None

    def _turn_numbers(self, conversations: List[Dict[str, Any]]) -> List[int]:
        return sorted(
            {
                message.get("turn_number")
                for message in conversations
                if message.get("role") == "user"
            }
        )

    def _conversation_doc(
        self,
        item: Dict[str, Any],
        target_turn_number: int,
    ) -> str:
        parts = []
        goal = item.get("conversation_goal")
        if goal:
            parts.append(f"conversation_goal: {self._clean_join(goal)}")
        profile = item.get("user_profile")
        if profile:
            parts.append(f"user_profile: {self._clean_join(profile)}")

        for message in item.get("conversations", []):
            turn_number = message.get("turn_number")
            if turn_number is None or turn_number > target_turn_number:
                continue
            role = message.get("role")
            content = message.get("content")
            if turn_number == target_turn_number and role != "user":
                continue
            if role == "music":
                role = "assistant_recommended_track"
                content = self._metadata_to_text(content)
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def build_conversation_index(self) -> None:
        train_dataset = load_dataset(self.train_dataset_name, split=self.train_split)
        docs = []
        train_track_ids = []

        for item in train_dataset:
            conversations = item.get("conversations", [])
            for turn_number in self._turn_numbers(conversations):
                track_id = self._ground_truth_for_turn(conversations, turn_number)
                if not track_id:
                    continue
                docs.append(self._conversation_doc(item, turn_number))
                train_track_ids.append(track_id)

        os.makedirs(self.index_dir, exist_ok=True)
        tokens = bm25s.tokenize([doc.lower() for doc in docs])
        index = bm25s.BM25()
        index.index(tokens)
        index.save(self.index_dir, corpus=docs)
        with open(os.path.join(self.index_dir, "train_track_ids.json"), "w", encoding="utf-8") as file:
            json.dump(train_track_ids, file)

    def _conversation_retrieval(self, query: str) -> List[str]:
        tokens = bm25s.tokenize([query.lower()])
        k = min(self.conversation_topk, len(self.train_track_ids))
        doc_scores = self.conversation_index.retrieve(tokens, k=k, return_as="tuple")

        scores = defaultdict(float)
        best_rank = {}
        for rank, item in enumerate(doc_scores.documents[0], start=1):
            track_id = self.train_track_ids[item["id"]]
            scores[track_id] += 3.0 / (self.rrf_k + rank)
            best_rank[track_id] = min(best_rank.get(track_id, rank), rank)

        return sorted(scores, key=lambda tid: (-scores[tid], best_rank[tid], tid))

    def _rrf_add(
        self,
        fused_scores: Dict[str, float],
        best_rank: Dict[str, int],
        ranked_items: Iterable[str],
        weight: float,
    ) -> None:
        for rank, track_id in enumerate(ranked_items, start=1):
            fused_scores[track_id] += weight / (self.rrf_k + rank)
            best_rank[track_id] = min(best_rank.get(track_id, rank), rank)

    def text_to_item_retrieval(self, query: str, topk: int) -> List[str]:
        ranked_lists = [
            (self._conversation_retrieval(query), 3.0),
            (self.bm25_metadata.text_to_item_retrieval(query, self.metadata_topk), 0.7),
        ]
        if self.semantic_metadata is not None:
            ranked_lists.append(
                (
                    self.semantic_metadata.text_to_item_retrieval(query, self.metadata_topk),
                    1.5,
                )
            )

        fused_scores = defaultdict(float)
        best_rank = {}
        for ranked_items, weight in ranked_lists:
            self._rrf_add(fused_scores, best_rank, ranked_items, weight)

        ranked_track_ids = sorted(
            fused_scores,
            key=lambda tid: (-fused_scores[tid], best_rank[tid], tid),
        )
        return ranked_track_ids[:topk]

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int) -> List[List[str]]:
        return [self.text_to_item_retrieval(query, topk) for query in queries]
