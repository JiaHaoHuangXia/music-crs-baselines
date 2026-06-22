"""Hybrid retrieval over labeled train conversations and track metadata.

The challenge objective is a single relevant track per turn.  This retriever
therefore treats the public train turns as supervised nearest-neighbor examples:
retrieve similar conversation states, vote for their ground-truth tracks, and
fuse that signal with catalog metadata retrieval.
"""

import json
import os
import re
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
        self._build_catalog_indexes()
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

    def _normalize(self, text: Any) -> str:
        text = self._clean_join(text).lower()
        text = re.sub(r"\([^)]*(anniversary|deluxe|remaster|expanded|edition|live)[^)]*\)", " ", text)
        text = re.sub(r"\b(remaster(?:ed)?|deluxe|expanded|anniversary|edition|explicit|clean|mono|stereo)\b", " ", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _tokenize_text(self, text: Any) -> set[str]:
        stopwords = {
            "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with",
            "me", "my", "i", "want", "need", "song", "track", "music", "something",
            "recommend", "recommendation", "please", "like", "more", "another",
        }
        return {
            token
            for token in self._normalize(text).split()
            if len(token) > 2 and token not in stopwords
        }

    def _build_catalog_indexes(self) -> None:
        self.artist_to_tracks = defaultdict(list)
        self.tag_to_tracks = defaultdict(list)
        self.year_to_tracks = defaultdict(list)
        self.decade_to_tracks = defaultdict(list)
        self.version_groups = defaultdict(list)
        self.token_to_tracks = defaultdict(list)
        self.track_popularity = {}
        self.track_tokens = {}

        for track_id, metadata in self.metadata_dict.items():
            popularity = float(metadata.get("popularity") or 0.0)
            self.track_popularity[track_id] = popularity

            title_norm = self._normalize(metadata.get("track_name"))
            artists = metadata.get("artist_name") or []
            if not isinstance(artists, list):
                artists = [artists]
            artist_norms = [self._normalize(artist) for artist in artists if self._normalize(artist)]
            primary_artist = artist_norms[0] if artist_norms else ""
            if title_norm and primary_artist:
                self.version_groups[(title_norm, primary_artist)].append(track_id)
            for artist_norm in artist_norms:
                self.artist_to_tracks[artist_norm].append(track_id)

            tags = metadata.get("tag_list") or []
            if not isinstance(tags, list):
                tags = [tags]
            for tag in tags:
                tag_norm = self._normalize(tag)
                if tag_norm:
                    self.tag_to_tracks[tag_norm].append(track_id)

            release_date = str(metadata.get("release_date") or "")
            match = re.search(r"(19|20)\d{2}", release_date)
            if match:
                year = match.group(0)
                self.year_to_tracks[year].append(track_id)
                self.decade_to_tracks[f"{year[:3]}0s"].append(track_id)

            metadata_text = " ".join(
                self._clean_join(metadata.get(field))
                for field in ["track_name", "artist_name", "album_name", "tag_list", "release_date"]
            )
            self.track_tokens[track_id] = self._tokenize_text(metadata_text)
            for token in self.track_tokens[track_id]:
                self.token_to_tracks[token].append(track_id)

        for index in [
            self.artist_to_tracks,
            self.tag_to_tracks,
            self.year_to_tracks,
            self.decade_to_tracks,
            self.version_groups,
            self.token_to_tracks,
        ]:
            for key, track_ids in index.items():
                index[key] = sorted(
                    dict.fromkeys(track_ids),
                    key=lambda tid: (-self.track_popularity.get(tid, 0.0), tid),
                )

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

    def _parse_query_state(self, query: str) -> Dict[str, Any]:
        user_messages = re.findall(r"(?im)^user:\s*(.+)$", query)
        current_user = user_messages[-1] if user_messages else query
        all_user_text = " ".join(user_messages) if user_messages else query
        lower_current = current_user.lower()

        previous_track_ids = re.findall(r"track_id:\s*([0-9a-fA-F-]{36})", query)
        turn_match = re.search(r"(?im)^turn_number:\s*(\d+)", query)
        turn_number = int(turn_match.group(1)) if turn_match else None

        positive_patterns = [
            "more like", "similar", "same vibe", "that vibe", "i like", "i liked",
            "love", "loved", "yes", "great", "good", "another one", "keep",
        ]
        negative_patterns = [
            "not ", "don't", "dont", "avoid", "less", "too ", "no ", "different",
            "instead", "change", "something else",
        ]
        change_artist_patterns = [
            "different artist", "another artist", "new artist", "other artist",
            "not the same artist",
        ]

        feedback = "neutral"
        if any(pattern in lower_current for pattern in positive_patterns):
            feedback = "positive"
        if any(pattern in lower_current for pattern in negative_patterns):
            feedback = "negative"
        change_artist = any(pattern in lower_current for pattern in change_artist_patterns)

        return {
            "current_user": current_user,
            "all_user_text": all_user_text,
            "previous_track_ids": list(dict.fromkeys(previous_track_ids)),
            "feedback": feedback,
            "change_artist": change_artist,
            "turn_number": turn_number,
            "tokens": self._tokenize_text(all_user_text),
        }

    def _structured_query(self, query: str, state: Dict[str, Any]) -> str:
        parts = [query, f"current_request: {state['current_user']}"]
        parts.append(f"all_user_preferences: {state['all_user_text']}")

        if state["previous_track_ids"]:
            previous_metadata = [
                self._metadata_to_text(track_id)
                for track_id in state["previous_track_ids"][-3:]
            ]
            label = "liked_previous_tracks" if state["feedback"] == "positive" else "previous_tracks"
            parts.append(f"{label}:\n" + "\n".join(previous_metadata))

        if state["change_artist"]:
            parts.append("constraint: use a different artist from previous recommendations")
        if state["feedback"] == "negative":
            parts.append("negative_feedback: avoid the immediately previous recommendation style")

        return "\n".join(parts)

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

    def _catalog_signal_candidates(self, state: Dict[str, Any], limit: int = 160) -> List[str]:
        scores = defaultdict(float)
        query_norm = self._normalize(state["all_user_text"])
        query_tokens = state["tokens"]

        for artist_norm, track_ids in self.artist_to_tracks.items():
            if artist_norm and artist_norm in query_norm:
                for rank, track_id in enumerate(track_ids[:40], start=1):
                    scores[track_id] += 4.0 / (rank ** 0.5)

        for tag_norm, track_ids in self.tag_to_tracks.items():
            tag_tokens = set(tag_norm.split())
            if tag_norm in query_norm or (tag_tokens and tag_tokens.issubset(query_tokens)):
                for rank, track_id in enumerate(track_ids[:30], start=1):
                    scores[track_id] += 1.6 / (rank ** 0.5)

        for decade in re.findall(r"\b(?:19|20)\d0s\b", query_norm):
            for rank, track_id in enumerate(self.decade_to_tracks.get(decade, [])[:40], start=1):
                scores[track_id] += 1.2 / (rank ** 0.5)
        for year in re.findall(r"\b(?:19|20)\d{2}\b", query_norm):
            for rank, track_id in enumerate(self.year_to_tracks.get(year, [])[:40], start=1):
                scores[track_id] += 1.5 / (rank ** 0.5)

        overlap_counts = defaultdict(int)
        for token in query_tokens:
            for track_id in self.token_to_tracks.get(token, [])[:300]:
                overlap_counts[track_id] += 1
        for track_id, overlap in overlap_counts.items():
            if overlap >= 3:
                scores[track_id] += overlap + min(self.track_popularity.get(track_id, 0.0), 100.0) / 200.0

        return sorted(scores, key=lambda tid: (-scores[tid], -self.track_popularity.get(tid, 0.0), tid))[:limit]

    def _feedback_candidates(self, state: Dict[str, Any], limit: int = 120) -> List[str]:
        if not state["previous_track_ids"]:
            return []

        scores = defaultdict(float)
        previous_ids = state["previous_track_ids"][-3:]
        for recency, track_id in enumerate(reversed(previous_ids), start=1):
            metadata = self.metadata_dict.get(track_id)
            if not metadata:
                continue

            title_norm = self._normalize(metadata.get("track_name"))
            artists = metadata.get("artist_name") or []
            if not isinstance(artists, list):
                artists = [artists]
            artist_norms = [self._normalize(artist) for artist in artists if self._normalize(artist)]
            primary_artist = artist_norms[0] if artist_norms else ""

            if state["feedback"] != "negative":
                if title_norm and primary_artist:
                    for rank, sibling_id in enumerate(self.version_groups.get((title_norm, primary_artist), [])[:12], start=1):
                        if sibling_id != track_id:
                            scores[sibling_id] += 3.0 / (rank + recency)
                if not state["change_artist"]:
                    for artist_norm in artist_norms:
                        for rank, sibling_id in enumerate(self.artist_to_tracks.get(artist_norm, [])[:50], start=1):
                            if sibling_id != track_id:
                                scores[sibling_id] += 1.4 / (rank ** 0.5 + recency)

            tags = metadata.get("tag_list") or []
            if not isinstance(tags, list):
                tags = [tags]
            tag_weight = 1.2 if state["feedback"] != "negative" else 0.5
            for tag in tags[:8]:
                tag_norm = self._normalize(tag)
                for rank, sibling_id in enumerate(self.tag_to_tracks.get(tag_norm, [])[:35], start=1):
                    if sibling_id != track_id:
                        scores[sibling_id] += tag_weight / (rank ** 0.5 + recency)

        return sorted(scores, key=lambda tid: (-scores[tid], -self.track_popularity.get(tid, 0.0), tid))[:limit]

    def _excluded_tracks(self, state: Dict[str, Any]) -> set[str]:
        excluded = set(state["previous_track_ids"])
        if state["feedback"] == "negative" and state["previous_track_ids"]:
            last_track = state["previous_track_ids"][-1]
            metadata = self.metadata_dict.get(last_track)
            if metadata:
                artists = metadata.get("artist_name") or []
                if not isinstance(artists, list):
                    artists = [artists]
                for artist in artists:
                    artist_norm = self._normalize(artist)
                    excluded.update(self.artist_to_tracks.get(artist_norm, [])[:80])
        if state["change_artist"] and state["previous_track_ids"]:
            for track_id in state["previous_track_ids"][-2:]:
                metadata = self.metadata_dict.get(track_id)
                if not metadata:
                    continue
                artists = metadata.get("artist_name") or []
                if not isinstance(artists, list):
                    artists = [artists]
                for artist in artists:
                    excluded.update(self.artist_to_tracks.get(self._normalize(artist), [])[:80])
        return excluded

    def _version_expand(self, ranked_track_ids: List[str], limit: int) -> List[str]:
        expanded = []
        seen = set()
        for track_id in ranked_track_ids:
            if track_id not in seen:
                expanded.append(track_id)
                seen.add(track_id)
            metadata = self.metadata_dict.get(track_id)
            if not metadata:
                continue
            title_norm = self._normalize(metadata.get("track_name"))
            artists = metadata.get("artist_name") or []
            if not isinstance(artists, list):
                artists = [artists]
            primary_artist = self._normalize(artists[0]) if artists else ""
            for sibling_id in self.version_groups.get((title_norm, primary_artist), [])[:4]:
                if sibling_id not in seen:
                    expanded.append(sibling_id)
                    seen.add(sibling_id)
            if len(expanded) >= limit:
                break
        return expanded[:limit]

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
        state = self._parse_query_state(query)
        structured_query = self._structured_query(query, state)
        turn_number = state.get("turn_number") or 1
        conversation_weight = 2.4 if turn_number <= 2 else 3.2
        feedback_weight = 2.0 if state["previous_track_ids"] else 0.0

        ranked_lists = [
            (self._conversation_retrieval(structured_query), conversation_weight),
            (self.bm25_metadata.text_to_item_retrieval(structured_query, self.metadata_topk), 0.9),
            (self._catalog_signal_candidates(state), 1.6),
        ]
        feedback_candidates = self._feedback_candidates(state)
        if feedback_candidates:
            ranked_lists.append((feedback_candidates, feedback_weight))
        if self.semantic_metadata is not None:
            ranked_lists.append(
                (
                    self.semantic_metadata.text_to_item_retrieval(structured_query, self.metadata_topk),
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
        excluded = self._excluded_tracks(state)
        filtered = [track_id for track_id in ranked_track_ids if track_id not in excluded]
        if len(filtered) < topk:
            filtered.extend(track_id for track_id in ranked_track_ids if track_id not in filtered)
        return self._version_expand(filtered, topk)

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int) -> List[List[str]]:
        return [self.text_to_item_retrieval(query, topk) for query in queries]
