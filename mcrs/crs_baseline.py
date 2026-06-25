import os
import re
import torch
from collections import defaultdict
from typing import Optional, Any, List, Dict
from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import load_retrieval_module
from mcrs.controlled_tags import normalize_music_tag, normalize_tag, normalized_music_tags
from mcrs.style_profiles import release_decade_text, weighted_metadata_lines

from mcrs.query_expansion.gemini_expander import GeminiExpander

class CRS_BASELINE:
    """
    Conversational Recommender System (CRS) baseline that wires together an LLM module and an item retrieval module over a music catalog and user profiles.
    Attributes:
        cache_dir: Local path for caching artifacts and indices.
        lm_type: Identifier/name for the LLM backend to load.
        retrieval_type: Retrieval backend to use (e.g., "bm25").
        item_db_name: Hugging Face dataset or DB name for item metadata.
        user_db_name: Hugging Face dataset or DB name for user metadata.
        split_types: Dataset split names to load (e.g., ["test_warm", "test_cold"]).
        corpus_types: Item fields used for retrieval (e.g., title, artist, album).
        device: Compute device for the LLM (e.g., "cuda", "cpu").
        dtype: Torch dtype used by the LLM.
        lm: Loaded LLM module used for response generation.
        retrieval: Retrieval module used to fetch candidate items.
        item_db: Item metadata database accessor.
        user_db: User profile database accessor.
        prompts_dir: Directory containing prompt templates.
        role_prompt: Loaded prompt templates keyed by role.
        session_memory: In-memory list of message dicts for the current session.
    """
    def __init__(self,
        lm_type="meta-llama/Llama-3.2-1B-Instruct",
        retrieval_type="bm25",
        item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
        track_split_types: list[str] = ["all_tracks"], # for test
        user_split_types: list[str] = ["all_users"],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir="./cache",
        device="cuda",
        attn_implementation="eager",
        dtype=torch.bfloat16,
        # New Gemini parameters
        use_gemini_expansion: bool = False,
        gemini_model_name: str = "gemini-3.1-flash-lite",
        gemini_cache_dir: str = "./cache/gemini_expansions",
        gemini_keyword_field_weights: Optional[dict[str, int]] = None,
        gemini_keyword_block_weight: int = 1,
        gemini_expansion_mode: str = "tag_query",
        gemini_topk_per_reference: int = 50,
        gemini_rrf_k: int = 60,
        include_original_query_in_fusion: bool = False,
        original_query_weight: float = 2.0,
        gemini_reference_weight: float = 1.0,
        gemini_max_reference_tracks: Optional[int] = None,
        gemini_fusion_method: str = "rrf",
        rerank_tag_weight: float = 0.04,
        rerank_artist_profile_weight: float = 0.04,
        rerank_decade_weight: float = 0.02,
        hybrid_bm25_corpus_types: Optional[list[str]] = None,
        hybrid_bm25_topk_per_reference: int = 100,
        hybrid_bm25_weight: float = 0.12,
        hybrid_artist_match_weight: float = 0.03,
        hybrid_album_match_weight: float = 0.02,
        hybrid_multi_source_weight: float = 0.03,
        use_query_type_routing: bool = False,
        query_type_candidate_topk: int = 100,
        query_type_rank_weight: float = 1.0,
        query_type_title_weight: float = 0.80,
        query_type_artist_weight: float = 0.55,
        query_type_album_weight: float = 0.35,
        query_type_decade_weight: float = 0.18,
        query_type_negative_weight: float = 0.35,
        retrieval_only: bool = False,
    ):
        """Initialize the CRS baseline components.

        Args:
            lm_type: LLM model identifier to load for response generation.
            retrieval_type: Retrieval backend name (e.g., "bm25").
            item_db_name: Dataset/DB name for item metadata.
            user_db_name: Dataset/DB name for user metadata.
            split_types: Dataset split names to load.
            corpus_types: Item metadata fields used for retrieval.
            cache_dir: Local directory for caching artifacts/indices.
            device: Compute device for the LLM (e.g., "cuda", "cpu").
            dtype: Torch dtype for the LLM weights/tensors.
        """
        self.cache_dir = cache_dir
        self.lm_type = lm_type
        self.retrieval_type = retrieval_type
        self.item_db_name = item_db_name
        self.user_db_name = user_db_name
        self.track_split_types = track_split_types
        self.user_split_types = user_split_types
        self.corpus_types = corpus_types
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.retrieval_only = retrieval_only

        # Gemini query expansion
        self.use_gemini_expansion = use_gemini_expansion
        self.gemini_keyword_field_weights = gemini_keyword_field_weights
        self.gemini_keyword_block_weight = gemini_keyword_block_weight
        self.gemini_expansion_mode = gemini_expansion_mode
        self.gemini_topk_per_reference = gemini_topk_per_reference
        self.gemini_rrf_k = gemini_rrf_k
        self.include_original_query_in_fusion = include_original_query_in_fusion
        self.original_query_weight = original_query_weight
        self.gemini_reference_weight = gemini_reference_weight
        self.gemini_max_reference_tracks = gemini_max_reference_tracks
        self.gemini_fusion_method = gemini_fusion_method
        self.rerank_tag_weight = rerank_tag_weight
        self.rerank_artist_profile_weight = rerank_artist_profile_weight
        self.rerank_decade_weight = rerank_decade_weight
        self.hybrid_bm25_corpus_types = hybrid_bm25_corpus_types or [
            "track_name",
            "artist_name",
            "album_name",
            "tag_list",
        ]
        self.hybrid_bm25_topk_per_reference = hybrid_bm25_topk_per_reference
        self.hybrid_bm25_weight = hybrid_bm25_weight
        self.hybrid_artist_match_weight = hybrid_artist_match_weight
        self.hybrid_album_match_weight = hybrid_album_match_weight
        self.hybrid_multi_source_weight = hybrid_multi_source_weight
        self.use_query_type_routing = use_query_type_routing
        self.query_type_candidate_topk = query_type_candidate_topk
        self.query_type_rank_weight = query_type_rank_weight
        self.query_type_title_weight = query_type_title_weight
        self.query_type_artist_weight = query_type_artist_weight
        self.query_type_album_weight = query_type_album_weight
        self.query_type_decade_weight = query_type_decade_weight
        self.query_type_negative_weight = query_type_negative_weight
        valid_gemini_modes = {
            "tag_query",
            "multi_query_fusion",
            "controlled_keyword_query",
        }
        if self.gemini_expansion_mode not in valid_gemini_modes:
            raise ValueError(
                f"Unknown gemini_expansion_mode='{self.gemini_expansion_mode}'. "
                f"Expected one of {sorted(valid_gemini_modes)}."
            )
        valid_fusion_methods = {
            "rrf",
            "max_similarity",
            "max_similarity_structured_rerank",
            "hybrid_structured_rerank",
        }
        if self.gemini_fusion_method not in valid_fusion_methods:
            raise ValueError(
                f"Unknown gemini_fusion_method='{self.gemini_fusion_method}'. "
                f"Expected one of {sorted(valid_fusion_methods)}."
            )
        if (
            self.use_gemini_expansion
            and self.gemini_expansion_mode == "multi_query_fusion"
            and self.retrieval_type not in {"bert", "sentence_transformer"}
        ):
            raise ValueError(
                "gemini_expansion_mode='multi_query_fusion' requires retrieval_type='bert' "
                "or retrieval_type='sentence_transformer'."
            )

        if self.use_gemini_expansion:
            self.gemini_expander = GeminiExpander(
                model_name=gemini_model_name,
                cache_dir=gemini_cache_dir,
                keyword_field_weights=gemini_keyword_field_weights,
            )
        else:
            self.gemini_expander = None

        self.lm = None if self.retrieval_only else load_lm_module(self.lm_type, self.device, self.attn_implementation, self.dtype)
        self.retrieval = load_retrieval_module(self.retrieval_type, self.item_db_name, self.track_split_types, self.corpus_types, self.cache_dir)
        self.lexical_retrieval = None
        if self.gemini_fusion_method == "hybrid_structured_rerank":
            self.lexical_retrieval = load_retrieval_module(
                "bm25",
                self.item_db_name,
                self.track_split_types,
                self.hybrid_bm25_corpus_types,
                self.cache_dir,
            )
        self.item_db = MusicCatalogDB(self.item_db_name, self.track_split_types, self.corpus_types)
        self.user_db = None if self.retrieval_only else UserProfileDB(self.user_db_name, self.user_split_types)
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "system_prompts")
        self.role_prompt = {} if self.retrieval_only else {
            "role_play": open(f"{self.prompts_dir}/roleplay.txt", "r", encoding="utf-8").read(),
            "personalization": open(f"{self.prompts_dir}/personalization.txt", "r", encoding="utf-8").read(),
            "response_generation": open(f"{self.prompts_dir}/response_generation.txt", "r", encoding="utf-8").read(),
        }
        self.session_memory = []

    def _gemini_reference_to_query(self, track: Dict[str, Any]) -> str:
        """Convert one Gemini reference track into metadata-style query text."""
        def clean_join(value: Any) -> str:
            if isinstance(value, list):
                return ", ".join(str(item).strip() for item in value if str(item).strip())
            return str(value).strip() if value is not None else ""

        values = {
            "track_name": clean_join(track.get("track_name")),
            "artist_name": clean_join(track.get("artist_name")),
            "album_name": clean_join(track.get("album_name")),
            "tag_list": clean_join(track.get("tag_list")),
            "artist_style_profile": normalized_music_tags(track.get("tag_list"), max_tags=8),
            "release_date": clean_join(track.get("release_date")),
            "release_decade": release_decade_text(track.get("release_date")),
        }
        return "\n".join(weighted_metadata_lines(values, self.corpus_types))

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[Any]],
        weights: List[float],
        topk: int,
    ) -> List[str]:
        """Fuse multiple ranked track lists using weighted reciprocal rank fusion."""
        fused_scores = defaultdict(float)
        best_rank = {}

        for list_idx, ranked_items in enumerate(ranked_lists):
            weight = weights[list_idx]
            for rank, item in enumerate(ranked_items, start=1):
                track_id = item[0] if isinstance(item, tuple) else item
                fused_scores[track_id] += weight / (self.gemini_rrf_k + rank)
                best_rank[track_id] = min(best_rank.get(track_id, rank), rank)

        ranked_track_ids = sorted(
            fused_scores,
            key=lambda track_id: (-fused_scores[track_id], best_rank[track_id], track_id),
        )
        return ranked_track_ids[:topk]

    def _max_similarity_fusion(
        self,
        ranked_lists: List[List[Any]],
        topk: int,
    ) -> List[str]:
        """Fuse scored ranked lists by each track's best cosine similarity."""
        max_scores = {}
        best_rank = {}

        for ranked_items in ranked_lists:
            for rank, item in enumerate(ranked_items, start=1):
                if isinstance(item, tuple):
                    track_id, score = item
                else:
                    track_id, score = item, 0.0

                if track_id not in max_scores or score > max_scores[track_id]:
                    max_scores[track_id] = score
                best_rank[track_id] = min(best_rank.get(track_id, rank), rank)

        ranked_track_ids = sorted(
            max_scores,
            key=lambda track_id: (-max_scores[track_id], best_rank[track_id], track_id),
        )
        return ranked_track_ids[:topk]

    def _metadata_for_track(self, track_id: str) -> dict[str, Any]:
        if hasattr(self.retrieval, "metadata_dict") and track_id in self.retrieval.metadata_dict:
            return self.retrieval.metadata_dict[track_id]
        return self.item_db.metadata_dict[track_id]

    @staticmethod
    def _overlap_score(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left)

    @staticmethod
    def _metadata_text_set(value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, list):
            values = value
        else:
            values = str(value).split(",")
        return {normalize_tag(item) for item in values if normalize_tag(item)}

    @staticmethod
    def _field_values_from_text(text: str, field: str) -> list[str]:
        pattern = re.compile(rf"{re.escape(field)}:\s*([^\n]+)", re.IGNORECASE)
        values = []
        for match in pattern.finditer(text or ""):
            values.extend(part.strip() for part in match.group(1).split(","))
        return [value for value in values if value]

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        phrase = normalize_tag(phrase)
        if not phrase:
            return False
        return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalize_tag(text)) is not None

    def _metadata_from_history_message(self, message: dict[str, Any]) -> Optional[dict[str, Any]]:
        content = str(message.get("content", ""))
        match = re.search(r"track_id:\s*([^,\n]+)", content, re.IGNORECASE)
        if not match:
            return None
        track_id = match.group(1).strip()
        return self.item_db.metadata_dict.get(track_id)

    def _query_type_features(
        self,
        session_memory: list[dict[str, Any]],
        retrieval_input: str,
    ) -> dict[str, Any]:
        current_query = str(session_memory[-1].get("content", "")) if session_memory else ""
        current_query_norm = normalize_tag(current_query)
        previous_metadata = [
            metadata
            for message in session_memory[:-1]
            if message.get("role") == "assistant"
            for metadata in [self._metadata_from_history_message(message)]
            if metadata is not None
        ]

        structured_terms = {
            "titles": set(self._field_values_from_text(retrieval_input, "track_titles")),
            "artists": set(self._field_values_from_text(retrieval_input, "artists")),
            "albums": set(self._field_values_from_text(retrieval_input, "albums")),
            "decades": set(self._field_values_from_text(retrieval_input, "era")),
            "avoid": set(self._field_values_from_text(retrieval_input, "avoid_terms")),
        }

        previous_artists = set()
        for metadata in previous_metadata:
            previous_artists.update(self._metadata_text_set(metadata.get("artist_name")))

        same_artist_intent = any(
            phrase in current_query_norm
            for phrase in [
                "same artist",
                "same band",
                "by them",
                "from them",
                "another one by",
                "more by",
            ]
        )
        explicit_artist_intent = any(
            phrase in current_query_norm
            for phrase in [" by ", " from ", " song by ", " track by ", " artist "]
        )
        explicit_title_intent = any(
            self._contains_phrase(current_query_norm, title)
            for title in structured_terms["titles"]
        )
        explicit_album_intent = any(
            self._contains_phrase(current_query_norm, album)
            for album in structured_terms["albums"]
        )
        decade_intent = bool(
            re.search(r"\b(19[5-9]0s|20[0-2]0s|50s|60s|70s|80s|90s|00s|2000s|2010s|10s)\b", current_query_norm)
        )
        negative_intent = any(
            phrase in current_query_norm
            for phrase in [
                "not ",
                "no ",
                "without ",
                "less ",
                "avoid ",
                "don't want",
                "do not want",
            ]
        )

        route_types = set()
        if same_artist_intent and previous_artists:
            route_types.add("same_artist")
        if explicit_title_intent:
            route_types.add("title")
        if explicit_album_intent:
            route_types.add("album")
        if explicit_artist_intent and any(
            self._contains_phrase(current_query_norm, artist)
            for artist in structured_terms["artists"]
        ):
            route_types.add("artist")
        if decade_intent:
            route_types.add("decade")
        if negative_intent and structured_terms["avoid"]:
            route_types.add("negative")

        return {
            "route_types": route_types,
            "titles": {normalize_tag(value) for value in structured_terms["titles"]},
            "artists": {normalize_tag(value) for value in structured_terms["artists"]},
            "albums": {normalize_tag(value) for value in structured_terms["albums"]},
            "decades": {normalize_tag(value) for value in structured_terms["decades"]},
            "avoid": {normalize_music_tag(value) for value in structured_terms["avoid"]},
            "previous_artists": previous_artists,
        }

    def _query_type_route_rerank(
        self,
        ranked_items: list[str],
        session_memory: list[dict[str, Any]],
        retrieval_input: str,
        topk: int = 20,
    ) -> list[str]:
        features = self._query_type_features(session_memory, retrieval_input)
        route_types = features["route_types"]
        if not route_types:
            return ranked_items[:topk]

        reranked = []

        for rank, track_id in enumerate(ranked_items, start=1):
            metadata = self._metadata_for_track(track_id)
            title = normalize_tag(metadata.get("track_name"))
            artist_set = self._metadata_text_set(metadata.get("artist_name"))
            album_set = self._metadata_text_set(metadata.get("album_name"))
            tag_set = set(normalized_music_tags(metadata.get("tag_list"), max_tags=80))
            tag_set.update(metadata.get("artist_style_profile", []))
            decade = normalize_tag(metadata.get("release_decade") or release_decade_text(metadata.get("release_date")))

            score = self.query_type_rank_weight / rank

            if "title" in route_types and any(self._contains_phrase(title, title_query) for title_query in features["titles"]):
                score += self.query_type_title_weight
            if "artist" in route_types and features["artists"] & artist_set:
                score += self.query_type_artist_weight
            if "same_artist" in route_types and features["previous_artists"] & artist_set:
                score += self.query_type_artist_weight
            if "album" in route_types and features["albums"] & album_set:
                score += self.query_type_album_weight
            if "decade" in route_types and features["decades"] and decade in features["decades"]:
                score += self.query_type_decade_weight

            if "negative" in route_types and features["avoid"] & (artist_set | album_set | tag_set):
                score -= self.query_type_negative_weight

            reranked.append((track_id, score, rank))

        reranked.sort(key=lambda item: (-item[1], item[2], item[0]))
        return [track_id for track_id, _, _ in reranked[:topk]]

    def _max_similarity_structured_rerank(
        self,
        ranked_lists: List[List[Any]],
        gemini_tracks: List[Dict[str, Any]],
        topk: int,
    ) -> List[str]:
        """Rerank max-similarity candidates with small structured music signals."""
        max_scores = {}
        best_rank = {}
        hit_counts = defaultdict(int)

        for ranked_items in ranked_lists:
            for rank, item in enumerate(ranked_items, start=1):
                if isinstance(item, tuple):
                    track_id, score = item
                else:
                    track_id, score = item, 0.0
                if track_id not in max_scores or score > max_scores[track_id]:
                    max_scores[track_id] = score
                best_rank[track_id] = min(best_rank.get(track_id, rank), rank)
                hit_counts[track_id] += 1

        query_tags = set()
        query_decades = set()
        for track in gemini_tracks:
            query_tags.update(normalized_music_tags(track.get("tag_list"), max_tags=12))
            decade = release_decade_text(track.get("release_date"))
            if decade:
                query_decades.add(decade)

        reranked = []
        for track_id, embedding_score in max_scores.items():
            metadata = self._metadata_for_track(track_id)
            track_tags = set(normalized_music_tags(metadata.get("tag_list"), max_tags=40))
            artist_profile = set(metadata.get("artist_style_profile", []))
            decade = metadata.get("release_decade") or release_decade_text(metadata.get("release_date"))

            tag_score = self._overlap_score(query_tags, track_tags)
            artist_score = self._overlap_score(query_tags, artist_profile)
            decade_score = 1.0 if decade and decade in query_decades else 0.0

            final_score = (
                embedding_score
                + self.rerank_tag_weight * tag_score
                + self.rerank_artist_profile_weight * artist_score
                + self.rerank_decade_weight * decade_score
            )
            reranked.append(
                (
                    track_id,
                    final_score,
                    embedding_score,
                    hit_counts[track_id],
                    best_rank[track_id],
                )
            )

        reranked.sort(key=lambda item: (-item[1], -item[2], -item[3], item[4], item[0]))
        return [track_id for track_id, *_ in reranked[:topk]]

    def _hybrid_structured_rerank(
        self,
        dense_ranked_lists: List[List[Any]],
        lexical_ranked_lists: List[List[Any]],
        gemini_tracks: List[Dict[str, Any]],
        topk: int,
    ) -> List[str]:
        """Fuse dense and BM25 candidates, then rerank with lightweight metadata signals."""
        dense_scores = {}
        lexical_scores = defaultdict(float)
        best_rank = {}
        source_hits = defaultdict(int)

        for ranked_items in dense_ranked_lists:
            for rank, item in enumerate(ranked_items, start=1):
                if isinstance(item, tuple):
                    track_id, score = item
                else:
                    track_id, score = item, 0.0
                dense_scores[track_id] = max(dense_scores.get(track_id, float("-inf")), score)
                best_rank[track_id] = min(best_rank.get(track_id, rank), rank)
                source_hits[track_id] += 1

        for ranked_items in lexical_ranked_lists:
            for rank, item in enumerate(ranked_items, start=1):
                track_id = item[0] if isinstance(item, tuple) else item
                lexical_scores[track_id] = max(lexical_scores[track_id], 1.0 / rank)
                best_rank[track_id] = min(best_rank.get(track_id, rank), rank)
                source_hits[track_id] += 1
                if track_id not in dense_scores:
                    dense_scores[track_id] = 0.0

        query_tags = set()
        query_decades = set()
        query_artists = set()
        query_albums = set()
        for track in gemini_tracks:
            query_tags.update(normalized_music_tags(track.get("tag_list"), max_tags=20))
            query_artists.update(self._metadata_text_set(track.get("artist_name")))
            query_albums.update(self._metadata_text_set(track.get("album_name")))
            decade = release_decade_text(track.get("release_date"))
            if decade:
                query_decades.add(decade)

        max_source_hits = max(source_hits.values(), default=1)
        reranked = []
        for track_id, embedding_score in dense_scores.items():
            metadata = self._metadata_for_track(track_id)
            track_tags = set(normalized_music_tags(metadata.get("tag_list"), max_tags=80))
            artist_profile = set(metadata.get("artist_style_profile", []))
            track_artists = self._metadata_text_set(metadata.get("artist_name"))
            track_albums = self._metadata_text_set(metadata.get("album_name"))
            decade = metadata.get("release_decade") or release_decade_text(metadata.get("release_date"))

            tag_score = self._overlap_score(query_tags, track_tags)
            artist_profile_score = self._overlap_score(query_tags, artist_profile)
            decade_score = 1.0 if decade and decade in query_decades else 0.0
            artist_match = 1.0 if query_artists and query_artists & track_artists else 0.0
            album_match = 1.0 if query_albums and query_albums & track_albums else 0.0
            source_score = source_hits[track_id] / max_source_hits

            final_score = (
                embedding_score
                + self.hybrid_bm25_weight * lexical_scores[track_id]
                + self.rerank_tag_weight * tag_score
                + self.rerank_artist_profile_weight * artist_profile_score
                + self.rerank_decade_weight * decade_score
                + self.hybrid_artist_match_weight * artist_match
                + self.hybrid_album_match_weight * album_match
                + self.hybrid_multi_source_weight * source_score
            )
            reranked.append(
                (
                    track_id,
                    final_score,
                    embedding_score,
                    lexical_scores[track_id],
                    source_hits[track_id],
                    best_rank[track_id],
                )
            )

        reranked.sort(key=lambda item: (-item[1], -item[2], -item[3], -item[4], item[5], item[0]))
        return [track_id for track_id, *_ in reranked[:topk]]

    def _gemini_multi_query_fusion_retrieval(
        self,
        conversation_text: str,
        session_id: Optional[str],
        turn_number: Optional[int],
        topk: int = 20,
    ) -> List[str]:
        """Retrieve with one BERT query per Gemini reference and fuse rankings."""
        if self.gemini_expander is None:
            return self.retrieval.text_to_item_retrieval(conversation_text, topk=topk)

        gemini_tracks = self.gemini_expander.expand_tracks(
            conversation_text,
            session_id=session_id,
            turn_number=turn_number,
        )
        if self.gemini_max_reference_tracks is not None:
            gemini_tracks = gemini_tracks[:self.gemini_max_reference_tracks]

        query_texts = [self._gemini_reference_to_query(track) for track in gemini_tracks]
        weights = [self.gemini_reference_weight] * len(query_texts)

        if self.include_original_query_in_fusion:
            query_texts.insert(0, conversation_text)
            weights.insert(0, self.original_query_weight)

        if hasattr(self.retrieval, "batch_text_to_item_retrieval_with_scores"):
            ranked_lists = self.retrieval.batch_text_to_item_retrieval_with_scores(
                query_texts,
                topk=self.gemini_topk_per_reference,
            )
        elif hasattr(self.retrieval, "batch_text_to_item_retrieval"):
            ranked_lists = self.retrieval.batch_text_to_item_retrieval(
                query_texts,
                topk=self.gemini_topk_per_reference,
            )
        else:
            ranked_lists = [
                self.retrieval.text_to_item_retrieval(query, topk=self.gemini_topk_per_reference)
                for query in query_texts
            ]

        if self.gemini_fusion_method == "hybrid_structured_rerank":
            lexical_ranked_lists = []
            if self.lexical_retrieval is not None:
                if hasattr(self.lexical_retrieval, "batch_text_to_item_retrieval"):
                    lexical_ranked_lists = self.lexical_retrieval.batch_text_to_item_retrieval(
                        query_texts,
                        topk=self.hybrid_bm25_topk_per_reference,
                    )
                else:
                    lexical_ranked_lists = [
                        self.lexical_retrieval.text_to_item_retrieval(
                            query,
                            topk=self.hybrid_bm25_topk_per_reference,
                        )
                        for query in query_texts
                    ]
            return self._hybrid_structured_rerank(
                ranked_lists,
                lexical_ranked_lists,
                gemini_tracks,
                topk=topk,
            )

        if self.gemini_fusion_method == "max_similarity_structured_rerank":
            return self._max_similarity_structured_rerank(
                ranked_lists,
                gemini_tracks,
                topk=topk,
            )

        if self.gemini_fusion_method == "max_similarity":
            return self._max_similarity_fusion(ranked_lists, topk=topk)

        return self._reciprocal_rank_fusion(ranked_lists, weights, topk=topk)
        
    def _reset_session_memory(self):
        """Clear all messages stored in the current session memory.
        """
        self.session_memory = []

    def _upload_session_memory(self, chat_history: List[Dict[str, Any]]):
        """Upload the session memory to the database.
        """
        self.session_memory = chat_history

    def _get_system_prompt(self, user_id: Optional[str] = None) -> str:
        """Build the system prompt, optionally personalized with a user profile.
        Args:
            user_id: Optional user identifier. When provided, includes a personalization segment derived from the user's profile.
        Returns:
            The final system prompt string used for the LLM.
        """
        system_prompt = self.role_prompt["role_play"] + self.role_prompt["response_generation"]
        if user_id:
            user_profile_str = self.user_db.id_to_profile_str(user_id)
            system_prompt += self.role_prompt["personalization"] + '\n' + user_profile_str
        return system_prompt

    def chat(self, user_query: str, user_id: Optional[str] = None) -> dict[str, Any]:
        """Run a single CRS turn: retrieve items and generate a response.
        Args:
            user_query: The user's latest message or request.
            user_id: Optional user identifier for personalization.
        Returns:
            A dictionary with keys:
                - user_id: The user identifier (may be None).
                - user_query: Echo of the input query.
                - retrieval_items: List of retrieved item IDs (top candidates).
                - recommend_item: Metadata for the top recommended item.
                - response: The generated assistant response string.
        """
        self.session_memory.append({"role": "user", "content": user_query})
        # stage0. system prompt
        system_prompt = "" if self.retrieval_only else self._get_system_prompt(user_id)
        # stage1. retrieval
        retrieval_input = "\n".join([f"{conversation['role']}: {conversation['content']}" for conversation in self.session_memory])
        retrieval_items = self.retrieval.text_to_item_retrieval(retrieval_input, topk=20)
        recommend_item = self.item_db.id_to_metadata(retrieval_items[0])
        if self.retrieval_only:
            return {
                "user_id": user_id,
                "user_query": user_query,
                "retrieval_items": retrieval_items,
                "recommend_item": recommend_item,
                "response": "",
            }
        # stage2. response generation
        response = self.lm.response_generation(system_prompt, self.session_memory, recommend_item)
        return {
            "user_id": user_id,
            "user_query": user_query,
            "retrieval_items": retrieval_items,
            "recommend_item": recommend_item,
            "response": response,
        }

    def batch_chat(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run multiple CRS turns in batch: retrieve items and generate responses.
        Args:
            batch_data: List of dictionaries, each containing:
                - user_query: The user's latest message or request.
                - user_id: Optional user identifier for personalization.
                - session_memory: List of chat history messages.
        Returns:
            A list of dictionaries, each with keys:
                - user_id: The user identifier (may be None).
                - user_query: Echo of the input query.
                - retrieval_items: List of retrieved item IDs (top candidates).
                - recommend_item: Metadata for the top recommended item.
                - response: The generated assistant response string.
        """
        # Prepare batch inputs
        sys_prompts = []
        retrieval_requests = []
        session_memories = []

        # Original version
        # for data in batch_data:
        #     user_query = data['user_query']
        #     user_id = data.get('user_id')
        #     session_memory = data['session_memory'].copy()
        #     session_memory.append({"role": "user", "content": user_query})

        #     sys_prompts.append(self._get_system_prompt(user_id))
        #     retrieval_input = "\n".join([f"{conversation['role']}: {conversation['content']}" for conversation in session_memory])
        #     retrieval_inputs.append(retrieval_input)
        #     session_memories.append(session_memory)

        # Gemini expansion version
        for data in batch_data:
            user_query = data['user_query']
            user_id = data.get('user_id')
            session_memory = data['session_memory'].copy()
            session_memory.append({"role": "user", "content": user_query})

            if not self.retrieval_only:
                sys_prompts.append(self._get_system_prompt(user_id))
            # Baseline retrieval method 
            # retrieval_input = "\n".join([f"{conversation['role']}: {conversation['content']}" for conversation in session_memory])
            # retrieval_inputs.append(retrieval_input)
            conversation_text = "\n".join([
                f"{conversation['role']}: {conversation['content']}"
                for conversation in session_memory
            ])

            if (
                self.use_gemini_expansion
                and self.gemini_expander is not None
                and self.gemini_expansion_mode == "multi_query_fusion"
            ):
                session_id = data.get("session_id")
                turn_number = data.get("turn_number")

                try:
                    retrieval_items = self._gemini_multi_query_fusion_retrieval(
                        conversation_text,
                        session_id=session_id,
                        turn_number=turn_number,
                        topk=20,
                    )
                    retrieval_requests.append(retrieval_items)
                except Exception as e:
                    print(f"Gemini multi-query fusion failed, using original query. Error: {e}")
                    retrieval_requests.append(conversation_text)

            elif self.use_gemini_expansion and self.gemini_expander is not None:
                session_id = data.get("session_id")
                turn_number = data.get("turn_number")

                try:
                    if self.gemini_expansion_mode == "controlled_keyword_query":
                        gemini_query = self.gemini_expander.expand_controlled_query(
                            conversation_text,
                            session_id=session_id,
                            turn_number=turn_number,
                        )
                    else:
                        gemini_query = self.gemini_expander.expand(
                            conversation_text,
                            session_id=session_id,
                            turn_number=turn_number,
                        )

                    retrieval_input = (
                        conversation_text
                        + "\n\nGemini-controlled BM25 search terms:\n"
                        + "\n\n".join(
                            [gemini_query] * max(1, self.gemini_keyword_block_weight)
                        )
                    )
                except Exception as e:
                    print(f"Gemini expansion failed, using original query. Error: {e}")
                    retrieval_input = conversation_text

                retrieval_requests.append(retrieval_input)
            else:
                retrieval_input = conversation_text
                retrieval_requests.append(retrieval_input)

            session_memories.append(session_memory)

        # Stage 1: Batch retrieval
        batch_retrieval_items = [None] * len(retrieval_requests)
        retrieval_topk = self.query_type_candidate_topk if self.use_query_type_routing else 20
        text_request_indices = [
            i for i, request in enumerate(retrieval_requests)
            if isinstance(request, str)
        ]
        text_requests = [retrieval_requests[i] for i in text_request_indices]

        if text_requests:
            if hasattr(self.retrieval, 'batch_text_to_item_retrieval'):
                text_retrieval_items = self.retrieval.batch_text_to_item_retrieval(text_requests, topk=retrieval_topk)
            else:
                # Fallback to sequential retrieval if batch method not available
                text_retrieval_items = [
                    self.retrieval.text_to_item_retrieval(inp, topk=retrieval_topk)
                    for inp in text_requests
                ]

            for request_index, retrieval_items in zip(text_request_indices, text_retrieval_items):
                if self.use_query_type_routing:
                    batch_retrieval_items[request_index] = self._query_type_route_rerank(
                        retrieval_items,
                        session_memories[request_index],
                        retrieval_requests[request_index],
                        topk=20,
                    )
                else:
                    batch_retrieval_items[request_index] = retrieval_items

        for i, request in enumerate(retrieval_requests):
            if not isinstance(request, str):
                if self.use_query_type_routing:
                    batch_retrieval_items[i] = self._query_type_route_rerank(
                        request,
                        session_memories[i],
                        "",
                        topk=20,
                    )
                else:
                    batch_retrieval_items[i] = request

        recommend_items = [
            None if self.retrieval_only else self.item_db.id_to_metadata(items[0])
            for items in batch_retrieval_items
        ]

        # Stage 2: Batch response generation
        if self.retrieval_only:
            responses = [""] * len(batch_data)
        elif hasattr(self.lm, 'batch_response_generation'):
            responses = self.lm.batch_response_generation(sys_prompts, session_memories, recommend_items)
        else:
            # Fallback to sequential generation if batch method not available
            responses = [self.lm.response_generation(sys_prompts[i], session_memories[i], recommend_items[i])
                        for i in range(len(batch_data))]

        # Prepare results
        results = []
        for i, data in enumerate(batch_data):
            results.append({
                "user_id": data.get('user_id'),
                "user_query": data['user_query'],
                "retrieval_items": batch_retrieval_items[i],
                "recommend_item": recommend_items[i],
                "response": responses[i],
            })

        return results
