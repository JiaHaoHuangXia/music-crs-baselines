import os
import math
import torch
from collections import defaultdict
from typing import Optional, Any, List, Dict
from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import load_retrieval_module
from mcrs.controlled_tags import controlled_tags, normalized_music_tags

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
        gemini_expansion_mode: str = "tag_query",
        gemini_topk_per_reference: int = 50,
        gemini_rrf_k: int = 60,
        include_original_query_in_fusion: bool = False,
        original_query_weight: float = 2.0,
        gemini_reference_weight: float = 1.0,
        gemini_max_reference_tracks: Optional[int] = None,
        gemini_fusion_method: str = "rrf",
        gemini_tag_rerank_weight: float = 0.15,
        gemini_popularity_weight: float = 0.02,
        gemini_artist_penalty: float = 0.05,
        gemini_max_tracks_per_artist: int = 3,
        gemini_tag_candidate_topk: int = 50,
        gemini_tag_candidate_weight: float = 0.75,
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

        # Gemini query expansion
        self.use_gemini_expansion = use_gemini_expansion
        self.gemini_expansion_mode = gemini_expansion_mode
        self.gemini_topk_per_reference = gemini_topk_per_reference
        self.gemini_rrf_k = gemini_rrf_k
        self.include_original_query_in_fusion = include_original_query_in_fusion
        self.original_query_weight = original_query_weight
        self.gemini_reference_weight = gemini_reference_weight
        self.gemini_max_reference_tracks = gemini_max_reference_tracks
        self.gemini_fusion_method = gemini_fusion_method
        self.gemini_tag_rerank_weight = gemini_tag_rerank_weight
        self.gemini_popularity_weight = gemini_popularity_weight
        self.gemini_artist_penalty = gemini_artist_penalty
        self.gemini_max_tracks_per_artist = gemini_max_tracks_per_artist
        self.gemini_tag_candidate_topk = gemini_tag_candidate_topk
        self.gemini_tag_candidate_weight = gemini_tag_candidate_weight
        valid_gemini_modes = {"tag_query", "multi_query_fusion"}
        if self.gemini_expansion_mode not in valid_gemini_modes:
            raise ValueError(
                f"Unknown gemini_expansion_mode='{self.gemini_expansion_mode}'. "
                f"Expected one of {sorted(valid_gemini_modes)}."
            )
        valid_fusion_methods = {
            "rrf",
            "max_similarity",
            "max_similarity_tag_rerank",
            "tag_candidate_rerank",
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
            )
        else:
            self.gemini_expander = None

        self.lm = load_lm_module(self.lm_type, self.device, self.attn_implementation, self.dtype)
        self.retrieval = load_retrieval_module(self.retrieval_type, self.item_db_name, self.track_split_types, self.corpus_types, self.cache_dir)
        self.item_db = MusicCatalogDB(self.item_db_name, self.track_split_types, self.corpus_types)
        self.user_db = UserProfileDB(self.user_db_name, self.user_split_types)
        self._tag_candidate_cache = None
        self._tag_idf_cache = None
        self._tag_embedding_cache = {}
        self._track_tag_specificity_cache = {}
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "system_prompts")
        self.role_prompt = {
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
            "controlled_tag_list": clean_join(controlled_tags(track.get("tag_list"))),
            "release_date": clean_join(track.get("release_date")),
        }
        return "\n".join(f"{field}: {values.get(field, '')}" for field in self.corpus_types)

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

    def _tag_idf(self) -> tuple[Dict[str, float], float]:
        """Compute IDF weights for normalized catalog tags."""
        if self._tag_idf_cache is not None:
            return self._tag_idf_cache

        metadata_dict = getattr(self.retrieval, "metadata_dict", {})
        document_count = max(len(metadata_dict), 1)
        document_frequency = defaultdict(int)

        for metadata in metadata_dict.values():
            for tag in set(normalized_music_tags(metadata.get("tag_list"), max_tags=80)):
                document_frequency[tag] += 1

        idf = {
            tag: math.log((1 + document_count) / (1 + frequency)) + 1
            for tag, frequency in document_frequency.items()
        }
        default_idf = sum(idf.values()) / len(idf) if idf else 1.0
        self._tag_idf_cache = (idf, default_idf)
        return self._tag_idf_cache

    def _tag_embeddings(self, tags: List[str]) -> Dict[str, torch.Tensor]:
        """Return cached normalized embeddings for individual tags."""
        missing_tags = [tag for tag in tags if tag not in self._tag_embedding_cache]
        if missing_tags and hasattr(self.retrieval, "_embed_texts"):
            embeddings = self.retrieval._embed_texts(missing_tags)
            for tag, embedding in zip(missing_tags, embeddings):
                self._tag_embedding_cache[tag] = embedding

        return {
            tag: self._tag_embedding_cache[tag]
            for tag in tags
            if tag in self._tag_embedding_cache
        }

    def _semantic_idf_tag_similarity(self, reference_tags: List[str], catalog_tags: List[str]) -> float:
        """Compare tag lists with normalization, semantic similarity, and IDF weights."""
        if not reference_tags or not catalog_tags:
            return 0.0

        catalog_tag_set = set(catalog_tags)
        idf, default_idf = self._tag_idf()
        reference_embeddings = self._tag_embeddings(reference_tags)
        catalog_embeddings = self._tag_embeddings(catalog_tags)

        weighted_score = 0.0
        total_weight = 0.0
        for reference_tag in reference_tags:
            weight = idf.get(reference_tag, default_idf)
            total_weight += weight

            if reference_tag in catalog_tag_set:
                best_score = 1.0
            elif reference_tag in reference_embeddings and catalog_embeddings:
                reference_embedding = reference_embeddings[reference_tag]
                best_score = max(
                    float(torch.dot(reference_embedding, catalog_embedding))
                    for catalog_embedding in catalog_embeddings.values()
                )
                best_score = max(0.0, min(best_score, 1.0))
            else:
                best_score = 0.0

            weighted_score += weight * best_score

        if total_weight == 0:
            return 0.0
        return weighted_score / total_weight

    def _controlled_tag_overlap_score(
        self,
        reference_track: Dict[str, Any],
        catalog_track_id: str,
    ) -> float:
        """Score how much a catalog track's tags match one Gemini reference."""
        if not hasattr(self.retrieval, "metadata_dict"):
            return 0.0

        catalog_metadata = self.retrieval.metadata_dict.get(catalog_track_id)
        if not catalog_metadata:
            return 0.0

        reference_tags = normalized_music_tags(reference_track.get("tag_list"), max_tags=20)
        catalog_tags = normalized_music_tags(catalog_metadata.get("tag_list"), max_tags=80)
        return self._semantic_idf_tag_similarity(reference_tags, catalog_tags)

    def _catalog_metadata(self, track_id: str) -> Dict[str, Any]:
        """Return catalog metadata from the retrieval index when available."""
        if hasattr(self.retrieval, "metadata_dict"):
            return self.retrieval.metadata_dict.get(track_id, {})
        try:
            return self.item_db.id_to_metadata(track_id)
        except Exception:
            return {}

    def _track_artist_key(self, track_id: str) -> str:
        metadata = self._catalog_metadata(track_id)
        artist = metadata.get("artist_name", "")
        if isinstance(artist, list):
            artist = ", ".join(str(item) for item in artist)
        return str(artist).strip().lower()

    def _track_popularity_score(self, track_id: str) -> float:
        metadata = self._catalog_metadata(track_id)
        try:
            popularity = float(metadata.get("popularity", 0.0) or 0.0)
        except (TypeError, ValueError):
            popularity = 0.0
        return popularity / 100.0

    def _track_tag_specificity_score(self, track_id: str) -> float:
        """Average IDF of a track's normalized tags."""
        if track_id in self._track_tag_specificity_cache:
            return self._track_tag_specificity_cache[track_id]

        metadata = self._catalog_metadata(track_id)
        tags = normalized_music_tags(metadata.get("tag_list"), max_tags=80)
        if not tags:
            self._track_tag_specificity_cache[track_id] = 0.0
            return 0.0

        idf, default_idf = self._tag_idf()
        score = sum(idf.get(tag, default_idf) for tag in tags) / len(tags)
        self._track_tag_specificity_cache[track_id] = score
        return score

    def _build_tag_candidate_cache(self) -> Dict[str, list[tuple[str, float]]]:
        """Build an inverted index from normalized tags to catalog tracks."""
        if self._tag_candidate_cache is not None:
            return self._tag_candidate_cache

        tag_to_tracks = defaultdict(list)
        metadata_dict = getattr(self.retrieval, "metadata_dict", {})
        for track_id, metadata in metadata_dict.items():
            tags = normalized_music_tags(metadata.get("tag_list"), max_tags=80)
            specificity = self._track_tag_specificity_score(track_id)
            for tag in tags:
                tag_to_tracks[tag].append((track_id, specificity))

        for tag, tracks in tag_to_tracks.items():
            tracks.sort(key=lambda item: (-item[1], item[0]))

        self._tag_candidate_cache = tag_to_tracks
        return self._tag_candidate_cache

    def _tag_candidate_lists(
        self,
        gemini_tracks: List[Dict[str, Any]],
    ) -> List[List[tuple[str, float]]]:
        """Generate extra candidate tracks from Gemini controlled tags."""
        tag_to_tracks = self._build_tag_candidate_cache()
        tag_ranked_lists = []
        idf, default_idf = self._tag_idf()

        for reference_track in gemini_tracks:
            reference_tags = set(normalized_music_tags(reference_track.get("tag_list"), max_tags=20))
            if not reference_tags:
                tag_ranked_lists.append([])
                continue

            candidate_scores = {}
            candidate_specificity = {}
            total_reference_weight = sum(idf.get(tag, default_idf) for tag in reference_tags)
            for tag in reference_tags:
                tag_weight = idf.get(tag, default_idf)
                for track_id, specificity in tag_to_tracks.get(tag, []):
                    candidate_scores[track_id] = candidate_scores.get(track_id, 0.0) + tag_weight
                    candidate_specificity[track_id] = max(
                        specificity,
                        candidate_specificity.get(track_id, 0.0),
                    )

            ranked = sorted(
                candidate_scores,
                key=lambda track_id: (
                    -(candidate_scores[track_id] / max(total_reference_weight, 1e-9)),
                    -candidate_specificity[track_id],
                    track_id,
                ),
            )
            tag_ranked_lists.append([
                (track_id, candidate_scores[track_id] / max(total_reference_weight, 1e-9))
                for track_id in ranked[: self.gemini_tag_candidate_topk]
            ])

        return tag_ranked_lists

    def _select_with_artist_diversity(
        self,
        ranked_track_ids: List[str],
        topk: int,
        artist_caps: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        """Prefer artist diversity while still backfilling to topk if needed."""
        artist_caps = artist_caps or {}
        default_cap = self.gemini_max_tracks_per_artist
        if default_cap <= 0 and not artist_caps:
            return ranked_track_ids[:topk]

        selected = []
        artist_counts = defaultdict(int)
        skipped = []

        for track_id in ranked_track_ids:
            artist_key = self._track_artist_key(track_id)
            artist_cap = artist_caps.get(artist_key, default_cap)
            if artist_cap <= 0 or artist_counts[artist_key] < artist_cap:
                selected.append(track_id)
                artist_counts[artist_key] += 1
            else:
                skipped.append(track_id)
            if len(selected) >= topk:
                return selected

        for track_id in skipped:
            if track_id not in selected:
                selected.append(track_id)
            if len(selected) >= topk:
                break

        return selected[:topk]

    def _adaptive_artist_caps(self, gemini_tracks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Allow more final tracks from artists repeatedly suggested by Gemini."""
        artist_counts = defaultdict(int)
        for track in gemini_tracks:
            artists = track.get("artist_name", "")
            if isinstance(artists, list):
                artist = ", ".join(str(item) for item in artists)
            else:
                artist = str(artists)
            artist_key = artist.strip().lower()
            if artist_key:
                artist_counts[artist_key] += 1

        caps = {}
        for artist_key, count in artist_counts.items():
            if count >= 5:
                caps[artist_key] = 14
            elif count == 4:
                caps[artist_key] = 10
            elif count == 3:
                caps[artist_key] = 7
            elif count == 2:
                caps[artist_key] = 5

        return caps

    def _max_similarity_tag_rerank_fusion(
        self,
        ranked_lists: List[List[Any]],
        gemini_tracks: List[Dict[str, Any]],
        topk: int,
    ) -> List[str]:
        """Fuse by embedding score, then add a controlled-tag overlap bonus."""
        fused_scores = {}
        embedding_scores = {}
        tag_scores = {}
        best_rank = {}

        for list_idx, ranked_items in enumerate(ranked_lists):
            reference_track = gemini_tracks[list_idx] if list_idx < len(gemini_tracks) else {}
            for rank, item in enumerate(ranked_items, start=1):
                if isinstance(item, tuple):
                    track_id, embedding_score = item
                else:
                    track_id, embedding_score = item, 0.0

                tag_score = self._controlled_tag_overlap_score(reference_track, track_id)
                combined_score = embedding_score + (self.gemini_tag_rerank_weight * tag_score)

                if track_id not in fused_scores or combined_score > fused_scores[track_id]:
                    fused_scores[track_id] = combined_score
                    embedding_scores[track_id] = embedding_score
                    tag_scores[track_id] = tag_score

                best_rank[track_id] = min(best_rank.get(track_id, rank), rank)

        ranked_track_ids = sorted(
            fused_scores,
            key=lambda track_id: (
                -fused_scores[track_id],
                -embedding_scores[track_id],
                -tag_scores[track_id],
                best_rank[track_id],
                track_id,
            ),
        )
        return ranked_track_ids[:topk]

    def _tag_candidate_rerank_fusion(
        self,
        embedding_ranked_lists: List[List[Any]],
        tag_ranked_lists: List[List[Any]],
        gemini_tracks: List[Dict[str, Any]],
        topk: int,
    ) -> List[str]:
        """Combine embedding candidates, tag candidates, popularity, and artist diversity."""
        fused_scores = defaultdict(float)
        embedding_scores = defaultdict(float)
        tag_scores = defaultdict(float)
        popularity_scores = defaultdict(float)
        best_rank = {}

        for list_idx, ranked_items in enumerate(embedding_ranked_lists):
            reference_track = gemini_tracks[list_idx] if list_idx < len(gemini_tracks) else {}
            for rank, item in enumerate(ranked_items, start=1):
                if isinstance(item, tuple):
                    track_id, embedding_score = item
                else:
                    track_id, embedding_score = item, 0.0

                tag_score = self._controlled_tag_overlap_score(reference_track, track_id)
                popularity_score = self._track_popularity_score(track_id)
                combined_score = (
                    embedding_score
                    + (self.gemini_tag_rerank_weight * tag_score)
                    + (self.gemini_popularity_weight * popularity_score)
                )

                if combined_score > fused_scores[track_id]:
                    fused_scores[track_id] = combined_score
                embedding_scores[track_id] = max(embedding_scores[track_id], embedding_score)
                tag_scores[track_id] = max(tag_scores[track_id], tag_score)
                popularity_scores[track_id] = max(popularity_scores[track_id], popularity_score)
                best_rank[track_id] = min(best_rank.get(track_id, rank), rank)

        for ranked_items in tag_ranked_lists:
            for rank, item in enumerate(ranked_items, start=1):
                if isinstance(item, tuple):
                    track_id, tag_candidate_score = item
                else:
                    track_id, tag_candidate_score = item, 0.0

                popularity_score = self._track_popularity_score(track_id)
                combined_score = (
                    self.gemini_tag_candidate_weight * tag_candidate_score
                    + self.gemini_popularity_weight * popularity_score
                )

                if combined_score > fused_scores[track_id]:
                    fused_scores[track_id] = combined_score
                tag_scores[track_id] = max(tag_scores[track_id], tag_candidate_score)
                popularity_scores[track_id] = max(popularity_scores[track_id], popularity_score)
                best_rank[track_id] = min(best_rank.get(track_id, 1000 + rank), 1000 + rank)

        ranked_track_ids = sorted(
            fused_scores,
            key=lambda track_id: (
                -fused_scores[track_id],
                -embedding_scores[track_id],
                -tag_scores[track_id],
                -popularity_scores[track_id],
                best_rank[track_id],
                track_id,
            ),
        )
        return self._select_with_artist_diversity(
            ranked_track_ids,
            topk=topk,
            artist_caps=self._adaptive_artist_caps(gemini_tracks),
        )

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

        if self.gemini_fusion_method == "tag_candidate_rerank":
            tag_ranked_lists = self._tag_candidate_lists(gemini_tracks)
            return self._tag_candidate_rerank_fusion(
                ranked_lists,
                tag_ranked_lists,
                gemini_tracks,
                topk=topk,
            )

        if self.gemini_fusion_method == "max_similarity_tag_rerank":
            return self._max_similarity_tag_rerank_fusion(
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
        system_prompt = self._get_system_prompt(user_id)
        # stage1. retrieval
        retrieval_input = "\n".join([f"{conversation['role']}: {conversation['content']}" for conversation in self.session_memory])
        retrieval_items = self.retrieval.text_to_item_retrieval(retrieval_input, topk=20)
        recommend_item = self.item_db.id_to_metadata(retrieval_items[0])
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
                    gemini_query = self.gemini_expander.expand(
                        conversation_text,
                        session_id=session_id,
                        turn_number=turn_number,
                    )

                    retrieval_input = (
                        conversation_text
                        + "\n\nGemini-expanded search terms:\n"
                        + gemini_query
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
        text_request_indices = [
            i for i, request in enumerate(retrieval_requests)
            if isinstance(request, str)
        ]
        text_requests = [retrieval_requests[i] for i in text_request_indices]

        if text_requests:
            if hasattr(self.retrieval, 'batch_text_to_item_retrieval'):
                text_retrieval_items = self.retrieval.batch_text_to_item_retrieval(text_requests, topk=20)
            else:
                # Fallback to sequential retrieval if batch method not available
                text_retrieval_items = [
                    self.retrieval.text_to_item_retrieval(inp, topk=20)
                    for inp in text_requests
                ]

            for request_index, retrieval_items in zip(text_request_indices, text_retrieval_items):
                batch_retrieval_items[request_index] = retrieval_items

        for i, request in enumerate(retrieval_requests):
            if not isinstance(request, str):
                batch_retrieval_items[i] = request

        recommend_items = [self.item_db.id_to_metadata(items[0]) for items in batch_retrieval_items]

        # Stage 2: Batch response generation
        if hasattr(self.lm, 'batch_response_generation'):
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
