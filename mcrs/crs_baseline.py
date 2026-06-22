import os
import torch
from collections import defaultdict
from typing import Optional, Any, List, Dict
from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import load_retrieval_module

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
        valid_gemini_modes = {"tag_query", "multi_query_fusion"}
        if self.gemini_expansion_mode not in valid_gemini_modes:
            raise ValueError(
                f"Unknown gemini_expansion_mode='{self.gemini_expansion_mode}'. "
                f"Expected one of {sorted(valid_gemini_modes)}."
            )
        valid_fusion_methods = {"rrf", "max_similarity"}
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

        return (
            f"track_name: {clean_join(track.get('track_name'))}\n"
            f"artist_name: {clean_join(track.get('artist_name'))}\n"
            f"album_name: {clean_join(track.get('album_name'))}\n"
            f"tag_list: {clean_join(track.get('tag_list'))}\n"
            f"release_date: {clean_join(track.get('release_date'))}"
        )

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

    def _format_retrieval_context(self, data: Dict[str, Any]) -> str:
        """Build optional structured context for retrieval-only ranking."""
        context_parts = []
        conversation_goal = data.get("conversation_goal")
        if conversation_goal:
            context_parts.append(f"conversation_goal: {conversation_goal}")
        user_profile = data.get("user_profile")
        if user_profile:
            context_parts.append(f"user_profile: {user_profile}")
        turn_number = data.get("turn_number")
        if turn_number is not None:
            context_parts.append(f"turn_number: {turn_number}")
        return "\n".join(context_parts)

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
            retrieval_context = self._format_retrieval_context(data)
            if retrieval_context:
                conversation_text = retrieval_context + "\n" + conversation_text

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
