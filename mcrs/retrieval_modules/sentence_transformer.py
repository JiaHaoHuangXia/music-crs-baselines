"""Sentence-transformer embedding retrieval for music track metadata."""

import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer

from mcrs.controlled_tags import metadata_field


class SENTENCE_TRANSFORMER_MODEL:
    """Embedding retriever using a sentence-transformers model.

    This is intended as a stronger semantic-similarity alternative to the
    baseline mean-pooled bert-base-uncased retriever.
    """

    def __init__(
        self,
        dataset_name,
        split_types,
        corpus_types,
        cache_dir: str = "./cache",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.corpus_types = corpus_types
        self.corpus_name = "_".join(corpus_types)
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.model_cache_name = model_name.replace("/", "__")
        self.index_dir = os.path.join(
            self.cache_dir,
            "sentence_transformer",
            self.model_cache_name,
            self.corpus_name,
        )
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.metadata_dict = self._load_corpus()
        self.model = SentenceTransformer(self.model_name, device=self.device)

        if os.path.exists(os.path.join(self.index_dir, "embeddings.pt")) and \
           os.path.exists(os.path.join(self.index_dir, "track_ids.json")):
            self.embeddings, self.track_ids = self._load_index()
        else:
            self.build_index()
            self.embeddings, self.track_ids = self._load_index()

    def _load_index(self) -> Tuple[torch.Tensor, List[str]]:
        embeddings = torch.load(os.path.join(self.index_dir, "embeddings.pt"), map_location="cpu")
        with open(os.path.join(self.index_dir, "track_ids.json"), "r", encoding="utf-8") as file:
            track_ids = json.load(file)
        return embeddings, track_ids

    def _load_corpus(self) -> Dict[str, Dict]:
        metadata_dataset = load_dataset(self.dataset_name)
        metadata_concat_dataset = concatenate_datasets(
            [metadata_dataset[split_type] for split_type in self.split_types]
        )
        return {item["track_id"]: item for item in metadata_concat_dataset}

    def _stringify_metadata(self, metadata: Dict[str, object]) -> str:
        parts = []
        for corpus_type in self.corpus_types:
            entity = metadata_field(metadata, corpus_type)
            if isinstance(entity, list):
                entity = ", ".join(entity)
            parts.append(f"{corpus_type}: {entity}")
        return "\n".join(parts)

    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = embeddings.detach().cpu()
        return F.normalize(embeddings, p=2, dim=1)

    def build_index(self) -> None:
        track_ids = list(self.metadata_dict.keys())
        corpus_texts = [
            self._stringify_metadata(self.metadata_dict[track_id])
            for track_id in track_ids
        ]
        os.makedirs(self.index_dir, exist_ok=True)
        embeddings = self._embed_texts(corpus_texts).contiguous()
        torch.save(embeddings, os.path.join(self.index_dir, "embeddings.pt"))
        with open(os.path.join(self.index_dir, "track_ids.json"), "w", encoding="utf-8") as file:
            json.dump(track_ids, file, indent=2)

    def text_to_item_retrieval(self, query: str, topk: int) -> List[str]:
        query_emb = self._embed_texts([query]).squeeze(0)
        scores = torch.matmul(self.embeddings, query_emb)
        topk = min(topk, scores.shape[0])
        top_indices = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids[idx] for idx in top_indices]

    def text_to_item_retrieval_with_scores(self, query: str, topk: int) -> List[Tuple[str, float]]:
        query_emb = self._embed_texts([query]).squeeze(0)
        scores = torch.matmul(self.embeddings, query_emb)
        topk = min(topk, scores.shape[0])
        top_values, top_indices = torch.topk(scores, k=topk)
        return [
            (self.track_ids[idx], float(score))
            for idx, score in zip(top_indices.tolist(), top_values.tolist())
        ]

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int) -> List[List[str]]:
        query_embs = self._embed_texts(queries)
        scores = torch.matmul(self.embeddings, query_embs.T)
        results = []
        topk = min(topk, scores.shape[0])
        for i in range(len(queries)):
            top_indices = torch.topk(scores[:, i], k=topk).indices.tolist()
            results.append([self.track_ids[idx] for idx in top_indices])
        return results

    def batch_text_to_item_retrieval_with_scores(self, queries: List[str], topk: int) -> List[List[Tuple[str, float]]]:
        query_embs = self._embed_texts(queries)
        scores = torch.matmul(self.embeddings, query_embs.T)
        results = []
        topk = min(topk, scores.shape[0])
        for i in range(len(queries)):
            top_values, top_indices = torch.topk(scores[:, i], k=topk)
            results.append([
                (self.track_ids[idx], float(score))
                for idx, score in zip(top_indices.tolist(), top_values.tolist())
            ])
        return results
