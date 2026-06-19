"""BERT embedding-based retrieval utilities for music track metadata.

This module embeds selected metadata fields using a BERT encoder, caches the
embeddings to disk, and retrieves items for a query using cosine similarity.
Token embeddings are averaged (mean pooling) to form a global embedding.
"""
import os
import json
from typing import List, Tuple, Dict

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn.functional as F

from mcrs.controlled_tags import metadata_field
from mcrs.style_profiles import attach_artist_style_profiles


class BERT_MODEL:
    """BERT-based embedding retriever over track metadata.
    Builds an embedding index from specified corpus fields (e.g., `track_name`,
    `artist_name`, `album_name`) and provides text-to-item retrieval via cosine
    similarity over mean-pooled token embeddings.
    """
    def __init__(self,
        dataset_name,
        split_types,
        corpus_types,
        cache_dir: str = "./cache",
        model_name: str = "bert-base-uncased",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 128
    ) -> None:
        """Initialize the BERT retriever.
        Args:
            dataset_name: Hugging Face dataset name containing track metadata.
            split_types: Dataset splits to load and concatenate.
            corpus_types: Metadata fields to include in the text corpus.
            cache_dir: Directory to cache the embedding index and artifacts.
            model_name: Hugging Face model id for the encoder.
            device: Torch device string. If None, chooses CUDA if available else CPU.
            batch_size: Batch size for embedding computation when building index.
            max_length: Max sequence length for tokenization.
        """
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.corpus_types = corpus_types
        self.corpus_name = "_".join(corpus_types)
        self.cache_dir = cache_dir
        self.index_dir = os.path.join(self.cache_dir, "bert", self.corpus_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.metadata_dict = self._load_corpus()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device).eval()

        if os.path.exists(os.path.join(self.index_dir, "embeddings.pt")) and \
           os.path.exists(os.path.join(self.index_dir, "track_ids.json")):
            self.embeddings, self.track_ids = self._load_index()
        else:
            self.build_index()
            self.embeddings, self.track_ids = self._load_index()

    def _load_index(self) -> Tuple[torch.Tensor, List[str]]:
        """Load cached embedding matrix and track id list.
        Returns:
            A tuple of (embeddings [num_items, dim], track_ids).
        """
        embeddings = torch.load(os.path.join(self.index_dir, "embeddings.pt"), map_location="cpu")
        track_ids = json.load(open(os.path.join(self.index_dir, "track_ids.json"), "r"))
        return embeddings, track_ids

    def _load_corpus(self) -> Dict[str, Dict]:
        """Load and combine metadata splits from the configured dataset.
        Returns:
            A mapping from `track_id` to its metadata dictionary.
        """
        metadata_dataset = load_dataset(self.dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in self.split_types])
        metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}
        return attach_artist_style_profiles(metadata_dict)

    def _stringify_metadata(self, metadata: Dict[str, object]) -> str:
        """Convert a metadata dict into a multi-line string for indexing.
        Args:
            metadata: Track metadata with fields listed in `self.corpus_types`.
        Returns:
            A newline-separated string with `field: value` per selected field.
        """
        metadata_str = ""
        for corpus_type in self.corpus_types:
            entity = metadata_field(metadata, corpus_type)
            if isinstance(entity, list):
                entity = ", ".join(entity)
            metadata_str += f"{corpus_type}: {entity}\n"
        return metadata_str

    def _mean_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings with attention mask.
        Args:
            last_hidden_states: [batch, seq_len, hidden]
            attention_mask: [batch, seq_len]
        Returns:
            [batch, hidden] mean-pooled embeddings.
        """
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        summed = torch.sum(last_hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def build_index(self) -> None:
        """Build and persist an embedding index over the loaded corpus."""
        track_ids = list(self.metadata_dict.keys())
        corpus_texts = []
        for track_id in track_ids:
            metadata = self.metadata_dict[track_id]
            corpus_texts.append(self._stringify_metadata(metadata))
        os.makedirs(self.index_dir, exist_ok=True)
        embeddings: List[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(corpus_texts), self.batch_size):
                batch_texts = corpus_texts[start:start + self.batch_size]
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                pooled = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])  # [b, h]
                pooled = F.normalize(pooled, p=2, dim=1)  # store normalized for cosine similarity
                embeddings.append(pooled.detach().cpu())

        embedding_mat = torch.cat(embeddings, dim=0).contiguous()  # [N, h]
        torch.save(embedding_mat, os.path.join(self.index_dir, "embeddings.pt"))
        with open(os.path.join(self.index_dir, "track_ids.json"), "w") as f:
            json.dump(track_ids, f, indent=2)

    def text_to_item_retrieval(self, query: str, topk: int) -> List[str]:
        """Retrieve top-k track IDs for a natural language query via cosine similarity.
        Args:
            query: The user text query to embed and compare against the corpus.
            k: Number of items to retrieve.
        Returns:
            A list of track IDs ordered by decreasing cosine similarity.
        """
        self.model.eval()
        with torch.no_grad():
            batch = self.tokenizer([query], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            query_emb = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])  # [1, h]
            query_emb = F.normalize(query_emb, p=2, dim=1).cpu().squeeze(0)  # [h]
        # cosine similarity since embeddings are L2-normalized: mat @ query
        scores = torch.matmul(self.embeddings, query_emb)  # [N]
        topk = min(topk, scores.shape[0])
        top_indices = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids[i] for i in top_indices]

    def text_to_item_retrieval_with_scores(self, query: str, topk: int) -> List[Tuple[str, float]]:
        """Retrieve top-k track IDs and cosine scores for one query."""
        self.model.eval()
        with torch.no_grad():
            batch = self.tokenizer([query], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            query_emb = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            query_emb = F.normalize(query_emb, p=2, dim=1).cpu().squeeze(0)

        scores = torch.matmul(self.embeddings, query_emb)
        topk = min(topk, scores.shape[0])
        top_values, top_indices = torch.topk(scores, k=topk)
        return [
            (self.track_ids[idx], float(score))
            for idx, score in zip(top_indices.tolist(), top_values.tolist())
        ]

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int) -> List[List[str]]:
        """Retrieve top-k track IDs for multiple queries in batch via cosine similarity.
        Args:
            queries: List of user text queries to embed and compare against the corpus.
            topk: Number of items to retrieve per query.
        Returns:
            A list of lists, where each inner list contains track IDs ordered by decreasing cosine similarity.
        """
        self.model.eval()
        with torch.no_grad():
            batch = self.tokenizer(queries, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            query_embs = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])  # [batch, h]
            query_embs = F.normalize(query_embs, p=2, dim=1).cpu()  # [batch, h]
        # Compute cosine similarity for all queries: [N, h] @ [h, batch] -> [N, batch]
        scores = torch.matmul(self.embeddings, query_embs.T)  # [N, batch]
        results = []
        topk = min(topk, scores.shape[0])
        for i in range(len(queries)):
            top_indices = torch.topk(scores[:, i], k=topk).indices.tolist()
            results.append([self.track_ids[idx] for idx in top_indices])
        return results

    def batch_text_to_item_retrieval_with_scores(self, queries: List[str], topk: int) -> List[List[Tuple[str, float]]]:
        """Retrieve top-k track IDs and cosine scores for multiple queries."""
        self.model.eval()
        with torch.no_grad():
            batch = self.tokenizer(queries, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            query_embs = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            query_embs = F.normalize(query_embs, p=2, dim=1).cpu()

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
