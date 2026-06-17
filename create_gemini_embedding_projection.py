"""
Create a UMAP map of catalog embeddings and Gemini reference embeddings.

This script is intended to be run offline after the matching retrieval embedding
cache exists. It writes a CSV that Streamlit can load without initializing an
embedding model.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Import sklearn/umap before torch on Windows to avoid occasional DLL/import issues.
from sklearn.preprocessing import normalize
from umap import UMAP

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from mcrs.controlled_tags import controlled_tags, metadata_field


DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
SPLIT_TYPES = ["all_tracks"]
MODEL_NAME = "bert-base-uncased"
SENTENCE_TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42


def clean_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    value = str(value).strip()
    return [value] if value else []


def field_to_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def pseudo_track_to_text(track: dict[str, Any], corpus_types: list[str]) -> str:
    values = {
        "track_name": ", ".join(clean_list_field(track.get("track_name"))),
        "artist_name": ", ".join(clean_list_field(track.get("artist_name"))),
        "album_name": ", ".join(clean_list_field(track.get("album_name"))),
        "tag_list": ", ".join(clean_list_field(track.get("tag_list"))),
        "controlled_tag_list": ", ".join(controlled_tags(track.get("tag_list"))),
        "release_date": str(track.get("release_date", "")).strip(),
    }
    return "\n".join(f"{field}: {values.get(field, '')}" for field in corpus_types)


def load_gemini_cache_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        for key in ["recommendations", "songs", "tracks", "reference_songs", "reference_tracks", "items"]:
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"Invalid Gemini cache format: {path}")

    rows = []
    for item in data[:5]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "track_name": clean_list_field(item.get("track_name"))[:1] or ["Unknown Track"],
                "artist_name": clean_list_field(item.get("artist_name"))[:1] or ["Unknown Artist"],
                "album_name": clean_list_field(item.get("album_name"))[:1] or ["Unknown Album"],
                "tag_list": clean_list_field(item.get("tag_list")),
                "release_date": str(item.get("release_date", "")).strip(),
            }
        )
    return rows


def parse_session_turn(path: Path) -> tuple[str, int | None]:
    name = path.stem
    if "_turn_" not in name:
        return name, None
    session_id, turn_text = name.rsplit("_turn_", 1)
    try:
        return session_id, int(turn_text)
    except ValueError:
        return session_id, None


def mean_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    summed = torch.sum(last_hidden_states * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
    max_length: int = 128,
) -> np.ndarray:
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0).numpy()


def embed_texts_sentence_transformer(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return normalize(embeddings)


def get_index_dir(cache_dir: Path, retrieval_type: str, corpus_types: list[str]) -> Path:
    corpus_name = "_".join(corpus_types)
    if retrieval_type == "bert":
        return cache_dir / "bert" / corpus_name
    if retrieval_type == "sentence_transformer":
        model_cache_name = SENTENCE_TRANSFORMER_MODEL_NAME.replace("/", "__")
        return cache_dir / "sentence_transformer" / model_cache_name / corpus_name
    raise ValueError(f"Unsupported projection retrieval type: {retrieval_type}")


def load_embedding_model(retrieval_type: str, device: str):
    if retrieval_type == "bert":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        return tokenizer, model
    if retrieval_type == "sentence_transformer":
        model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME, device=device)
        return None, model
    raise ValueError(f"Unsupported projection retrieval type: {retrieval_type}")


def embed_reference_texts(
    texts: list[str],
    retrieval_type: str,
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if retrieval_type == "bert":
        return embed_texts(
            texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
        )
    if retrieval_type == "sentence_transformer":
        return embed_texts_sentence_transformer(texts, model=model, batch_size=batch_size)
    raise ValueError(f"Unsupported projection retrieval type: {retrieval_type}")


def load_catalog_metadata() -> dict[str, dict[str, Any]]:
    dataset = load_dataset(DATASET_NAME)
    full_dataset = concatenate_datasets([dataset[split] for split in SPLIT_TYPES])
    return {item["track_id"]: item for item in full_dataset}


def assign_broad_genre(tag_text: str) -> str:
    tag_text = str(tag_text).lower()
    if any(term in tag_text for term in ["metal", "hardcore", "screamo", "post-hardcore", "emocore"]):
        return "metal/hardcore"
    if any(term in tag_text for term in ["hip hop", "rap", "trap"]):
        return "hip hop/rap"
    if any(term in tag_text for term in ["electronic", "electronica", "edm", "house", "techno", "dance"]):
        return "electronic/dance"
    if any(term in tag_text for term in ["rock", "punk", "alternative", "indie"]):
        return "rock/indie"
    if any(term in tag_text for term in ["pop", "dance-pop", "synthpop"]):
        return "pop"
    if any(term in tag_text for term in ["jazz", "blues", "soul", "funk"]):
        return "jazz/soul/funk"
    if any(term in tag_text for term in ["classical", "instrumental", "ambient", "soundtrack"]):
        return "ambient/classical"
    return "other"


def build_catalog_rows(
    track_ids: list[str],
    coords: np.ndarray,
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for idx, track_id in enumerate(track_ids):
        item = metadata[track_id]
        tag_list = field_to_text(item.get("tag_list", ""))
        rows.append(
            {
                "point_type": "catalog",
                "projection_method": "",
                "projection_retrieval_type": "",
                "projection_corpus_types": "",
                "session_id": "",
                "turn_number": "",
                "gemini_reference_rank": "",
                "retrieved_rank": "",
                "cosine_similarity": "",
                "track_id": track_id,
                "track_name": field_to_text(item.get("track_name", "")),
                "artist_name": field_to_text(item.get("artist_name", "")),
                "album_name": field_to_text(item.get("album_name", "")),
                "tag_list": tag_list,
                "controlled_tag_list": field_to_text(metadata_field(item, "controlled_tag_list")),
                "release_date": field_to_text(item.get("release_date", "")),
                "broad_genre": assign_broad_genre(tag_list),
                "umap_x": coords[idx, 0],
                "umap_y": coords[idx, 1],
            }
        )
    return rows


def build_ground_truth_rows(
    devset_table_path: Path,
    catalog_rows_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not devset_table_path.exists():
        return []

    df = pd.read_csv(devset_table_path)
    required = {"session_id", "turn_number", "ground_truth_track_id"}
    if not required.issubset(df.columns):
        return []

    rows = []
    for row in df[["session_id", "turn_number", "ground_truth_track_id"]].drop_duplicates().itertuples(index=False):
        base = catalog_rows_by_id.get(row.ground_truth_track_id)
        if base is None:
            continue
        copy = dict(base)
        copy["point_type"] = "ground_truth"
        copy["session_id"] = row.session_id
        copy["turn_number"] = int(row.turn_number)
        rows.append(copy)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Streamlit UMAP data for Gemini reference embeddings.")
    parser.add_argument("--cache_dir", default="./cache", help="Repository cache directory.")
    parser.add_argument(
        "--projection_retrieval_type",
        choices=["bert", "sentence_transformer"],
        default="bert",
        help="Embedding cache/model used for the UMAP projection.",
    )
    parser.add_argument(
        "--corpus_types",
        nargs="+",
        default=["track_name", "artist_name", "album_name", "tag_list", "release_date"],
        help="Corpus fields matching the embedding cache.",
    )
    parser.add_argument(
        "--gemini_cache_dir",
        default="./cache/gemini_expansions_devset_first100",
        help="Folder containing Gemini cache JSON files.",
    )
    parser.add_argument(
        "--devset_table",
        default="./visualize_streamlit/devset_gemini_ground_truth_table.csv",
        help="Optional devset table used to duplicate ground-truth points.",
    )
    parser.add_argument(
        "--output",
        default="./visualize_streamlit/gemini_embedding_projection.csv",
        help="Output CSV consumed by Streamlit.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument(
        "--topk_retrieved_per_reference",
        type=int,
        default=20,
        help="Number of nearest catalog tracks to highlight for each Gemini reference.",
    )
    args = parser.parse_args()

    index_dir = get_index_dir(
        cache_dir=Path(args.cache_dir),
        retrieval_type=args.projection_retrieval_type,
        corpus_types=args.corpus_types,
    )
    embeddings_path = index_dir / "embeddings.pt"
    track_ids_path = index_dir / "track_ids.json"

    if not embeddings_path.exists() or not track_ids_path.exists():
        raise FileNotFoundError(
            f"Missing embedding cache in {index_dir}. "
            "Run retrieval once with matching retrieval_type and corpus_types first."
        )

    catalog_embeddings = torch.load(embeddings_path, map_location="cpu").numpy()
    with track_ids_path.open("r", encoding="utf-8") as file:
        track_ids = json.load(file)

    catalog_embeddings = normalize(catalog_embeddings)
    reducer = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.05,
        metric="cosine",
        random_state=RANDOM_SEED,
        transform_seed=RANDOM_SEED,
    )
    catalog_coords = reducer.fit_transform(catalog_embeddings)

    metadata = load_catalog_metadata()
    catalog_rows = build_catalog_rows(track_ids, catalog_coords, metadata)
    for row in catalog_rows:
        row["projection_method"] = "umap"
        row["projection_retrieval_type"] = args.projection_retrieval_type
        row["projection_corpus_types"] = ", ".join(args.corpus_types)
    catalog_rows_by_id = {row["track_id"]: row for row in catalog_rows}

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_embedding_model(
        retrieval_type=args.projection_retrieval_type,
        device=device,
    )

    gemini_rows = []
    gemini_texts = []
    gemini_cache_files = sorted(Path(args.gemini_cache_dir).glob("*.json"))
    for cache_file in gemini_cache_files:
        session_id, turn_number = parse_session_turn(cache_file)
        try:
            tracks = load_gemini_cache_file(cache_file)
        except Exception as error:
            print(f"Skipping {cache_file}: {error}")
            continue

        for rank, track in enumerate(tracks, start=1):
            gemini_texts.append(pseudo_track_to_text(track, args.corpus_types))
            tag_list = ", ".join(track["tag_list"])
            gemini_rows.append(
                {
                    "point_type": "gemini_reference",
                    "projection_method": "umap",
                    "projection_retrieval_type": args.projection_retrieval_type,
                    "projection_corpus_types": ", ".join(args.corpus_types),
                    "session_id": session_id,
                    "turn_number": turn_number,
                    "gemini_reference_rank": rank,
                    "retrieved_rank": "",
                    "cosine_similarity": "",
                    "track_id": "",
                    "track_name": ", ".join(track["track_name"]),
                    "artist_name": ", ".join(track["artist_name"]),
                    "album_name": ", ".join(track["album_name"]),
                    "tag_list": tag_list,
                    "release_date": track["release_date"],
                    "broad_genre": assign_broad_genre(tag_list),
                }
            )

    retrieved_rows = []
    if gemini_texts:
        gemini_embeddings = embed_reference_texts(
            gemini_texts,
            retrieval_type=args.projection_retrieval_type,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
        )
        gemini_embeddings = normalize(gemini_embeddings)
        gemini_coords = reducer.transform(gemini_embeddings)
        for idx, row in enumerate(gemini_rows):
            row["umap_x"] = gemini_coords[idx, 0]
            row["umap_y"] = gemini_coords[idx, 1]

        similarities = np.matmul(gemini_embeddings, catalog_embeddings.T)
        topk = min(args.topk_retrieved_per_reference, similarities.shape[1])
        for gemini_idx, gemini_row in enumerate(gemini_rows):
            top_indices = np.argsort(-similarities[gemini_idx])[:topk]
            for retrieved_rank, catalog_idx in enumerate(top_indices, start=1):
                track_id = track_ids[catalog_idx]
                retrieved_row = dict(catalog_rows_by_id[track_id])
                retrieved_row["point_type"] = "retrieved_track"
                retrieved_row["session_id"] = gemini_row["session_id"]
                retrieved_row["turn_number"] = gemini_row["turn_number"]
                retrieved_row["gemini_reference_rank"] = gemini_row["gemini_reference_rank"]
                retrieved_row["retrieved_rank"] = retrieved_rank
                retrieved_row["cosine_similarity"] = float(similarities[gemini_idx, catalog_idx])
                retrieved_rows.append(retrieved_row)

    ground_truth_rows = build_ground_truth_rows(Path(args.devset_table), catalog_rows_by_id)
    output_rows = catalog_rows + ground_truth_rows + retrieved_rows + gemini_rows
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved {len(output_rows)} points to {output_path}")
    print("Projection method: UMAP")
    print(f"Projection retrieval type: {args.projection_retrieval_type}")
    print(f"Corpus fields: {args.corpus_types}")


if __name__ == "__main__":
    main()
