import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import sklearn before torch on Windows to avoid possible DLL/import issues
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, concatenate_datasets


# ============================================================
# CONFIG
# ============================================================

GEMINI_CACHE_DIR = Path("./cache/gemini_expansions")

BERT_CACHE_DIR = Path("./cache/bert/track_name_artist_name_album_name_tag_list_release_date")

DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
SPLIT_TYPES = ["all_tracks"]

CORPUS_TYPES = [
    "track_name",
    "artist_name",
    "album_name",
    "tag_list",
    "release_date",
]

MODEL_NAME = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOPK_PER_GEMINI_TRACK = 10

OUTPUT_DIR = Path("./gemini_embedding_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def clean_list_field(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def pseudo_track_to_text(track):
    """
    Convert one Gemini pseudo-track into the same metadata-style text
    used by the BERT retriever.
    """
    track_name = ", ".join(clean_list_field(track.get("track_name", [])))
    artist_name = ", ".join(clean_list_field(track.get("artist_name", [])))
    album_name = ", ".join(clean_list_field(track.get("album_name", [])))
    tag_list = ", ".join(clean_list_field(track.get("tag_list", [])))
    release_date = str(track.get("release_date", ""))

    return (
        f"track_name: {track_name}\n"
        f"artist_name: {artist_name}\n"
        f"album_name: {album_name}\n"
        f"tag_list: {tag_list}\n"
        f"release_date: {release_date}"
    )


def catalog_track_to_text(metadata):
    parts = []

    for field in CORPUS_TYPES:
        value = metadata.get(field, "")
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        else:
            value = str(value)

        parts.append(f"{field}: {value}")

    return "\n".join(parts)


def load_gemini_cache_file(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both formats:
    # [ {...}, {...} ]
    # {"recommendations": [ {...}, {...} ]}
    if isinstance(data, dict):
        for key in ["recommendations", "songs", "tracks", "reference_songs", "reference_tracks", "items"]:
            if key in data:
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"Invalid Gemini cache format in {cache_path}")

    cleaned = []

    for item in data:
        if not isinstance(item, dict):
            continue

        cleaned.append({
            "track_name": clean_list_field(item.get("track_name", ["Unknown Track"]))[:1],
            "artist_name": clean_list_field(item.get("artist_name", ["Unknown Artist"]))[:1],
            "album_name": clean_list_field(item.get("album_name", ["Unknown Album"]))[:1],
            "tag_list": clean_list_field(item.get("tag_list", [])),
            "release_date": str(item.get("release_date", "")),
        })

    return cleaned


def parse_session_turn_from_filename(path):
    """
    Expected cache filename:
    <session_id>_turn_<turn_number>.json
    """
    name = path.stem

    if "_turn_" not in name:
        return name, None

    session_id, turn_number = name.rsplit("_turn_", 1)

    try:
        turn_number = int(turn_number)
    except ValueError:
        turn_number = None

    return session_id, turn_number


def mean_pool(last_hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    summed = torch.sum(last_hidden_states * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(texts, tokenizer, model, batch_size=16, max_length=128):
    embeddings = []

    model.eval()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)

            embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0).numpy()


def load_catalog_embeddings():
    embeddings_path = BERT_CACHE_DIR / "embeddings.pt"
    track_ids_path = BERT_CACHE_DIR / "track_ids.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing catalog embeddings: {embeddings_path}\n"
            "Run BERT retrieval once first to build the cache."
        )

    if not track_ids_path.exists():
        raise FileNotFoundError(
            f"Missing catalog track IDs: {track_ids_path}\n"
            "Run BERT retrieval once first to build the cache."
        )

    embeddings = torch.load(embeddings_path, map_location="cpu").numpy()

    with open(track_ids_path, "r", encoding="utf-8") as f:
        track_ids = json.load(f)

    return embeddings, track_ids


def load_catalog_metadata():
    dataset = load_dataset(DATASET_NAME)

    all_splits = []
    for split in SPLIT_TYPES:
        all_splits.append(dataset[split])

    full_dataset = concatenate_datasets(all_splits)

    metadata_dict = {}

    for item in full_dataset:
        metadata_dict[item["track_id"]] = item

    return metadata_dict


def metadata_field_to_str(value):
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    print(f"Using device: {DEVICE}")

    print("Loading catalog BERT embeddings...")
    catalog_embeddings, catalog_track_ids = load_catalog_embeddings()

    print("Loading catalog metadata...")
    catalog_metadata = load_catalog_metadata()

    print("Loading BERT model for Gemini pseudo-track embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    cache_files = sorted(GEMINI_CACHE_DIR.glob("*.json"))

    if not cache_files:
        raise FileNotFoundError(f"No Gemini cache files found in {GEMINI_CACHE_DIR}")

    all_rows = []

    print(f"Found {len(cache_files)} Gemini cache files.")

    for cache_file in cache_files:
        print(f"Processing {cache_file.name}")

        session_id, turn_number = parse_session_turn_from_filename(cache_file)

        try:
            pseudo_tracks = load_gemini_cache_file(cache_file)
        except Exception as e:
            print(f"Skipping invalid cache file {cache_file}: {e}")
            continue

        if len(pseudo_tracks) == 0:
            continue

        pseudo_texts = [pseudo_track_to_text(track) for track in pseudo_tracks]
        pseudo_embeddings = embed_texts(pseudo_texts, tokenizer, model)

        # Similarity: [num_pseudo_tracks, num_catalog_tracks]
        sims = np.matmul(pseudo_embeddings, catalog_embeddings.T)

        for pseudo_idx, track in enumerate(pseudo_tracks):
            top_indices = np.argsort(-sims[pseudo_idx])[:TOPK_PER_GEMINI_TRACK]

            pseudo_track_name = ", ".join(track["track_name"])
            pseudo_artist_name = ", ".join(track["artist_name"])
            pseudo_album_name = ", ".join(track["album_name"])
            pseudo_tags = ", ".join(track["tag_list"])
            pseudo_release_date = track["release_date"]

            for rank, catalog_idx in enumerate(top_indices, start=1):
                catalog_track_id = catalog_track_ids[catalog_idx]
                meta = catalog_metadata[catalog_track_id]

                all_rows.append({
                    "cache_file": cache_file.name,
                    "session_id": session_id,
                    "turn_number": turn_number,

                    "gemini_pseudo_rank": pseudo_idx + 1,
                    "gemini_track_name": pseudo_track_name,
                    "gemini_artist_name": pseudo_artist_name,
                    "gemini_album_name": pseudo_album_name,
                    "gemini_tag_list": pseudo_tags,
                    "gemini_release_date": pseudo_release_date,

                    "catalog_similarity_rank": rank,
                    "cosine_similarity": float(sims[pseudo_idx, catalog_idx]),

                    "catalog_track_id": catalog_track_id,
                    "catalog_track_name": metadata_field_to_str(meta.get("track_name", "")),
                    "catalog_artist_name": metadata_field_to_str(meta.get("artist_name", "")),
                    "catalog_album_name": metadata_field_to_str(meta.get("album_name", "")),
                    "catalog_tag_list": metadata_field_to_str(meta.get("tag_list", "")),
                    "catalog_release_date": metadata_field_to_str(meta.get("release_date", "")),
                })

        # Also save a heatmap between the 5 Gemini pseudo-tracks
        if len(pseudo_tracks) > 1:
            pseudo_sim_matrix = cosine_similarity(pseudo_embeddings)

            labels = [
                f"{i+1}. {', '.join(track['track_name'])}"
                for i, track in enumerate(pseudo_tracks)
            ]

            plt.figure(figsize=(8, 6))
            plt.imshow(pseudo_sim_matrix)
            plt.colorbar(label="Cosine similarity")
            plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
            plt.yticks(range(len(labels)), labels, fontsize=8)
            plt.title(f"Gemini pseudo-track similarity\n{session_id} turn {turn_number}")
            plt.tight_layout()

            heatmap_path = OUTPUT_DIR / f"{session_id}_turn_{turn_number}_gemini_similarity_heatmap.png"
            plt.savefig(heatmap_path, dpi=200)
            plt.close()

    df = pd.DataFrame(all_rows)

    output_csv = OUTPUT_DIR / "gemini_to_catalog_similarity_table.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8")

    output_excel = OUTPUT_DIR / "gemini_to_catalog_similarity_table.xlsx"
    df.to_excel(output_excel, index=False)

    print("\nDone.")
    print(f"Saved CSV table: {output_csv}")
    print(f"Saved Excel table: {output_excel}")
    print(f"Saved heatmaps in: {OUTPUT_DIR}")

    print("\nPreview:")
    print(df.head(20))


if __name__ == "__main__":
    main()