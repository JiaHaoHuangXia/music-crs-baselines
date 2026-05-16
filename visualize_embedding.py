import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# Optional but recommended:
# pip install umap-learn
import umap

from mcrs.db_item import MusicCatalogDB
from mcrs.retrieval_modules.bert import BERT_MODEL

# =========================
# CONFIG
# =========================

CACHE_DIR = "./cache"

DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
SPLIT_TYPES = ["all_tracks"]

CORPUS_TYPES = [
    "track_name",
    "artist_name",
    "album_name",
    "tag_list",
    "release_date",
]

OUTPUT_DIR = "./embedding_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_POINTS = 12000  # reduce if plotting is slow; set None to use all tracks
RANDOM_SEED = 42


# =========================
# HELPERS
# =========================

def get_bert_index_dir(cache_dir, corpus_types):
    corpus_name = "_".join(corpus_types)
    return os.path.join(cache_dir, "bert", corpus_name)


def load_embeddings_and_ids(index_dir):
    embeddings_path = os.path.join(index_dir, "embeddings.pt")
    track_ids_path = os.path.join(index_dir, "track_ids.json")

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")

    if not os.path.exists(track_ids_path):
        raise FileNotFoundError(f"Missing track IDs file: {track_ids_path}")

    embeddings = torch.load(embeddings_path, map_location="cpu")
    track_ids = json.load(open(track_ids_path, "r", encoding="utf-8"))

    embeddings = embeddings.numpy()

    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded track IDs: {len(track_ids)}")

    return embeddings, track_ids


def metadata_to_dataframe(track_ids, item_db):
    rows = []

    for track_id in track_ids:
        metadata = item_db.metadata_dict[track_id]

        def clean_field(value):
            if isinstance(value, list):
                return ", ".join(str(v) for v in value)
            return str(value)

        rows.append({
            "track_id": track_id,
            "track_name": clean_field(metadata.get("track_name", "")),
            "artist_name": clean_field(metadata.get("artist_name", "")),
            "album_name": clean_field(metadata.get("album_name", "")),
            "tag_list": clean_field(metadata.get("tag_list", "")),
            "release_date": clean_field(metadata.get("release_date", "")),
        })

    return pd.DataFrame(rows)


def assign_broad_genre(tag_text):
    tag_text = str(tag_text).lower()

    if any(x in tag_text for x in ["metal", "hardcore", "screamo", "post-hardcore", "emocore"]):
        return "metal/hardcore"
    if any(x in tag_text for x in ["hip hop", "rap", "trap"]):
        return "hip hop/rap"
    if any(x in tag_text for x in ["electronic", "electronica", "edm", "house", "techno", "dance"]):
        return "electronic/dance"
    if any(x in tag_text for x in ["rock", "punk", "alternative", "indie"]):
        return "rock/indie"
    if any(x in tag_text for x in ["pop", "dance-pop", "synthpop"]):
        return "pop"
    if any(x in tag_text for x in ["jazz", "blues", "soul", "funk"]):
        return "jazz/soul/funk"
    if any(x in tag_text for x in ["classical", "instrumental", "ambient", "soundtrack"]):
        return "ambient/classical"
    return "other"


def sample_points(embeddings, track_ids, max_points=None, seed=42):
    if max_points is None or len(track_ids) <= max_points:
        return embeddings, track_ids

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(track_ids), size=max_points, replace=False)
    indices = np.sort(indices)

    sampled_embeddings = embeddings[indices]
    sampled_track_ids = [track_ids[i] for i in indices]

    print(f"Sampled {max_points} tracks from {len(track_ids)} total tracks.")

    return sampled_embeddings, sampled_track_ids


def plot_embedding(df, x_col, y_col, color_col, title, output_path):
    categories = sorted(df[color_col].unique())

    plt.figure(figsize=(12, 9))

    for category in categories:
        subset = df[df[color_col] == category]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            s=5,
            alpha=0.65,
            label=category
        )

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved plot: {output_path}")


# =========================
# MAIN
# =========================

def main():
    index_dir = get_bert_index_dir(CACHE_DIR, CORPUS_TYPES)
    print(f"Using BERT index directory: {index_dir}")

    embeddings_path = os.path.join(index_dir, "embeddings.pt")
    track_ids_path = os.path.join(index_dir, "track_ids.json")

    if not os.path.exists(embeddings_path) or not os.path.exists(track_ids_path):
        print("BERT embedding cache not found. Building BERT index first...")

        _ = BERT_MODEL(
            dataset_name=DATASET_NAME,
            split_types=SPLIT_TYPES,
            corpus_types=CORPUS_TYPES,
            cache_dir=CACHE_DIR,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=32,
            max_length=128,
        )

        print("BERT index built.")

    embeddings, track_ids = load_embeddings_and_ids(index_dir)

    embeddings, track_ids = sample_points(
        embeddings,
        track_ids,
        max_points=MAX_POINTS,
        seed=RANDOM_SEED
    )

    print("Normalizing embeddings...")
    embeddings = normalize(embeddings)

    print("Loading track metadata...")
    item_db = MusicCatalogDB(
        dataset_name=DATASET_NAME,
        split_types=SPLIT_TYPES,
        corpus_types=CORPUS_TYPES,
    )

    df = metadata_to_dataframe(track_ids, item_db)
    df["broad_genre"] = df["tag_list"].apply(assign_broad_genre)

    # =========================
    # PCA 2D
    # =========================

    print("Running PCA 2D...")
    pca_2d = PCA(n_components=2, random_state=RANDOM_SEED)
    pca_coords = pca_2d.fit_transform(embeddings)

    df["pca_x"] = pca_coords[:, 0]
    df["pca_y"] = pca_coords[:, 1]

    print("PCA explained variance ratio:", pca_2d.explained_variance_ratio_)

    plot_embedding(
        df=df,
        x_col="pca_x",
        y_col="pca_y",
        color_col="broad_genre",
        title="PCA 2D of BERT Track Metadata Embeddings",
        output_path=os.path.join(OUTPUT_DIR, "pca_2d_track_embeddings.png")
    )

    # =========================
    # PCA 50D → UMAP 2D
    # =========================

    print("Running PCA 50D before UMAP...")
    pca_50d = PCA(n_components=50, random_state=RANDOM_SEED)
    embeddings_50d = pca_50d.fit_transform(embeddings)

    print("Running UMAP 2D...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.05,
        metric="cosine",
        random_state=RANDOM_SEED,
        low_memory=True,
    )

    umap_coords = reducer.fit_transform(embeddings_50d)

    df["umap_x"] = umap_coords[:, 0]
    df["umap_y"] = umap_coords[:, 1]

    plot_embedding(
        df=df,
        x_col="umap_x",
        y_col="umap_y",
        color_col="broad_genre",
        title="UMAP 2D of BERT Track Metadata Embeddings",
        output_path=os.path.join(OUTPUT_DIR, "umap_2d_track_embeddings.png")
    )

    # Save coordinates for later inspection
    csv_path = os.path.join(OUTPUT_DIR, "track_embedding_coordinates.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved coordinates CSV: {csv_path}")

    print("\nDone.")
    print("Open these files:")
    print(f"- {os.path.join(OUTPUT_DIR, 'pca_2d_track_embeddings.png')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'umap_2d_track_embeddings.png')}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()