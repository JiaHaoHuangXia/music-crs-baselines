import datasets
print("Imported datasets first", flush=True)

import torch
print("Imported torch", flush=True)

from mcrs.retrieval_modules.bert import BERT_MODEL
print("Imported BERT_MODEL", flush=True)

from mcrs.db_item import MusicCatalogDB
print("Imported MusicCatalogDB", flush=True)

pseudo_tracks = [
    {
        "track_name": ["Understanding in a Car Crash"],
        "artist_name": ["Thursday"],
        "album_name": ["Full Collapse"],
        "tag_list": ["post-hardcore", "emo", "intense", "dramatic", "screamo vocals", "raw energy", "emotional", "dynamic"],
        "release_date": "2001-04-10"
    },
    {
        "track_name": ["Baby, You Wouldn't Last a Minute on the Creek"],
        "artist_name": ["Chiodos"],
        "album_name": ["All's Well That Ends Well"],
        "tag_list": ["post-hardcore", "screamo", "theatrical", "high energy", "dramatic", "clean and harsh vocals", "melodic chaos"],
        "release_date": "2005-07-26"
    },
    {
        "track_name": ["The Artist in the Ambulance"],
        "artist_name": ["Thrice"],
        "album_name": ["The Artist in the Ambulance"],
        "tag_list": ["post-hardcore", "alternative rock", "intense", "melodic", "driving rhythm", "emotional", "anthemic"],
        "release_date": "2003-07-22"
    },
    {
        "track_name": ["Smile in Your Sleep"],
        "artist_name": ["Silverstein"],
        "album_name": ["Discovering the Waterfront"],
        "tag_list": ["post-hardcore", "emo", "melodic", "emotional intensity", "screamed vocals", "catchy hooks", "dramatic"],
        "release_date": "2005-08-16"
    },
    {
        "track_name": ["Reinventing Your Exit"],
        "artist_name": ["Underoath"],
        "album_name": ["They're Only Chasing Safety"],
        "tag_list": ["post-hardcore", "metalcore influence", "intense", "atmospheric", "screamo", "dynamic contrast", "emotional"],
        "release_date": "2004-06-15"
    }
]


def pseudo_tracks_to_query(pseudo_tracks):
    parts = []

    for track in pseudo_tracks:
        track_name = ", ".join(track.get("track_name", []))
        artist_name = ", ".join(track.get("artist_name", []))
        album_name = ", ".join(track.get("album_name", []))
        tag_list = ", ".join(track.get("tag_list", []))
        release_date = track.get("release_date", "")

        parts.append(
            f"track_name: {track_name}\n"
            f"artist_name: {artist_name}\n"
            f"album_name: {album_name}\n"
            f"tag_list: {tag_list}\n"
            f"release_date: {release_date}"
        )

    return "\n\n".join(parts)


def main():
    dataset_name = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
    split_types = ["all_tracks"]

    # Important: include tag_list because your pseudo-tracks rely heavily on tags
    corpus_types = [
        "track_name",
        "artist_name",
        "album_name",
        "tag_list",
        "release_date"
    ]

    print("Building expanded query...")
    expanded_query = pseudo_tracks_to_query(pseudo_tracks)

    print("\nExpanded query:")
    print("=" * 80)
    print(expanded_query)
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("\nLoading BERT retriever...")
    retriever = BERT_MODEL(
        dataset_name=dataset_name,
        split_types=split_types,
        corpus_types=corpus_types,
        cache_dir="./cache",
        device=device,
        batch_size=32,
        max_length=128
    )

    print("\nRetrieving top 20 tracks...")
    track_ids = retriever.text_to_item_retrieval(expanded_query, topk=20)

    print("\nTop 20 retrieved track IDs:")
    for i, track_id in enumerate(track_ids, start=1):
        print(f"{i}. {track_id}")

    print("\nLoading catalog to inspect metadata...")
    item_db = MusicCatalogDB(
        dataset_name=dataset_name,
        split_types=split_types,
        corpus_types=corpus_types
    )

    print("\nTop 20 retrieved tracks with metadata:")
    print("=" * 80)
    for i, track_id in enumerate(track_ids, start=1):
        print(f"{i}. {item_db.id_to_metadata(track_id)}")
        print("-" * 80)


if __name__ == "__main__":
    main()