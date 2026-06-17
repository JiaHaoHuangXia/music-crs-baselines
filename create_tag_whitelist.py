"""
Create a controlled tag whitelist from the TalkPlay track metadata dataset.

The whitelist can be built in two ways:
- popular: keep the most frequent clean catalog tags.
- music_vocabulary: map clean catalog tags to a compact music-focused vocabulary.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import concatenate_datasets, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
DEFAULT_SPLITS = ["all_tracks"]

NOISE_EXACT = {
    "",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "10 of 10 stars",
    "seen live",
    "favorite",
    "favorites",
    "favourite",
    "favourites",
    "fav",
    "favs",
    "loved",
    "love",
    "good",
    "great",
    "awesome",
    "beautiful",
    "nice",
    "cool",
    "amazing",
    "my music",
    "playlist",
    "spotify",
    "lastfm",
    "last fm",
    "radio",
    "heard live",
    "heard on pandora",
    "male vocalists",
    "female vocalists",
    "american",
    "british",
    "usa",
    "uk",
}

NOISE_PATTERNS = [
    re.compile(r"^\d+\s+of\s+10\s+stars?$"),
    re.compile(r"^\d+\s*stars?$"),
    re.compile(r"^rated\d+$"),
    re.compile(r"^my\s+.+"),
    re.compile(r".*\bfavorite\b.*"),
    re.compile(r".*\bfavourite\b.*"),
    re.compile(r".*\bloved tracks?\b.*"),
    re.compile(r".*\bplaylist\b.*"),
    re.compile(r".*\bfav(e|s)?\b.*"),
    re.compile(r".*\bfavorites?\b.*"),
    re.compile(r".*\bfavourites?\b.*"),
    re.compile(r".*\bmy\b.*"),
    re.compile(r".*\btop\b.*\bsongs?\b.*"),
    re.compile(r".*\btag\b.*"),
]

MUSIC_VOCABULARY = {
    "acoustic",
    "afrobeat",
    "alternative",
    "alternative rock",
    "ambient",
    "angry",
    "blues",
    "britpop",
    "chill",
    "chillout",
    "classic rock",
    "classical",
    "club",
    "country",
    "dance",
    "dance-pop",
    "dark",
    "death metal",
    "disco",
    "downtempo",
    "dream pop",
    "drum and bass",
    "dubstep",
    "electro",
    "electronic",
    "electronica",
    "emo",
    "energetic",
    "experimental",
    "folk",
    "funk",
    "funk metal",
    "garage rock",
    "gothic",
    "grunge",
    "happy",
    "hard rock",
    "hardcore",
    "heavy metal",
    "hip hop",
    "hip-hop",
    "house",
    "indie",
    "indie pop",
    "indie rock",
    "industrial",
    "instrumental",
    "jazz",
    "latin",
    "lo-fi",
    "melancholic",
    "metal",
    "new wave",
    "nostalgic",
    "nu metal",
    "pop",
    "pop punk",
    "post-hardcore",
    "post-punk",
    "progressive rock",
    "psychedelic",
    "punk",
    "punk rock",
    "r&b",
    "rap",
    "rap metal",
    "rap rock",
    "reggae",
    "relaxing",
    "rnb",
    "rock",
    "sad",
    "screamo",
    "shoegaze",
    "singer-songwriter",
    "soul",
    "soundtrack",
    "synthpop",
    "techno",
    "thrash metal",
    "trance",
    "trip hop",
}


def vocabulary_matcher(vocabulary: set[str]):
    terms = sorted(vocabulary)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(terms)

    def match(tag: str) -> tuple[str, float]:
        query = vectorizer.transform([tag])
        similarities = cosine_similarity(query, matrix)[0]
        best_index = int(similarities.argmax())
        return terms[best_index], float(similarities[best_index])

    return match


def normalize_tag(tag: Any) -> str:
    tag = str(tag or "").strip().lower()
    tag = tag.replace("_", " ").replace("/", " / ")
    tag = re.sub(r"\s+", " ", tag)
    return tag.strip(" -_.,;:!?")


def is_noise(tag: str) -> bool:
    if tag in NOISE_EXACT:
        return True
    if len(tag) < 3:
        return True
    if any(pattern.match(tag) for pattern in NOISE_PATTERNS):
        return True
    if re.fullmatch(r"\d{1,2}s?", tag):
        return True
    return False


def has_bad_shape(tag: str, max_words: int) -> bool:
    words = tag.split()
    if len(words) > max_words:
        return True
    if len(tag) > 32:
        return True
    if re.search(r"(.)\1{3,}", tag):
        return True
    if len(words) == 1 and len(tag) > 14 and "-" not in tag:
        return True
    return False


def is_exact_or_contained_music_term(tag: str) -> str | None:
    if tag in MUSIC_VOCABULARY:
        return tag
    contained = [term for term in MUSIC_VOCABULARY if term in tag]
    if not contained:
        return None
    return max(contained, key=len)


def iter_tags(value: Any):
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            yield item
        return
    yield value


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a music-focused tag whitelist from track metadata.")
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument(
        "--selection_method",
        choices=["popular", "music_vocabulary"],
        default="popular",
        help=(
            "'popular' keeps the most frequent clean catalog tags. "
            "'music_vocabulary' maps tags to a compact predefined music vocabulary."
        ),
    )
    parser.add_argument("--min_frequency", type=int, default=25)
    parser.add_argument("--max_tags", type=int, default=400)
    parser.add_argument(
        "--max_words",
        type=int,
        default=4,
        help="Remove candidate tags longer than this many words before vocabulary matching.",
    )
    parser.add_argument(
        "--min_vocab_similarity",
        type=float,
        default=0.58,
        help="Minimum character-ngram similarity to the controlled music vocabulary.",
    )
    parser.add_argument("--output_json", default="config/tag_whitelist.json")
    parser.add_argument("--output_csv", default="config/tag_whitelist.csv")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)
    full_dataset = concatenate_datasets([dataset[split] for split in args.splits])

    counts: Counter[str] = Counter()
    for row in full_dataset:
        for raw_tag in iter_tags(row.get("tag_list")):
            tag = normalize_tag(raw_tag)
            if tag:
                counts[tag] += 1

    canonical_counts: Counter[str] = Counter()
    canonical_sources: dict[str, Counter[str]] = defaultdict(Counter)
    rejected_counts = Counter()
    match_vocabulary = None
    if args.selection_method == "music_vocabulary":
        match_vocabulary = vocabulary_matcher(MUSIC_VOCABULARY)

    for tag, frequency in counts.most_common():
        if frequency < args.min_frequency:
            rejected_counts["below_min_frequency"] += 1
            continue
        if is_noise(tag):
            rejected_counts["noise"] += 1
            continue
        if has_bad_shape(tag, args.max_words):
            rejected_counts["bad_shape"] += 1
            continue

        if args.selection_method == "popular":
            canonical_tag = tag
        else:
            exact_term = is_exact_or_contained_music_term(tag)
            if exact_term is not None:
                canonical_tag = exact_term
                similarity = 1.0 if tag == exact_term else 0.95
            else:
                canonical_tag, similarity = match_vocabulary(tag)
                if similarity < args.min_vocab_similarity:
                    rejected_counts["low_vocab_similarity"] += 1
                    continue

        canonical_counts[canonical_tag] += frequency
        canonical_sources[canonical_tag][tag] += frequency

    candidates = []
    for canonical_tag, frequency in canonical_counts.most_common(args.max_tags):
        top_sources = [
            source_tag
            for source_tag, _ in canonical_sources[canonical_tag].most_common(5)
        ]
        candidates.append(
            {
                "tag": canonical_tag,
                "frequency": frequency,
                "source_examples": "; ".join(top_sources),
            }
        )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_name": args.dataset_name,
                "splits": args.splits,
                "selection_method": args.selection_method,
                "min_frequency": args.min_frequency,
                "max_tags": args.max_tags,
                "max_words": args.max_words,
                "min_vocab_similarity": args.min_vocab_similarity,
                "method": (
                    "frequency + noise/tag-shape rules"
                    if args.selection_method == "popular"
                    else "frequency + tag-shape rules + character-ngram similarity to controlled music vocabulary"
                ),
                "rejected_counts": dict(rejected_counts),
                "tags": candidates,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["tag", "frequency", "source_examples"])
        writer.writeheader()
        writer.writerows(candidates)

    print(f"Loaded {len(full_dataset)} tracks.")
    print(f"Found {len(counts)} unique normalized tags.")
    print(f"Rejected candidates: {dict(rejected_counts)}")
    print(f"Saved {len(candidates)} whitelist tags to {output_json} and {output_csv}.")


if __name__ == "__main__":
    main()
