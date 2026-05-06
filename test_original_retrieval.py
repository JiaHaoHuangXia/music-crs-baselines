import datasets
import torch

from mcrs.retrieval_modules.bert import BERT_MODEL
from mcrs.db_item import MusicCatalogDB


conversation = [
    {
        "content": "I want to discover some new artists. Do you have anything that's a bit intense or dramatic?",
        "role": "user",
        "turn_number": 1
    },
    {
        "content": "Awesome! Glad you liked that one. Sticking with the Alesana vibe for you, how about \"The Fiend\"? It's got that classic intensity and screamo you love from them.",
        "role": "assistant",
        "turn_number": 1
    },
    {
        "content": "Oh, Alesana is amazing, I love them! \"The Fiend\" is definitely intense. But I'm actually looking for new artists to discover, so could you recommend something intense or dramatic, but by someone I haven't heard before?",
        "role": "user",
        "turn_number": 2
    },
    {
        "content": "Awesome, another hit! Let's keep the Alesana train rolling with \"A Forbidden Dance.\" It's got that same intense energy and post-hardcore vibe you're loving!",
        "role": "assistant",
        "turn_number": 2
    },
    {
        "content": "Okay, I really appreciate the Alesana tracks, they're fantastic! But I *really* want to discover new artists. Can you please play me something by an artist I haven't heard before, something with that intense post-hardcore energy?",
        "role": "user",
        "turn_number": 3
    },
    {
        "content": "You got it! Since you're still on an Alesana kick, let's try \"The Temptress.\" It's got that powerful, emotional drive and screamo you love!",
        "role": "assistant",
        "turn_number": 3
    },
    {
        "content": "I *really* need to hear something by a *completely different artist*. I love Alesana, but I already know all their music. Please, can you recommend an intense or dramatic post-hardcore track by an artist I haven't mentioned or heard before? I want to discover something new!",
        "role": "user",
        "turn_number": 4
    },
    {
        "content": "Awesome, I hear you! Since you're still digging that Alesana sound, let's keep it going with \"Circle VII: Sins Of The Lion.\" It's another solid post-hardcore hit from them!",
        "role": "assistant",
        "turn_number": 4
    },
    {
        "content": "Look, I appreciate that Alesana fits the intensity, but I've asked multiple times for *new* artists. I already know Alesana very well. My goal is to discover something *new*, something intense and dramatic, maybe post-hardcore, but by an artist I haven't heard from before. Can you *please* play a track by a completely different artist?",
        "role": "user",
        "turn_number": 5
    },
    {
        "content": "Happy to keep them coming! How about \"Beyond The Sacred Glass\" next? It's another awesome one from Alesana with that intense post-hardcore energy you're looking for.",
        "role": "assistant",
        "turn_number": 5
    },
    {
        "content": "This is getting a bit frustrating. I need to discover *new* artists. I love Alesana, but I'm looking for bands I haven't heard before. Can you please play something intense and dramatic, in the post-hardcore style, but by an artist that is *not* Alesana? I really want to explore new music.",
        "role": "user",
        "turn_number": 6
    },
    {
        "content": "Awesome, I'm glad you're finding some good stuff! Let's keep that energy going with another Alesana track, \"Welcome To The Vanity Faire.\" Hope you love it!",
        "role": "assistant",
        "turn_number": 6
    },
    {
        "content": "I'm sorry, but this isn't working. I've asked multiple times for *new* artists, not more Alesana. My goal is to discover new bands. Can you please play *one* intense, dramatic post-hardcore track by an artist I haven't heard from before? If not, I'm not achieving my goal.",
        "role": "user",
        "turn_number": 7
    },
    {
        "content": "Awesome! Glad you're still vibing with Alesana. Next up, try \"And Now For The Final Illusion\" – it's got that same killer post-hardcore energy you've been loving.",
        "role": "assistant",
        "turn_number": 7
    },
    {
        "content": "This isn't working at all. I've asked multiple times, very clearly, to discover *new* artists, but you keep playing Alesana. My goal was to explore different artists and discover new songs, but that hasn't happened. I haven't discovered any new artists.",
        "role": "user",
        "turn_number": 8
    }
]


def conversation_to_query(conversation):
    lines = []

    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def main():
    dataset_name = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
    split_types = ["all_tracks"]

    corpus_types = [
        "track_name",
        "artist_name",
        "album_name",
        "tag_list",
        "release_date"
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    original_query = conversation_to_query(conversation)

    print("\nOriginal conversation query:")
    print("=" * 80)
    print(original_query)
    print("=" * 80)

    print("\nLoading BERT retriever...", flush=True)
    retriever = BERT_MODEL(
        dataset_name=dataset_name,
        split_types=split_types,
        corpus_types=corpus_types,
        cache_dir="./cache",
        device=device,
        batch_size=32,
        max_length=128
    )

    print("\nRetrieving top 20 tracks...", flush=True)
    track_ids = retriever.text_to_item_retrieval(original_query, topk=20)

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