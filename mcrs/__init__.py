from .crs_baseline import CRS_BASELINE
import torch

def load_crs_baseline(
    lm_type="meta-llama/Llama-3.2-1B-Instruct",
    retrieval_type="bm25",
    item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
    track_split_types: list[str] = ["all_tracks"],
    user_split_types: list[str] = ["all_users"],
    corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
    cache_dir="./cache",
    device="cuda",
    attn_implementation="eager",
    dtype=torch.bfloat16,
    use_gemini_expansion=False,
    gemini_model_name="gemini-3.1-flash-lite",
    gemini_cache_dir="./cache/gemini_expansions",
):
    return CRS_BASELINE(lm_type, retrieval_type, item_db_name, user_db_name, track_split_types, user_split_types, corpus_types, cache_dir, device, attn_implementation, dtype, use_gemini_expansion, gemini_model_name, gemini_cache_dir)
