"""
Refresh the devset artifacts used by the Streamlit dashboard.

Run this after a completed first-rows devset evaluation so the conversation
details and BM25 explanation page point to the same model run.
"""

import argparse
from pathlib import Path

from export_bm25_explanation_to_csv import export_bm25_explanations
from export_devset_conversation_details import export_details


PROJECT_ROOT = Path(__file__).resolve().parent
STREAMLIT_DIR = PROJECT_ROOT / "visualize_streamlit"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh Streamlit dashboard artifacts from one devset evaluation run."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory containing predictions.json and ground_truth.json.",
    )
    parser.add_argument(
        "--gemini_cache_dir",
        type=Path,
        default=PROJECT_ROOT / "cache" / "gemini_expansions_devset_first100",
        help="Gemini cache directory kept for backwards-compatible command lines.",
    )
    parser.add_argument(
        "--bm25_keyword_cache_dir",
        type=Path,
        default=None,
        help=(
            "Gemini controlled-keyword cache used for the BM25 explanation page. "
            "Defaults to --gemini_cache_dir for backwards compatibility."
        ),
    )
    parser.add_argument(
        "--skip_embedding_projection",
        action="store_true",
        help="Deprecated no-op kept for old commands.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Deprecated no-op kept for old commands.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    gemini_cache_dir = args.gemini_cache_dir
    if not gemini_cache_dir.is_absolute():
        gemini_cache_dir = PROJECT_ROOT / gemini_cache_dir
    bm25_keyword_cache_dir = args.bm25_keyword_cache_dir or args.gemini_cache_dir
    if not bm25_keyword_cache_dir.is_absolute():
        bm25_keyword_cache_dir = PROJECT_ROOT / bm25_keyword_cache_dir

    conversation_output = STREAMLIT_DIR / "devset_conversation_details.json"
    bm25_explanation_output = STREAMLIT_DIR / "bm25_explanation_table.csv"

    print("Refreshing Streamlit artifacts from:")
    print(f"  run_dir: {run_dir}")
    print(f"  gemini_cache_dir: {gemini_cache_dir}")
    print(f"  bm25_keyword_cache_dir: {bm25_keyword_cache_dir}")

    export_details(run_dir=run_dir, output_path=conversation_output)
    export_bm25_explanations(
        run_dir=run_dir,
        gemini_cache_dir=bm25_keyword_cache_dir,
        output_path=bm25_explanation_output,
    )
    print("Streamlit artifacts refreshed.")


if __name__ == "__main__":
    main()
