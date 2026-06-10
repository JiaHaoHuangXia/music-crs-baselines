"""
Refresh all devset artifacts used by the Streamlit dashboard.

Run this after a completed first-rows devset evaluation so the case-study page,
Gemini comparison table, and embedding map all point to the same model run.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from export_devset_conversation_details import export_details
from export_devset_gemini_cache_to_csv import export_cache


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
        help="Gemini cache used by the same run.",
    )
    parser.add_argument(
        "--corpus_types",
        nargs="+",
        default=["track_name", "artist_name", "album_name", "tag_list", "release_date"],
        help="Corpus fields matching the embedding cache used for the projection.",
    )
    parser.add_argument(
        "--projection_retrieval_type",
        choices=["bert", "sentence_transformer"],
        default="bert",
        help="Embedding retriever used for gemini_embedding_projection.csv.",
    )
    parser.add_argument(
        "--skip_embedding_projection",
        action="store_true",
        help="Skip rebuilding gemini_embedding_projection.csv.",
    )
    parser.add_argument(
        "--topk_retrieved_per_reference",
        type=int,
        default=10,
        help="Nearest catalog tracks to highlight for each Gemini reference in the embedding map.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Optional device override for Gemini reference embedding creation.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    gemini_cache_dir = args.gemini_cache_dir
    if not gemini_cache_dir.is_absolute():
        gemini_cache_dir = PROJECT_ROOT / gemini_cache_dir

    conversation_output = STREAMLIT_DIR / "devset_conversation_details.json"
    gemini_ground_truth_output = STREAMLIT_DIR / "devset_gemini_ground_truth_table.csv"
    embedding_projection_output = STREAMLIT_DIR / "gemini_embedding_projection.csv"

    print("Refreshing Streamlit artifacts from:")
    print(f"  run_dir: {run_dir}")
    print(f"  gemini_cache_dir: {gemini_cache_dir}")

    export_details(run_dir=run_dir, output_path=conversation_output)
    export_cache(
        cache_dir=gemini_cache_dir,
        ground_truth_path=run_dir / "ground_truth.json",
        output_path=gemini_ground_truth_output,
    )

    if args.skip_embedding_projection:
        print("Skipped embedding projection refresh.")
        return

    projection_command = [
        sys.executable,
        str(PROJECT_ROOT / "create_gemini_embedding_projection.py"),
        "--gemini_cache_dir",
        str(gemini_cache_dir),
        "--devset_table",
        str(gemini_ground_truth_output),
        "--output",
        str(embedding_projection_output),
        "--projection_retrieval_type",
        args.projection_retrieval_type,
        "--topk_retrieved_per_reference",
        str(args.topk_retrieved_per_reference),
        "--corpus_types",
        *args.corpus_types,
    ]
    if args.device:
        projection_command.extend(["--device", args.device])

    print("Refreshing embedding projection...")
    subprocess.run(projection_command, cwd=PROJECT_ROOT, check=True)
    print("Streamlit artifacts refreshed.")


if __name__ == "__main__":
    main()
