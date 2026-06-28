from pathlib import Path
import json
import html
import math

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DEVSET_CONVERSATION_PATH = ROOT / "visualize_streamlit" / "devset_conversation_details.json"
BM25_EXPLANATION_PATH = ROOT / "visualize_streamlit" / "bm25_explanation_table.csv"
MINILM_EXPLANATION_PATH = ROOT / "visualize_streamlit" / "minilm_reference_retrieval_table.csv"
MINILM_SCORES_PATH = ROOT / "exp" / "first_100" / "devset_gemini_multiquery_minilm_streamlit_top20" / "scores.json"

REFERENCE_LINKS = {
    "RecSys Challenge 2026": "https://www.recsyschallenge.com/2026/",
    "Music-CRS Challenge": "https://nlp4musa.github.io/music-crs-challenge/",
    "talkpl-ai Hugging Face": "https://huggingface.co/talkpl-ai",
    "Baseline repository": "https://github.com/JiaHaoHuangXia/music-crs-baselines",
}

DEVSET_SUBSET_RESULTS = [
    {
        "experiment": "Clean baseline BERT + raw conversation",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0025,
        "ndcg@10": 0.011400438250866413,
        "ndcg@20": 0.013322326781414569,
        "catalog_diversity": 0.014467506532684667,
        "lexical_diversity": 0.4384776745579862,
        "total_catalog_size": 47071,
        "source": "Historical local evaluator",
        "note": "Original dense baseline using raw conversation text. It shows why exact-track retrieval is difficult with dense embeddings alone.",
    },
    {
        "experiment": "Clean baseline BM25 + raw conversation",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0175,
        "ndcg@10": 0.0936229490574533,
        "ndcg@20": 0.12222444085071704,
        "catalog_diversity": 0.040916912748826244,
        "lexical_diversity": 0.4877534278682761,
        "total_catalog_size": 47071,
        "source": "Historical local evaluator",
        "note": "Original BM25 baseline using raw conversation text and basic catalog fields.",
    },
    {
        "experiment": "Clean baseline BM25 + tag_list + raw conversation",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0175,
        "ndcg@10": 0.09162452088491813,
        "ndcg@20": 0.12487742965820994,
        "catalog_diversity": 0.048926090374115695,
        "lexical_diversity": 0.49106931629997636,
        "total_catalog_size": 47071,
        "source": "Historical local evaluator",
        "note": "BM25 baseline with tag_list added to the retrieval corpus. Tags improve top-20 recall but add noise.",
    },
    {
        "experiment": "BM25 + Gemini controlled keywords + Logistic Regression reranker",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0325,
        "ndcg@10": 0.11469899631494865,
        "ndcg@20": 0.15177859138557978,
        "catalog_diversity": 0.053217479977055934,
        "lexical_diversity": 0.5275797855262313,
        "total_catalog_size": 47071,
        "source": "Latest local evaluator",
        "note": "Current best local BM25 run. Gemini extracts controlled search terms; BM25 generates candidates; Logistic Regression reranks candidates using interpretable query-candidate features.",
    },
    {
        "experiment": "MiniLM + Gemini references + artist profile + decade",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0375,
        "ndcg@10": 0.07805621984489183,
        "ndcg@20": 0.09567336416729033,
        "catalog_diversity": 0.0687684561619681,
        "lexical_diversity": 0.518107476635514,
        "total_catalog_size": 47071,
        "source": "Latest local evaluator",
        "note": "Best retained MiniLM-only embedding run for Streamlit: Gemini reference tracks are embedded against catalog metadata enriched with artist profile and release decade.",
    },
]


st.set_page_config(
    page_title="Music CRS TFM Dashboard",
    page_icon="music",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --accent: #c4462f;
        --ink: #202124;
        --muted: #5f6368;
        --line: #e5e7eb;
        --soft: #f7f7f4;
    }
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1320px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    h1 {
        font-size: 2.2rem;
    }
    .tfm-callout {
        border-left: 4px solid var(--accent);
        background: var(--soft);
        padding: 0.9rem 1rem;
        margin: 0.4rem 0 1rem 0;
        color: var(--ink);
    }
    .small-muted {
        color: var(--muted);
        font-size: 0.92rem;
    }
    .router-box {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin-top: 0.8rem;
    }
    .router-box-label {
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 0.25rem;
    }
    .router-box-value {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 600;
        line-height: 1.35;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
    }
    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
    .message-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        margin: 0.55rem 0 0.15rem 0;
        text-transform: uppercase;
    }
    .message-label.music {
        color: #9a5a00;
    }
    .message-label.assistant {
        color: #46616f;
    }
    .message-label.user {
        color: #a33d31;
    }
    .transcript-window {
        max-height: 430px;
        overflow-y: auto;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fbfbfb;
        padding: 0.85rem;
        margin-bottom: 1rem;
    }
    .transcript-row {
        display: flex;
        margin: 0.45rem 0;
        width: 100%;
    }
    .transcript-row.left {
        justify-content: flex-start;
    }
    .transcript-row.right {
        justify-content: flex-end;
    }
    .transcript-bubble {
        max-width: 76%;
        border-radius: 8px;
        padding: 0.65rem 0.8rem;
        line-height: 1.45;
        border: 1px solid transparent;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }
    .transcript-bubble.user {
        background: #f4f4f6;
        border-color: #ececf0;
        color: #172033;
    }
    .transcript-bubble.current-user {
        background: #fff7f5;
        border-color: #f1b5ad;
        color: #172033;
    }
    .transcript-bubble.music {
        background: #eaf2ff;
        border-color: #cfe0f7;
        color: #0c3f87;
    }
    .transcript-bubble.assistant {
        background: #eaf7ed;
        border-color: #cfecd5;
        color: #17652f;
    }
    .transcript-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64707d;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
    }
    .transcript-text {
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(value):
    if pd.isna(value):
        return ""
    return (
        str(value)
        .replace("\u00e2\u20ac\u201d", "-")
        .replace("\u00e2\u2020\u2019", "->")
        .replace("\u00e2\u20ac\u2122", "'")
        .replace("\u00e2\u20ac\u0153", '"')
        .replace("\u00e2\u20ac\ufffd", '"')
    )


@st.cache_data
def load_devset_conversations(path):
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    return {
        (row["session_id"], int(row["turn_number"])): row
        for row in rows
    }


@st.cache_data
def load_bm25_explanations(path):
    if not path.exists():
        return pd.DataFrame()

    usecols = [
        "session_id",
        "turn_number",
        "rank",
        "track_name",
        "artist_name",
        "album_name",
        "tag_list",
        "release_decade",
        "is_ground_truth",
        "route_types",
        "keyword_json",
        "final_bm25_query",
        "matched_fields",
        "matched_terms",
        "matched_term_count",
        "track_name_matches",
        "artist_name_matches",
        "album_name_matches",
        "tag_list_matches",
        "release_decade_matches",
    ]
    df = pd.read_csv(path, usecols=usecols)
    text_columns = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_string_dtype(df[column])
    ]
    for column in text_columns:
        df[column] = df[column].map(clean_text)

    df["turn_number"] = pd.to_numeric(df["turn_number"], errors="coerce").astype("Int64")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    df["matched_term_count"] = pd.to_numeric(
        df["matched_term_count"],
        errors="coerce",
    ).fillna(0)
    if "is_ground_truth" in df.columns:
        df["is_ground_truth"] = df["is_ground_truth"].astype(str).str.lower().isin(["true", "1"])
    return df


@st.cache_data
def load_minilm_explanations(path):
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    text_columns = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_string_dtype(df[column])
    ]
    for column in text_columns:
        df[column] = df[column].map(clean_text)

    if "turn_number" in df.columns:
        df["turn_number"] = pd.to_numeric(df["turn_number"], errors="coerce").astype("Int64")
    if "rank" in df.columns:
        df["rank_number"] = pd.to_numeric(df["rank"], errors="coerce")
    if "gemini_reference_rank" in df.columns:
        df["gemini_reference_rank_number"] = pd.to_numeric(df["gemini_reference_rank"], errors="coerce")
    if "cosine_similarity" in df.columns:
        df["cosine_similarity"] = pd.to_numeric(df["cosine_similarity"], errors="coerce")
    if "is_ground_truth" in df.columns:
        df["is_ground_truth"] = df["is_ground_truth"].astype(str).str.lower().isin(["true", "1"])
    return df


@st.cache_data
def load_score_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_metric(value, digits=4):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def metric_grid(results):
    best_ndcg = results.sort_values("ndcg@20", ascending=False).iloc[0]
    col1, col2, col3, col4 = st.columns(4)

    if "composite_score" in results:
        best = results.sort_values("composite_score", ascending=False).iloc[0]
        col1.metric("Best composite score", format_metric(best["composite_score"]), best["experiment"])
    else:
        col1.metric("Models shown", len(results))
    col2.metric("Best nDCG@20", format_metric(best_ndcg["ndcg@20"]), best_ndcg["experiment"])
    col3.metric(
        "Best catalog diversity",
        format_metric(results["catalog_diversity"].max()) if "catalog_diversity" in results else "n/a",
    )
    col4.metric(
        "Best lexical diversity",
        format_metric(results["lexical_diversity"].max()) if "lexical_diversity" in results else "n/a",
    )


def render_header(title, caption):
    st.title(title)
    st.caption(caption)

    st.markdown(
        """
        <div class="tfm-callout">
        The central question is not only whether a recommendation sounds plausible, but whether the system
        retrieves the exact hidden ground-truth track inside the top 20. This dashboard keeps that distinction visible.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_project_overview():
    st.title("Music Conversational Recommender System")
    st.caption("A guided dashboard for understanding the challenge, the models, and the evaluation results.")

    st.markdown(
        """
        <div class="tfm-callout">
        This project studies a conversational music recommender system: given a multi-turn dialogue,
        the system must retrieve the exact catalog track that best matches the user's current request
        and generate a natural-language recommendation response.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### What This Challenge Asks The Model To Do")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 1. Understand The Conversation")
        st.write(
            "The model receives previous user messages, previous assistant replies, and previous music "
            "recommendations. The current turn may depend on corrections or preferences from earlier turns."
        )
    with col2:
        st.markdown("##### 2. Retrieve Tracks")
        st.write(
            "For each evaluated turn, the system returns up to 20 track IDs from a catalog of 47,071 songs."
        )
    with col3:
        st.markdown("##### 3. Generate A Response")
        st.write(
            "The system also writes a conversational answer explaining or presenting the recommendation."
        )

    st.markdown("#### Why Exact-Track Retrieval Matters")
    st.write(
        "The main ranking metric is nDCG@20. In this setup, each evaluated turn has one hidden "
        "ground-truth track. A musically reasonable recommendation can still score zero if that exact "
        "track is not in the top 20. This is why the dashboard separates retrieval quality from response quality."
    )

    st.markdown("#### Final Models Kept In This Dashboard")
    model_rows = [
        {
            "model": "BM25 + Gemini keywords + Logistic Regression reranker",
            "role": "Main retrieval model",
            "how_it_works": (
                "Gemini extracts controlled search terms from the conversation, BM25 retrieves candidate tracks "
                "from catalog metadata, and Logistic Regression reranks the candidates with interpretable features."
            ),
            "why_kept": "Best local nDCG@20 and the most aligned with exact hidden-track retrieval.",
        },
        {
            "model": "MiniLM + Gemini references + artist profile + decade",
            "role": "Embedding explanation model",
            "how_it_works": (
                "Gemini proposes reference tracks, and MiniLM retrieves nearby catalog tracks using enriched "
                "metadata with track, artist, album, artist style profile, and release decade."
            ),
            "why_kept": "Useful for analyzing semantic retrieval behavior, even though it is weaker than BM25 for exact-track nDCG@20.",
        },
    ]
    st.dataframe(pd.DataFrame(model_rows), width="stretch", hide_index=True)

    st.markdown("#### Main Finding")
    st.write(
        "The strongest retrieval strategy was not free-form generation or pure semantic embedding search. "
        "The best result came from constraining Gemini to extract catalog-relevant keywords, using BM25 for "
        "lexical candidate retrieval, and applying a lightweight Logistic Regression reranker to refine the top-20 list."
    )

    st.markdown("#### How To Use This Dashboard")
    st.markdown(
        """
        - Start with **Model Results** to see which retrieval strategies worked best.
        - Use **BM25 Explanation** to inspect the final model: controlled Gemini terms, retrieved tracks, and matched metadata evidence.
        - Use **MiniLM Explanation** to inspect the retained embedding model and compare its semantic retrieval behavior with BM25.
        """
    )

    st.divider()
    render_references()


def render_model_results():
    render_header(
        "Model Results",
        "Compare the final cleaned retrieval models kept for the project.",
    )

    st.markdown("#### Devset Subset Scores")
    st.caption(
        "Local evaluation on the first 50 conversations of TalkPlayData-Challenge-Dataset, with all 8 turns evaluated per conversation."
    )
    devset_df = pd.DataFrame(DEVSET_SUBSET_RESULTS)
    metric_grid(devset_df)
    display_columns = [
        "experiment",
        "split",
        "turns_evaluated",
        "ndcg@20",
        "catalog_diversity",
        "lexical_diversity",
        "note",
    ]
    st.dataframe(devset_df[display_columns], width="stretch", hide_index=True)

    st.markdown("#### Interpretation")
    st.markdown(
        """
        - For this challenge, the most important retrieval signal is nDCG@20: whether the exact target track appears in the top 20.
        - The current best local BM25 system uses Gemini as a controlled lexical query extractor and uses Logistic Regression to rerank BM25 candidates with interpretable query-candidate features.
        - Dense embedding variants retrieve semantically plausible neighborhoods, but they are weak for this evaluation because nDCG@20 rewards exact track recovery.
        - The embedding model is kept for analysis and explainability, while the learned BM25 reranker is the strongest nDCG@20 model in this branch.
        """
    )


def render_devset_conversation(detail):
    st.markdown("#### Conversation And Model Output")
    if detail is None:
        st.info("Conversation details are not available for this selected turn.")
        return

    st.markdown("##### Context Available To The Model")
    st.info(
        "The conversation history shown here is replayed from the original devset. "
        "Previous assistant responses are the dataset's provided responses, not responses generated by our model."
    )
    transcript_parts = ['<div class="transcript-window">']

    if not detail["conversation_history"]:
        transcript_parts.append(
            '<div class="small-muted">This is the first turn; no earlier conversation context is available.</div>'
        )

    for message in detail["conversation_history"]:
        content = message["content"]
        is_music_message = content.startswith("Recommended track:")
        if is_music_message:
            label = "Dataset music recommendation"
            side = "right"
            kind = "music"
        elif message["role"] == "assistant":
            label = "Dataset assistant response"
            side = "right"
            kind = "assistant"
        else:
            label = "User message"
            side = "left"
            kind = "user"

        transcript_parts.append(
            f'<div class="transcript-row {side}">'
            f'<div class="transcript-bubble {kind}">'
            f'<div class="transcript-label">{label}</div>'
            f'<div class="transcript-text">{html.escape(content)}</div>'
            "</div></div>"
        )

    transcript_parts.append(
        '<div class="transcript-row left">'
        '<div class="transcript-bubble current-user">'
        '<div class="transcript-label">Current user request</div>'
        f'<div class="transcript-text">{html.escape(detail["current_user_request"])}</div>'
        "</div></div>"
    )
    transcript_parts.append("</div>")
    st.markdown("".join(transcript_parts), unsafe_allow_html=True)

    predicted_tracks = detail["predicted_tracks"]
    ground_truth_id = detail["ground_truth_track"]["track_id"]
    ground_truth_rank = next(
        (
            rank
            for rank, track in enumerate(predicted_tracks, start=1)
            if track["track_id"] == ground_truth_id
        ),
        None,
    )
    top_track = predicted_tracks[0] if predicted_tracks else None
    rank_text = f"#{ground_truth_rank}" if ground_truth_rank else "Not in top 20"

    st.markdown("##### Model Prediction")
    metric_col1, metric_col2, metric_col3 = st.columns([1.1, 0.8, 1.1])
    metric_col1.metric(
        "Top recommendation",
        top_track["track_name"] if top_track else "n/a",
    )
    if top_track:
        metric_col1.caption(top_track["artist_name"])
    metric_col2.metric("Ground-truth rank", rank_text)
    metric_col3.metric(
        "Ground-truth track",
        detail["ground_truth_track"]["track_name"],
    )
    metric_col3.caption(detail["ground_truth_track"]["artist_name"])

    response_col, ranking_col = st.columns([1.15, 1])
    with response_col:
        st.markdown("##### Generated Response")
        if str(detail.get("predicted_response", "")).strip():
            st.write(detail["predicted_response"])
        else:
            st.info("This artifact was generated from a retrieval-only run, so no Llama response was saved.")

    with ranking_col:
        st.markdown("##### Ranked Tracks")
        ranking_table = pd.DataFrame(
            [
                {
                    "rank": rank,
                    "track": track["track_name"],
                    "artist": track["artist_name"],
                    "is_ground_truth": track["track_id"] == ground_truth_id,
                }
                for rank, track in enumerate(predicted_tracks, start=1)
            ]
        )
        st.dataframe(ranking_table, width="stretch", hide_index=True, height=360)


def parse_keyword_json(value):
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def split_terms(value):
    return [term.strip() for term in str(value or "").split(",") if term.strip()]


def unique_bm25_query_lines(query_text):
    seen = set()
    lines = []
    for raw_line in str(query_text or "").splitlines():
        line = raw_line.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines)


def format_router_route(route_types):
    route = str(route_types or "none").strip()
    labels = {
        "artist": "Prioritize artist matches",
        "title": "Prioritize track-title matches",
        "album": "Prioritize album matches",
        "decade": "Prioritize release-era matches",
        "negative": "Avoid rejected terms",
        "same_artist": "Prefer the same artist",
        "none": "No special reranking",
    }
    parts = [part.strip() for part in route.split(",") if part.strip()]
    if not parts:
        return labels["none"]
    return " + ".join(labels.get(part, part.replace("_", " ").title()) for part in parts)


def parse_bool_value(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_top20_retrieval_summary(df, session_col="session_id", turn_col="turn_number"):
    if df.empty or "is_ground_truth" not in df.columns or "rank" not in df.columns:
        return {}, pd.DataFrame()

    rows = []
    work_df = df.copy()
    work_df["rank_number"] = pd.to_numeric(work_df["rank"], errors="coerce")
    work_df["is_ground_truth_bool"] = work_df["is_ground_truth"].map(parse_bool_value)

    for (session_id, turn_number), group in work_df.groupby([session_col, turn_col]):
        top20 = group[group["rank_number"].between(1, 20, inclusive="both")]
        hit_rows = top20[top20["is_ground_truth_bool"]].sort_values("rank_number")
        if hit_rows.empty:
            ground_truth_rank = None
            ndcg = 0.0
            rank_bucket = "Not in top 20"
        else:
            ground_truth_rank = int(hit_rows.iloc[0]["rank_number"])
            ndcg = 1.0 / math.log2(ground_truth_rank + 1)
            if ground_truth_rank == 1:
                rank_bucket = "1"
            elif ground_truth_rank <= 5:
                rank_bucket = "2-5"
            elif ground_truth_rank <= 10:
                rank_bucket = "6-10"
            else:
                rank_bucket = "11-20"

        rows.append(
            {
                "session_id": session_id,
                "turn_number": turn_number,
                "ground_truth_rank": ground_truth_rank,
                "hit_at_20": ground_truth_rank is not None,
                "ndcg_at_20": ndcg,
                "rank_bucket": rank_bucket,
            }
        )

    turn_metrics = pd.DataFrame(rows)
    if turn_metrics.empty:
        return {}, turn_metrics

    summary = {
        "turns": len(turn_metrics),
        "ndcg_at_20": turn_metrics["ndcg_at_20"].mean(),
        "hit_rate_at_20": turn_metrics["hit_at_20"].mean(),
        "found": int(turn_metrics["hit_at_20"].sum()),
        "missed": int((~turn_metrics["hit_at_20"]).sum()),
    }
    return summary, turn_metrics


def render_bm25_explanation(df, conversation_details):
    render_header(
        "BM25 Explanation",
        "Inspect the lexical evidence behind BM25 retrieval and the learned reranker.",
    )
    if df.empty:
        st.warning("BM25 explanation data was not found.")
        st.code(
            "python refresh_streamlit_artifacts.py --run_dir exp/first_100/devset_bm25_gemini_keywords_query_type_router "
            "--gemini_cache_dir cache/gemini_keywords_devset_first100 --projection_retrieval_type sentence_transformer "
            "--corpus_types track_name artist_name album_name --topk_retrieved_per_reference 20",
            language="powershell",
        )
        return

    st.markdown(
        """
        **How to read this page.** BM25 is not an embedding model, so the useful visualization is not a 2D semantic map.
        This view explains the lexical matching process: Gemini extracts controlled search terms, BM25 retrieves
        catalog tracks whose metadata contains those terms, and the Logistic Regression reranker refines the order
        using interpretable query-candidate features.
        """
    )

    retrieval_summary, turn_metrics = build_top20_retrieval_summary(df)
    if retrieval_summary:
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("nDCG@20", f"{retrieval_summary['ndcg_at_20']:.4f}")
        metric_col2.metric("Hit rate@20", f"{retrieval_summary['hit_rate_at_20']:.1%}")
        metric_col3.metric("Target found", f"{retrieval_summary['found']}/{retrieval_summary['turns']}")
        metric_col4.metric("Avg matched terms", format_metric(df["matched_term_count"].mean(), digits=2))

    session_options = sorted(df["session_id"].dropna().unique())
    selected_session = st.sidebar.selectbox(
        "Session",
        session_options,
        key="bm25_session",
    )
    session_df = df[df["session_id"] == selected_session].copy()
    turn_options = sorted(session_df["turn_number"].dropna().unique(), key=int)
    selected_turn = st.sidebar.selectbox(
        "Turn",
        turn_options,
        key="bm25_turn",
    )
    turn_df = session_df[session_df["turn_number"] == selected_turn].sort_values("rank").copy()
    if turn_df.empty:
        st.info("No BM25 explanation rows are available for this turn.")
        return

    detail = conversation_details.get((selected_session, int(selected_turn)))
    render_devset_conversation(detail)

    first_row = turn_df.iloc[0]
    keyword_payload = parse_keyword_json(first_row.get("keyword_json", "{}"))
    route_types = first_row.get("route_types", "none")

    st.markdown("#### Controlled Query")
    st.caption(
        "This shows how the user's conversation is converted into BM25 search terms. Gemini extracts structured "
        "music clues, then the model searches the catalog for metadata that contains those clues."
    )
    left, right = st.columns([1, 1])
    with left:
        st.markdown("##### Gemini Extracted Fields")
        st.caption("The useful entities and descriptors Gemini found in the conversation.")
        keyword_rows = [
            {"field": field, "values": ", ".join(values) if isinstance(values, list) else ""}
            for field, values in keyword_payload.items()
            if values
        ]
        st.dataframe(pd.DataFrame(keyword_rows), width="stretch", hide_index=True, height=240)
    with right:
        st.markdown("##### BM25 Query Terms")
        st.caption("The final readable search terms sent to BM25.")
        st.code(unique_bm25_query_lines(first_row.get("final_bm25_query", "")), language="text")
        router_decision = html.escape(format_router_route(route_types))
        st.markdown(
            f"""
            <div class="router-box">
                <div class="router-box-label">Ranking focus</div>
                <div class="router-box-value">{router_decision}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "This indicates which metadata clues the model decided to prioritize when ordering the retrieved tracks."
        )

    st.markdown("#### Top-20 Retrieval Evidence")
    st.caption(
        "The final ranked tracks for this turn. The ground_truth column marks the correct answer when it appears."
    )
    if "tag_list" not in turn_df.columns:
        turn_df = turn_df.copy()
        turn_df["tag_list"] = ""
    display_table = turn_df[
        [
            "rank",
            "track_name",
            "artist_name",
            "album_name",
            "release_decade",
            "tag_list",
            "is_ground_truth",
        ]
    ].rename(
        columns={
            "track_name": "track",
            "artist_name": "artist",
            "album_name": "album",
            "release_decade": "decade",
            "tag_list": "tags",
            "is_ground_truth": "ground_truth",
        }
    )
    st.dataframe(display_table, width="stretch", hide_index=True, height=420)

    st.markdown("#### Matched-Term Matrix")
    st.caption(
        "A compact yes/no view of which query terms matched each retrieved track. It helps explain why BM25 placed "
        "certain tracks in the ranking."
    )
    selected_terms = []
    for field in [
        "track_titles",
        "artists",
        "albums",
        "genres",
        "moods",
        "instruments",
        "themes",
        "era",
        "must_include_terms",
        "avoid_terms",
    ]:
        values = keyword_payload.get(field, [])
        if isinstance(values, list):
            selected_terms.extend(str(value).strip().lower() for value in values if str(value).strip())
    selected_terms = list(dict.fromkeys(selected_terms))[:24]

    if not selected_terms:
        st.info("No Gemini keyword terms were exported for this turn.")
    else:
        matrix_rows = []
        for row in turn_df.itertuples(index=False):
            matched = {term.lower() for term in split_terms(row.matched_terms)}
            matrix_row = {
                "rank": int(row.rank),
                "track": row.track_name,
                "artist": row.artist_name,
                "matches": int(row.matched_term_count),
            }
            for term in selected_terms:
                matrix_row[term] = "yes" if term in matched else ""
            matrix_rows.append(matrix_row)

        matrix_df = pd.DataFrame(matrix_rows)
        term_columns = [term for term in selected_terms if term in matrix_df.columns]
        st.dataframe(
            matrix_df,
            width="stretch",
            hide_index=True,
            height=520,
            column_config={
                "rank": st.column_config.NumberColumn("rank", width="small"),
                "track": st.column_config.TextColumn("track", width="medium"),
                "artist": st.column_config.TextColumn("artist", width="medium"),
                "matches": st.column_config.NumberColumn("matches", width="small"),
                **{
                    term: st.column_config.TextColumn(term, width="small")
                    for term in term_columns
                },
            },
        )

    st.markdown("#### Field Match Breakdown")
    st.caption(
        "A summary of where the matches came from: track title, artist, album, tags, or release decade."
    )
    field_columns = [
        "track_name_matches",
        "artist_name_matches",
        "album_name_matches",
        "tag_list_matches",
        "release_decade_matches",
    ]
    field_summary = []
    for column in field_columns:
        field_summary.append(
            {
                "field": column.replace("_matches", ""),
                "tracks_with_match": int(turn_df[column].fillna("").astype(str).str.len().gt(0).sum()),
                "all_matched_terms": ", ".join(
                    sorted(
                        {
                            term
                            for value in turn_df[column].fillna("")
                            for term in split_terms(value)
                        }
                    )
                ),
            }
        )
    st.dataframe(pd.DataFrame(field_summary), width="stretch", hide_index=True)


def render_minilm_explanation(df, conversation_details, scores):
    render_header(
        "MiniLM Explanation",
        "Inspect the MiniLM embedding model without PCA or UMAP projection.",
    )
    if df.empty:
        st.warning("MiniLM explanation data was not found.")
        st.code(
            "python export_minilm_explanation_to_csv.py "
            "--run_dir exp/first_100/devset_gemini_multiquery_minilm_streamlit_top20 "
            "--gemini_cache_dir cache/gemini_expansions_devset_first100 "
            "--output_csv visualize_streamlit/minilm_reference_retrieval_table.csv "
            "--topk_retrieved_per_reference 20 "
            "--corpus_types track_name artist_name album_name artist_style_profile release_decade "
            "--device cuda",
            language="powershell",
        )
        return

    st.markdown(
        """
        **How to read this page.** Gemini first proposes five reference tracks from the conversation.
        Each reference track is embedded with MiniLM using the same catalog metadata fields:
        track, artist, album, artist style profile, and release decade. The tables below show the final
        top-20 recommendation list and the nearest catalog tracks retrieved from each Gemini reference.
        """
    )

    final_df = df[df["row_type"] == "final_recommendation"].copy()
    ground_truth_df = df[df["row_type"] == "ground_truth"].copy()
    reference_df = df[df["row_type"] == "gemini_reference"].copy()
    retrieved_df = df[df["row_type"] == "retrieved_from_reference"].copy()

    hit_rows = final_df[final_df["is_ground_truth"]]
    found_turns = hit_rows[["session_id", "turn_number"]].drop_duplicates()
    total_turns = ground_truth_df[["session_id", "turn_number"]].drop_duplicates()
    turn_count = int(scores.get("subset_turns", len(total_turns)))
    hit_rate = len(found_turns) / turn_count if turn_count else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("nDCG@20", format_metric(scores.get("ndcg@20")))
    col2.metric("Hit rate@20", f"{hit_rate:.1%}")
    col3.metric("Target found", f"{len(found_turns)}/{turn_count}")

    session_options = sorted(reference_df["session_id"].dropna().unique())
    if not session_options:
        st.info("No Gemini reference tracks are available in the MiniLM explanation table.")
        return

    selected_session = st.sidebar.selectbox("Session", session_options, key="minilm_session")
    session_turns = reference_df[reference_df["session_id"] == selected_session]["turn_number"].dropna().unique()
    turn_options = sorted([int(turn) for turn in session_turns])
    selected_turn = st.sidebar.selectbox("Turn", turn_options, key="minilm_turn")

    turn_final = final_df[
        (final_df["session_id"] == selected_session) & (final_df["turn_number"] == selected_turn)
    ].sort_values("rank_number").copy()
    turn_ground_truth = ground_truth_df[
        (ground_truth_df["session_id"] == selected_session) & (ground_truth_df["turn_number"] == selected_turn)
    ].copy()
    turn_references = reference_df[
        (reference_df["session_id"] == selected_session) & (reference_df["turn_number"] == selected_turn)
    ].sort_values("gemini_reference_rank_number").copy()
    turn_retrieved = retrieved_df[
        (retrieved_df["session_id"] == selected_session) & (retrieved_df["turn_number"] == selected_turn)
    ].sort_values(
        ["gemini_reference_rank_number", "rank_number"]
    ).copy()

    detail = conversation_details.get((selected_session, int(selected_turn)))
    render_devset_conversation(detail)

    st.markdown("#### Ground Truth And Gemini References")
    summary_rows = []
    for row in turn_ground_truth.itertuples(index=False):
        summary_rows.append(
            {
                "type": "Ground truth",
                "rank": "",
                "track": row.track_name,
                "artist": row.artist_name,
                "album": row.album_name,
                "tags": row.tag_list,
            }
        )
    for row in turn_references.itertuples(index=False):
        summary_rows.append(
            {
                "type": "Gemini reference",
                "rank": str(int(row.gemini_reference_rank_number)),
                "track": row.track_name,
                "artist": row.artist_name,
                "album": row.album_name,
                "tags": row.tag_list,
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    st.markdown("#### Final Top-20 Recommendations")
    final_table = turn_final[
        ["rank", "track_name", "artist_name", "album_name", "tag_list", "is_ground_truth"]
    ].rename(
        columns={
            "track_name": "track",
            "artist_name": "artist",
            "album_name": "album",
            "tag_list": "tags",
        }
    )
    st.dataframe(final_table, width="stretch", hide_index=True, height=420)

    st.markdown("#### MiniLM Nearest Tracks For Each Gemini Reference")
    st.caption("Each table shows the catalog tracks retrieved from one Gemini-generated reference song before final fusion.")
    if turn_retrieved.empty:
        st.info("No per-reference MiniLM retrieval rows are available for this turn.")
        return

    ground_truth_id = turn_ground_truth.iloc[0]["track_id"] if not turn_ground_truth.empty else None
    reference_labels = {
        int(row.gemini_reference_rank_number): f"{int(row.gemini_reference_rank_number)}. {row.track_name} - {row.artist_name}"
        for row in turn_references.itertuples(index=False)
    }
    for reference_rank, reference_rows in turn_retrieved.groupby("gemini_reference_rank_number"):
        reference_rows = reference_rows.sort_values("rank_number").copy()
        found = reference_rows[reference_rows["track_id"] == ground_truth_id]
        status = "target not found"
        if not found.empty:
            status = f"target found at rank {int(found['rank_number'].min())}"
        label = reference_labels.get(int(reference_rank), f"Gemini reference {int(reference_rank)}")
        with st.expander(f"{label} - {status}"):
            table = reference_rows[
                ["rank", "cosine_similarity", "track_name", "artist_name", "album_name", "tag_list", "is_ground_truth"]
            ].rename(
                columns={
                    "track_name": "track",
                    "artist_name": "artist",
                    "album_name": "album",
                    "tag_list": "tags",
                }
            )
            st.dataframe(table, width="stretch", hide_index=True, height=min(420, 84 + 36 * len(table)))


def render_references():
    st.subheader("Challenge References")
    for label, url in REFERENCE_LINKS.items():
        st.markdown(f"- [{label}]({url})")


def set_active_page(page):
    st.session_state.active_page = page


def main():
    valid_pages = {
        "Project Overview",
        "Model Results",
        "MiniLM Explanation",
        "BM25 Explanation",
    }
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Project Overview"
    elif st.session_state.active_page not in valid_pages:
        st.session_state.active_page = "Project Overview"

    with st.sidebar:
        st.header("Navigation")
        st.button(
            "Project Overview",
            width="stretch",
            type="primary" if st.session_state.active_page == "Project Overview" else "secondary",
            on_click=set_active_page,
            args=("Project Overview",),
        )
        st.button(
            "Model Results",
            width="stretch",
            type="primary" if st.session_state.active_page == "Model Results" else "secondary",
            on_click=set_active_page,
            args=("Model Results",),
        )
        st.button(
            "MiniLM Explanation",
            width="stretch",
            type="primary" if st.session_state.active_page == "MiniLM Explanation" else "secondary",
            on_click=set_active_page,
            args=("MiniLM Explanation",),
        )
        st.button(
            "BM25 Explanation",
            width="stretch",
            type="primary" if st.session_state.active_page == "BM25 Explanation" else "secondary",
            on_click=set_active_page,
            args=("BM25 Explanation",),
        )

        page = st.session_state.active_page
        st.divider()
        if page == "Project Overview":
            st.caption("Start here for the challenge goal, data, and model families.")
        elif page == "Model Results":
            st.caption("Compare model performance and understand each metric.")
        elif page == "MiniLM Explanation":
            st.caption("Inspect MiniLM references, final recommendations, and per-reference retrieval tables.")
        elif page == "BM25 Explanation":
            st.caption("Inspect controlled BM25 query terms, matches, and reranker evidence.")

    if page == "Project Overview":
        render_project_overview()
    elif page == "MiniLM Explanation":
        minilm_explanation_df = load_minilm_explanations(MINILM_EXPLANATION_PATH)
        devset_conversations = load_devset_conversations(DEVSET_CONVERSATION_PATH)
        minilm_scores = load_score_json(MINILM_SCORES_PATH)
        render_minilm_explanation(minilm_explanation_df, devset_conversations, minilm_scores)
    elif page == "BM25 Explanation":
        bm25_explanation_df = load_bm25_explanations(BM25_EXPLANATION_PATH)
        devset_conversations = load_devset_conversations(DEVSET_CONVERSATION_PATH)
        render_bm25_explanation(bm25_explanation_df, devset_conversations)
    else:
        render_model_results()


if __name__ == "__main__":
    main()
