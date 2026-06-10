from pathlib import Path
import json
import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
TFM_ROOT = ROOT.parent
CSV_PATH = ROOT / "visualize_streamlit" / "gemini_to_catalog_similarity_table.csv"
DEVSET_GEMINI_PATH = ROOT / "visualize_streamlit" / "devset_gemini_ground_truth_table.csv"
DEVSET_CONVERSATION_PATH = ROOT / "visualize_streamlit" / "devset_conversation_details.json"
EMBEDDING_PROJECTION_PATH = ROOT / "visualize_streamlit" / "gemini_embedding_projection.csv"
SCORES_DIR = TFM_ROOT / "music-crs-evaluator" / "exp" / "scores" / "devset"
PREDICTIONS_DIR = TFM_ROOT / "music-crs-evaluator" / "exp" / "inference" / "devset"

REFERENCE_LINKS = {
    "RecSys Challenge 2026": "https://www.recsyschallenge.com/2026/",
    "Music-CRS Challenge": "https://nlp4musa.github.io/music-crs-challenge/",
    "talkpl-ai Hugging Face": "https://huggingface.co/talkpl-ai",
    "Codabench": "https://www.codabench.org/competitions/15786/",
    "Baseline repository": "https://github.com/JiaHaoHuangXia/music-crs-baselines",
}

EXPERIMENT_NOTES = {
    "random": "Lower bound. High catalog coverage but almost no retrieval accuracy.",
    "popularity": "Lower bound. Recommends globally frequent tracks, so diversity is very low.",
    "prediction": "Current devset BM25 baseline artifact in the evaluator folder.",
    "llama1b_bm25_devset": "Llama 3.2 1B response generator with BM25 retrieval on devset.",
}

MANUAL_RESULTS = [
    {
        "experiment": "BM25 + tag_list",
        "split": "Blind A",
        "ndcg@20": 0.1807,
        "catalog_diversity": 0.0307,
        "lexical_diversity": 0.6723,
        "llm_judge_score": 2.8000,
        "composite_score": 0.2956,
        "source": "Codabench",
        "note": "Best observed model so far. Tags give BM25 strong exact lexical signals.",
    },
    {
        "experiment": "BM25 + Gemini + tag_list",
        "split": "Blind A",
        "ndcg@20": 0.1630,
        "catalog_diversity": 0.0307,
        "lexical_diversity": 0.6626,
        "llm_judge_score": 2.2500,
        "composite_score": 0.2446,
        "source": "Codabench",
        "note": "Gemini tag expansion is competitive, but it still underperforms plain BM25 + tags.",
    },
    {
        "experiment": "BM25",
        "split": "Blind A",
        "ndcg@20": 0.1357,
        "catalog_diversity": 0.0214,
        "lexical_diversity": 0.6376,
        "llm_judge_score": 2.3000,
        "composite_score": 0.2312,
        "source": "Codabench",
        "note": "Strong lexical baseline without tag_list in the catalog fields.",
    },
    {
        "experiment": "BERT + Gemini",
        "split": "Blind A",
        "ndcg@20": 0.0159,
        "catalog_diversity": 0.0233,
        "lexical_diversity": 0.6591,
        "llm_judge_score": 3.1500,
        "composite_score": 0.2374,
        "source": "Codabench",
        "note": "Best LLM judge score, but retrieval remains far below BM25 variants.",
    },
    {
        "experiment": "BERT + Gemini + tag + conversation",
        "split": "Blind A",
        "ndcg@20": 0.0175,
        "catalog_diversity": 0.0163,
        "lexical_diversity": 0.6565,
        "llm_judge_score": 1.8000,
        "composite_score": 0.1360,
        "source": "Codabench",
        "note": "Keeping the conversation helps slightly over some Gemini-BERT variants, but not enough.",
    },
    {
        "experiment": "BERT + Gemini + tag",
        "split": "Blind A",
        "ndcg@20": 0.0150,
        "catalog_diversity": 0.0240,
        "lexical_diversity": 0.6530,
        "llm_judge_score": 2.0000,
        "composite_score": 0.1502,
        "source": "Codabench",
        "note": "Plausible semantic expansion, weak exact hidden-track recovery.",
    },
    {
        "experiment": "BERT",
        "split": "Blind A",
        "ndcg@20": 0.0112,
        "catalog_diversity": 0.0117,
        "lexical_diversity": 0.6466,
        "llm_judge_score": 1.4500,
        "composite_score": 0.1052,
        "source": "Codabench",
        "note": "Dense retrieval alone performs poorly for this exact-track task.",
    },
    {
        "experiment": "BERT + tag_list",
        "split": "Blind A",
        "ndcg@20": 0.0000,
        "catalog_diversity": 0.0130,
        "lexical_diversity": 0.6325,
        "llm_judge_score": 2.3500,
        "composite_score": 0.1658,
        "source": "Codabench",
        "note": "Adding noisy tags to dense embeddings collapses exact-track retrieval in this run.",
    },
]

DEVSET_SUBSET_RESULTS = [
    {
        "experiment": "BM25 + Gemini + tag_list",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0150,
        "ndcg@10": 0.08928339778789217,
        "ndcg@20": 0.1206117972983663,
        "catalog_diversity": 0.04661043954876676,
        "lexical_diversity": 0.4865810019518543,
        "total_catalog_size": 47071,
        "source": "Local evaluator",
        "note": "Previous first-50 devset reference run.",
    },
    {
        "experiment": "BERT + Gemini multi-query fusion + tag_list",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0050,
        "ndcg@10": 0.0172569051749283,
        "ndcg@20": 0.022337180121647184,
        "catalog_diversity": 0.037793970810052896,
        "lexical_diversity": 0.4600994125621328,
        "total_catalog_size": 47071,
        "source": "Local evaluator",
        "note": "Each Gemini reference track is embedded separately; rankings are fused with RRF.",
    },
    {
        "experiment": "BERT + Gemini multi-query fusion + tag_list + conversation",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0050,
        "ndcg@10": 0.01896599825802791,
        "ndcg@20": 0.025201424361161535,
        "catalog_diversity": 0.04100189076076565,
        "lexical_diversity": 0.46515103482632125,
        "total_catalog_size": 47071,
        "source": "Local evaluator",
        "note": "Hybrid fusion: Gemini reference queries plus original conversation with higher weight.",
    },
    {
        "experiment": "BERT + Gemini multi-query fusion, no tag_list",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0100,
        "ndcg@10": 0.03209145337491858,
        "ndcg@20": 0.03769050564476374,
        "catalog_diversity": 0.03526587495485543,
        "lexical_diversity": 0.4684398570861453,
        "total_catalog_size": 47071,
        "source": "Local evaluator",
        "note": "No-tag rerun. This overwrote the original folder, but improved over the tag-list multi-query variant.",
    },
    {
        "experiment": "BERT + Gemini multi-query fusion + conversation, no tag_list",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0050,
        "ndcg@10": 0.02742079093061506,
        "ndcg@20": 0.0336234679375138,
        "catalog_diversity": 0.03354507021308237,
        "lexical_diversity": 0.47563773788635444,
        "total_catalog_size": 47071,
        "source": "Local evaluator",
        "note": "No-tag hybrid rerun. The original conversation did not help as much as in the tag-list setting.",
    },
    {
        "experiment": "BERT + Gemini top-2 fusion, no tag_list/release_date",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0100,
        "ndcg@10": 0.02586563444205212,
        "ndcg@20": 0.03042129429837016,
        "catalog_diversity": 0.036583034139916294,
        "lexical_diversity": 0.4625166473342785,
        "total_catalog_size": 47071,
        "source": "Local evaluator",
        "note": "Uses only Gemini references 1 and 2. Corpus fields: track, artist, album.",
    },
    {
        "experiment": "BERT + Gemini top-2 fusion, no tag_list/release_date, topk folder",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0100,
        "ndcg@10": 0.02586563444205212,
        "ndcg@20": 0.03042129429837016,
        "catalog_diversity": 0.036583034139916294,
        "lexical_diversity": 0.46150862068965515,
        "total_catalog_size": 47071,
        "source": "Local evaluator",
        "note": "This result folder was named topk10, but the config still used topk=50. Config has now been fixed; rerun for a true topk=10 score.",
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


def tag_overlap(gemini_tags, catalog_tags):
    gemini_set = {
        tag.strip().lower()
        for tag in str(gemini_tags).split(",")
        if tag.strip()
    }
    catalog_set = {
        tag.strip().lower()
        for tag in str(catalog_tags).split(",")
        if tag.strip()
    }

    if not gemini_set or not catalog_set:
        return 0, ""

    overlap = gemini_set.intersection(catalog_set)
    return len(overlap), ", ".join(sorted(overlap))


@st.cache_data
def load_gemini_table(path):
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

    overlaps = df.apply(
        lambda row: tag_overlap(row["gemini_tag_list"], row["catalog_tag_list"]),
        axis=1,
    )
    df["tag_overlap_count"] = [x[0] for x in overlaps]
    df["tag_overlap_terms"] = [x[1] for x in overlaps]
    df["gemini_label"] = (
        df["gemini_pseudo_rank"].astype(str)
        + ". "
        + df["gemini_track_name"].astype(str)
        + " - "
        + df["gemini_artist_name"].astype(str)
    )
    df["catalog_label"] = (
        df["catalog_similarity_rank"].astype(str)
        + ". "
        + df["catalog_track_name"].astype(str)
        + " - "
        + df["catalog_artist_name"].astype(str)
    )
    return df


@st.cache_data
def load_devset_ground_truth_table(path):
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

    df["gemini_label"] = (
        df["gemini_reference_rank"].astype(str)
        + ". "
        + df["gemini_track_name"].astype(str)
        + " - "
        + df["gemini_artist_name"].astype(str)
    )
    df["ground_truth_label"] = (
        df["ground_truth_track_name"].astype(str)
        + " - "
        + df["ground_truth_artist_name"].astype(str)
    )
    return df


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
def load_embedding_projection(path):
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
        turn_numbers = pd.to_numeric(df["turn_number"], errors="coerce").astype("Int64")
        df["turn_number_key"] = turn_numbers.astype(str).replace("<NA>", "")
    if "gemini_reference_rank" in df.columns:
        ranks = pd.to_numeric(df["gemini_reference_rank"], errors="coerce").astype("Int64")
        df["gemini_reference_rank_key"] = ranks.astype(str).replace("<NA>", "")
    if "retrieved_rank" in df.columns:
        ranks = pd.to_numeric(df["retrieved_rank"], errors="coerce").astype("Int64")
        df["retrieved_rank_key"] = ranks.astype(str).replace("<NA>", "")

    return df


@st.cache_data
def load_score_files(scores_dir):
    rows = []
    if not scores_dir.exists():
        return pd.DataFrame()

    for path in sorted(scores_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        experiment = path.stem
        rows.append(
            {
                "experiment": experiment,
                "ndcg@1": data.get("ndcg@1"),
                "ndcg@10": data.get("ndcg@10"),
                "ndcg@20": data.get("ndcg@20"),
                "catalog_diversity": data.get("catalog_diversity"),
                "lexical_diversity": data.get("lexical_diversity"),
                "total_catalog_size": data.get("total_catalog_size"),
                "note": EXPERIMENT_NOTES.get(experiment, ""),
                "file": str(path),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data
def load_predictions(predictions_dir):
    options = {}
    if not predictions_dir.exists():
        return options

    for path in sorted(predictions_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            options[path.stem] = pd.DataFrame(json.load(f))
    return options


def format_metric(value, digits=4):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def metric_grid(results):
    best = results.sort_values("composite_score", ascending=False).iloc[0]
    best_ndcg = results.sort_values("ndcg@20", ascending=False).iloc[0]
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Best composite score", format_metric(best["composite_score"]), best["experiment"])
    col2.metric("Best nDCG@20", format_metric(best_ndcg["ndcg@20"]), best_ndcg["experiment"])
    col3.metric("Experiments", len(results))
    col4.metric("Best LLM judge score", format_metric(results["llm_judge_score"].max()))


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

    st.markdown("#### Models Compared")
    model_rows = [
        {
            "model": "BM25",
            "idea": "Keyword matching over track metadata.",
            "strength": "Good when the conversation shares exact words with track names, artists, albums, or tags.",
            "risk": "Cannot understand paraphrases deeply.",
        },
        {
            "model": "BM25 + tag_list",
            "idea": "BM25 with user-generated/music metadata tags included in the searchable catalog text.",
            "strength": "Best observed system because tags give strong lexical clues.",
            "risk": "Still depends on exact vocabulary overlap.",
        },
        {
            "model": "BERT",
            "idea": "Dense embedding retrieval using text representations.",
            "strength": "Can group semantically similar metadata.",
            "risk": "Semantic neighbors are not always the exact hidden target track.",
        },
        {
            "model": "Gemini expansion",
            "idea": "Gemini generates reference songs/tags from the conversation to enrich the retrieval query.",
            "strength": "Can improve interpretation and response-side quality.",
            "risk": "Can cause query drift by suggesting plausible songs that point away from the target.",
        },
    ]
    st.dataframe(pd.DataFrame(model_rows), width="stretch", hide_index=True)

    st.markdown("#### How To Use This Dashboard")
    st.markdown(
        """
        - Start with **Model Results** to see which retrieval strategies worked best.
        - Use **Devset Case Study** to inspect one conversation turn where the ground truth is known.
        - Use **Blindset Retrieval Explorer** to see how Gemini reference tracks map into nearby catalog regions.
        """
    )


def render_model_results():
    render_header(
        "Model Results",
        "Compare retrieval and response metrics for the model variants tested in this project.",
    )
    manual_df = pd.DataFrame(MANUAL_RESULTS).sort_values(
        "composite_score",
        ascending=False,
    )
    metric_grid(manual_df)

    st.markdown("#### Blindset-A Codabench Scores")
    st.caption(
        "Official Blindset-A submission results. The true target tracks are hidden, so only aggregate metrics are available."
    )
    st.dataframe(manual_df, width="stretch", hide_index=True)

    blind_metric = st.segmented_control(
        "Compare models by",
        ["composite_score", "ndcg@20", "llm_judge_score", "lexical_diversity", "catalog_diversity"],
        default="composite_score",
    )
    blind_fig = px.bar(
        manual_df.sort_values(blind_metric, ascending=False),
        x=blind_metric,
        y="experiment",
        orientation="h",
        color="experiment",
        text=blind_metric,
        title=f"Blind-A comparison by {blind_metric}",
    )
    blind_fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    blind_fig.update_layout(
        height=520,
        showlegend=False,
        xaxis_title=blind_metric,
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(blind_fig, width="stretch")

    st.markdown("#### Devset Subset Scores")
    st.caption(
        "Local evaluation on the first 50 conversations of TalkPlayData-Challenge-Dataset, "
        "with all 8 turns evaluated per conversation. These scores are not directly comparable "
        "to the Blindset-A Codabench results."
    )
    devset_df = pd.DataFrame(DEVSET_SUBSET_RESULTS)
    st.dataframe(devset_df, width="stretch", hide_index=True)

    st.markdown("#### Interpretation")
    st.markdown(
        """
        - For this challenge, the most important retrieval signal is nDCG@20: whether the exact target track appears in the top 20.
        - BM25 + tag_list is the strongest current system, with the best nDCG@20 and composite score.
        - Gemini can improve natural-language response quality, but it does not consistently improve exact hidden-track retrieval.
        - BERT variants retrieve semantically plausible neighborhoods, but they are weak for this evaluation because nDCG@20 rewards exact track recovery.
        - The most useful thesis comparison is BM25 vs BM25 + tag_list vs BERT/Gemini variants, because it shows the difference between lexical matching and semantic query drift.
        """
    )


def render_prediction_explorer(prediction_options):
    st.subheader("Prediction Explorer")
    if not prediction_options:
        st.warning("No prediction files were found.")
        return

    names = list(prediction_options.keys())
    selected_name = st.selectbox("Prediction file", names)
    df = prediction_options[selected_name].copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", len(df))
    col2.metric("Sessions", df["session_id"].nunique() if "session_id" in df else "n/a")
    col3.metric("Users", df["user_id"].nunique() if "user_id" in df else "n/a")
    col4.metric("Max turn", int(df["turn_number"].max()) if "turn_number" in df else "n/a")

    session_options = sorted(df["session_id"].dropna().unique())
    selected_session = st.selectbox("Session", session_options)
    session_df = df[df["session_id"] == selected_session].copy()

    turns = sorted(session_df["turn_number"].dropna().unique())
    selected_turn = st.selectbox("Turn", turns)
    row = session_df[session_df["turn_number"] == selected_turn].iloc[0]

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("#### Top 20 Track IDs")
        tracks = pd.DataFrame(
            {
                "rank": range(1, len(row["predicted_track_ids"]) + 1),
                "track_id": row["predicted_track_ids"],
            }
        )
        st.dataframe(tracks, width="stretch", hide_index=True)

    with right:
        st.markdown("#### Generated Response")
        st.write(clean_text(row.get("predicted_response", "")))

    st.markdown("#### Raw Row")
    st.json(row.to_dict())


def render_gemini_analysis(df):
    render_header(
        "Blindset Retrieval Explorer",
        "Inspect how Gemini-generated reference tracks map to nearby real catalog tracks in BERT embedding space.",
    )
    if df.empty:
        st.warning("Gemini similarity table was not found.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", len(df))
    col2.metric("Sessions", df["session_id"].nunique())
    col3.metric(
        "Gemini references",
        df[["session_id", "turn_number", "gemini_pseudo_rank"]].drop_duplicates().shape[0],
    )
    col4.metric("Avg similarity", format_metric(df["cosine_similarity"].mean()))

    st.markdown(
        """
        **How to read this page.** Blindset-A does not reveal the correct target tracks, so this view cannot
        say whether Gemini found the right answer. Instead, it diagnoses *where Gemini points retrieval*:
        for each conversation turn, Gemini generated five reference songs, and this page shows their nearest
        real catalog neighbors in BERT embedding space.
        """
    )

    session_options = sorted(df["session_id"].dropna().unique())
    selected_session = st.sidebar.selectbox("Session", session_options)
    session_df = df[df["session_id"] == selected_session].copy()

    turn_options = sorted(session_df["turn_number"].dropna().unique())
    selected_turn = st.sidebar.selectbox("Turn", turn_options)
    turn_df = session_df[session_df["turn_number"] == selected_turn].copy()

    gemini_options = sorted(turn_df["gemini_label"].dropna().unique())
    selected_gemini = st.sidebar.selectbox(
        "Gemini-generated reference track",
        gemini_options,
        help="Choose one song suggested by Gemini. The nearest catalog matches below will update.",
    )
    selected_df = turn_df[turn_df["gemini_label"] == selected_gemini].copy()

    min_similarity = st.sidebar.slider(
        "Minimum cosine similarity",
        float(df["cosine_similarity"].min()),
        float(df["cosine_similarity"].max()),
        float(df["cosine_similarity"].min()),
        step=0.001,
    )
    selected_df = selected_df[selected_df["cosine_similarity"] >= min_similarity]

    if selected_df.empty:
        st.info("No catalog matches pass the current similarity filter.")
        return

    first_row = selected_df.iloc[0]
    left, right = st.columns([1, 1.5])
    with left:
        st.markdown("#### Selected Gemini Reference")
        st.write(f"Track: {first_row['gemini_track_name']}")
        st.write(f"Artist: {first_row['gemini_artist_name']}")
        st.write(f"Album: {first_row['gemini_album_name']}")
        st.write(f"Release date: {first_row['gemini_release_date']}")
    with right:
        st.markdown("#### Extracted Tags")
        st.info(first_row["gemini_tag_list"])

    fig = px.bar(
        selected_df.sort_values("catalog_similarity_rank"),
        x="cosine_similarity",
        y="catalog_label",
        orientation="h",
        color="tag_overlap_count",
        hover_data=[
            "catalog_track_id",
            "catalog_album_name",
            "catalog_tag_list",
            "tag_overlap_terms",
        ],
        title="Nearest real catalog tracks by BERT cosine similarity",
    )
    fig.update_layout(
        yaxis_title="Catalog track",
        xaxis_title="Cosine similarity",
        yaxis=dict(autorange="reversed"),
        height=560,
    )
    st.plotly_chart(fig, width="stretch")

    display_cols = [
        "catalog_similarity_rank",
        "cosine_similarity",
        "catalog_track_id",
        "catalog_track_name",
        "catalog_artist_name",
        "catalog_album_name",
        "catalog_tag_list",
        "tag_overlap_count",
        "tag_overlap_terms",
        "catalog_release_date",
    ]
    st.dataframe(
        selected_df[display_cols].sort_values("catalog_similarity_rank"),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### All Gemini References For This Turn")
    session_summary = (
        turn_df.groupby(
            ["gemini_pseudo_rank", "gemini_track_name", "gemini_artist_name"],
            as_index=False,
        )
        .agg(
            avg_similarity=("cosine_similarity", "mean"),
            max_similarity=("cosine_similarity", "max"),
            avg_tag_overlap=("tag_overlap_count", "mean"),
            retrieved_catalog_tracks=("catalog_track_id", "nunique"),
        )
    )

    fig2 = px.scatter(
        session_summary,
        x="avg_similarity",
        y="avg_tag_overlap",
        size="retrieved_catalog_tracks",
        color="gemini_pseudo_rank",
        hover_data=[
            "gemini_track_name",
            "gemini_artist_name",
            "max_similarity",
            "retrieved_catalog_tracks",
        ],
        title="Do the Gemini references point to the same catalog region?",
    )
    fig2.update_layout(
        xaxis_title="Average cosine similarity to top catalog matches",
        yaxis_title="Average tag overlap",
        height=480,
    )
    st.plotly_chart(fig2, width="stretch")

    st.dataframe(
        session_summary.sort_values("gemini_pseudo_rank"),
        width="stretch",
        hide_index=True,
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
        st.write(detail["predicted_response"])

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


def render_devset_comparison(df, conversation_details):
    render_header(
        "Devset Case Study",
        "Inspect conversations where the true target track is known, then compare Gemini references with that target.",
    )
    if df.empty:
        st.warning("Devset Gemini ground-truth table was not found.")
        return

    session_turn_count = df[["session_id", "turn_number"]].drop_duplicates().shape[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", len(df))
    col2.metric("Sessions", df["session_id"].nunique())
    col3.metric("Evaluated turns", session_turn_count)
    col4.metric("Avg tag overlap", format_metric(df["tag_overlap_count"].mean(), digits=2))

    st.markdown(
        """
        **How to read this page.** The devset contains known ground-truth tracks, so it can be used as a
        controlled case study. For each selected turn, you can inspect the conversation, the model response,
        where the correct track appeared in the top 20, and whether Gemini's generated reference songs share
        metadata tags with the true target.
        """
    )

    session_options = sorted(df["session_id"].dropna().unique())
    selected_session = st.sidebar.selectbox(
        "Session",
        session_options,
        key="devset_session",
    )
    session_df = df[df["session_id"] == selected_session].copy()

    turn_options = sorted(session_df["turn_number"].dropna().unique())
    selected_turn = st.sidebar.selectbox(
        "Turn",
        turn_options,
        key="devset_turn",
    )
    turn_df = session_df[session_df["turn_number"] == selected_turn].copy()

    selected_detail = conversation_details.get((selected_session, int(selected_turn)))
    first_row = turn_df.sort_values("gemini_reference_rank").iloc[0]

    render_devset_conversation(selected_detail)

    st.markdown("#### Target Track And Gemini References")
    st.markdown("##### Ground-Truth Track")
    target_col1, target_col2, target_col3, target_col4 = st.columns([1, 1, 1.5, 1])
    target_col1.write(f"Track: {first_row['ground_truth_track_name']}")
    target_col2.write(f"Artist: {first_row['ground_truth_artist_name']}")
    target_col3.write(f"Album: {first_row['ground_truth_album_name']}")
    target_col4.write(f"Release date: {first_row['ground_truth_release_date']}")
    st.info(first_row["ground_truth_tag_list"])

    st.markdown("##### Gemini-Generated Reference Tracks")
    st.caption(
        "Gemini produced these five reference songs as query-expansion clues. "
        "They are not final recommendations; they help explain where Gemini tried to steer retrieval."
    )
    reference_display = turn_df[
        [
            "gemini_reference_rank",
            "gemini_track_name",
            "gemini_artist_name",
            "gemini_album_name",
            "gemini_tag_list",
            "tag_overlap_count",
            "tag_overlap_terms",
        ]
    ].sort_values("gemini_reference_rank")
    st.dataframe(reference_display, width="stretch", hide_index=True, height=260)

    st.markdown("#### Five Gemini References Compared With The Target")
    fig = px.bar(
        turn_df.sort_values("gemini_reference_rank"),
        x="tag_overlap_count",
        y="gemini_label",
        orientation="h",
        text="tag_overlap_count",
        color="tag_overlap_count",
        hover_data=[
            "gemini_tag_list",
            "tag_overlap_terms",
            "ground_truth_track_name",
            "ground_truth_artist_name",
        ],
        title="Exact tag overlap with the ground-truth track",
    )
    fig.update_layout(
        height=420,
        xaxis_title="Number of shared tags",
        yaxis_title="Gemini-generated reference track",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch")

    display_columns = [
        "gemini_reference_rank",
        "gemini_track_name",
        "gemini_artist_name",
        "gemini_tag_list",
        "tag_overlap_count",
        "tag_overlap_terms",
        "ground_truth_track_name",
        "ground_truth_artist_name",
    ]
    st.dataframe(
        turn_df[display_columns].sort_values("gemini_reference_rank"),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Global Devset Tag Alignment")
    per_reference = (
        df.groupby("gemini_reference_rank", as_index=False)
        .agg(
            average_tag_overlap=("tag_overlap_count", "mean"),
            turns_with_overlap=("tag_overlap_count", lambda values: int((values > 0).sum())),
        )
    )
    global_fig = px.bar(
        per_reference,
        x="gemini_reference_rank",
        y="average_tag_overlap",
        text="average_tag_overlap",
        title="Average ground-truth tag overlap by Gemini reference position",
    )
    global_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    global_fig.update_layout(
        height=400,
        xaxis_title="Gemini reference rank",
        yaxis_title="Average shared tags",
    )
    st.plotly_chart(global_fig, width="stretch")
    st.dataframe(per_reference, width="stretch", hide_index=True)


def render_embedding_map(df, conversation_details):
    render_header(
        "Devset Embedding Map",
        "Place Gemini-generated reference tracks inside the catalog embedding space.",
    )
    if df.empty:
        st.warning("Embedding projection data was not found.")
        st.write(
            "Create it after the matching BERT cache exists. This is an offline step, so Streamlit stays lightweight."
        )
        st.code(
            "python create_gemini_embedding_projection.py "
            "--gemini_cache_dir ./cache/gemini_expansions_devset_first100",
            language="powershell",
        )
        return

    st.markdown(
        """
        **How to read this page.** The gray cloud is the real challenge catalog projected from retrieval embeddings
        into two PCA dimensions. The highlighted points show the selected turn's ground-truth track and the five
        Gemini-generated reference tracks. If the Gemini points are far from the ground truth, that is visual evidence
        of query drift.

        The final top-20 recommendations, Gemini references, and embedding projection are loaded from exported
        dashboard files. After running a new model, refresh those files with `refresh_streamlit_artifacts.py`
        so this page and the Devset Case Study page describe the same run.
        """
    )

    catalog_df = df[df["point_type"] == "catalog"].copy()
    gemini_df = df[df["point_type"] == "gemini_reference"].copy()
    ground_truth_df = df[df["point_type"] == "ground_truth"].copy()
    retrieved_df = df[df["point_type"] == "retrieved_track"].copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Catalog tracks", len(catalog_df))
    col2.metric("Gemini references", len(gemini_df))
    col3.metric("Ground-truth points", len(ground_truth_df))
    col4.metric("Retrieved highlights", len(retrieved_df))

    if gemini_df.empty:
        st.info("No Gemini reference points are available in the projection CSV.")
        return

    projection_type = (
        df["projection_retrieval_type"].dropna().astype(str).iloc[0]
        if "projection_retrieval_type" in df.columns and not df["projection_retrieval_type"].dropna().empty
        else "unknown"
    )
    projection_fields = (
        df["projection_corpus_types"].dropna().astype(str).iloc[0]
        if "projection_corpus_types" in df.columns and not df["projection_corpus_types"].dropna().empty
        else "unknown"
    )
    projection_label = {
        "bert": "BERT",
        "sentence_transformer": "MiniLM sentence-transformer",
    }.get(projection_type, projection_type)
    st.caption(f"Projection: {projection_label}. Corpus fields: {projection_fields}.")

    session_options = sorted(gemini_df["session_id"].dropna().unique())
    selected_session = st.sidebar.selectbox(
        "Session",
        session_options,
        key="embedding_session",
    )
    session_gemini = gemini_df[gemini_df["session_id"] == selected_session].copy()

    turn_options = sorted(session_gemini["turn_number_key"].dropna().unique(), key=lambda value: int(value))
    selected_turn = st.sidebar.selectbox(
        "Turn",
        turn_options,
        key="embedding_turn",
    )
    turn_gemini = session_gemini[session_gemini["turn_number_key"] == selected_turn].copy()
    turn_ground_truth = ground_truth_df[
        (ground_truth_df["session_id"] == selected_session)
        & (ground_truth_df["turn_number_key"] == selected_turn)
    ].copy()
    turn_retrieved = retrieved_df[
        (retrieved_df["session_id"] == selected_session)
        & (retrieved_df["turn_number_key"] == selected_turn)
    ].copy()

    turn_gemini = turn_gemini.sort_values("gemini_reference_rank_key").copy()
    reference_label_by_rank = {
        row.gemini_reference_rank_key: f"{row.gemini_reference_rank_key}. {row.track_name} - {row.artist_name}"
        for row in turn_gemini.itertuples(index=False)
    }
    reference_options = ["All"] + list(reference_label_by_rank.values())
    selected_reference = st.sidebar.selectbox(
        "Gemini reference",
        reference_options,
        key="embedding_reference",
    )
    visible_retrieved = turn_retrieved
    visible_gemini = turn_gemini
    if selected_reference != "All":
        selected_reference_rank = selected_reference.split(".", 1)[0]
        visible_retrieved = turn_retrieved[
            turn_retrieved["gemini_reference_rank_key"] == selected_reference_rank
        ]
        visible_gemini = turn_gemini[
            turn_gemini["gemini_reference_rank_key"] == selected_reference_rank
        ]

    reference_colors = {
        "1": "#c4462f",
        "2": "#2f80c4",
        "3": "#7b4fb3",
        "4": "#2f9e67",
        "5": "#d18a00",
    }
    if not turn_retrieved.empty:
        turn_retrieved["gemini_reference_track"] = turn_retrieved["gemini_reference_rank_key"].map(
            reference_label_by_rank
        )
    if not visible_retrieved.empty:
        visible_retrieved["gemini_reference_track"] = visible_retrieved["gemini_reference_rank_key"].map(
            reference_label_by_rank
        )

    selected_detail = conversation_details.get((selected_session, int(selected_turn)))
    final_recommendations = []
    if selected_detail is not None:
        catalog_by_track_id = catalog_df.set_index("track_id")
        for rank, track in enumerate(selected_detail.get("predicted_tracks", []), start=1):
            track_id = track.get("track_id")
            if track_id not in catalog_by_track_id.index:
                continue
            catalog_row = catalog_by_track_id.loc[track_id]
            final_recommendations.append(
                {
                    "rank": rank,
                    "track_id": track_id,
                    "track_name": track.get("track_name", catalog_row["track_name"]),
                    "artist_name": track.get("artist_name", catalog_row["artist_name"]),
                    "album_name": track.get("album_name", catalog_row["album_name"]),
                    "tag_list": catalog_row["tag_list"],
                    "pca_x": catalog_row["pca_x"],
                    "pca_y": catalog_row["pca_y"],
                }
            )
    final_recommendations_df = pd.DataFrame(final_recommendations)

    genre_options = ["All"] + sorted(catalog_df["broad_genre"].dropna().unique())
    selected_genre = st.sidebar.selectbox(
        "Catalog background",
        genre_options,
        key="embedding_genre",
    )
    visible_catalog = catalog_df
    if selected_genre != "All":
        visible_catalog = catalog_df[catalog_df["broad_genre"] == selected_genre]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=visible_catalog["pca_x"],
            y=visible_catalog["pca_y"],
            mode="markers",
            name="Catalog tracks",
            marker=dict(size=4, color="#c7cbd1", opacity=0.28),
            text=visible_catalog["track_name"] + " - " + visible_catalog["artist_name"],
            customdata=visible_catalog[["album_name", "broad_genre", "tag_list"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Album: %{customdata[0]}<br>"
                "Genre group: %{customdata[1]}<br>"
                "Tags: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    if not turn_ground_truth.empty:
        fig.add_trace(
            go.Scatter(
                x=turn_ground_truth["pca_x"],
                y=turn_ground_truth["pca_y"],
                mode="markers",
                name="Ground-truth track",
                marker=dict(size=18, color="#2364aa", symbol="star", line=dict(width=1, color="#ffffff")),
                text=turn_ground_truth["track_name"] + " - " + turn_ground_truth["artist_name"],
                customdata=turn_ground_truth[["album_name", "tag_list"]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Album: %{customdata[0]}<br>"
                    "Tags: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    if not final_recommendations_df.empty:
        fig.add_trace(
            go.Scatter(
                x=final_recommendations_df["pca_x"],
                y=final_recommendations_df["pca_y"],
                mode="markers+text",
                name="Final top-20 recommendations",
                marker=dict(
                    size=12,
                    color="#111827",
                    symbol="square",
                    line=dict(width=1, color="#ffffff"),
                    opacity=0.9,
                ),
                text=final_recommendations_df["rank"].astype(str),
                textposition="bottom center",
                customdata=final_recommendations_df[
                    ["rank", "track_name", "artist_name", "album_name", "tag_list"]
                ],
                hovertemplate=(
                    "<b>Final recommendation #%{customdata[0]}</b><br>"
                    "%{customdata[1]} - %{customdata[2]}<br>"
                    "Album: %{customdata[3]}<br>"
                    "Tags: %{customdata[4]}<extra></extra>"
                ),
            )
        )

    for reference_rank, reference_gemini in visible_gemini.groupby("gemini_reference_rank_key"):
        color = reference_colors.get(str(reference_rank), "#c4462f")
        reference_label = reference_label_by_rank.get(str(reference_rank), f"Gemini reference {reference_rank}")
        reference_retrieved = visible_retrieved[
            visible_retrieved["gemini_reference_rank_key"] == str(reference_rank)
        ]

        if not reference_retrieved.empty:
            fig.add_trace(
                go.Scatter(
                    x=reference_retrieved["pca_x"],
                    y=reference_retrieved["pca_y"],
                    mode="markers",
                    name=f"Retrieved from {reference_label}",
                    marker=dict(
                        size=10,
                        color=color,
                        symbol="circle-open",
                        line=dict(width=2),
                    ),
                    customdata=reference_retrieved[
                        [
                            "track_name",
                            "artist_name",
                            "album_name",
                            "gemini_reference_track",
                            "retrieved_rank_key",
                            "cosine_similarity",
                            "tag_list",
                        ]
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]} - %{customdata[1]}</b><br>"
                        "Album: %{customdata[2]}<br>"
                        "From Gemini: %{customdata[3]}<br>"
                        "Retrieved rank for that Gemini track: %{customdata[4]}<br>"
                        "Cosine similarity: %{customdata[5]:.4f}<br>"
                        "Tags: %{customdata[6]}<extra></extra>"
                    ),
                )
            )

        fig.add_trace(
            go.Scatter(
                x=reference_gemini["pca_x"],
                y=reference_gemini["pca_y"],
                mode="markers+text",
                name=reference_label,
                marker=dict(size=16, color=color, symbol="diamond", line=dict(width=1, color="#ffffff")),
                text=reference_gemini["gemini_reference_rank_key"],
                textposition="top center",
                customdata=reference_gemini[["track_name", "artist_name", "album_name", "tag_list"]],
                hovertemplate=(
                    "<b>Gemini reference %{text}</b><br>"
                    "%{customdata[0]} - %{customdata[1]}<br>"
                    "Album: %{customdata[2]}<br>"
                    "Tags: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"PCA map of {projection_label} metadata embeddings",
        height=660,
        xaxis_title="PCA component 1",
        yaxis_title="PCA component 2",
        legend_title="Gemini source",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Highlighted Points")
    summary_rows = []
    if not turn_ground_truth.empty:
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

    for row in turn_gemini.sort_values("gemini_reference_rank_key").itertuples(index=False):
        summary_rows.append(
            {
                "type": "Gemini reference",
                "rank": row.gemini_reference_rank_key,
                "track": row.track_name,
                "artist": row.artist_name,
                "album": row.album_name,
                "tags": row.tag_list,
            }
        )

    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    st.markdown("#### Final Top-20 Recommendations")
    st.caption(
        "These are the final model recommendations shown in the Devset Case Study page, projected onto the same PCA map."
    )
    if final_recommendations_df.empty:
        st.info("No final recommendation details are available for this selected turn.")
    else:
        final_table = final_recommendations_df[
            ["rank", "track_name", "artist_name", "album_name", "tag_list"]
        ].rename(
            columns={
                "track_name": "track",
                "artist_name": "artist",
                "album_name": "album",
                "tag_list": "tags",
            }
        )
        st.dataframe(final_table, width="stretch", hide_index=True, height=360)

    st.markdown("#### BERT Nearest Tracks For Each Gemini Reference")
    st.caption(
        "These are diagnostic nearest-neighbor tracks around each Gemini reference, not the final recommendation list."
    )
    if visible_retrieved.empty:
        st.info(
            "No retrieved-track highlights are available. Regenerate the projection CSV with the updated script."
        )
    else:
        retrieved_table = visible_retrieved[
            [
                "gemini_reference_rank_key",
                "gemini_reference_track",
                "retrieved_rank_key",
                "cosine_similarity",
                "track_name",
                "artist_name",
                "album_name",
                "tag_list",
            ]
        ].rename(
            columns={
                "gemini_reference_rank_key": "gemini_reference",
                "gemini_reference_track": "source_gemini_track",
                "retrieved_rank_key": "retrieved_rank",
                "track_name": "track",
                "artist_name": "artist",
                "album_name": "album",
                "tag_list": "tags",
            }
        )
        retrieved_table["_gemini_reference_sort"] = pd.to_numeric(
            retrieved_table["gemini_reference"],
            errors="coerce",
        )
        retrieved_table["_retrieved_rank_sort"] = pd.to_numeric(
            retrieved_table["retrieved_rank"],
            errors="coerce",
        )
        retrieved_table = retrieved_table.sort_values(
            ["_gemini_reference_sort", "_retrieved_rank_sort"]
        ).drop(columns=["_gemini_reference_sort", "_retrieved_rank_sort"])
        st.dataframe(
            retrieved_table,
            width="stretch",
            hide_index=True,
            height=360,
        )


def render_global_gemini(df):
    st.subheader("Global Blindset Similarity")
    if df.empty:
        st.warning("Gemini similarity table was not found.")
        return

    fig = px.histogram(
        df,
        x="cosine_similarity",
        nbins=40,
        title="Distribution of Gemini reference-track to catalog-track similarity",
    )
    fig.update_layout(
        xaxis_title="Cosine similarity",
        yaxis_title="Count",
        height=420,
    )
    st.plotly_chart(fig, width="stretch")

    pseudo_summary = (
        df.groupby(
            [
                "session_id",
                "turn_number",
                "gemini_pseudo_rank",
                "gemini_track_name",
                "gemini_artist_name",
            ],
            as_index=False,
        )
        .agg(
            avg_similarity=("cosine_similarity", "mean"),
            max_similarity=("cosine_similarity", "max"),
            avg_tag_overlap=("tag_overlap_count", "mean"),
        )
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Highest Average Similarity")
        st.dataframe(
            pseudo_summary.sort_values("avg_similarity", ascending=False).head(10),
            width="stretch",
            hide_index=True,
        )
    with col_b:
        st.markdown("#### Lowest Average Similarity")
        st.dataframe(
            pseudo_summary.sort_values("avg_similarity", ascending=True).head(10),
            width="stretch",
            hide_index=True,
        )


def render_references():
    st.subheader("Challenge References")
    for label, url in REFERENCE_LINKS.items():
        st.markdown(f"- [{label}]({url})")

    st.markdown("#### Current Working Hypothesis")
    st.markdown(
        """
        The strongest path for the TFM demo is to show that exact lexical signals matter strongly in this dataset:
        BM25 benefits from tag and metadata overlap, while dense semantic expansion can produce musically plausible
        but evaluation-wrong recommendations. Gemini should therefore be used as a restrained tag extractor for BM25.
        """
    )


def set_active_page(page):
    st.session_state.active_page = page


def main():
    gemini_df = load_gemini_table(CSV_PATH)
    devset_gemini_df = load_devset_ground_truth_table(DEVSET_GEMINI_PATH)
    devset_conversations = load_devset_conversations(DEVSET_CONVERSATION_PATH)
    embedding_projection_df = load_embedding_projection(EMBEDDING_PROJECTION_PATH)

    if "active_page" not in st.session_state:
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
            "Devset Case Study",
            width="stretch",
            type="primary" if st.session_state.active_page == "Devset Case Study" else "secondary",
            on_click=set_active_page,
            args=("Devset Case Study",),
        )
        st.button(
            "Devset Embedding Map",
            width="stretch",
            type="primary" if st.session_state.active_page == "Devset Embedding Map" else "secondary",
            on_click=set_active_page,
            args=("Devset Embedding Map",),
        )
        st.button(
            "Blindset Retrieval Explorer",
            width="stretch",
            type="primary" if st.session_state.active_page == "Blindset Retrieval Explorer" else "secondary",
            on_click=set_active_page,
            args=("Blindset Retrieval Explorer",),
        )

        page = st.session_state.active_page
        st.divider()
        if page == "Project Overview":
            st.caption("Start here for the challenge goal, data, and model families.")
        elif page == "Model Results":
            st.caption("Compare model performance and understand each metric.")
        elif page == "Devset Case Study":
            st.caption("Inspect conversations with known target tracks.")
        elif page == "Devset Embedding Map":
            st.caption("See Gemini references and target tracks inside the BERT embedding space.")
        else:
            st.caption("Explore Gemini references against nearby catalog tracks.")

    if page == "Project Overview":
        render_project_overview()
    elif page == "Blindset Retrieval Explorer":
        render_gemini_analysis(gemini_df)
        st.divider()
        render_global_gemini(gemini_df)
        st.divider()
        render_references()
    elif page == "Devset Case Study":
        render_devset_comparison(devset_gemini_df, devset_conversations)
    elif page == "Devset Embedding Map":
        render_embedding_map(embedding_projection_df, devset_conversations)
    else:
        render_model_results()


if __name__ == "__main__":
    main()
