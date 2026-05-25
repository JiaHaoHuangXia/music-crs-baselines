from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
TFM_ROOT = ROOT.parent
CSV_PATH = ROOT / "visualize_streamlit" / "gemini_to_catalog_similarity_table.csv"
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


def render_model_results():
    render_header(
        "Model Results",
        "Blind-A Codabench evaluation results for the model variants tested in this TFM.",
    )
    manual_df = pd.DataFrame(MANUAL_RESULTS).sort_values(
        "composite_score",
        ascending=False,
    )
    metric_grid(manual_df)

    st.markdown("#### Blind-A Codabench Scores")
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

    st.markdown("#### Interpretation")
    st.markdown(
        """
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
        "Gemini Embedding Explorer",
        "Inspect how Gemini-generated reference tracks map to real catalog tracks in BERT embedding space.",
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
        **What was previously called a pseudo-track?** Gemini generated five example songs from a conversation
        to act as search references. They are not evaluation answers and may not even exist in the challenge
        catalog. Selecting a reference below changes the real catalog neighbors shown in the chart and table.
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


def render_global_gemini(df):
    st.subheader("Global Gemini Similarity")
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

    if "active_page" not in st.session_state:
        st.session_state.active_page = "Gemini Embeddings"

    with st.sidebar:
        st.header("Navigation")
        st.button(
            "Gemini Embeddings",
            width="stretch",
            type="primary" if st.session_state.active_page == "Gemini Embeddings" else "secondary",
            on_click=set_active_page,
            args=("Gemini Embeddings",),
        )
        st.button(
            "Model Results",
            width="stretch",
            type="primary" if st.session_state.active_page == "Model Results" else "secondary",
            on_click=set_active_page,
            args=("Model Results",),
        )

        page = st.session_state.active_page
        st.divider()
        if page == "Gemini Embeddings":
            st.caption("Filters update the catalog matches shown on this page.")
        else:
            st.caption("Codabench Blind-A scores for your tested models.")

    if page == "Gemini Embeddings":
        render_gemini_analysis(gemini_df)
        st.divider()
        render_global_gemini(gemini_df)
        st.divider()
        render_references()
    else:
        render_model_results()


if __name__ == "__main__":
    main()
