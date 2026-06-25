from pathlib import Path
import json
import html
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
TFM_ROOT = ROOT.parent
DEVSET_CONVERSATION_PATH = ROOT / "visualize_streamlit" / "devset_conversation_details.json"
EMBEDDING_PROJECTION_PATH = ROOT / "visualize_streamlit" / "gemini_embedding_projection.csv"
BM25_EXPLANATION_PATH = ROOT / "visualize_streamlit" / "bm25_explanation_table.csv"
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
        "experiment": "BM25 + Gemini controlled keywords + query-type router",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0325,
        "ndcg@10": 0.11541612893572054,
        "ndcg@20": 0.15073852033934126,
        "catalog_diversity": 0.054407172144207684,
        "lexical_diversity": 0.0,
        "total_catalog_size": 47071,
        "source": "Local evaluator",
        "note": "Best local first-50 run. Gemini extracts controlled BM25 terms, then a conservative query-type router reranks only clear artist/title/album/decade/negative-intent turns.",
    },
    {
        "experiment": "BM25 + Gemini controlled keywords, artist/title weighted block x2",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0325,
        "ndcg@10": 0.11414415511650364,
        "ndcg@20": 0.14930209229441888,
        "catalog_diversity": 0.05126298570244949,
        "lexical_diversity": 0.5262852351157841,
        "total_catalog_size": 47071,
        "source": "Local evaluator",
        "note": "Previous best local BM25 run. The final query repeats Gemini controlled keywords twice and gives artist/title fields higher weight.",
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
        "source": "Local evaluator, baseline-models-first50 branch",
        "note": "Upstream baseline code on the first 50 devset conversations, using raw conversation text and catalog fields track, artist, album, tag_list, and release_date.",
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
        "source": "Local evaluator, baseline-models-first50 branch",
        "note": "Upstream baseline code on the first 50 devset conversations, using raw conversation text and catalog fields track, artist, album, and release_date.",
    },
    {
        "experiment": "MiniLM + artist profile + decade + BM25 hybrid reranker",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0325,
        "ndcg@10": 0.08925567039128843,
        "ndcg@20": 0.10458119860417942,
        "catalog_diversity": 0.06827983259331648,
        "lexical_diversity": 0.0,
        "total_catalog_size": 47071,
        "source": "Local evaluator",
        "note": "Current best local embedding/reranking run. Adds BM25 lexical candidates over title, artist, album, and tags before structured reranking.",
    },
    {
        "experiment": "MiniLM + artist profile + decade + structured reranker",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0400,
        "ndcg@10": 0.08130748412086836,
        "ndcg@20": 0.09812400600576054,
        "catalog_diversity": 0.06694142890527076,
        "lexical_diversity": 0.0,
        "total_catalog_size": 47071,
        "source": "Local evaluator",
        "note": "Previous best local enriched embedding run before adding BM25 lexical candidate generation.",
    },
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
        "experiment": "Clean baseline BERT + raw conversation",
        "split": "Devset first 50 conversations",
        "turns_evaluated": 400,
        "ndcg@1": 0.0025,
        "ndcg@10": 0.011400438250866413,
        "ndcg@20": 0.013322326781414569,
        "catalog_diversity": 0.014467506532684667,
        "lexical_diversity": 0.4384776745579862,
        "total_catalog_size": 47071,
        "source": "Local evaluator, baseline-models-first50 branch",
        "note": "Upstream BERT baseline code on the first 50 devset conversations, using raw conversation text.",
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
        - Use **Devset Embedding Map** to inspect the embedding-based Gemini experiments and example turns with known ground truth.
        - Use **BM25 Explanation** to see how the current BM25 + Gemini keyword model builds and matches its query.
        """
    )

    st.divider()
    render_references()


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
        - The current best local devset system is BM25 with Gemini controlled keywords and a conservative query-type router.
        - The clean upstream baseline comparison shows that BM25 is much stronger than BERT for exact target-track recovery.
        - Adding tag_list to the clean BM25 baseline improves nDCG@20, but its noisy terms can slightly disturb top-10 ranking.
        - The biggest gain came from using Gemini as a controlled lexical query extractor instead of a free-form similar-song generator.
        - Dense embedding variants retrieve semantically plausible neighborhoods, but they are weak for this evaluation because nDCG@20 rewards exact track recovery.
        - The most useful thesis comparison is clean BM25 vs clean BM25 + tag_list vs BM25 + Gemini controlled keywords, because it shows how lexical evidence improves exact retrieval.
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
        "Inspect the lexical evidence behind BM25 retrieval and the query-type router.",
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
        catalog tracks whose metadata contains those terms, and the query-type router only changes the order for
        clear artist/title/album/decade/negative-intent cases.
        """
    )

    turn_count = df[["session_id", "turn_number"]].drop_duplicates().shape[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Explanation rows", len(df))
    col2.metric("Evaluated turns", turn_count)
    col3.metric("Avg matched terms", format_metric(df["matched_term_count"].mean(), digits=2))
    col4.metric("Ground-truth rows", int(df["is_ground_truth"].sum()))

    retrieval_summary, turn_metrics = build_top20_retrieval_summary(df)
    if retrieval_summary:
        st.markdown("#### Retrieval Quality Summary")
        st.caption(
            "Overall performance for this exported devset run: whether the correct track appears in the top 20, "
            "and how high it is ranked when it appears."
        )
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("nDCG@20", f"{retrieval_summary['ndcg_at_20']:.4f}")
        metric_col2.metric("Hit rate@20", f"{retrieval_summary['hit_rate_at_20']:.1%}")
        metric_col3.metric("Target found", f"{retrieval_summary['found']}/{retrieval_summary['turns']}")
        metric_col4.metric("Target missed", f"{retrieval_summary['missed']}")

        rank_order = ["1", "2-5", "6-10", "11-20", "Not in top 20"]
        rank_distribution = (
            turn_metrics["rank_bucket"]
            .value_counts()
            .reindex(rank_order, fill_value=0)
            .rename_axis("ground_truth_rank")
            .reset_index(name="turns")
        )
        rank_distribution["share"] = rank_distribution["turns"] / retrieval_summary["turns"]
        fig = px.bar(
            rank_distribution,
            x="ground_truth_rank",
            y="turns",
            text=rank_distribution["share"].map(lambda value: f"{value:.1%}"),
            title="Ground-truth rank distribution",
        )
        fig.update_traces(marker_color="#2f80c4", textposition="outside")
        fig.update_layout(
            height=360,
            xaxis_title="Where the correct track appeared",
            yaxis_title="Conversation turns",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

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
        "The final ranked tracks for this turn. The match columns show which Gemini/BM25 terms were found in each "
        "track's metadata, and the ground_truth column marks the correct answer when it appears."
    )
    display_table = turn_df[
        [
            "rank",
            "track_name",
            "artist_name",
            "album_name",
            "release_decade",
            "matched_term_count",
            "matched_fields",
            "matched_terms",
            "is_ground_truth",
        ]
    ].rename(
        columns={
            "track_name": "track",
            "artist_name": "artist",
            "album_name": "album",
            "release_decade": "decade",
            "matched_term_count": "matched_terms_n",
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


def build_oracle_rows(ground_truth_df, retrieved_df, k_values):
    if ground_truth_df.empty or retrieved_df.empty:
        return pd.DataFrame()

    retrieved = retrieved_df.copy()
    retrieved["retrieved_rank_number"] = pd.to_numeric(
        retrieved["retrieved_rank"],
        errors="coerce",
    )
    ground_truth = ground_truth_df[
        ["session_id", "turn_number_key", "track_id", "track_name", "artist_name"]
    ].rename(
        columns={
            "track_id": "ground_truth_track_id",
            "track_name": "ground_truth_track",
            "artist_name": "ground_truth_artist",
        }
    )
    joined = retrieved.merge(
        ground_truth,
        on=["session_id", "turn_number_key"],
        how="inner",
    )
    joined["is_ground_truth_neighbor"] = (
        joined["track_id"] == joined["ground_truth_track_id"]
    )

    rows = []
    total_turns = ground_truth[["session_id", "turn_number_key"]].drop_duplicates().shape[0]
    for k in k_values:
        within_k = joined[joined["retrieved_rank_number"] <= k]
        hit_turns = (
            within_k[within_k["is_ground_truth_neighbor"]]
            [["session_id", "turn_number_key"]]
            .drop_duplicates()
            .shape[0]
        )
        rows.append(
            {
                "k": k,
                "hit_turns": hit_turns,
                "miss_turns": total_turns - hit_turns,
                "total_turns": total_turns,
                "hit_rate": hit_turns / total_turns if total_turns else 0.0,
            }
        )

    return pd.DataFrame(rows)


def normalize_match_value(value):
    return str(value or "").strip().lower()


def build_miss_breakdown_rows(ground_truth_df, gemini_df, retrieved_df, k):
    if ground_truth_df.empty or gemini_df.empty or retrieved_df.empty:
        return pd.DataFrame()

    retrieved = retrieved_df.copy()
    retrieved["retrieved_rank_number"] = pd.to_numeric(
        retrieved["retrieved_rank"],
        errors="coerce",
    )
    retrieved = retrieved[retrieved["retrieved_rank_number"] <= k]

    ground_truth = ground_truth_df[
        ["session_id", "turn_number_key", "track_id", "track_name", "artist_name", "album_name"]
    ].rename(
        columns={
            "track_id": "ground_truth_track_id",
            "track_name": "ground_truth_track",
            "artist_name": "ground_truth_artist",
            "album_name": "ground_truth_album",
        }
    )
    retrieved_with_truth = retrieved.merge(
        ground_truth[["session_id", "turn_number_key", "ground_truth_track_id"]],
        on=["session_id", "turn_number_key"],
        how="inner",
    )
    found_turns = set(
        retrieved_with_truth[retrieved_with_truth["track_id"] == retrieved_with_truth["ground_truth_track_id"]]
        .apply(lambda row: (row["session_id"], row["turn_number_key"]), axis=1)
        .tolist()
    )

    gemini = gemini_df[
        ["session_id", "turn_number_key", "track_name", "artist_name", "album_name"]
    ].rename(
        columns={
            "track_name": "gemini_track",
            "artist_name": "gemini_artist",
            "album_name": "gemini_album",
        }
    )
    joined = ground_truth.merge(gemini, on=["session_id", "turn_number_key"], how="left")
    joined["turn_key"] = list(zip(joined["session_id"], joined["turn_number_key"]))
    misses = joined[~joined["turn_key"].isin(found_turns)].copy()

    if misses.empty:
        return pd.DataFrame()

    misses["same_track_title"] = (
        misses["ground_truth_track"].map(normalize_match_value)
        == misses["gemini_track"].map(normalize_match_value)
    )
    misses["same_artist"] = (
        misses["ground_truth_artist"].map(normalize_match_value)
        == misses["gemini_artist"].map(normalize_match_value)
    )
    misses["same_album"] = (
        misses["ground_truth_album"].map(normalize_match_value)
        == misses["gemini_album"].map(normalize_match_value)
    )

    rows = []
    for (_, _), group in misses.groupby(["session_id", "turn_number_key"]):
        exact_title_and_artist = bool((group["same_track_title"] & group["same_artist"]).any())
        same_artist_and_album = bool((group["same_artist"] & group["same_album"]).any())
        same_artist = bool(group["same_artist"].any())
        same_album = bool(group["same_album"].any())

        if exact_title_and_artist:
            category = "Gemini named the target, but retrieval still missed it"
        elif same_artist_and_album:
            category = "Same artist and album"
        elif same_artist:
            category = "Same artist only"
        elif same_album:
            category = "Same album only"
        else:
            category = "Different artist and album"

        rows.append({"miss_reason": category})

    order = [
        "Gemini named the target, but retrieval still missed it",
        "Same artist and album",
        "Same artist only",
        "Same album only",
        "Different artist and album",
    ]
    breakdown = (
        pd.DataFrame(rows)
        .value_counts("miss_reason")
        .rename("turns")
        .reset_index()
    )
    breakdown["miss_reason"] = pd.Categorical(
        breakdown["miss_reason"],
        categories=order,
        ordered=True,
    )
    breakdown = breakdown.sort_values("miss_reason")
    breakdown["share_of_misses"] = breakdown["turns"] / breakdown["turns"].sum()
    return breakdown


def render_oracle_check(ground_truth_df, gemini_df, retrieved_df, selected_session, selected_turn):
    st.markdown("#### Can Gemini's Reference Songs Find The Correct Track?")
    st.markdown(
        """
        For each conversation turn, Gemini generated 5 reference songs. For each reference song, we retrieved
        the closest catalog tracks. This diagnostic checks whether the true target track appeared anywhere in
        those candidates. If it did not appear here, a later ranking or fusion step cannot recommend it.
        """
    )

    if ground_truth_df.empty or retrieved_df.empty:
        st.info("Oracle data is not available. Regenerate the embedding projection with retrieved-track highlights.")
        return

    max_rank = int(pd.to_numeric(retrieved_df["retrieved_rank"], errors="coerce").max())
    candidate_k_values = [1, 5, 10, 20, 50, 100, 200]
    k_values = [k for k in candidate_k_values if k <= max_rank]
    if max_rank not in k_values:
        k_values.append(max_rank)
    k_values = sorted(set(k_values))

    oracle_df = build_oracle_rows(ground_truth_df, retrieved_df, k_values)
    if oracle_df.empty:
        st.info("Oracle data could not be calculated from the current projection file.")
        return

    latest_k = int(oracle_df["k"].max())
    latest_row = oracle_df[oracle_df["k"] == latest_k].iloc[0]
    hit_turns = int(latest_row["hit_turns"])
    total_turns = int(latest_row["total_turns"])
    miss_turns = int(latest_row["miss_turns"])
    hit_rate = float(latest_row["hit_rate"])
    miss_rate = 1.0 - hit_rate

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Evaluated turns", f"{total_turns}")
    metric_col2.metric("Target found", f"{hit_turns}")
    metric_col3.metric("Target missed", f"{miss_turns}")
    metric_col4.metric("Target reachable rate", f"{hit_rate:.1%}")

    found_missed = pd.DataFrame(
        [
            {"result": "Found target", "turns": hit_turns, "share": hit_rate},
            {"result": "Missed target", "turns": miss_turns, "share": miss_rate},
        ]
    )
    fig = go.Figure()
    for row in found_missed.itertuples(index=False):
        color = "#2f9e67" if row.result == "Found target" else "#d9dee7"
        fig.add_trace(
            go.Bar(
                y=["Gemini reference retrieval"],
                x=[row.share],
                name=row.result,
                orientation="h",
                marker=dict(color=color),
                customdata=[[row.turns, row.share]],
                text=[f"{row.result}: {row.share:.1%}"],
                textposition="inside",
                hovertemplate=(
                    f"<b>{row.result}</b><br>"
                    "Turns: %{customdata[0]}<br>"
                    "Share: %{customdata[1]:.1%}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="stack",
        height=220,
        xaxis_title=f"Share of turns after checking 5 x top {latest_k} nearest tracks",
        yaxis_title="",
        yaxis_tickformat=".0%",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        legend_title="Result",
        margin=dict(l=10, r=10, t=20, b=40),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Interpretation: with the current exported data, Gemini's 5 reference songs found the correct track "
        f"for {hit_turns} of {total_turns} turns after checking the top {latest_k} nearest catalog matches "
        "for each reference song. This is an upper-limit diagnostic, not the final recommender score."
    )

    st.markdown("##### Why Did The Missed Turns Fail?")
    st.caption(
        "This breakdown uses only the metadata fields used by the latest embedding model: track name, artist name, "
        "and album name."
    )
    miss_breakdown = build_miss_breakdown_rows(ground_truth_df, gemini_df, retrieved_df, latest_k)
    if miss_breakdown.empty:
        st.info("There are no missed turns to break down in the current exported data.")
    else:
        st.dataframe(
            miss_breakdown.assign(
                share_of_misses=miss_breakdown["share_of_misses"].map(lambda value: f"{value:.1%}")
            ),
            width="stretch",
            hide_index=True,
        )

    turn_ground_truth = ground_truth_df[
        (ground_truth_df["session_id"] == selected_session)
        & (ground_truth_df["turn_number_key"] == selected_turn)
    ].copy()
    turn_retrieved = retrieved_df[
        (retrieved_df["session_id"] == selected_session)
        & (retrieved_df["turn_number_key"] == selected_turn)
    ].copy()

    if turn_ground_truth.empty or turn_retrieved.empty:
        return

    gt_row = turn_ground_truth.iloc[0]
    gt_track_id = gt_row["track_id"]
    turn_retrieved["retrieved_rank_number"] = pd.to_numeric(
        turn_retrieved["retrieved_rank"],
        errors="coerce",
    )
    hits = turn_retrieved[turn_retrieved["track_id"] == gt_track_id].copy()

    if hits.empty:
        st.warning(
            "For this selected turn, the ground-truth track does not appear in the exported nearest neighbors "
            "of any Gemini reference."
        )
    else:
        best_hit = hits.sort_values("retrieved_rank_number").iloc[0]
        st.success(
            "For this selected turn, Gemini can reach the ground-truth track: "
            f"{gt_row['track_name']} by {gt_row['artist_name']} appears at nearest-neighbor rank "
            f"{int(best_hit['retrieved_rank_number'])} from Gemini reference {best_hit['gemini_reference_rank_key']}."
        )

    reference_summary = (
        turn_retrieved.assign(is_ground_truth=turn_retrieved["track_id"] == gt_track_id)
        .groupby("gemini_reference_rank_key", as_index=False)
        .agg(
            best_ground_truth_rank=(
                "retrieved_rank_number",
                lambda values: int(values.min()) if len(values) else None,
            ),
            contains_ground_truth=("is_ground_truth", "max"),
        )
    )
    reference_summary = reference_summary.rename(
        columns={
            "gemini_reference_rank_key": "gemini_reference",
            "contains_ground_truth": "target_found",
        }
    )
    if not hits.empty:
        hit_rank_by_reference = hits.groupby("gemini_reference_rank_key")["retrieved_rank_number"].min()
        reference_summary["best_ground_truth_rank"] = reference_summary["gemini_reference"].map(
            hit_rank_by_reference
        )
    else:
        reference_summary["best_ground_truth_rank"] = None
    rank_values = pd.to_numeric(
        reference_summary["best_ground_truth_rank"],
        errors="coerce",
    )
    reference_summary["best_ground_truth_rank"] = (
        rank_values.astype("Int64").astype(str).replace("<NA>", "not found")
    )
    st.dataframe(reference_summary, width="stretch", hide_index=True)


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
        into two UMAP dimensions. The highlighted points show the selected turn's ground-truth track and the five
        Gemini-generated reference tracks. If the Gemini points are far from the ground truth, that is visual evidence
        of query drift.

        UMAP is used because this page focuses on local neighborhoods: whether Gemini references, retrieved tracks,
        final recommendations, and ground-truth tracks appear near each other. Retrieval is still computed in the
        original embedding space, not directly from this 2D map.

        The final top-20 recommendations, Gemini references, and embedding projection are loaded from exported
        dashboard files. After running a new model, refresh those files with `refresh_streamlit_artifacts.py`
        so this page describes the latest exported embedding run.
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
    projection_method = (
        df["projection_method"].dropna().astype(str).iloc[0]
        if "projection_method" in df.columns and not df["projection_method"].dropna().empty
        else "umap" if {"umap_x", "umap_y"}.issubset(df.columns) else "pca"
    )
    x_coord = "umap_x" if "umap_x" in df.columns else "pca_x"
    y_coord = "umap_y" if "umap_y" in df.columns else "pca_y"
    projection_method_label = projection_method.upper()
    st.caption(
        f"Projection: {projection_method_label} using {projection_label}. Corpus fields: {projection_fields}."
    )

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

    render_oracle_check(ground_truth_df, gemini_df, retrieved_df, selected_session, selected_turn)

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
                    x_coord: catalog_row[x_coord],
                    y_coord: catalog_row[y_coord],
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
        go.Scatter(
            x=visible_catalog[x_coord],
            y=visible_catalog[y_coord],
            mode="markers",
            name="Catalog tracks",
            marker=dict(size=3, color="#c7cbd1", opacity=0.16),
            text=visible_catalog["track_name"] + " - " + visible_catalog["artist_name"],
            customdata=visible_catalog[["album_name", "broad_genre"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Album: %{customdata[0]}<br>"
                "Genre group: %{customdata[1]}<extra></extra>"
            ),
        )
    )

    if not final_recommendations_df.empty:
        fig.add_trace(
            go.Scatter(
                x=final_recommendations_df[x_coord],
                y=final_recommendations_df[y_coord],
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
                    ["rank", "track_name", "artist_name", "album_name"]
                ],
                hovertemplate=(
                    "<b>Final recommendation #%{customdata[0]}</b><br>"
                    "%{customdata[1]} - %{customdata[2]}<br>"
                    "Album: %{customdata[3]}<extra></extra>"
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
                    x=reference_retrieved[x_coord],
                    y=reference_retrieved[y_coord],
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
                        ]
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]} - %{customdata[1]}</b><br>"
                        "Album: %{customdata[2]}<br>"
                        "From Gemini: %{customdata[3]}<br>"
                        "Retrieved rank for that Gemini track: %{customdata[4]}<br>"
                        "Cosine similarity: %{customdata[5]:.4f}<extra></extra>"
                    ),
                )
            )

        fig.add_trace(
            go.Scatter(
                x=reference_gemini[x_coord],
                y=reference_gemini[y_coord],
                mode="markers+text",
                name=reference_label,
                marker=dict(size=16, color=color, symbol="diamond", line=dict(width=1, color="#ffffff")),
                text=reference_gemini["gemini_reference_rank_key"],
                textposition="top center",
                customdata=reference_gemini[["track_name", "artist_name", "album_name"]],
                hovertemplate=(
                    "<b>Gemini reference %{text}</b><br>"
                    "%{customdata[0]} - %{customdata[1]}<br>"
                    "Album: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    if not turn_ground_truth.empty:
        fig.add_trace(
            go.Scatter(
                x=turn_ground_truth[x_coord],
                y=turn_ground_truth[y_coord],
                mode="markers",
                name="Ground-truth track",
                marker=dict(size=20, color="#2364aa", symbol="star", line=dict(width=2, color="#ffffff")),
                text=turn_ground_truth["track_name"] + " - " + turn_ground_truth["artist_name"],
                customdata=turn_ground_truth[["album_name"]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Album: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"{projection_method_label} map of {projection_label} metadata embeddings",
        height=660,
        xaxis_title=f"{projection_method_label} dimension 1",
        yaxis_title=f"{projection_method_label} dimension 2",
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
        f"These are the final model recommendations for the selected turn, projected onto the same {projection_method_label} map."
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

    st.markdown(f"#### {projection_label} Nearest Tracks For Each Gemini Reference")
    st.caption(
        "Each table shows the catalog tracks retrieved from one Gemini-generated reference song before fusion."
    )
    if visible_retrieved.empty:
        st.info(
            "No retrieved-track highlights are available. Regenerate the projection CSV with the updated script."
        )
    else:
        target_track_id = None
        if not turn_ground_truth.empty:
            target_track_id = turn_ground_truth.iloc[0]["track_id"]

        for reference_rank, reference_rows in visible_retrieved.groupby("gemini_reference_rank_key"):
            reference_rows = reference_rows.copy()
            reference_rows["_retrieved_rank_sort"] = pd.to_numeric(
                reference_rows["retrieved_rank_key"],
                errors="coerce",
            )
            reference_rows = reference_rows.sort_values("_retrieved_rank_sort")
            reference_label = reference_label_by_rank.get(
                str(reference_rank),
                f"{reference_rank}. Gemini reference",
            )
            found_rows = reference_rows[reference_rows["track_id"] == target_track_id]
            if target_track_id is not None and not found_rows.empty:
                found_rank = int(found_rows["_retrieved_rank_sort"].min())
                status = f"target found at rank {found_rank}"
            else:
                status = "target not found"

            with st.expander(f"{reference_label} - {status}", expanded=(selected_reference != "All")):
                table = reference_rows[
                    [
                        "retrieved_rank_key",
                        "cosine_similarity",
                        "track_name",
                        "artist_name",
                        "album_name",
                        "tag_list",
                    ]
                ].rename(
                    columns={
                        "retrieved_rank_key": "retrieved_rank",
                        "track_name": "track",
                        "artist_name": "artist",
                        "album_name": "album",
                        "tag_list": "tags",
                    }
                )
                st.dataframe(
                    table,
                    width="stretch",
                    hide_index=True,
                    height=min(420, 84 + 36 * len(table)),
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
    valid_pages = {
        "Project Overview",
        "Model Results",
        "Devset Embedding Map",
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
            "Devset Embedding Map",
            width="stretch",
            type="primary" if st.session_state.active_page == "Devset Embedding Map" else "secondary",
            on_click=set_active_page,
            args=("Devset Embedding Map",),
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
        elif page == "Devset Embedding Map":
            st.caption("See Gemini references and target tracks inside the BERT embedding space.")
        elif page == "BM25 Explanation":
            st.caption("Inspect controlled BM25 query terms, matches, and router evidence.")

    if page == "Project Overview":
        render_project_overview()
    elif page == "Devset Embedding Map":
        embedding_projection_df = load_embedding_projection(EMBEDDING_PROJECTION_PATH)
        devset_conversations = load_devset_conversations(DEVSET_CONVERSATION_PATH)
        render_embedding_map(embedding_projection_df, devset_conversations)
    elif page == "BM25 Explanation":
        bm25_explanation_df = load_bm25_explanations(BM25_EXPLANATION_PATH)
        devset_conversations = load_devset_conversations(DEVSET_CONVERSATION_PATH)
        render_bm25_explanation(bm25_explanation_df, devset_conversations)
    else:
        render_model_results()


if __name__ == "__main__":
    main()
