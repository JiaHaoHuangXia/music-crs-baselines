import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "visualize_streamlit/gemini_to_catalog_similarity_table.csv"


# ============================================================
# HELPERS
# ============================================================

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


def shorten_text(text, max_len=120):
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    overlaps = df.apply(
        lambda row: tag_overlap(row["gemini_tag_list"], row["catalog_tag_list"]),
        axis=1
    )

    df["tag_overlap_count"] = [x[0] for x in overlaps]
    df["tag_overlap_terms"] = [x[1] for x in overlaps]

    df["gemini_label"] = (
        df["gemini_pseudo_rank"].astype(str)
        + ". "
        + df["gemini_track_name"].astype(str)
        + " — "
        + df["gemini_artist_name"].astype(str)
    )

    df["catalog_label"] = (
        df["catalog_similarity_rank"].astype(str)
        + ". "
        + df["catalog_track_name"].astype(str)
        + " — "
        + df["catalog_artist_name"].astype(str)
    )

    return df


df = load_data(CSV_PATH)


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Gemini + BERT Retrieval Analysis",
    layout="wide"
)

st.title("Gemini + BERT Retrieval Analysis")
st.caption(
    "Interactive dashboard for analyzing how Gemini pseudo-tracks map to real catalog tracks in BERT embedding space."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Filters")

session_options = sorted(df["session_id"].dropna().unique())
selected_session = st.sidebar.selectbox("Session ID", session_options)

session_df = df[df["session_id"] == selected_session].copy()

turn_options = sorted(session_df["turn_number"].dropna().unique())
selected_turn = st.sidebar.selectbox("Turn number", turn_options)

turn_df = session_df[session_df["turn_number"] == selected_turn].copy()

gemini_options = sorted(turn_df["gemini_label"].dropna().unique())
selected_gemini = st.sidebar.selectbox("Gemini pseudo-track", gemini_options)

selected_df = turn_df[turn_df["gemini_label"] == selected_gemini].copy()

min_similarity = st.sidebar.slider(
    "Minimum cosine similarity",
    float(df["cosine_similarity"].min()),
    float(df["cosine_similarity"].max()),
    float(df["cosine_similarity"].min()),
    step=0.001,
)

selected_df = selected_df[selected_df["cosine_similarity"] >= min_similarity]


# ============================================================
# SUMMARY METRICS
# ============================================================

st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total rows", len(df))
col2.metric("Sessions", df["session_id"].nunique())
col3.metric("Gemini pseudo-tracks", df[["session_id", "turn_number", "gemini_pseudo_rank"]].drop_duplicates().shape[0])
col4.metric("Average similarity", f"{df['cosine_similarity'].mean():.4f}")

st.divider()


# ============================================================
# SELECTED GEMINI PSEUDO-TRACK
# ============================================================

st.subheader("Selected Gemini pseudo-track")

first_row = selected_df.iloc[0]

col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### Gemini output")
    st.write(f"**Track:** {first_row['gemini_track_name']}")
    st.write(f"**Artist:** {first_row['gemini_artist_name']}")
    st.write(f"**Album:** {first_row['gemini_album_name']}")
    st.write(f"**Release date:** {first_row['gemini_release_date']}")

with col_right:
    st.markdown("### Gemini tags")
    st.info(first_row["gemini_tag_list"])


# ============================================================
# BAR CHART: TOP CATALOG MATCHES
# ============================================================

st.subheader("Top catalog matches for this Gemini pseudo-track")

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
    height=550,
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TABLE: TOP MATCHES
# ============================================================

st.subheader("Detailed retrieval table")

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
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# SESSION-LEVEL VIEW
# ============================================================

st.divider()
st.subheader("Session-level comparison: all 5 Gemini pseudo-tracks")

session_summary = (
    turn_df
    .groupby(["gemini_pseudo_rank", "gemini_track_name", "gemini_artist_name"], as_index=False)
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
    title="Do the 5 Gemini pseudo-tracks point to similar catalog regions?",
)

fig2.update_layout(
    xaxis_title="Average cosine similarity to top catalog matches",
    yaxis_title="Average tag overlap",
    height=500,
)

st.plotly_chart(fig2, use_container_width=True)

st.dataframe(
    session_summary.sort_values("gemini_pseudo_rank"),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# GLOBAL ANALYSIS
# ============================================================

st.divider()
st.subheader("Global similarity distribution")

fig3 = px.histogram(
    df,
    x="cosine_similarity",
    nbins=40,
    title="Distribution of Gemini pseudo-track → catalog track similarity",
)

fig3.update_layout(
    xaxis_title="Cosine similarity",
    yaxis_title="Count",
    height=400,
)

st.plotly_chart(fig3, use_container_width=True)


st.subheader("Most and least aligned Gemini pseudo-tracks")

pseudo_summary = (
    df
    .groupby(["session_id", "turn_number", "gemini_pseudo_rank", "gemini_track_name", "gemini_artist_name"], as_index=False)
    .agg(
        avg_similarity=("cosine_similarity", "mean"),
        max_similarity=("cosine_similarity", "max"),
        avg_tag_overlap=("tag_overlap_count", "mean"),
    )
)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Highest average similarity")
    st.dataframe(
        pseudo_summary.sort_values("avg_similarity", ascending=False).head(10),
        use_container_width=True,
        hide_index=True,
    )

with col_b:
    st.markdown("### Lowest average similarity")
    st.dataframe(
        pseudo_summary.sort_values("avg_similarity", ascending=True).head(10),
        use_container_width=True,
        hide_index=True,
    )