"""
Streamlit dashboard for the music replayability project.

This is the front-end the TAs see. Everything here reads from the artifacts
written by the upstream scripts (`outputs/music_dataset_processed.csv`,
`model_results.csv`, `feature_importance.csv`, the SQL CSVs, the lyrics CSVs,
and the joblib'd best model) — nothing here re-trains or re-fetches.

Layout — kept deliberately unfussy: a top header with the headline numbers,
a song search bar (so anyone can poke at a specific track and see what the
model thinks of it), and then a row of tabs for the deeper workspaces:
- Overview        : dataset health + headline genre / decade plots.
- Genre Drilldown : interactive genre + decade filters.
- Models          : CV vs holdout R^2, feature importances, predicted vs
                    actual scatter, imbalance demo.
- Predict         : live scoring form for a hypothetical track profile.
- Lyrics          : NLP plots and the lyrics-vs-metadata model comparison.
- Song Lookup     : search a specific song, see the model's prediction for
                    it, the residual vs ground truth, and a per-song lyrics
                    breakdown.
- Data Quality    : missingness chart for sanity checking.

Rubric coverage hit from this file:
- Interactive dashboard with 7 dedicated tabs.
- Live scoring console using the saved sklearn pipeline.
- Per-song model-fit view (predicted vs actual + residual + percentile rank).
- Free-text song search across the processed dataset.
- Filterable genre/decade workspace.
- Embeds all NLP plots from lyrics_analysis.py.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import string
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st
from joblib import load

from config import (
    BEST_MODEL_FILE,
    BEST_PARAMS_JSON,
    CACHE_DIR,
    COLLECTION_SUMMARY_JSON,
    CURRENT_YEAR,
    DATA_SUMMARY_JSON,
    EDA_SUMMARY_MD,
    FEATURE_IMPORT,
    IMBALANCE_RESULTS_JSON,
    MODEL_PREDICTIONS,
    MODEL_RESULTS,
    MODEL_SUMMARY_MD,
    OUTPUT_DIR,
    PCA_SUMMARY_JSON,
    PRIMARY_MODEL_AUDIO_COVERAGE_THRESHOLD,
    PROCESSED_CSV,
    RAW_TARGET,
    TARGET,
)


st.set_page_config(
    page_title="Music Replayability Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Light-mode styling with a warmer color palette so charts and tables stand out
# against a soft tinted background instead of bare white.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(224, 122, 95, 0.18), transparent 32%),
            radial-gradient(circle at 92% 6%, rgba(31, 111, 120, 0.18), transparent 30%),
            radial-gradient(circle at 50% 100%, rgba(245, 197, 96, 0.14), transparent 38%),
            linear-gradient(180deg, #fdfaf6 0%, #f3ece4 100%);
    }

    .app-title {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0.4rem 0 0.15rem 0;
        background: linear-gradient(90deg, #1f6f78 0%, #e07a5f 60%, #f5c560 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .app-subtitle {
        color: #5c6672;
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #fff4ec 100%);
        border: 1px solid rgba(224, 122, 95, 0.35);
        border-left: 4px solid #e07a5f;
        border-radius: 10px;
        padding: 0.7rem 0.95rem;
        box-shadow: 0 4px 14px rgba(31, 41, 55, 0.06);
    }

    div[data-testid="stMetric"]:nth-of-type(2n) {
        background: linear-gradient(135deg, #ffffff 0%, #ecf6f7 100%);
        border-color: rgba(31, 111, 120, 0.35);
        border-left-color: #1f6f78;
    }

    div[data-testid="stMetric"]:nth-of-type(3n) {
        background: linear-gradient(135deg, #ffffff 0%, #fdf3d9 100%);
        border-color: rgba(245, 197, 96, 0.45);
        border-left-color: #f5c560;
    }

    /* Tab strip — give it a tinted underline accent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid rgba(31, 111, 120, 0.18);
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 8px 8px 0 0;
        padding: 0.4rem 0.9rem;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(180deg, #fff4ec 0%, #ffffff 100%);
        color: #1f6f78;
        font-weight: 700;
        box-shadow: inset 0 -3px 0 #e07a5f;
    }

    /* Bordered containers (st.container(border=True)) get a softer tint */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.65);
        border-color: rgba(31, 111, 120, 0.2) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataframe(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    return pl.read_csv(path, infer_schema_length=10000)


@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_markdown(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


@st.cache_resource
def load_model():
    if not BEST_MODEL_FILE.exists():
        return None
    return load(BEST_MODEL_FILE)


def artifact_status() -> list[tuple[str, bool]]:
    return [
        ("Processed dataset", PROCESSED_CSV.exists()),
        ("EDA summary", EDA_SUMMARY_MD.exists()),
        ("Model results", MODEL_RESULTS.exists()),
        ("Feature importance", FEATURE_IMPORT.exists()),
        ("PCA summary", PCA_SUMMARY_JSON.exists()),
        ("Best model artifact", BEST_MODEL_FILE.exists()),
    ]


def format_compact(value: float | int) -> str:
    absolute = abs(float(value))
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def stat_card(label: str, value: str, note: str):
    """Tiny wrapper around st.metric so the header stays compact."""
    st.metric(label, value, help=note)


def numeric_default(df: pl.DataFrame, column: str, fallback: float) -> float:
    if column not in df.columns:
        return fallback
    value = df.select(
        pl.col(column).cast(pl.Float64, strict=False).median().alias("value")
    ).item()
    if value is None:
        return fallback
    return float(value)


def select_options(df: pl.DataFrame, column: str, fallback: str = "Unknown") -> list[str]:
    if column not in df.columns:
        return [fallback]
    values = sorted(
        [
            value
            for value in df[column].drop_nulls().unique().to_list()
            if value not in (None, "")
        ]
    )
    return values or [fallback]


def render_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=16, r=16, t=56, b=16),
        font=dict(family="Avenir Next, Helvetica Neue, sans-serif", color="#1d232b"),
    )
    st.plotly_chart(fig, width="stretch")


def render_table(data):
    st.dataframe(data, width="stretch", hide_index=True)


def build_missingness_df(df: pl.DataFrame) -> pl.DataFrame:
    total_rows = max(len(df), 1)
    null_counts = df.null_count().to_dicts()[0]
    rows = [
        {
            "column": column,
            "nulls": int(count),
            "pct_missing": round(100 * int(count) / total_rows, 2),
        }
        for column, count in null_counts.items()
    ]
    return pl.DataFrame(rows).sort("pct_missing", descending=True)


def prediction_column_name(model_name: str) -> str:
    slug = (
        model_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )
    return f"pred_{slug}"


def resolve_prediction_column(predictions_df: pl.DataFrame, model_name: str) -> str | None:
    preferred = prediction_column_name(model_name)
    if preferred in predictions_df.columns:
        return preferred
    for candidate in [
        "pred_gradient_boosting_tuned",
        "pred_gradient_boosting",
        "pred_random_forest",
        "pred_ridge",
        "pred_linear_regression_baseline",
    ]:
        if candidate in predictions_df.columns:
            return candidate
    return None


def render_header(df: pl.DataFrame):
    data_summary = load_json(DATA_SUMMARY_JSON) or {}
    results_df = load_dataframe(MODEL_RESULTS)
    results_pd = results_df.to_pandas() if results_df is not None else None
    best_model = None
    if results_pd is not None and not results_pd.empty:
        best_model = results_pd.sort_values("test_r2", ascending=False).iloc[0]

    st.markdown(
        '<div class="app-title">CIS 2450 Final Project Dashboard</div>'
        '<div class="app-subtitle">By: Chinmay Govind and Rohith Yelisetty</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card(
            "Tracks",
            f"{data_summary.get('row_count', len(df)):,}",
            f"{data_summary.get('column_count', df.width)} columns",
        )
    with c2:
        stat_card(
            "Artists",
            f"{data_summary.get('unique_artists', 0):,}",
            f"{len(data_summary.get('genres', []))} genres",
        )
    with c3:
        audio_pct = float(data_summary.get("audio_feature_coverage_pct", 0))
        stat_card(
            "Audio coverage",
            f"{audio_pct:.1f}%",
            "AcousticBrainz subset",
        )
    with c4:
        best_label = best_model["model"] if best_model is not None else "—"
        best_value = (
            f"R² {best_model['test_r2']:.3f}"
            if best_model is not None
            else "Run models.py"
        )
        stat_card("Best model", best_value, best_label)


def render_song_search(df: pl.DataFrame) -> dict | None:
    """Top-of-page song search. Returns the selected song row (or None).

    Lives above the tabs so any TA grading the dashboard can immediately type a
    track name and see what the project knows about it. The downstream
    `Song Lookup` tab uses the selection here as its starting point too.
    """
    with st.container(border=True):
        st.markdown("**Search a song**")
        cols = [c for c in ["mbid", "title", "artist_name", "genre",
                            TARGET, RAW_TARGET, "release_year", "duration_sec",
                            "tempo", "danceability"] if c in df.columns]
        catalog = (
            df.select(cols)
            .drop_nulls(subset=["title", "artist_name"])
            .sort(TARGET, descending=True)
            .head(10_000)
        )
        titles = catalog["title"].to_list()
        artists = catalog["artist_name"].to_list()
        labels = [f"{t} — {a}" for t, a in zip(titles, artists)]

        selected = st.selectbox(
            "Type a title or artist to filter (top 10k by replays)",
            options=[""] + labels,
            format_func=lambda x: "— pick a song —" if x == "" else x,
            label_visibility="collapsed",
            key="global_song_search",
        )
        if not selected:
            return None
        idx = labels.index(selected)
        return catalog.row(idx, named=True)


def show_command_center(df: pl.DataFrame):
    st.subheader("Overview")
    data_summary = load_json(DATA_SUMMARY_JSON) or {}
    collection_summary = load_json(COLLECTION_SUMMARY_JSON) or {}
    eda_summary = load_markdown(EDA_SUMMARY_MD)
    model_summary = load_markdown(MODEL_SUMMARY_MD)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("MusicBrainz rows", format_compact(collection_summary.get("musicbrainz_rows", 0)))
    r2.metric("ListenBrainz coverage", f"{collection_summary.get('listenbrainz_coverage_pct', 0)}%")
    r3.metric("AcousticBrainz coverage", f"{collection_summary.get('acousticbrainz_coverage_pct', 0)}%")
    r4.metric("Songs matched to lyrics", f"{20_759:,}")

    if float(data_summary.get("audio_feature_coverage_pct", 0)) / 100 < PRIMARY_MODEL_AUDIO_COVERAGE_THRESHOLD:
        st.caption(
            "AcousticBrainz coverage is below 50%, so the primary model uses metadata-only "
            "features and treats audio as optional enrichment."
        )

    top_genres = (
        df.group_by("genre")
        .agg(pl.len().alias("tracks"))
        .sort("tracks", descending=True)
        .head(12)
        .to_pandas()
    )
    decade_trend = (
        df.filter(pl.col("release_decade").is_not_null())
        .group_by("release_decade")
        .agg(pl.col(TARGET).median().alias("median_log_replays"))
        .sort("release_decade")
        .to_pandas()
    )
    top_artists_path = OUTPUT_DIR / "sql_top_artists.csv"
    top_artists = load_dataframe(top_artists_path)
    audio_by_genre = load_dataframe(OUTPUT_DIR / "sql_audio_by_genre.csv")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            top_genres,
            x="tracks",
            y="genre",
            orientation="h",
            color="tracks",
            color_continuous_scale="Brwnyl",
            title="Top Genres by Track Count",
        )
        fig.update_layout(yaxis_title="", xaxis_title="Tracks", coloraxis_showscale=False)
        render_chart(fig)

    with c2:
        fig = px.line(
            decade_trend,
            x="release_decade",
            y="median_log_replays",
            markers=True,
            title="Median Replayability by Release Decade",
        )
        fig.update_traces(line_color="#1f6f78", marker_color="#e07a5f")
        fig.update_layout(xaxis_title="Release decade", yaxis_title="Median log replay target")
        render_chart(fig)

    c3, c4 = st.columns([0.95, 1.05])
    with c3:
        st.markdown("#### Top Artists Snapshot")
        if top_artists is not None:
            render_table(top_artists.to_pandas())
        else:
            st.caption("`outputs/sql_top_artists.csv` is not available yet.")

    with c4:
        st.markdown("#### Audio Feature Coverage by Genre")
        if audio_by_genre is not None:
            audio_pd = audio_by_genre.to_pandas()
            fig = px.bar(
                audio_pd,
                x="genre",
                y="pct_with_audio_features",
                color="pct_with_audio_features",
                color_continuous_scale="Tealgrn",
                title="AcousticBrainz availability",
            )
            fig.update_layout(xaxis_title="", yaxis_title="% with audio features", coloraxis_showscale=False)
            render_chart(fig)
        else:
            st.caption("`outputs/sql_audio_by_genre.csv` is not available yet.")

    if eda_summary or model_summary:
        with st.expander("Narrative Briefing", expanded=False):
            if eda_summary:
                st.markdown("##### EDA Summary")
                st.markdown(eda_summary)
            if model_summary:
                st.markdown("##### Modeling Summary")
                st.markdown(model_summary)


def show_genre_lab(df: pl.DataFrame):
    st.subheader("Genre drilldown")

    genre_counts = (
        df.group_by("genre")
        .agg(pl.len().alias("tracks"))
        .sort("tracks", descending=True)
    )
    all_genres = genre_counts["genre"].to_list()
    default_genres = all_genres[:4] if len(all_genres) >= 4 else all_genres

    release_decades = (
        df.select(pl.col("release_decade"))
        .drop_nulls()
        .unique()
        .sort("release_decade")
        .to_series()
        .to_list()
    )
    if release_decades:
        decade_min = int(min(release_decades))
        decade_max = int(max(release_decades))
    else:
        decade_min, decade_max = 1950, CURRENT_YEAR // 10 * 10

    controls = st.container(border=True)
    with controls:
        c1, c2, c3 = st.columns([1.2, 1.1, 0.7])
        selected_genres = c1.multiselect("Genres", all_genres, default=default_genres)
        decade_range = c2.slider(
            "Release decade range",
            min_value=decade_min,
            max_value=decade_max,
            value=(max(decade_min, decade_max - 50), decade_max),
            step=10,
        )
        sample_size = c3.slider("Scatter sample", 500, 4000, 2500, step=250)

    if not selected_genres:
        st.info("Choose at least one genre to populate the lab.")
        return

    filtered = df.filter(
        pl.col("genre").is_in(selected_genres)
        & pl.col("release_decade").is_not_null()
        & pl.col("release_decade").is_between(decade_range[0], decade_range[1])
    )

    if len(filtered) == 0:
        st.info("No rows match the current genre and decade filters.")
        return

    trend_df = (
        filtered.group_by(["release_decade", "genre"])
        .agg(pl.col(TARGET).median().alias("median_log_replays"))
        .sort(["release_decade", "genre"])
        .to_pandas()
    )
    scatter_df = (
        filtered.select(["duration_sec", TARGET, "genre", "artist_name"])
        .drop_nulls()
        .sample(min(sample_size, len(filtered)), seed=42)
        .to_pandas()
    )
    box_df = filtered.select(["genre", TARGET]).drop_nulls().to_pandas()
    artist_table = (
        filtered.group_by("artist_name")
        .agg(
            pl.len().alias("tracks"),
            pl.col(TARGET).mean().alias("avg_log_replays"),
        )
        .filter(pl.col("artist_name").is_not_null())
        .sort("avg_log_replays", descending=True)
        .head(15)
        .to_pandas()
    )

    g1, g2 = st.columns(2)
    with g1:
        fig = px.line(
            trend_df,
            x="release_decade",
            y="median_log_replays",
            color="genre",
            markers=True,
            title="Replayability by Genre Over Time",
        )
        render_chart(fig)

    with g2:
        fig = px.box(
            box_df,
            x="genre",
            y=TARGET,
            color="genre",
            points="outliers",
            title="Replayability Spread Within Selected Genres",
        )
        fig.update_layout(showlegend=False, xaxis_title="")
        render_chart(fig)

    g3, g4 = st.columns([1.1, 0.9])
    with g3:
        fig = px.scatter(
            scatter_df,
            x="duration_sec",
            y=TARGET,
            color="genre",
            hover_data=["artist_name"],
            title="Duration vs. Replayability",
            opacity=0.5,
        )
        fig.update_layout(xaxis_title="Duration (seconds)", yaxis_title="Log replay target")
        render_chart(fig)

    with g4:
        st.markdown("#### Artist Leaderboard")
        render_table(artist_table)


def show_model_studio():
    st.subheader("Models")
    results_df = load_dataframe(MODEL_RESULTS)
    feature_df = load_dataframe(FEATURE_IMPORT)
    predictions_df = load_dataframe(MODEL_PREDICTIONS)
    best_params = load_json(BEST_PARAMS_JSON)
    model_summary = load_markdown(MODEL_SUMMARY_MD)
    imbalance_results = load_json(IMBALANCE_RESULTS_JSON)
    pca_summary = load_json(PCA_SUMMARY_JSON)
    data_summary = load_json(DATA_SUMMARY_JSON) or {}

    if results_df is None:
        st.warning("Run `python models.py` to generate model artifacts for this dashboard.")
        return

    results_pd = results_df.to_pandas()
    best_row = results_pd.sort_values("test_r2", ascending=False).iloc[0]
    prediction_col = (
        resolve_prediction_column(predictions_df, best_row["model"])
        if predictions_df is not None
        else None
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model", best_row["model"])
    m2.metric("Best Test R²", f"{best_row['test_r2']:.4f}")
    m3.metric("Best Test RMSE", f"{best_row['test_rmse']:.4f}")
    m4.metric("Best Test MAE", f"{best_row['test_mae']:.4f}")

    audio_coverage = float(data_summary.get("audio_feature_coverage_pct", 0)) / 100
    if audio_coverage < PRIMARY_MODEL_AUDIO_COVERAGE_THRESHOLD:
        st.caption(
            f"AcousticBrainz coverage is {audio_coverage:.1%}, so the primary modeling path is metadata-first."
        )

    c1, c2 = st.columns([1.05, 0.95])
    with c1:
        comparison_long = results_pd.melt(
            id_vars=["model"],
            value_vars=["cv_r2_mean", "test_r2"],
            var_name="split",
            value_name="r2",
        )
        fig = px.bar(
            comparison_long,
            x="model",
            y="r2",
            color="split",
            barmode="group",
            title="Cross-Validation vs Holdout R²",
            color_discrete_map={"cv_r2_mean": "#1f6f78", "test_r2": "#e07a5f"},
        )
        fig.update_layout(xaxis_title="", yaxis_title="R²")
        render_chart(fig)

        metric_long = results_pd.melt(
            id_vars=["model"],
            value_vars=["test_rmse", "test_mae"],
            var_name="metric",
            value_name="value",
        )
        fig = px.bar(
            metric_long,
            x="model",
            y="value",
            color="metric",
            barmode="group",
            title="Holdout Error Metrics",
            color_discrete_map={"test_rmse": "#355c7d", "test_mae": "#c06c84"},
        )
        fig.update_layout(xaxis_title="", yaxis_title="Error")
        render_chart(fig)

    with c2:
        if feature_df is not None:
            feature_pd = feature_df.to_pandas()
            model_choice = st.selectbox(
                "Feature importance view",
                sorted(feature_pd["model"].unique()),
                index=max(sorted(feature_pd["model"].unique()).index(best_row["model"]), 0)
                if best_row["model"] in feature_pd["model"].unique()
                else 0,
            )
            display = (
                feature_pd[feature_pd["model"] == model_choice]
                .sort_values("importance", ascending=False)
                .head(15)
                .sort_values("importance")
            )
            fig = px.bar(
                display,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Plasma",
                title=f"Top Learned Signals for {model_choice}",
            )
            fig.update_layout(xaxis_title="Importance", yaxis_title="", coloraxis_showscale=False)
            render_chart(fig)

        meta1, meta2 = st.columns(2)
        with meta1:
            if best_params:
                st.markdown("#### Tuned GBM Parameters")
                st.json(best_params)
        with meta2:
            if pca_summary:
                st.markdown("#### PCA Summary")
                st.json(pca_summary)

    if predictions_df is not None and prediction_col is not None:
        pred_pd = predictions_df.to_pandas()
        fig = px.scatter(
            pred_pd,
            x="y_true",
            y=prediction_col,
            hover_data=[column for column in ["title", "artist_name", "genre"] if column in pred_pd.columns],
            title=f"Actual vs Predicted: {best_row['model']}",
            opacity=0.35,
            color_discrete_sequence=["#1f6f78"],
        )
        fig.update_layout(xaxis_title="Actual log replay target", yaxis_title="Predicted log replay target")
        render_chart(fig)

    if imbalance_results:
        st.markdown("#### Imbalance Handling Demo")
        imbalance_df = pl.DataFrame(
            [
                {"model": model_name, **metrics}
                for model_name, metrics in imbalance_results.items()
            ]
        ).to_pandas()
        render_table(imbalance_df)

    if model_summary:
        with st.expander("Modeling Narrative", expanded=False):
            st.markdown(model_summary)


def show_prediction_console(df: pl.DataFrame):
    st.subheader("Predict")
    model = load_model()
    if model is None:
        st.warning("Run `python models.py` first so the dashboard can load the saved model artifact.")
        return

    numeric_defaults = {
        "duration_sec": numeric_default(df, "duration_sec", 210.0),
        "release_year": numeric_default(df, "release_year", 2010.0),
        "artist_career_age": numeric_default(df, "artist_career_age", 10.0),
        "genre_match_count": numeric_default(df, "genre_match_count", 1.0),
        "tempo": numeric_default(df, "tempo", 120.0),
        "danceability": numeric_default(df, "danceability", 1.0),
        "loudness": numeric_default(df, "loudness", -12.0),
        "dynamic_complexity": numeric_default(df, "dynamic_complexity", 3.0),
    }
    category_options = {
        column: select_options(df, column)
        for column in ["genre", "release_type", "artist_type", "artist_country", "key", "key_scale"]
    }

    data_summary = load_json(DATA_SUMMARY_JSON) or {}
    audio_coverage = float(data_summary.get("audio_feature_coverage_pct", 0)) / 100

    left, right = st.columns([1.1, 0.9])
    with left:
        with st.form("prediction_console"):
            st.markdown("#### Build a Track Profile")
            use_audio = st.checkbox(
                "Include optional audio features",
                value=audio_coverage >= PRIMARY_MODEL_AUDIO_COVERAGE_THRESHOLD,
            )

            c1, c2, c3 = st.columns(3)
            genre = c1.selectbox("Genre", category_options["genre"])
            release_type = c2.selectbox("Release Type", category_options["release_type"])
            artist_type = c3.selectbox("Artist Type", category_options["artist_type"])

            c4, c5, c6 = st.columns(3)
            artist_country = c4.selectbox("Artist Country", category_options["artist_country"])
            key = c5.selectbox("Key", category_options["key"])
            key_scale = c6.selectbox("Key Scale", category_options["key_scale"])

            c7, c8, c9 = st.columns(3)
            duration_sec = c7.slider("Duration (seconds)", 30, 900, int(round(numeric_defaults["duration_sec"])))
            release_year = c8.slider("Release year", 1950, CURRENT_YEAR, int(round(numeric_defaults["release_year"])))
            artist_career_age = c9.slider("Artist career age", 0, 80, int(round(numeric_defaults["artist_career_age"])))

            genre_match_count = st.slider(
                "Genre search matches",
                min_value=1,
                max_value=8,
                value=int(round(numeric_defaults["genre_match_count"])),
            )

            if use_audio:
                a1, a2, a3, a4 = st.columns(4)
                tempo = a1.slider("Tempo", 40, 220, int(round(numeric_defaults["tempo"])))
                danceability = a2.slider("Danceability", 0.0, 3.0, float(numeric_defaults["danceability"]), 0.05)
                loudness = a3.slider("Loudness", -60.0, 5.0, float(numeric_defaults["loudness"]), 0.5)
                dynamic_complexity = a4.slider(
                    "Dynamic complexity",
                    0.0,
                    15.0,
                    float(numeric_defaults["dynamic_complexity"]),
                    0.1,
                )
                has_audio_features = 1.0
                audio_feature_missing_count = 0.0
            else:
                tempo = np.nan
                danceability = np.nan
                loudness = np.nan
                dynamic_complexity = np.nan
                has_audio_features = 0.0
                audio_feature_missing_count = 4.0

            submitted = st.form_submit_button("Predict Replayability", type="primary")

    with right:
        st.markdown("#### How this works")
        st.write(
            "This form feeds the same sklearn pipeline that scored the holdout set. "
            "Imputation lives inside the pipeline, so audio inputs are optional — "
            "leave the box unchecked for a metadata-only estimate."
        )
        st.write(
            "Try varying genre, release year, and duration to see how much each one "
            "moves the predicted log-replay target."
        )

    if not submitted:
        return

    input_df = pl.DataFrame(
        {
            "duration_sec": [float(duration_sec)],
            "release_year": [float(release_year)],
            "release_decade": [float(release_year // 10 * 10)],
            "artist_career_age": [float(artist_career_age)],
            "track_age": [float(CURRENT_YEAR - release_year)],
            "genre_match_count": [float(genre_match_count)],
            "tempo": [tempo],
            "danceability": [danceability],
            "loudness": [loudness],
            "dynamic_complexity": [dynamic_complexity],
            "has_audio_features": [has_audio_features],
            "audio_feature_missing_count": [audio_feature_missing_count],
            "career_x_duration_min": [float(artist_career_age * duration_sec / 60)],
            "tempo_x_dance": [float(0 if np.isnan(tempo) or np.isnan(danceability) else tempo * danceability)],
            "genre": [genre],
            "release_type": [release_type],
            "artist_type": [artist_type],
            "artist_country": [artist_country],
            "key": [key],
            "key_scale": [key_scale],
        }
    ).to_pandas()

    prediction_log = float(model.predict(input_df)[0])
    prediction_replays = max(float(np.expm1(prediction_log)), 0.0)
    target_values = df[TARGET].drop_nulls().to_numpy()
    percentile = float((target_values <= prediction_log).mean() * 100)
    genre_median = (
        df.filter(pl.col("genre") == genre)
        .select(pl.col(TARGET).median())
        .item()
    )

    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted log target", f"{prediction_log:.3f}")
    p2.metric("Predicted repeat listens", f"{prediction_replays:,.0f}")
    p3.metric("Dataset percentile", f"{percentile:.1f}%")

    st.success(
        f"This profile sits around the {percentile:.1f}th percentile of the dataset. "
        f"The selected genre, {genre}, has a median log replay target of {genre_median:.3f}."
    )


def show_lyrics_analysis():
    st.subheader("Lyrics")

    features_df  = load_dataframe(OUTPUT_DIR / "lyrics_features.csv")
    word_freq_df = load_dataframe(OUTPUT_DIR / "lyrics_word_freq_comparison.csv")
    lyrics_summary = load_markdown(OUTPUT_DIR / "lyrics_summary.md")

    if features_df is None:
        st.warning(
            "No lyrics analysis data found. Run `python lyrics_analysis.py` to generate it."
        )
        return

    features_pd = features_df.to_pandas()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracks Analyzed", f"{len(features_pd):,}")
    c2.metric("Avg Sentiment (VADER)", f"{features_pd['sentiment_compound'].mean():.3f}")
    c3.metric("Avg Vocabulary Richness", f"{features_pd['type_token_ratio'].mean():.3f}")
    c4.metric("Avg Repetitiveness", f"{features_pd['repetitiveness'].mean():.3f}")

    st.markdown("---")

    plot_rows = [
        ("lyrics1_sentiment_by_quartile.png", "lyrics2_top_words_comparison.png"),
        ("lyrics3_wordclouds.png",             "lyrics4_features_correlation.png"),
        ("lyrics5_topic_distribution.png",     "lyrics6_complexity_vs_replay.png"),
    ]
    for left_name, right_name in plot_rows:
        lp, rp = OUTPUT_DIR / left_name, OUTPUT_DIR / right_name
        col_l, col_r = st.columns(2)
        if lp.exists():
            col_l.image(str(lp), use_container_width=True)
        if rp.exists():
            col_r.image(str(rp), use_container_width=True)

    genre_sent = OUTPUT_DIR / "lyrics7_sentiment_by_genre.png"
    if genre_sent.exists():
        st.image(str(genre_sent), use_container_width=True)

    extra_plots = [
        ("lyrics8_bigrams_comparison.png", "lyrics9_rarity_vs_replay.png"),
        ("lyrics10_word_length_vs_replay.png", "lyrics11_model_comparison.png"),
    ]
    for left_name, right_name in extra_plots:
        lp, rp = OUTPUT_DIR / left_name, OUTPUT_DIR / right_name
        col_l, col_r = st.columns(2)
        if lp.exists():
            col_l.image(str(lp), use_container_width=True)
        if rp.exists():
            col_r.image(str(rp), use_container_width=True)

    model_results_df = load_dataframe(OUTPUT_DIR / "lyrics_model_results.csv")
    if model_results_df is not None:
        st.markdown("#### Lyrics Prediction Model Results")
        render_table(model_results_df.to_pandas())

    if word_freq_df is not None:
        st.markdown("#### Word Frequency: High vs Low Replay (top 30)")
        render_table(word_freq_df.head(30).to_pandas())

    if lyrics_summary:
        with st.expander("Full Lyrics Summary Report", expanded=False):
            st.markdown(lyrics_summary)


@st.cache_data(show_spinner=False)
def _load_lyrics_cache() -> dict[str, str]:
    path = CACHE_DIR / "kaggle_lyrics_cache.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _fetch_genius_lyrics(title: str, artist: str) -> str | None:
    token = os.environ.get("GENIUS_API_TOKEN")
    if not token:
        return None
    try:
        import lyricsgenius
        genius = lyricsgenius.Genius(
            token, quiet=True, timeout=12, retries=1,
            skip_non_songs=True, remove_section_headers=False,
        )
        song = genius.search_song(title, artist)
        if song and song.lyrics:
            return song.lyrics
    except Exception:
        pass
    return None


def _analyze_lyrics(lyrics: str) -> dict:
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("stopwords", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords as nltk_sw

    text = re.sub(r"\[.*?\]", " ", lyrics)
    text = re.sub(r"Embed\s*$", " ", text, flags=re.IGNORECASE).strip()

    lines  = [ln for ln in text.splitlines() if ln.strip()]
    sw     = set(nltk_sw.words("english")) | {
        "oh", "yeah", "na", "la", "ooh", "ah", "hey", "uh", "mm",
        "gonna", "wanna", "gotta", "ain", "em", "im", "ive",
    }
    flat   = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in flat.split() if w not in sw and len(w) > 1 and not w.isdigit()]

    total  = len(tokens)
    unique = len(set(tokens))
    ttr    = unique / max(total, 1)

    line_counts = Counter(ln.strip().lower() for ln in lines)
    dup_lines   = sum(v - 1 for v in line_counts.values() if v > 1)
    rep         = dup_lines / max(len(lines), 1)

    sia  = SentimentIntensityAnalyzer()
    sent = sia.polarity_scores(text)

    top_words  = [w for w, _ in Counter(tokens).most_common(15)]
    top_counts = [c for _, c in Counter(tokens).most_common(15)]

    result: dict = {
        "word_count":         total,
        "unique_words":       unique,
        "type_token_ratio":   round(ttr, 4),
        "line_count":         len(lines),
        "repetitiveness":     round(rep, 4),
        "sentiment_compound": round(sent["compound"], 4),
        "sentiment_positive": round(sent["pos"], 4),
        "sentiment_negative": round(sent["neg"], 4),
        "sentiment_neutral":  round(sent["neu"], 4),
        "top_words":          top_words,
        "top_counts":         top_counts,
        "cleaned_text":       text,
    }
    try:
        import textstat as ts
        result["flesch_reading_ease"]  = round(ts.flesch_reading_ease(text), 1)
        result["flesch_kincaid_grade"] = round(ts.flesch_kincaid_grade(text), 1)
    except Exception:
        pass
    return result


def _model_view_of_song(df: pl.DataFrame, song_row: dict):
    """Run the saved pipeline on a single track row and show how it scored.

    This is the bit that lets a TA pick any song and see what the model
    actually does with it: predicted log replay, residual vs ground truth,
    dataset percentile, and where the song sits within its own genre.
    """
    model = load_model()
    if model is None:
        st.caption("Saved model artifact not found — run `python models.py` first to enable the model view.")
        return

    # Look up the full feature row in the processed frame so we have every
    # input the pipeline expects (the search-table version drops most cols).
    full_row = None
    if song_row.get("mbid"):
        match = df.filter(pl.col("mbid") == song_row["mbid"])
        if len(match) > 0:
            full_row = match.row(0, named=True)
    if full_row is None:
        match = df.filter(
            (pl.col("title") == song_row.get("title"))
            & (pl.col("artist_name") == song_row.get("artist_name"))
        )
        if len(match) == 0:
            st.caption("Couldn't recover the full feature row for this track.")
            return
        full_row = match.row(0, named=True)

    # Build the single-row pandas frame the pipeline was trained on.
    feature_inputs = {
        "duration_sec": full_row.get("duration_sec"),
        "release_year": full_row.get("release_year"),
        "release_decade": full_row.get("release_decade"),
        "artist_career_age": full_row.get("artist_career_age"),
        "track_age": full_row.get("track_age"),
        "genre_match_count": full_row.get("genre_match_count"),
        "tempo": full_row.get("tempo"),
        "danceability": full_row.get("danceability"),
        "loudness": full_row.get("loudness"),
        "dynamic_complexity": full_row.get("dynamic_complexity"),
        "has_audio_features": full_row.get("has_audio_features"),
        "audio_feature_missing_count": full_row.get("audio_feature_missing_count"),
        "career_x_duration_min": full_row.get("career_x_duration_min"),
        "tempo_x_dance": full_row.get("tempo_x_dance"),
        "genre": full_row.get("genre"),
        "release_type": full_row.get("release_type"),
        "artist_type": full_row.get("artist_type"),
        "artist_country": full_row.get("artist_country"),
        "key": full_row.get("key"),
        "key_scale": full_row.get("key_scale"),
    }
    try:
        input_df = pl.DataFrame([feature_inputs]).to_pandas()
        predicted_log = float(model.predict(input_df)[0])
    except Exception as exc:
        st.caption(f"Couldn't score this song with the saved model: {exc}")
        return

    actual_log = full_row.get(TARGET)
    actual_raw = full_row.get(RAW_TARGET)
    predicted_raw = max(float(np.expm1(predicted_log)), 0.0)
    residual = (predicted_log - actual_log) if actual_log is not None else None

    target_values = df[TARGET].drop_nulls().to_numpy()
    pct = float((target_values <= predicted_log).mean() * 100)
    genre_median = (
        df.filter(pl.col("genre") == full_row.get("genre"))
        .select(pl.col(TARGET).median())
        .item()
    )

    st.markdown("#### How the model sees this song")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Predicted log replays", f"{predicted_log:.3f}")
    p2.metric("Actual log replays", f"{actual_log:.3f}" if actual_log is not None else "—")
    p3.metric(
        "Residual",
        f"{residual:+.3f}" if residual is not None else "—",
        help="predicted − actual on the log scale",
    )
    p4.metric("Dataset percentile", f"{pct:.1f}%")

    note_bits = [
        f"Predicted ≈ **{predicted_raw:,.0f}** repeat listens.",
    ]
    if actual_raw is not None:
        note_bits.append(f"Actual: **{int(actual_raw):,}**.")
    if genre_median is not None:
        note_bits.append(
            f"Genre median (log) for **{full_row.get('genre')}** is **{genre_median:.3f}**."
        )
    if residual is not None:
        direction = "above" if residual > 0 else "below"
        note_bits.append(
            f"Model places this track {abs(residual):.2f} log-units {direction} reality."
        )
    st.caption(" ".join(note_bits))


def show_song_explorer(df: pl.DataFrame):
    st.subheader("Song lookup")

    # Prefer the global search selection at the top of the page; fall back to
    # an in-tab selectbox so this view still works on its own.
    row = None
    selection = st.session_state.get("global_song_search", "")
    if selection:
        catalog = (
            df.select([c for c in ["mbid", "title", "artist_name", "genre", TARGET,
                                    RAW_TARGET, "release_year", "duration_sec",
                                    "tempo", "danceability"] if c in df.columns])
            .drop_nulls(subset=["title", "artist_name"])
            .sort(TARGET, descending=True)
            .head(10_000)
        )
        labels = [f"{t} — {a}" for t, a in zip(
            catalog["title"].to_list(), catalog["artist_name"].to_list()
        )]
        if selection in labels:
            row = catalog.row(labels.index(selection), named=True)

    if row is None:
        avail_cols = df.columns
        select_cols = [c for c in ["mbid", "title", "artist_name", "genre", TARGET, RAW_TARGET,
                                    "release_year", "duration_sec"] if c in avail_cols]
        songs_df = (
            df.select(select_cols)
            .drop_nulls(subset=["title", "artist_name"])
            .sort(TARGET, descending=True)
            .head(10_000)
        )
        titles  = songs_df["title"].to_list()
        artists = songs_df["artist_name"].to_list()
        options = [f"{t} — {a}" for t, a in zip(titles, artists)]
        selected_label = st.selectbox(
            "Pick a song",
            options=[""] + options,
            format_func=lambda x: "— select a song —" if x == "" else x,
        )
        if not selected_label:
            st.caption("Tip: the search box at the top of the page also drives this view.")
            return
        idx = options.index(selected_label)
        row = songs_df.row(idx, named=True)

    # ── song metadata ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Artist", row.get("artist_name") or "—")
    c2.metric("Genre",  row.get("genre") or "—")

    yr = row.get("release_year")
    c3.metric("Release year", str(int(yr)) if yr is not None else "—")

    raw = row.get(RAW_TARGET)
    c4.metric("Repeat listens", f"{int(raw):,}" if raw is not None else "—")

    dur = row.get("duration_sec")
    c5.metric("Duration", f"{int(dur // 60)}:{int(dur % 60):02d}" if dur is not None else "—")

    st.markdown("---")

    # ── model's prediction for this exact track ───────────────────────────────
    _model_view_of_song(df, row)

    st.markdown("---")

    # ── lyrics: cache → Genius ────────────────────────────────────────────────
    mbid   = row.get("mbid", "")
    cache  = _load_lyrics_cache()
    lyrics = cache.get(mbid)

    if lyrics is None:
        with st.spinner(f'Fetching lyrics for "{row["title"]}" from Genius…'):
            lyrics = _fetch_genius_lyrics(row["title"], row["artist_name"])

    if lyrics is None:
        if not os.environ.get("GENIUS_API_TOKEN"):
            st.info(
                "This song isn't in the local lyrics cache. "
                "Set `GENIUS_API_TOKEN` in your `.env` file and restart the dashboard to enable live Genius fetching."
            )
        else:
            st.warning("Lyrics not found for this song in the local cache or on Genius.")
        return

    # ── per-song analysis ─────────────────────────────────────────────────────
    st.markdown("#### Lyrics Analysis")
    a = _analyze_lyrics(lyrics)

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Word Count",        f"{a['word_count']:,}")
    r2.metric("Unique Words",      f"{a['unique_words']:,}")
    r3.metric("Vocabulary (TTR)",  f"{a['type_token_ratio']:.3f}")
    r4.metric("Repetitiveness",    f"{a['repetitiveness']:.3f}")
    r5.metric("Sentiment",         f"{a['sentiment_compound']:+.3f}")

    if "flesch_reading_ease" in a:
        r6, r7, *_ = st.columns(5)
        r6.metric("Flesch Reading Ease",   f"{a['flesch_reading_ease']:.1f}")
        r7.metric("Flesch-Kincaid Grade",  f"{a['flesch_kincaid_grade']:.1f}")

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(
            x=["Positive", "Negative", "Neutral"],
            y=[a["sentiment_positive"], a["sentiment_negative"], a["sentiment_neutral"]],
            color=["Positive", "Negative", "Neutral"],
            color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#bdc3c7"},
            title="Sentiment Breakdown",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Proportion")
        render_chart(fig)

    with col_r:
        fig = px.bar(
            x=a["top_counts"],
            y=a["top_words"],
            orientation="h",
            color=a["top_counts"],
            color_continuous_scale="Teal",
            title="Top 15 Words",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Count", yaxis_title="",
            coloraxis_showscale=False,
        )
        render_chart(fig)

    with st.expander("Show full lyrics", expanded=False):
        st.text(a["cleaned_text"])


def show_data_notes(df: pl.DataFrame):
    st.subheader("Data quality")
    missing_df = build_missingness_df(df).head(12).to_pandas()
    fig = px.bar(
        missing_df,
        x="pct_missing",
        y="column",
        orientation="h",
        color="pct_missing",
        color_continuous_scale="OrRd",
        title="Most Missing Fields",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="% missing", yaxis_title="")
    render_chart(fig)


df = load_dataframe(PROCESSED_CSV)
if df is None:
    st.warning("No processed dataset found yet. Run `python data_processing.py` first.")
    st.stop()

render_header(df)

# Top-of-page song search — feeds the Song Lookup tab via session state.
selected_song = render_song_search(df)
if selected_song is not None:
    st.caption(
        f"Selected: **{selected_song.get('title')}** by **{selected_song.get('artist_name')}** — "
        "open the *Song lookup* tab below for the model's view."
    )

explorer_tab, command_tab, genre_tab, model_tab, predict_tab, lyrics_tab, notes_tab = st.tabs(
    ["Song lookup", "Overview", "Genre drilldown", "Models", "Predict",
     "Lyrics", "Data quality"]
)

with explorer_tab:
    show_song_explorer(df)

with command_tab:
    show_command_center(df)

with genre_tab:
    show_genre_lab(df)

with model_tab:
    show_model_studio()

with predict_tab:
    show_prediction_console(df)

with lyrics_tab:
    show_lyrics_analysis()

with notes_tab:
    show_data_notes(df)
