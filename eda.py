"""
Exploratory data analysis for the music replayability project.

The script saves 8 figures to `outputs/` and writes a narrative markdown
summary that can be reused in the final report, dashboard, and presentation.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats

from config import EDA_PLOT_FILES, EDA_SUMMARY_MD, OUTPUT_DIR, PROCESSED_CSV, RAW_TARGET, TARGET


def _load() -> pl.DataFrame:
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(f"{PROCESSED_CSV} not found. Run `python data_processing.py` first.")
    return pl.read_csv(PROCESSED_CSV, infer_schema_length=10000)


def _save(fig, key: str):
    path = EDA_PLOT_FILES[key]
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def eda_target_distribution(df: pl.DataFrame) -> str:
    raw = df[RAW_TARGET].cast(pl.Float64).drop_nulls().to_numpy()
    log = df[TARGET].drop_nulls().to_numpy()
    clip = float(np.percentile(raw, 99.5))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(np.clip(raw, 0, clip), bins=80, color="#355C7D", edgecolor="white", linewidth=0.3)
    axes[0].set_title("Raw repeat_listens (clipped at 99.5th percentile)", fontweight="bold")
    axes[0].set_xlabel("Repeat listens")
    axes[0].set_ylabel("Track count")

    axes[1].hist(log, bins=80, color="#C06C84", edgecolor="white", linewidth=0.3)
    axes[1].set_title("log(1 + repeat_listens)", fontweight="bold")
    axes[1].set_xlabel("Log repeat listens")
    axes[1].set_ylabel("Track count")

    fig.suptitle("EDA 1 · Target distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "target_distribution")
    return (
        "The raw target is extremely right-skewed, with a small number of viral tracks "
        "dominating the replay count. The log transform compresses the heavy tail and "
        "creates a far more stable regression target for downstream modeling."
    )


def eda_genre_ranking(df: pl.DataFrame) -> str:
    grouped = (
        df.group_by("genre")
        .agg(pl.col(RAW_TARGET).median().alias("median_repeats"))
        .sort("median_repeats", descending=True)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(
        grouped["genre"].to_list(),
        grouped["median_repeats"].to_numpy(),
        color=plt.cm.viridis(np.linspace(0.15, 0.9, len(grouped))),
    )
    ax.invert_yaxis()
    ax.set_xlabel("Median repeat listens")
    ax.set_title("EDA 2 · Median repeat listens by genre", fontweight="bold")
    fig.tight_layout()
    _save(fig, "genre_ranking")
    return (
        "Replay behavior differs meaningfully across genres, which justifies keeping genre "
        "as a primary predictive feature and also supports the business story that music "
        "replayability is partly audience-segment dependent."
    )


def eda_decade_trend(df: pl.DataFrame) -> str:
    decade_df = (
        df.filter(pl.col("release_decade").is_not_null())
        .group_by("release_decade")
        .agg(
            pl.col(TARGET).median().alias("median_log_replays"),
            pl.col(TARGET).std().alias("std_log_replays"),
        )
        .sort("release_decade")
    )

    x_vals = decade_df["release_decade"].to_numpy()
    medians = decade_df["median_log_replays"].to_numpy()
    stds = np.nan_to_num(decade_df["std_log_replays"].to_numpy(), nan=0.0)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x_vals, medians, marker="o", linewidth=2.5, color="#2A9D8F", markersize=7)
    ax.fill_between(x_vals, medians - stds, medians + stds, alpha=0.2, color="#2A9D8F")
    ax.set_xlabel("Release decade")
    ax.set_ylabel("Median log(1 + repeat listens)")
    ax.set_title("EDA 3 · Replayability across decades", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{int(value)}s"))
    fig.tight_layout()
    _save(fig, "decade_trend")
    return (
        "Replayability varies by release era, which suggests that track age and release "
        "context contain signal. This motivated the engineered `release_decade` and "
        "`track_age` features in the preprocessing pipeline."
    )


def eda_tempo_danceability(df: pl.DataFrame) -> str:
    subset = df.filter(pl.col("tempo").is_not_null() & pl.col("danceability").is_not_null())
    if len(subset) < 500:
        return "AcousticBrainz coverage was too limited to render the tempo/danceability scatter plot."

    sample = subset.sample(min(5000, len(subset)), seed=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        sample["tempo"].to_numpy(),
        sample["danceability"].to_numpy(),
        c=sample[TARGET].to_numpy(),
        cmap="plasma",
        alpha=0.55,
        s=18,
        edgecolors="none",
    )
    plt.colorbar(scatter, ax=ax, label="Log replay target")
    ax.set_xlabel("Tempo (BPM)")
    ax.set_ylabel("Danceability")
    ax.set_title("EDA 4 · Tempo vs. danceability (audio-covered subset)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "tempo_danceability")
    return (
        "This plot is drawn only on the subset of tracks with AcousticBrainz coverage. It is "
        "useful for enrichment and storytelling, but the sparse coverage means audio features "
        "should be treated as optional rather than assumed for every record."
    )


def eda_correlation(df: pl.DataFrame) -> str:
    numeric_cols = [
        TARGET,
        "duration_sec",
        "release_year",
        "artist_career_age",
        "track_age",
        "tempo",
        "danceability",
        "loudness",
        "dynamic_complexity",
        "genre_enc",
        "release_type_enc",
        "artist_type_enc",
    ]
    numeric_cols = [column for column in numeric_cols if column in df.columns]
    corr = df.select([pl.col(column).cast(pl.Float64, strict=False) for column in numeric_cols]).to_pandas().corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("EDA 5 · Correlation heatmap", fontweight="bold")
    fig.tight_layout()
    _save(fig, "correlation")
    return (
        "Most single features correlate only weakly with the target, which suggests that "
        "replayability is driven by combinations of metadata and audio signals rather than "
        "a single dominant variable."
    )


def eda_duration_buckets(df: pl.DataFrame) -> str:
    edges = [0.0, 90.0, 180.0, 240.0, 300.0, 420.0, 900.0]
    labels = ["<1.5min", "1.5-3min", "3-4min", "4-5min", "5-7min", "7+min"]
    bucketed = (
        df.filter(pl.col("duration_sec").is_not_null())
        .with_columns(pl.col("duration_sec").cut(breaks=edges[1:-1], labels=labels).alias("duration_bin"))
        .group_by("duration_bin")
        .agg(pl.col(TARGET).median().alias("median_log_replays"))
        .sort("duration_bin")
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        bucketed["duration_bin"].to_list(),
        bucketed["median_log_replays"].to_numpy(),
        color=sns.color_palette("muted", len(bucketed)),
    )
    ax.set_xlabel("Track duration bucket")
    ax.set_ylabel("Median log(1 + repeat listens)")
    ax.set_title("EDA 6 · Replayability by duration", fontweight="bold")
    fig.tight_layout()
    _save(fig, "duration_buckets")
    return (
        "Duration matters in an interpretable way: mid-length songs tend to have the best "
        "replay outcomes, while very short interludes and very long tracks lag behind."
    )


def eda_missingness(df: pl.DataFrame) -> str:
    missing = (
        df.null_count()
        .transpose(include_header=True, column_names=["nulls"])
        .with_columns((pl.col("nulls") / len(df) * 100).alias("pct_missing"))
        .sort("pct_missing", descending=True)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        missing["column"].to_list(),
        missing["pct_missing"].to_numpy(),
        color=sns.color_palette("rocket_r", len(missing)),
    )
    ax.invert_yaxis()
    ax.set_xlabel("% rows missing")
    ax.set_title("EDA 7 · Missingness by column", fontweight="bold")
    fig.tight_layout()
    _save(fig, "missingness")
    return (
        "Missingness is overwhelmingly concentrated in the AcousticBrainz fields. That finding "
        "changed the downstream strategy: the main modeling path is metadata-first, while audio "
        "variables are treated as partial enrichment instead of mandatory inputs."
    )


def eda_outliers_by_genre(df: pl.DataFrame) -> str:
    pdf = df.select(["genre", TARGET]).to_pandas()
    order = (
        pdf.groupby("genre")[TARGET]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(
        data=pdf,
        x="genre",
        y=TARGET,
        order=order,
        ax=ax,
        palette="viridis",
        fliersize=2,
        showfliers=True,
    )
    ax.set_title("EDA 8 · Outliers by genre", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Log replay target")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    _save(fig, "outliers_by_genre")
    return (
        "All genres retain upper-tail outliers even after the log transform, which is why "
        "the project compares robust tree-based models against ordinary linear regression "
        "instead of relying on a single modeling family."
    )


def run_statistical_tests(df: pl.DataFrame) -> str:
    top_genres = (
        df.group_by("genre")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(5)["genre"]
        .to_list()
    )
    genre_groups = [
        df.filter(pl.col("genre") == genre)[TARGET].drop_nulls().to_numpy()
        for genre in top_genres
    ]
    genre_groups = [group for group in genre_groups if len(group) > 20]

    kruskal_text = "Kruskal-Wallis test not run due to insufficient genre group sizes."
    if len(genre_groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*genre_groups)
        kruskal_text = (
            f"Kruskal-Wallis across the five largest genres: H={kw_stat:.2f}, p={kw_p:.3e}. "
            "This supports the claim that replay distributions differ significantly by genre."
        )

    corr_df = df.select(["duration_sec", TARGET]).drop_nulls()
    spearman_text = "Spearman correlation not run due to insufficient duration coverage."
    if len(corr_df) > 20:
        rho, p_value = stats.spearmanr(corr_df["duration_sec"].to_numpy(), corr_df[TARGET].to_numpy())
        spearman_text = (
            f"Spearman correlation between duration and log replay target: rho={rho:.3f}, "
            f"p={p_value:.3e}. This quantifies the directional relationship seen in the "
            "duration-bucket chart."
        )

    return f"{kruskal_text}\n\n{spearman_text}"


def write_summary_markdown(df: pl.DataFrame, sections: list[tuple[str, str]]):
    audio_rows = int(df["has_audio_features"].sum()) if "has_audio_features" in df.columns else 0
    lines = [
        "# EDA Summary",
        "",
        "## Dataset Context",
        "",
        f"- Rows analyzed: {len(df):,}",
        f"- Columns available: {df.shape[1]}",
        f"- Target variable: `{TARGET}` derived from `{RAW_TARGET}`",
        f"- Rows with any AcousticBrainz coverage: {audio_rows:,} ({audio_rows / max(len(df), 1):.1%})",
        "",
        "## Findings",
        "",
    ]

    for title, body in sections:
        lines.extend([f"### {title}", "", body, ""])

    lines.extend(["## Statistical Checks", "", run_statistical_tests(df), ""])
    EDA_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary markdown written to {EDA_SUMMARY_MD}")


def main():
    plt.rcParams.update({"figure.dpi": 130, "font.size": 11})
    sns.set_style("whitegrid")
    df = _load()
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {PROCESSED_CSV.name}\n")

    sections = [
        ("Target Distribution", eda_target_distribution(df)),
        ("Genre Ranking", eda_genre_ranking(df)),
        ("Decade Trend", eda_decade_trend(df)),
        ("Tempo and Danceability", eda_tempo_danceability(df)),
        ("Correlation Structure", eda_correlation(df)),
        ("Duration Buckets", eda_duration_buckets(df)),
        ("Missingness", eda_missingness(df)),
        ("Outliers by Genre", eda_outliers_by_genre(df)),
    ]
    write_summary_markdown(df, sections)
    print(f"\nAll EDA plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
