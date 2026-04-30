"""
Data wrangling, feature engineering, and SQL analytics.

This module turns the three pickled API caches into the single processed CSV
that everything downstream (EDA, models, dashboard) reads from. We do all the
heavy joins/aggregations in Polars because it's faster on this data size, and
then hand the same frame to DuckDB for the SQL analytics so we can show
SQL-style group-bys without leaving Python.

What happens here, end to end:
1. Load cached MusicBrainz / ListenBrainz / AcousticBrainz pickles.
2. Deduplicate MusicBrainz: a song that came back under multiple genre searches
   gets collapsed to a single MBID with all its genre tags merged into a list,
   and `genre_match_count` records how many searches it appeared in (used as a
   feature later).
3. 3-way left join on MBID (MB + LB + AB).
4. Build the regression target: `repeat_listens = total_listen_count -
   total_user_count` — i.e. listens beyond the first per unique listener.
   Then `log_repeat_listens = log(1 + repeat_listens)` to stabilize the
   right-skewed tail.
5. Engineer the rest of the features: duration_sec, release_decade, track_age,
   artist_career_age, has_audio_features (binary), audio_feature_missing_count
   (0–4), and two interaction terms (tempo * danceability and
   artist_career_age * duration_minutes).
6. Winsorize the continuous numeric features at the 1st/99th percentile so a
   handful of extreme outliers don't dominate the linear models.
7. Label-encode all categoricals (genre, release_type, artist_type, country,
   key, key_scale) into `_enc` columns; original strings are kept for display.
8. Run 8 DuckDB queries: per-genre summary, decade trend, top artists, country
   breakdown, duration buckets, genre overlap, and two audio-coverage views.
"""
from __future__ import annotations

import json
import pickle

import duckdb
import numpy as np
import polars as pl

from config import (
    AB_CACHE_FILE,
    AUDIO_CATEGORICAL_FEATURES,
    AUDIO_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CLASSIFICATION_TARGET,
    CURRENT_YEAR,
    DATA_SUMMARY_JSON,
    FEATURE_COLS,
    LB_CACHE_FILE,
    MB_CACHE_FILE,
    NUMERIC_FEATURES,
    PROCESSED_CSV,
    RAW_TARGET,
    SQL_REPORT_CSV,
    TARGET,
)


def _load_pickle(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing cache: {path}. Run `python data_collection.py` first.")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _aggregate_musicbrainz_rows(mb_rows: list[dict]) -> pl.DataFrame:
    """Collapse duplicate MBIDs introduced by multi-genre search collection."""
    df = pl.DataFrame(mb_rows, infer_schema_length=5000)
    df = df.filter(pl.col("mbid").is_not_null())

    grouped = (
        df.group_by("mbid")
        .agg(
            pl.first("title").alias("title"),
            pl.first("duration_ms").alias("duration_ms"),
            pl.first("disambiguation").alias("disambiguation"),
            pl.first("release_type").alias("release_type"),
            pl.first("release_year").alias("release_year"),
            pl.first("artist_name").alias("artist_name"),
            pl.first("artist_type").alias("artist_type"),
            pl.first("artist_country").alias("artist_country"),
            pl.first("artist_begin").alias("artist_begin"),
            pl.col("genre").drop_nulls().unique().sort().alias("genre_tags"),
        )
        .with_columns(
            pl.col("genre_tags").list.first().fill_null("Unknown").alias("genre"),
            pl.col("genre_tags").list.join(" | ").alias("genre_tags_text"),
            pl.col("genre_tags").list.len().cast(pl.Int32).alias("genre_match_count"),
        )
        .drop("genre_tags")
    )
    print(f"[MB] Deduplicated {len(df):,} raw rows to {len(grouped):,} unique MBIDs.")
    return grouped


def _mapping_to_frame(mapping: dict[str, dict], schema: dict[str, pl.DataType] | None = None) -> pl.DataFrame:
    rows = [{"mbid": mbid, **values} for mbid, values in mapping.items()]
    if not rows:
        return pl.DataFrame(schema=schema or {"mbid": pl.String})
    return pl.DataFrame(rows, schema=schema, infer_schema_length=5000)


def _winsorize(df: pl.DataFrame, columns: list[str], lower_q: float = 0.01, upper_q: float = 0.99) -> pl.DataFrame:
    for column in columns:
        if column not in df.columns:
            continue
        stats = df.select(
            pl.col(column).quantile(lower_q).alias("lower"),
            pl.col(column).quantile(upper_q).alias("upper"),
        ).to_dicts()[0]
        lower = stats.get("lower")
        upper = stats.get("upper")
        if lower is None or upper is None:
            continue
        df = df.with_columns(pl.col(column).clip(lower, upper).alias(column))
    return df


def _add_categorical_encodings(df: pl.DataFrame) -> pl.DataFrame:
    for column in CATEGORICAL_FEATURES:
        if column not in df.columns:
            df = df.with_columns(pl.lit("Unknown").alias(column))
        df = df.with_columns(pl.col(column).fill_null("Unknown").cast(pl.String).alias(column))
        mapping = {value: idx for idx, value in enumerate(df[column].unique().sort().to_list())}
        df = df.with_columns(
            pl.col(column).replace_strict(mapping, default=0).cast(pl.Int32).alias(f"{column}_enc")
        )
    return df


def build_dataframe() -> pl.DataFrame:
    mb_rows = _load_pickle(MB_CACHE_FILE)
    lb_rows = _load_pickle(LB_CACHE_FILE)
    ab_rows = _load_pickle(AB_CACHE_FILE) if AB_CACHE_FILE.exists() else {}

    mb_df = _aggregate_musicbrainz_rows(mb_rows)
    lb_df = _mapping_to_frame(
        lb_rows,
        schema={
            "mbid": pl.String,
            "total_listen_count": pl.Int64,
            "total_user_count": pl.Int64,
        },
    )
    df = mb_df.join(lb_df, on="mbid", how="left")

    if ab_rows:
        ab_df = _mapping_to_frame(ab_rows)
        df = df.join(ab_df, on="mbid", how="left")
    else:
        for column in ["tempo", "danceability", "loudness", "dynamic_complexity"]:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(column))
        for column in ["key", "key_scale"]:
            df = df.with_columns(pl.lit(None).cast(pl.String).alias(column))

    for column in ["tempo", "danceability", "loudness", "dynamic_complexity"]:
        if column in df.columns:
            df = df.with_columns(pl.col(column).cast(pl.Float64, strict=False).alias(column))

    before = len(df)
    df = df.with_columns(
        (pl.col("total_listen_count") - pl.col("total_user_count")).alias(RAW_TARGET)
    ).filter(pl.col(RAW_TARGET).is_not_null() & (pl.col(RAW_TARGET) >= 0))
    print(f"[merge] Dropped {before - len(df):,} rows without a valid target. Remaining: {len(df):,}.")

    df = df.with_columns(
        pl.col("duration_ms").cast(pl.Float64, strict=False).alias("duration_ms"),
        pl.col("release_year").cast(pl.Int32, strict=False).alias("release_year"),
        pl.col("artist_begin").cast(pl.Int32, strict=False).alias("artist_begin_year"),
    )

    df = df.with_columns(
        (pl.col("duration_ms") / 1000).clip(0, 900).alias("duration_sec"),
        pl.when(pl.col("release_year").is_between(1900, CURRENT_YEAR))
        .then(pl.col("release_year"))
        .otherwise(None)
        .alias("release_year"),
    )

    df = df.with_columns(
        (pl.col("release_year") // 10 * 10).alias("release_decade"),
        (CURRENT_YEAR - pl.col("release_year")).clip(0, 120).alias("track_age"),
        (pl.col("release_year") - pl.col("artist_begin_year")).clip(0, 80).alias("artist_career_age"),
        (pl.col(RAW_TARGET).cast(pl.Float64) + 1).log().alias(TARGET),
    )

    audio_columns = ["tempo", "danceability", "loudness", "dynamic_complexity"]
    audio_presence_expr = None
    for column in audio_columns:
        expr = pl.col(column).is_not_null()
        audio_presence_expr = expr if audio_presence_expr is None else (audio_presence_expr | expr)

    missing_audio_expr = None
    for column in audio_columns:
        expr = pl.col(column).is_null().cast(pl.Int32)
        missing_audio_expr = expr if missing_audio_expr is None else (missing_audio_expr + expr)

    df = df.with_columns(
        pl.when(audio_presence_expr).then(1).otherwise(0).cast(pl.Int8).alias("has_audio_features"),
        missing_audio_expr.cast(pl.Int32).alias("audio_feature_missing_count"),
        (pl.col("artist_career_age").fill_null(0) * pl.col("duration_sec").fill_null(0) / 60)
        .alias("career_x_duration_min"),
        pl.when(pl.col("tempo").is_not_null() & pl.col("danceability").is_not_null())
        .then(pl.col("tempo") * pl.col("danceability"))
        .otherwise(None)
        .alias("tempo_x_dance"),
    )

    df = _winsorize(
        df,
        columns=["tempo", "danceability", "loudness", "dynamic_complexity", "duration_sec"],
    )
    df = _add_categorical_encodings(df)

    ordered_columns = [
        "mbid",
        "title",
        "artist_name",
        "genre",
        "genre_tags_text",
        "genre_match_count",
        "release_type",
        "release_year",
        "release_decade",
        "artist_type",
        "artist_country",
        "artist_begin_year",
        "artist_career_age",
        "track_age",
        "duration_ms",
        "duration_sec",
        "tempo",
        "danceability",
        "loudness",
        "dynamic_complexity",
        "key",
        "key_scale",
        "has_audio_features",
        "audio_feature_missing_count",
        "career_x_duration_min",
        "tempo_x_dance",
        "total_listen_count",
        "total_user_count",
        RAW_TARGET,
        TARGET,
    ]
    encoding_columns = [f"{column}_enc" for column in CATEGORICAL_FEATURES if f"{column}_enc" in df.columns]
    remaining_columns = [col for col in df.columns if col not in ordered_columns + encoding_columns]
    df = df.select([col for col in ordered_columns if col in df.columns] + encoding_columns + remaining_columns)

    print(f"[merge] Final processed frame: {df.shape[0]:,} rows x {df.shape[1]} columns.")
    return df


def export_data_summary(df: pl.DataFrame):
    rows_with_audio = int(df["has_audio_features"].sum()) if "has_audio_features" in df.columns else 0
    complete_audio_rows = (
        df.filter(
            pl.col("tempo").is_not_null()
            & pl.col("danceability").is_not_null()
            & pl.col("loudness").is_not_null()
            & pl.col("dynamic_complexity").is_not_null()
        ).height
        if all(column in df.columns for column in AUDIO_NUMERIC_FEATURES[:4])
        else 0
    )
    summary = {
        "row_count": df.height,
        "column_count": df.width,
        "unique_artists": df.select(pl.col("artist_name").n_unique()).item(),
        "unique_tracks": df.select(pl.col("mbid").n_unique()).item(),
        "genres": sorted(df["genre"].drop_nulls().unique().to_list()),
        "median_repeat_listens": float(df.select(pl.col(RAW_TARGET).median()).item()),
        "median_log_repeat_listens": float(df.select(pl.col(TARGET).median()).item()),
        "rows_with_any_audio_features": rows_with_audio,
        "rows_with_complete_audio_features": complete_audio_rows,
        "audio_feature_coverage_pct": round(100 * rows_with_audio / max(df.height, 1), 2),
        "complete_audio_feature_coverage_pct": round(100 * complete_audio_rows / max(df.height, 1), 2),
        "null_counts": {
            row["column"]: int(row["nulls"])
            for row in (
                df.null_count()
                .transpose(include_header=True, column_names=["nulls"])
                .to_dicts()
            )
        },
    }
    with open(DATA_SUMMARY_JSON, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[summary] Data summary written to {DATA_SUMMARY_JSON.name}.")


def run_sql_analytics(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Run DuckDB SQL queries over the processed data."""
    print("\n[SQL] Running DuckDB analytics...")
    con = duckdb.connect(":memory:")
    con.register("tracks", df.to_pandas())

    queries = {
        "genre_summary": """
            SELECT
                genre,
                COUNT(*) AS n_tracks,
                ROUND(AVG(repeat_listens), 1) AS avg_repeats,
                ROUND(MEDIAN(repeat_listens), 1) AS median_repeats,
                ROUND(AVG(log_repeat_listens), 3) AS avg_log_repeats
            FROM tracks
            GROUP BY genre
            ORDER BY median_repeats DESC
        """,
        "decade_trend": """
            SELECT
                release_decade,
                COUNT(*) AS n_tracks,
                ROUND(AVG(log_repeat_listens), 3) AS avg_log_repeats,
                ROUND(STDDEV(log_repeat_listens), 3) AS std_log_repeats
            FROM tracks
            WHERE release_decade IS NOT NULL
            GROUP BY release_decade
            ORDER BY release_decade
        """,
        "top_artists": """
            SELECT
                artist_name,
                COUNT(*) AS n_tracks,
                ROUND(AVG(repeat_listens), 1) AS avg_repeats,
                MAX(repeat_listens) AS peak_repeats
            FROM tracks
            WHERE artist_name IS NOT NULL
            GROUP BY artist_name
            HAVING COUNT(*) >= 5
            ORDER BY avg_repeats DESC
            LIMIT 25
        """,
        "country_breakdown": """
            SELECT
                artist_country,
                COUNT(*) AS n_tracks,
                ROUND(AVG(log_repeat_listens), 3) AS avg_log_repeats
            FROM tracks
            WHERE artist_country IS NOT NULL
              AND artist_country <> 'Unknown'
            GROUP BY artist_country
            HAVING COUNT(*) >= 30
            ORDER BY avg_log_repeats DESC
            LIMIT 30
        """,
        "duration_buckets": """
            SELECT
                CASE
                    WHEN duration_sec < 90 THEN '0_<1.5min'
                    WHEN duration_sec < 180 THEN '1_1.5-3min'
                    WHEN duration_sec < 240 THEN '2_3-4min'
                    WHEN duration_sec < 300 THEN '3_4-5min'
                    WHEN duration_sec < 420 THEN '4_5-7min'
                    ELSE '5_7+min'
                END AS duration_bucket,
                COUNT(*) AS n_tracks,
                ROUND(AVG(log_repeat_listens), 3) AS avg_log_repeats
            FROM tracks
            WHERE duration_sec IS NOT NULL
            GROUP BY duration_bucket
            ORDER BY duration_bucket
        """,
        "genre_overlap": """
            SELECT
                genre_match_count,
                COUNT(*) AS n_tracks,
                ROUND(AVG(log_repeat_listens), 3) AS avg_log_repeats
            FROM tracks
            GROUP BY genre_match_count
            ORDER BY genre_match_count
        """,
        "audio_coverage": """
            SELECT
                COUNT(*) AS total_rows,
                SUM(CASE WHEN has_audio_features = 1 THEN 1 ELSE 0 END) AS with_audio_features,
                ROUND(100.0 * AVG(has_audio_features), 2) AS pct_with_audio_features,
                ROUND(AVG(CASE WHEN has_audio_features = 1 THEN log_repeat_listens END), 3) AS avg_log_replays_with_audio,
                ROUND(AVG(CASE WHEN has_audio_features = 0 THEN log_repeat_listens END), 3) AS avg_log_replays_without_audio
            FROM tracks
        """,
        "audio_by_genre": """
            SELECT
                genre,
                COUNT(*) AS n_tracks,
                SUM(CASE WHEN has_audio_features = 1 THEN 1 ELSE 0 END) AS with_audio_features,
                ROUND(100.0 * AVG(has_audio_features), 2) AS pct_with_audio_features
            FROM tracks
            GROUP BY genre
            ORDER BY pct_with_audio_features DESC, n_tracks DESC
        """,
    }

    results: dict[str, pl.DataFrame] = {}
    summary_rows = []
    for name, sql in queries.items():
        result = pl.from_pandas(con.execute(sql).fetch_df())
        results[name] = result
        result.write_csv(SQL_REPORT_CSV.parent / f"sql_{name}.csv")
        summary_rows.append({"section": name, "rows": len(result)})
        print(f"  [SQL] {name}: {len(result)} rows")

    pl.DataFrame(summary_rows).write_csv(SQL_REPORT_CSV)
    con.close()
    print(f"[SQL] Wrote query outputs to {SQL_REPORT_CSV.parent}.")
    return results


def make_balanced_classification_split(df: pl.DataFrame) -> pl.DataFrame:
    """Create a derived high-replay classification target."""
    threshold = float(np.percentile(df[RAW_TARGET].to_numpy(), 75))
    df = df.with_columns(
        (pl.col(RAW_TARGET) > threshold).cast(pl.Int8).alias(CLASSIFICATION_TARGET)
    )
    positives = int(df[CLASSIFICATION_TARGET].sum())
    print(
        f"[balance] Threshold={threshold:.0f}; positives={positives:,} / {len(df):,} "
        f"({positives / max(len(df), 1):.1%})."
    )
    return df


def main():
    df = build_dataframe()
    df = make_balanced_classification_split(df)
    df.write_csv(PROCESSED_CSV)
    print(f"\n[write] Processed dataset written to {PROCESSED_CSV}.")

    export_data_summary(df)
    run_sql_analytics(df)

    print("\n[diagnostic] Null counts for key features:")
    diagnostic_cols = [col for col in FEATURE_COLS + [TARGET, CLASSIFICATION_TARGET] if col in df.columns]
    nulls = (
        df.select(diagnostic_cols)
        .null_count()
        .transpose(include_header=True, column_names=["nulls"])
        .sort("nulls", descending=True)
    )
    print(nulls)
    print("\nNext steps: `python eda.py`, `python models.py`, and `streamlit run dashboard.py`.")


if __name__ == "__main__":
    main()
