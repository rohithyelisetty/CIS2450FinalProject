"""Shared configuration for the full music replayability project."""
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# Raw caches and source-level exports
MB_CACHE_FILE = CACHE_DIR / "musicbrainz_cache.pkl"
LB_CACHE_FILE = CACHE_DIR / "listenbrainz_cache.pkl"
AB_CACHE_FILE = CACHE_DIR / "acousticbrainz_cache.pkl"
LYRICS_CACHE = CACHE_DIR / "genius_lyrics_cache.pkl"

RAW_MB_CSV = OUTPUT_DIR / "raw_musicbrainz.csv"
RAW_LB_CSV = OUTPUT_DIR / "raw_listenbrainz.csv"
RAW_AB_CSV = OUTPUT_DIR / "raw_acousticbrainz.csv"
RAW_LYRICS_CSV = OUTPUT_DIR / "raw_genius_lyrics_sample.csv"
COLLECTION_SUMMARY_JSON = OUTPUT_DIR / "collection_summary.json"


# Processed data and analytics artifacts
PROCESSED_CSV = OUTPUT_DIR / "music_dataset_processed.csv"
DATA_SUMMARY_JSON = OUTPUT_DIR / "data_summary.json"
SQL_REPORT_CSV = OUTPUT_DIR / "sql_analysis_report.csv"
EDA_SUMMARY_MD = OUTPUT_DIR / "eda_summary.md"
MODEL_SUMMARY_MD = OUTPUT_DIR / "model_summary.md"


# Modeling artifacts
MODEL_RESULTS = OUTPUT_DIR / "model_results.csv"
FEATURE_IMPORT = OUTPUT_DIR / "feature_importance.csv"
MODEL_PREDICTIONS = OUTPUT_DIR / "model_predictions.csv"
BEST_MODEL_FILE = OUTPUT_DIR / "best_gbm_model.joblib"
BEST_PARAMS_JSON = OUTPUT_DIR / "best_gbm_params.json"
IMBALANCE_RESULTS_JSON = OUTPUT_DIR / "imbalance_results.json"
PCA_SUMMARY_JSON = OUTPUT_DIR / "pca_summary.json"


# Plot outputs
EDA_PLOT_FILES = {
    "target_distribution": OUTPUT_DIR / "eda1_target_distribution.png",
    "genre_ranking": OUTPUT_DIR / "eda2_genre_ranking.png",
    "decade_trend": OUTPUT_DIR / "eda3_decade_trend.png",
    "tempo_danceability": OUTPUT_DIR / "eda4_tempo_danceability.png",
    "correlation": OUTPUT_DIR / "eda5_correlation.png",
    "duration_buckets": OUTPUT_DIR / "eda6_duration_buckets.png",
    "missingness": OUTPUT_DIR / "eda7_missingness.png",
    "outliers_by_genre": OUTPUT_DIR / "eda8_outliers_by_genre.png",
}
MODEL_COMPARISON_PLOT = OUTPUT_DIR / "model_comparison.png"
FEATURE_IMPORTANCE_PLOT = OUTPUT_DIR / "feature_importance.png"
PREDICTIONS_PLOT = OUTPUT_DIR / "pred_vs_actual.png"


HEADERS = {
    "User-Agent": "CIS2450ReplayabilityProject/1.0 "
                  "(music-replayability-analysis@example.com)"
}

GENRES = [
    "pop", "rock", "hip-hop", "jazz", "classical", "electronic",
    "r&b", "country", "metal", "folk", "latin", "soul", "punk",
    "reggae", "blues", "indie", "dance", "ambient", "funk", "gospel",
]

PER_GENRE_DEFAULT = 5000
LB_BATCH_SIZE = 500
AB_REQUEST_TIMEOUT = 10
RANDOM_STATE = 42
CURRENT_YEAR = date.today().year


RAW_TARGET = "repeat_listens"
TARGET = "log_repeat_listens"
CLASSIFICATION_TARGET = "is_high_replay"


NUMERIC_FEATURES = [
    "duration_sec",
    "release_year",
    "release_decade",
    "artist_career_age",
    "track_age",
    "genre_match_count",
    "has_audio_features",
    "audio_feature_missing_count",
    "career_x_duration_min",
]

AUDIO_NUMERIC_FEATURES = [
    "tempo",
    "danceability",
    "loudness",
    "dynamic_complexity",
    "tempo_x_dance",
]

AUDIO_CATEGORICAL_FEATURES = [
    "key",
    "key_scale",
]

CORE_CATEGORICAL_FEATURES = [
    "genre",
    "release_type",
    "artist_type",
    "artist_country",
]

CATEGORICAL_FEATURES = CORE_CATEGORICAL_FEATURES + AUDIO_CATEGORICAL_FEATURES
FEATURE_COLS = NUMERIC_FEATURES + AUDIO_NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Treat AcousticBrainz as optional enrichment when coverage is sparse.
PRIMARY_MODEL_AUDIO_COVERAGE_THRESHOLD = 0.50
