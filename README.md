# Music Replayability Final Project

This project predicts **music replayability** using linked data from multiple music APIs:

- **MusicBrainz** for recording and artist metadata
- **ListenBrainz** for popularity and listening counts
- **AcousticBrainz** for optional low-level audio enrichment where coverage exists
- **Genius** for optional sampled lyrical context

The main prediction target is:

- `repeat_listens = total_listen_count - total_user_count`
- `log_repeat_listens = log(1 + repeat_listens)`

The overall workflow is modular and mirrors the final deliverable requirements: data collection, preprocessing, EDA, modeling, and dashboarding.

## Project Structure

- `config.py`
  Shared constants, paths, feature definitions, and output locations.
- `data_collection.py`
  Pulls data from the external APIs, writes raw CSVs, and caches responses to pickle files.
- `data_processing.py`
  Deduplicates recordings, links the sources together with Polars joins, engineers features, and runs DuckDB SQL analysis.
- `eda.py`
  Generates 8 EDA plots and writes `outputs/eda_summary.md`.
- `models.py`
  Trains baseline and advanced models, includes a PCA-based regression path, performs 5-fold CV and RandomizedSearchCV, saves model artifacts, and writes `outputs/model_summary.md`.
- `dashboard.py`
  Streamlit dashboard with tabs for Overview, EDA, Modeling, Live Predictor, and Genre Explorer.
- `RUBRIC_EXPLANATION.md`
  Point-by-point explanation of how the final codebase addresses the rubric.

## Installation

```bash
pip install -r requirements.txt
```

If you want the optional Genius stage, set:

```bash
export GENIUS_API_TOKEN="your_token_here"
```

## Recommended Run Order

```bash
python data_collection.py --per-genre 5000
python data_processing.py
python eda.py
python models.py
streamlit run dashboard.py
```

## Important Notes

- The collection script is designed to target **50,000+ cleaned rows** by collecting up to 100,000 raw MusicBrainz rows across 20 genres.
- API responses are cached locally in `cache/` so you do not need to re-fetch everything on every run.
- The preprocessing script fixes the earlier artist life-span issue, removes invalid targets, and deduplicates recordings that appeared in multiple genre searches.
- AcousticBrainz coverage is partial, so the project treats audio variables as an enrichment layer rather than a required source for every row.
- The main modeling path is designed to remain stable even when audio coverage is sparse.
- The dashboard depends on the processed dataset and model artifacts in `outputs/`.

## Main Output Files

After a full run, the `outputs/` folder should contain files such as:

- `raw_musicbrainz.csv`
- `raw_listenbrainz.csv`
- `raw_acousticbrainz.csv`
- `music_dataset_processed.csv`
- `sql_*.csv`
- `eda_summary.md`
- `model_results.csv`
- `feature_importance.csv`
- `model_predictions.csv`
- `best_gbm_model.joblib`
- `best_gbm_params.json`
- `pca_summary.json`
- `model_summary.md`

## Dashboard Demo Content

The Streamlit dashboard includes:

- **Overview**
  Dataset size, genre coverage, collection coverage, and SQL-driven summaries.
- **EDA**
  Interactive target, missingness, genre, and trend views.
- **Modeling**
  Cross-validation vs. holdout results, feature importance, and prediction diagnostics.
- **Live Predictor**
  User-controlled inputs to estimate replayability for a hypothetical track.
- **Genre Explorer**
  Filtered view of genre trends, track duration relationships, and top artists.
