# Music Replayability Analysis

Predicts **music replayability** by linking data from multiple music APIs and a large-scale lyrics corpus.

**Authors:** Chinmay Govind · Rohith Yelisetty

## Project Structure

```
config.py              shared constants, paths, feature lists
data_collection.py     fetches MusicBrainz / ListenBrainz / AcousticBrainz APIs
data_processing.py     Polars joins, feature engineering, DuckDB SQL analytics
eda.py                 8 EDA plots → outputs/eda_summary.md
models.py              7 regression models + classification, PCA, tuning → outputs/model_summary.md
lyrics_analysis.py     loads Kaggle lyrics, NLP features, LDA topics, 11 plots, prediction model
dashboard.py           Streamlit dashboard (6 tabs)
```
## Data Sources

| Source | What it provides |
|---|---|
| MusicBrainz | Recording metadata: title, artist, genre, release type, country |
| ListenBrainz | Popularity signals: total listens, unique listeners, repeat listens |
| AcousticBrainz | Optional low-level audio features (BPM, key, energy, danceability) |
| Kaggle 5M Lyrics | NLP features: sentiment, vocabulary richness, LDA topics, repetitiveness |

**Prediction target:** `log_repeat_listens = log(1 + total_listens - unique_listeners)`


## Setup

```bash
pip install -r requirements.txt
```

For lyrics analysis, download the Kaggle dataset and place it at `lyrics/ds2.csv`:
```
https://www.kaggle.com/datasets/nikhilnayak123/5-million-song-lyrics-dataset
```

## Run Order

```bash
python data_collection.py --per-genre 5000   # fetch ~50k songs across 20 genres
python data_processing.py                    # join, clean, engineer features
python eda.py                                # EDA plots
python models.py                             # train regression + classification models
python lyrics_analysis.py                   # NLP analysis (uses pickle cache after first run)
streamlit run dashboard.py                  # launch dashboard
```

API responses are cached in `cache/` — subsequent runs skip re-fetching. The Kaggle CSV scan (9.2 GB) is also cached after the first run of `lyrics_analysis.py`.

## Dashboard Tabs

| Tab | Contents |
|---|---|
| Command Center | Dataset health cards, genre distribution, decade replayability trend, top-25 artist leaderboard, EDA + model summaries |
| Genre Lab | Interactive genre/decade filter with trend lines, box plots, duration scatter, artist leaderboard |
| Model Studio | CV vs holdout R² chart, RMSE/MAE comparison, feature importance, tuned GBM params, actual vs predicted scatter |
| Prediction Console | Live scoring form — enter a track profile and get a predicted log replay count + dataset percentile |
| Lyrics Analysis | 11 NLP/model plots, lyrics prediction model results table, word frequency comparison, full summary report |
| Song Explorer | Typeahead search over top 10k songs; per-song sentiment, top words, vocabulary metrics, lyrics viewer |

## Key Results

- **Best regression model:** Tuned GBM — holdout R² ≈ 0.21, RMSE ≈ 2.26
- **Lyrics vs metadata:** Lyrics alone explain ~4% of replay variance; metadata GBM reaches ~20%; combining both gives R² ≈ 0.208
- **23,641 songs** matched to lyrics via exact + fuzzy artist/title matching
- **LDA topics:** 8 coherent themes (multilingual, romantic, hip-hop, spiritual, rootsy, dance/rock, conversational, temporal)

## Main Output Files

```
outputs/
  music_dataset_processed.csv     cleaned + joined dataset (~55k rows)
  model_results.csv               CV and holdout scores for all models
  feature_importance.csv          GBM feature importances
  best_gbm_model.joblib           saved tuned GBM
  lyrics_features.csv             per-song NLP feature matrix
  lyrics_model_results.csv        lyrics vs metadata vs combined model scores
  lyrics_summary.md               full lyrics analysis narrative
  eda_summary.md                  EDA narrative
  model_summary.md                modeling narrative
  lyrics1_*.png ... lyrics11_*.png  NLP + model visualizations
```
