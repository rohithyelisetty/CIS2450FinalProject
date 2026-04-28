# Rubric Explanation

This document explains how the final codebase addresses the major rubric categories for the CIS 2450 final project. It focuses on the **codebase and dashboard deliverable**. Submission timing, live presentation, recording quality, and team logistics are outside what code alone can guarantee.

## 1. Project Proposal

This section is administrative rather than code-based. The repository cannot prove whether the original proposal was submitted on time or whether all teammates were added correctly. What the codebase does show is that the final scope is aligned with the original idea of predicting replayability from linked music datasets, even though the implementation has evolved beyond the original proposal.

## 2. Intermediate Check-In

This section is also partly administrative, but the current repository now contains all of the components needed to support a strong check-in:

- A defined target variable (`repeat_listens` and `log_repeat_listens`)
- An EDA script with more than 3 meaningful visuals
- A baseline model plus stronger follow-up models
- A modular plan of action reflected directly in the code layout

In other words, while the repo cannot retroactively prove the timing of the check-in, it now contains the exact evidence that those check-in questions asked for.

## 3. Difficulty

The project now goes beyond a standard single-table regression workflow. It includes several higher-difficulty concepts used in a way that is relevant to the problem.

### Concept 1: Record Linking / Entity Linking

The project links the same recording across **MusicBrainz**, **ListenBrainz**, and **AcousticBrainz** using the MusicBrainz recording ID (`mbid`). This is a genuine entity-linking step because the final dataset is not obtained from one source. It is built by combining metadata, listening behavior, and audio features from distinct APIs.

Why this counts:

- It is implemented directly in `data_collection.py` and `data_processing.py`
- It is necessary for the project objective because replayability cannot be modeled from one source alone
- It enriches the final dataset with new fields that materially affect analysis and modeling

### Concept 2: Feature Engineering

The preprocessing stage engineers multiple new features instead of only using raw columns. Examples include:

- `track_age`
- `artist_career_age`
- `genre_match_count`
- `has_audio_features`
- `audio_feature_missing_count`
- `career_x_duration_min`
- `tempo_x_dance`

Why this counts:

- The features are motivated by the EDA and domain logic
- Interaction terms help the non-linear models capture combined effects
- These new variables are used directly in the models and in the final interpretation

### Concept 3: PCA / Feature Reduction

The modeling stage now includes a dedicated **PCA + Ridge** pipeline. PCA is applied after preprocessing to reduce the transformed feature space into a smaller set of orthogonal components while preserving most of the variance.

Why this counts:

- PCA is implemented as an actual model path in `models.py`, not just mentioned in prose
- It is relevant to the project because the transformed feature matrix can become high-dimensional after one-hot encoding
- The PCA model is evaluated directly against the baseline and other models, so the dimensionality-reduction tradeoff is measurable

### Concept 4: Ensemble Models + Hyperparameter Tuning

The modeling stage includes both **Random Forest** and **Gradient Boosting**, which are ensemble models, and then tunes Gradient Boosting with **RandomizedSearchCV** using 5-fold cross-validation on the training split.

Why this counts:

- The models are implemented correctly inside sklearn pipelines
- Hyperparameter tuning is separated from the test set, which avoids leakage
- The tuned model is compared back against the baseline and untuned models rather than being reported in isolation

### Additional Depth Present

The codebase also includes:

- Feature importance analysis from both tree models and regularized linear models
- Class imbalance handling through a derived classification task with class weighting
- A second analytics layer with DuckDB SQL on top of the processed dataset

## 4a. EDA

The project includes a dedicated `eda.py` script that produces **8 visuals**, which exceeds the minimum request for 3 meaningful EDA visuals and is strong enough to support the final presentation requirement of 3-5 polished charts.

The EDA covers:

- Target distribution before and after log transform
- Replayability by genre
- Replayability across decades
- Tempo vs. danceability scatter
- Correlation heatmap
- Replayability by track duration bucket
- Missingness analysis
- Outlier analysis by genre

Why this is rubric-aligned:

- It provides context for the dataset and the target variable
- It studies distributions, relationships, missing values, and outliers
- The findings are written out in `outputs/eda_summary.md`, so the analysis is not just visual but also interpreted
- The EDA directly informs preprocessing choices and model selection

There is also a lightweight statistical layer in the EDA summary through tests such as Kruskal-Wallis and Spearman correlation, which strengthens the analytical justification.

## 4b. Data Pre-processing and Feature Engineering

The preprocessing work is one of the strongest parts of the final codebase.

It includes:

- Efficient multi-source API collection with local caching
- Deduplication of tracks that appear in multiple MusicBrainz genre searches
- Polars-based joins across sources
- Null handling without throwing away most of the data
- Explicit handling of partial AcousticBrainz coverage
- Outlier control through clipping and winsorization
- Creation of new engineered features
- Categorical handling for modeling through one-hot encoding inside the sklearn pipeline
- Derived imbalance label for a secondary classification exercise
- Numeric scaling where appropriate for linear models

Why this is rubric-aligned:

- The choices are informed by EDA findings
- Missingness is handled appropriately instead of ignored
- Sparse audio coverage is treated as optional enrichment rather than forcing the entire dataset to depend on it
- The preprocessing is modular, reproducible, and separated cleanly from modeling
- The repository demonstrates more than just loading data and moving directly into models

## 5a. Model Implementation

The final modeling pipeline is organized in `models.py` and includes:

- Linear Regression baseline
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Tuned Gradient Boosting Regressor

Why this is correctly implemented:

- There is a proper train/test split
- Missing values are imputed inside the sklearn pipeline
- Categorical variables are handled through one-hot encoding
- Scaling is applied inside the pipeline where it makes sense
- The test set is not used during tuning
- The baseline is present and serves as a comparison point
- The primary model path remains valid even when AcousticBrainz only covers a subset of tracks

This directly addresses the rubric requirement for a baseline model plus additional models with clear justification.

## 5b. Model Assessment and Hyperparameter Tuning

The project includes a dedicated evaluation framework rather than reporting a single score.

It uses:

- **RMSE**
- **MAE**
- **R^2**
- **5-fold cross-validation**
- **PCA-based dimensionality reduction** as a directly evaluated model variant
- **RandomizedSearchCV** for tuned Gradient Boosting

It also includes a derived imbalance-handling demo for classification using:

- Accuracy
- F1
- ROC-AUC

Why this is rubric-aligned:

- Multiple evaluation metrics are used correctly for the task
- Hyperparameter tuning is explicit and methodical
- Tuned results are compared to untuned and baseline models
- Outputs are persisted to CSV, JSON, plots, and markdown summaries for transparency

## 7. Code Quality / Readability

The codebase is now explicitly modular and maps closely to the expected project lifecycle:

- `config.py`
- `data_collection.py`
- `data_processing.py`
- `eda.py`
- `models.py`
- `dashboard.py`
- `README.md`
- `RUBRIC_EXPLANATION.md`

Why this is rubric-aligned:

- Each file has a clear purpose
- Reusable constants live in one place
- Output artifacts are standardized in `outputs/`
- The code is broken into functions instead of one monolithic notebook
- The README gives a clean run order for a TA or teammate

## 8. Application of Course Topics

This project applies several course topics in a way that is connected to the actual objective. At least the following are clearly represented:

1. **Polars**
   Used for data cleaning, joins, transformations, and feature engineering.
2. **SQL**
   Used through DuckDB analytical queries in `data_processing.py`.
3. **Joins**
   Core to combining MusicBrainz, ListenBrainz, and AcousticBrainz.
4. **Relational Database / Analytical SQL Layer**
   The processed dataset is queried as a relational table through DuckDB.
5. **Record Linking**
   Recordings are linked across APIs using `mbid`.
6. **Supervised Learning Models**
   Multiple regression models are trained and evaluated.
7. **PCA**
   Dimensionality reduction is implemented and evaluated through the PCA + Ridge model path.
8. **Different Methods of Hyperparameter Tuning**
   Manual baseline comparison plus randomized hyperparameter search are both used.

Because these topics are all directly tied to the replayability objective, they are not just included superficially.

## 9. Quality of Dashboard Demo

The final dashboard is implemented in `dashboard.py` as a multi-tab Streamlit app with:

- **Overview**
- **EDA**
- **Modeling**
- **Live Predictor**
- **Genre Explorer**

Why this is rubric-aligned:

- It is interactive rather than static
- It covers both EDA and modeling results
- It includes a live prediction workflow, which makes the modeling results feel actionable
- It exposes SQL-style summaries, feature importance, and filtered exploration in a single interface
- It is polished enough to support a dashboard demo rather than just displaying raw tables

## 10. Final Deliverables / Codebase Expectations

The project structure is consistent with the final deliverable instructions:

- Modular codebase
- Local storage of collected data
- Interactive dashboard
- Readable scripts rather than a single tangled notebook
- Documentation for running and explaining the deliverable

## 11. Presentation and Recording

Not covered here by design, per the request.

## 12. Other Penalties

This section is administrative and cannot be guaranteed by code alone. The repository cannot prove on-time submission, contribution balance, or final Gradescope logistics. What it can do is minimize preventable deductions by being organized, complete, readable, and runnable.
