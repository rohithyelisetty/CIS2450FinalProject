# EDA Summary

## Dataset Context

- Rows analyzed: 55,579
- Columns available: 39
- Target variable: `log_repeat_listens` derived from `repeat_listens`
- Rows with any AcousticBrainz coverage: 23,794 (42.8%)

## Findings

### Target Distribution

The raw target is extremely right-skewed, with a small number of viral tracks dominating the replay count. The log transform compresses the heavy tail and creates a far more stable regression target for downstream modeling.

### Genre Ranking

Replay behavior differs meaningfully across genres, which justifies keeping genre as a primary predictive feature and also supports the business story that music replayability is partly audience-segment dependent.

### Decade Trend

Replayability varies by release era, which suggests that track age and release context contain signal. This motivated the engineered `release_decade` and `track_age` features in the preprocessing pipeline.

### Tempo and Danceability

This plot is drawn only on the subset of tracks with AcousticBrainz coverage. It is useful for enrichment and storytelling, but the sparse coverage means audio features should be treated as optional rather than assumed for every record.

### Correlation Structure

Most single features correlate only weakly with the target, which suggests that replayability is driven by combinations of metadata and audio signals rather than a single dominant variable.

### Duration Buckets

Duration matters in an interpretable way: mid-length songs tend to have the best replay outcomes, while very short interludes and very long tracks lag behind.

### Missingness

Missingness is overwhelmingly concentrated in the AcousticBrainz fields. That finding changed the downstream strategy: the main modeling path is metadata-first, while audio variables are treated as partial enrichment instead of mandatory inputs.

### Outliers by Genre

All genres retain upper-tail outliers even after the log transform, which is why the project compares robust tree-based models against ordinary linear regression instead of relying on a single modeling family.

## Statistical Checks

Kruskal-Wallis across the five largest genres: H=6395.09, p=0.000e+00. This supports the claim that replay distributions differ significantly by genre.

Spearman correlation between duration and log replay target: rho=0.042, p=1.874e-22. This quantifies the directional relationship seen in the duration-bucket chart.
