# Modeling Summary

## Evaluation Setup

- Train/test split: 80/20
- Cross-validation: 5-fold on the training split
- Metrics: RMSE, MAE, and R^2 for regression; accuracy/F1/ROC-AUC for the imbalance demo
- Missing values handled with in-pipeline imputation to avoid leakage
- AcousticBrainz coverage in the processed dataset: 42.8%
- Primary model feature set: metadata-first; sparse AcousticBrainz features were treated as optional enrichment
- PCA model retained 16 components and explained 95.2% of variance

## Best Model

- Best test model: Gradient Boosting (tuned)
- Test R^2: 0.3118
- Test RMSE: 2.3716
- Test MAE: 1.8755

## Tuned Gradient Boosting Parameters

```json
{
  "best_score": 0.32105807853687646,
  "best_params": {
    "model__subsample": 0.7,
    "model__n_estimators": 500,
    "model__min_samples_leaf": 20,
    "model__max_depth": 5,
    "model__learning_rate": 0.05
  }
}
```

## Top Learned Signals

- cat__genre_classical: 0.19702
- cat__genre_indie: 0.13296
- num__duration_sec: 0.13085
- num__track_age: 0.09461
- num__release_year: 0.08935
- cat__genre_r&b: 0.06215
- num__audio_feature_missing_count: 0.04854
- cat__genre_punk: 0.04399
- num__has_audio_features: 0.03866
- cat__genre_metal: 0.02261

## Imbalance Demo

- LogReg_no_class_weight: accuracy=0.766, f1=0.327, roc_auc=0.737
- LogReg_balanced: accuracy=0.691, f1=0.506, roc_auc=0.737