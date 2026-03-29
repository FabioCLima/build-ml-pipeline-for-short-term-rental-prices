# ML Pipeline for Short-Term Rental Price Prediction in NYC

End-to-end reproducible ML pipeline for predicting short-term rental prices in New York City, built with MLflow, Weights & Biases, and Hydra.

## Project Links

- **GitHub Repository:** [FabioCLima/build-ml-pipeline-for-short-term-rental-prices](https://github.com/FabioCLima/build-ml-pipeline-for-short-term-rental-prices)
- **W&B Project:** [fabio_lima07-mlops/nyc_airbnb](https://wandb.ai/fabio_lima07-mlops/nyc_airbnb)

## Overview

A property management company receives new Airbnb listing data weekly and needs to retrain a price estimation model with the same cadence. This project implements a fully automated, reproducible pipeline that handles data ingestion, cleaning, validation, training, hyperparameter optimization and model testing — all tracked in W&B.

## Pipeline Architecture

```text
download → basic_cleaning → data_check → data_split → train_random_forest → test_regression_model
```

- **`download`** — fetches raw CSV sample and uploads to W&B as `sample.csv`
- **`basic_cleaning`** — filters price outliers, converts `last_review` to datetime, removes listings outside NYC boundaries
- **`data_check`** — runs pytest-based data validation (row count, price range, geo boundaries, neighbourhood distribution)
- **`data_split`** — stratified split into `trainval_data.csv` and `test_data.csv`
- **`train_random_forest`** — trains a sklearn Pipeline (preprocessing + RandomForest), logs metrics and exports model to W&B
- **`test_regression_model`** — evaluates the `prod`-tagged model against the held-out test set

## Model Performance

Best model: `random_forest_export:prod` (selected by lowest MAE on validation set)

| Set | R² | MAE |
|-----|----|-----|
| Validation | 0.5654 | $33.80 |
| Test | 0.5808 | $33.29 |

No overfitting — test performance is comparable to validation performance.

## Best Hyperparameters

Found via Hydra multi-run sweep (15 combinations):

```yaml
modeling:
  max_tfidf_features: 30
  random_forest:
    max_features: 0.33
    n_estimators: 100
    max_depth: 15
    min_samples_split: 4
    min_samples_leaf: 3
```

## Setup

### Requirements

- Ubuntu 22.04/24.04 or macOS
- Python 3.13
- Conda (Miniconda or Anaconda)

### Create environment

```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

### Configure W&B

```bash
wandb login [your API key]
```

## Running the Pipeline

Run the full pipeline:

```bash
mlflow run .
```

Run individual steps:

```bash
mlflow run . -P steps=basic_cleaning
mlflow run . -P steps=data_check
mlflow run . -P steps=train_random_forest
```

Run only the test step (requires a model tagged as `prod` in W&B):

```bash
mlflow run . -P steps=test_regression_model
```

### Hyperparameter sweep

```bash
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
```

### Run a released version on new data

```bash
mlflow run https://github.com/FabioCLima/build-ml-pipeline-for-short-term-rental-prices.git \
  -v 1.0.1 \
  -P hydra_options="etl.sample='sample2.csv'"
```

## Key Design Decisions

- **No imputation in `basic_cleaning`** — missing values are handled inside the sklearn `Pipeline` in `train_random_forest` to prevent data leakage
- **Geo boundary filter** — listings outside NYC coordinates (lon: -74.25 to -73.50, lat: 40.5 to 41.2) are removed in `basic_cleaning`; caught by `test_proper_boundaries` when running on `sample2.csv` (v1.0.0 → v1.0.1 fix)
- **Stratified split** by `neighbourhood_group` to preserve borough distribution across train/val/test
- **TF-IDF on listing name** — the `name` column contributes meaningfully to price prediction via a minimal NLP step

## Releases

- **v1.0.0** — Initial release with best hyperparameters
- **v1.0.1** — Fix: remove listings outside NYC geographic boundaries in `basic_cleaning`

## Future Improvements

- Extended EDA with geospatial visualizations (price heatmap by neighbourhood)
- Explore gradient boosting models (XGBoost, LightGBM) as alternatives to Random Forest
- Add CI/CD workflow to automatically retrain when new data arrives
- Implement model monitoring to detect data drift in production

## License

[License](LICENSE.txt)
