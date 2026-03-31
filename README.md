# ML Pipeline for Short-Term Rental Price Prediction in NYC

An end-to-end, reproducible ML pipeline for predicting short-term rental prices in New York City, built with MLflow, Weights & Biases, and Hydra.

![Python](https://img.shields.io/badge/python-3.13-blue)
![MLflow](https://img.shields.io/badge/mlflow-tracked-blue)
![W&B](https://img.shields.io/badge/W%26B-nyc__airbnb-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

## Project Links

- **GitHub:** [FabioCLima/build-ml-pipeline-for-short-term-rental-prices](https://github.com/FabioCLima/build-ml-pipeline-for-short-term-rental-prices)
- **W&B Project:** [fabio_lima07-mlops/nyc_airbnb](https://wandb.ai/fabio_lima07-mlops/nyc_airbnb)

---

## About This Project

A property management company receives new Airbnb listing data every week and needs to retrain a price estimation model at the same cadence. I built this project to demonstrate how MLOps principles solve that operational problem.

The core challenge is not just training a good model — it is making the entire workflow **reproducible**, **auditable**, and **retrainable on demand**. Every artifact (raw data, cleaned data, trained model) is versioned in Weights & Biases. Every run is tracked with MLflow. Every configuration value — from price bounds to hyperparameters — is managed by Hydra and can be overridden from the command line without touching the code.

### What I built

- A **6-step pipeline** orchestrated by MLflow Projects, where each step runs in its own isolated Conda environment.
- A **data validation layer** using pytest fixtures that pull live W&B artifacts, testing for schema consistency, geographic boundaries, price ranges, and neighbourhood distribution drift (KL divergence).
- A **scikit-learn inference pipeline** that bundles all preprocessing (ordinal encoding, one-hot encoding, zero imputation, date-to-days feature engineering, TF-IDF on listing names) with the Random Forest model into a single serializable object — ensuring identical transformations at training and inference time.
- A **hyperparameter sweep** using Hydra's multi-run feature to evaluate 15 configurations, with the best model promoted to the `prod` tag in W&B.
- A **retraining workflow**: running `mlflow run . -P hydra_options="etl.sample=sample2.csv"` on any tagged release ingests new data and produces a new versioned model without any code changes.

---

## Pipeline Architecture

```text
download → basic_cleaning → data_check → data_split → train_random_forest → test_regression_model
```

| Step | Description |
| ---- | ----------- |
| `download` | Fetches a raw CSV sample and uploads it to W&B as `sample.csv` |
| `basic_cleaning` | Filters price outliers, converts `last_review` to datetime, removes listings outside NYC boundaries |
| `data_check` | Runs pytest-based data validation (row count, price range, geographic boundaries, neighbourhood distribution) |
| `data_split` | Stratified split into `trainval_data.csv` and `test_data.csv` |
| `train_random_forest` | Trains a scikit-learn `Pipeline` (preprocessing + Random Forest), logs metrics, exports model to W&B |
| `test_regression_model` | Evaluates the model tagged as `prod` on the held-out test set |

## Model Performance

Best model: `random_forest_export:prod` (selected by lowest validation MAE)

| Set        | R²     | MAE    |
|------------|--------|--------|
| Validation | 0.5654 | $33.80 |
| Test       | 0.5808 | $33.29 |

Test performance is comparable to validation — no significant overfitting observed.

## Best Hyperparameters

Found via a Hydra multi-run sweep (15 combinations):

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

---

## Setup

### Requirements

- Ubuntu 22.04/24.04 or macOS
- Python 3.13
- Conda (Miniconda or Anaconda)

### Create the environment

```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

### Configure W&B

```bash
wandb login [your API key]
```

---

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

### Hyperparameter Sweep

```bash
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
```

### Run a Released Version on New Data

```bash
mlflow run https://github.com/FabioCLima/build-ml-pipeline-for-short-term-rental-prices.git \
  -v 1.0.1 \
  -P hydra_options="etl.sample='sample2.csv'"
```

---

## Key Design Decisions

- **No imputation in `basic_cleaning`** — missing values are handled inside the scikit-learn `Pipeline` in `train_random_forest` to prevent data leakage.
- **Geographic boundary filter** — listings outside NYC coordinates (longitude: -74.25 to -73.50, latitude: 40.5 to 41.2) are removed in `basic_cleaning`; this issue was caught by `test_proper_boundaries` when running on `sample2.csv` and fixed in `v1.0.1`.
- **Stratified split** — the split is stratified by `neighbourhood_group` to preserve borough distribution across train, validation, and test sets.
- **TF-IDF on listing name** — the `name` column contributes useful signal for price prediction through a lightweight NLP step.

## Releases

| Version | Description                                                                 |
| ------- | --------------------------------------------------------------------------- |
| v1.0.0  | Initial release with best hyperparameters                                   |
| v1.0.1  | Fix: removed listings outside NYC geographic boundaries in `basic_cleaning` |

## Future Improvements

- Extend the EDA with geospatial visualizations (price heatmaps by neighbourhood)
- Explore gradient boosting models (XGBoost, LightGBM) as alternatives to Random Forest
- Add a CI/CD workflow to automatically retrain the pipeline when new data arrives
- Implement model monitoring to detect data drift in production

## License

[License](LICENSE.txt)
