# Components

Pre-built reusable MLflow components used in the pipeline.

## Project Links

- **W&B Project:** [fabio_lima07-mlops/nyc_airbnb](https://wandb.ai/fabio_lima07-mlops/nyc_airbnb)
- **GitHub:** [FabioCLima/build-ml-pipeline-for-short-term-rental-prices](https://github.com/FabioCLima/build-ml-pipeline-for-short-term-rental-prices)

## Requirements

Conda (Miniconda or Anaconda) and MLflow must be installed:

```bash
conda install mlflow=3.3.2
```

## Available Components

- `get_data` — downloads the raw data sample from the source and uploads it to W&B
- `train_val_test_split` — splits data into train/validation and test sets
- `test_regression_model` — evaluates the production model against the held-out test set

Each component follows the MLflow project structure: `conda.yml`, `MLproject`, `run.py`.
