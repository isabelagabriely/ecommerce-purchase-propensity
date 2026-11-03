# E-commerce Purchase Propensity

Machine Learning Engineering project for predicting the purchase propensity of e-commerce users based on their browsing behavior.

## Business Problem

An e-commerce company needs a predictive system capable of identifying whether a browsing session is likely to result in a purchase.

The project uses the **Online Shoppers Purchasing Intention Dataset**, with `Revenue` as the binary target variable.

## Project Goals

The main goal is to build a reproducible Machine Learning pipeline covering:

- data ingestion and preprocessing;
- classical classification models with Scikit-Learn;
- data and pipeline versioning with DVC;
- experiment tracking with MLflow;
- model registration with MLflow Model Registry;
- dependency management with Poetry;
- containerized execution with Docker;
- automated tests and code quality checks.

## Planned Models

The initial candidate models are:

- Logistic Regression;
- Random Forest.

Model selection will be based on validation metrics appropriate for the binary classification problem.

## Project Structure

```text
ecommerce-purchase-propensity/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   └── ecommerce_purchase_propensity/
│       ├── data/
│       └── modeling/
├── tests/
├── .dockerignore
├── .env.example
├── .gitignore
└── README.md
```

## Development Environment

The project uses Python 3.13 and Poetry for dependency management.

Install all project and development dependencies with:

```bash
poetry install
```

Run commands inside the managed environment with:

```bash
poetry run <command>
```

Production and development dependencies are maintained separately in pyproject.toml, while poetry.lock ensures reproducible installations.

## Code Quality

The project uses Ruff for linting and formatting and Pytest for automated tests.

Run linting with:

```bash
poetry run ruff check .
```

Check formatting with:

```bash
poetry run ruff format --check .
```

Run tests with:

```bash
poetry run pytest
```

Application logging is configured through configs/logging.yaml, with the log level controlled by the LOG_LEVEL environment variable.

## Dataset

This project uses the **Online Shoppers Purchasing Intention Dataset**
from the UCI Machine Learning Repository.

The dataset contains 12,330 browsing sessions and 18 attributes.
The binary target variable is `Revenue`, which indicates whether the
session resulted in a purchase.

The raw dataset can be downloaded automatically with:

```bash
poetry run python -m ecommerce_purchase_propensity.data.download
```

To force a new download:

```bash
poetry run python -m ecommerce_purchase_propensity.data.download --force
```

The download command validates the expected dataset schema and number
of records before the data is used by the pipeline.

#### Dataset attribution
Sakar, C. & Kastro, Y. (2018).
Online Shoppers Purchasing Intention Dataset.
UCI Machine Learning Repository.

DOI: 10.24432/C5F88Q

## DVC Pipeline

The project uses DVC for data versioning and pipeline reproducibility.

Initialize dependencies and reproduce the current pipeline with:

```bash
poetry run dvc repro
```

The current pipeline starts with automated dataset ingestion:

```text
download
   ↓
data/raw/online_shoppers_intention.csv
```

Inspect the pipeline graph with:

```bash
poetry run dvc dag
```

Check whether pipeline dependencies or outputs have changed with:

```bash
poetry run dvc status
```

DVC stores generated data in its local cache while Git tracks the
pipeline definition and lock metadata through `dvc.yaml` and `dvc.lock`.

As additional stages are implemented, the pipeline will be extended
with preprocessing, training, evaluation, and model registration.

## Data Preprocessing

The raw dataset is validated and split into reproducible training and
test datasets.

The preprocessing stage performs:

- schema validation;
- binary target normalization;
- stratified train/test splitting;
- fixed random seed for reproducibility.

Pipeline parameters are stored in `params.yaml`.

Current configuration:

```yaml
data:
  test_size: 0.20
  random_state: 42
```

Run the complete data pipeline with:

```bash
poetry run dvc repro
```

The current DVC pipeline is:

```text
download
   ↓
preprocess
   ├── data/processed/train.csv
   └── data/processed/test.csv
```

Feature scaling and categorical encoding are intentionally not applied
before the train/test split. These transformations will be fitted only
on training data inside the Scikit-Learn modeling pipeline to prevent
data leakage.

## Model Training

The project currently compares two classical Scikit-Learn classifiers:

- Logistic Regression;
- Random Forest.

Both models use the same preprocessing architecture through a
Scikit-Learn `Pipeline`.

Numeric features are processed with median imputation and
standardization. Categorical features use most-frequent imputation
and one-hot encoding.

```text
features
   ↓
ColumnTransformer
   ├── numeric
   │     ├── median imputation
   │     └── StandardScaler
   │
   └── categorical
         ├── most-frequent imputation
         └── OneHotEncoder
                 ↓
             classifier
```

Model selection uses 5-fold stratified cross-validation on the
training dataset only.

The primary selection metric is **Average Precision**, while ROC-AUC,
F1, precision, recall, and accuracy are also tracked.

The held-out test dataset is used only after the champion model has
been selected.

Run the complete pipeline with:

```bash
poetry run dvc repro
```

View final DVC metrics with:

```bash
poetry run dvc metrics show
```

Cross-validation results are stored in:

```text
reports/model_comparison.csv
```

Final held-out test metrics are stored in:

```text
reports/test_metrics.json
```

## MLflow Experiment Tracking

Model experiments are tracked with MLflow.

Start the local MLflow Tracking Server from the project root:

```bash
poetry run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

The MLflow UI is available at:

```text
http://127.0.0.1:5000
```

The training pipeline creates one MLflow run for each candidate model.

Each run records:

- model hyperparameters;
- cross-validation configuration;
- cross-validation metrics;
- fitted Scikit-Learn pipeline;
- model signature;
- input example;
- DVC lock metadata;
- pipeline parameters.

The model with the highest cross-validation Average Precision is tagged
as the champion. Only the champion is evaluated on the held-out test set,
and its final test metrics are added to the corresponding MLflow run.

Run tracked training through DVC with:

```bash
poetry run dvc repro train
```

MLflow metadata is stored locally in `mlflow.db`, while model artifacts
are stored under `mlartifacts/`. Both are excluded from Git.

## MLflow Model Registry

After model comparison and held-out evaluation, the champion model is registered in the MLflow Model Registry.

The registered model name and alias are configured through environment variables:

```text
MLFLOW_REGISTERED_MODEL_NAME=purchase-propensity-classifier
MLFLOW_MODEL_ALIAS=champion
```

The current champion can be referenced independently of its numeric model version:

```text
models:/purchase-propensity-classifier@champion
```

The registration process locates the most recent champion MLflow run, registers its logged Scikit-Learn pipeline, attaches evaluation metadata to the model version, assigns the `champion` alias, and validates that the alias resolves to the registered version.

After running a new training experiment, register the resulting champion with:

```bash
poetry run dvc repro --force --single-item register
```

The `--single-item` option prevents model registration from implicitly reproducing the training stage. The `--force` option is used because registration changes external MLflow Registry state that is not represented solely by DVC file outputs.

The current DVC pipeline is:

```text
download
   ↓
preprocess
   ↓
train
   ↓
register
```

For a new modeling experiment, run:

```bash
poetry run dvc repro train
```

Then promote the resulting champion separately:

```bash
poetry run dvc repro --force --single-item register
```

## Docker

The complete Machine Learning pipeline can run in containers without relying on the host Python environment.

Docker Compose defines two services:

```text
mlflow
   └── MLflow Tracking Server and Model Registry

pipeline
   └── DVC pipeline execution
       ├── download
       ├── preprocess
       ├── train
       └── register
```

Build the Docker images with:

```bash
docker compose build
```

Start the MLflow Tracking Server:

```bash
docker compose up -d mlflow
```

The MLflow UI is available at:

```text
http://localhost:5000
```

Check the container status with:

```bash
docker compose ps
```

Run the complete Machine Learning pipeline inside Docker:

```bash
docker compose run --rm pipeline
```

The pipeline container connects to the MLflow service through the internal Docker network using:

```text
http://mlflow:5000
```

The Docker image runs DVC with SCM integration disabled because the `.git` directory is intentionally excluded from the image. Git and DVC integration remain enabled normally in the development environment.

MLflow metadata, experiments, model artifacts, and Model Registry state are persisted in the `mlflow-data` Docker volume. This allows the MLflow state to survive container recreation.

Stop the Docker environment with:

```bash
docker compose down
```

This preserves the MLflow volume.

To completely reset the Docker environment, including MLflow experiments, registered models, and artifacts, run:

```bash
docker compose down -v
```

The `-v` option removes the persistent MLflow volume and should only be used when a complete reset is intended.
