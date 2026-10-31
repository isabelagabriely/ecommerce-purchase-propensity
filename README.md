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