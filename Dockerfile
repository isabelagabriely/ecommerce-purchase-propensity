FROM python:3.13-slim

ARG POETRY_VERSION=2.5.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock README.md ./
COPY src ./src

RUN poetry install --only main --no-interaction --no-ansi \
    && rm -rf "${POETRY_CACHE_DIR}"

COPY configs ./configs
COPY params.yaml ./
COPY dvc.yaml dvc.lock ./
COPY .dvc ./.dvc
COPY .dvcignore ./

RUN poetry run dvc config core.no_scm true

RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /mlflow \
    && chown -R app:app /app /mlflow

USER app

CMD ["poetry", "run", "dvc", "repro"]
