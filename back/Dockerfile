FROM python:3.10-slim as python

LABEL maintainer="Evgeny Nizhibitsky <nizhib@github.com>"

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=60

WORKDIR /app

# Build the deps

FROM python as build

ENV POETRY_VERSION=1.3.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN pip install "poetry==$POETRY_VERSION"
COPY pyproject.toml poetry.lock /app/
RUN poetry install --no-interaction --no-ansi -vvv

# Build the app

FROM python as runtime

ARG CHECKPOINTS_ROOT=/root/.cache/torch/hub/checkpoints/
ARG UNET_URL=https://github.com/nizhib/portrait-demo/releases/download/v0.3/unet_resnext50-46a04131.pth
ADD ${UNET_URL} ${CHECKPOINTS_ROOT}

COPY --from=build /app /app
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=5000", "--log-level=debug"]
