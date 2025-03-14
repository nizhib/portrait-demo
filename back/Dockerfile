FROM python:3.12-slim-bookworm AS python

LABEL maintainer="Evgeny Nizhibitsky <nizhib@github.com>"

WORKDIR /app

# Build the deps

FROM python AS build

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/

WORKDIR /app
RUN uv sync --frozen

# Build the app

FROM python AS runtime

ARG CHECKPOINTS_ROOT=/root/.cache/torch/hub/checkpoints/
ARG UNET_URL=https://github.com/nizhib/portrait-demo/releases/download/v0.3/unet_resnext50-46a04131.pth
ADD ${UNET_URL} ${CHECKPOINTS_ROOT}

COPY --from=build /app /app
COPY --from=build /bin /bin
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD ["uv", "run", "uvicorn", "app:app", "--host=0.0.0.0", "--port=5000", "--log-level=info"]
