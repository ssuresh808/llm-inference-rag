# syntax=docker/dockerfile:1

# --- builder: resolve dependencies into a venv with uv ---------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# INCLUDE_LOCAL_EMBEDDINGS=true also installs the heavy `local` group (torch +
# bge-large) for full-local `docker compose`. Default false = slim cloud image
# (hosted Voyage embeddings, no torch) that fits a 512MB instance (ADR-019).
ARG INCLUDE_LOCAL_EMBEDDINGS=false

# Install runtime deps first (cached layer); exclude dev deps for a lean image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
    $(if [ "$INCLUDE_LOCAL_EMBEDDINGS" = "true" ]; then echo "--group local"; fi)

# --- runtime: slim image with just the venv + source ----------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser

# Bring in the resolved virtualenv and the application source only.
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser README.md ./

USER appuser
EXPOSE 8000 8501

# Default to the API. docker-compose overrides the command for the frontend.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
