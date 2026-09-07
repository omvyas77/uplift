# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
# EconML publishes no linux/arm64 wheel, so on Apple Silicon (and any arm64
# runner) it builds from source and needs a C/C++ toolchain. This lives in the
# BUILDER stage only - the runtime image below never sees a compiler.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
# Dependencies first, without the project, so a source edit does not bust the
# dependency layer.
RUN uv sync --frozen --no-dev --extra causal --extra serve --no-install-project
COPY src/ src/
RUN uv sync --frozen --no-dev --extra causal --extra serve

FROM python:3.12-slim AS runtime
# LightGBM needs libgomp at runtime; the slim image does not ship it, and the
# failure mode is an unhelpful dlopen error at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/ src/
COPY --chown=app:app pyproject.toml README.md ./
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "uplift.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
