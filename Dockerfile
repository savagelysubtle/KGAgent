# ============================================================================
# KG Agent Backend - Multi-stage Dockerfile
# ============================================================================
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ============================================================================
# Builder stage - Install dependencies
# ============================================================================
FROM base AS builder

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./
COPY README.md ./

# Create virtual environment and install dependencies
RUN uv sync --frozen --no-dev

# ============================================================================
# Runtime stage - Final image
# ============================================================================
FROM base AS runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    # Required for Playwright/Crawl4AI
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Install Playwright browsers (Chromium only for smaller image)
RUN playwright install chromium && \
    playwright install-deps chromium

# Copy application code
COPY src/ ./src/
COPY main.py ./

# Create directories for data persistence
RUN mkdir -p /app/storage/screenshots /app/storage/pdfs /app/cache \
    /app/data/chroma_db /app/data/chunks /app/data/parsed /app/data/raw \
    /app/models/embeddings /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

EXPOSE 8000

# Default command - use uvicorn with multiple workers
CMD ["uvicorn", "src.kg_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
