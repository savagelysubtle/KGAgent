FROM python:3.11-slim

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .[dev]

# Install Playwright browsers
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy application code
COPY src/ ./src/

# Create directories
RUN mkdir -p /app/storage/screenshots /app/storage/pdfs /app/cache

EXPOSE 8000

CMD ["uvicorn", "kg_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
