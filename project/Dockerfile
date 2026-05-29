# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app

WORKDIR $APP_HOME

# Install system dependencies
# libgomp1: required for LightGBM (OpenMP runtime)
# build-essential, gcc: sometimes needed for building wheels during pip install
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, configs, and model artifacts
COPY src/ ./src/
COPY configs/ ./configs/
COPY artifacts/modeling/models/ ./artifacts/modeling/models/

# Create non-root user for security isolation
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app $APP_HOME
USER app

# Expose application port
EXPOSE 8000

# Healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI service
CMD ["uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8000"]