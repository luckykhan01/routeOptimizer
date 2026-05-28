FROM python:3.10-slim

WORKDIR /app

# system dependencies required for geographical libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# project files
COPY . .

# environment
ENV PYTHONPATH=/app
