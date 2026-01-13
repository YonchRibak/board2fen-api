# --- Stage 1: Build Stage ---
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /build

# Install minimal compilers for pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies to /install to keep the final image clean
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime Stage ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install ONLY the C-libraries OpenCV needs to run (without the GUI bloat)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the pre-installed dependencies from the builder
COPY --from=builder /install /usr/local

# Copy application code into /app/api/ to preserve import structure
# (code uses "from api.config import settings", etc.)
COPY . /app/api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
