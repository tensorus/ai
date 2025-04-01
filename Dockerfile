FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for storage
RUN mkdir -p /data/tensorus

# Environment variables
ENV PYTHONPATH=/app
ENV TENSORUS_STORAGE_PATH=/data/tensorus

# Expose ports for API and Dashboard
EXPOSE 8000 8501

# Default command to run the API
CMD ["python", "run_api.py", "--host", "0.0.0.0", "--port", "8000"] 