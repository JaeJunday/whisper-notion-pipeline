FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Whisper
RUN pip install --no-cache-dir openai-whisper

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p input output logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu

# Default command
CMD ["python", "run.py", "--help"]