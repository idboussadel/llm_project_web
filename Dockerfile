# Multi-stage build for faster Railway deployment
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch first (CPU version is smaller and faster)
# This allows better caching
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other ML packages
RUN pip install --no-cache-dir \
    transformers>=4.30.0 \
    peft>=0.6.0 \
    accelerate>=0.20.0 \
    safetensors>=0.4.0 \
    pytorch-forecasting>=1.0.0 \
    pytorch-lightning>=2.0.0

# Install remaining requirements (without PyTorch - already installed)
COPY requirements-optimized.txt .
RUN pip install --no-cache-dir -r requirements-optimized.txt

# Final stage - minimal runtime
FROM python:3.11-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgfortran5 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Expose port (default to 8080 if PORT not set)
EXPOSE 8080

# Use gunicorn with shell form to expand PORT variable
CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:${PORT:-8080} --timeout 120 wsgi:app"]

