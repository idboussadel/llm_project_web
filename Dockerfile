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

# Expose port
EXPOSE 8080

# OPTIMIZED: Use --preload to load models once, then fork workers
# This reduces memory usage and startup time via copy-on-write
# PyTorch thread limits set in app/__init__.py prevent CPU oversubscription
# Using gevent for async I/O (API calls) while keeping 1 worker for model safety
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

CMD ["gunicorn", "--preload", "-w", "1", "-b", "0.0.0.0:8080", "--timeout", "300", "--worker-class", "gevent", "--worker-connections", "1000", "wsgi:app"]

