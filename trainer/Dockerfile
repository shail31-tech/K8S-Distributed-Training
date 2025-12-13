FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------
# Install Python dependencies
# ----------------------------

# Copy requirements file first (better caching)
COPY requirements.txt /app/

# Install dependencies including PyTorch CPU
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ----------------------------
# Copy application source
# ----------------------------
COPY train.py test.py /app/

# Ensure model directory exists
RUN mkdir -p /model

# ----------------------------
# Default command: start training
# ----------------------------
CMD ["python", "/app/train.py"]
