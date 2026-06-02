FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    STOCKLAB_TORCH_THREADS=4

# CPU-only PyTorch first (no CUDA — matches the no-GPU design and keeps the
# image small), then the remaining dependencies.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Production WSGI server (pure-Python, cross-platform). Redis must be reachable
# via STOCKLAB_REDIS_URL (see docker-compose.yml).
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--threads=6", "--call", "stocklab.web:create_app"]
