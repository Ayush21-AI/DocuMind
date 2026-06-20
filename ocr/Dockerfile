# Use a minimal and secure Python image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/

# Install dependencies in a virtual environment to avoid cluttering the global Python environment
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire ocr module, preserving `ocr/src/main/...`
COPY src /app/ocr/src

# CPU threading and memory optimization for containerized inference
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    ORT_DISABLE_CPU_AFFINITY=1 \
    OPENCV_FOR_THREADS_NUM=2 \
    MALLOC_ARENA_MAX=2

CMD ["uvicorn", "ocr.src.main.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
