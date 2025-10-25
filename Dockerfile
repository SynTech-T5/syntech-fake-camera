# -------------------------------------------------------
# ✅ GPU Base Image with CUDA 12.2 (NVIDIA Official)
# -------------------------------------------------------
FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

# -------------------------------------------------------
# 📦 Install system dependencies & FFmpeg 6.1 with NVENC/NVDEC
# -------------------------------------------------------
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntuhandbook1/apps -y && \
    apt-get update && \
    apt-get install -y \
    ffmpeg \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# 🏗️ Working directory
# -------------------------------------------------------
WORKDIR /app

# -------------------------------------------------------
# 📄 Install Python dependencies
# -------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# -------------------------------------------------------
# 🧠 Copy project files
# -------------------------------------------------------
COPY . /app

# -------------------------------------------------------
# 🧩 Verify FFmpeg & NVENC Support (for debugging)
# -------------------------------------------------------
RUN echo "🔍 Checking FFmpeg + NVENC availability..." && \
    ffmpeg -version && \
    ffmpeg -encoders | grep nvenc || true && \
    ffmpeg -decoders | grep cuvid || true

# -------------------------------------------------------
# 🚀 Default command
# -------------------------------------------------------
CMD ["python3", "model_ffmpeg_cuda.py"]
