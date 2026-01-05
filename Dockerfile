FROM node:18-alpine AS build-step
WORKDIR /build

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ .
RUN npm run build

FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    ca-certificates \
    cmake \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# --- whisper.cpp (CLI) ---
# Build the CLI and download a small English model.
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git /opt/whisper.cpp \
    && make -C /opt/whisper.cpp -j4 \
    && ln -sf /opt/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli \
    && mkdir -p /models \
    && curl -L -o /models/ggml-tiny.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin

ENV WHISPERCPP_BIN=whisper-cli
ENV WHISPERCPP_MODEL_DIR=/models

COPY requirements.txt .
RUN pip install --no-cache-dir --use-pep517 -r requirements.txt

COPY . .

COPY --from=build-step /build/dist ./client

ENV FLASK_STATIC_FOLDER=/app/client

EXPOSE 8000

# Use threads so one slow ASR request doesn't block everything.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--worker-class", "gthread", "--timeout", "360", "--keep-alive", "5", "--bind", "0.0.0.0:8000", "app:app"]