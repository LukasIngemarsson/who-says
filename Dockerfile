FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --use-pep517 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:8000", "app:app"]

# CMD ["python", "main.py"]
