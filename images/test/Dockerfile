FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git ca-certificates wget curl && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY ../../requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir ctranslate2==4.3.1

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb 
RUN dpkg -i cuda-keyring_1.0-1_all.deb 
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends libcudnn8 libcudnn8-dev

COPY ../../ /app
EXPOSE 6006
CMD ["bash"]