# One image for the app (mlops-ui) and the worker (mlops-worker): the same code and the same kraken.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# git: the release clone of MaDoc; libgl1 / libglib2.0-0: OpenCV (augmentation)
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# PyTorch for CUDA 12.8 first (the RTX 50 series needs it); kraken 7.0.2 accepts torch 2.4 to 2.10
RUN pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# not root: the volumes are made with the rights of this user
RUN useradd -m -u 1000 mlops && mkdir -p /data /runs /release && chown mlops /data /runs /release
USER mlops

CMD ["python", "-m", "mlops.ui"]
