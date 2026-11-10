# The images of the platform, in one file:
#   target "platform"  -> madoc-mlops:latest            the app (mlops-ui) and the worker (mlops-worker): kraken
#   target "baselines" -> madoc-mlops-baselines:latest  the other recognizers: the same, plus transformers / peft
# Both start from the same "deps" stage (system packages, PyTorch, requirements.txt), and the code is copied LAST:
# a change of the code rebuilds only that last step, nothing is downloaded again. The libraries are downloaded only
# when requirements.txt or a version below changes, and the pip cache (kept by BuildKit between builds, never in
# the image) means even then only what changed is downloaded. A slow PyPI is retried, not fatal.

FROM python:3.12-slim AS deps

ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
# git: the release clone of MaDoc; libgl1 / libglib2.0-0: OpenCV (augmentation)
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# PyTorch for CUDA 12.8 first (the RTX 50 series needs it); kraken 7.0.2 accepts torch 2.4 to 2.10
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# not root: the volumes are made with the rights of this user
RUN useradd -m -u 1000 mlops && mkdir -p /data /runs /release /hf-cache && chown mlops /data /runs /release /hf-cache


# the other recognizers (mlops/baselines): CRNN, TrOCR-style, Qwen2.5-VL-3B
FROM deps AS baselines
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install transformers==4.57.1 peft==0.17.1 accelerate==1.10.1
COPY . .
USER mlops
CMD ["python", "-m", "mlops.worker"]


# the app and the worker (the last stage: what "docker build ." makes)
FROM deps AS platform
COPY . .
USER mlops
CMD ["python", "-m", "mlops.ui"]