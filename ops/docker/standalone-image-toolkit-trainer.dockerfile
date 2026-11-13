# Image trainer for every non-FLUX image task (G.O.D builds this file for all ImageModelTypes except FLUX).
# Two isolated Python stacks:
#  - system python3 (the AI Toolkit base image): AI Toolkit + a Musubi venv, same pins as G.O.D's reference toolkit
#    trainer at 14c3208 (#1420). Used for FLUX.2 Klein, Qwen-Image-Edit-2511, HiDream O1, MOSS and all edit tasks.
#  - /opt/venv-ours: our engine (trainer/), same ComfyUI revision + numerical stack as the G.O.D legacy image
#    evaluator. Used for z-image, qwen-image and krea2 text-to-image.
# Base pinned by digest (the reference uses diagonalge/ai-toolkit:latest; this is that tag's digest on 2026-11-13).
FROM diagonalge/ai-toolkit@sha256:c24f8bb95bf1dc8da7cd6158a763f2c9782783ad7648dc4047c5757ef3447db8

ARG AI_TOOLKIT_COMMIT=e9c72b762748e4d31837fbeb3b87d01a5b143321
ARG MUSUBI_COMMIT=30c658c4f4b0bf05038b3346eff9670259b10fc7
ARG COMFYUI_COMMIT=694815f498295080a0e15a1502edc9dba841b110

WORKDIR /app/ai-toolkit
RUN git fetch origin "${AI_TOOLKIT_COMMIT}" && \
    git checkout "${AI_TOOLKIT_COMMIT}" && \
    pip install --no-cache-dir -r requirements.txt

RUN git init /app/musubi-tuner && \
    cd /app/musubi-tuner && \
    git remote add origin https://github.com/kohya-ss/musubi-tuner.git && \
    git fetch --depth 1 origin "${MUSUBI_COMMIT}" && \
    git checkout FETCH_HEAD && \
    python3 -m venv /opt/musubi && \
    /opt/musubi/bin/pip install --upgrade pip && \
    /opt/musubi/bin/pip install --no-cache-dir \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && \
    /opt/musubi/bin/pip install --no-cache-dir -e . lycoris-lora

# Our engine stack (was python:3.11-slim + system pip; now an isolated venv on the base image's python3).
RUN git init /opt/ComfyUI && cd /opt/ComfyUI && \
    git remote add origin https://github.com/comfyanonymous/ComfyUI.git && \
    git fetch --depth 1 origin "${COMFYUI_COMMIT}" && git checkout FETCH_HEAD && rm -rf .git
RUN python3 -m venv /opt/venv-ours && \
    /opt/venv-ours/bin/pip install --upgrade pip && \
    /opt/venv-ours/bin/pip install --timeout 120 --retries 10 --no-cache-dir \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 && \
    /opt/venv-ours/bin/pip install --timeout 120 --retries 10 --no-cache-dir -r /opt/ComfyUI/requirements.txt \
        diffusers==0.39.0 transformers==5.10.2 huggingface-hub==1.18.0 safetensors==0.8.0 pydantic==2.13.4 pillow numpy
RUN mkdir -p /opt/godtrainer/assets && \
    curl -fL -o /opt/godtrainer/assets/qwen_image_vae.safetensors https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors && \
    curl -fL -o /opt/godtrainer/assets/flux2-vae.safetensors https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/vae/flux2-vae.safetensors

COPY trainer /app/trainer
COPY scripts /app/scripts
COPY templates /app/templates
RUN /opt/venv-ours/bin/python -c "import torch; assert torch.__version__.startswith('2.9.1')" && \
    /opt/musubi/bin/python -c "import torch; assert torch.__version__.startswith('2.11.0')" && \
    python3 -c "import yaml, PIL" && mkdir -p /app/checkpoints

ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 COMFY_ROOT=/opt/ComfyUI \
    ENGINE_PYTHON=/opt/venv-ours/bin/python TOOLKIT_TEMPLATES=/app/templates \
    MUSUBI_PYTHON=/opt/musubi/bin/python MUSUBI_ACCELERATE=/opt/musubi/bin/accelerate
WORKDIR /app
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3", "/app/scripts/image_trainer.py"]
