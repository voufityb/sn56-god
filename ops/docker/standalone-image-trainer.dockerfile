FROM python:3.11-slim

# Same ComfyUI revision + numerical stack as the G.O.D image evaluator.
ARG COMFYUI_COMMIT=694815f498295080a0e15a1502edc9dba841b110
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 COMFY_ROOT=/opt/ComfyUI PYTHONPATH=/app/trainer
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git init /opt/ComfyUI && cd /opt/ComfyUI && \
    git remote add origin https://github.com/comfyanonymous/ComfyUI.git && \
    git fetch --depth 1 origin "${COMFYUI_COMMIT}" && git checkout FETCH_HEAD && rm -rf .git
RUN pip install --timeout 120 --retries 10 --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --timeout 120 --retries 10 --no-cache-dir -r /opt/ComfyUI/requirements.txt \
    diffusers==0.39.0 transformers==5.10.2 huggingface-hub==1.18.0 safetensors==0.8.0 pydantic==2.13.4 pillow numpy
RUN mkdir -p /opt/godtrainer/assets && \
    curl -fL -o /opt/godtrainer/assets/qwen_image_vae.safetensors https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors && \
    curl -fL -o /opt/godtrainer/assets/flux2-vae.safetensors https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/vae/flux2-vae.safetensors
COPY trainer /app/trainer
COPY scripts /app/scripts
RUN python -c "import torch; assert torch.__version__.startswith('2.9.1')" && mkdir -p /app/checkpoints
ENTRYPOINT ["python", "/app/scripts/image_trainer.py"]
