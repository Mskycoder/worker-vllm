FROM nvidia/cuda:12.1.0-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# ------------------------------------------------------------------
# Install PyTorch 2.3 + CUDA 12.8, vLLM 0.10.1 + gptoss and FlashInfer
# ------------------------------------------------------------------
RUN python3 -m pip install --upgrade pip && \
    # 1️⃣  Torch + CUDA 12.8  (stable wheels live in the cu128 index)
    python3 -m pip install --pre \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        torch==2.3.0+cu128 torchvision==0.18.0+cu128 --index-strategy unsafe-best-match && \
    # 2️⃣  vLLM build that understands MXFP-4 MoE (lives on the gpt-oss wheel index)
    python3 -m pip install --pre \
        --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
        vllm==0.10.1+gptoss \
        --index-strategy unsafe-best-match && \
    # 3️⃣  FlashInfer kernels for Blackwell / Hopper (package is *flashinfer-python*)
    python3 -m pip install flashinfer-python==0.1.5 \
        -i https://flashinfer.ai/whl/cu128/torch2.3

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME="openai/gpt-oss-120b"
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 

ENV PYTHONPATH="/:/vllm-workspace"
ENV MODEL_NAME="openai/gpt-oss-120b"
# optional but useful defaults
ENV DTYPE="auto" \
    MAX_MODEL_LEN="32768" \
    GPU_MEMORY_UTILIZATION="0.95"

COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]
