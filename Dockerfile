# CPU-only image of the full torch + FAISS service, for any Docker host (e.g. Google Cloud Run).
# The index and both models are baked in at build time so the container starts in seconds.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/user/.cache/huggingface \
    EMBED_DEVICE=cpu \
    RERANK_DEVICE=cpu \
    PORT=7860

# Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# CPU wheel of torch first, so the PyPI install below doesn't pull ~3 GB of CUDA libraries.
RUN pip install --user --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
COPY --chown=user requirements.txt .
RUN grep -vE "^(torch|pytest|httpx)==" requirements.txt > /tmp/requirements-serve.txt \
    && pip install --user -r /tmp/requirements-serve.txt

COPY --chown=user app ./app
COPY --chown=user scripts ./scripts
COPY --chown=user results ./results

RUN python -m scripts.build_index --device cpu \
    && python -c "from app.reranker import CrossEncoderReranker; CrossEncoderReranker(device='cpu')"

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT}"]
