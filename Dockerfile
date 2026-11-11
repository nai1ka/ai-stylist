FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=4 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libegl1 libgles2 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-docker.txt ./
# Explicit CPU wheels prevent pulling CUDA libraries on Linux.
RUN python -m pip install torch==2.15.0 torchvision==0.29.2 \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements-docker.txt \
    && python -m pip check

COPY app.py ./
COPY stylist ./stylist
COPY data ./data
COPY scripts ./scripts
COPY README.md SOURCES.md MODEL_NOTES.md ./
RUN useradd --create-home --uid 10001 stylist \
    && mkdir -p /app/models \
    && chown -R stylist:stylist /app
USER stylist

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
