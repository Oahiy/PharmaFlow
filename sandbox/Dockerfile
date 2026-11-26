FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && uv pip install --system --no-cache pandas==3.1.1 rdkit==2026.9.2 matplotlib==3.11.2

RUN groupadd --gid 10001 sandbox \
    && useradd --uid 10001 --gid 10001 --create-home sandbox \
    && install -d /work/input \
    && install -d -o 10001 -g 10001 /work/output /work/.matplotlib

ENV MPLBACKEND=Agg MPLCONFIGDIR=/work/.matplotlib PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

WORKDIR /work
USER 10001:10001
CMD ["python", "/work/main.py"]
