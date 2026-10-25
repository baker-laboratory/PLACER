# Linux x86_64 / NVIDIA inference image, independent of Python packaging work.
FROM mambaorg/micromamba:2.9.0@sha256:7b93def90e48d6d7744938e31c29172e03b19bc9f56f0db512039f18efd21529 AS base

FROM base AS environment

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/conda-linux-64.lock /tmp/conda-linux-64.lock
RUN micromamba install --yes --name base --file /tmp/conda-linux-64.lock && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Git and build artifacts stay out of the final runtime image.
FROM environment AS se3-builder
USER root
RUN apt-get update && \
    apt-get install --yes --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER
RUN python -m pip wheel --no-deps --no-build-isolation --no-cache-dir \
    --wheel-dir /tmp/se3-wheels \
    "se3-transformer @ git+https://github.com/NVIDIA/DeepLearningExamples.git@729963dd47e7c8bd462ad10bfac7a7b0b604e6dd#subdirectory=DGLPyTorch/DrugDiscovery/SE3Transformer"

FROM environment AS assembled
COPY --from=se3-builder --chown=$MAMBA_USER:$MAMBA_USER /tmp/se3-wheels /tmp/se3-wheels
RUN python -m pip install --no-deps --no-cache-dir /tmp/se3-wheels/*.whl && \
    rm -rf /tmp/se3-wheels

# Copy the completed environment with root ownership into a fresh runtime stage.
# This avoids duplicating the entire environment in a recursive-chown layer.
FROM base AS runtime
USER root
RUN chown root:root /opt/conda && chmod 0755 /opt/conda && \
    mkdir -p /opt/placer /work && chmod 0755 /opt/placer /work
COPY --from=assembled --chown=0:0 /opt/conda/ /opt/conda/
COPY --chown=0:0 PLACER.py run_PLACER.py LICENSE README.md /opt/placer/
COPY --chown=0:0 modules/ /opt/placer/modules/
COPY --chown=0:0 data/ /opt/placer/data/
COPY --chown=0:0 weights/ /opt/placer/weights/
COPY --chown=0:0 examples/inputs/dnHEM1_apo.pdb /opt/placer/examples/inputs/dnHEM1_apo.pdb
USER $MAMBA_USER

ARG PLACER_REVISION=unknown
ARG BUILD_CREATED=""
LABEL org.opencontainers.image.title="PLACER CUDA inference" \
      org.opencontainers.image.description="Local prototype for Linux x86_64 / NVIDIA CUDA inference; includes PLACER model weights and chemical resources." \
      org.opencontainers.image.source="https://github.com/patrickrochemn/PLACER" \
      org.opencontainers.image.url="https://github.com/baker-laboratory/PLACER" \
      org.opencontainers.image.revision="$PLACER_REVISION" \
      org.opencontainers.image.created="$BUILD_CREATED" \
      org.opencontainers.image.version="unreleased" \
      org.opencontainers.image.licenses="BSD-3-Clause"

ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    PYTHONPATH=/opt/placer \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DGLBACKEND=pytorch \
    OMP_NUM_THREADS=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "/opt/placer/run_PLACER.py"]
CMD ["--help"]
