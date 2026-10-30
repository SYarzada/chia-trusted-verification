FROM ghcr.io/ucb-bar/chia-opencode:latest

USER root

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       build-essential \
       bison \
       flex \
       gawk \
       git \
       make \
       ca-certificates \
       pkg-config \
       python3 \
       libreadline-dev \
       tcl-dev \
       libffi-dev \
       zlib1g-dev \
       z3 \
    && rm -rf /var/lib/apt/lists/*

ARG YOSYS_REF=7b0c1fe49
RUN git clone --recursive https://github.com/YosysHQ/yosys.git /tmp/yosys \
    && cd /tmp/yosys \
    && git checkout "$YOSYS_REF" \
    && git submodule update --init --recursive \
    && make config-gcc \
    && make -j4 \
    && make install \
    && rm -rf /tmp/yosys

RUN python -m pip install --no-cache-dir click

ARG SBY_REF=v0.68
RUN git clone https://github.com/YosysHQ/sby.git /tmp/sby \
    && cd /tmp/sby \
    && git checkout "$SBY_REF" \
    && make install \
    && rm -rf /tmp/sby

RUN command -v yosys \
    && yosys -V \
    && command -v yosys-smtbmc \
    && command -v z3 \
    && z3 --version \
    && command -v sby

USER ray
