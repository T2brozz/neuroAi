FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIPENV_VENV_IN_PROJECT=1
ENV PIPENV_NOSPIN=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    git \
    python3-pip \
    python3-venv \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && pip install pipenv

WORKDIR /workspace

COPY Pipfile Pipfile.lock* /

RUN PIPENV_PYTHON=3.11 pipenv install --deploy

CMD ["pipenv", "run", "jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]