FROM mambaorg/micromamba:1.5.8
ARG PYVER=3.11
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba create -y -n app -c conda-forge python=${PYVER} rdkit=2023.09.* \
 && micromamba clean -a -y
WORKDIR /app
COPY env.lock ./
RUN micromamba run -n app python -m pip install --upgrade pip \
 && micromamba run -n app pip install --no-cache-dir -r env.lock --extra-index-url https://download.pytorch.org/whl/cpu
COPY . /app
RUN micromamba run -n app pip install --no-cache-dir -e .
CMD ["micromamba","run","-n","app","pytest","-q"]
