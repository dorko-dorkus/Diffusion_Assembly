FROM mambaorg/micromamba:1.5.8

# Build arguments
ARG PYVER=3.11
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Create the base environment with RDKit available
RUN micromamba create -y -n app -c conda-forge python=${PYVER} rdkit=2023.09.* \
 && micromamba clean -a -y

WORKDIR /app

# Install python dependencies first to leverage Docker layer caching
COPY env.lock ./
RUN micromamba run -n app python -m pip install --upgrade pip \
 && micromamba run -n app pip install --no-cache-dir -r env.lock --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the project and install in editable mode from the source tree
COPY . /app
WORKDIR /app/assembly_diffusion
RUN micromamba run -n app pip install --no-cache-dir -e ..

# Sanity check that core modules can be imported
RUN micromamba run -n app python - <<'PY'
import rdkit
import assembly_diffusion
print('imports ok')
PY

# Default command runs the test suite
WORKDIR /app
CMD ["micromamba","run","-n","app","pytest","-q"]
