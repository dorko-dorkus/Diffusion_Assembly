FROM condaforge/miniforge3

WORKDIR /app

COPY env.lock env.lock
COPY env/conda.yml conda.yml

RUN conda env create -f conda.yml && conda clean -afy

SHELL ["conda", "run", "-n", "assembly_diffusion", "/bin/bash", "-c"]

CMD ["python", "-c", "import rdkit, torch; print(rdkit.__version__)"]
