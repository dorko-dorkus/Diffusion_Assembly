# Diffusion Assembly

Prototype package for molecular assembly diffusion.

## Installation

The project targets Python 3.10+ and is distributed as a standard Python package.
To get started locally:

1. **Clone the repository**

   ```bash
   git clone https://github.com/username/Diffusion_Assembly.git
   cd Diffusion_Assembly
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the dependencies**

   Install PyTorch first (choose the variant appropriate for your hardware),
   then install the package itself in editable mode:

   ```bash
   pip install torch
   pip install -e .
   ```

### RDKit

RDKit is an optional dependency used for molecule manipulation and loading the QM9 dataset. It is often easier to install a pinned version via conda:

```bash
conda install -c conda-forge rdkit=2024.09.6
```

Without RDKit, features such as canonical SMILES generation and QM9 dataset loading will be unavailable and will raise informative errors.

### Dataset Handling

The QM9 dataset is downloaded on demand. By default files are stored in a
`qm9_raw` directory, but the location can be overridden by setting the
`QM9_DATA_DIR` environment variable. Downloads are verified against a
SHA-256 checksum and corrupted archives trigger a clear error so they can be
re-downloaded.

## Usage

Once installed, the package exposes a small command line interface that can
perform a minimal molecule sampling demo. From the project root run:

```bash
python -m assembly_diffusion sample
```

The command constructs a simple molecular graph, runs the diffusion sampler and
prints a canonical SMILES string of the generated molecule. Additional utilities
for training and evaluation are available in the `assembly_diffusion` module and
can serve as starting points for custom experiments.

## Testing

Run the test suite (if present) with:

```bash
pytest
```

