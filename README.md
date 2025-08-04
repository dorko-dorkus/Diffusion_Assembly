# Diffusion Assembly

Prototype package for molecular assembly diffusion.

## Installation

Create and activate a virtual environment, then install the package requirements:

```bash
pip install torch
pip install -e .
```

### RDKit

RDKit is an optional dependency used for molecule manipulation and loading the QM9 dataset. It is often easier to install via conda:

```bash
conda install -c conda-forge rdkit
```

Without RDKit, features such as canonical SMILES generation and QM9 dataset loading will be unavailable and will raise informative errors.

## Testing

Run the test suite (if present) with:

```bash
pytest
```

