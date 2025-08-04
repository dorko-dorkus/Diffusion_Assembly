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

### Dataset Handling

The QM9 dataset is downloaded on demand. By default files are stored in a
`qm9_raw` directory, but the location can be overridden by setting the
`QM9_DATA_DIR` environment variable. Downloads are verified against a
SHA-256 checksum and corrupted archives trigger a clear error so they can be
re-downloaded.

## Testing

Run the test suite (if present) with:

```bash
pytest
```

