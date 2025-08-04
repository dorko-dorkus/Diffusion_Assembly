import hashlib
import os
import tarfile
import urllib.request

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .graph import MoleculeGraph
from .backbone import ATOM_MAP

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
QM9_SHA256 = "45255048ac6d83ea4b923ecdf7d6fb6dc62bfec5e80fbc5bcfd93a62157a31db"
DEFAULT_DATA_DIR = os.environ.get("QM9_DATA_DIR", "qm9_raw")


def _verify_sha256(path: str, expected: str) -> bool:
    """Return ``True`` if ``path`` matches the expected SHA256 hash."""
    if not os.path.exists(path):
        return False
    hash_sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest() == expected


def download_qm9(data_dir: str = DEFAULT_DATA_DIR, *, url: str = URL, sha256: str = QM9_SHA256) -> None:
    """Download and extract the QM9 dataset if necessary.

    Parameters
    ----------
    data_dir:
        Directory where the QM9 data should be stored. Defaults to ``QM9_DATA_DIR``
        environment variable or ``qm9_raw``.
    url:
        Location of the dataset archive.
    sha256:
        Expected SHA256 hash of the archive for integrity checking.
    """

    os.makedirs(data_dir, exist_ok=True)
    archive_path = os.path.join(data_dir, "gdb9.tar.gz")
    if not _verify_sha256(archive_path, sha256):
        print("Downloading QM9 dataset...")
        try:
            urllib.request.urlretrieve(url, archive_path)
        except Exception as e:  # pragma: no cover - network failure
            raise RuntimeError(f"Failed to download QM9 dataset: {e}") from e
        if not _verify_sha256(archive_path, sha256):
            raise RuntimeError("Checksum mismatch for downloaded QM9 archive.")
        print("Download complete.")

    if not os.path.exists(os.path.join(data_dir, "gdb9.sdf")):
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
                print("Extraction complete.")
        except tarfile.TarError as e:  # pragma: no cover - corrupted archive
            os.remove(archive_path)
            raise RuntimeError("Failed to extract QM9 archive. File may be corrupted and has been removed.") from e


def load_qm9_chon(max_heavy: int = 12, data_dir: str = DEFAULT_DATA_DIR):
    """Load molecules from the QM9 dataset restricted to C/H/O/N.

    Parameters
    ----------
    max_heavy:
        Maximum number of non-hydrogen atoms allowed in a molecule.
    data_dir:
        Location of the QM9 data directory.
    """

    try:
        import torch
    except ImportError as e:  # pragma: no cover - runtime check
        raise ImportError(
            "PyTorch is required to load QM9 data. Install it via 'pip install torch'."
        ) from e
    try:
        from rdkit.Chem import rdmolfiles
    except ImportError as e:  # pragma: no cover - runtime check
        raise ImportError(
            "RDKit is required to load QM9 data. Install it, e.g., via 'conda install -c conda-forge rdkit'."
        ) from e

    sdf_path = os.path.join(data_dir, "gdb9.sdf")
    if not os.path.exists(sdf_path):
        download_qm9(data_dir)
    mols = []
    suppl = rdmolfiles.SDMolSupplier(sdf_path)
    for mol in suppl:
        if mol is None:
            continue
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        if all(a in ['C', 'H', 'O', 'N'] for a in atoms) and sum(a != 'H' for a in atoms) <= max_heavy:
            bonds = torch.zeros((len(atoms), len(atoms)))
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                order = int(bond.GetBondTypeAsDouble())
                bonds[i, j] = bonds[j, i] = order
            mols.append(MoleculeGraph(atoms, bonds))
    return mols


class QM9CHON_Dataset(Dataset):
    """Torch dataset for the filtered QM9 molecules."""

    def __init__(self, max_heavy: int = 12, data_dir: str = DEFAULT_DATA_DIR):
        self.data = load_qm9_chon(max_heavy, data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_graphs(batch):
    """Pack a list of :class:`MoleculeGraph` objects into padded tensors."""

    atom_ids = [
        torch.tensor([ATOM_MAP.get(a, 0) for a in m.atoms], dtype=torch.long)
        for m in batch
    ]
    atom_tensor = pad_sequence(atom_ids, batch_first=True)

    max_atoms = atom_tensor.size(1)
    bond_tensor = atom_tensor.new_zeros(len(batch), max_atoms, max_atoms, dtype=torch.float)
    for i, m in enumerate(batch):
        n = m.bonds.size(0)
        bond_tensor[i, :n, :n] = m.bonds.float()

    return atom_tensor, bond_tensor


def get_dataloader(batch_size: int = 32, max_heavy: int = 12, data_dir: str = DEFAULT_DATA_DIR):
    """Return a dataloader over the filtered QM9 molecules."""

    dataset = QM9CHON_Dataset(max_heavy, data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs)
