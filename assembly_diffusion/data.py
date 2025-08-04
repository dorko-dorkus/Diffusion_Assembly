import os
import urllib.request
import tarfile

from torch.utils.data import Dataset, DataLoader

from .graph import MoleculeGraph

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"


def download_qm9():
    """Download and extract the QM9 dataset if necessary."""
    os.makedirs("qm9_raw", exist_ok=True)
    archive_path = "qm9_raw/gdb9.tar.gz"
    if not os.path.exists(archive_path):
        print("Downloading QM9 dataset...")
        urllib.request.urlretrieve(URL, archive_path)
        print("Download complete.")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path="qm9_raw")
            print("Extraction complete.")


def load_qm9_chon(max_heavy: int = 12):
    """Load molecules from the QM9 dataset restricted to C/H/O/N."""
    from rdkit.Chem import rdmolfiles
    import torch

    if not os.path.exists("qm9_raw/gdb9.sdf"):
        download_qm9()
    mols = []
    suppl = rdmolfiles.SDMolSupplier("qm9_raw/gdb9.sdf")
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
    def __init__(self, max_heavy: int = 12):
        self.data = load_qm9_chon(max_heavy)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(batch_size: int = 32, max_heavy: int = 12):
    dataset = QM9CHON_Dataset(max_heavy)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
