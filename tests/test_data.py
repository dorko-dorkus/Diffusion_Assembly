import torch
from assembly_diffusion.graph import MoleculeGraph
from assembly_diffusion.data import QM9CHON_Dataset, get_dataloader


def test_qm9_dataset_loading(monkeypatch):
    dummy = [MoleculeGraph([6], torch.zeros((1, 1), dtype=torch.int)) for _ in range(2)]
    monkeypatch.setattr('assembly_diffusion.data.load_qm9_chon', lambda max_heavy: dummy)
    dataset = QM9CHON_Dataset()
    assert len(dataset) == 2
    assert isinstance(dataset[0], MoleculeGraph)


def test_get_dataloader(monkeypatch):
    dummy = [MoleculeGraph([6], torch.zeros((1, 1), dtype=torch.int)) for _ in range(3)]
    monkeypatch.setattr('assembly_diffusion.data.load_qm9_chon', lambda max_heavy: dummy)
    loader = get_dataloader(batch_size=2)
    batch = next(iter(loader))
    assert isinstance(batch, list)
    assert len(batch) == 2
    assert isinstance(batch[0], MoleculeGraph)
