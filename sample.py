import json
import pandas as pd
import torch

from assembly_diffusion.graph import MoleculeGraph
from assembly_diffusion.forward import ForwardKernel
from assembly_diffusion.mask import FeasibilityMask
from assembly_diffusion.backbone import GNNBackbone
from assembly_diffusion.policy import ReversePolicy
from assembly_diffusion.sampler import Sampler
from assembly_diffusion.guidance import AssemblyPrior


def main():
    """Generate samples and save them to ``sample.parquet``.

    The script runs two phases for each guidance variant:
    1. Draw 1k trajectories for diagnostics.
    2. Draw 10k final graphs and persist them as a Parquet file.
    """

    torch.manual_seed(0)

    x_init = MoleculeGraph(["C", "O"], torch.zeros((2, 2), dtype=torch.int64))
    kernel = ForwardKernel()
    mask = FeasibilityMask()
    policy = ReversePolicy(GNNBackbone())
    sampler = Sampler(policy, mask)

    prior = AssemblyPrior(coeff=0.5, target=12)

    def prior_guidance(logits, x, t, m):
        new_logits = prior.reweight(logits.clone(), x)
        return new_logits[:-1] - logits[:-1]

    variants = {"unguided": None, "assembly_prior": prior_guidance}

    records = []

    for name, guide in variants.items():
        for _ in range(1000):
            sampler.trajectory(kernel.T, x_init, guidance=guide, gamma=1.0)

        for _ in range(10000):
            x = sampler.sample(kernel.T, x_init, guidance=guide, gamma=1.0)
            try:
                smiles = x.canonical_smiles()
            except Exception:
                smiles = None
            records.append(
                {
                    "variant": name,
                    "atoms": json.dumps(x.atoms),
                    "bonds": json.dumps(x.bonds.tolist()),
                    "smiles": smiles,
                }
            )

    df = pd.DataFrame(records)
    df.to_parquet("sample.parquet")


if __name__ == "__main__":
    main()
