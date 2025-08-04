import argparse


def sample_demo():
    """Run a minimal sampling demo."""
    try:
        import torch
        from .graph import MoleculeGraph
        from .forward import ForwardKernel
        from .mask import FeasibilityMask
        from .backbone import GNNBackbone
        from .policy import ReversePolicy
        from .sampler import Sampler
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}")

    x_init = MoleculeGraph(['C', 'O'], torch.zeros((2, 2)))
    kernel = ForwardKernel()
    mask = FeasibilityMask()
    policy = ReversePolicy(GNNBackbone())
    sampler = Sampler(policy, mask)
    x = sampler.sample(kernel.T, x_init)
    print(x.canonical_smiles())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Assembly Diffusion command line interface")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("sample", help="Run a simple sampling demo")
    args = parser.parse_args(argv)

    if args.command == "sample":
        sample_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
