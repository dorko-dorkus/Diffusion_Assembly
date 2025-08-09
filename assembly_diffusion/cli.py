import argparse

from .config import SamplingConfig


def sample_demo(args):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_init = MoleculeGraph(
        ['C', 'O'], torch.zeros((2, 2), dtype=torch.int64, device=device)
    )
    kernel = ForwardKernel()
    mask = FeasibilityMask()
    policy = ReversePolicy(GNNBackbone()).to(device)
    sampler = Sampler(policy, mask)

    cfg = SamplingConfig()
    if args.gamma is not None:
        cfg.guidance_gamma = args.gamma
    if args.guidance_mode is not None:
        cfg.guidance_mode = args.guidance_mode

    if cfg.guidance_gamma != 0.0:
        # Dummy guidance hook for demonstration purposes
        def dummy_guidance(logits, x, t, mask):
            return torch.zeros_like(logits[:-1])

        x = sampler.sample(kernel.T, x_init, guidance=dummy_guidance, gamma=cfg.guidance_gamma)
    else:
        x = sampler.sample(kernel.T, x_init)
    print(x.canonical_smiles())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Assembly Diffusion command line interface")
    sub = parser.add_subparsers(dest="command")
    sample_parser = sub.add_parser("sample", help="Run a simple sampling demo")
    sample_parser.add_argument("--gamma", type=float, default=None, help="Guidance strength; overrides config if set")
    sample_parser.add_argument(
        "--guidance-mode",
        type=str,
        default=None,
        choices=["A_exact", "A_lower", "heuristic"],
        help="Score type for guidance",
    )

    args = parser.parse_args(argv)

    if args.command == "sample":
        sample_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
