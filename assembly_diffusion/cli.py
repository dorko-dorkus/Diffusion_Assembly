import argparse
from typing import List, Tuple

import torch

from .config import SamplingConfig, load_run_config
from .logging_config import get_logger

logger = get_logger(__name__)


class PolicyGrammarAdapter:
    """Adapter bridging the feasibility mask and policy for batched sampling."""

    def __init__(self, mask, policy) -> None:  # pragma: no cover - simple container
        self.mask = mask
        self.policy = policy

    def enumerate_actions(self, state) -> List[Tuple]:
        feas_map = self.mask.mask_edits(state)
        return [a for a in feas_map.keys() if a != "STOP"]

    def actions_to_mask(self, states, cand_actions, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B = len(states)
        A = max((len(a) for a in cand_actions), default=0)
        out = torch.zeros((B, A), dtype=torch.bool, device=device)
        for i, (s, actions) in enumerate(zip(states, cand_actions)):
            feas_map = self.mask.mask_edits(s)
            for j, a in enumerate(actions):
                out[i, j] = bool(feas_map.get(a, 0))
        return out

    def apply_action(self, state, action):
        x = state.copy() if hasattr(state, "copy") else state
        if action == "STOP":
            return x
        if isinstance(action, tuple) and action and action[0] == "ADD":
            _, site, atom = action
            x.add_atom(atom, site)
            return x
        i, j, b = action
        x.bonds[i, j] = x.bonds[j, i] = b
        return x

    def is_terminal(self, state) -> bool:
        feas_map = self.mask.mask_edits(state)
        return not any(k != "STOP" and v for k, v in feas_map.items())


class BatchedPolicy:
    """Wrap ``ReversePolicy`` to expose a batched ``logits`` method."""

    def __init__(self, policy) -> None:  # pragma: no cover - simple container
        self.policy = policy

    def logits(self, states, cand_actions, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B = len(states)
        A = max((len(a) for a in cand_actions), default=0)
        out = torch.full((B, A), float("-inf"), device=device)
        for i, (s, actions) in enumerate(zip(states, cand_actions)):
            mask = {a: 1 for a in actions}
            mask["STOP"] = 1
            logits = self.policy.logits(s, t=1, mask=mask)
            out[i, : len(actions)] = logits[:-1]
        return out


def sample_demo(args):
    """Run a minimal sampling demo using the batched guided sampler."""
    try:
        from .graph import MoleculeGraph
        from .mask import FeasibilityMask
        from .backbone import GNNBackbone
        from .policy import ReversePolicy
        from .sampler import sample_with_guidance
    except ModuleNotFoundError as exc:  # pragma: no cover - import check
        raise SystemExit(f"Missing dependency: {exc.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_init = MoleculeGraph(['C', 'O'], torch.zeros((2, 2), dtype=torch.int64, device=device))

    mask = FeasibilityMask()
    policy = ReversePolicy(GNNBackbone()).to(device)
    grammar = PolicyGrammarAdapter(mask, policy)
    batch_policy = BatchedPolicy(policy)

    cfg = SamplingConfig()
    if args.gamma is not None:
        cfg.guidance_gamma = args.gamma
    if args.guidance_mode is not None:
        cfg.guidance_mode = args.guidance_mode

    init_states = [x_init.copy() for _ in range(64)]
    final_states, _ = sample_with_guidance(batch_policy, grammar, init_states, cfg, device=device)
    try:
        logger.info(final_states[0].canonical_smiles())
    except ImportError:  # pragma: no cover - optional RDKit
        logger.warning("RDKit not installed; skipping SMILES output")


def build_ai(method: str):
    """Instantiate the configured assembly index estimator."""

    if method == "surrogate":
        from .ai_surrogate import AISurrogate

        return AISurrogate()
    if method == "assemblymc":
        from .ai_mc import AssemblyMC

        return AssemblyMC()
    raise ValueError(f"Unknown ai.method '{method}'")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Assembly Diffusion command line interface")
    parser.add_argument("--config", type=str, help="Path to run configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Print parsed configuration and exit")
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

    cfg = None
    if args.config:
        cfg = load_run_config(args.config)
        if args.dry_run:
            print(cfg)
            return
        # Instantiate AI to exercise config selection
        build_ai(cfg.ai.method)

    if args.command == "sample":
        sample_demo(args)
    else:
        if args.config and not args.dry_run:
            # Config was supplied but no command; show parsed config
            print(cfg)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
