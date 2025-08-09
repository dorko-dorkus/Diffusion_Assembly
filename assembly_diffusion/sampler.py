import torch
from torch.distributions import Categorical

from .mask import FeasibilityMask
from .policy import ReversePolicy
from .graph import MoleculeGraph
from .guidance import reweight, assembly_guidance_scores

class Sampler:
    """Run reverse diffusion sampling using a policy and feasibility mask."""
    def __init__(self, policy: ReversePolicy, mask: FeasibilityMask):
        self.policy = policy
        self.masker = mask

    def sample(
        self,
        T: int,
        x_init: MoleculeGraph,
        guidance=None,
        gamma: float = 0.0,
        clip_range: tuple[float, float] = (-float("inf"), float("inf")),
    ) -> MoleculeGraph:
        """Generate a single sample by running ``T`` reverse steps.

        Parameters
        ----------
        T:
            Number of reverse diffusion steps.
        x_init:
            Starting molecular graph at time ``T``.
        guidance: callable, optional
            Function producing guidance scores ``ΔS`` for candidate edits. It
            must accept ``(logits, x, t, mask)`` and return a tensor aligned
            with the non-STOP logits.
        gamma: float, optional
            Scale factor for the guidance scores.
        clip_range: tuple of float, optional
            Range ``(a, b)`` used to clip ``ΔS`` before scaling.
        """

        x = x_init.copy()
        for t in range(T, 0, -1):
            mask = self.masker.mask_edits(x)
            logits = self.policy.logits(x, t, mask)
            if guidance is not None:
                delta = guidance(logits, x, t, mask)
                logits = reweight(logits, x, delta, gamma, clip_range)
            probs = torch.softmax(logits, dim=0)
            if not torch.isfinite(probs).any():
                return x
            dist = Categorical(probs)
            idx = dist.sample().item()
            action = self.policy._actions[idx]
            if action == "STOP":
                break
            if isinstance(action, tuple) and action and action[0] == "ADD":
                _, site, atom = action
                x.add_atom(atom, site)
            else:
                i, j, b = action
                x.bonds[i, j] = x.bonds[j, i] = b
        return x

    def trajectory(
        self,
        T: int,
        x_init: MoleculeGraph,
        guidance=None,
        gamma: float = 0.0,
        clip_range: tuple[float, float] = (-float("inf"), float("inf")),
    ):
        """Return the sequence of intermediate graphs for diagnostics.

        The returned list starts from ``x_init`` (``G_T``) and includes each
        intermediate prediction down to the final ``Ĝ_0``.
        """

        x = x_init.copy()
        traj = [x.copy()]
        for t in range(T, 0, -1):
            mask = self.masker.mask_edits(x)
            logits = self.policy.logits(x, t, mask)
            if guidance is not None:
                delta = guidance(logits, x, t, mask)
                logits = reweight(logits, x, delta, gamma, clip_range)
            probs = torch.softmax(logits, dim=0)
            if not torch.isfinite(probs).any():
                break
            dist = Categorical(probs)
            idx = dist.sample().item()
            action = self.policy._actions[idx]
            if action == "STOP":
                break
            if isinstance(action, tuple) and action and action[0] == "ADD":
                _, site, atom = action
                x.add_atom(atom, site)
            else:
                i, j, b = action
                x.bonds[i, j] = x.bonds[j, i] = b
            traj.append(x.copy())
        return traj


@torch.no_grad()
def sample_with_guidance(policy, grammar, init_states, cfg, device=None):
    """Sample trajectories with optional guidance scores."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gamma = float(cfg.guidance_gamma)

    states = list(init_states)
    finished = [False] * len(states)
    traj = [[] for _ in states]

    for step in range(cfg.max_steps):
        if all(finished):
            break
        cand_actions = [grammar.enumerate_actions(s) if not f else [] for s, f in zip(states, finished)]
        masks = grammar.actions_to_mask(states, cand_actions, device=device)
        logits = policy.logits(states, cand_actions, device=device)

        if gamma != 0.0:
            scores = assembly_guidance_scores(states, cand_actions, grammar, mode=cfg.guidance_mode)
            logits = logits + gamma * scores

        logits = logits.masked_fill(~masks, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        probs[~torch.isfinite(probs).any(dim=-1)] = 0.0
        idx = torch.zeros(len(states), dtype=torch.long, device=device)
        valid = probs.sum(dim=-1) > 0
        if valid.any():
            idx_valid = torch.multinomial(probs[valid].clamp(min=0), num_samples=1).squeeze(1)
            idx[valid] = idx_valid

        for i, aidx in enumerate(idx.tolist()):
            if finished[i]:
                continue
            if aidx < len(cand_actions[i]):
                a = cand_actions[i][aidx]
                states[i] = grammar.apply_action(states[i], a)
                traj[i].append(a)
                finished[i] = grammar.is_terminal(states[i])
            else:
                finished[i] = True

    return states, traj
