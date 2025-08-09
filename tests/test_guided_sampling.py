from __future__ import annotations
import torch
from assembly_diffusion.sampler import sample_with_guidance

class ToyState:
    def __init__(self, v=0):
        self.v = v

    def num_edges(self):
        return self.v

    def is_acyclic(self):
        return True

    def num_nodes(self):
        return self.v + 1

    def num_connected_components(self):
        return 1


class ToyGrammar:
    def enumerate_actions(self, s):
        return [0, 1, 2]

    def actions_to_mask(self, states, cand_actions, device=None):
        B = len(states)
        A = max(len(a) for a in cand_actions)
        return torch.ones((B, A), dtype=torch.bool, device=device)

    def apply_action(self, s, a):
        return ToyState(s.v + (a + 1))

    def is_terminal(self, s):
        return s.v >= 3

    def heuristic_A(self, s):
        return s.num_edges()


class ToyPolicy:
    def logits(self, states, cand_actions, device=None):
        B = len(states)
        A = max(len(a) for a in cand_actions)
        return torch.zeros((B, A), device=device)


class Cfg:
    guidance_gamma = 0.0
    guidance_mode = "A_exact"
    max_steps = 4


def test_guidance_pushes_up_A():
    torch.manual_seed(0)
    toyG = ToyGrammar()
    pol = ToyPolicy()
    init = [ToyState(0)] * 128

    final_b, _ = sample_with_guidance(pol, toyG, init, Cfg)
    A_b = sum(s.num_edges() for s in final_b) / len(final_b)

    Cfg.guidance_gamma = 0.8
    final_g, _ = sample_with_guidance(pol, toyG, init, Cfg)
    A_g = sum(s.num_edges() for s in final_g) / len(final_g)
    assert A_g >= A_b
