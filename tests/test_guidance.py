import pathlib
import sys

import pytest

# Import torch lazily so the test suite can run without the dependency.
torch = pytest.importorskip("torch")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembly_diffusion.guidance import (
    AssemblyPrior,
    additive_guidance,
    linear_lambda_schedule,
)
from assembly_diffusion.graph import MoleculeGraph
from assembly_diffusion.assembly_index import approx_AI


def test_assembly_prior_reweight():
    graph = MoleculeGraph(["C", "C"], torch.zeros((2, 2), dtype=torch.int64))
    logits = torch.zeros(3)
    prior = AssemblyPrior(coeff=0.5, target=1)
    new_logits = prior.reweight(logits.clone(), graph)

    ai = approx_AI(graph)
    penalty = 0.5 * (ai - 1)
    expected = torch.tensor([-penalty, -penalty, 0.0])
    assert torch.allclose(new_logits, expected)


class ToyGrammar:
    """Minimal grammar returning preset ``Â`` scores."""

    def __init__(self, scores):
        self.scores = scores

    def apply_action(self, state, action):
        return action

    def A_hat(self, state):
        return self.scores[state]


def test_additive_guidance_monotone():
    grammar = ToyGrammar({"a": 1.0, "b": 3.0})
    state = ""
    actions = ["a", "b"]
    A_vals = torch.tensor(
        [grammar.A_hat(grammar.apply_action(state, a)) for a in actions]
    )
    logits = torch.zeros(len(actions))

    lambdas = [linear_lambda_schedule(t, 3, 1.0) for t in range(3)]
    probs = []
    for lam in lambdas:
        new_logits = additive_guidance(logits, A_vals, lam)
        probs.append(torch.softmax(new_logits, dim=-1)[0].item())

    assert probs[0] < probs[1] < probs[2]
