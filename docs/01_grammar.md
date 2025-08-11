# Grammar and Primitives (G, P)

The diffusion process optimizes over an explicit molecular grammar **G** with
primitive set **P**.

## Primitive set P
- Atoms: {C, N, O, H} with valence caps {C:4, N:3, O:2, H:1}.
- Bond orders: single and double; triple and aromatic bonds are excluded in v0.

## Grammar G
- **State space:** `MoleculeGraph` objects (see `assembly_diffusion/graph.py`)
  whose atoms are drawn from P and whose bonds are integer orders.
- **Start state:** a minimal seed graph used for sampling.
- **Production rules:**
  1. **Bond edit** `(i, j, b)` sets the bond order between existing atoms
     `i` and `j` to `b ∈ {0,1,2}` (`0` deletes the bond).
  2. **Atom insertion** `("ADD", i, a)` attaches a new atom `a ∈ {C,N,O,H}`
     to atom `i` with a single bond.
  3. **STOP** terminates growth.

Each rule adds at most one bond, aligning with the assembly-theory intuition of
"build by attaching one bond per step." Feasibility of actions is enforced by
`FeasibilityMask` (`assembly_diffusion/mask.py`), which performs valence checks
and optional RDKit sanitization.

## Grammar G′
`G′` extends `G` with an additional production rule that relabels existing
atoms:

4. **Atom substitution** `("SUB", i, a)` replaces the atom at index `i` with a
   new element `a ∈ {C,N,O,H}` while preserving existing bonds.

The remainder of the state space, start state and feasibility checks mirror the
original grammar.  This variant explores the impact of allowing atom relabeling
in the assembly process.

### Acyclic exactness
For molecules whose bond graph is a tree, the exact assembly index
`A*(x|G,P)` equals the number of edges because each step contributes a single
bond. Calibrators in `assembly_diffusion/calibrators/strings.py` and
`assembly_diffusion/calibrators/trees.py` validate this property.

All experimental results are conditioned on this choice of `(G, P)`; changing
the grammar or primitives alters the objective and conclusions.
