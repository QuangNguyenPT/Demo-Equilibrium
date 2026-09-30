"""HRM-X v0.6: constructive concept induction on top of v0.5's structural operator inference.

v0.6 (ADR-013, `vault/10 - Research Program/architecture_spec.md`): when no written family explains
the evidence, the mind constructs new features from primitives (`construct.py`), fits their linear
closure exactly, and stores verified constructions as reusable concepts (`concept_memory.py`).
`model.py` is the trainable, checkpointed model; `python -m hrm_x` its CLI; `language.py` the text
interface. What follows describes v0.5, which v0.6 contains unchanged:

HRM-X v0.5: structural operator inference, under value-of-computation control.

A hypothesis is no longer a program that has to be found by search. It is a
*family* -- the exact solution set of a law set -- whose rational parameters are
fitted exactly, and which may answer a query only where Theorem 3 says the
evidence determines it. The v0.4 program-search route is still here in
`hrm_x/domains/arith.py` and `hrm_x/library.py`, because it is the baseline the
new route is measured against in `experiments/x_v05.py`, not because it is the
default.

Spec: vault/01 - Architecture Blueprint/HRM-X Specification v0.5 - Structure-Preserving Operator Inference.md
Decision: vault/06 - Devlogs & ADRs/ADR-010 Structural Operator Inference.md
"""
