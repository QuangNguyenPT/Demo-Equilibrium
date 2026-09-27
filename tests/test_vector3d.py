"""Vector3D domain, the continuous depth axis, and the bugs benchmark v2 exposed.

Every test here corresponds to a defect that benchmark v1 could not see and v2
did, and each is written to fail loudly if that defect returns. The failure
modes were subtle: in every case the system still *ran*, still produced
plausible trace output, and still reported a solve rate — just a wrong one.
"""
import itertools

import numpy as np
import pytest

from hrm.config import PerceptionConfig, SubstrateConfig, VerifyConfig
from hrm.learning.generators import Vector3DTasks
from hrm.perception import Program, Vector3DDomain
from hrm.perception.vector3d import (ROTATION_SET, SCALES, Project, Reflect, Rotate, Scale,
                                     VectorProgram, _program_matrix)
from hrm.substrate import Codebook
from hrm.verify import Verifier

CB = Codebook(SubstrateConfig(dim=1024))
DOM = Vector3DDomain(CB, PerceptionConfig())
VERIFIER = Verifier(VerifyConfig())
GEN = Vector3DTasks()
RNG = np.random.default_rng(11)


def prog(*ops) -> VectorProgram:
    return VectorProgram(tuple(ops))


def _reapply(ops, v):
    """Apply ops in order, independent of VectorProgram, as a cross-check."""
    out = np.asarray(v, dtype=np.float64)
    for op in ops:
        out = op.run(out)
    return out


class TestJSONRoundTrip:
    """The stored benchmark is the only path that goes through JSON.

    The base Domain casts to int64, correct for the grid and sequence alphabets.
    Applied to a continuous vector it truncated 0.861382 -> 0 and 1.73627 -> 1,
    so every demonstration in the written benchmark described a different
    function than the one generated, and the domain scored 0 on all composed
    tasks. In-memory probes passed throughout; only the round trip failed.
    """

    def test_from_json_preserves_fractional_components(self):
        v = [0.861382, -0.817264, 1.73627]
        assert np.allclose(DOM.from_json(v), v)

    def test_from_json_is_not_truncated_to_int(self):
        out = DOM.from_json([0.5, 0.25, 0.75])
        assert out.dtype.kind == "f"
        assert np.allclose(out, [0.5, 0.25, 0.75])

    def test_round_trip_survives_tolist(self):
        v = np.array([0.049984, 0.419916, 2.042699])
        assert np.allclose(DOM.from_json(DOM.to_json(v)), v)

    def test_generated_task_survives_write_and_read(self):
        """A generated task must mean the same thing after a JSON round trip."""
        task = GEN.generate("d3_rotate", np.random.default_rng(3))
        for inp, out in list(task.demos) + [(task.test_input, task.expected)]:
            assert np.allclose(DOM.from_json(DOM.to_json(inp)), np.asarray(inp))
            assert np.allclose(DOM.from_json(DOM.to_json(out)), np.asarray(out))


class TestComplexityLimit:
    """Input width must not be mistaken for permission to think.

    complexity_limit derived a cap from input size (input_size // 4). A 3-vector
    reports size 3 at any depth, so three demonstrations implied a limit of 2
    and every program of complexity 3+ was rejected as "memorization". The
    reasoner looked incapable of reasoning past two steps because a guard had
    forbidden it.
    """

    def test_vector_domain_is_not_capped_by_input_width(self):
        demos = [(np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6]))] * 3
        assert VERIFIER.complexity_limit(DOM, demos) == VerifyConfig().max_program_complexity

    def test_deep_program_is_not_called_memorization(self):
        ops = (Rotate("x", 90), Rotate("y", 90), Scale(2.0), Reflect(0))
        p = prog(*ops)
        demos = [(v, p.apply(v)) for v in RNG.normal(size=(3, 3))]
        report = VERIFIER.verify(DOM, p, demos)
        assert not any("memorization" in r for r in report.reasons)

    def test_length_scaled_domains_keep_the_relative_cap(self):
        """The heuristic must still apply where input width tracks program depth."""
        from hrm.perception import SequenceDomain
        seq = SequenceDomain(CB, PerceptionConfig())
        demos = [(np.arange(4), np.arange(4))] * 3
        assert seq.size_bounds_complexity is True
        assert VERIFIER.complexity_limit(seq, demos) < VerifyConfig().max_program_complexity


class TestRedundancy:
    """Composability must be decided by algebra, not by family name.

    The base test prunes any two single steps from the same family when that
    family is declared closed. For grid translations that is right. For 3D
    rotations it is wrong, and the effect was that synthesis pruned every
    rotate pair, leaving no composition of depth 3 or more reachable.
    """

    def test_different_axes_are_not_redundant(self):
        a, b = prog(Rotate("x", 90)), prog(Rotate("d", 120))
        assert not DOM.redundant(a, b)

    def test_same_axis_reduction_is_redundant(self):
        a, b = prog(Rotate("x", 90)), prog(Rotate("x", 180))
        assert DOM.redundant(a, b)

    def test_mixed_kinds_are_not_redundant(self):
        assert not DOM.redundant(prog(Rotate("x", 90)), prog(Scale(2.0)))

    def test_redundancy_agrees_with_brute_force(self):
        """Every pair: pruned iff the composition is literally a single op."""
        cands = DOM.candidates(())
        mats = [_program_matrix(p) for p in cands]
        for i, j in itertools.product(range(len(cands)), repeat=2):
            composed = mats[j] @ mats[i]
            brute = any(np.allclose(composed, mk, atol=1e-9) for mk in mats)
            assert DOM.redundant(cands[i], cands[j]) == brute, (cands[i], cands[j])


class TestVocabulary:
    def test_no_invalid_axis_angle_pairs(self):
        """A 90 deg turn about the cube diagonal is not exactly representable."""
        for p in DOM.candidates(()):
            op = p.ops[0]
            if isinstance(op, Rotate):
                allowed = dict(ROTATION_SET)[op.axis]
                assert op.degrees in allowed, op

    def test_project_is_not_enumerable(self):
        """Project would make the gravity family solvable, so it is not a decoy."""
        assert not any(isinstance(p.ops[0], Project) for p in DOM.candidates(()))

    def test_gravity_is_unreachable_within_synthesis_depth(self):
        want = np.array([0.0, -0.4, 0.0])
        cands = DOM.candidates(())
        assert not any(np.allclose(p.apply(np.array([0.3, -0.4, 0.5])), want, atol=1e-9)
                       for p in cands)
        assert not any(
            np.allclose(DOM.compose(a, b).apply(np.array([0.3, -0.4, 0.5])), want, atol=1e-9)
            for a in cands for b in cands)

    def test_scales_are_distinct_single_ops(self):
        factors = [p.ops[0].factor for p in DOM.candidates(()) if isinstance(p.ops[0], Scale)]
        assert len(factors) == len(set(factors)) == len(SCALES)


class TestDepthAxis:
    """A depth label is a claim about the task and must be true of the task."""

    @pytest.mark.parametrize("family,depth", [("d2_rotate", 2), ("d3_rotate", 3),
                                              ("d4_rotate", 4)])
    def test_shortest_explanation_matches_the_label(self, family, depth):
        for seed in range(3):
            ops = GEN._deep_product(np.random.default_rng(seed), depth)
            assert GEN._shortest_depth(ops, depth) == depth

    def test_ground_truth_reproduces_demos_exactly(self):
        ops = GEN._deep_product(np.random.default_rng(2), 3)
        p = prog(*ops)
        vecs = RNG.normal(size=(4, 3))
        for v in vecs:
            assert np.allclose(p.apply(v), _reapply(ops, v), atol=1e-9)

    def test_deep_task_survives_verification(self):
        """The whole point: a depth-3 program must be accepted, not called memorized."""
        ops = GEN._deep_product(np.random.default_rng(4), 3)
        p = prog(*ops)
        vecs = RNG.normal(size=(3, 3))
        demos = [(v, p.apply(v)) for v in vecs]
        report = VERIFIER.verify(DOM, p, demos)
        assert report.exact, report.reasons
        assert report.passed, report.reasons

    def test_deep_product_is_not_merely_relabelled(self):
        """Guards the original defect: k random rotations that collapse to one op.

        The first version of this benchmark composed k random rotations and
        called the family d{k}. Because the rotations generate the octahedral
        group, most such products need one or two steps, so every "deep" family
        was secretly a depth-1 family and measured nothing.
        """
        rotations = [Rotate(a, d) for a, angles in ROTATION_SET for d in angles]
        shallow = sum(1 for seed in range(20)
                      if GEN._shortest_depth(tuple(RNG.choice(rotations) for _ in range(3)), 3) != 3)
        assert shallow > 0, "rotations alone should not be able to express depth 3"
