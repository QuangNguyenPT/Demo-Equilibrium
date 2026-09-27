"""Domains (grid, sequence), the domain registry, and the independent verifier."""
import numpy as np
import pytest

from hrm.config import PerceptionConfig, SubstrateConfig, VerifyConfig
from hrm.perception import (DomainRegistry, GridDomain, Program, ProgramError, SequenceDomain)
from hrm.perception.grid import (ColorCycle, Dihedral, GridProgram, RecolorMap, Translate)
from hrm.perception.grid.symmetry import D4, D4_INVERSE, apply as sym_apply
from hrm.perception.sequence import Affine, CumSum, Diff, Rotate, SequenceProgram
from hrm.substrate import Codebook, PhasorVector
from hrm.verify import Verifier

CB = Codebook(SubstrateConfig(dim=1024))
GRID = GridDomain(CB, PerceptionConfig(grid_max_shift=3))
SEQ = SequenceDomain(CB, PerceptionConfig())
VERIFIER = Verifier(VerifyConfig())
RNG = np.random.default_rng(5)


def prog(*prims) -> GridProgram:
    return GridProgram(tuple(prims))


def grid_demos(program, n=3, size=8):
    demos = []
    for _ in range(n):
        g = np.zeros((size, size), dtype=np.int64)
        x, y = RNG.integers(1, size - 4, 2)
        g[y:y + 2, x:x + 2] = int(RNG.integers(1, 9))
        g[y + 2, x] = int(RNG.integers(1, 9))
        demos.append((g, program.apply(g)))
    return demos


def top_by_shared(domain, demos):
    task = domain.encode_task(demos)
    cands = domain.candidates(demos)
    sims = [task.shared @ c.signature(CB) for c in cands]
    return cands[int(np.argmax(sims))], max(sims)


class TestGridOperators:
    def test_dihedral_closure_and_inverse(self):
        g = RNG.integers(0, 5, size=(4, 7))
        for name, inv in D4_INVERSE.items():
            assert np.array_equal(sym_apply(inv, sym_apply(name, g)), g), name

    def test_eight_distinct_dihedral_variants(self):
        g = np.arange(12).reshape(3, 4)
        variants = {D4[n](g).tobytes() + str(D4[n](g).shape).encode() for n in D4}
        assert len(variants) == 8

    @pytest.mark.parametrize("shift", [(2, -1, 0), (-3, 2, 1), (0, 0, -1)])
    def test_vectorized_translate_matches_reference_loop(self, shift):
        vol = RNG.integers(0, 4, size=(3, 6, 7))
        dx, dy, dz = shift
        ref = np.zeros_like(vol)
        for z, y, x in zip(*np.nonzero(vol)):
            if 0 <= x + dx < 7 and 0 <= y + dy < 6 and 0 <= z + dz < 3:
                ref[z + dz, y + dy, x + dx] = vol[z, y, x]
        assert np.array_equal(prog(Translate(dx, dy, dz)).apply(vol), ref)

    def test_color_cycle_has_period_nine_on_nonzero_colors(self):
        g = RNG.integers(0, 10, size=(5, 5))
        assert np.array_equal(prog(ColorCycle(9)).apply(g), g)
        assert np.array_equal(prog(*[ColorCycle(1)] * 9).apply(g), g)

    def test_payload_round_trip(self):
        p = prog(Translate(2, -1), RecolorMap(((1, 3), (2, 5))), Dihedral("r90"))
        back = GRID.program_from_payload(p.to_payload())
        assert back.describe() == p.describe()
        g = RNG.integers(0, 4, size=(6, 6))
        assert np.array_equal(back.apply(g), p.apply(g))

    def test_non_grid_input_is_a_program_error(self):
        with pytest.raises(ProgramError):
            prog(Translate(1, 0)).apply([1, 2, 3])


class TestGridEncodingAlgebra:
    def test_translation_task_points_at_its_program(self):
        best, sim = top_by_shared(GRID, grid_demos(prog(Translate(2, 0))))
        assert best.describe() == "translate(dx=2, dy=0)" and sim > 0.5

    def test_color_cycle_task_points_at_its_program(self):
        best, sim = top_by_shared(GRID, grid_demos(prog(ColorCycle(3))))
        assert best.describe() == "color_cycle(shift=3)" and sim > 0.9

    def test_observed_recolor_becomes_a_candidate(self):
        demos = []
        for _ in range(3):
            g = np.zeros((6, 6), dtype=np.int64)
            g[1, 1:3], g[4, 2:5] = 1, 2
            demos.append((g, prog(RecolorMap(((1, 3), (2, 5)))).apply(g)))
        assert "recolor({1: 3, 2: 5})" in {c.describe() for c in GRID.candidates(demos)}

    @pytest.mark.parametrize("x,ok", [
        (np.zeros((3, 3), dtype=np.int64), True),
        (np.zeros((2, 3, 3), dtype=np.int64), True),
        ([1, 2, 3], False),
        (np.zeros((3, 3)), False),                 # float grid
        (np.full((2, 2), 12), False),              # color out of range
        ([[1, 2], [3]], False),                    # ragged
    ])
    def test_accepts(self, x, ok):
        assert GRID.accepts([(x, x)]) is ok

    def test_similarity_is_one_iff_equal(self):
        a = RNG.integers(0, 5, size=(5, 5))
        b = a.copy()
        b[0, 0] = (b[0, 0] + 1) % 5
        assert GRID.similarity(a, a) == 1.0
        assert 0.0 < GRID.similarity(a, b) < 1.0
        assert GRID.similarity(a, np.zeros((4, 4), dtype=np.int64)) == 0.0


class TestSequenceDomain:
    def test_ops(self):
        assert Rotate(1).run([1, 2, 3]) == [3, 1, 2]
        assert CumSum().run([1, 2, 3]) == [1, 3, 6]
        with pytest.raises(ProgramError):
            Diff().run([4])

    def test_offset_task_points_at_its_program(self):
        demos = [([1, 4, 2, 8], [4, 7, 5, 11]), ([0, 5], [3, 8]), ([9, 9, 1], [12, 12, 4])]
        best, sim = top_by_shared(SEQ, demos)
        assert best.describe() == "affine(a=1, b=3)" and sim > 0.95

    def test_fitted_affine_outside_enumerated_range(self):
        demos = [([1, 2, 3], [-13, -6, 1]), ([0, 5], [-20, 15])]
        assert "affine(a=7, b=-20)" in {c.describe() for c in SEQ.candidates(demos)}

    def test_payload_round_trip(self):
        p = SequenceProgram((Affine(2, -1), Rotate(2), Diff()))
        back = SEQ.program_from_payload(p.to_payload())
        assert back.describe() == p.describe()
        assert back.apply([3, 1, 4, 1, 5]) == p.apply([3, 1, 4, 1, 5])

    def test_similarity_is_one_iff_equal(self):
        assert SEQ.similarity([1, 2, 3], [1, 2, 3]) == 1.0
        assert 0.0 < SEQ.similarity([1, 2, 3], [1, 2, 4]) < 1.0
        assert SEQ.similarity([1, 2], [[1, 2]]) == 0.0


class TestRegistry:
    def test_routes_by_representation(self):
        reg = DomainRegistry([GRID, SEQ])
        assert reg.for_demos([([1, 2], [2, 3])]) is SEQ
        assert reg.for_demos([(np.ones((2, 2), dtype=np.int64),) * 2]) is GRID
        with pytest.raises(LookupError):
            reg.for_demos([("text", "more text")])

    def test_duplicate_name_rejected(self):
        with pytest.raises(ValueError):
            DomainRegistry([GRID, GRID])


class TestVerifier:
    def test_correct_program_passes(self):
        demos = grid_demos(prog(Translate(2, 0)))
        rep = VERIFIER.verify(GRID, prog(Translate(2, 0)), demos)
        assert rep.passed and rep.exact and rep.margin == 1.0 and not rep.reasons

    def test_wrong_program_fails_with_reason(self):
        rep = VERIFIER.verify(GRID, prog(Translate(1, 1)), grid_demos(prog(Translate(2, 0))))
        assert not rep.passed and "not exact" in rep.reasons[0]

    def test_memorization_refused_even_when_exact(self):
        tiny = [(np.array([[1, 2]]), np.array([[4, 5]])), (np.array([[3, 0]]), np.array([[6, 0]]))]
        assert VERIFIER.verify(GRID, prog(ColorCycle(3)), tiny).passed
        rep = VERIFIER.verify(GRID, prog(*[ColorCycle(1)] * 3), tiny)
        assert rep.exact and not rep.passed and "memorization" in rep.reasons[-1]

    def test_mdl_prefers_simpler(self):
        assert VERIFIER.mdl(prog(Translate(2, 0)), 1.0) > \
            VERIFIER.mdl(prog(Translate(1, 0), Translate(1, 0)), 1.0)

    def test_inapplicable_program_counts_as_failure(self):
        ev = VERIFIER.evaluate(SEQ, SequenceProgram((Diff(),)), [([5], []), ([1, 3], [2])])
        assert not ev.exact and ev.margin == 0.0

    def test_energy(self):
        demos = [([1, 2], [2, 3])]
        assert VERIFIER.energy(SEQ, SequenceProgram((Affine(1, 1),)), demos) == (0.0, 1.0)
        energy, margin = VERIFIER.energy(SEQ, SequenceProgram((Affine(1, 2),)), demos)
        assert 0.0 < energy < 1.0 and margin < 1.0

    def test_nondeterministic_program_refused(self):
        class Toggle(Program):
            """Exact on its first call, different on the next: only determinism can catch it."""
            calls = 0

            def apply(self, x):
                Toggle.calls += 1
                return [0] if Toggle.calls % 2 else [1]

            def describe(self):
                return "toggle"

            def signature(self, cb):
                return PhasorVector.identity(cb.dim)

            @property
            def complexity(self):
                return 1

            def to_payload(self):
                return {}

        rep = VERIFIER.verify(SEQ, Toggle(), [([0], [0])])
        assert rep.exact and not rep.passed
        assert "non-deterministic program" in rep.reasons
