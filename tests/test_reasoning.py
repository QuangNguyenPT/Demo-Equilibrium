"""Two-level reasoning: controller rails, equilibrium engine, reasoner end-to-end."""
import numpy as np
import pytest

from hrm.config import (ModelConfig, PerceptionConfig, ReasoningConfig, SubstrateConfig)
from hrm.knowledge import KnowledgeRecord, KnowledgeStore
from hrm.perception import DomainRegistry, GridDomain, SequenceDomain
from hrm.perception.grid import ColorCycle, GridProgram, Translate
from hrm.perception.sequence import Affine, Reverse, SequenceProgram
from hrm.reasoning import (Action, BudgetController, CandidateSet, ControllerFeatures,
                           EquilibriumEngine, PriorTable, QHead, Reasoner, RestartPolicy)
from hrm.substrate import Codebook
from hrm.trace import Trace
from hrm.verify import Verifier

CFG = ModelConfig(substrate=SubstrateConfig(dim=512),
                  perception=PerceptionConfig(grid_max_shift=3),
                  reasoning=ReasoningConfig(max_ticks=300, restart_budget=3))
CB = Codebook(CFG.substrate)
GRID = GridDomain(CB, CFG.perception)
SEQ = SequenceDomain(CB, CFG.perception)
RNG = np.random.default_rng(11)


def reasoner(**kw) -> Reasoner:
    return Reasoner(CFG, CB, DomainRegistry([GRID, SEQ]), **kw)


def grid_task(program, n=3, size=8):
    demos = []
    for _ in range(n + 1):
        g = np.zeros((size, size), dtype=np.int64)
        x, y = RNG.integers(1, size - 4, 2)
        g[y:y + 2, x:x + 2] = int(RNG.integers(1, 7))
        demos.append((g, program.apply(g)))
    return demos[:n], demos[n][0], demos[n][1]


def features(**kw) -> ControllerFeatures:
    base = dict(residual=1.0, residual_slope=0.0, energy=1.0, diversity=1.0,
                memory_confidence=0.0, verification_margin=0.0, plateau_count=0,
                restarts_used=0)
    base.update(kw)
    return ControllerFeatures(**base)


class TestController:
    def test_conjunctive_accept(self):
        c = BudgetController(CFG.reasoning)
        assert c.decide(features(residual=1e-9, energy=0.0,
                                 verification_margin=1.0)).action is Action.ACCEPT

    def test_false_attractor_restarts_never_accepts(self):
        c = BudgetController(CFG.reasoning)
        assert c.decide(features(residual=1e-9, energy=0.9)).action is Action.RESTART

    def test_abstain_when_budget_exhausted(self):
        c = BudgetController(CFG.reasoning)
        f = features(residual=1e-9, energy=0.9, restarts_used=CFG.reasoning.restart_budget)
        assert c.decide(f).action is Action.ABSTAIN

    def test_budget_exhausted_abstains_even_when_semantic_passes(self):
        """Regression: kinetic + semantic + NOT verified + budget spent used to
        fall through to CONTINUE and spin to max_ticks (measured 2x on benchmark
        v1 at residual 0.0, energy 0.05, margin 0.84). It must terminate."""
        c = BudgetController(CFG.reasoning)
        f = features(residual=1e-9, energy=0.05, verification_margin=0.84,
                     plateau_count=CFG.reasoning.plateau_cycles * CFG.reasoning.ticks_per_cycle,
                     restarts_used=CFG.reasoning.restart_budget)
        d = c.decide(f)
        assert d.action is Action.ABSTAIN
        assert "unreachable" in d.reason

    def test_stuck_run_never_continues_once_budget_is_spent(self):
        """A plateaued run (or a false attractor) with the restart budget spent
        must terminate. Continuing there could only chase an ACCEPT that a
        non-converging hypothesis cannot reach. A run still making progress
        legitimately CONTINUEs and is bounded by max_ticks instead."""
        c = BudgetController(CFG.reasoning)
        stuck = (CFG.reasoning.plateau_cycles * CFG.reasoning.ticks_per_cycle)
        checked = 0
        for energy in (0.0, 0.05, 0.5, 0.99):
            for margin in (0.0, 0.84, 0.97, 1.0):
                for residual in (0.0, 0.04, 0.06, 1.0):
                    for plateau in (0, 3, 39, 40, 400):
                        f = features(residual=residual, energy=energy,
                                     verification_margin=margin, plateau_count=plateau,
                                     restarts_used=CFG.reasoning.restart_budget)
                        action = c.decide(f).action
                        false_attractor = residual < CFG.reasoning.eps_kinetic and \
                            energy >= CFG.reasoning.eps_energy
                        if plateau >= stuck or false_attractor:
                            checked += 1
                            assert action is not Action.CONTINUE
                        elif action is not Action.ACCEPT:
                            # not stuck, rails not met: legitimately keep going,
                            # bounded by max_ticks
                            assert action is Action.CONTINUE
        assert checked > 100                       # the sweep really did cover the stuck half

    def test_lambda_stays_in_km_interval(self):
        c = BudgetController(CFG.reasoning)
        for e in (0.0, 0.5, 5.0, 1e9):
            assert CFG.reasoning.lam_min <= c.decide(features(energy=e)).lam <= CFG.reasoning.lam_max

    def test_adversarial_qhead_cannot_grant_accept(self):
        q = QHead()
        q.w[:] = 1e6                                    # maximally eager weights
        c = BudgetController(CFG.reasoning, qhead=q)
        for plateau in range(0, 50, 7):
            d = c.decide(features(residual=0.5, energy=0.5, plateau_count=plateau))
            assert d.action is not Action.ACCEPT

    def test_restart_anneals_and_parse_pick_avoids_failures(self):
        rp = RestartPolicy(CFG.reasoning, np.random.default_rng(0))
        s0 = rp.sigma()
        rp.kick(np.ones((2, 8), dtype=np.complex64))
        assert rp.sigma() == pytest.approx(s0 * CFG.reasoning.restart_gamma)
        picks = [rp.pick_parse(4, current=0, failed=[1]) for _ in range(300)]
        assert 0 not in picks and picks.count(1) < picks.count(2)
        assert rp.pick_parse(1, current=0, failed=[]) is None


class TestQHead:
    def test_td_update_reduces_error(self):
        q = QHead(lr=0.2)
        from hrm.reasoning import Transition, featurize
        t = Transition(featurize(features()), 1, 1.0, None)
        first = q.update([t])
        for _ in range(20):
            last = q.update([t])
        assert last < first


class TestEngine:
    def _setup(self, program):
        demos, _, _ = grid_task(program)
        verifier = Verifier(CFG.verify)
        task = GRID.encode_task(demos)
        cands = CandidateSet.build(GRID.candidates(demos), CB)
        return demos, task, cands, (lambda p: verifier.energy(GRID, p, demos))

    def test_deep_level_converges_to_the_explanation(self):
        _, task, cands, energy_fn = self._setup(GridProgram((Translate(2, 0),)))
        res = EquilibriumEngine(CFG.reasoning, seed=1).run(task, cands, energy_fn, Trace())
        assert res.status == "accept"
        assert res.program.describe() == "translate(dx=2, dy=0)"

    def test_state_stays_on_torus(self):
        _, task, cands, _ = self._setup(GridProgram((Translate(2, 0),)))
        engine = EquilibriumEngine(CFG.reasoning, seed=1)
        z = engine.init_state(task.shared)
        scores = engine.resonator.task_scores(cands, task)
        for _ in range(5):
            z, _ = engine.tick(z, cands, scores, 0.5)
        assert np.allclose(np.abs(z), 1.0, atol=1e-4)

    def test_cached_conjugates_follow_candidate_changes(self):
        _, task, cands, _ = self._setup(GridProgram((Translate(2, 0),)))
        before = cands.conj_t
        cands.raise_prior(GridProgram((Translate(1, 0), ColorCycle(2))), 0.5, CB)
        assert cands.conj_t.shape[1] == before.shape[1] + 1
        direct = (task.shared.data @ np.conj(cands.signatures).T).real / cands.signatures.shape[1]
        assert np.allclose(cands.similarities(task.shared.data), direct)

    def test_restart_swaps_parse_and_uses_new_encoding(self):
        demos = [(RNG.integers(0, 5, (6, 6)), RNG.integers(0, 5, (6, 6))) for _ in range(3)]
        verifier = Verifier(CFG.verify)
        task = GRID.encode_task(demos)
        cands = CandidateSet.build(GRID.candidates(demos), CB)
        seen = []

        def hook(t, failed):
            nxt = (t.parse + 1) % GRID.n_parses
            seen.append(nxt)
            return GRID.encode_task(demos, nxt), cands

        trace = Trace()
        res = EquilibriumEngine(CFG.reasoning, seed=2).run(
            task, cands, lambda p: verifier.energy(GRID, p, demos), trace, on_restart=hook)
        assert res.status in ("abstain", "ceiling")
        swaps = [e.detail for e in trace.of_kind("restart") if "parse" in e.detail]
        assert swaps and all(d["parse"] != d["failed_parse"] for d in swaps)
        assert res.task.parse == seen[-1]


class TestReasoner:
    def test_fast_path_solves_translation(self):
        demos, test_in, expected = grid_task(GridProgram((Translate(2, 0),)))
        sol = reasoner().solve(demos, test_in)
        assert sol.status == "fast" and sol.ticks == 0
        assert np.array_equal(sol.answer, expected)
        assert sol.confidence == 1.0 and sol.report.passed

    def test_sequence_task_routed_and_solved(self):
        sol = reasoner().solve([([1, 4, 2], [4, 7, 5]), ([0, 9], [3, 12])], [5, 5, 1])
        assert sol.domain == "sequence" and sol.answer == [8, 8, 4] and sol.solved

    def test_composite_is_synthesized(self):
        program = GridProgram((Translate(1, 0), ColorCycle(2)))
        demos, test_in, expected = grid_task(program)
        sol = reasoner().solve(demos, test_in)
        assert sol.status == "synthesized"
        assert np.array_equal(sol.answer, expected)

    def test_sequence_composite_is_synthesized(self):
        program = SequenceProgram((Affine(2, 0), Reverse()))
        demos = [(x, program.apply(x)) for x in ([1, 2, 3], [4, 0, 7, 1], [5, 6])]
        sol = reasoner().solve(demos, [9, 8])
        assert sol.status == "synthesized" and sol.answer == [16, 18]

    def test_abstains_instead_of_guessing(self):
        demos = [(RNG.integers(0, 5, (6, 6)), RNG.integers(0, 5, (6, 6))) for _ in range(3)]
        sol = reasoner().solve(demos, np.zeros((6, 6), dtype=np.int64))
        assert sol.status == "abstained" and sol.answer is None and sol.confidence == 0.0

    def test_explanation_that_cannot_apply_to_test_input_abstains(self):
        sol = reasoner().solve([([1, 3, 6], [2, 3]), ([2, 2, 5], [0, 3])], [7])
        assert sol.status == "abstained"
        assert "does not apply" in sol.trace.of_kind("abstain")[-1].detail["reason"]

    def test_ambiguity_lowers_confidence_and_offers_second_guess(self):
        sol = reasoner().solve([([1, 1], [2, 2])], [3, 5])
        assert sol.solved and sol.confidence < 0.5
        assert sol.second_guess is not None and sol.second_guess != sol.answer

    def test_consolidated_prior_turns_synthesis_into_recognition(self):
        program = GridProgram((Translate(1, 0), ColorCycle(2)))
        priors = PriorTable()
        priors.consolidate("grid", program, amount=0.5)
        demos, test_in, expected = grid_task(program)
        sol = reasoner(priors=priors).solve(demos, test_in)
        assert sol.status == "fast" and np.array_equal(sol.answer, expected)

    def test_ssd_knowledge_turns_synthesis_into_recognition(self, tmp_path):
        with KnowledgeStore(tmp_path / "k", CFG.substrate, CFG.knowledge) as store:
            r = reasoner(knowledge=store)
            program = GridProgram((Translate(1, 0), ColorCycle(2)))
            demos, test_in, _ = grid_task(program)
            first = r.solve(demos, test_in, "t1")
            assert first.status == "synthesized"
            assert store.stats()["rows"] == 0                  # reasoning never writes
            store.write(first.task_key,
                        KnowledgeRecord("program", first.domain, first.program.describe(),
                                        first.program.to_payload(), first.program.complexity,
                                        "t1", "synthesized"), first.report)
            demos2, test2, expected2 = grid_task(program)
            second = r.solve(demos2, test2, "t2")
            assert second.status == "fast" and second.retrieved
            assert np.array_equal(second.answer, expected2)
