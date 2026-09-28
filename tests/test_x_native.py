"""The native backend must be indistinguishable from the Python reference (the blueprint).

Every check below runs the same inputs through both and demands identical results, including
failure outcomes, explanations and cost counters. Skipped when the library is not built.
"""
import random

import numpy as np
import pytest

from hrm_x import native
from hrm_x.domains import grid as G
from hrm_x.rules import Lit, Rule, matches, op_index, saturate
from hrm_x.workspace import Why, Workspace

pytestmark = pytest.mark.skipif(not native.available(), reason="native backend not built (scripts/build_native.ps1)")


def random_grid(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    g = np.zeros((h, w), dtype=np.int64)
    for _ in range(rng.integers(1, 6)):
        r, c = rng.integers(0, h), rng.integers(0, w)
        hh, ww = rng.integers(1, 4), rng.integers(1, 4)
        g[r:r + hh, c:c + ww] = rng.integers(1, 5)
    return g


def test_components_match_the_reference():
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = random_grid(rng, int(rng.integers(1, 12)), int(rng.integers(1, 12)))
        labels, n = native.components(g)
        objs = G.objects(g)
        assert n == len(objs)
        for k, o in enumerate(objs, start=1):
            assert set(zip(*np.nonzero(labels == k))) == set(o.cells)


def test_every_object_step_matches_the_reference_including_failures():
    rng = np.random.default_rng(1)
    for _ in range(60):
        g = random_grid(rng, int(rng.integers(3, 10)), int(rng.integers(3, 10)))
        for step in G.candidate_steps([(g, g)]):
            if step[0] == "grid":
                continue
            for strict in (G.LENIENT, G.EXERCISED, G.APPLICABLE):
                try:
                    ref = G.apply_step_reference(step, g, strict)
                except G.Invalid:
                    ref = None
                got = native.object_step(step[0], step[1], g, strict)
                assert (ref is None) == (got is None), (step, strict)
                if ref is not None:
                    assert np.array_equal(ref, got), (step, strict)


def _drive(ws, rng: random.Random, steps: int):
    """A random but reproducible history of adds, unions (some conflicting) and rebuilds."""
    ids = [ws.const("Q", k) for k in range(3)] + [ws.add("v", (), k) for k in range(6)]
    for step in range(steps):
        roll = rng.random()
        if roll < 0.45:
            op = rng.choice(["f", "g"])
            ids.append(ws.add(op, [rng.choice(ids) for _ in range(1 if op == "f" else 2)]))
        elif roll < 0.9:
            a, b = rng.choice(ids), rng.choice(ids)
            ws.union(a, b, Why(f"u{step}", frozenset({f"s{step}"})))
        else:
            ws.rebuild()
    ws.rebuild()
    return ids


def test_egraph_histories_are_identical_on_both_backends():
    for seed in range(40):
        py, nat = Workspace(), native.NativeWorkspace()
        ids_py = _drive(py, random.Random(seed), 120)
        ids_nat = _drive(nat, random.Random(seed), 120)
        assert ids_py == ids_nat
        assert py.classes() == nat.classes()
        assert [py.find(i) for i in ids_py] == [nat.find(i) for i in ids_nat]
        assert [py.value(i) for i in ids_py] == [nat.value(i) for i in ids_nat]
        assert [list(py.nodes(c)) for c in py.classes()] == [list(nat.nodes(c)) for c in nat.classes()]
        assert py.ops == nat.ops
        assert [(c.values, c.assumptions) for c in py.conflicts] == [(c.values, c.assumptions) for c in nat.conflicts]
        for a in ids_py[:25]:
            for b in ids_py[:25]:
                if py.same(a, b):
                    assert py.explain(a, b) == nat.explain(a, b)


def test_ematch_and_saturation_agree():
    fold = Rule("fold", ("add", "?a", "?b"),
                lambda w, s: w.const("Q", w.value(s["?a"]) + w.value(s["?b"])),
                lambda w, s: w.value(s["?a"]) is not None and w.value(s["?b"]) is not None)
    comm = Rule("comm", ("add", "?a", "?b"), ("add", "?b", "?a"))
    assoc = Rule("assoc", ("add", ("add", "?a", "?b"), "?c"), ("add", "?a", ("add", "?b", "?c")))
    anyc = Rule("anyc", Lit(None), lambda w, s: None)
    for seed in range(10):
        results = []
        for ws in (Workspace(), native.NativeWorkspace()):
            rng = random.Random(seed)
            leaves = [ws.const("Q", rng.randint(0, 5)) for _ in range(4)] + [ws.add("x", (), k) for k in range(3)]
            for _ in range(12):
                leaves.append(ws.add("add", [rng.choice(leaves), rng.choice(leaves)]))
            index = op_index(ws)
            found = [(r.name, matches(ws, r, index)) for r in (comm, assoc, anyc)]
            sat = saturate(ws, [fold, comm, assoc], max_firings=300)
            results.append((found, (sat.firings, sat.rounds, sat.saturated), ws.ops, ws.classes(),
                            [ws.value(c) for c in ws.classes()]))
        assert results[0] == results[1]


def test_domain_solvers_agree_across_backends(monkeypatch):
    from hrm_x.domains.arith import Arith, ArithTask
    from hrm_x.mind import Mind

    def run(backend: str):
        monkeypatch.setenv("HRMX_BACKEND", backend)
        mind, dom = Mind(), Arith()
        first = mind.solve(dom, ArithTask.from_text(["2 × 2 = 4", "2 × 3 = 6"], "10 × 3.1", oracle=lambda a, b: a * b))
        laws, _ = dom.consolidate(mind, "×", max_size=5, probes=12)
        later = [mind.solve(dom, ArithTask.from_text([], q)) for q in ("1000 × 3.1", "2 × 3 + 40 × 5.5")]
        return ([str(l) for l in laws], first.answer, [(r.answer, r.ops, r.cycles) for r in later])

    assert run("python") == run("native")
