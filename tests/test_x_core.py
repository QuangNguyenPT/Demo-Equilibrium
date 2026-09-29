"""HRM-X core invariants: equality is congruent and explained, contradictions name their premises,
laws are conjectured only when tested, and the controller stops when no action is worth its cost."""
from fractions import Fraction

from hrm_x.induction import behaviour_classes, consistent, discover_laws
from hrm_x.rules import Rule, saturate
from hrm_x.workspace import Why, Workspace


def test_congruence_closure():
    ws = Workspace()
    a, b = ws.add("a"), ws.add("b")
    fa, fb = ws.add("f", [a]), ws.add("f", [b])
    ws.union(a, b, Why("given", frozenset({"p"})))
    ws.rebuild()
    assert ws.same(fa, fb)
    assert ws.assumptions(fa, fb) == frozenset({"p"})       # f(a)=f(b) is explained by a=b


def test_contradiction_is_recorded_not_merged_and_names_premises():
    ws = Workspace()
    x = ws.add("x")
    ws.union(x, ws.const("Q", 4), Why("given", frozenset({"s0"})))
    ws.union(x, ws.const("Q", 5), Why("given", frozenset({"s1"})))
    assert ws.value(x) == 4
    assert len(ws.conflicts) == 1 and ws.conflicts[0].assumptions == frozenset({"s0", "s1"})


def test_a_constant_survives_when_its_class_is_the_smaller_one():
    """Regression: union-by-size swapped the roots but not the constants, dropping the value."""
    ws = Workspace()
    four = ws.const("Q", 4)
    big = ws.add("x")
    ws.union(big, ws.add("y"), Why("given"))                 # class of size 2, no constant
    ws.union(four, big, Why("given"))                        # the constant's class (size 1) is smaller
    assert ws.value(big) == 4 and ws.value(four) == 4


def test_congruence_conflict_names_the_equality_that_caused_it():
    ws = Workspace()
    a, b = ws.add("a"), ws.add("b")
    ws.union(ws.add("f", [a]), ws.const("Q", 4), Why("given", frozenset({"s0"})))
    ws.union(ws.add("f", [b]), ws.const("Q", 5), Why("given", frozenset({"s1"})))
    ws.union(a, b, Why("given", frozenset({"s2"})))
    ws.rebuild()
    assert ws.conflicts and ws.conflicts[-1].assumptions == frozenset({"s0", "s1", "s2"})


def test_rule_firing_carries_the_premises_it_used():
    """A computed value is explained down to the facts its inputs came from."""
    ws = Workspace()
    q0 = ws.add("q0")
    ws.union(q0, ws.const("Q", 2), Why("given", frozenset({"s0"})))
    s = ws.add("sub", [q0, ws.const("Q", 3)])
    q1 = ws.add("q1")
    ws.union(q1, s, Why("event", frozenset({"s2"})))
    fold = Rule("exact:sub", ("sub", "?a", "?b"),
                lambda w, m: w.const("Q", w.value(m["?a"]) - w.value(m["?b"])),
                lambda w, m: w.value(m["?a"]) is not None and w.value(m["?b"]) is not None)
    saturate(ws, [fold], 100)
    assert ws.value(q1) == -1
    assert ws.assumptions(q1, ws.const_node(q1)) == frozenset({"s0", "s2"})


def test_extraction_picks_the_cheapest_equal_term():
    ws = Workspace()
    slow, fast = ws.add("slow"), ws.add("fast")
    ws.union(slow, fast, Why("law"))
    cost = {"slow": 100.0, "fast": 1.0}
    best = ws.extract(slow, lambda w, node, kids: cost[node[0]] + sum(kids))
    assert best == (1.0, ("fast", None, ()))


def test_behaviour_classes_merge_equivalent_programs_and_sum_their_mass():
    progs = [("x+y", 3), ("y+x", 3), ("x", 1)]
    ev = lambda p, xy: {"x+y": xy[0] + xy[1], "y+x": xy[1] + xy[0], "x": xy[0]}[p]
    classes = behaviour_classes(progs, ev, [(1, 2), (3, 5)])
    assert len(classes) == 2
    sums = [h for h in classes if h.program == "x+y"][0]
    assert sums.members == 2 and sums.weight == 2 * 2 ** -3
    assert [h.program for h in consistent(classes, [((2, 2), 4)], ev)] == ["x+y"]


def test_laws_are_tested_conjectures_and_redundant_ones_are_pruned():
    universe = ["?x", "?y", ("add", "?x", "?y"), ("add", "?y", "?x"), ("sub", "?x", "?y"), ("sub", "?y", "?x")]

    def ev(t, env):
        if isinstance(t, str):
            return env[t]
        a, b = ev(t[1], env), ev(t[2], env)
        return a + b if t[0] == "add" else a - b

    envs = [{"?x": Fraction(i), "?y": Fraction(2 * i + 1)} for i in range(8)]
    make = lambda law: [Rule(str(law), law.lhs, law.rhs), Rule(str(law), law.rhs, law.lhs)]
    laws = discover_laws(universe, ev, envs, make)
    assert [str(l) for l in laws] == ["add(x, y) = add(y, x)"]      # subtraction is not commutative
