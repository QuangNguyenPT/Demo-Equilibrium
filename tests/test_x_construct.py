"""HRM-X v0.6: concept construction, concept memory, the language interface.

Each test pins one claim of `vault/10 - Research Program/architecture_spec.md`; the failure
condition is the point of the test, not the happy path.
"""
import math
from fractions import Fraction as F

import pytest

from hrm_x import construct as C
from hrm_x.concept_memory import ConceptMemory
from hrm_x.concepts import ConceptConfig
from hrm_x.domains.concept import Concept, ConceptTask
from hrm_x.language import Conversation, DecimalReader, LearnedReader, parse
from hrm_x.mind import Controller, Mind


def _task(op, pairs, query, symbol="?", oracle=True):
    return ConceptTask.from_op(op, [(F(a), F(b)) for a, b in pairs], (F(query[0]), F(query[1])),
                               symbol=symbol, oracle=op if oracle else None)


def _labels(size):
    return {C.render(f.expr) for f in C.features(size)}


# ---------- construction: the feature space ----------

def test_max_and_square_are_constructed_not_written():
    """Neither is among the 68 written families; both appear as short features."""
    assert "(x * x)" in _labels(3)
    assert "ifgt(x, y, x, y)" in _labels(5)


def test_affine_images_share_one_feature_and_the_base_span_is_not_new():
    """min = x + y - max and |x - y| = 2 max - x - y: one class. x - y and xy + 1 are in span{1,x,y,xy}."""
    labels = _labels(5)
    assert "ifgt(x, y, y, x)" not in labels             # min: same class as max
    assert "(x - y)" not in labels and "((x * y) + 1)" not in labels
    keys = {}
    for f in C.features(7):
        keys.setdefault(C.render(f.expr), f)
    assert len(C.features(7)) == len({tuple(f.values) for f in C.features(7)})


def test_evaluation_is_exact_and_undefined_is_none():
    expr = next(f.expr for f in C.features(3) if C.render(f.expr) == "(x / y)")
    assert C.evaluate(expr, F(1), F(3)) == F(1, 3)
    assert C.evaluate(expr, F(1), F(0)) is None


def test_ordering_cannot_change_the_answer():
    """Construction is exhaustive within a level, so the order of the features cannot matter."""
    task = _task(max, [(2, 3), (4, 1), (5, 5), (1, 6), (8, 2)], (7, 2))
    cfg = ConceptConfig()
    feats = C.features(7)
    fn_of = lambda f: (lambda x, y, e=f.expr: C.evaluate(e, x, y))
    label = lambda f: C.render(f.expr)
    by_mdl = C.construct(task.evidence(), cfg, sorted(feats, key=lambda f: f.bits), fn_of, label)
    import random
    shuffled = list(feats)
    random.Random(3).shuffle(shuffled)
    by_rnd = C.construct(task.evidence(), cfg, shuffled, fn_of, label)
    assert by_mdl.fitted[0].family.name == by_rnd.fitted[0].family.name
    assert by_mdl.fitted[0].params == by_rnd.fitted[0].params
    assert by_mdl.examined == by_rnd.examined == len(feats)          # exhaustive within the level


# ---------- the mind: construct, refuse, remember, transfer ----------

def test_the_mind_constructs_max_and_remembers_it_as_a_candidate():
    mind = Mind(Controller())
    r = mind.solve(Concept(), _task(max, [(2, 3), (4, 1), (5, 5), (1, 6)], (7, 2), "⊕"))
    assert r.status == "answer" and r.answer == F(7)
    mem = ConceptMemory.of(mind)
    assert [mem.text(k) for k in range(len(mem.concepts))] == ["ifgt(x, y, x, y)"]
    assert mem.concepts[0].status == "candidate"


def test_a_second_task_admits_the_concept_and_it_then_lowers_the_evidence_needed():
    """Transfer (U7): |x - y| from 4 rows is refused from scratch, answered once max is a concept."""
    absdiff = lambda a, b: abs(a - b)
    fresh = Mind(Controller()).solve(Concept(), _task(absdiff, [(2, 7), (9, 4), (3, 3), (1, 8)], (6, 10), "⊖"))
    assert fresh.status == "abstain"

    mind = Mind(Controller())
    mind.solve(Concept(), _task(max, [(2, 3), (4, 1), (5, 5), (1, 6)], (7, 2), "⊕"))
    mind.solve(Concept(), _task(max, [(6, 1), (2, 9), (4, 4), (8, 3)], (1, 5), "⊗"))
    assert ConceptMemory.of(mind).concepts[0].status == "admitted"
    r = mind.solve(Concept(), _task(absdiff, [(2, 7), (9, 4), (3, 3), (1, 8)], (6, 10), "⊖"))
    assert r.status == "answer" and r.answer == F(4)
    assert not any(s.target.startswith("level 1") for s in r.trace if s.action == "propose")

    no_memory = Mind(Controller())
    no_memory.knowledge = mind.knowledge
    ablated = no_memory.solve(Concept(use_memory=False), _task(absdiff, [(2, 7), (9, 4), (3, 3), (1, 8)], (6, 10), "⊘"))
    assert ablated.status == "abstain"


def test_an_operator_outside_the_primitives_is_refused_not_guessed():
    gcd = lambda a, b: F(math.gcd(int(a), int(b)))
    r = Mind(Controller()).solve(Concept(), _task(gcd, [(4, 6), (9, 6), (5, 7), (8, 12)], (10, 15)))
    assert r.status == "abstain"


def test_in_language_operators_do_not_construct():
    """Adaptive compute (U5): multiplication is answered at level 0; no construction is paid for."""
    r = Mind(Controller()).solve(Concept(), _task(lambda a, b: a * b, [(2, 3), (3, 4), (1, 5)], (4, 6)))
    assert r.status == "answer" and r.answer == F(24)
    assert [s.target for s in r.trace if s.action == "propose"] == ["level 0"]


def test_construction_off_is_v05():
    r = Mind(Controller()).solve(Concept(construct=None), _task(max, [(2, 3), (4, 1), (5, 5), (1, 6)], (7, 2)))
    assert r.status == "abstain"


def test_concept_memory_survives_save_and_load(tmp_path):
    mind = Mind(Controller())
    mind.solve(Concept(), _task(max, [(2, 3), (4, 1), (5, 5), (1, 6)], (7, 2), "⊕"))
    path = tmp_path / "mind.json"
    mind.save(str(path))
    back = Mind.load(str(path))
    mem = ConceptMemory.of(back)
    assert mem.text(0) == "ifgt(x, y, x, y)"
    assert mem.fn(0)(F(3), F(9)) == F(9)


def test_a_retired_concept_is_kept_but_no_longer_found():
    mem = ConceptMemory()
    k = mem.record(lambda x, y: max(x, y), (0, 1, 0, 1, 14), "t", [])
    mem.fail(k)
    mem.fail(k)
    mem.fail(k)
    assert mem.concepts[k].status == "retired"
    assert mem.find(lambda x, y: max(x, y)) is None


# ---------- language ----------

@pytest.mark.parametrize("line", ["3 ⊕ 5 = 8", "8 = 3 ⊕ 5", "⊕(3, 5) = 8", "3 ⊕ 5 gives 8",
                                  "combining 3 and 5 with ⊕ gives 8", "if you ⊕ 3 and 5, you get 8",
                                  "3 and 5 make 8 under ⊕"])
def test_every_statement_template_reads_the_same_roles(line):
    p = parse(line)
    assert (p.kind, p.op, p.a, p.b, p.c) == ("statement", "⊕", F(3), F(5), F(8))


def test_free_text_is_refused_with_a_reason():
    assert parse("the larger of the two numbers, please").kind == "refused"


def test_a_new_symbol_for_an_old_operation_finds_the_old_concept_and_a_word_binds_to_it():
    """Representation abstraction (U1) and language-to-structure: identity is behaviour, not symbol."""
    chat = Conversation()
    chat.say("2 ⊕ 3 = 3; 4 ⊕ 1 = 4; 5 ⊕ 5 = 5; 1 ⊕ 6 = 6")
    assert "= 7" in chat.say("7 ⊕ 2 = ?")
    chat.say("combining 6 and 1 with ★ gives 6; if you ★ 2 and 9, you get 9; 4 ★ 4 is 4; ★(8, 3) = 8")
    reply = chat.say("what is 1 ★ 5?")
    assert "= 5" in reply
    assert chat.concept_of["⊕"] == chat.concept_of["★"]
    assert chat.say("call ★ larger") == "Noted: ★ is 'larger'."
    assert chat.say("what is larger of 30 and 12?").startswith("larger of 30 and 12 is 30")


def test_the_learned_reader_reads_a_notation_it_was_never_told():
    """EQB-style input learning: digit values and base 7 from written sums only, then exact reading."""
    pytest.importorskip("torch")
    import random
    from hrm_x.reader import make_sums
    reader = LearnedReader.learn(make_sums(random.Random(0), 120, base=7, max_int=300))
    assert reader is not None and reader.base == 7
    assert reader.read("12") == F(9) and reader.read("-3.5") == -(F(3) + F(5, 7))
    assert parse("12 ⊕ 3 = 15", reader).c == F(12)
    assert DecimalReader().read("15") == F(15)
