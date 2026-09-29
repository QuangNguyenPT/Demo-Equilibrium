"""Held-out free text: does the learned perception transfer to surface forms it never saw?

The parser is trained only on `TEMPLATES`. Every earlier story experiment evaluated on forms drawn
from the same closed set (`EVENT_TEXT` in x_story repeats them), so "generalizes to unseen text" was
never measured. This experiment measures it, against a reader written independently of the parser.

Two questions, and they have different answers:

1. **Perception.** Does the tagger's reading of a held-out sentence match a correct one? Measured
   per-sentence, split by *what* is held out: only the verb (word order unchanged), the clause order
   (passive / "to B from A"), the question word order, or the whole frame.
2. **Verification.** When perception is wrong, does the mind notice? `Story.verify` replays the
   tagger's own `Reading` through a different *representation* (a plain dict rather than the
   e-graph), which is a real check on the derivation and no check at all on the reading. A
   misread that is internally complete is therefore reported as `verified` with a confident answer.
   The rate of that is reported here as `confident_wrong_on_misread`, because a benchmark that
   cannot fail proves nothing and this is the failure the closed-set evaluation was hiding.

The reference reader below is a hand-written lexicon+regex grammar over the held-out forms. It shares
no code path with the parser, no weights, and no templates; the only thing it has in common is the
`Frame` output type, so that the two readings can be compared. It lives in the experiment, not in the
mind, on purpose: putting it in the mind would be a second non-learned perception path, which is an
architecture decision (see ADR-009), not a measurement.

  python -m experiments.x_perception
"""
from __future__ import annotations

import random
import re
import time
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from experiments.common import record, table
from hrm_x.domains.story import (Frame, Parser, Story, StoryTask, frames_for, singular,
                                 tokenize, TRAIN_ITEMS, TRAIN_NAMES)
from hrm_x.mind import UNKNOWN, Mind

SEED = 0
# Names and items the parser's vocabulary has never seen, and that the reference reader must also
# handle without a list of them (it is role-based, not name-based).
HELDOUT_NAMES = ["Zoltan", "Quenby", "Xavier", "Wendell", "Priya", "Kofi", "Leilani", "Oskar"]
HELDOUT_ITEMS = ["oranges", "pencils", "stamps"]

# ---------- the independent reader ----------

def _rx(pattern: str) -> "re.Pattern":
    """Case-insensitive, because the domain capitalizes questions ("How many ...") and not the
    templates' own lower-case function words, and a reader that only matched one of those two
    conventions would look like a perception failure when it is a spelling one."""
    return re.compile(pattern, re.IGNORECASE)

# (pattern, schema, what each capture group is). Written by hand against these sentences only; the
# parser has never seen any of these strings. `A` is the initiator, `B` the other party, `n` an
# integer, `i` a plural noun, `q` a question word. Order matters: the first match wins, so longer
# patterns precede their prefixes. Naming the groups is what makes the passive forms readable: in
# "{B} is given n {i} by {A}" the text order is recipient-first, but A is still the giver.
# Whitespace is explicit because the domain's own text puts a space before its final ".", the way the
# templates and the tokenizer do; a reader that assumed "word." would silently match nothing.
REFERENCE: List[Tuple[re.Pattern, str, Tuple[str, ...]]] = [
    (_rx(r"^(\w+)\s+hands\s+(\d+)\s+(\w+)\s+to\s+(\w+)\s*\.$"), "TRANSFER", ("A", "n", "i", "B")),
    (_rx(r"^(\w+)\s+passes\s+(\d+)\s+(\w+)\s+over\s+to\s+(\w+)\s*\.$"), "TRANSFER", ("A", "n", "i", "B")),
    (_rx(r"^(\w+)\s+passes\s+(\d+)\s+(\w+)\s+to\s+(\w+)\s*\.$"), "TRANSFER", ("A", "n", "i", "B")),
    (_rx(r"^(\w+)\s+transfers\s+(\d+)\s+(\w+)\s+to\s+(\w+)\s*\.$"), "TRANSFER", ("A", "n", "i", "B")),
    (_rx(r"^(\w+)\s+is\s+given\s+(\d+)\s+(\w+)\s+by\s+(\w+)\s*\.$"), "TRANSFER", ("B", "n", "i", "A")),
    (_rx(r"^(\w+)\s+gets\s+(\d+)\s+(\w+)\s+from\s+(\w+)\s*\.$"), "TRANSFER", ("B", "n", "i", "A")),
    (_rx(r"^(\w+)\s+accepts\s+(\d+)\s+(\w+)\s+from\s+(\w+)\s*\.$"), "TRANSFER", ("B", "n", "i", "A")),
    (_rx(r"^(\w+)\s+collects\s+(\d+)\s+(\w+)\s*\.$"), "ACQUIRE", ("A", "n", "i")),
    (_rx(r"^(\w+)\s+gathers\s+(\d+)\s+(\w+)\s*\.$"), "ACQUIRE", ("A", "n", "i")),
    (_rx(r"^(\w+)\s+gobbles\s+(\d+)\s+(\w+)\s*\.$"), "CONSUME", ("A", "n", "i")),
    (_rx(r"^(\w+)\s+throws\s+away\s+(\d+)\s+(\w+)\s*\.$"), "CONSUME", ("A", "n", "i")),
    (_rx(r"^(\w+)\s+holds\s+(\d+)\s+(\w+)\s*\.$"), "STATE", ("A", "n", "i")),
    (_rx(r"^(\w+)\s+has\s+how\s+many\s+(\w+)\s*\?$"), "Q_ONE", ("A", "i")),
    (_rx(r"^how\s+many\s+(\w+)\s+does\s+(\w+)\s+have\s+in\s+total\s*\?$"), "Q_ONE", ("i", "A")),
    (_rx(r"^how\s+many\s+(\w+)\s+do\s+(\w+)\s+and\s+(\w+)\s+have\s+altogether\s*\?$"), "Q_SUM", ("i", "A", "B")),
]

# The same grammar extended to the forms the parser *was* trained on, so the experiment can report a
# control next to the transfer number. These are the 12 TEMPLATES and nothing else; the parser has
# seen every one of them, so a gap here is a gap in the control, not in the held-out result.
REFERENCE_SEEN: List[Tuple[re.Pattern, str, Tuple[str, ...]]] = [
    (_rx(r"^(\w+)\s+has\s+(\d+)\s+(\w+)\s*\.$"), "STATE", ("A", "n", "i")),
    (_rx(r"^(\w+)\s+owns\s+(\d+)\s+(\w+)\s*\.$"), "STATE", ("A", "n", "i")),
    (_rx(r"^(\w+)\s+gives\s+(\w+)\s+(\d+)\s+(\w+)\s*\.$"), "TRANSFER", ("A", "B", "n", "i")),
    (_rx(r"^(\w+)\s+gives\s+(\d+)\s+(\w+)\s+to\s+(\w+)\s*\.$"), "TRANSFER", ("A", "n", "i", "B")),
    (_rx(r"^(\w+)\s+receives\s+(\d+)\s+(\w+)\s+from\s+(\w+)\s*\.$"), "TRANSFER", ("B", "n", "i", "A")),
    (_rx(r"^(\w+)\s+takes\s+(\d+)\s+(\w+)\s+from\s+(\w+)\s*\.$"), "TRANSFER", ("B", "n", "i", "A")),
    (_rx(r"^(\w+)\s+buys\s+(\d+)\s+(\w+)\s*\.$"), "ACQUIRE", ("A", "n", "i")),
    (_rx(r"^(\w+)\s+finds\s+(\d+)\s+(\w+)\s*\.$"), "ACQUIRE", ("A", "n", "i")),
    (_rx(r"^(\w+)\s+eats\s+(\d+)\s+(\w+)\s*\.$"), "CONSUME", ("A", "n", "i")),
    (_rx(r"^(\w+)\s+loses\s+(\d+)\s+(\w+)\s*\.$"), "CONSUME", ("A", "n", "i")),
    (_rx(r"^how\s+many\s+(\w+)\s+does\s+(\w+)\s+have\s*\?$"), "Q_ONE", ("i", "A")),
    (_rx(r"^how\s+many\s+(\w+)\s+do\s+(\w+)\s+and\s+(\w+)\s+have\s+together\s*\?$"), "Q_SUM", ("i", "A", "B")),
]


def reference_frame(sentence: str, seen: bool = False) -> Optional[Frame]:
    """The correct reading of a sentence, or None if the reader does not cover it.

    `seen=True` restricts the reader to the trained-on templates, which is how the held-out set is
    proven to be held out: a sentence the *seen* grammar cannot read is by construction a sentence the
    parser was never shown.
    """
    s = sentence.strip()
    for pattern, schema, labels in (REFERENCE_SEEN if seen else REFERENCE + REFERENCE_SEEN):
        m = pattern.match(s)
        if m is None:
            continue
        g = dict(zip(labels, m.groups()))
        a = g["A"].lower() if "A" in g else None
        b = g["B"].lower() if "B" in g else None
        n = Fraction(g["n"]) if "n" in g else None
        item = singular(g["i"]) if "i" in g else None
        if schema == "TRANSFER":
            return Frame("TRANSFER", (), a, b, n, item)      # A initiates, B receives
        if schema == "Q_SUM":
            return Frame("Q_SUM", (a, b), None, None, None, item)
        if schema == "Q_ONE":
            return Frame("Q_ONE", (a,), None, None, None, item)
        return Frame(schema, (a,), None, None, n, item)
    return None


def key(frame: Optional[Frame]) -> Optional[tuple]:
    if frame is None:
        return None
    return (frame.schema, frame.owner, frame.src, frame.dst, frame.amount,
            singular(frame.item) if frame.item else None)


def predicted_frame(parser: Parser, sentence: str) -> Optional[Frame]:
    readings = frames_for(parser, sentence)
    return readings[0][0] if readings else None


# ---------- probe sets, by what is held out ----------

def probes() -> List[Tuple[str, str, str, Frame]]:
    """(group, sentence, what changed, correct reading). Names/items are unseen by the parser."""
    a, b = "Zoltan", "Priya"
    out: List[Tuple[str, str, str, Frame]] = [
        ("control (seen form)", f"{a} gives {b} 3 oranges.", "none",
         Frame("TRANSFER", (), a.lower(), b.lower(), Fraction(3), "orange")),
        ("control (seen form)", f"{a} buys 3 oranges.", "none",
         Frame("ACQUIRE", (a.lower(),), None, None, Fraction(3), "orange")),
        ("control (seen form)", f"how many oranges does {a} have ?", "none",
         Frame("Q_ONE", (a.lower(),), None, None, None, "orange")),

        ("unseen verb, same order", f"{a} transfers 3 oranges to {b} .", "verb",
         Frame("TRANSFER", (), a.lower(), b.lower(), Fraction(3), "orange")),
        ("unseen verb, same order", f"{a} collects 3 oranges.", "verb",
         Frame("ACQUIRE", (a.lower(),), None, None, Fraction(3), "orange")),
        ("unseen verb, same order", f"{a} gobbles 3 oranges.", "verb",
         Frame("CONSUME", (a.lower(),), None, None, Fraction(3), "orange")),
        ("unseen verb, same order", f"{a} holds 3 oranges.", "verb",
         Frame("STATE", (a.lower(),), None, None, Fraction(3), "orange")),
        ("unseen verb, same order", f"{b} gets 3 oranges from {a} .", "verb",
         Frame("TRANSFER", (), a.lower(), b.lower(), Fraction(3), "orange")),
        ("unseen verb, same order", f"{b} accepts 3 oranges from {a} .", "verb",
         Frame("TRANSFER", (), a.lower(), b.lower(), Fraction(3), "orange")),

        ("clause order", f"{a} hands 3 oranges to {b} .", "word order",
         Frame("TRANSFER", (), a.lower(), b.lower(), Fraction(3), "orange")),
        ("clause order", f"{a} passes 3 oranges over to {b} .", "word order + filler",
         Frame("TRANSFER", (), a.lower(), b.lower(), Fraction(3), "orange")),
        ("clause order", f"{b} is given 3 oranges by {a} .", "passive voice",
         Frame("TRANSFER", (), a.lower(), b.lower(), Fraction(3), "orange")),

        ("question order", f"{a} has how many oranges ?", "question word order",
         Frame("Q_ONE", (a.lower(),), None, None, None, "orange")),
        ("question order", f"how many oranges does {a} have in total ?", "question suffix",
         Frame("Q_ONE", (a.lower(),), None, None, None, "orange")),
        ("question order", f"how many oranges do {a} and {b} have altogether ?", "question suffix",
         Frame("Q_SUM", (a.lower(), b.lower()), None, None, None, "orange")),
    ]
    # No row is dropped: a probe the reference reader cannot read would silently shrink the metric.
    # `tests/test_x_perception.py` asserts that the reader agrees with every expectation here.
    return out


def score(parser: Parser, rows: List[Tuple[str, str, str, Frame]]) -> Dict[str, dict]:
    """Accuracy of the tagger's top reading against the reference, by group."""
    groups: Dict[str, Dict[str, int]] = {}
    for group, sentence, _changed, frame in rows:
        got, want = key(predicted_frame(parser, sentence)), key(frame)
        g = groups.setdefault(group, {"n": 0, "schema": 0, "exact": 0, "slots": 0, "slot_total": 0,
                                     "item": 0, "parties": 0, "no_reading": 0})
        g["n"] += 1
        if got is None:
            g["no_reading"] += 1
        else:
            g["schema"] += int(got[0] == want[0])
            g["item"] += int(got[5] == want[5])
            if want[0] == "TRANSFER":
                g["parties"] += int((got[2], got[3]) == (want[2], want[3]))
            else:
                g["parties"] += int(got[1] == want[1])
            g["slots"] += sum(int(a == b) for a, b in
                              ((got[1], want[1]), (got[2], want[2]), (got[3], want[3]),
                               (got[4], want[4]), (got[5], want[5])))
            g["slot_total"] += 5
        g["exact"] += int(got == want)
    out = {}
    for name, g in groups.items():
        out[name] = {"n": g["n"], "schema": g["schema"] / g["n"], "exact": g["exact"] / g["n"],
                     "slots": g["slots"] / max(g["slot_total"], 1), "item": g["item"] / g["n"],
                     "parties": g["parties"] / g["n"], "no_reading": g["no_reading"] / g["n"]}
    return out


# ---------- end to end, on held-out text only ----------

# Held-out event wording, chosen from REFERENCE so the truth and the reference agree.
GIVE = ["{a} hands {n} {i} to {b} .", "{a} passes {n} {i} to {b} .",
        "{b} is given {n} {i} by {a} .", "{b} gets {n} {i} from {a} ."]
TAKE = ["{a} gobbles {n} {i} .", "{a} throws away {n} {i} ."]
GET = ["{a} collects {n} {i} .", "{a} gathers {n} {i} ."]
STATE_HELD = ["{a} holds {n} {i} .", "{a} has {n} {i} ."]
ASK_HELD = ["{a} has how many {i} ?", "how many {i} does {a} have in total ?"]


def own_truth(start: Dict[str, int], events: List[Tuple], ask: List[str]) -> Optional[int]:
    """The experiment's own replay, sharing no code with `story.simulate`."""
    state = dict(start)
    for kind, a, b, n in events:
        if kind == "give":
            state[a] -= n
            state[b] += n
        else:
            state[a] += n if kind == "acquire" else -n
        if min(state.values()) < 0:
            return None
    return sum(state[x] for x in ask)


def held_out_story(rng: random.Random, held_state: bool, held_ask: bool) -> Tuple[List[str], str, Optional[int], bool]:
    """A story whose events are always held-out wording; state/question optionally so too."""
    a, b = rng.sample(HELDOUT_NAMES, 2)
    item = rng.choice(HELDOUT_ITEMS)
    na, nb = rng.randint(5, 20), rng.randint(5, 20)
    start = {a: na, b: nb}
    sentences = [rng.choice(STATE_HELD if held_state else ["{a} has {n} {i} ."]).format(a=a, n=na, i=item),
                 rng.choice(STATE_HELD if held_state else ["{a} has {n} {i} ."]).format(a=b, n=nb, i=item)]
    events, state = [], dict(start)
    kinds = ["give", "give", "acquire", "consume"]
    for _ in range(rng.randint(1, 2)):
        kind = rng.choice(kinds)
        x, y = rng.sample([a, b], 2)
        cap = state[x] if kind in ("give", "consume") else 10
        if cap < 1:
            continue
        n = rng.randint(1, cap)
        text = rng.choice(GIVE if kind == "give" else (GET if kind == "acquire" else TAKE))
        sentences.append(text.format(a=x, b=y, n=n, i=item))
        events.append((kind, x, y, n))
        if kind == "give":
            state[x] -= n
            state[y] += n
        else:
            state[x] += n if kind == "acquire" else -n
    if not events:
        return [], "", None, False
    asker = rng.choice([a, b])
    q = (rng.choice(ASK_HELD).format(a=asker, i=item) if held_ask
         else f"How many {item} does {asker} have ?")
    return sentences, q, own_truth(start, events, [asker]), True


def end_to_end(parser: Parser, n: int, held_state: bool, held_ask: bool) -> Dict[str, float]:
    """Correct, abstain, and *confident wrong*, plus whether perception was even right.

    `confident_wrong_on_misread` is the number that matters: stories the mind answered and marked
    verified, where the reference reading of the same sentences disagrees. Each of those is a wrong
    answer the mind is confident in, and the existing verification cannot see it.
    """
    rng = random.Random(SEED + 1)
    dom = Story(parser)
    correct = abstain = wrong = 0
    total = 0
    misread_total = misread_confident = misread_safe = 0
    for _ in range(n * 4):
        if total >= n:
            break
        sentences, q, truth, ok = held_out_story(rng, held_state, held_ask)
        if not ok or truth is None:
            continue
        total += 1
        # Loud, not silent: a sentence the reference reader cannot read would drop out of the
        # misread count and quietly report a clean bill of health for the verifier.
        want = [key(reference_frame(s)) for s in sentences] + [key(reference_frame(q))]
        if any(w is None for w in want):
            raise RuntimeError(f"reference reader does not cover {sentences + [q]!r}")
        got = [key(predicted_frame(parser, s)) for s in sentences] + [key(predicted_frame(parser, q))]
        result = Mind().solve(dom, StoryTask(sentences, q))
        if result.status == "answer" and result.answer == truth:
            correct += 1
        elif result.status in ("abstain", "ask") or result.answer is UNKNOWN or result.answer is None:
            abstain += 1
        else:
            wrong += 1
        if want != got:
            misread_total += 1
            if result.status == "answer" and result.answer != truth:
                misread_confident += 1
            else:
                misread_safe += 1
    return {"n": total, "correct": correct / max(total, 1), "abstain": abstain / max(total, 1),
            "wrong": wrong / max(total, 1),
            "misread": misread_total / max(total, 1),
            "confident_wrong_on_misread": misread_confident / max(misread_total, 1),
            "safe_on_misread": misread_safe / max(misread_total, 1),
            "counts": {"n": total, "correct": correct, "abstain": abstain, "wrong": wrong,
                       "misread": misread_total, "confident_wrong_on_misread": misread_confident,
                       "safe_on_misread": misread_safe}}


def main() -> None:
    t0 = time.time()
    parser = Parser()
    loss = parser.train(seed=SEED)
    train_s = time.time() - t0
    print(f"[PARSER] {parser.params} params, loss {loss:.4f}, {train_s:.0f}s")

    rows = probes()
    by_group = score(parser, rows)
    table_rows = [[g, m["n"], f"{m['schema']:.2f}", f"{m['parties']:.2f}", f"{m['item']:.2f}",
                   f"{m['slots']:.2f}", f"{m['exact']:.2f}"]
                  for g, m in by_group.items()]
    print(table(table_rows, ["held out", "n", "schema", "parties", "item", "slots", "exact"]))

    e2e = {
        "events held out": end_to_end(parser, 120, held_state=False, held_ask=False),
        "events + state held out": end_to_end(parser, 120, held_state=True, held_ask=False),
        "events + question held out": end_to_end(parser, 120, held_state=False, held_ask=True),
    }
    e2e_rows = [[k, v["n"], f"{v['correct']:.2f}", f"{v['abstain']:.2f}", f"{v['wrong']:.2f}",
                 f"{v['misread']:.2f}", f"{v['confident_wrong_on_misread']:.2f}"]
                for k, v in e2e.items()]
    print(table(e2e_rows, ["split", "n", "correct", "abstain", "wrong", "misread",
                           "confident wrong on misread"]))

    m = {"perception_by_group": by_group,
         "end_to_end": e2e,
         "held_out_verbs_transfer": by_group.get("unseen verb, same order", {}).get("exact"),
         "clause_order_transfers": by_group.get("clause order", {}).get("exact"),
         "question_order_transfers": by_group.get("question order", {}).get("exact"),
         "control_exact": by_group.get("control (seen form)", {}).get("exact")}
    for k, v in e2e.items():
        m[f"{k} :: confident wrong on misread"] = v["confident_wrong_on_misread"]
    worst = e2e["events held out"]
    m["conclusion"] = (
        "Held-out free text does not transfer, and the part that fails is syntax rather than "
        f"vocabulary. The top reading is exact {m['control_exact']:.0%} of the time on seen forms, "
        f"{m['held_out_verbs_transfer']:.0%} with only the verb changed, "
        f"{m['clause_order_transfers']:.0%} with only the clause order changed and "
        f"{m['question_order_transfers']:.0%} with the question reordered. End to end, "
        f"{worst['misread']:.0%} of held-out stories ({worst['counts']['misread']} of "
        f"{worst['counts']['n']}) contain at least one misread sentence, and the answer is right "
        f"{worst['correct']:.0%} of the time at best. The tagger learned template "
        "positions, not a compositional reader: an unseen verb is fine if its slots land where "
        "training put them, and reordering them loses the roles. On the second question the "
        "measurement went against the expectation, which is worth stating plainly: `Story.verify` "
        "replays the tagger's own reading through a plain dictionary, so it cannot see a misreading, "
        "yet only "
        f"{worst['counts']['confident_wrong_on_misread']} of {worst['counts']['misread']} misread "
        "stories came back as a confident wrong answer. That safety comes from `reading_defect`, "
        "which rejects a reading that is missing a slot and makes the mind abstain -- not from "
        "verification. The remaining confident errors are the dangerous class: readings that are "
        "complete and wrong (a consume verb read as an acquire, say), which no structural check can "
        "catch. So free text is unsupported today, and the honest reason is the reader, not the "
        "reasoner; the reasoner's own guard is what keeps the failure from being worse.")
    print(f"\n[CONCLUSION] {m['conclusion']}")
    record("x_perception",
           {"seed": SEED, "parser_params": parser.params, "parser_train_seconds": train_s,
            "parser_training_data": "the 12 TEMPLATES only; no held-out form is trained on",
            "reference_reader": "hand-written regex+lexicon grammar over the held-out forms, "
                                "independent of the parser, used only for scoring",
            "held_out": sorted({r[0] for r in rows}),
            "unseen_names": HELDOUT_NAMES, "unseen_items": HELDOUT_ITEMS}, m)


if __name__ == "__main__":
    main()
