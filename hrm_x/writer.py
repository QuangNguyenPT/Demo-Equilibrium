"""Writer: the answer and its explanation, rendered from the actual computation (spec v0.4 §8).

It reads only the Result -- the trace of actions, the chosen hypothesis, its derivation note,
its verification, and the abstention reason -- so every claim in the text is something the mind
did. It is a fixed realizer, not a learned generator: fluent surface language is future work
(a Reader/Writer transformer trained on imagined traces), faithfulness is the property kept here.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from hrm_x.mind import Result

ACTION_WORDS = {"propose": "considered", "derive": "worked out", "verify": "checked", "ask": "asked"}


def _value(v: Any) -> str:
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f"{float(v):g}"
    if isinstance(v, tuple) and v and isinstance(v[0], tuple):
        return "\n" + "\n".join(" ".join(str(c) for c in row) for row in v)
    return str(v)


def explain(result: Result, question: str = "") -> str:
    lines = []
    if result.status == "answer":
        lines.append(f"{question.strip()} -> {_value(result.answer)}" if question else f"Answer: {_value(result.answer)}")
        chosen = max((h for h in result.hypotheses if h.answer == result.answer and h.weight > 0),
                     key=lambda h: h.weight, default=None)
        if chosen is not None:
            lines.append(f"Reading: {chosen.key}.")
            lines.append(f"Checked independently: {'yes' if chosen.verified else 'no (not worth the cost)'}; "
                         f"confidence {result.confidence:.2f}.")
    else:
        lines.append(f"I don't know{': ' + question.strip() if question else ''}. Why: {result.reason}.")
    asked = [s for s in result.trace if s.action == "ask"]
    if asked:
        lines.append("I asked: " + "; ".join(s.target for s in asked) + ".")
    lines.append("Steps: " + "; ".join(f"{ACTION_WORDS.get(s.action, s.action)} "
                                       f"{s.note or s.target}"[:140] for s in result.trace) + ".")
    lines.append(f"Cost: {result.cycles} decisions, {result.ops} operations.")
    return "\n".join(lines)
