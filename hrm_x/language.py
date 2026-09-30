"""Language interface: surface text -> structured evidence -> the mind -> an explained answer.

What it does, and what it deliberately does not:

  * **Perception of numerals is learned or exact, never guessed.** `DecimalReader` parses
    decimal, fractional and negative numerals exactly. `LearnedReader` wraps the Place-Value
    Reader (hrm_x/reader.py): its digit values and base are learned by self-supervision from
    written sums only, snapped to integers, and accepted only if every sum then holds exactly --
    so a notation the system has never been told (say base 7) is read exactly after it has seen
    sums written in it. This is the EQB input-learning path kept where it measurably works.
  * **Relations are read into roles, not into tokens.** Six statement templates (infix, reversed,
    functional, "gives", "combining ... with", "if you ... you get") all produce the same
    ((x, y), z) evidence. The operator symbol is only a name for a task; the concept a task uses
    is identified by *behaviour* (concept memory keys), so a new symbol for an old operation finds
    the old concept.
  * **Words can be bound to constructed concepts.** "call ⊕ larger" names the concept the mind
    formed for ⊕; afterwards "what is larger of 3 and 9?" is answered from memory with no evidence.
  * **Not free text.** ADR-009 stands: anything outside these templates is refused with the reason,
    never paraphrased into a guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hrm_x.concept_memory import ConceptMemory
from hrm_x.domains.concept import Concept, ConceptTask
from hrm_x.mind import Controller, Mind, Result

Q = Fraction
NUM = r"-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?"
SYM = r"[^\s\d().,=?/+\-*]+|[+\-*/]"


class DecimalReader:
    """Exact reading of decimal numerals, fractions a/b and negatives."""

    def read(self, s: str) -> Q:
        return Q(s)


class LearnedReader:
    """Numerals in a learned place-value notation (digit values and base from written sums).

    Only the snapped, verified parameters are used to read: `read` is an exact Fraction map, and
    an unverified reader refuses rather than reading approximately.
    """

    def __init__(self, theta: Sequence[int], base: int) -> None:
        self.theta, self.base = list(theta), int(base)

    @staticmethod
    def learn(sums: Sequence[Tuple[str, str, str]], seed: int = 0) -> Optional["LearnedReader"]:
        from hrm_x.reader import train_reader          # torch only here
        _, rep = train_reader(sums, seed=seed)
        return LearnedReader(*rep.snapped) if rep.snapped is not None else None

    def read(self, s: str) -> Q:
        neg = s.startswith("-")
        s = s.lstrip("-")
        if "/" in s:
            a, b = s.split("/")
            return (-1 if neg else 1) * self.read(a) / self.read(b)
        ip, _, fp = s.partition(".")
        v = Q(0)
        for j, c in enumerate(ip):
            v += self.theta[int(c)] * Q(self.base) ** (len(ip) - 1 - j)
        for j, c in enumerate(fp):
            v += self.theta[int(c)] * Q(self.base) ** (-(j + 1))
        return -v if neg else v


STATEMENTS = [
    re.compile(rf"^(?P<a>{NUM})\s*(?P<op>{SYM})\s*(?P<b>{NUM})\s*(?:=|is|equals|gives|makes)\s*(?P<c>{NUM})$"),
    re.compile(rf"^(?P<c>{NUM})\s*=\s*(?P<a>{NUM})\s*(?P<op>{SYM})\s*(?P<b>{NUM})$"),
    re.compile(rf"^(?P<op>{SYM})\s*\(\s*(?P<a>{NUM})\s*,\s*(?P<b>{NUM})\s*\)\s*(?:=|is)\s*(?P<c>{NUM})$"),
    re.compile(rf"^combining\s+(?P<a>{NUM})\s+(?:and|with)\s+(?P<b>{NUM})\s+(?:under|with|by)\s+(?P<op>{SYM})\s+gives\s+(?P<c>{NUM})$"),
    re.compile(rf"^if you\s+(?P<op>{SYM})\s+(?P<a>{NUM})\s+and\s+(?P<b>{NUM})\s*,?\s*you get\s+(?P<c>{NUM})$"),
    re.compile(rf"^(?P<a>{NUM})\s+and\s+(?P<b>{NUM})\s+make\s+(?P<c>{NUM})\s+under\s+(?P<op>{SYM})$"),
]
QUERIES = [
    re.compile(rf"^(?P<a>{NUM})\s*(?P<op>{SYM})\s*(?P<b>{NUM})\s*=\s*\?$"),
    re.compile(rf"^what is\s+(?P<a>{NUM})\s*(?P<op>{SYM})\s*(?P<b>{NUM})\s*\??$"),
    re.compile(rf"^(?P<op>{SYM})\s*\(\s*(?P<a>{NUM})\s*,\s*(?P<b>{NUM})\s*\)\s*=\s*\?$"),
    re.compile(rf"^what do you get from\s+(?P<a>{NUM})\s+and\s+(?P<b>{NUM})\s+under\s+(?P<op>{SYM})\s*\??$"),
]
NAMING = re.compile(rf"^(?:call|name)\s+(?P<op>{SYM})\s+(?P<word>[a-z][a-z_]*)$")
NAMED_QUERY = re.compile(rf"^what is\s+(?P<word>[a-z][a-z_]*)\s+of\s+(?P<a>{NUM})\s+and\s+(?P<b>{NUM})\s*\??$")


@dataclass
class Parsed:
    kind: str                            # statement | query | naming | named_query | refused
    op: str = ""
    a: Optional[Q] = None
    b: Optional[Q] = None
    c: Optional[Q] = None
    word: str = ""
    reason: str = ""


def parse(line: str, reader: Any = None) -> Parsed:
    """One line of text into a structured act; anything outside the templates is refused."""
    reader = reader or DecimalReader()
    t = " ".join(line.strip().split())
    t = t[:-1] if t.endswith(".") else t
    for pat in STATEMENTS:
        m = pat.match(t)
        if m:
            return Parsed("statement", m["op"], reader.read(m["a"]), reader.read(m["b"]), reader.read(m["c"]))
    for pat in QUERIES:
        m = pat.match(t)
        if m:
            return Parsed("query", m["op"], reader.read(m["a"]), reader.read(m["b"]))
    m = NAMING.match(t)
    if m:
        return Parsed("naming", m["op"], word=m["word"])
    m = NAMED_QUERY.match(t)
    if m:
        return Parsed("named_query", word=m["word"], a=reader.read(m["a"]), b=reader.read(m["b"]))
    return Parsed("refused", reason="not one of the statement, query or naming templates (ADR-009: no free text)")


def _num(v: Any) -> str:
    return str(v) if not isinstance(v, Q) or v.denominator != 1 else str(v.numerator)


def explain(result: Result, task: ConceptTask, domain: Concept, mind: Mind) -> str:
    """Answer (or refusal) plus how it was reached, from the trace -- never from a template of intent."""
    q = task.question
    if result.status != "answer":
        return f"I do not know {_num(q[0])} {task.symbol} {_num(q[1])}: {result.reason}."
    winner = max((h for h in result.hypotheses if h.verified and h.answer == result.answer and h.payload),
                 key=lambda h: h.weight, default=None)
    head = f"{_num(q[0])} {task.symbol} {_num(q[1])} = {_num(result.answer)}"
    if winner is None:
        return head + "."
    fam = winner.payload.family.name
    rows = len(task.evidence())
    surplus = winner.payload.surplus
    asked = sum(1 for s in result.trace if s.action == "ask")
    how = ""
    if fam.startswith("construct["):
        how = (f"; the operation was not in my language, so I constructed {fam[len('construct['):]} and "
               f"fitted it exactly: it predicted {surplus} of the {rows} examples it was not fitted to")
    elif fam.startswith("lib:"):
        how = f"; I used the concept '{fam[4:].split(']')[0].split('+')[0]}' I formed earlier ({surplus} examples predicted)"
    else:
        how = f"; the reading {fam} predicted {surplus} of {rows} examples"
    if asked:
        how += f", after asking {asked} question{'s' if asked > 1 else ''}"
    return head + how + " and the dual check re-derived the same number."


@dataclass
class Conversation:
    """Stateful text interface over one Mind: statements accumulate per operator symbol."""

    mind: Mind = field(default_factory=lambda: Mind(Controller()))
    domain: Concept = field(default_factory=Concept)
    reader: Any = field(default_factory=DecimalReader)
    evidence: Dict[str, List[Tuple[Q, Q, Q]]] = field(default_factory=dict)
    concept_of: Dict[str, int] = field(default_factory=dict)       # symbol -> concept index

    def say(self, text: str) -> str:
        replies = [r for r in (self._one(line) for line in re.split(r"[;\n]", text) if line.strip()) if r]
        return " ".join(replies)

    def _one(self, line: str) -> str:
        p = parse(line, self.reader)
        if p.kind == "statement":
            self.evidence.setdefault(p.op, []).append((p.a, p.b, p.c))
            return ""
        if p.kind == "query":
            task = ConceptTask(p.op, list(self.evidence.get(p.op, [])), (p.a, p.b))
            result = self.mind.solve(self.domain, task)
            k = self._concept_used(result)
            if k is not None:
                self.concept_of[p.op] = k
            return explain(result, task, self.domain, self.mind)
        if p.kind == "naming":
            k = self.concept_of.get(p.op)
            if k is None:
                return f"I have not formed a concept for {p.op} yet, so there is nothing to name."
            ConceptMemory.of(self.mind).name(k, p.word)
            return f"Noted: {p.op} is '{p.word}'."
        if p.kind == "named_query":
            mem = ConceptMemory.of(self.mind)
            k = mem.by_name(p.word)
            if k is None:
                return f"I know no concept called '{p.word}'."
            v = mem.fn(k)(p.a, p.b)
            return f"{p.word} of {_num(p.a)} and {_num(p.b)} is {_num(v)} (concept {mem.text(k)})."
        return f"I cannot read '{line.strip()}': {p.reason}."

    def _concept_used(self, result: Result) -> Optional[int]:
        """Index of the stored concept behind a verified answer (behaviour identity, not symbol)."""
        if result.status != "answer":
            return None
        winner = max((h for h in result.hypotheses if h.verified and h.answer == result.answer and h.payload),
                     key=lambda h: h.weight, default=None)
        origin = None if winner is None else self.domain._origin.get(winner.payload.family.name)
        return None if origin is None else ConceptMemory.of(self.mind).find(origin[1])
