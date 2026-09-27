"""Perception: the Domain protocol.

A domain turns raw examples into phasor scenes and supplies the executable
vocabulary (programs) the reasoning section searches over. Everything
domain-specific lives behind this protocol; the engine, the verifier, the
knowledge store and the learning loops never look inside an example.

Understanding, operationally: a program that reproduces every demonstration
exactly (judged by `Domain.equal`), preferred by simplicity.
"""
from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from hrm.substrate import Codebook, PhasorVector, bundle

Example = Tuple[Any, Any]           # (input, output) demonstration


class ProgramError(ValueError):
    """A program is not applicable to this input (not a bug, a mismatch)."""


class Program(ABC):
    """An explanation with two synchronized faces: executable (``apply``, what
    verification runs) and algebraic (``signature``, what the resonator and the
    knowledge store operate on)."""

    provenance: str = "search"

    @abstractmethod
    def apply(self, x: Any) -> Any:
        """Execute on one input; raise ProgramError when not applicable."""

    @abstractmethod
    def describe(self) -> str:
        """Canonical text: equal strings mean the same program."""

    @abstractmethod
    def signature(self, cb: Codebook) -> PhasorVector:
        ...

    @property
    @abstractmethod
    def complexity(self) -> int:
        """Number of non-identity factors (the MDL cost)."""

    @abstractmethod
    def to_payload(self) -> Dict[str, Any]:
        """JSON-serializable form, decoded by ``Domain.program_from_payload``."""

    @property
    def family(self) -> str:
        """Concept region this program belongs to (its kinds of steps, not their values)."""
        return self.describe()

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self.describe()}]"


@dataclass
class TaskEncoding:
    demos: List[Example]
    pair_vectors: List[PhasorVector]    # per-demo transformation out / in
    shared: PhasorVector                # coherent invariant core across demos
    deltas: List[PhasorVector]          # example-specific residues
    parse: int                          # which parse hypothesis produced it


class Domain(ABC):
    name: str = "domain"
    n_parses: int = 1                   # alternative parses (e.g. segmentations)

    def __init__(self, cb: Codebook):
        self.cb = cb

    @abstractmethod
    def accepts(self, demos: Sequence[Example]) -> bool:
        """True when every example is representable in this domain."""

    @abstractmethod
    def encode(self, x: Any, parse: int = 0) -> PhasorVector:
        ...

    @abstractmethod
    def candidates(self, demos: Sequence[Example]) -> List[Program]:
        """Data-conditioned base hypotheses for these demonstrations."""

    @abstractmethod
    def compose(self, first: Program, then: Program) -> Program:
        """Program that applies ``first`` and then ``then``."""

    closed_families: Tuple[str, ...] = ()

    def redundant(self, first: Program, then: Program) -> bool:
        """True when composing ``first`` and ``then`` adds no expressive power.

        The default test is a *family-level* claim: two single steps of a family
        that is closed under composition collapse to one step (two grid
        translations are one translation). Domains where that closure fails —
        rotations about different axes, for instance — must override this, or
        bounded search prunes the very compositions it needs to find.
        """
        return (first.complexity == 1 and then.complexity == 1
                and first.family == then.family and first.family in self.closed_families)

    @abstractmethod
    def equal(self, pred: Any, target: Any) -> bool:
        ...

    @abstractmethod
    def similarity(self, pred: Any, target: Any) -> float:
        """Graded agreement in [0, 1]; exactly 1 iff ``equal``."""

    @abstractmethod
    def size(self, x: Any) -> int:
        """Number of atomic elements in an input (bounds program complexity)."""

    #: Whether `size` grows with the program's natural depth, and may therefore
    #: be used as a relative bound on program complexity. False for fixed-width
    #: inputs: a 3-vector is size 3 whether the program rotates it once or five
    #: times, so deriving a complexity cap from it would cap the *reasoner* at a
    #: depth the input width has nothing to do with.
    size_bounds_complexity: bool = True

    @abstractmethod
    def program_from_payload(self, payload: Dict[str, Any]) -> Program:
        ...

    @abstractmethod
    def vocabulary(self) -> str:
        """The program language, described for the language section's prompts."""

    format_example: Tuple[Any, str] = ((), "")   # (input, rule) shown to the language section

    @abstractmethod
    def program_from_terms(self, terms: List[Tuple[str, Any]]) -> Program:
        """Build a program from parsed ``name(args)`` terms; ValueError if invalid."""

    def parse_program(self, text: str) -> Program:
        """Inverse of ``Program.describe``: ``"translate(dx=1, dy=0) . color_cycle(shift=2)"``.
        Raises ValueError on anything malformed (language output is untrusted)."""
        return self.program_from_terms(split_terms(text))

    def render(self, x: Any) -> str:
        """Plain-text form of one input/output: rows of numbers separated by ' / '."""
        arr = np.asarray(x)
        if arr.ndim <= 1:
            return " ".join(str(v) for v in arr.ravel().tolist())
        return " / ".join(" ".join(str(v) for v in row) for row in arr.reshape(-1, arr.shape[-1]))

    def to_json(self, x: Any) -> Any:
        """JSON-serializable form of one input/output (for episodic memory)."""
        return np.asarray(x).tolist()

    def from_json(self, obj: Any) -> Any:
        return np.asarray(obj, dtype=np.int64)

    def encode_task(self, demos: Sequence[Example], parse: int = 0) -> TaskEncoding:
        pairs = [self.encode(out, parse) / self.encode(inp, parse) for inp, out in demos]
        shared = bundle(pairs)       # identical transformations add coherently
        return TaskEncoding(list(demos), pairs, shared, [p / shared for p in pairs], parse)

    def task_key(self, task: TaskEncoding) -> PhasorVector:
        """Knowledge-store key: the task's invariant core, tagged with this domain."""
        return bundle([task.shared, self.cb.symbol(f"domain.{self.name}")])


_TERM = re.compile(r"^([a-z_][a-z0-9_]*)(?:\((.*)\))?$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def split_terms(text: str) -> List[Tuple[str, Any]]:
    """``"a(x=1, y=-2) . b(r90) . c"`` -> ``[("a", {"x": 1, "y": -2}), ("b", "r90"), ("c", None)]``.

    Arguments are keyword literals, one literal (``{1: 3}``), or one bare word.
    """
    terms: List[Tuple[str, Any]] = []
    for raw in text.strip().split(" . "):
        m = _TERM.match(raw.strip())
        if m is None:
            raise ValueError(f"malformed program term {raw!r}")
        name, args = m.group(1), m.group(2)
        if args is None or not args.strip():
            terms.append((name, None))
        elif "=" in args and not args.lstrip().startswith("{"):
            kwargs = {}
            for part in args.split(","):
                key, sep, value = part.partition("=")
                if not sep or not key.strip().isidentifier():
                    raise ValueError(f"malformed argument {part!r} in {raw!r}")
                kwargs[key.strip()] = _literal(value)
            terms.append((name, kwargs))
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.strip()):
            terms.append((name, args.strip()))
        else:
            terms.append((name, _literal(args)))
    return terms


def int_kwargs(args: Any, allowed: set, bound: int) -> bool:
    """True if `args` is a non-empty dict of allowed keys with bounded int values."""
    return (isinstance(args, dict) and bool(args) and set(args) <= allowed
            and all(type(v) is int and abs(v) <= bound for v in args.values()))


def _literal(text: str) -> Any:
    """Python literal from untrusted text; every failure becomes ValueError.

    A bare identifier is read as that identifier's name, so domains whose
    arguments are named values (``rotate(axis=z, degrees=90)``) round-trip
    through ``describe`` without the parser needing to know the vocabulary.
    """
    if len(text) > 512:
        raise ValueError("argument too long")
    stripped = text.strip()
    if _IDENT.fullmatch(stripped):
        return stripped
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        raise ValueError(f"not a literal: {stripped!r}") from exc


class DomainRegistry:
    """Name -> domain instance; resolves which domain can represent a task."""

    def __init__(self, domains: Iterable[Domain] = ()):
        self._domains: Dict[str, Domain] = {}
        for d in domains:
            self.register(d)

    def register(self, domain: Domain) -> None:
        if domain.name in self._domains:
            raise ValueError(f"domain '{domain.name}' already registered")
        self._domains[domain.name] = domain

    def get(self, name: str) -> Domain:
        return self._domains[name]

    def for_demos(self, demos: Sequence[Example]) -> Domain:
        for domain in self._domains.values():
            if domain.accepts(demos):
                return domain
        raise LookupError("no registered domain can represent these examples")

    def __iter__(self):
        return iter(self._domains.values())

    def __len__(self) -> int:
        return len(self._domains)
