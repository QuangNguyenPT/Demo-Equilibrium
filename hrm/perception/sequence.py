"""Sequence domain: integer sequences in, integer sequences out.

Encoding: bundle_i POS^i VAL^{x_i}, bound with LEN^n. Adding b to every
element IS binding VAL^b and rotating by k IS (up to wrap-around) binding
POS^k, so those programs are found algebraically; the rest (scaling,
reverse, sort, running sums, differences) carry named symbols and are found
by energy measurement and synthesis.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hrm.config import PerceptionConfig
from hrm.perception.base import Domain, Example, Program, ProgramError, int_kwargs
from hrm.substrate import Codebook, PhasorVector, bundle


class SeqOp(ABC):
    name: ClassVar[str] = "op"

    @abstractmethod
    def run(self, xs: List[int]) -> List[int]:
        ...

    @abstractmethod
    def signature(self, cb: Codebook) -> PhasorVector:
        ...

    @abstractmethod
    def describe(self) -> str:
        ...

    @property
    def identity(self) -> bool:
        return False


@dataclass(frozen=True)
class Affine(SeqOp):
    a: int = 1
    b: int = 0
    name: ClassVar[str] = "affine"

    def run(self, xs: List[int]) -> List[int]:
        return [self.a * x + self.b for x in xs]

    def signature(self, cb: Codebook) -> PhasorVector:
        shift = cb.power("seq.VAL", self.b)
        return shift if self.a == 1 else cb.symbol(f"seq.scale.{self.a}") * shift

    def describe(self) -> str:
        return f"affine(a={self.a}, b={self.b})"

    @property
    def identity(self) -> bool:
        return self.a == 1 and self.b == 0


@dataclass(frozen=True)
class Rotate(SeqOp):
    k: int = 1                     # out[i] = in[i - k]
    name: ClassVar[str] = "rotate"

    def run(self, xs: List[int]) -> List[int]:
        if not xs:
            return []
        k = self.k % len(xs)
        return xs[-k:] + xs[:-k] if k else list(xs)

    def signature(self, cb: Codebook) -> PhasorVector:
        return cb.power("seq.POS", self.k)

    def describe(self) -> str:
        return f"rotate(k={self.k})"


@dataclass(frozen=True)
class Reverse(SeqOp):
    name: ClassVar[str] = "reverse"

    def run(self, xs: List[int]) -> List[int]:
        return xs[::-1]

    def signature(self, cb: Codebook) -> PhasorVector:
        return cb.symbol("seq.reverse")

    def describe(self) -> str:
        return "reverse"


@dataclass(frozen=True)
class Sort(SeqOp):
    descending: bool = False
    name: ClassVar[str] = "sort"

    def run(self, xs: List[int]) -> List[int]:
        return sorted(xs, reverse=self.descending)

    def signature(self, cb: Codebook) -> PhasorVector:
        return cb.symbol("seq.sort.desc" if self.descending else "seq.sort.asc")

    def describe(self) -> str:
        return "sort(desc)" if self.descending else "sort(asc)"


@dataclass(frozen=True)
class CumSum(SeqOp):
    name: ClassVar[str] = "cumsum"

    def run(self, xs: List[int]) -> List[int]:
        return [int(v) for v in np.cumsum(xs)] if xs else []

    def signature(self, cb: Codebook) -> PhasorVector:
        return cb.symbol("seq.cumsum")

    def describe(self) -> str:
        return "cumsum"


@dataclass(frozen=True)
class Diff(SeqOp):
    name: ClassVar[str] = "diff"

    def run(self, xs: List[int]) -> List[int]:
        if len(xs) < 2:
            raise ProgramError("diff needs at least two elements")
        return [b - a for a, b in zip(xs, xs[1:])]

    def signature(self, cb: Codebook) -> PhasorVector:
        return cb.symbol("seq.diff") * cb.power("seq.LEN", -1)

    def describe(self) -> str:
        return "diff"


OPS: Dict[str, type] = {cls.name: cls for cls in (Affine, Rotate, Reverse, Sort, CumSum, Diff)}


@dataclass
class SequenceProgram(Program):
    ops: Tuple[SeqOp, ...] = field(default_factory=tuple)
    provenance: str = "search"

    def apply(self, x: Any) -> List[int]:
        xs = as_sequence(x)
        if xs is None:
            raise ProgramError("sequence program needs a 1-D integer sequence")
        for op in self.ops:
            xs = op.run(xs)
        return xs

    def describe(self) -> str:
        return " . ".join(op.describe() for op in self.ops) or "identity"

    def signature(self, cb: Codebook) -> PhasorVector:
        sig = PhasorVector.identity(cb.dim, cb.dtype)
        for op in self.ops:
            sig = sig * op.signature(cb)
        return sig

    @property
    def complexity(self) -> int:
        return sum(1 for op in self.ops if not op.identity)

    @property
    def family(self) -> str:
        return "+".join(op.name for op in self.ops if not op.identity) or "identity"

    def to_payload(self) -> Dict[str, Any]:
        return {"ops": [{"op": op.name, **asdict(op)} for op in self.ops]}

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], provenance: str = "memory") -> "SequenceProgram":
        ops = []
        for spec in payload["ops"]:
            spec = dict(spec)
            ops.append(OPS[spec.pop("op")](**spec))
        return cls(tuple(ops), provenance)


class SequenceDomain(Domain):
    name = "sequence"
    closed_families = ("affine", "rotate", "reverse", "sort")
    format_example = ((1, 2, 3), "affine(a=2, b=0) . reverse")

    def __init__(self, cb: Codebook, cfg: PerceptionConfig):
        super().__init__(cb)
        self.max_scale = cfg.seq_max_scale
        self.max_offset = cfg.seq_max_offset

    def accepts(self, demos: Sequence[Example]) -> bool:
        return len(demos) > 0 and all(as_sequence(x) is not None for pair in demos for x in pair)

    def encode(self, x: Any, parse: int = 0) -> PhasorVector:
        xs = as_sequence(x)
        if xs is None:
            raise ValueError("not an integer sequence")
        length = self.cb.power("seq.LEN", len(xs))
        if not xs:
            return length
        return bundle(self.cb.power("seq.POS", i) * self.cb.power("seq.VAL", v)
                      for i, v in enumerate(xs)) * length

    def candidates(self, demos: Sequence[Example]) -> List[Program]:
        m, o = self.max_scale, self.max_offset
        ops: List[SeqOp] = [Affine(1, 0)]
        ops += [Affine(a, b) for a in range(-m, m + 1) for b in range(-o, o + 1)
                if a != 0 and (a, b) != (1, 0)]
        longest = max((len(as_sequence(i) or []) for i, _ in demos), default=0)
        ops += [Rotate(k) for k in range(1, longest)]
        ops += [Reverse(), Sort(False), Sort(True), CumSum(), Diff()]
        fitted = self._fitted_affine(demos)
        if fitted is not None and fitted not in ops:
            ops.append(fitted)
        return [SequenceProgram((op,)) for op in ops]

    @staticmethod
    def _fitted_affine(demos: Sequence[Example]) -> Optional[Affine]:
        """Integer y = a x + b fitted across all demos, if the data implies one."""
        xs: List[int] = []
        ys: List[int] = []
        for inp, out in demos:
            i, o = as_sequence(inp), as_sequence(out)
            if i is None or o is None or len(i) != len(o):
                return None
            xs += i
            ys += o
        if len(set(xs)) < 2:
            return None
        a, b = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
        a_i, b_i = int(round(a)), int(round(b))
        if a_i == 0 or any(a_i * x + b_i != y for x, y in zip(xs, ys)):
            return None
        return Affine(a_i, b_i)

    def compose(self, first: Program, then: Program) -> Program:
        if not (isinstance(first, SequenceProgram) and isinstance(then, SequenceProgram)):
            raise TypeError("sequence domain composes SequencePrograms only")
        return SequenceProgram(first.ops + then.ops, provenance="synthesized")

    def equal(self, pred: Any, target: Any) -> bool:
        p, t = as_sequence(pred), as_sequence(target)
        return p is not None and p == t

    def similarity(self, pred: Any, target: Any) -> float:
        p, t = as_sequence(pred), as_sequence(target)
        if p is None or t is None:
            return 0.0
        if p == t:
            return 1.0
        # order-sensitive terms (exact, close) + an order-free value multiset term:
        # "right values, wrong order" earns credit, giving search a gradient
        n = max(len(p), len(t))
        exact = sum(1 for a, b in zip(p, t) if a == b) / n
        close = sum(np.exp(-abs(a - b)) for a, b in zip(p, t)) / n
        multiset = sum((Counter(p) & Counter(t)).values()) / n
        return float(0.35 * exact + 0.25 * close + 0.25 * multiset + 0.15 * min(len(p), len(t)) / n)

    def size(self, x: Any) -> int:
        return len(as_sequence(x) or [])

    def program_from_payload(self, payload: Dict[str, Any]) -> Program:
        return SequenceProgram.from_payload(payload)

    def vocabulary(self) -> str:
        return ("affine(a=INT, b=INT) maps every x to a*x+b; rotate(k=INT) shifts elements "
                "right by k (wrapping); reverse; sort(asc) or sort(desc); cumsum gives running "
                "totals; diff gives differences of neighbours")

    def program_from_terms(self, terms: List[Tuple[str, Any]]) -> Program:
        ops: List[SeqOp] = []
        for name, args in terms:
            if name == "identity" and args is None:
                continue
            if name == "affine" and int_kwargs(args, {"a", "b"}, 1000):
                ops.append(Affine(**args))
            elif name == "rotate" and int_kwargs(args, {"k"}, 1000):
                ops.append(Rotate(**args))
            elif name == "sort" and args in ("asc", "desc"):
                ops.append(Sort(args == "desc"))
            elif name in ("reverse", "cumsum", "diff") and args is None:
                ops.append(OPS[name]())
            else:
                raise ValueError(f"not a sequence operation: {name}({args})")
        return SequenceProgram(tuple(ops), provenance="language")


def as_sequence(x: Any) -> Optional[List[int]]:
    """Plain list of ints for a 1-D integer sequence, else None."""
    try:
        arr = np.asarray(x)
    except (ValueError, TypeError):
        return None
    if arr.ndim != 1 or (arr.size and arr.dtype.kind not in "iu"):
        return None
    return [int(v) for v in arr.tolist()]
