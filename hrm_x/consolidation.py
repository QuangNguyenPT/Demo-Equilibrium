"""Library growth by MDL gain (doc 03, section 7): verified concepts become new basis features.

A solved task leaves a fitted reading h_t (family + exact parameters). A candidate macro mu is the
function h_t itself. mu enters the library when
    dL(mu) = sum_t [ L(sol_t | Lambda) - L(sol_t | Lambda + mu) ] - L(mu)  > 0,
with L(sol | Lambda) = family bits + parameter bits (the same code as the posterior's Occam term).
Once in the library, mu is a feature: new families are the supports over {mu, x, y, 1} that contain mu,
so the hypothesis language GROWS (e.g. x^y + x becomes expressible after x^y was learned).

How this relates to `hrm_x/library.py`. That module mines PROGRAM subtrees: it compresses programs
that are already known into named DSL primitives, so the program language grows. This one operates on
CONCEPT readings: a fitted family plus exact rationals becomes a macro, so the hypothesis language of
`hrm_x/concepts.py` grows. The two act at different levels of the same architecture, and both are kept:
a program primitive can expose a concept that consolidation then has to read, and a macro can make a
concept expressible that a program must then call.

Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from hrm_x.concepts import FAMILIES, Family, Fitted, param_bits

Q = Fraction


@dataclass
class Macro:
    name: str
    fn: Callable[[Q, Q], Optional[Q]]
    bits: float                       # L(mu): the description length of its definition


@dataclass
class MacroLibrary:
    macros: List[Macro] = field(default_factory=list)

    def families(self) -> List[Family]:
        fam = list(FAMILIES)
        basis = {"x": lambda x, y: x, "y": lambda x, y: y, "1": lambda x, y: Q(1)}
        for m in self.macros:
            for r in range(0, 3):
                for extra in itertools.combinations(["x", "y", "1"], r):
                    feats = (m.fn,) + tuple(basis[k] for k in extra)
                    fam.append(Family(f"lib[{m.name}]" + ("+" + "+".join(extra) if extra else ""), feats,
                                      bits=math.log2(max(1, len(self.macros))) + 1 + len(extra) + math.log2(4),
                                      laws=f"linear in library feature {m.name}"))
        return fam


def instance_fn(f: Fitted) -> Callable[[Q, Q], Optional[Q]]:
    def fn(x: Q, y: Q, f=f) -> Optional[Q]:
        r = f.family.row(x, y)
        return None if r is None else sum((a * b for a, b in zip(r, f.params)), Q(0))
    return fn


def description_bits(f: Fitted) -> float:
    return f.family.bits + param_bits(f.params)


def consolidate(lib: MacroLibrary, solved: Sequence[Tuple[str, Fitted]],
                unsolved: Sequence[Tuple[str, list]] = (), cfg=None, min_gain: float = 0.0) -> List[Dict]:
    """Sleep-time abstraction. Candidates = behaviour classes of the winning readings of solved tasks.
    Gain (bits) of adding mu to the library, over the whole corpus of attempted tasks:
      solved tasks   : L(sol) - L(sol | mu)                      (re-description saving)
      unsolved tasks : [bits_bottom + (n-1) log2|Y|] - L(sol | mu)   if a mu-family now fits with surplus >= 1
                       (the task was only describable as 'outside my language' plus its raw data)
    minus L(mu). DreamCoder-style: abstraction is kept only if it compresses the corpus."""
    # Lazy: `concepts` is the heavy L-level module and this is sleep-time code. Keeping the import
    # inside the function lets callers use the Macro/MacroLibrary types without paying for a fit.
    from hrm_x.concepts import ConceptConfig, fit
    cfg = cfg or ConceptConfig()
    probes = [(Q(a), Q(b)) for a, b in [(1, 2), (2, 3), (3, 2), (2, 5), (5, 1), (1, 1)]]
    groups: Dict[Tuple, List[Tuple[str, Fitted]]] = {}
    for name, f in solved:
        fp = tuple(instance_fn(f)(x, y) for x, y in probes)
        groups.setdefault(fp, []).append((name, f))
    report = []
    for fp, members in groups.items():
        f0 = members[0][1]
        L_mu = description_bits(f0)
        mu = Macro(f"{f0.family.name}|{'/'.join(str(p) for p in f0.params)}", instance_fn(f0), L_mu)
        trial = MacroLibrary(lib.macros + [mu])
        fams = [f for f in trial.families() if f.name.startswith(f"lib[{mu.name}]")]
        with_mu = math.log2(len(trial.macros)) + 1 + math.log2(4)
        gain_solved = sum(description_bits(f) - with_mu for _, f in members)
        gain_unsolved, newly = 0.0, []
        for name, evidence in unsolved:
            bel = fit(evidence, cfg, fams)
            good = [f for f in bel.fitted if f.surplus >= 1]
            if good:
                best = min(good, key=lambda f: f.family.bits + param_bits(f.params))
                gain_unsolved += cfg.bits_bottom + (len(evidence) - 1) * cfg.log2_answer_space - (best.family.bits + param_bits(best.params))
                newly.append(name)
        gain = gain_solved + gain_unsolved - L_mu
        added = gain > min_gain
        if added:
            lib.macros.append(mu)
        report.append({"macro": mu.name, "support": [m for m, _ in members], "gain_bits": round(gain, 2),
                       "gain_solved": round(gain_solved, 2), "gain_unsolved": round(gain_unsolved, 2),
                       "newly_solvable": newly, "added": added})
    return report
