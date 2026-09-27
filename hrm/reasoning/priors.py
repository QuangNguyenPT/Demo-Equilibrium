"""Consolidated knowledge held in the core (RAM).

When the learning section consolidates a program out of the SSD knowledge
store, it lands here: the program joins every future candidate set of its
domain with a learned prior, so it is found without a knowledge read. This is
the "knowledge flows into the neurons" path; the table is part of the core's
weights and is saved, promoted and rolled back with them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from hrm.perception.base import Domain, DomainRegistry, Program

_Key = Tuple[str, str]


class PriorTable:
    def __init__(self):
        self._entries: Dict[_Key, Tuple[float, Program]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def prior(self, domain: str, program: Program) -> float:
        entry = self._entries.get((domain, program.describe()))
        return entry[0] if entry else 0.0

    def has(self, domain: str, description: str) -> bool:
        return (domain, description) in self._entries

    def consolidate(self, domain: str, program: Program, amount: float,
                    cap: float = 1.0) -> float:
        """Raise the program's prior by `amount` (capped); returns the new prior."""
        key = (domain, program.describe())
        prior = min(cap, self._entries.get(key, (0.0, program))[0] + amount)
        self._entries[key] = (prior, program)
        return prior

    def forget(self, domain: str, description: str) -> bool:
        """Drop a consolidated program (e.g. a human said it produced a wrong answer)."""
        return self._entries.pop((domain, description), None) is not None

    def programs(self, domain: Domain) -> List[Tuple[Program, float]]:
        return [(program, prior) for (name, _), (prior, program) in self._entries.items()
                if name == domain.name]

    def snapshot(self) -> Dict[_Key, Tuple[float, Program]]:
        return dict(self._entries)

    def restore(self, snap: Dict[_Key, Tuple[float, Program]]) -> None:
        self._entries = dict(snap)

    def to_list(self) -> List[Dict[str, Any]]:
        return [{"domain": d, "prior": prior, "payload": program.to_payload()}
                for (d, _), (prior, program) in sorted(self._entries.items())]

    def load_list(self, rows: List[Dict[str, Any]], domains: DomainRegistry) -> None:
        entries: Dict[_Key, Tuple[float, Program]] = {}
        for row in rows:
            program = domains.get(row["domain"]).program_from_payload(row["payload"])
            program.provenance = "consolidated"
            entries[(row["domain"], program.describe())] = (float(row["prior"]), program)
        self._entries = entries
