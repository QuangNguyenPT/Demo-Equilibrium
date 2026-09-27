"""Reasoning trace: the legible record of how an answer was (or was not) reached."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class TraceEvent:
    tick: int
    kind: str            # fast | step | refresh | restart | accept | abstain | synthesis
    hypothesis: str = ""
    residual: float = 0.0
    energy: float = 0.0
    lam: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)


class Trace:
    def __init__(self, task_id: str = ""):
        self.task_id = task_id
        self.events: List[TraceEvent] = []
        self.started = time.time()

    def log(self, event: TraceEvent) -> None:
        self.events.append(event)

    def of_kind(self, kind: str) -> List[TraceEvent]:
        return [e for e in self.events if e.kind == kind]

    def failed_parses(self) -> List[int]:
        return [e.detail["failed_parse"] for e in self.events
                if e.kind == "restart" and "failed_parse" in e.detail]

    def to_dict(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "duration_s": time.time() - self.started,
                "events": [asdict(e) for e in self.events]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
