"""Shared experiment plumbing: seeded runs, result records, and a results directory under version control."""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict

RESULTS = Path(__file__).resolve().parent / "results"


def record(name: str, config: Dict[str, Any], metrics: Dict[str, Any]) -> Path:
    """Write experiments/results/<name>.json with the config needed to reproduce it."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    payload = {"experiment": name, "config": config, "metrics": metrics,
               "python": sys.version.split()[0], "platform": platform.platform(),
               "recorded": time.strftime("%Y-%m-%d %H:%M:%S")}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[RESULT] wrote {path.name}")
    return path


def table(rows, headers) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))
    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows])
