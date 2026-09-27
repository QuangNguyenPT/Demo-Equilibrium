"""The core's weights: everything learned that lives in RAM.

Small by design: the resonator's search scalars, the H-level mixing logits,
the Q-head and the consolidated prior table. The large, growing part of what
the model knows lives in the SSD knowledge store. A core snapshot is plain
JSON so every generation is inspectable and diffable.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from hrm.perception.base import DomainRegistry
from hrm.reasoning import QHead, Reasoner

CORE_FORMAT = 1


def core_to_dict(reasoner: Reasoner, qhead: QHead) -> Dict[str, Any]:
    engine = reasoner.engine
    return {
        "format": CORE_FORMAT,
        "resonator": {k: v.tolist() for k, v in engine.resonator.params.items()},
        "mixing": {"base_logits": engine.mixing.base_logits.tolist(),
                   "feature_scale": engine.mixing.feature_scale},
        "qhead": qhead.w.tolist(),
        "priors": reasoner.priors.to_list(),
    }


def core_from_dict(data: Dict[str, Any], reasoner: Reasoner, qhead: QHead,
                   domains: DomainRegistry) -> None:
    if data.get("format") != CORE_FORMAT:
        raise ValueError(f"unsupported core format {data.get('format')}")
    engine = reasoner.engine
    for key, current in engine.resonator.params.items():
        value = np.asarray(data["resonator"][key], dtype=np.float64)
        if value.shape != current.shape:
            raise ValueError(f"core resonator.{key} has shape {value.shape}, "
                             f"expected {current.shape}")
        engine.resonator.params[key] = value
    logits = np.asarray(data["mixing"]["base_logits"], dtype=np.float64)
    if logits.shape != engine.mixing.base_logits.shape:
        raise ValueError("core mixing logits do not match n_streams")
    engine.mixing.base_logits = logits
    engine.mixing.feature_scale = float(data["mixing"]["feature_scale"])
    w = np.asarray(data["qhead"], dtype=np.float64)
    if w.shape != qhead.w.shape:
        raise ValueError("core qhead weights have the wrong shape")
    qhead.w = w
    reasoner.priors.load_list(data["priors"], domains)
