"""model.hrmc: the whole trained core in one pickle-free file whose certificate is recomputed on load (F1).

Layout
    0      magic b"HRMCORE\\0"
    8      u32 format version
    12     u64 header length L
    20     header: canonical JSON (config, tensor directory, certificate claims, probes, knowledge
           schema, lineage)
    ...    zero padding to 4096; each tensor's raw little-endian bytes, 4096-aligned
    end-32 SHA-256 of every preceding byte
The loader refuses a file whose SHA-256, any tensor hash, the config, the recomputed certificate
(Γ < 1 and agreement with the claims) or the probe self-test fails.
"""

import copy
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from hrm_eqb.config import ModelConfig
from hrm_eqb.field import collate, encode_task
from hrm_eqb.knowledge import schema_hash, schema_of
from hrm_eqb.model import HRMEQB
from hrm_eqb.tasks import Task, generate

MAGIC = b"HRMCORE\0"
VERSION = 1
ALIGN = 4096
PROBE_SEED = 20260927
PROBE_TOLERANCE = 1e-3
CLAIM_TOLERANCE = 1e-6


class HRMCError(ValueError):
    """The file is not a valid, untampered HRM-EQB core."""


def _pad(n: int) -> int:
    return (-n) % ALIGN


def probe_tasks() -> List[Task]:
    """Fixed, seeded tasks the core must reproduce on load."""
    rng = np.random.default_rng(PROBE_SEED)
    return [generate("sequence", "reverse", rng), generate("text", "upper", rng),
            generate("grid", "flip", rng, max_side=5), generate("sequence", "offset", rng)]


@torch.no_grad()
def probe_logits(model: HRMEQB) -> np.ndarray:
    """First answer logits of the probe tasks, on a CPU copy with an empty brain (core only)."""
    cpu = copy.deepcopy(model).to("cpu").eval()
    cpu.memory.silent, cpu.memory.overlay = True, None
    batch = collate([encode_task(t) for t in probe_tasks()])
    return cpu(batch, segments=1)[0].logits[:, :4, :].numpy().astype(np.float64)


def certificate(model: HRMEQB) -> Dict[str, Any]:
    """Γ, Γ_A, the four measured norms and r_n for the configured n_low (from SVDs of the weights)."""
    b = model.core.bounds()
    rate, weights = b.perron(model.cfg.n_low)
    return {"gamma": b.gamma(), "gamma_a": b.gamma_a(), "rate": rate, "weights": list(weights),
            "norms": {"s_l": b.s_l, "s_h": b.s_h, "v_h": b.v_h, "v_l": b.v_l},
            "n_low": model.cfg.n_low}


def save(model: HRMEQB, path: Path, lineage: Optional[Dict[str, Any]] = None) -> str:
    """Write the core; returns the file's SHA-256 (its identity in lineage records)."""
    cfg = model.cfg
    schema = schema_of(cfg.memory_side, cfg.key_dim, cfg.value_quaternions)
    meta = {
        "config": cfg.to_dict(),
        "certificate": certificate(model),
        "probes": {"seed": PROBE_SEED, "tolerance": PROBE_TOLERANCE, "logits": probe_logits(model).round(6).tolist()},
        "knowledge": {"schema": schema, "schema_sha256": schema_hash(schema)},
        "lineage": {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(lineage or {})},
    }
    state = {k: v.detach().cpu().float().contiguous() for k, v in model.core_state().items()}
    return write(path, state, meta)


def write(path: Path, state: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> str:
    """Serialize tensors + metadata in the .hrmc layout (tensor directory and hashes computed here)."""
    directory, blobs, offset = {}, [], 0
    for name in sorted(state):
        raw = state[name].detach().cpu().float().contiguous().numpy().tobytes()
        directory[name] = {"dtype": "float32", "shape": list(state[name].shape), "offset": offset,
                           "nbytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        blobs.append(raw + b"\0" * _pad(len(raw)))
        offset += len(raw) + _pad(len(raw))
    header = {"format": "hrmc", "version": VERSION, "tensors": directory, **meta}
    head = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    prefix = MAGIC + struct.pack("<IQ", VERSION, len(head)) + head
    body = prefix + b"\0" * _pad(len(prefix)) + b"".join(blobs)
    digest = hashlib.sha256(body).digest()
    Path(path).write_bytes(body + digest)
    return digest.hex()


def read_header(path: Path) -> Tuple[Dict[str, Any], bytes, int]:
    """(header, whole file, offset of the first tensor) after magic, version and SHA-256 checks."""
    raw = Path(path).read_bytes()
    if len(raw) < 52 or raw[:8] != MAGIC:
        raise HRMCError("not an .hrmc file (bad magic)")
    if hashlib.sha256(raw[:-32]).digest() != raw[-32:]:
        raise HRMCError("file SHA-256 mismatch: corrupted or tampered")
    version, hlen = struct.unpack("<IQ", raw[8:20])
    if version != VERSION:
        raise HRMCError(f"unsupported .hrmc version {version}")
    base = 20 + hlen
    return json.loads(raw[20:base]), raw, base + _pad(base)


def load(path: Path) -> Tuple[HRMEQB, Dict[str, Any]]:
    """Rebuild and verify the core (knowledge not attached: an empty brain until one is)."""
    header, raw, base = read_header(path)
    cfg = ModelConfig.from_dict(header["config"])
    state = {}
    for name, entry in header["tensors"].items():
        start = base + entry["offset"]
        blob = raw[start:start + entry["nbytes"]]
        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            raise HRMCError(f"tensor {name} hash mismatch")
        state[name] = torch.from_numpy(np.frombuffer(blob, dtype=np.float32).copy()).reshape(entry["shape"])
    model = HRMEQB(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if sorted(missing) != sorted(HRMEQB.knowledge_names()) or unexpected:
        raise HRMCError(f"tensor directory mismatch: missing={missing} unexpected={unexpected}")
    _verify_certificate(model, header["certificate"])
    got = probe_logits(model)
    want = np.asarray(header["probes"]["logits"])
    if got.shape != want.shape or float(np.abs(got - want).max()) > header["probes"]["tolerance"]:
        raise HRMCError("probe self-test failed: the weights do not compute what they computed when saved")
    return model, header


def _verify_certificate(model: HRMEQB, claims: Dict[str, Any]) -> None:
    measured = certificate(model)
    if not measured["gamma"] < 1.0:
        raise HRMCError(f"recomputed Γ = {measured['gamma']:.4f} >= 1: convergence not certified")
    for key in ("gamma", "gamma_a", "rate"):
        if abs(measured[key] - claims[key]) > CLAIM_TOLERANCE:
            raise HRMCError(f"certificate claim {key}={claims[key]} disagrees with the weights ({measured[key]})")
