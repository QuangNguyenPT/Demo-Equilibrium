"""knowledge.pt: the model's knowledge as a normal PyTorch archive, rewritten in place on SSD (F2).

`torch.load(path, weights_only=True)` reads it anywhere. At runtime the row tensors are
mapped MAP_SHARED, so a write lands on disk without loading the file into RAM. Every write
is journaled first (write-ahead log, fsync), then applied; recovery replays a complete
journal and drops a torn one, so after any crash the rows are either all old or all new
(spec §14, F3). `seal()` rewrites the archive atomically so its zip checksums are valid again.
"""

import hashlib
import json
import os
import pickle
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

ROW_TENSORS = ("values", "usage", "confidence", "generation")
FIXED_TENSORS = ("keys_a", "keys_b")
WAL_MAGIC = b"HRMWAL1\0"
_DTYPES = {"FloatStorage": np.float32, "LongStorage": np.int64}


def schema_of(side: int, key_dim: int, value_quaternions: int) -> Dict[str, int]:
    return {"side": side, "key_dim": key_dim, "value_quaternions": value_quaternions}


def schema_hash(schema: Dict[str, int]) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def create(path: Path, keys_a: torch.Tensor, keys_b: torch.Tensor, values: torch.Tensor,
           schema: Dict[str, int]) -> None:
    """Write a fresh knowledge file (usage, confidence, generation start at zero)."""
    slots = values.shape[0]
    payload = {
        "schema": json.dumps(schema, sort_keys=True),
        "keys_a": keys_a.detach().cpu().float().contiguous(),
        "keys_b": keys_b.detach().cpu().float().contiguous(),
        "values": values.detach().cpu().float().contiguous(),
        "usage": torch.zeros(slots, dtype=torch.float32),
        "confidence": torch.zeros(slots, dtype=torch.float32),
        "generation": torch.zeros(slots, dtype=torch.int64),
    }
    _atomic_save(payload, Path(path))


def _atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    with open(tmp, "rb+") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class _StorageKeys(pickle.Unpickler):
    """Reads data.pkl without building tensors: name -> (storage key, dtype, offset, shape)."""

    def find_class(self, module: str, name: str):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return lambda storage, offset, size, stride, *rest: (storage, int(offset), tuple(size), tuple(stride))
        if module == "torch" and name in _DTYPES:
            return name
        if (module, name) == ("collections", "OrderedDict"):
            return dict
        raise pickle.UnpicklingError(f"unexpected global {module}.{name} in knowledge file")

    def persistent_load(self, pid):
        return pid[2], pid[1]


def _payload_offsets(path: Path) -> Dict[str, Tuple[int, np.dtype, Tuple[int, ...]]]:
    """name -> (byte offset of the tensor's data in the file, dtype, shape)."""
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
        prefix = pkl[: -len("data.pkl")]
        entries = _StorageKeys(zf.open(pkl)).load()
        infos = {i.filename: i for i in zf.infolist()}
    out = {}
    with open(path, "rb") as fh:
        for name, spec in entries.items():
            if not isinstance(spec, tuple):
                continue
            (key, storage_type), offset, shape, _ = spec
            info = infos[f"{prefix}data/{key}"]
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"{name} is compressed; cannot map it in place")
            fh.seek(info.header_offset)
            head = fh.read(30)
            start = info.header_offset + 30 + struct.unpack("<H", head[26:28])[0] + struct.unpack("<H", head[28:30])[0]
            dtype = np.dtype(_DTYPES[storage_type])
            out[name] = (start + offset * dtype.itemsize, dtype, shape)
    return out


class KnowledgeFile:
    """An open knowledge file: fixed keys in RAM, row tensors mapped from disk for in-place writes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.wal = self.path.with_name(self.path.name + ".wal")
        recover(self.path)
        data = torch.load(self.path, weights_only=True, mmap=True, map_location="cpu")
        self.schema = json.loads(data["schema"])
        self.fixed = {n: data[n].clone() for n in FIXED_TENSORS}
        del data
        self.maps: Dict[str, np.memmap] = {}
        for name, (offset, dtype, shape) in _payload_offsets(self.path).items():
            if name in ROW_TENSORS:
                self.maps[name] = np.memmap(self.path, dtype=dtype, mode="r+", offset=offset, shape=shape)

    def tensor(self, name: str) -> torch.Tensor:
        """Zero-copy torch view of a mapped row tensor (reads see writes immediately)."""
        return torch.from_numpy(self.maps[name])

    def write_rows(self, rows: np.ndarray, updates: Dict[str, np.ndarray]) -> None:
        """Journal then apply new contents for `rows` of the given row tensors."""
        rows = np.asarray(rows, dtype=np.int64)
        journal = _encode_journal(rows, {k: np.ascontiguousarray(v, dtype=self.maps[k].dtype)
                                         for k, v in updates.items()})
        with open(self.wal, "wb") as fh:
            fh.write(journal)
            fh.flush()
            os.fsync(fh.fileno())
        self._apply(rows, updates)
        os.remove(self.wal)

    def _apply(self, rows: np.ndarray, updates: Dict[str, np.ndarray]) -> None:
        for name, new in updates.items():
            self.maps[name][rows] = new
            self.maps[name].flush()

    def close(self) -> None:
        for m in self.maps.values():
            m.flush()
        self.maps.clear()


def _encode_journal(rows: np.ndarray, updates: Dict[str, np.ndarray]) -> bytes:
    header = json.dumps({"rows": int(rows.size),
                         "fields": {k: [str(v.dtype), list(v.shape)] for k, v in updates.items()}}).encode()
    body = struct.pack("<I", len(header)) + header + rows.tobytes() + b"".join(v.tobytes() for v in updates.values())
    return WAL_MAGIC + body + struct.pack("<I", zlib.crc32(body))


def _decode_journal(raw: bytes):
    """(rows, updates) of a complete journal, or None if it is torn or corrupt."""
    if len(raw) < len(WAL_MAGIC) + 8 or not raw.startswith(WAL_MAGIC):
        return None
    body, crc = raw[len(WAL_MAGIC):-4], struct.unpack("<I", raw[-4:])[0]
    if zlib.crc32(body) != crc:
        return None
    hlen = struct.unpack("<I", body[:4])[0]
    header = json.loads(body[4:4 + hlen])
    pos = 4 + hlen
    rows = np.frombuffer(body[pos:pos + 8 * header["rows"]], dtype=np.int64)
    pos += 8 * header["rows"]
    updates = {}
    for name, (dtype, shape) in header["fields"].items():
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        updates[name] = np.frombuffer(body[pos:pos + n], dtype=dtype).reshape(shape)
        pos += n
    return rows, updates


def recover(path: Path) -> str:
    """Replay a complete journal or drop a torn one; returns 'clean', 'replayed' or 'discarded'."""
    wal = Path(path).with_name(Path(path).name + ".wal")
    if not wal.exists():
        return "clean"
    decoded = _decode_journal(wal.read_bytes())
    if decoded is None:
        os.remove(wal)
        return "discarded"
    rows, updates = decoded
    offsets = _payload_offsets(Path(path))
    for name, new in updates.items():
        offset, dtype, shape = offsets[name]
        view = np.memmap(path, dtype=dtype, mode="r+", offset=offset, shape=shape)
        view[rows] = new
        view.flush()
        del view
    os.remove(wal)
    return "replayed"


def seal(path: Path) -> None:
    """Rewrite the archive atomically (valid zip checksums); no KnowledgeFile may be open on it."""
    data = torch.load(path, weights_only=True, map_location="cpu")
    _atomic_save({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}, Path(path))
