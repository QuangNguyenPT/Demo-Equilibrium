"""SSD-resident knowledge: the dynamic part of the model file.

Layout inside the model bundle (``<model>/knowledge/``):
    keys.<seq>.f16     raw memmap, (capacity, 2*dim) float16, interleaved re/im
    knowledge.sqlite   items (programs, facts) with evidence, outcome log, meta

RAM holds only a small index of active items and one streamed key chunk at a
time; keys and payloads stay on disk. Invariants the store enforces itself:
  * unverified knowledge is never written;
  * a key row referenced by a committed item is never modified in place:
    merges write a new row, compaction writes a new key file and switches to
    it in one transaction. Rollback and crash recovery therefore reduce to
    "the database decides which rows are live";
  * a key row is flushed to disk before the transaction referencing it commits;
  * the store refuses to open under a different substrate (dim, seed), because
    every stored key would be meaningless.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple

import numpy as np

from hrm.config import KnowledgeConfig, SubstrateConfig
from hrm.substrate import PhasorVector

log = logging.getLogger("hrm.knowledge")

FORMAT_VERSION = 1
STATUSES = ("active", "quarantined", "graduated", "evicted")
# keys kept through compaction; graduated keys stay so a core rollback can reactivate them
_KEYED_STATUSES = ("active", "quarantined", "graduated")
SOURCES = ("self", "human")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS items (
    id          INTEGER PRIMARY KEY,
    key_row     INTEGER UNIQUE,
    kind        TEXT    NOT NULL,
    domain      TEXT    NOT NULL,
    signature   TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    complexity  INTEGER NOT NULL,
    status      TEXT    NOT NULL,
    status_gen  INTEGER NOT NULL DEFAULT 0,
    successes   INTEGER NOT NULL DEFAULT 0,
    failures    INTEGER NOT NULL DEFAULT 0,
    usage       INTEGER NOT NULL DEFAULT 0,
    margin_mean REAL    NOT NULL DEFAULT 0,
    margin_m2   REAL    NOT NULL DEFAULT 0,
    origin      TEXT    NOT NULL,
    provenance  TEXT    NOT NULL,
    created     REAL    NOT NULL,
    last_used   REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS items_status ON items(status);
CREATE INDEX IF NOT EXISTS items_signature ON items(domain, signature);
CREATE TABLE IF NOT EXISTS outcomes (
    id       INTEGER PRIMARY KEY,
    item_id  INTEGER NOT NULL REFERENCES items(id),
    task_id  TEXT    NOT NULL,
    success  INTEGER NOT NULL,
    margin   REAL    NOT NULL,
    source   TEXT    NOT NULL,
    at       REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    task_id    TEXT PRIMARY KEY,
    domain     TEXT NOT NULL,
    demos      TEXT NOT NULL,
    test_input TEXT,
    created    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS answers (
    task_id    TEXT PRIMARY KEY REFERENCES episodes(task_id),
    status     TEXT NOT NULL,
    program    TEXT,
    family     TEXT,
    answer     TEXT,
    confidence REAL NOT NULL,
    items      TEXT NOT NULL,
    verdict    TEXT,
    at         REAL NOT NULL
);
"""


class KnowledgeStoreError(RuntimeError):
    """The store cannot honour a request without breaking an invariant."""


class Rollback(Exception):
    """Raise inside ``store.transaction()`` to discard it without an error."""


class Verified(Protocol):
    """What the write gate needs from a verification report."""
    passed: bool
    margin: float


@dataclass(frozen=True)
class KnowledgeRecord:
    """A piece of knowledge a domain wants stored."""
    kind: str                   # "program", "fact", ...
    domain: str
    signature: str              # canonical description; equal signature = same knowledge
    payload: Dict[str, Any]     # JSON-serializable; decoded by the owning domain
    complexity: int
    origin: str                 # task id or source that produced it
    provenance: str             # "synthesized", "taught", ...


@dataclass(frozen=True)
class KnowledgeItem:
    id: int
    kind: str
    domain: str
    signature: str
    payload: Dict[str, Any]
    complexity: int
    status: str
    successes: int
    failures: int
    usage: int
    margin_mean: float
    origin: str
    provenance: str
    created: float
    last_used: float

    @property
    def confidence(self) -> float:
        return self.successes / (self.successes + self.failures + 1)


@dataclass(frozen=True)
class Retrieved:
    item: KnowledgeItem
    similarity: float
    score: float


@dataclass(frozen=True)
class Episode:
    """Episodic memory: the demonstrations a task arrived with (JSON-decoded)."""
    task_id: str
    domain: str
    demos: List[Any]
    test_input: Any
    created: float


@dataclass(frozen=True)
class AnswerRecord:
    """What the model answered, and which knowledge items produced the answer."""
    task_id: str
    status: str
    program: Optional[str]
    family: Optional[str]
    answer: Any
    confidence: float
    items: List[int]
    verdict: Optional[str]
    at: float


class KnowledgeStore:
    """Thread-safe; one instance per model bundle directory."""

    def __init__(self, directory: str | Path, substrate: SubstrateConfig,
                 cfg: KnowledgeConfig):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.dim = substrate.dim
        self._width = 2 * substrate.dim                 # float16 values per row
        self._row_bytes = self._width * 2
        self.row_limit = max(int(cfg.max_disk_gb * 2 ** 30) // self._row_bytes, 1)
        self._lock = threading.RLock()
        self._depth = 0
        self._pending_usage: Dict[int, int] = {}
        self._index: Dict[int, Tuple[int, str, int, int, int]] = {}
        self._arrays: Optional[Tuple[np.ndarray, ...]] = None
        self._db = sqlite3.connect(str(self.dir / "knowledge.sqlite"),
                                   isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._init_meta(substrate)
        self._remove_stray_key_files()
        self._keys = self._map_keys()
        self._load_index()

    # --- lifecycle ---------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            if self._db is None:
                return
            self.flush()
            self._keys = None
            self._db.close()
            self._db = None

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._index)

    def flush(self) -> None:
        """Persist buffered usage counters (retrieval does not write per call)."""
        with self._lock:
            if not self._pending_usage:
                return
            pending, self._pending_usage = self._pending_usage, {}
            now = time.time()
            with self.transaction():
                self._db.executemany(
                    "UPDATE items SET usage = usage + ?, last_used = ? WHERE id = ?",
                    [(n, now, item_id) for item_id, n in pending.items()])

    @contextmanager
    def transaction(self) -> Iterator["KnowledgeStore"]:
        """Atomic group of changes; nests via savepoints. Raise ``Rollback`` to discard."""
        with self._lock:
            savepoint = f"sp{self._depth}"
            self._db.execute("BEGIN IMMEDIATE" if self._depth == 0 else f"SAVEPOINT {savepoint}")
            self._depth += 1
            try:
                yield self
            except BaseException as exc:
                self._depth -= 1
                if self._depth == 0:
                    self._db.execute("ROLLBACK")
                else:
                    self._db.execute(f"ROLLBACK TO {savepoint}")
                    self._db.execute(f"RELEASE {savepoint}")
                self._load_index()
                if not isinstance(exc, Rollback):
                    raise
            else:
                self._depth -= 1
                self._db.execute("COMMIT" if self._depth == 0 else f"RELEASE {savepoint}")

    # --- write path ----------------------------------------------------------------
    def write(self, key: PhasorVector, record: KnowledgeRecord,
              report: Verified) -> Optional[int]:
        """Store verified knowledge; returns the item id, or None if refused."""
        if not report.passed:
            log.info("[KNOWLEDGE] refused unverified %s/%s", record.domain, record.signature)
            return None
        vec = self._check_key(key)
        payload = json.dumps(record.payload, sort_keys=True)
        with self._lock, self.transaction():
            self.flush()
            # allocate first: reaching the disk cap may compact and renumber rows
            row = self._allocate_row()
            dup = self._find_duplicate(vec, record)
            if dup is not None:
                item_id, old_row = dup
                merged = self._read_row(old_row) + vec
                merged = merged / np.maximum(np.abs(merged), 1e-12)
                self._write_row(row, merged.astype(np.complex64))   # copy-on-write
                self._db.execute(
                    "UPDATE items SET key_row = ?, successes = successes + 1, last_used = ? "
                    "WHERE id = ?", (row, time.time(), item_id))
                _, domain, succ, fail, cx = self._index[item_id]
                self._index[item_id] = (row, domain, succ + 1, fail, cx)
                self._arrays = None
                return item_id
            self._write_row(row, vec)
            item_id = self._insert_item(row, record, payload, report.margin)
            self._index[item_id] = (row, record.domain, 1, 0, record.complexity)
            self._arrays = None
            return item_id

    def report_outcome(self, item_id: int, success: bool, margin: float,
                       task_id: str, source: str = "self") -> None:
        """Record evidence for or against an item (Welford update of its margin)."""
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        with self._lock, self.transaction():
            row = self._db.execute(
                "SELECT successes, failures, margin_mean, margin_m2 FROM items WHERE id = ?",
                (item_id,)).fetchone()
            if row is None:
                raise KeyError(item_id)
            succ = row["successes"] + (1 if success else 0)
            fail = row["failures"] + (0 if success else 1)
            n = succ + fail
            delta = margin - row["margin_mean"]
            mean = row["margin_mean"] + delta / n
            m2 = row["margin_m2"] + delta * (margin - mean)
            now = time.time()
            self._db.execute(
                "UPDATE items SET successes = ?, failures = ?, margin_mean = ?, margin_m2 = ?, "
                "last_used = ? WHERE id = ?", (succ, fail, mean, m2, now, item_id))
            self._db.execute(
                "INSERT INTO outcomes (item_id, task_id, success, margin, source, at) "
                "VALUES (?, ?, ?, ?, ?, ?)", (item_id, task_id, int(success), margin, source, now))
            if item_id in self._index:
                key_row, domain, _, _, cx = self._index[item_id]
                self._index[item_id] = (key_row, domain, succ, fail, cx)
                self._arrays = None

    def set_status(self, item_id: int, status: str, generation: int = 0) -> None:
        """Change an item's status; `generation` records which model generation did it."""
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with self._lock, self.transaction():
            row = self._db.execute("SELECT key_row FROM items WHERE id = ?",
                                   (item_id,)).fetchone()
            if row is None:
                raise KeyError(item_id)
            if status == "active" and row["key_row"] is None:
                raise KnowledgeStoreError(f"item {item_id} lost its key at compaction; "
                                          "it cannot be reactivated")
            self._db.execute("UPDATE items SET status = ?, status_gen = ? WHERE id = ?",
                             (status, generation, item_id))
            self._load_index()

    def reactivate_graduated(self, after_generation: int) -> Tuple[List[int], List[int]]:
        """Undo graduations made after a generation (core rollback).
        Returns (reactivated ids, ids whose key was already evicted)."""
        with self._lock, self.transaction():
            rows = self._db.execute(
                "SELECT id, key_row FROM items WHERE status = 'graduated' AND status_gen > ?",
                (after_generation,)).fetchall()
            back = [r["id"] for r in rows if r["key_row"] is not None]
            lost = [r["id"] for r in rows if r["key_row"] is None]
            self._db.executemany("UPDATE items SET status = 'active', status_gen = ? WHERE id = ?",
                                 [(after_generation, i) for i in back])
            self._load_index()
            return back, lost

    # --- read path -----------------------------------------------------------------
    def retrieve(self, query: PhasorVector, k: Optional[int] = None,
                 domain: Optional[str] = None) -> List[Retrieved]:
        """Top-k active items by similarity + confidence bonus - complexity penalty."""
        k = k or self.cfg.retrieval_k
        with self._lock:
            ids, rows, domains, conf, cx = self._active_arrays()
            if domain is not None:
                mask = domains == domain
                ids, rows, conf, cx = ids[mask], rows[mask], conf[mask], cx[mask]
            if len(ids) == 0:
                return []
            sims = self._similarities(self._check_key(query), rows)
            scores = sims + self.cfg.beta_conf * conf - self.cfg.beta_cx * cx / 10.0
            kk = min(k, len(ids))
            top = np.argpartition(-scores, kk - 1)[:kk]
            top = top[np.argsort(-scores[top])]
            items = self._fetch([int(ids[i]) for i in top])
            for i in top:
                item_id = int(ids[i])
                self._pending_usage[item_id] = self._pending_usage.get(item_id, 0) + 1
            return [Retrieved(items[int(ids[i])], float(sims[i]), float(scores[i])) for i in top]

    def get(self, item_id: int) -> KnowledgeItem:
        with self._lock:
            items = self._fetch([item_id])
            if item_id not in items:
                raise KeyError(item_id)
            return items[item_id]

    def key_of(self, item_id: int) -> PhasorVector:
        with self._lock:
            row = self._db.execute("SELECT key_row FROM items WHERE id = ?",
                                   (item_id,)).fetchone()
            if row is None or row["key_row"] is None:
                raise KeyError(item_id)
            return PhasorVector(self._read_row(row["key_row"]))

    def items(self, status: str = "active") -> List[KnowledgeItem]:
        with self._lock:
            ids = [r["id"] for r in self._db.execute(
                "SELECT id FROM items WHERE status = ? ORDER BY id", (status,))]
            fetched = self._fetch(ids)
            return [fetched[i] for i in ids]

    # --- episodic memory & answer log -------------------------------------------------
    def record_episode(self, task_id: str, domain: str, demos: Any, test_input: Any) -> None:
        """Store (or extend) the demonstrations of a task; values must be JSON-serializable."""
        with self._lock, self.transaction():
            self._db.execute(
                "INSERT INTO episodes (task_id, domain, demos, test_input, created) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(task_id) DO UPDATE SET "
                "domain = excluded.domain, demos = excluded.demos, "
                "test_input = excluded.test_input",
                (task_id, domain, json.dumps(demos), json.dumps(test_input), time.time()))

    def episode(self, task_id: str) -> Episode:
        with self._lock:
            r = self._db.execute("SELECT * FROM episodes WHERE task_id = ?",
                                 (task_id,)).fetchone()
            if r is None:
                raise KeyError(task_id)
            return Episode(r["task_id"], r["domain"], json.loads(r["demos"]),
                           json.loads(r["test_input"]), r["created"])

    def record_answer(self, task_id: str, status: str, program: Optional[str],
                      family: Optional[str], answer: Any, confidence: float,
                      items: List[int]) -> None:
        with self._lock, self.transaction():
            self._db.execute(
                "INSERT INTO answers (task_id, status, program, family, answer, confidence, "
                "items, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(task_id) DO UPDATE SET "
                "status = excluded.status, program = excluded.program, "
                "family = excluded.family, answer = excluded.answer, "
                "confidence = excluded.confidence, items = excluded.items, verdict = NULL, "
                "at = excluded.at",
                (task_id, status, program, family, json.dumps(answer), confidence,
                 json.dumps(items), time.time()))

    def answer(self, task_id: str) -> AnswerRecord:
        with self._lock:
            r = self._db.execute("SELECT * FROM answers WHERE task_id = ?",
                                 (task_id,)).fetchone()
            if r is None:
                raise KeyError(task_id)
            return AnswerRecord(r["task_id"], r["status"], r["program"], r["family"],
                                json.loads(r["answer"]), r["confidence"], json.loads(r["items"]),
                                r["verdict"], r["at"])

    def set_verdict(self, task_id: str, verdict: str) -> None:
        with self._lock, self.transaction():
            cur = self._db.execute("UPDATE answers SET verdict = ? WHERE task_id = ?",
                                   (verdict, task_id))
            if cur.rowcount == 0:
                raise KeyError(task_id)

    # --- maintenance ---------------------------------------------------------------
    def compact(self) -> int:
        """Rewrite live keys into a fresh file; returns the number of rows reclaimed."""
        with self._lock:
            if self._depth:
                raise KnowledgeStoreError("compact() cannot run inside a transaction")
            self.flush()
            live = self._db.execute(
                f"SELECT id, key_row FROM items WHERE status IN {_KEYED_STATUSES} "
                "AND key_row IS NOT NULL ORDER BY key_row").fetchall()
            old_rows = np.array([r["key_row"] for r in live], dtype=np.int64)
            n_rows = self._meta_int("n_rows")
            capacity = min(max(math.ceil(len(live) * 1.25), self.cfg.initial_rows), self.row_limit)
            seq = self._meta_int("keys_seq") + 1
            new_name = f"keys.{seq}.f16"
            self._create_key_file(new_name, capacity)
            new_keys = np.memmap(self.dir / new_name, dtype=np.float16, mode="r+",
                                 shape=(capacity, self._width))
            chunk = self.cfg.retrieval_chunk
            for lo in range(0, len(old_rows), chunk):
                hi = min(lo + chunk, len(old_rows))
                new_keys[lo:hi] = self._keys[old_rows[lo:hi]]
            new_keys.flush()
            del new_keys
            old_name = self._meta("keys_file")
            with self.transaction():
                self._db.execute(f"UPDATE items SET key_row = NULL "
                                 f"WHERE status NOT IN {_KEYED_STATUSES}")
                # two phases keep key_row UNIQUE while rows are renumbered
                self._db.executemany("UPDATE items SET key_row = ? WHERE id = ?",
                                     [(-(i + 1), r["id"]) for i, r in enumerate(live)])
                self._db.execute("UPDATE items SET key_row = -key_row - 1 WHERE key_row < 0")
                self._meta_set("keys_file", new_name)
                self._meta_set("keys_seq", seq)
                self._meta_set("n_rows", len(live))
            self._keys = None
            self._keys = self._map_keys()
            (self.dir / old_name).unlink(missing_ok=True)
            self._load_index()
            log.info("[KNOWLEDGE] compacted %d -> %d rows", n_rows, len(live))
            return n_rows - len(live)

    def integrity_problems(self) -> List[str]:
        """Empty list when the store is consistent; used by the learning gate."""
        with self._lock:
            problems: List[str] = []
            if self._db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                problems.append("sqlite integrity_check failed")
            n_rows = self._meta_int("n_rows")
            if n_rows > self._keys.shape[0]:
                problems.append(f"n_rows {n_rows} exceeds key capacity {self._keys.shape[0]}")
            bad = self._db.execute(
                f"SELECT COUNT(*) FROM items WHERE status IN {_KEYED_STATUSES} "
                "AND (key_row IS NULL OR key_row < 0 OR key_row >= ?)", (n_rows,)).fetchone()[0]
            if bad:
                problems.append(f"{bad} keyed item(s) point outside the key file")
            _, rows, _, _, _ = self._active_arrays()
            for lo in range(0, len(rows), self.cfg.retrieval_chunk):
                block = self._keys[np.sort(rows[lo:lo + self.cfg.retrieval_chunk])]
                mags = np.abs(block.astype(np.float32).view(np.complex64))
                if mags.size and np.abs(mags - 1.0).max() > 5e-3:
                    problems.append("an active key is not a unit phasor")
                    break
            return problems

    def stats(self) -> Dict[str, int]:
        with self._lock:
            n_rows = self._meta_int("n_rows")
            keyed = self._db.execute(
                f"SELECT COUNT(*) FROM items WHERE status IN {_KEYED_STATUSES} "
                "AND key_row IS NOT NULL").fetchone()[0]
            return {"active": len(self._index), "rows": n_rows, "dead_rows": n_rows - keyed,
                    "capacity": int(self._keys.shape[0]), "row_limit": self.row_limit,
                    "key_file_bytes": int(self._keys.shape[0]) * self._row_bytes}

    # --- internals: meta & files ---------------------------------------------------
    def _init_meta(self, substrate: SubstrateConfig) -> None:
        self._db.executescript(_SCHEMA)
        if self._meta("format_version") is None:
            self._create_key_file("keys.0.f16", min(self.cfg.initial_rows, self.row_limit))
            for key, value in (("format_version", FORMAT_VERSION), ("dim", substrate.dim),
                               ("seed", substrate.seed), ("n_rows", 0),
                               ("keys_seq", 0), ("keys_file", "keys.0.f16")):
                self._meta_set(key, value)
            return
        if self._meta_int("format_version") != FORMAT_VERSION:
            raise KnowledgeStoreError(f"unsupported knowledge format "
                                      f"{self._meta('format_version')}")
        stored = (self._meta_int("dim"), self._meta_int("seed"))
        if stored != (substrate.dim, substrate.seed):
            raise KnowledgeStoreError(
                f"substrate mismatch: store built with dim={stored[0]}, seed={stored[1]}; "
                f"opened with dim={substrate.dim}, seed={substrate.seed}")

    def _meta(self, key: str) -> Optional[str]:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row["value"]

    def _meta_int(self, key: str) -> int:
        return int(self._meta(key))

    def _meta_set(self, key: str, value: Any) -> None:
        self._db.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                         "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))

    def _create_key_file(self, name: str, rows: int) -> None:
        with open(self.dir / name, "wb") as fh:
            fh.truncate(rows * self._row_bytes)

    def _map_keys(self) -> np.memmap:
        path = self.dir / self._meta("keys_file")
        rows = path.stat().st_size // self._row_bytes
        return np.memmap(path, dtype=np.float16, mode="r+", shape=(rows, self._width))

    def _remove_stray_key_files(self) -> None:
        current = self._meta("keys_file")
        for path in self.dir.glob("keys.*.f16"):
            if path.name != current:
                log.info("[KNOWLEDGE] removing stray key file %s", path.name)
                path.unlink()

    # --- internals: rows -----------------------------------------------------------
    def _check_key(self, key: PhasorVector) -> np.ndarray:
        if key.dim != self.dim:
            raise ValueError(f"key dim {key.dim} != store dim {self.dim}")
        if not np.allclose(np.abs(key.data), 1.0, atol=1e-3):
            raise ValueError("knowledge keys must be unit phasors")
        return key.data.astype(np.complex64)

    def _allocate_row(self) -> int:
        row = self._meta_int("n_rows")
        if row >= self._keys.shape[0]:
            if self._keys.shape[0] < self.row_limit:
                self._grow(min(self._keys.shape[0] * 2, self.row_limit))
            elif self._depth > 1:
                raise KnowledgeStoreError("key storage is at its disk cap inside a "
                                          "transaction; run compact() between cycles")
            else:
                self._reclaim()
                row = self._meta_int("n_rows")
        self._meta_set("n_rows", row + 1)
        return row

    def _reclaim(self) -> None:
        """At the disk cap with no enclosing transaction: evict, then compact.
        Graduated items go first (the core already holds them), then the coldest active."""
        self._db.execute("COMMIT")          # close the write's own transaction first
        self._depth -= 1
        try:
            keyed = self._db.execute("SELECT COUNT(*) FROM items WHERE status IN "
                                     "('active', 'graduated')").fetchone()[0]
            n_evict = max(1, math.ceil(keyed * self.cfg.evict_fraction))
            ids = [r["id"] for r in self._db.execute(
                "SELECT id FROM items WHERE status IN ('active', 'graduated') "
                "ORDER BY status = 'graduated' DESC, usage ASC, last_used ASC LIMIT ?",
                (n_evict,))]
            self._db.executemany("UPDATE items SET status = 'evicted' WHERE id = ?",
                                 [(i,) for i in ids])
            log.warning("[KNOWLEDGE] disk cap reached: evicted %d coldest items", len(ids))
            self.compact()
        finally:
            self._db.execute("BEGIN IMMEDIATE")
            self._depth += 1
        if self._meta_int("n_rows") >= self._keys.shape[0]:
            raise KnowledgeStoreError("key storage full even after eviction")

    def _grow(self, rows: int) -> None:
        self._keys.flush()
        self._keys = None                   # unmap before resizing (required on Windows)
        with open(self.dir / self._meta("keys_file"), "r+b") as fh:
            fh.truncate(rows * self._row_bytes)
        self._keys = self._map_keys()

    def _write_row(self, row: int, vec: np.ndarray) -> None:
        self._keys[row] = vec.astype(np.complex64).view(np.float32).astype(np.float16)
        self._keys.flush()

    def _read_row(self, row: int) -> np.ndarray:
        return self._keys[row].astype(np.float32).view(np.complex64).copy()

    def _similarities(self, query: np.ndarray, rows: np.ndarray) -> np.ndarray:
        q = query.view(np.float32)
        sims = np.empty(len(rows), dtype=np.float32)
        order = np.argsort(rows)              # read in file order: sequential I/O
        chunk = self.cfg.retrieval_chunk
        for lo in range(0, len(order), chunk):
            sel = order[lo:lo + chunk]
            sims[sel] = self._keys[rows[sel]].astype(np.float32) @ q
        return sims / self.dim

    # --- internals: items & index ----------------------------------------------------
    def _insert_item(self, row: int, record: KnowledgeRecord, payload: str,
                     margin: float) -> int:
        now = time.time()
        cur = self._db.execute(
            "INSERT INTO items (key_row, kind, domain, signature, payload, complexity, status, "
            "successes, margin_mean, origin, provenance, created, last_used) "
            "VALUES (?, ?, ?, ?, ?, ?, 'active', 1, ?, ?, ?, ?, ?)",
            (row, record.kind, record.domain, record.signature, payload, record.complexity,
             margin, record.origin, record.provenance, now, now))
        return int(cur.lastrowid)

    def _find_duplicate(self, vec: np.ndarray,
                        record: KnowledgeRecord) -> Optional[Tuple[int, int]]:
        best: Optional[Tuple[int, int]] = None
        best_sim = self.cfg.dup_sim
        for r in self._db.execute(
                "SELECT id, key_row FROM items WHERE status = 'active' AND domain = ? "
                "AND signature = ?", (record.domain, record.signature)):
            sim = float(np.vdot(self._read_row(r["key_row"]), vec).real / self.dim)
            if sim > best_sim:
                best, best_sim = (r["id"], r["key_row"]), sim
        return best

    def _load_index(self) -> None:
        self._index = {
            r["id"]: (r["key_row"], r["domain"], r["successes"], r["failures"], r["complexity"])
            for r in self._db.execute(
                "SELECT id, key_row, domain, successes, failures, complexity FROM items "
                "WHERE status = 'active'")}
        self._arrays = None

    def _active_arrays(self) -> Tuple[np.ndarray, ...]:
        if self._arrays is None:
            entries = list(self._index.items())
            ids = np.array([i for i, _ in entries], dtype=np.int64)
            rows = np.array([e[0] for _, e in entries], dtype=np.int64)
            domains = np.array([e[1] for _, e in entries], dtype=object)
            conf = np.array([e[2] / (e[2] + e[3] + 1) for _, e in entries], dtype=np.float32)
            cx = np.array([e[4] for _, e in entries], dtype=np.float32)
            self._arrays = (ids, rows, domains, conf, cx)
        return self._arrays

    def _fetch(self, ids: List[int]) -> Dict[int, KnowledgeItem]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        out: Dict[int, KnowledgeItem] = {}
        for r in self._db.execute(f"SELECT * FROM items WHERE id IN ({marks})", ids):
            out[r["id"]] = KnowledgeItem(
                id=r["id"], kind=r["kind"], domain=r["domain"], signature=r["signature"],
                payload=json.loads(r["payload"]), complexity=r["complexity"],
                status=r["status"], successes=r["successes"], failures=r["failures"],
                usage=r["usage"] + self._pending_usage.get(r["id"], 0),
                margin_mean=r["margin_mean"], origin=r["origin"], provenance=r["provenance"],
                created=r["created"], last_used=r["last_used"])
        return out
