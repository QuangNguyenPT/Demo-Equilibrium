"""SSD knowledge store: persistence, write gate, copy-on-write, rollback, crash safety."""
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from hrm.config import KnowledgeConfig, SubstrateConfig
from hrm.knowledge import (KnowledgeRecord, KnowledgeStore, KnowledgeStoreError, Rollback)
from hrm.substrate import PhasorVector

DIM = 256
SUB = SubstrateConfig(dim=DIM, seed=11)
ROW_BYTES = 2 * DIM * 2
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Report:
    passed: bool = True
    margin: float = 1.0


def cfg(**kw) -> KnowledgeConfig:
    base = dict(initial_rows=4, retrieval_chunk=16)
    base.update(kw)
    return KnowledgeConfig(**base)


def key(seed: int) -> PhasorVector:
    return PhasorVector.random(DIM, np.random.default_rng(seed))


def record(sig: str, domain: str = "grid", **kw) -> KnowledgeRecord:
    base = dict(kind="program", domain=domain, signature=sig, payload={"sig": sig},
                complexity=1, origin="t", provenance="synthesized")
    base.update(kw)
    return KnowledgeRecord(**base)


@pytest.fixture
def store(tmp_path):
    with KnowledgeStore(tmp_path / "k", SUB, cfg()) as s:
        yield s


def test_round_trip_survives_reopen_and_growth(tmp_path):
    with KnowledgeStore(tmp_path / "k", SUB, cfg()) as s:
        ids = [s.write(key(i), record(f"r{i}", payload={"i": i}), Report()) for i in range(50)]
        assert s.stats()["capacity"] >= 50            # grew past initial_rows=4
    with KnowledgeStore(tmp_path / "k", SUB, cfg()) as s:
        assert len(s) == 50
        for i in (0, 17, 49):
            top = s.retrieve(key(i), k=1)[0]
            assert top.item.id == ids[i] and top.item.payload == {"i": i}
            assert top.similarity > 0.99
        assert s.integrity_problems() == []


def test_keys_live_in_a_memmapped_file(store):
    store.write(key(1), record("a"), Report())
    assert isinstance(store._keys, np.memmap)
    key_file = next(store.dir.glob("keys.*.f16"))
    assert key_file.stat().st_size == store.stats()["capacity"] * ROW_BYTES


def test_unverified_write_refused(store):
    assert store.write(key(1), record("guess"), Report(passed=False)) is None
    assert len(store) == 0 and store.stats()["rows"] == 0


def test_non_unit_key_rejected(store):
    bad = PhasorVector(key(1).data * np.complex64(2.0))
    with pytest.raises(ValueError, match="unit phasors"):
        store.write(bad, record("a"), Report())


@pytest.mark.parametrize("other", [SubstrateConfig(dim=DIM, seed=12),
                                   SubstrateConfig(dim=2 * DIM, seed=11)])
def test_substrate_mismatch_refused(tmp_path, other):
    KnowledgeStore(tmp_path / "k", SUB, cfg()).close()
    with pytest.raises(KnowledgeStoreError, match="substrate mismatch"):
        KnowledgeStore(tmp_path / "k", other, cfg())


def test_same_signature_near_key_merges_copy_on_write(store):
    k = key(1)
    first = store.write(k, record("shift"), Report())
    old_key = store.key_of(first).data.copy()
    near = k.noisy(0.05, np.random.default_rng(0))
    assert store.write(near, record("shift"), Report()) == first
    item = store.get(first)
    assert item.successes == 2 and len(store) == 1
    assert store.stats()["dead_rows"] == 1                 # old row retired, not overwritten
    assert np.allclose(store._read_row(0), old_key, atol=2e-3)


def test_same_signature_other_context_is_a_new_item(store):
    a = store.write(key(1), record("shift"), Report())
    b = store.write(key(2), record("shift"), Report())
    assert a != b and len(store) == 2


def test_rollback_discards_writes_and_reuses_rows(store):
    store.write(key(0), record("keep"), Report())
    with store.transaction():
        for i in range(1, 4):
            store.write(key(i), record(f"tentative{i}"), Report())
        raise Rollback
    assert len(store) == 1 and store.stats()["rows"] == 1
    assert all(r.item.signature == "keep" for r in store.retrieve(key(2)))
    store.write(key(9), record("after"), Report())
    assert store.stats()["rows"] == 2 and store.integrity_problems() == []


def test_nested_savepoint_rolls_back_only_inner(store):
    with store.transaction():
        store.write(key(1), record("outer"), Report())
        with store.transaction():
            store.write(key(2), record("inner"), Report())
            raise Rollback
    assert [i.signature for i in store.items()] == ["outer"]


def test_error_in_transaction_rolls_back_and_propagates(store):
    with pytest.raises(RuntimeError, match="boom"):
        with store.transaction():
            store.write(key(1), record("x"), Report())
            raise RuntimeError("boom")
    assert len(store) == 0


def test_quarantine_excluded_until_reactivated(store):
    item = store.write(key(1), record("wrong"), Report())
    store.set_status(item, "quarantined")
    assert store.retrieve(key(1)) == []
    store.set_status(item, "active")
    assert store.retrieve(key(1), k=1)[0].item.id == item


def test_outcomes_drive_confidence_and_ranking(store):
    q = key(1)
    a = store.write(q, record("a"), Report())
    b = store.write(q, record("b"), Report())
    store.report_outcome(a, False, 0.2, "t1")
    store.report_outcome(a, False, 0.1, "t2", source="human")
    store.report_outcome(b, True, 0.9, "t3")
    assert store.get(a).confidence < store.get(b).confidence
    assert store.retrieve(q, k=1)[0].item.id == b
    with pytest.raises(ValueError):
        store.report_outcome(a, True, 1.0, "t4", source="oracle")


def test_domain_filter(store):
    store.write(key(1), record("g", domain="grid"), Report())
    s = store.write(key(1), record("s", domain="sequence"), Report())
    assert [r.item.id for r in store.retrieve(key(1), domain="sequence")] == [s]


def test_usage_counters_persist(tmp_path):
    with KnowledgeStore(tmp_path / "k", SUB, cfg()) as s:
        item = s.write(key(1), record("a"), Report())
        for _ in range(3):
            s.retrieve(key(1))
    with KnowledgeStore(tmp_path / "k", SUB, cfg()) as s:
        assert s.get(item).usage == 3


def test_compaction_keeps_every_mapping(store):
    ids = [store.write(key(i), record(f"r{i}"), Report()) for i in range(20)]
    for i in ids[::2]:
        store.set_status(i, "evicted")
    old_files = set(store.dir.glob("keys.*.f16"))
    assert store.compact() == 10
    for i, item_id in enumerate(ids):
        if i % 2:
            assert store.retrieve(key(i), k=1)[0].item.id == item_id
        else:
            with pytest.raises(KeyError):
                store.key_of(item_id)
    assert store.integrity_problems() == []
    assert set(store.dir.glob("keys.*.f16")).isdisjoint(old_files)


def test_graduated_keys_survive_compaction_and_rollback_reactivates(store):
    item = store.write(key(1), record("kept"), Report())
    store.set_status(item, "graduated", generation=3)
    store.compact()
    assert store.retrieve(key(1)) == []                  # graduated: not retrieved
    assert store.reactivate_graduated(after_generation=3) == ([], [])
    back, lost = store.reactivate_graduated(after_generation=2)
    assert back == [item] and lost == []
    assert store.retrieve(key(1), k=1)[0].item.id == item


def test_rollback_reports_graduations_whose_key_was_evicted(store):
    item = store.write(key(1), record("a"), Report())
    store.set_status(item, "evicted", generation=5)
    store.compact()                                      # key dropped for good
    store._db.execute("UPDATE items SET status = 'graduated' WHERE id = ?", (item,))
    assert store.reactivate_graduated(after_generation=4) == ([], [item])


def test_compaction_refused_inside_transaction(store):
    with pytest.raises(KnowledgeStoreError):
        with store.transaction():
            store.compact()


def test_disk_cap_evicts_coldest(tmp_path):
    limit_gb = 8 * ROW_BYTES / 2 ** 30                  # room for exactly 8 keys
    with KnowledgeStore(tmp_path / "k", SUB, cfg(max_disk_gb=limit_gb,
                                                 evict_fraction=0.25)) as s:
        ids = [s.write(key(i), record(f"r{i}"), Report()) for i in range(8)]
        for i in range(4, 8):                           # make the last four hot
            for _ in range(3):
                s.retrieve(key(i), k=1)
        new = s.write(key(100), record("new"), Report())
        active = {it.id for it in s.items()}
        assert new in active and set(ids[4:]) <= active
        assert len(active) <= 8 and s.stats()["capacity"] <= 8
        assert s.integrity_problems() == []


def test_disk_cap_inside_outer_transaction_raises(tmp_path):
    limit_gb = 2 * ROW_BYTES / 2 ** 30
    with KnowledgeStore(tmp_path / "k", SUB, cfg(max_disk_gb=limit_gb)) as s:
        with pytest.raises(KnowledgeStoreError, match="disk cap"):
            with s.transaction():
                for i in range(3):
                    s.write(key(i), record(f"r{i}"), Report())
        assert len(s) == 0


def test_integrity_check_detects_corrupted_key(store):
    item = store.write(key(1), record("a"), Report())
    store._keys[0] = np.float16(3.0)                     # bypass the API: corrupt the row
    assert any("unit phasor" in p for p in store.integrity_problems())
    assert item is not None


def test_crash_between_key_write_and_commit(tmp_path):
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(ROOT)!r})
        import numpy as np
        from hrm.config import KnowledgeConfig, SubstrateConfig
        from hrm.knowledge import KnowledgeRecord, KnowledgeStore
        from hrm.substrate import PhasorVector

        class R:
            passed, margin = True, 1.0

        def rec(sig):
            return KnowledgeRecord("program", "grid", sig, {{}}, 1, "t", "synthesized")

        def key(seed):
            return PhasorVector.random({DIM}, np.random.default_rng(seed))

        s = KnowledgeStore({str(tmp_path / 'k')!r}, SubstrateConfig(dim={DIM}, seed=11),
                           KnowledgeConfig(initial_rows=4))
        s.write(key(0), rec("a"), R())
        s.write(key(1), rec("b"), R())
        KnowledgeStore._insert_item = lambda *a, **k: os._exit(3)   # die after the key row
        s.write(key(2), rec("c"), R())
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 3, proc.stderr
    with KnowledgeStore(tmp_path / "k", SUB, cfg()) as s:
        assert len(s) == 2 and s.stats()["rows"] == 2
        assert s.integrity_problems() == []
        assert s.retrieve(key(1), k=1)[0].item.signature == "b"
        s.write(key(3), record("d"), Report())
        assert s.stats()["rows"] == 3
