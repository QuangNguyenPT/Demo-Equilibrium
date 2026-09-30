"""HRM-EQB files and runtime: .hrmc (F1), knowledge.pt (F2, F3), Mind (Q1 protocol, V1 writes), study (G1), CLI."""

import json
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hrm_eqb import hrmc  # noqa: E402
from hrm_eqb import knowledge as kn  # noqa: E402
from hrm_eqb.__main__ import main  # noqa: E402
from hrm_eqb.bench import generated, schedule_invariance  # noqa: E402
from hrm_eqb.config import PRESETS  # noqa: E402
from hrm_eqb.mind import CORE, KNOWLEDGE, STATE, Answer, Mind, Question, task_from_json  # noqa: E402
from hrm_eqb.model import HRMEQB  # noqa: E402
from hrm_eqb.study import LEDGER, study_once  # noqa: E402
from hrm_eqb.tasks import generate  # noqa: E402
from hrm_eqb.train import TrainConfig  # noqa: E402

TINY = PRESETS["tiny"]
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    torch.manual_seed(0)
    directory = tmp_path_factory.mktemp("eqb") / "model"
    Mind.create(directory, HRMEQB(TINY), {"generation": 0, "note": "test"})
    return directory


# ---------- .hrmc ----------

def test_hrmc_round_trip_is_exact(tmp_path):
    model = HRMEQB(TINY)
    hrmc.save(model, tmp_path / "m.hrmc")
    loaded, header = hrmc.load(tmp_path / "m.hrmc")
    for name, tensor in model.core_state().items():
        assert torch.equal(tensor, loaded.core_state()[name]), name
    assert header["certificate"]["gamma"] < 1.0
    assert (tmp_path / "m.hrmc").read_bytes()[:8] == hrmc.MAGIC


def _meta(path):
    header, _, _ = hrmc.read_header(path)
    return {k: v for k, v in header.items() if k not in ("format", "version", "tensors")}


def test_hrmc_refuses_corruption_and_forgery(tmp_path):
    model = HRMEQB(TINY)
    good = tmp_path / "good.hrmc"
    hrmc.save(model, good)
    raw = bytearray(good.read_bytes())
    raw[len(raw) // 2] ^= 1
    (tmp_path / "flip.hrmc").write_bytes(bytes(raw))
    with pytest.raises(hrmc.HRMCError, match="SHA-256"):
        hrmc.load(tmp_path / "flip.hrmc")

    state = model.core_state()
    forged = {k: v.clone() for k, v in state.items()}
    forged["readout.weight"] += 0.5                      # every hash recomputed, probes now disagree
    hrmc.write(tmp_path / "weights.hrmc", forged, _meta(good))
    with pytest.raises(hrmc.HRMCError, match="probe"):
        hrmc.load(tmp_path / "weights.hrmc")

    meta = _meta(good)
    meta["certificate"]["gamma"] -= 0.05                 # a flattering, false certificate
    hrmc.write(tmp_path / "claim.hrmc", state, meta)
    with pytest.raises(hrmc.HRMCError, match="disagrees"):
        hrmc.load(tmp_path / "claim.hrmc")

    missing = {k: v for k, v in state.items() if k != "halt.bias"}
    hrmc.write(tmp_path / "missing.hrmc", missing, _meta(good))
    with pytest.raises(hrmc.HRMCError, match="directory"):
        hrmc.load(tmp_path / "missing.hrmc")


# ---------- knowledge.pt ----------

def _knowledge(tmp_path):
    path = tmp_path / "knowledge.pt"
    kn.create(path, torch.randn(4, 2), torch.randn(4, 2), torch.zeros(16, 8), kn.schema_of(4, 4, 2))
    return path


def test_knowledge_rows_are_written_in_place_and_persist(tmp_path):
    path = _knowledge(tmp_path)
    kf = kn.KnowledgeFile(path)
    kf.write_rows(np.array([2, 7]), {"values": np.full((2, 8), 1.5), "usage": np.array([1.0, 2.0])})
    assert float(kf.tensor("values")[7, 3]) == 1.5              # the mapped view sees the write
    assert not kf.wal.exists()
    kf.close()
    data = torch.load(path, weights_only=True)                   # a normal .pt for any reader
    assert float(data["values"][2, 0]) == 1.5 and float(data["usage"][7]) == 2.0
    assert float(data["values"][3].abs().sum()) == 0.0


def test_knowledge_journal_makes_writes_atomic(tmp_path):
    """F3: after a crash the rows are all old (torn journal) or all new (complete journal)."""
    path = _knowledge(tmp_path)
    wal = path.with_name(path.name + ".wal")
    journal = kn._encode_journal(np.array([1, 5]), {"values": np.full((2, 8), 4.0, dtype=np.float32)})
    wal.write_bytes(journal)                                     # crash after journaling, before applying
    assert kn.recover(path) == "replayed"
    assert float(torch.load(path, weights_only=True)["values"][5, 0]) == 4.0
    wal.write_bytes(kn._encode_journal(np.array([0]), {"values": np.full((1, 8), 9.0, dtype=np.float32)})[:-3])
    assert kn.recover(path) == "discarded"                       # torn journal: nothing applied
    assert float(torch.load(path, weights_only=True)["values"][0, 0]) == 0.0
    assert not wal.exists()


def test_seal_restores_valid_zip_checksums(tmp_path):
    path = _knowledge(tmp_path)
    kf = kn.KnowledgeFile(path)
    kf.write_rows(np.array([3]), {"values": np.ones((1, 8))})
    kf.close()
    del kf
    assert zipfile.ZipFile(path).testzip() is not None           # in-place writes leave stale CRCs
    kn.seal(path)
    assert zipfile.ZipFile(path).testzip() is None
    assert float(torch.load(path, weights_only=True)["values"][3, 0]) == 1.0


# ---------- Mind ----------

def text_task(answer: bool = True):
    return task_from_json({"domain": "text", "train": [{"input": "cat", "output": "CAT"},
                                                       {"input": "stone", "output": "STONE"}],
                           "test": [{"input": "owl", **({"output": "OWL"} if answer else {})}]})


def test_mind_answers_or_asks_with_reasons(model_dir):
    mind = Mind(model_dir, CPU)
    try:
        for task in (text_task(), generate("grid", "flip", np.random.default_rng(0), 5)):
            result = mind.ask(task, n_views=3)
            assert isinstance(result, (Answer, Question))
            if isinstance(result, Question):
                assert result.reasons and result.candidates and result.bits >= 0.0
            else:
                assert result.certified_error <= TINY.solve_tol
        level = mind.hormones.level
        assert bool(((mind.hormones.lower <= level) & (level <= mind.hormones.upper)).all())
    finally:
        mind.close()
    state = json.loads((model_dir / STATE).read_text())
    assert state["episodes"] >= 2 and set(state["hormones"]) == {"dopamine", "serotonin", "noradrenaline",
                                                                 "acetylcholine"}


def test_unverified_write_never_reaches_disk(model_dir):
    """V1: an untrained model cannot re-derive the demos, so its write is refused and the file is unchanged."""
    before = (model_dir / KNOWLEDGE).read_bytes()
    mind = Mind(model_dir, CPU)
    try:
        report = mind.learn(text_task(False), np.frombuffer(b"OWL", dtype=np.uint8).astype(np.int64)[None, :],
                            steps=3)
        sudoku = generate("sudoku", "sudoku", np.random.default_rng(0))
        refused = mind.learn(sudoku, sudoku.answer, steps=1)
    finally:
        mind.close()
    assert not report.accepted and report.loo_after > 0
    assert not refused.accepted and "two demonstrations" in refused.reason
    assert (model_dir / KNOWLEDGE).read_bytes() == before


def test_committed_write_changes_exactly_its_rows(model_dir):
    mind = Mind(model_dir, CPU)
    try:
        values = np.array(mind.kfile.maps["values"])
        rows = np.array([1, 9])
        delta = np.full((2, values.shape[1]), 0.25, dtype=np.float32)
        mind._commit(rows, delta)
        after = np.array(mind.kfile.maps["values"])
    finally:
        mind.close()
    assert np.allclose(after[rows], values[rows] + delta)
    untouched = np.setdiff1d(np.arange(values.shape[0]), rows)
    assert np.array_equal(after[untouched], values[untouched])
    reread = torch.load(model_dir / KNOWLEDGE, weights_only=True)
    assert np.allclose(reread["values"].numpy()[rows], after[rows])


def test_schema_mismatch_is_refused(tmp_path):
    directory = tmp_path / "m"
    Mind.create(directory, HRMEQB(TINY))
    kn.create(directory / KNOWLEDGE, torch.randn(4, 2), torch.randn(4, 2), torch.zeros(16, 8), kn.schema_of(4, 4, 2))
    with pytest.raises(ValueError, match="schema"):
        Mind(directory, CPU)


# ---------- study and bench ----------

def test_study_records_a_ledger_row_and_locks_the_harness(model_dir, monkeypatch):
    import hrm_eqb.study as study
    monkeypatch.setattr(study, "TrainConfig", _small_train)
    result = study_once(model_dir, minutes=0.01, n_per_family=1, device=CPU, log=lambda s: None)
    rows = (model_dir / LEDGER).read_text().splitlines()
    assert rows[0].startswith("time\t") and len(rows) == 2
    assert rows[1].split("\t")[7] in ("keep", "discard")
    assert result.margin > 0
    state = json.loads((model_dir / STATE).read_text())
    state["harness_sha256"] = "tampered"
    (model_dir / STATE).write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="harness"):
        study_once(model_dir, minutes=0.01, n_per_family=1, device=CPU, log=lambda s: None)


def _small_train(**kw):
    """The study's TrainConfig with CPU-sized batches."""
    return TrainConfig(**{**kw, "batch_tokens": 512, "batch_pairs": 50_000})


def test_bench_suites_run_and_schedule_invariance_holds(model_dir):
    model, _ = hrmc.load(model_dir / CORE)
    result = generated(model, CPU, n_per_family=1)
    assert result.attempted == 23 and 0.0 <= result.accuracy <= 1.0
    assert schedule_invariance(model, CPU, n_per_family=1).accuracy == 1.0


# ---------- CLI ----------

def test_cli_create_info_ask_learn_seal(tmp_path, capsys):
    directory = str(tmp_path / "cli")
    assert main(["create", directory, "--preset", "tiny", "--steps", "1", "--cpu",
                 "--checkpoint", str(tmp_path / "checkpoint.pt")]) == 0
    capsys.readouterr()
    assert main(["info", directory, "--cpu"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["certificate"]["gamma"] < 1.0 and info["state"]["generation"] == 0
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"domain": "text", "train": [{"input": "cat", "output": "CAT"},
                                                             {"input": "dog", "output": "DOG"}],
                                "test": [{"input": "sun"}]}))
    assert main(["ask", directory, str(task), "--views", "2", "--cpu"]) == 0
    asked = json.loads(capsys.readouterr().out)
    assert asked["kind"] in ("answer", "question")
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps({"output": "SUN"}))
    assert main(["learn", directory, str(task), str(answer), "--steps", "1", "--cpu"]) == 0
    assert "accepted" in json.loads(capsys.readouterr().out)
    assert main(["seal", directory, "--cpu"]) == 0
    assert zipfile.ZipFile(tmp_path / "cli" / KNOWLEDGE).testzip() is None



def test_hormone_drives_move_in_the_documented_directions():
    """§6 table: DA follows reward error, 5-HT patience gain, NE agreement, ACh surprise."""
    from hrm_eqb.neuromodulation import drive
    level = torch.tensor([0.4, 0.5, 0.6, 0.3], dtype=torch.float64)
    base = drive(0.0, 0.0, 0.5, 2.0, 0.2, level)
    assert drive(0.8, 0.0, 0.5, 2.0, 0.2, level)[0] > base[0] > drive(-0.8, 0.0, 0.5, 2.0, 0.2, level)[0]
    assert drive(None, 0.0, 0.5, 2.0, 0.2, level)[0] == level[0]            # no reward observed: unchanged
    assert drive(0.0, 0.3, 0.5, 2.0, 0.2, level)[1] > base[1] > drive(0.0, -0.3, 0.5, 2.0, 0.2, level)[1]
    assert drive(0.0, 0.0, 0.0, 2.0, 0.2, level)[2] > base[2] > drive(0.0, 0.0, 2.0, 2.0, 0.2, level)[2]
    assert drive(0.0, 0.0, 0.5, 2.0, 0.9, level)[3] > base[3]
    assert bool(((base >= 0) & (base <= 1)).all())
