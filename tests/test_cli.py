"""Command line: create, ask, feedback with correction, explain, report, study, sleep."""
import json

from hrm.__main__ import main
from hrm.learning import Harness, SequenceTasks


class Offsets(SequenceTasks):
    families = ("offset",)


def test_cli_round_trip(tmp_path, capsys):
    bench = Harness.build(tmp_path / "bench.json", [Offsets()], per_family=1,
                          decoys_per_domain=1, seed=1)
    model = str(tmp_path / "m")
    assert main(["create", model, "--dim", "512", "--benchmark", str(bench)]) == 0
    assert main(["ask", model, "1 1 -> 2 2 ; 3 5 -> ?", "--task-id", "amb"]) == 0
    assert "[ANSWER]" in capsys.readouterr().out
    assert main(["feedback", model, "amb", "wrong", "--correction", "6 10"]) == 0
    assert "affine(a=2, b=0)" in capsys.readouterr().out
    assert main(["explain", model, "amb"]) == 0
    assert "[EXPLAIN]" in capsys.readouterr().out
    assert main(["report", model]) == 0
    assert "[COMPETENCE]" in capsys.readouterr().out
    assert main(["ask", model, "just words"]) == 2
    assert main(["feedback", model, "amb", "wrong", "--correction", "six ten"]) == 2
    assert main(["sleep", model]) == 0
    assert main(["study", model, "--tasks", "2", "--sleep-every", "5"]) == 0
    assert "tasks=2" in capsys.readouterr().out
    assert json.loads((tmp_path / "m" / "manifest.json").read_text())["active_generation"] >= 0
