"""Tests for the graph-view pruner.

The pruner decides what a person sees in the graph. If it silently keeps
docstring prose or drops the module/class skeleton, the "simplified" view is
either still noisy or useless, so each rule is pinned here.

Attribution: opencode/big-pickle (big-pickle), 2026-09-27.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from simplify_graph import (  # noqa: E402
    is_dunder,
    is_prose,
    is_test,
    keep_node,
    prune,
    summarize,
)


def node(nid, label, **kw):
    base = {"id": nid, "label": label, "source_file": "hrm/model.py"}
    base.update(kw)
    return base


PROSE_NODE = node("p1", "The native language section.", source_location="L40")
DUNDER_NODE = node("d1", "dot-__truediv__()")
MODULE_NODE = node("m1", "hrm/model.py")
CLASS_NODE = node("c1", "Model", _callable_class=True, _callable=True)
FUNC_NODE = node("f1", "step", _callable=True)
TEST_FUNC = node("t1", "test_step", _callable=True, source_file="tests/test_model.py")


def test_prose_docstring_node_is_detected():
    assert is_prose(PROSE_NODE)


def test_symbols_are_not_mistaken_for_prose():
    for n in (MODULE_NODE, CLASS_NODE, FUNC_NODE):
        assert not is_prose(n)


def test_dunder_detected():
    assert is_dunder(DUNDER_NODE)
    assert not is_dunder(FUNC_NODE)


def test_test_detection():
    assert is_test(TEST_FUNC)
    assert not is_test(FUNC_NODE)


def test_prose_never_survives_even_if_functions_requested():
    assert not keep_node(PROSE_NODE, {"module", "functions"}, True)


def test_dunder_never_survives():
    assert not keep_node(DUNDER_NODE, {"module", "functions"}, True)


def test_structure_survives_by_default():
    assert keep_node(MODULE_NODE, set(), False)
    assert keep_node(CLASS_NODE, set(), False)


def test_functions_excluded_by_default():
    assert not keep_node(FUNC_NODE, set(), False)


def test_functions_included_on_request():
    assert keep_node(FUNC_NODE, {"functions"}, False)


def test_tests_excluded_by_default_and_includable():
    assert not keep_node(TEST_FUNC, {"functions"}, False)
    assert keep_node(TEST_FUNC, {"functions"}, True)


def test_prune_drops_dangling_edges():
    graph = {
        "nodes": [MODULE_NODE, CLASS_NODE, PROSE_NODE],
        "links": [
            {"source": "m1", "target": "c1", "relation": "contains"},
            {"source": "m1", "target": "p1", "relation": "rationale_for"},
            {"source": "c1", "target": "ghost", "relation": "calls"},
        ],
    }
    out = prune(graph, set(), False)
    assert {n["id"] for n in out["nodes"]} == {"m1", "c1"}
    # the rationale edge dies with its prose node; the ghost edge dies too
    assert [l["source"] for l in out["links"]] == ["m1"]


def test_prune_preserves_graph_envelope_fields():
    graph = {"directed": True, "multigraph": False, "graph": {"name": "x"},
             "nodes": [MODULE_NODE], "links": [], "hyperedges": [{"a": 1}]}
    out = prune(graph, set(), False)
    assert out["directed"] is True
    assert out["graph"] == {"name": "x"}
    assert out["hyperedges"] == []


def test_summarize_reports_reduction():
    graph = {"nodes": [MODULE_NODE, CLASS_NODE, FUNC_NODE, PROSE_NODE], "links": []}
    out = prune(graph, set(), False)
    text = summarize(graph, out)
    assert "4 -> 2" in text
    assert "50.0% kept" in text


def test_empty_graph_is_handled():
    out = prune({"nodes": [], "links": []}, set(), False)
    assert out["nodes"] == [] and out["links"] == []


def test_cli_writes_pruned_file(tmp_path, monkeypatch):
    import simplify_graph

    src = tmp_path / "graph.json"
    src.write_text(json.dumps({"nodes": [MODULE_NODE, CLASS_NODE, PROSE_NODE],
                               "links": []}), encoding="utf-8")
    out = tmp_path / "sub" / "graph.simple.json"
    rc = simplify_graph.main(["--graph", str(src), "--out", str(out)])
    assert rc == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert len(written["nodes"]) == 2


def test_cli_errors_when_graph_missing(tmp_path):
    import simplify_graph

    rc = simplify_graph.main(["--graph", str(tmp_path / "nope.json"),
                              "--out", str(tmp_path / "o.json")])
    assert rc == 1


def test_real_repo_graph_is_mostly_noise():
    """The whole justification for this script, asserted against the real graph."""
    real = Path(__file__).resolve().parents[1] / "graphify-out" / "graph.json"
    if not real.exists():
        pytest.skip("graph.json not built yet")
    graph = json.loads(real.read_text(encoding="utf-8"))
    before = len(graph["nodes"])
    kept = prune(graph, set(), False)["nodes"]
    after = len(kept)
    assert after < before * 0.5, f"expected a big reduction, got {before} -> {after}"
    # no docstring prose may reach the simplified view
    assert not any(is_prose(n) for n in kept)


def test_no_kept_label_contains_a_space():
    """Regression: the first pruner used `^[^()]+$`, so every docstring that
    documented a tensor signature -- "(B, thoughts + n, coda_dim) cross-attention
    memory ..." -- was kept and exported as a note named after a sentence
    fragment. A surviving node whose label has a space is always wrong, so
    assert the property directly instead of trusting is_prose()."""
    real = Path(__file__).resolve().parents[1] / "graphify-out" / "graph.json"
    if not real.exists():
        pytest.skip("graph.json not built yet")
    graph = json.loads(real.read_text(encoding="utf-8"))
    offenders = [n["label"] for n in prune(graph, set(), False)["nodes"]
                 if " " in str(n.get("label") or "")]
    assert not offenders, f"{len(offenders)} prose labels survived, e.g. {offenders[:3]}"


def test_signature_docstrings_are_recognised_as_prose():
    """The exact shape that defeated the first rule."""
    for label in [
        "(B, thoughts + n, coda_dim) cross-attention memory from detached segment",
        "(header, whole file, offset of the first tensor) after magic, version and",
        "(..., 2m) real - (..., m) complex, pairing consecutive reals..",
    ]:
        assert is_prose({"label": label})


def test_real_symbols_are_not_prose():
    for label in ["AlphaThing", "Model", "HRMEQB", "_Run", "hrm/model.py",
                  "encode_batch()", "Model.forward"]:
        assert not is_prose({"label": label})
