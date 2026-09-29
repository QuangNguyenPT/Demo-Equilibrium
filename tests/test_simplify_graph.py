"""Tests for the graph-view pruner.

The pruner decides what a person sees in the graph. If it silently keeps
docstring prose, keeps names this repository never defines, or drops the
module/class skeleton, the "simplified" view is either still noisy or useless,
so each rule is pinned here.

The provenance rule was added on 2026-09-29, after the first export wrote notes
named `Any.md`, `Any_1.md` ... `Any_24.md`, `ndarray.md`, `textwrap.md` and
`parametrize.md` into the Obsidian view: 207 code nodes carried no `source_file`
at all because they were imported names, and 43 more were `type: external`
stdlib stubs.

Attribution: opencode/big-pickle (big-pickle), 2026-09-27 for the prose / dunder /
test rules; opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29
for provenance resolution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from simplify_graph import (  # noqa: E402
    Definitions,
    is_dunder,
    is_external,
    is_prose,
    is_proven,
    is_test,
    keep_node,
    prune,
    project_definitions,
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

# What a repository of this shape defines. `Model` and `step` are here because
# `hrm/model.py` has them; `Any`, `ndarray` and `textwrap` are deliberately not.
DEF = Definitions(names={"Model", "step", "simpler", "grid", "device"},
                  stems={"model", "grid", "device"})


# ---------- shape rules ----------


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
    assert not keep_node(PROSE_NODE, {"module", "functions"}, True, DEF)


def test_dunder_never_survives():
    assert not keep_node(DUNDER_NODE, {"module", "functions"}, True, DEF)


def test_structure_survives_by_default():
    assert keep_node(MODULE_NODE, set(), False, DEF)
    assert keep_node(CLASS_NODE, set(), False, DEF)


def test_functions_excluded_by_default():
    assert not keep_node(FUNC_NODE, set(), False, DEF)


def test_functions_included_on_request():
    assert keep_node(FUNC_NODE, {"functions"}, False, DEF)


def test_tests_excluded_by_default_and_includable():
    assert not keep_node(TEST_FUNC, {"functions"}, False, DEF)
    assert keep_node(TEST_FUNC, {"functions"}, True, DEF)


# ---------- provenance ----------


def test_an_imported_name_with_no_source_file_is_dropped():
    """`Any`, `ndarray`, `fixture`: used as classes at the use site, defined
    nowhere in this repository."""
    for nid, label in (("i1", "Any"), ("i2", "ndarray"), ("i3", "fixture"),
                       ("i4", "parametrize"), ("i5", "no_grad")):
        imported = node(nid, label, source_file="", source_location="",
                        _callable_class=True)
        assert not is_proven(imported, DEF)
        assert not keep_node(imported, {"functions"}, False, DEF)


def test_a_real_symbol_the_extractor_failed_to_attribute_survives():
    """`Rule` in `hrm_x/rules.py` and `Solution` in `hrm/reasoning/reasoner.py`
    reach the graph with no `source_file`. They are still this repository's."""
    local = Definitions(names={"Rule"}, stems=set())
    orphan = node("o1", "Rule", source_file="", _callable_class=True)
    assert is_proven(orphan, local)
    assert keep_node(orphan, set(), False, local)


def test_an_external_module_needs_a_file_of_that_name_in_the_repository():
    """A name collision must not resurrect a stdlib module."""
    json_stub = node("e1", "json", source_file="", external=True)
    assert is_external(json_stub)
    assert not is_proven(json_stub, DEF)                    # no json.py here
    assert is_proven(json_stub, Definitions(names={"json"}, stems={"json"}))
    # and a same-named *function* is not enough
    assert not is_proven(json_stub, Definitions(names={"json"}, stems=set()))


def test_a_dotted_label_is_resolved_by_its_last_segment():
    dotted = node("d2", "torch.nn.functional", source_file="", external=True)
    assert is_proven(dotted, Definitions(names=set(), stems={"functional"}))


def test_definitions_come_from_this_repository_and_ignore_generated_trees(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "thing.py").write_text(
        "class Widget:\n    pass\n\n\ndef spin(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "graphify-out").mkdir()
    (tmp_path / "graphify-out" / "junk.py").write_text(
        "class Generated:\n    pass\n", encoding="utf-8")
    found = project_definitions(tmp_path)
    assert {"Widget", "spin", "thing"} <= found.names
    assert "thing" in found.stems
    assert "Generated" not in found.names, "generated trees are not project source"
    assert "junk" not in found.stems


# ---------- pruning ----------


def test_prune_drops_dangling_edges():
    graph = {
        "nodes": [MODULE_NODE, CLASS_NODE, PROSE_NODE],
        "links": [
            {"source": "m1", "target": "c1", "relation": "contains"},
            {"source": "m1", "target": "p1", "relation": "rationale_for"},
            {"source": "c1", "target": "ghost", "relation": "calls"},
        ],
    }
    out = prune(graph, set(), False, DEF)
    assert {n["id"] for n in out["nodes"]} == {"m1", "c1"}
    # the rationale edge dies with its prose node; the ghost edge dies too
    assert [l["source"] for l in out["links"]] == ["m1"]


def test_prune_preserves_graph_envelope_fields():
    graph = {"directed": True, "multigraph": False, "graph": {"name": "x"},
             "nodes": [MODULE_NODE], "links": [], "hyperedges": [{"a": 1}]}
    out = prune(graph, set(), False, DEF)
    assert out["directed"] is True
    assert out["graph"] == {"name": "x"}
    assert out["hyperedges"] == []


def test_summarize_reports_reduction():
    graph = {"nodes": [MODULE_NODE, CLASS_NODE, FUNC_NODE, PROSE_NODE], "links": []}
    out = prune(graph, set(), False, DEF)
    text = summarize(graph, out, DEF)
    assert "4 -> 2" in text
    assert "50.0% kept" in text


def test_empty_graph_is_handled():
    out = prune({"nodes": [], "links": []}, set(), False, DEF)
    assert out["nodes"] == [] and out["links"] == []


# ---------- the CLI ----------


def test_cli_writes_pruned_file(tmp_path):
    import simplify_graph

    src = tmp_path / "graph.json"
    src.write_text(json.dumps({"nodes": [MODULE_NODE, CLASS_NODE, PROSE_NODE],
                               "links": []}), encoding="utf-8")
    out = tmp_path / "sub" / "graph.simple.json"
    rc = simplify_graph.main(["--graph", str(src), "--out", str(out),
                              "--root", str(tmp_path)])
    assert rc == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert len(written["nodes"]) == 2


def test_cli_errors_when_graph_missing(tmp_path):
    import simplify_graph

    rc = simplify_graph.main(["--graph", str(tmp_path / "nope.json"),
                              "--out", str(tmp_path / "o.json")])
    assert rc == 1


# ---------- against the real graph ----------


def _real_graph():
    real = Path(__file__).resolve().parents[1] / "graphify-out" / "graph.json"
    if not real.exists():
        pytest.skip("graph.json not built yet")
    return json.loads(real.read_text(encoding="utf-8"))


def _real_definitions():
    return project_definitions(Path(__file__).resolve().parents[1])


def test_real_repo_graph_is_mostly_noise():
    """The whole justification for this script, asserted against the real graph."""
    graph = _real_graph()
    before = len(graph["nodes"])
    kept = prune(graph, set(), False, _real_definitions())["nodes"]
    assert len(kept) < before * 0.5, f"expected a big reduction, got {before} -> {len(kept)}"
    assert not any(is_prose(n) for n in kept)


def test_no_kept_label_contains_a_space():
    """Regression: the first pruner used `^[^()]+$`, so every docstring that
    documented a tensor signature -- "(B, thoughts + n, coda_dim) cross-attention
    memory ..." -- was kept and exported as a note named after a sentence
    fragment. A surviving node whose label has a space is always wrong, so
    assert the property directly instead of trusting is_prose()."""
    offenders = [n["label"] for n in prune(_real_graph(), set(), False,
                                           _real_definitions())["nodes"]
                 if " " in str(n.get("label") or "")]
    assert not offenders, f"{len(offenders)} prose labels survived, e.g. {offenders[:3]}"


def test_no_kept_node_is_an_import_the_repository_never_defines():
    """The regression that put `Any.md` ... `Any_24.md` in the Obsidian view."""
    defined = _real_definitions()
    kept = prune(_real_graph(), set(), False, defined)["nodes"]
    offenders = [n["label"] for n in kept if not is_proven(n, defined)]
    assert not offenders, (f"{len(offenders)} unprovenanced nodes survived, "
                           f"e.g. {offenders[:5]}")
    for junk in ("Any", "ndarray", "Textwrap", "textwrap", "fixture", "parametrize"):
        assert junk not in {n["label"] for n in kept}


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
