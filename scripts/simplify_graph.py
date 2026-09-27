"""Prune graphify's raw graph down to a human-reviewable graph.

    python scripts/simplify_graph.py                 # write graphify-out/graph.simple.json
    python scripts/simplify_graph.py --keep functions --keep-tests
    python scripts/simplify_graph.py --html          # also render graph.html from the pruned graph

Why this exists
---------------
`graphify . --code-only` indexes every symbol it can reach, which is correct for
impact analysis and unusable as a picture. Measured on this repo (88 py files,
~9.8k lines) the raw graph is 1,846 nodes:

    642  functions                      34.8%
    397  docstring "rationale" prose    21.5%   <- a whole node per docstring
    387  test functions                21.0%
    218  modules / files                11.8%
    140  classes                         7.6%
     66  dunders (`dot-__truediv__`)     3.6%

`rationale_for` is 398 of the 5,059 edges, so a fifth of the picture is a
sentence lifted out of a docstring ("The native language section."). The
structure a person actually wants -- which modules exist, how they relate, and
where the docs for them live -- is the 358 module/class nodes.

Pruning is therefore a *view* concern, not an extraction concern: the full graph
stays on disk untouched and remains the input to `affected` / `path` / `query`.
This script only writes a second, smaller file.

Attribution: opencode/big-pickle (big-pickle), 2026-09-27.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

DUNDER = re.compile(r"^dot-|__\w+__")

# A symbol is an identifier, a dotted name, an optional call, or a file path.
# Anything else that contains a space is a lifted docstring, not a navigable
# anchor. Note the earlier version of this rule required the label to *lack*
# parentheses, which missed every docstring that documents a tensor signature --
# "(B, thoughts + n, coda_dim) cross-attention memory ..." -- so 148 such notes
# survived the first export. Signatures are full of parens; the discriminator
# has to be "is this a symbol", not "does this avoid punctuation".
SYMBOLISH = re.compile(r"^[A-Za-z_][\w]*(?:\.[\w]+)*(?:\(\))?$")
PATHISH = re.compile(r"^[\w./\\-]+\.(?:py|md|json|toml|ps1|hpp|cpp|h|cc)$")


def node_source(node: dict) -> str:
    return str(node.get("source_file") or node.get("file") or "")


def is_test(node: dict) -> bool:
    src = node_source(node)
    return src.startswith("tests/") or "/tests/" in src or src.endswith("_test.py")


def is_prose(node: dict) -> bool:
    """A whole docstring promoted to a node, rather than a symbol."""
    label = str(node.get("label") or "").strip()
    if not label:
        return False
    if SYMBOLISH.match(label) or PATHISH.match(label):
        return False
    return " " in label


def is_dunder(node: dict) -> bool:
    return bool(DUNDER.search(str(node.get("label") or "")))


def keep_node(node: dict, keep: set[str], keep_tests: bool) -> bool:
    """Decide whether a node survives.

    The default view is *structure*: modules and classes, which is the 358-node
    skeleton a person can actually read. Functions, tests, dunders and docstring
    prose are opt-out noise. Docstring prose and dunders are never kept because
    they are not navigational anchors, no matter what was requested.
    """
    if is_prose(node) or is_dunder(node):
        return False
    if is_test(node):
        return keep_tests or "tests" in keep
    if node.get("_callable_class"):
        return True
    if node.get("type") in {"package", "external"}:
        return True
    if node.get("_callable"):
        return "functions" in keep
    # a module/file: structural, always kept
    return True


def prune(graph: dict, keep: set[str], keep_tests: bool) -> dict:
    nodes = graph.get("nodes", [])
    links = graph.get("links", [])
    kept = [n for n in nodes if keep_node(n, keep, keep_tests)]
    ids = {n["id"] for n in kept}
    out_links = [l for l in links if l.get("source") in ids and l.get("target") in ids]
    return {
        "directed": graph.get("directed", True),
        "multigraph": graph.get("multigraph", False),
        "graph": graph.get("graph", {}),
        "nodes": kept,
        "links": out_links,
        "hyperedges": [],
    }


def summarize(before: dict, after: dict) -> str:
    b, a = len(before.get("nodes", [])), len(after.get("nodes", []))
    bl, al = len(before.get("links", [])), len(after.get("links", []))
    lines = [
        f"nodes {b} -> {a}  ({100 * a / b:.1f}% kept)" if b else "nodes -> 0",
        f"links {bl} -> {al}",
        "",
        "kept by kind:",
    ]
    for label, count in Counter(
        "prose" if is_prose(n) else
        "dunder" if is_dunder(n) else
        "test" if is_test(n) else
        "class" if n.get("_callable_class") else
        "function" if n.get("_callable") else
        "module" for n in after["nodes"]
    ).most_common():
        lines.append(f"  {label:<10} {count}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graph", default="graphify-out/graph.json")
    ap.add_argument("--out", default="graphify-out/graph.simple.json")
    ap.add_argument(
        "--keep",
        action="append",
        choices=["functions", "tests"],
        default=None,
        help="extra symbol levels to retain (repeatable). "
             "Default: structure only (modules + classes).",
    )
    ap.add_argument("--keep-tests", action="store_true")
    ap.add_argument("--html", action="store_true", help="also render graph.html")
    args = ap.parse_args(argv)

    keep = set(args.keep or [])
    src = Path(args.graph)
    if not src.exists():
        print(f"error: {src} not found; run the graph update first", file=sys.stderr)
        return 1
    graph = json.loads(src.read_text(encoding="utf-8"))
    after = prune(graph, keep, args.keep_tests)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(after, ensure_ascii=False, indent=1), encoding="utf-8")
    print(summarize(graph, after))
    print(f"\nwrote {out}")

    if args.html:
        import subprocess

        r = subprocess.run(
            ["graphify", "export", "html", "--graph", str(out),
             "--output", str(out.with_suffix(".html"))],
            capture_output=True, text=True,
        )
        print(r.stdout.strip() or r.stderr.strip())
        return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
