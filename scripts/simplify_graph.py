"""Prune graphify's raw graph down to a human-reviewable graph.

    python scripts/simplify_graph.py                 # write graphify-out/graph.simple.json
    python scripts/simplify_graph.py --keep functions --keep-tests
    python scripts/simplify_graph.py --html          # also render graph.html from the pruned graph

Why this exists
---------------
`graphify . --code-only` indexes every symbol it can reach, which is correct for
impact analysis and unusable as a picture. Re-measured on this repo (115 py files,
~17.2k lines) on 2026-09-28, the raw graph is 2,686 nodes over 7,194 links:

    982  functions                      36.6%
    653  docstring "rationale" prose    24.3%   <- a whole node per docstring
    422  test functions                15.7%
    301  modules / files                11.2%
    187  classes                         7.0%
     96  dunders (`dot-__truediv__`)     3.6%
     45  packages / externals            1.7%

(Exact counts, and therefore the percentages, drift every time the code does; the
proportions are the point, not the digits. Reproduce with
`python scripts/simplify_graph.py`, whose `kept by kind:` block reports the same
partition for whatever graph is on disk.)

`rationale_for` is 552 of the 7,194 edges, and the 653 prose nodes it points at are
a quarter of the picture: a sentence lifted out of a docstring ("The native language
section."). The structure a person actually wants -- which modules exist, how they
relate, and where the docs for them live -- is the 346 module/package and 187 class
nodes, which is exactly what the default view keeps (2,686 -> 533, 19.8%).

Pruning is therefore a *view* concern, not an extraction concern: the full graph
stays on disk untouched and remains the input to `affected` / `path` / `query`.
This script only writes a second, smaller file.


Resolving unprovenanced nodes
-----------------------------
A second, quieter leak survived the first round of pruning. 207 code nodes carried
no `source_file` and no `source_location` at all: `Any` (25), `ndarray` (30),
`Path` (14), `Tensor` (17), plus pytest's `fixture` (10) and `parametrize` (9).
graphify's AST pass records them as class-shaped nodes because they are used as
classes at the use site, but nothing in this repo defines them -- they are imported
names from the standard library, numpy, torch and pytest. Because they carry
`_callable_class`, `keep_node` returned True for all of them, and `graphify export
obsidian` wrote one note per name into `vault/_graph-view/`: `Any.md`, `Any_1.md` ...
`Any_24.md`, `ndarray.md`, `textwrap.md`, `parametrize.md`. 43 more nodes were
`type: external` module stubs (`os`, `re`, `torch`, `numpy`), which the same rule
kept on purpose.

The discriminator is provenance, not naming: a node survives if this repo defines
its label somewhere in its own source. `_DEFINITION` collects `class`/`def`/
`struct`/`enum`/`interface`/`function` names plus module stems from the project's
own code, and `is_proven()` accepts a node that carries a `source_file`, or one
whose label is in that set. Real project symbols that graphify failed to attribute
(`Rule` in `hrm_x/rules.py`, `Solution` in `hrm/reasoning/reasoner.py`) are kept by
the lookup; imported ones are dropped. External modules go with them.

Attribution: opencode/big-pickle (big-pickle), 2026-09-27 for the prose/dunder/
test pruning, 2026-09-28 for the symbol-shape rules; opencode/stealth/
space-bunny-alpha (space-bunny-alpha), 2026-09-29 for provenance resolution.
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

# Code suffixes scanned for project-owned definitions. `docs/`, `vault/` and the
# generated trees are excluded by directory name, so only real source counts.
CODE_SUFFIXES = (".py", ".hpp", ".cpp", ".h", ".cc", ".c", ".ps1")
SKIP_DIRS = {
    ".git", ".claude", ".obsidian", ".pytest_cache", "__pycache__",
    "graphify-out", "vault", "legacy", "models", "runs", "node_modules",
}
SKIP_DIR_PREFIXES = (".gfy-",)

# Every construct that introduces a name a person could navigate to. The keyword
# is REQUIRED: a looser rule that captured the first identifier on any line
# matched `from pathlib import Path`, `pytest.fixture` and every other mere
# mention, so `Path`, `fixture`, `dtype`, `op` and `payload` all entered the
# definitions set and 14 `Path` notes came straight back into the export.
_DEFINITION = re.compile(
    r"^[ \t]*(?:@[A-Za-z_][\w.]*[ \t]*(?:\([^\n]*\))?[ \t]*\r?\n[ \t]*)*"
    r"(?:async[ \t]+def|def|class|struct|enum|union|interface|function|func)\b"
    r"[ \t\*&<>:]*([A-Za-z_]\w*)",
    re.M,
)


class Definitions:
    """What this project defines, in the two senses the pruner needs.

    `names` is every class and function plus every module stem; `stems` is only
    the module stems. The split matters for external nodes: `json` and `utility`
    are stdlib modules, and they must not survive merely because some unrelated
    function is also called `utility` -- an external node is legitimate here only
    if this repo ships a file of that name.
    """

    def __init__(self, names: set[str], stems: set[str]) -> None:
        self.names = names
        self.stems = stems


def project_definitions(root: Path) -> Definitions:
    """Scan the project's own code for the names it defines.

    One pass over every code file: module stems first, then the definition regex.
    Cheap -- this repo is ~17k lines -- and it is what makes `is_proven` a fact
    about this repository rather than a guess about the name.
    """
    names: set[str] = set()
    stems: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
            continue
        if any(part in SKIP_DIRS or part.startswith(SKIP_DIR_PREFIXES)
               for part in path.parts):
            continue
        names.add(path.stem)
        stems.add(path.stem)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in _DEFINITION.finditer(text):
            names.add(match.group(1))
    return Definitions(names, stems)


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


def is_external(node: dict) -> bool:
    """A stdlib / third-party module stub, which this repo never defines."""
    return node.get("type") == "external" or bool(node.get("external"))


def is_proven(node: dict, defined: Definitions) -> bool:
    """Whether this project's own source can account for the node.

    A `source_file` is graphify's own attribution and is taken at face value. A
    code node with none is an imported name; it survives only if the label is
    defined somewhere in the repository, which is what recovers `Rule` and
    `Solution` and discards `Any`, `ndarray`, `fixture` and `parametrize`.
    Dotted labels (`torch.nn.functional`) are resolved by their last segment.
    An external module stub is held to the stricter test: the repository must
    actually ship a file of that name.
    """
    if is_external(node):
        label = str(node.get("label") or "").strip()
        return bool(label) and label.rsplit(".", 1)[-1] in defined.stems
    if node_source(node):
        return True
    label = str(node.get("label") or "").strip()
    return bool(label) and label.rsplit(".", 1)[-1] in defined.names


def keep_node(node: dict, keep: set[str], keep_tests: bool,
              defined: Definitions) -> bool:
    """Decide whether a node survives.

    The default view is *structure*: modules and classes, which is the skeleton a
    person can actually read. Functions, tests, dunders and docstring prose are
    opt-out noise. Docstring prose and dunders are never kept because they are not
    navigational anchors, no matter what was requested. Imported names are never
    kept because nothing in this repo defines them -- see `is_proven`.
    """
    if is_prose(node) or is_dunder(node):
        return False
    if not is_proven(node, defined):
        return False
    if is_test(node):
        return keep_tests or "tests" in keep
    if node.get("_callable_class"):
        return True
    if node.get("type") == "package":
        return True
    if node.get("_callable"):
        return "functions" in keep
    # a module/file: structural, always kept
    return True


def prune(graph: dict, keep: set[str], keep_tests: bool,
          defined: Definitions) -> dict:
    nodes = graph.get("nodes", [])
    links = graph.get("links", [])
    kept = [n for n in nodes if keep_node(n, keep, keep_tests, defined)]
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


def summarize(before: dict, after: dict, defined: Definitions) -> str:
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
        "imported" if not is_proven(n, defined) else
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
    ap.add_argument(
        "--root",
        default=".",
        help="project root scanned for the definitions that give a node "
             "provenance (default: the current directory).",
    )
    args = ap.parse_args(argv)

    keep = set(args.keep or [])
    src = Path(args.graph)
    if not src.exists():
        print(f"error: {src} not found; run the graph update first", file=sys.stderr)
        return 1
    graph = json.loads(src.read_text(encoding="utf-8"))
    root = Path(args.root).resolve()
    defined = project_definitions(root)
    after = prune(graph, keep, args.keep_tests, defined)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(after, ensure_ascii=False, indent=1), encoding="utf-8")
    print(summarize(graph, after, defined))
    print(f"\nwrote {out}")
    print(f"definitions resolved from {root}: {len(defined.names)} "
          f"({len(defined.stems)} module stems)")

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
