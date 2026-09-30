"""Curated Obsidian code map: one note per subsystem, linked to the notes that govern and changed it.

    python scripts/knowledge_map.py            # (re)write vault/09 - Code Map/
    python scripts/knowledge_map.py --check    # print the plan, write nothing

graphify's graph.json stays the symbol-level index for impact queries. Exported
into the vault it produced one note per function and per docstring (5,219 notes
across two exports), which Obsidian drew as two giant blobs that buried the ~45
real notes. This view is deliberately coarse: subsystems, never functions.

Links come from two sources only, both deterministic:
  - a note that names a repo path (`hrm_eqb/mind.py`, `hrm/reasoning/`) governs it;
  - a session note's `commits:` frontmatter lists the commits it produced, and
    `git show` says which files each commit changed ("what changed per session").
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

OUT_DIR = Path("vault") / "09 - Code Map"
MARKER = "generated: knowledge_map"
CODE_ROOTS = ("hrm", "hrm_x", "experiments", "scripts")
CODE_SUFFIXES = (".py", ".ps1", ".cpp", ".h")
SKIP_NOTE_DIRS = ("07 - Templates", "09 - Code Map", "_graph-view")
INDEX_DIRS = ("00 - Meta & Indexes",)
SESSION_DIR = "08 - Chat Context & Sessions"
HISTORY = 8


@dataclass(frozen=True)
class Subsystem:
    name: str
    purpose: str
    paths: Tuple[str, ...]      # exact files, or prefixes ending in "/"


SUBSYSTEMS: Tuple[Subsystem, ...] = (
    # `hrm_eqb/` used to be seven subsystems here. It moved to `legacy/hrm_eqb/` on
    # 2026-09-29: an audit found zero files outside its own test suite importing it,
    # and `.graphifyignore` already excludes `legacy/` from the knowledge graph
    # ("must not appear in impact queries"). Documenting it in the *live* code map
    # while excluding it from the graph would be incoherent. Its module inventory
    # is now `legacy/README.md`; the case for re-taking its method is ADR-011.
    Subsystem("HRM-X Workspace & Rules",
              "E-graph with provenance: congruence closure, conflicts with blame, rule saturation, extraction.",
              ("hrm_x/workspace.py", "hrm_x/rules.py")),
    Subsystem("HRM-X Mind & Induction",
              "Operator hypotheses as programs, law discovery, the VOC controller, modulators, trace writer.",
              ("hrm_x/mind.py", "hrm_x/induction.py", "hrm_x/writer.py", "hrm_x/__init__.py")),
    Subsystem("HRM-X DSL Library",
              "Structural MDL primitive mining, exact abstraction/expansion, and the measured credit a "
              "library earns in search.",
              ("hrm_x/library.py",)),
    Subsystem("HRM-X Concept Induction (v0.5)",
              "Law-closed hypothesis families fitted exactly over Q, the determinacy test of Theorem 3, "
              "the open-world coverage rule and BOTTOM, corpus-MDL consolidation, phasor binding, "
              "and the group-theoretic grid and place-value layers.",
              ("hrm_x/qlinalg.py", "hrm_x/concepts.py", "hrm_x/consolidation.py",
               "hrm_x/situation.py", "hrm_x/algebra.py", "hrm_x/operators.py", "hrm_x/reader.py")),
    Subsystem("HRM-X Concept Construction (v0.6)",
              "New basis features built from primitives when the written language runs out, exact "
              "fitting of their linear closure, concept memory (candidate -> admitted -> retired), "
              "the language interface with a learned numeral reader, and the measured-but-unused "
              "FHRR signature / mushroom-body router.",
              ("hrm_x/construct.py", "hrm_x/concept_memory.py", "hrm_x/language.py",
               "hrm_x/signature.py", "hrm_x/mbi.py")),
    Subsystem("HRM-X Domains", "Arithmetic concepts, word-problem stories, grid object programs; their verifiers.",
              ("hrm_x/domains/",)),
    Subsystem("HRM-X Native Backend",
              "C++ e-graph and grid kernels behind a C ABI, parity-tested against the Python reference.",
              ("hrm_x/native/",)),
    Subsystem("HRM-X Experiments", "E1-E12 runners with baselines and ablations; results recorded as JSON.",
              ("experiments/",)),
    Subsystem("HRM Substrate (FHRR)", "Phasor algebra, name-derived codebook, backend validation gate.",
              ("hrm/substrate/",)),
    Subsystem("HRM Perception", "Domain protocol; grid, sequence and 3-D domains; the program language.",
              ("hrm/perception/",)),
    Subsystem("HRM Reasoning", "Fast path, H-equilibrium, synthesis search, halting and restarts.",
              ("hrm/reasoning/",)),
    Subsystem("HRM Verification", "Independent judge: exact replay, determinism, anti-memorization.",
              ("hrm/verify/",)),
    Subsystem("HRM Knowledge Store", "SSD store: memmap fp16 keys plus a sqlite journal.",
              ("hrm/knowledge/",)),
    Subsystem("HRM Language", "Language section backends, inner speech, text notation.", ("hrm/language/",)),
    Subsystem("HRM Metacognition", "Self-model (competence), curiosity, intent regression.",
              ("hrm/metacognition/",)),
    Subsystem("HRM Learning Loops", "Online learner, generators, frozen harness, sleep cycle, self-study.",
              ("hrm/learning/",)),
    Subsystem("HRM Model Facade & CLI", "Bundle directory, core weights, config, traces and the CLI.",
              ("hrm/model.py", "hrm/core.py", "hrm/config.py", "hrm/trace.py", "hrm/__main__.py",
               "hrm/__init__.py")),
    Subsystem("Tooling & Benchmarks", "Benchmark fetch and generation, graph and code-map tooling.",
              ("scripts/",)),
)

# Full repo paths only. Bare names ("train.py") were tried and dropped: old session
# notes name files that later moved to legacy/, so a bare name linked sessions to the
# wrong subsystem.
PATH_RE = re.compile(r"\b((?:hrm_eqb|hrm_x|hrm|experiments|scripts|tests)/[\w/]*\w(?:\.py|\.ps1|\.cpp|\.h)?)")


def assign(path: str) -> Optional[str]:
    """Subsystem owning a repo-relative posix path: exact file first, then the longest prefix."""
    for s in SUBSYSTEMS:
        if path in s.paths:
            return s.name
    best, depth = None, -1
    for s in SUBSYSTEMS:
        for p in s.paths:
            if p.endswith("/") and path.startswith(p) and len(p) > depth:
                best, depth = s.name, len(p)
    return best


def code_files(root: Path) -> List[str]:
    out = []
    for top in CODE_ROOTS:
        for p in (root / top).rglob("*"):
            if p.suffix in CODE_SUFFIXES and p.is_file() and "__pycache__" not in p.parts:
                out.append(p.relative_to(root).as_posix())
    return sorted(out)


def summary(root: Path, path: str) -> Tuple[str, int]:
    """(first docstring or leading-comment line, line count)."""
    text = (root / path).read_text(encoding="utf-8", errors="replace")
    lines = text.count("\n") + 1
    first = ""
    if path.endswith(".py"):
        try:
            doc = ast.get_docstring(ast.parse(text)) or ""
        except SyntaxError:
            doc = ""
        first = doc.strip().splitlines()[0] if doc.strip() else ""
    elif path.endswith((".cpp", ".h")):
        for line in text.splitlines():
            stripped = line.strip().lstrip("/*").strip()
            if line.lstrip().startswith(("//", "/*")) and stripped:
                first = stripped
                break
    else:
        for line in text.splitlines():
            if line.startswith("#") and line.strip("# ").strip():
                first = line.strip("# ").strip()
                break
    return first.replace("|", "\\|"), lines


def resolve(module: str, known: Set[str]) -> Optional[str]:
    base = module.replace(".", "/")
    for cand in (f"{base}.py", f"{base}/__init__.py"):
        if cand in known:
            return cand
    return None


def imports(root: Path, path: str, known: Set[str]) -> Set[str]:
    """Repo modules imported by a .py file (absolute and relative imports, submodule-from-package)."""
    if not path.endswith(".py"):
        return set()
    try:
        tree = ast.parse((root / path).read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()
    package = path.rsplit("/", 1)[0].replace("/", ".") if "/" in path else ""
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                parts = parts[: len(parts) - node.level + 1]
                mod = ".".join(parts + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            names = [mod] + [f"{mod}.{a.name}" for a in node.names]
        else:
            continue
        for name in names:
            hit = resolve(name, known)
            if hit and hit != path:
                found.add(hit)
    return found


def mentions(text: str, known: Iterable[str]) -> Set[str]:
    """Repo modules a note names by full path or package prefix (`hrm/reasoning/`)."""
    known = list(known)
    hits: Set[str] = set()
    for m in PATH_RE.findall(text):
        if m in known:
            hits.add(m)
        else:
            hits.update(k for k in known if k.startswith(m + "/"))
    return hits


def frontmatter_commits(text: str) -> List[str]:
    """`commits:` from YAML frontmatter, as an inline list or indented `- hash` items."""
    if not text.startswith("---"):
        return []
    end = text.find("\n---", 3)
    block = text[3:end] if end > 0 else ""
    out: List[str] = []
    lines = block.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("commits:"):
            rest = line.split(":", 1)[1].strip()
            if rest.startswith("["):
                out = [c.strip(" '\"") for c in rest.strip("[]").split(",") if c.strip()]
            else:
                for item in lines[i + 1:]:
                    if not item.startswith((" ", "-")):
                        break
                    if item.strip().startswith("-"):
                        out.append(item.strip()[1:].strip(" '\""))
            break
    return [c for c in out if re.fullmatch(r"[0-9a-f]{7,40}", c)]


def git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else ""


@dataclass
class Note:
    name: str
    folder: str
    text: str

    @property
    def is_session(self) -> bool:
        return self.folder.startswith(SESSION_DIR) and self.name.startswith("Session - ")

    @property
    def is_index(self) -> bool:
        return self.folder.startswith(INDEX_DIRS) or self.name.startswith("00 - ")


def vault_notes(root: Path) -> List[Note]:
    vault = root / "vault"
    out = []
    for p in sorted(vault.rglob("*.md")):
        rel = p.relative_to(vault)
        if rel.parts[0] in SKIP_NOTE_DIRS or len(rel.parts) == 1:
            continue
        out.append(Note(p.stem, rel.parts[0], p.read_text(encoding="utf-8", errors="replace")))
    return out


@dataclass
class Map:
    modules: Dict[str, List[str]]            # subsystem -> modules
    info: Dict[str, Tuple[str, int]]         # module -> (summary, lines)
    depends: Dict[str, Set[str]]             # subsystem -> subsystems it imports
    tests: Dict[str, Set[str]]               # subsystem -> test files
    documents: Dict[str, Set[str]]           # subsystem -> governing note names
    sessions: Dict[str, Dict[str, Set[str]]]  # subsystem -> session -> commits ("" = named in text)
    history: Dict[str, List[str]]            # subsystem -> "date hash subject"
    unassigned: List[str]


def build(root: Path, use_git: bool = True) -> Map:
    files = code_files(root)
    known = set(files)
    modules: Dict[str, List[str]] = defaultdict(list)
    unassigned = []
    for f in files:
        owner = assign(f)
        if owner:
            modules[owner].append(f)
        else:
            unassigned.append(f)
    info = {f: summary(root, f) for f in files}

    depends: Dict[str, Set[str]] = defaultdict(set)
    for f in files:
        src = assign(f)
        for dep in imports(root, f, known):
            dst = assign(dep)
            if src and dst and dst != src:
                depends[src].add(dst)

    tests: Dict[str, Set[str]] = defaultdict(set)
    for t in sorted((root / "tests").glob("test_*.py")):
        rel = t.relative_to(root).as_posix()
        for dep in imports(root, rel, known):
            if assign(dep):
                tests[assign(dep)].add(rel)

    documents: Dict[str, Set[str]] = defaultdict(set)
    sessions: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for note in vault_notes(root):
        named = {assign(m) for m in mentions(note.text, files)} - {None}
        if note.is_session:
            for s in named:
                sessions[s][note.name].add("")
            for commit in frontmatter_commits(note.text) if use_git else []:
                for changed in git(root, "show", "--name-only", "--format=", commit).split():
                    owner = assign(changed)
                    if owner:
                        sessions[owner][note.name].add(commit)
        elif not note.is_index:
            for s in named:
                documents[s].add(note.name)

    history: Dict[str, List[str]] = {}
    for s in SUBSYSTEMS:
        if use_git and modules.get(s.name):
            log = git(root, "log", f"-{HISTORY}", "--date=short", "--format=%ad %h %s", "--", *modules[s.name])
            history[s.name] = [line for line in log.splitlines() if line.strip()]
    return Map(dict(modules), info, depends, tests, documents, sessions, history, unassigned)


def _links(names: Iterable[str]) -> List[str]:
    return [f"- [[{n}]]" for n in sorted(names)]


def render_subsystem(s: Subsystem, m: Map) -> str:
    used_by = sorted(o for o, deps in m.depends.items() if s.name in deps)
    lines = ["---", "type: code-subsystem", MARKER, "tags:", "  - code-map", "---", "",
             f"# {s.name}", "", f"> {s.purpose}", "", "## Modules", "",
             "| module | what it does | lines |", "|---|---|---|"]
    for f in m.modules.get(s.name, []):
        text, count = m.info[f]
        lines.append(f"| `{f}` | {text} | {count} |")
    sections = (("Depends on", _links(m.depends.get(s.name, ()))),
                ("Used by", _links(used_by)),
                ("Governed by (specs, ADRs, concepts, guides)", _links(m.documents.get(s.name, ()))))
    for title, body in sections:
        lines += ["", f"## {title}", ""] + (body or ["- none"])
    lines += ["", "## Changed or discussed in sessions", ""]
    touched = m.sessions.get(s.name, {})
    if touched:
        for session in sorted(touched):
            commits = sorted(c for c in touched[session] if c)
            via = f" — commits {', '.join(f'`{c}`' for c in commits)}" if commits else " — named in the log"
            lines.append(f"- [[{session}]]{via}")
    else:
        lines.append("- none recorded")
    lines += ["", "## Tests", "", ", ".join(f"`{t}`" for t in sorted(m.tests.get(s.name, ()))) or "none"]
    lines += ["", "## Recent history", ""] + ([f"- {h}" for h in m.history.get(s.name, [])] or ["- none"])
    return "\n".join(lines) + "\n"


def render_index(m: Map) -> str:
    lines = ["---", "type: moc", MARKER, "tags:", "  - code-map", "  - moc", "---", "",
             "# Code Subsystems", "",
             "Generated by `scripts/knowledge_map.py` (run by `scripts/update_knowledge_graph.ps1`).",
             "One note per subsystem; functions are deliberately absent. For symbol-level impact",
             "queries use `graphify query` on `graphify-out/graph.json`. Hand-written companion: [[Code Map]].", ""]
    groups = (("`hrm_x/` — HRM-X v0.5, the current model (structural operator inference)", "HRM-X "),
              ("`hrm_eqb/` — HRM-EQB baseline (neural, certified)", "EQB "),
              ("`hrm/` — program-search baseline", "HRM "), ("Tooling", "Tooling"))
    for title, prefix in groups:
        lines += [f"## {title}", ""]
        lines += [f"- [[{s.name}]] — {s.purpose}" for s in SUBSYSTEMS if s.name.startswith(prefix)]
        lines.append("")
    lines += ["## Dependencies", "", "```mermaid", "flowchart LR"]
    ids = {s.name: f"S{i}" for i, s in enumerate(SUBSYSTEMS)}
    lines += [f'  {ids[s.name]}["{s.name}"]' for s in SUBSYSTEMS]
    for src in sorted(m.depends):
        lines += [f"  {ids[src]} --> {ids[dst]}" for dst in sorted(m.depends[src])]
    lines += ["```", "", "## What each session changed", "", "| session | subsystems |", "|---|---|"]
    per_session: Dict[str, Set[str]] = defaultdict(set)
    for sub, touched in m.sessions.items():
        for session in touched:
            per_session[session].add(sub)
    for session in sorted(per_session):
        lines.append(f"| [[{session}]] | {', '.join(sorted(per_session[session]))} |")
    return "\n".join(lines) + "\n"


def plan(m: Map) -> Dict[str, str]:
    """File name -> contents for every generated note."""
    notes = {f"{s.name}.md": render_subsystem(s, m) for s in SUBSYSTEMS}
    notes["00 - Code Subsystems.md"] = render_index(m)
    return notes


def write(out: Path, notes: Dict[str, str]) -> Tuple[int, int]:
    """Write notes; remove only stale files that carry the generated marker. Returns (written, removed)."""
    out.mkdir(parents=True, exist_ok=True)
    removed = 0
    for p in out.glob("*.md"):
        if p.name not in notes and MARKER in p.read_text(encoding="utf-8", errors="replace"):
            p.unlink()
            removed += 1
    for name, text in notes.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    return len(notes), removed


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    m = build(root)
    if m.unassigned:
        print(f"[CODEMAP] unassigned modules (add them to SUBSYSTEMS): {', '.join(m.unassigned)}", file=sys.stderr)
    notes = plan(m)
    edges = sum(len(v) for v in m.depends.values())
    print(f"[CODEMAP] {len(SUBSYSTEMS)} subsystems, {sum(len(v) for v in m.modules.values())} modules, "
          f"{edges} dependency edges")
    if args.check:
        return 1 if m.unassigned else 0
    written, removed = write(root / OUT_DIR, notes)
    print(f"[CODEMAP] wrote {written} notes to {OUT_DIR.as_posix()}/ (removed {removed} stale)")
    return 1 if m.unassigned else 0


if __name__ == "__main__":
    raise SystemExit(main())
