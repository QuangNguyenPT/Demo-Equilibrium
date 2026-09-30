"""The curated code map stays small, complete, link-clean, and never deletes hand-written notes."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import knowledge_map as km  # noqa: E402

LINK = re.compile(r"\[\[([^\]|#]+)")


def test_every_module_belongs_to_a_subsystem():
    """A new file that fits no subsystem must be noticed, not silently dropped from the map."""
    files = km.code_files(ROOT)
    assert files, "no code found"
    assert [f for f in files if km.assign(f) is None] == []


def test_assign_prefers_exact_then_longest_prefix():
    assert km.assign("hrm_x/mind.py") == "HRM-X Mind & Induction"
    assert km.assign("hrm/perception/grid/programs.py") == "HRM Perception"
    assert km.assign("hrm/model.py") == "HRM Model Facade & CLI"
    assert km.assign("tests/test_cli.py") is None


def test_imports_resolve_absolute_relative_and_submodule_from_package(tmp_path):
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text("")
    (tmp_path / "pkg" / "sub" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "sub" / "b.py").write_text(
        "import os\nfrom pkg import a\nfrom .. import a as again\nfrom pkg.sub import c\n")
    (tmp_path / "pkg" / "sub" / "c.py").write_text("")
    known = {"pkg/__init__.py", "pkg/a.py", "pkg/sub/__init__.py", "pkg/sub/b.py", "pkg/sub/c.py"}
    got = km.imports(tmp_path, "pkg/sub/b.py", known)
    assert {"pkg/a.py", "pkg/sub/c.py"} <= got
    assert "pkg/sub/b.py" not in got


def test_mentions_accept_full_paths_and_packages_never_bare_names():
    """Bare names are rejected: old notes say `train.py` for files that later moved to legacy/."""
    known = ["hrm_x/mind.py", "hrm_x/writer.py", "hrm/reasoning/equilibrium.py", "hrm/reasoning/qhead.py"]
    text = "Edited `hrm_x/mind.py` and the whole hrm/reasoning/ package; see writer.py and legacy/writer.py."
    got = km.mentions(text, known)
    assert got == {"hrm_x/mind.py", "hrm/reasoning/equilibrium.py", "hrm/reasoning/qhead.py"}


def test_frontmatter_commits_inline_and_list_forms():
    inline = "---\ntitle: x\ncommits: [483c0c1, 1553230]\n---\nbody"
    listed = "---\ncommits:\n  - 483c0c1\n  - not-a-hash\ntags: []\n---\n"
    assert km.frontmatter_commits(inline) == ["483c0c1", "1553230"]
    assert km.frontmatter_commits(listed) == ["483c0c1"]
    assert km.frontmatter_commits("no frontmatter commits: [483c0c1]") == []


def test_map_is_small_and_every_link_resolves():
    """The point of the map: tens of notes, and no dangling links (each would be a phantom graph node)."""
    notes = km.plan(km.build(ROOT, use_git=False))
    assert len(notes) <= 30
    existing = {p.stem for p in (ROOT / "vault").rglob("*.md")} | {n[:-3] for n in notes}
    dangling = {target.strip() for text in notes.values() for target in LINK.findall(text)} - existing
    assert dangling == set()


def test_write_removes_only_its_own_stale_notes(tmp_path):
    (tmp_path / "hand written.md").write_text("# mine\n", encoding="utf-8")
    (tmp_path / "Old Subsystem.md").write_text(f"---\n{km.MARKER}\n---\n", encoding="utf-8")
    written, removed = km.write(tmp_path, {"New.md": f"---\n{km.MARKER}\n---\n"})
    assert (written, removed) == (1, 1)
    assert (tmp_path / "hand written.md").exists()
    assert not (tmp_path / "Old Subsystem.md").exists()
