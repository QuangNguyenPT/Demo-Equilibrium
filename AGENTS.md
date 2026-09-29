# AGENTS.md — working agreements for this project

This repository is worked on by **multiple agents**, sequentially and in
parallel. These conventions exist so that a change is attributable and
reproducible regardless of which agent made it.

## 1. No personal credentials

**Routine updates need no credential at all.** The graph's LLM backend defaults to
`claude-cli`, which drives the `claude` CLI already on PATH. `.env` is empty for
normal work.

- Do **not** export a personal API key in your own shell and rely on it.
- Do **not** commit credentials, tokens, or machine-specific paths.
- `.\scripts\update_knowledge_graph.ps1` loads `.env` itself (see `.env.example`)
  and degrades to `-Mode code-only` when a requested backend has no key.
- `-Backend claude-cli` (the default) and `-Backend ollama` need no key. Only a
  deliberately hosted backend (`kimi`, `claude`, `openai`, `deepseek`) needs one,
  and then it is the project's key, in `.env`.

Rationale: if graph extraction or any other step depended on one agent's
personal key, the project's shared state would be a function of whose machine
ran it. That is a per-harness concern masquerading as project state.

## 2. Attribution

Record **who made a change** in the artifact itself, not in a chat transcript.

**Change notes** (`vault/06 - Devlogs & ADRs/`, `vault/08 - Chat Context & Sessions/`):
carry `agent:` and `model:` in the YAML frontmatter:

```yaml
---
title: "ADR-00N: ..."
type: adr
agent: "opencode/big-pickle"
model: "big-pickle"
date: 2026-09-27
---
```

**Code changes**: note the agent inline in the docstring when behaviour is
non-obvious, and list the changed files with their agent in the session log.
Do not stamp generated files, vendored code, or the frozen benchmark.

**Graph updates**: pass `-Agent` / `-Model` so the run is attributable:

```powershell
.\scripts\update_knowledge_graph.ps1 -Mode code-only -Agent "opencode/big-pickle" -Model "big-pickle"
```

**Tests are the attribution floor**: a change to behaviour without a test that
fails without it is not finished. This is how a reviewer knows *what* changed
independ of *who* changed it.

## 3. What agents must not do to shared state

- **Never edit a frozen benchmark or `hrm/learning/harness.py` casually.** The
  scorer and the benchmark file are hashed into a per-model lock; changing either
  breaks every model's harness lock by design. Make a new benchmark version and
  let a human re-sign.
- **Never overwrite `data/benchmark/benchmark-v1.json`.** It is the ADR-003
  baseline; v2 is a separate file so the baseline stays comparable.
- **Never point the Obsidian export at the vault root.** `graphify export obsidian`
  writes one note per graph node; aimed at `vault/` it produced 1,922 derived
  files at its root that buried the 47 real notes and then re-indexed themselves.
  It now targets `vault/_graph-view/`, which is inside the vault (so Obsidian
  renders it) but quarantined in one subfolder and listed in `.graphifyignore`,
  and it exports the **pruned** graph so note names are real symbols. Since
  2026-09-29 the pruner also resolves **provenance**: a node survives only if
  graphify attributed a `source_file` to it, or this repository's own source
  defines its label — which is what keeps imported names (`Any`, `ndarray`,
  `fixture`) and stdlib stubs out of the view. Measured on the real graph:
  3,236 → 356 nodes (11.0%), 651 → 445 notes. See ADR-010.
  `vault/00 - Meta & Indexes/Code Map.md` is the human-facing code↔doc map.
- **Never write core weights outside the sleep gate** (or a human
  promote/rollback). `hrm/learning` writes knowledge; the reasoner never does.
- **Attribute honestly, including negative results.** If an experiment shows a
  component does not help, record that plainly. A flattering but wrong writeup
  is the most expensive failure mode in a multi-agent project, because the next
  agent will trust it.

## 3a. Text files: never round-trip them through PowerShell

`Get-Content` + `Set-Content -Encoding utf8` in PowerShell 5.1 reads a BOM-less
UTF-8 file as Windows-1252 and writes the mojibake back with a BOM added. This
silently corrupted 5 files on 2026-09-27 (`â€”` for `—`, `ðŸ` for emoji). Use the
editor tools, which write UTF-8 without a BOM. If a file must be rewritten in
bulk, do it in Python and verify the byte size afterwards.

## 4. The knowledge graph is a view, not the source of truth

`graphify` extracts code locally (AST) and markdown through the `claude-cli`
backend, which needs no API key. If that backend is unavailable (no `claude` CLI,
or a session rate limit), the script degrades to `-Mode code-only` and the graph
holds code nodes but **zero** note nodes, so sessions, ADRs and concepts will not
appear in it. Do not present that as a bug in the drawing; the edges do not exist
yet. A re-run retries whatever failed, so fix the cause and re-run rather than
hand-editing the graph.

- `.\scripts\update_knowledge_graph.ps1` is the sanctioned update path. It runs
  the update, prunes the result, and refreshes the Obsidian view in one command.
- The raw graph is deliberately over-granular (every function, every docstring
  promoted to a node). `scripts/simplify_graph.py` prunes to the module/class
  skeleton and writes `graphify-out/graph.simple.json` + `graph.html`.
- Use the **full** graph for `query` / `path` / `affected`; pruning is only for
  display and for the Obsidian export.

## 5. Measure before claiming

- Run the full suite (`python -m pytest`) and report the real numbers.
- Prefer a measurement over an assertion. If an ablation is flat, say it is flat
  and give the cause.
- When fixing a bug, add a test that fails without the fix, and (where cheap)
  mutation-check that the test actually detects the reintroduction.
- A benchmark that cannot fail proves nothing. If a new instrument reports a
  perfect score, treat that as a prompt to make it harder, not as success.

## 6. Where things are documented

| Concern | Location |
|---|---|
| **Which code implements which document** | `vault/00 - Meta & Indexes/Code Map.md` |
| Architecture as built (normative) | `HRM-X Specification v0.5 - Structural Operator Inference` (ADR-010) |
| Baseline architectures (normative for their own package only) | `HRM-EQB Specification v0.2` (ADR-005/006), `ADR-003 Unified Cognitive Model` |
| Why program search was replaced | `ADR-005 Neural HRM-Phi Replaces Program Search` |
| What the benchmark can and cannot show | `ADR-004 Discriminating Benchmark and Reasoning Ceiling` |
| Why v2 did not prove the model | `05 - Engineering & Guides/Why v2 Did Not Prove the Model.md` |
| Session history / decision chain | `08 - Chat Context & Sessions/00 - Session Index.md` |
| Repo-level agent rules | this file |

