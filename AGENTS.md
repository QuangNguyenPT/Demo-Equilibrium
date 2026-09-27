# AGENTS.md — working agreements for this project

This repository is worked on by **multiple agents**, sequentially and in
parallel. These conventions exist so that a change is attributable and
reproducible regardless of which agent made it.

## 1. No personal credentials

Project-scoped secrets live in **`.env`** at the repo root (gitignored; see
`.env.example` for the template). Every agent reads the same file.

- Do **not** export a personal API key in your own shell and rely on it.
- Do **not** commit credentials, tokens, or machine-specific paths.
- `.\scripts\update_knowledge_graph.ps1` loads `.env` itself and degrades to
  `-Mode code-only` (no secret needed) when the key is absent. Routine updates
  should use the default path and require no credential at all.

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
- **Never write core weights outside the sleep gate** (or a human
  promote/rollback). `hrm/learning` writes knowledge; the reasoner never does.
- **Attribute honestly, including negative results.** If an experiment shows a
  component does not help, record that plainly. A flattering but wrong writeup
  is the most expensive failure mode in a multi-agent project, because the next
  agent will trust it.

## 4. Measure before claiming

- Run the full suite (`python -m pytest`) and report the real numbers.
- Prefer a measurement over an assertion. If an ablation is flat, say it is flat
  and give the cause.
- When fixing a bug, add a test that fails without the fix, and (where cheap)
  mutation-check that the test actually detects the reintroduction.
- A benchmark that cannot fail proves nothing. If a new instrument reports a
  perfect score, treat that as a prompt to make it harder, not as success.

## 5. Where things are documented

| Concern | Location |
|---|---|
| Architecture as built | `ADR-003 Unified Cognitive Model` |
| What the benchmark can and cannot show | `ADR-004 Discriminating Benchmark and Reasoning Ceiling` |
| Why the deep engine is not load-bearing | `05 - Engineering & Guides/Why v2 Did Not Prove the Model.md` |
| Session history / decision chain | `08 - Chat Context & Sessions/00 - Session Index.md` |
| Repo-level agent rules | this file |
