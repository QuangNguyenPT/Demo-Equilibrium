# Update knowledge graph after modifying the project.
# Run: .\scripts\update_knowledge_graph.ps1
# Called by convention after every modification session.
#
# This is a MULTI-AGENT project. Three rules follow from that:
#
#   1. NO CREDENTIAL IS NEEDED. The default LLM backend is `claude-cli`, which
#      drives the `claude` CLI already on PATH. There is no API key and nothing
#      for an agent to export in a private shell. Markdown extraction -- the
#      part that makes ADRs and sessions appear as graph nodes -- therefore works
#      out of the box for every agent.
#
#   2. If you ever do need a hosted backend, its key is PROJECT-scoped in
#      $ProjectRoot\.env (gitignored), loaded below. Never a personal key.
#
#   3. Attribution is recorded by the caller. Pass -Agent (and optionally
#      -Model) so the change is attributable without depending on who ran it.

param(
    [string]$Mode = "auto",            # auto | deep | code-only
    [string]$Backend = "claude-cli",   # claude-cli (no key) | ollama (no key) | kimi | claude | openai | deepseek
    [switch]$NoObsidian,               # skip the browsable vault view
    [string]$Agent = "",               # e.g. "opencode/big-pickle" -- who ran this
    [string]$Model = ""                # e.g. "big-pickle" -- which model acted
)

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$EnvFile     = Join-Path $ProjectRoot ".env"

# Deliberately NOT $ErrorActionPreference = "Stop". graphify writes progress to
# stderr, which PowerShell surfaces as NativeCommandError; with Stop that aborts
# the script mid-way, so a failed semantic pass (e.g. a CLI session limit) also
# silently skipped the Obsidian export below. Exit codes are checked explicitly.

# The Obsidian view is a curated code map (scripts/knowledge_map.py): one note
# per subsystem, linked to the specs, ADRs and sessions that govern or changed
# it. graphify's own `export obsidian` wrote one note per function/docstring --
# 2,084 notes at the vault root plus 3,211 in vault/_graph-view -- which Obsidian
# drew as two giant blobs burying the ~45 real notes. graph.json is still built
# for symbol-level `graphify query` / impact analysis; it is just not exported.

# Hosted-backend credentials, if the project ever needs one. Project-scoped, and
# optional: `claude-cli` and `code-only` need nothing.
if (Test-Path -LiteralPath $EnvFile) {
    Get-Content -LiteralPath $EnvFile -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $name, $value = $line.Split("=", 2)
            $name = $name.Trim()
            if ($name -and -not [Environment]::GetEnvironmentVariable($name)) {
                [Environment]::SetEnvironmentVariable($name, $value.Trim())
            }
        }
    }
    Write-Host "[GRAPHIFY] Loaded project-scoped config from .env (not a personal credential)." -ForegroundColor DarkGray
}

Write-Host "[GRAPHIFY] Updating knowledge graph for $ProjectRoot" -ForegroundColor Cyan
if ($Agent) {
    $who = if ($Model) { "$Agent ($Model)" } else { $Agent }
    Write-Host "[GRAPHIFY] Attribution: $who" -ForegroundColor DarkGray
}

# Resolve the backend before building flags. `claude-cli` is the default because
# it needs no secret; fall back to code-only (never to a personal key) if the
# CLI is missing, so a routine update still works on a bare machine.
$BackendKeyFor = @{
    "kimi" = "MOONSHOT_API_KEY"; "claude" = "ANTHROPIC_API_KEY"
    "openai" = "OPENAI_API_KEY"; "deepseek" = "DEEPSEEK_API_KEY"
}
$useBackend = $false
if ($Mode -ne "code-only") {
    if ($Backend -eq "claude-cli") {
        if (Get-Command claude -ErrorAction SilentlyContinue) {
            $useBackend = $true
        } else {
            Write-Host "[GRAPHIFY] 'claude' CLI not on PATH; no credential-free backend available." -ForegroundColor Yellow
        }
    } elseif ($Backend -eq "ollama") {
        $useBackend = [bool](Get-Command ollama -ErrorAction SilentlyContinue)
        if (-not $useBackend) { Write-Host "[GRAPHIFY] ollama not on PATH." -ForegroundColor Yellow }
    } else {
        $var = $BackendKeyFor[$Backend]
        if ($var -and [Environment]::GetEnvironmentVariable($var)) {
            $useBackend = $true
        } else {
            Write-Host "[GRAPHIFY] Backend '$Backend' needs $var in .env; ask the project owner." -ForegroundColor Yellow
        }
    }
}
if (-not $useBackend) {
    if ($Mode -ne "code-only") {
        Write-Host "[GRAPHIFY] Falling back to -Mode code-only (no LLM; vault notes stay out of the graph)." -ForegroundColor Yellow
    }
    $Mode = "code-only"
} else {
    Write-Host "[GRAPHIFY] Semantic backend: $Backend" -ForegroundColor DarkGray
}

$flags = @(".", "--update")
switch ($Mode) {
    "deep"      { $flags += @("--mode", "deep") }
    "code-only" { $flags += "--code-only" }
}
if ($useBackend) { $flags += @("--backend", $Backend) }

Push-Location $ProjectRoot
try {
    & graphify @flags
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[GRAPHIFY] Graph updated successfully." -ForegroundColor Green
    } else {
        Write-Host "[GRAPHIFY] graphify exited with code $LASTEXITCODE (graph kept as-is; export continues)." -ForegroundColor Yellow
    }

    if (-not $NoObsidian) {
        if (Get-Command python -ErrorAction SilentlyContinue) {
            & python (Join-Path $PSScriptRoot "knowledge_map.py") --root $ProjectRoot
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[GRAPHIFY] code map reported unassigned modules; add them to SUBSYSTEMS in scripts/knowledge_map.py" -ForegroundColor Yellow
            }
        } else {
            Write-Host "[GRAPHIFY] python not on PATH; skipped the code map." -ForegroundColor Yellow
        }
    }
} finally {
    Pop-Location
}
