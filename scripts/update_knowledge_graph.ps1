# Update knowledge graph after modifying the project.
# Run: .\scripts\update_knowledge_graph.ps1
# Called by convention after every modification session.
#
# This is a MULTI-AGENT project. Two rules follow from that:
#
#   1. No personal credentials. Nothing here reads an agent's personal API key
#      or expects one to be exported in a private shell. Project-scoped secrets
#      live in $ProjectRoot\.env (gitignored) and are loaded here, so every agent
#      extracts against the same configuration. `code-only` needs no secret at all
#      and is the default path for routine updates.
#
#   2. Attribution is recorded by the caller. Pass -Agent (and optionally
#      -Model) so the change is attributable in the graph and the session log
#      without depending on who happened to run the script.

param(
    [string]$Mode = "auto",       # auto | deep | code-only
    [switch]$NoObsidian = $false,
    [string]$Agent = "",          # e.g. "opencode/big-pickle" -- who ran this
    [string]$Model = ""           # e.g. "big-pickle" -- which model acted
)

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$VaultDir    = Join-Path $ProjectRoot "vault"
$EnvFile     = Join-Path $ProjectRoot ".env"

# Project-scoped configuration, shared by every agent on this project.
# Absent file is fine: code-only mode needs no secret.
if (Test-Path -LiteralPath $EnvFile) {
    Get-Content -LiteralPath $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $name, $value = $line.Split("=", 2)
            if (-not [Environment]::GetEnvironmentVariable($name)) {
                [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim())
            }
        }
    }
    Write-Host "[GRAPHIFY] Loaded project config from .env (not a personal credential)." -ForegroundColor DarkGray
}

Write-Host "[GRAPHIFY] Updating knowledge graph for $ProjectRoot" -ForegroundColor Cyan
if ($Agent) {
    $who = if ($Model) { "$Agent ($Model)" } else { $Agent }
    Write-Host "[GRAPHIFY] Attribution: $who" -ForegroundColor DarkGray
}

# Deep mode needs the project-scoped key. Degrade loudly rather than fail, and
# never prompt for a personal one.
if ($Mode -eq "deep" -and -not $env:GEMINI_API_KEY) {
    Write-Host "[GRAPHIFY] No project-scoped GEMINI_API_KEY in .env; falling back to code-only." -ForegroundColor Yellow
    Write-Host "[GRAPHIFY] Ask the project owner to add it to $EnvFile -- do NOT use a personal key." -ForegroundColor Yellow
    $Mode = "code-only"
}

$flags = @(".", "--update")
if (-not $NoObsidian) {
    $flags += @("--obsidian", "--obsidian-dir", $VaultDir)
}
switch ($Mode) {
    "deep"      { $flags += "--mode"; $flags += "deep" }
    "code-only" { $flags += "--code-only" }
}

Push-Location $ProjectRoot
try {
    & graphify @flags
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[GRAPHIFY] Graph updated successfully." -ForegroundColor Green
    } else {
        Write-Host "[GRAPHIFY] graphify exited with code $LASTEXITCODE" -ForegroundColor Yellow
    }
} finally {
    Pop-Location
}
