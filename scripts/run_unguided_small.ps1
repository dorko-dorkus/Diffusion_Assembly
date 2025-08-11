#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

# --- Preflight ---------------------------------------------------------------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Error "Missing command: python"; exit 2
}

if (-not $env:ASSEMBLYMC_BIN) {
  Write-Error "ERROR: ASSEMBLYMC_BIN is not set."
  Write-Host "Set it to your locally built AssemblyMC.exe path (personal use only)." -ForegroundColor Yellow
  exit 2
}
if (-not (Test-Path $env:ASSEMBLYMC_BIN)) {
  Write-Error "ERROR: ASSEMBLYMC_BIN points to a non-existent file: $($env:ASSEMBLYMC_BIN)"
  exit 2
}

$required = @(
  "scripts/sample.py",
  "scripts/compute_ai.py",
  "scripts/aggregate.py",
  "scripts/fit_slope.py",
  "scripts/plots.py"
)
foreach ($f in $required) {
  if (-not (Test-Path $f)) {
    Write-Error "ERROR: missing $f (finish Batch 1)"
    exit 2
  }
}

# --- Params -----------------------------------------------------------------
if (-not $env:N) { $N = 1000 } else { $N = [int]$env:N }
if (-not $env:TRIALS) { $TRIALS = 2000 } else { $TRIALS = [int]$env:TRIALS }
if (-not $env:TIMEOUT_S) { $TIMEOUT_S = 2 } else { $TIMEOUT_S = [double]$env:TIMEOUT_S }
if (-not $env:RUN_ROOT) { $RUN_ROOT = "runs" } else { $RUN_ROOT = $env:RUN_ROOT }
$STAMP = Get-Date -Format "yyyyMMddTHHmmss"
$RUN_DIR = Join-Path $RUN_ROOT "unguided_small_$STAMP"
New-Item -ItemType Directory -Force -Path (Join-Path $RUN_DIR "plots") | Out-Null

@"
ai:
  method: assemblymc
  trials: ${TRIALS}
  timeout_s: ${TIMEOUT_S}
"@ | Set-Content (Join-Path $RUN_DIR "run.yml")

# --- Pipeline ---------------------------------------------------------------
Write-Host ">>> Sampling $N molecules"
python scripts/sample.py --n $N --out "$RUN_DIR/samples.csv" | Tee-Object "$RUN_DIR/sample.log"

Write-Host ">>> Computing A* via AssemblyMC (trials=$TRIALS, timeout_s=$TIMEOUT_S)"
python scripts/compute_ai.py --in "$RUN_DIR/samples.csv" --out "$RUN_DIR/ai.csv" --method assemblymc --trials $TRIALS --timeout-s $TIMEOUT_S | Tee-Object "$RUN_DIR/ai.log"

Write-Host ">>> Aggregating metrics"
python scripts/aggregate.py --in "$RUN_DIR/ai.csv" --out "$RUN_DIR/agg.csv" | Tee-Object "$RUN_DIR/agg.log"

Write-Host ">>> Fitting slope (bootstrap=1000)"
python scripts/fit_slope.py --in "$RUN_DIR/agg.csv" --out "$RUN_DIR/slope.json" --bootstrap 1000 | Tee-Object "$RUN_DIR/slope.log"

Write-Host ">>> Plotting"
python scripts/plots.py --in "$RUN_DIR/agg.csv" --outdir "$RUN_DIR/plots" | Tee-Object "$RUN_DIR/plots.log"

Write-Host
Write-Host "Run outputs:"
Write-Host "  CSV:    $RUN_DIR/ai.csv"
Write-Host "  Agg:    $RUN_DIR/agg.csv"
Write-Host "  Slope:  $RUN_DIR/slope.json"
Write-Host "  Plots:  $RUN_DIR/plots/"
