#!/usr/bin/env bash
set -euo pipefail

# --- Preflight ---------------------------------------------------------------
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1" >&2; exit 2; }; }
need python

# Required env var per permission model
if [[ -z "${ASSEMBLYMC_BIN:-}" ]]; then
  echo "ERROR: ASSEMBLYMC_BIN is not set." >&2
  echo "Set it to your locally built AssemblyMC.exe path (personal use only)." >&2
  exit 2
fi
if [[ ! -f "$ASSEMBLYMC_BIN" ]]; then
  echo "ERROR: ASSEMBLYMC_BIN points to a non-existent file: $ASSEMBLYMC_BIN" >&2
  exit 2
fi

# Check required helper scripts
for f in scripts/sample.py scripts/compute_ai.py scripts/aggregate.py scripts/fit_slope.py scripts/plots.py; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f (finish Batch 1)"; exit 2; }
done

# --- Params -----------------------------------------------------------------
N="${N:-1000}"
TRIALS="${TRIALS:-2000}"
TIMEOUT_S="${TIMEOUT_S:-2}"
RUN_ROOT="${RUN_ROOT:-runs}"
STAMP="$(date +%Y%m%dT%H%M%S)"
RUN_DIR="$RUN_ROOT/unguided_small_$STAMP"
mkdir -p "$RUN_DIR/plots"

# Minimal config snapshot (bin resolved by env in your adapter)
cat > "$RUN_DIR/run.yml" <<YAML
ai:
  method: assemblymc
  trials: ${TRIALS}
  timeout_s: ${TIMEOUT_S}
YAML

# --- Pipeline ---------------------------------------------------------------
echo ">>> Sampling ${N} molecules"
python scripts/sample.py --n "$N" --out "$RUN_DIR/samples.parquet" | tee "$RUN_DIR/sample.log"

echo ">>> Computing A* via AssemblyMC (trials=$TRIALS, timeout_s=$TIMEOUT_S)"
python scripts/compute_ai.py --in "$RUN_DIR/samples.parquet" --out "$RUN_DIR/ai.csv" \
  --method assemblymc --trials "$TRIALS" --timeout-s "$TIMEOUT_S" | tee "$RUN_DIR/ai.log"

echo ">>> Aggregating metrics"
python scripts/aggregate.py --in "$RUN_DIR/ai.csv" --out "$RUN_DIR/agg.csv" | tee "$RUN_DIR/agg.log"

echo ">>> Fitting slope (bootstrap=1000)"
python scripts/fit_slope.py --in "$RUN_DIR/agg.csv" --out "$RUN_DIR/slope.json" --bootstrap 1000 | tee "$RUN_DIR/slope.log"

echo ">>> Plotting"
python scripts/plots.py --in "$RUN_DIR/agg.csv" --outdir "$RUN_DIR/plots" | tee "$RUN_DIR/plots.log"

echo
echo "Run outputs:"
echo "  CSV:    $RUN_DIR/ai.csv"
echo "  Agg:    $RUN_DIR/agg.csv"
echo "  Slope:  $RUN_DIR/slope.json"
echo "  Plots:  $RUN_DIR/plots/"
