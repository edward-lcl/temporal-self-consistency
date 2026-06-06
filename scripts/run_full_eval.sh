#!/usr/bin/env bash
# Run the full evaluation workflow end-to-end.
# Assumes prediction files are in ./predictions/ in the canonical format.
set -euo pipefail

PRED_DIR="${1:-./predictions}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== TSCT full evaluation ==="
echo "Predictions dir: $PRED_DIR"

cd "$REPO_ROOT/src/evaluation"

echo "[1/3] Running full analysis across all prediction files..."
TSCT_PREDICTIONS_DIR="$PRED_DIR" python full_analysis.py

echo "[2/3] Generating figures..."
python paper_plots.py

echo "[3/3] Generating results table..."
python generate_results_table.py

echo "Done. Figures in src/evaluation/figures/, tables as results_table.{md,tex,csv}"
