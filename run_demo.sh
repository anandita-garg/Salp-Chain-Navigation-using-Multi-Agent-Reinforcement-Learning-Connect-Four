#!/usr/bin/env bash
# =============================================================================
#  run_demo.sh  —  Salp Chain MARL · Demo Runner
# =============================================================================
#  Place this file in the repository root alongside demo.py and run:
#
#      bash run_demo.sh
#
#  What it does:
#    1. Verifies Python 3.8+
#    2. Installs missing dependencies (torch, numpy, pandas, tqdm, matplotlib)
#    3. Runs demo.py — which imports FastEnv directly from each training script
#       and runs 1 episode per trained algorithm to show they work
# =============================================================================
set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
CYN='\033[0;36m'; BLD='\033[1m';    RST='\033[0m'

info() { echo -e "${CYN}[INFO]${RST}  $*"; }
ok()   { echo -e "${GRN}[ OK ]${RST}  $*"; }
die()  { echo -e "${RED}[FAIL]${RST}  $*" >&2; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_SCRIPT="${SCRIPT_DIR}/demo.py"

echo ""
echo -e "${BLD}${CYN}══════════════════════════════════════════════════════════${RST}"
echo -e "${BLD}${CYN}   Salp Chain Navigation · MARL Demo${RST}"
echo -e "${BLD}${CYN}══════════════════════════════════════════════════════════${RST}"
echo ""

# ── 1. Python check ──────────────────────────────────────────────────────────
info "Checking Python ..."
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        if "$cmd" -c "import sys; assert sys.version_info >= (3,8)" 2>/dev/null; then
            PYTHON="$cmd"
            ok "Found $cmd  ($($cmd --version 2>&1))"
            break
        fi
    fi
done
[[ -z "$PYTHON" ]] && die "Python 3.8+ not found. Install from https://python.org"

# ── 2. Dependencies ──────────────────────────────────────────────────────────
info "Installing dependencies from requirements.txt ..."
$PYTHON -m pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
ok "Dependencies installed"

# ── 3. Check demo.py is present ──────────────────────────────────────────────
[[ -f "$DEMO_SCRIPT" ]] || \
    die "demo.py not found in ${SCRIPT_DIR}. Place demo.py alongside run_demo.sh."

# ── 4. Run ───────────────────────────────────────────────────────────────────
echo ""
info "Running demo ..."
echo ""
$PYTHON "$DEMO_SCRIPT"

echo ""
ok "Done."
echo ""
