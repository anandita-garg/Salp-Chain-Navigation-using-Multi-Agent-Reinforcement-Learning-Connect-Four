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

# ── 0. Install Python 3.12 ───────────────────────────────────────────────────
info "Ensuring Python 3.12 is installed ..."

if ! command -v python3.12 &>/dev/null; then
    info "Installing Python 3.12 ..."
    apt update
    apt install -y software-properties-common

    # Add deadsnakes for newer Python
    add-apt-repository ppa:deadsnakes/ppa -y
    apt update

    apt install -y python3.12 python3.12-venv python3.12-distutils
    ok "Python 3.12 installed"
fi

PYTHON=python3.12

# ── 1. Create virtual environment ────────────────────────────────────────────
VENV_DIR="${SCRIPT_DIR}/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment ..."
    $PYTHON -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# ── 2. Upgrade pip ───────────────────────────────────────────────────────────
info "Upgrading pip ..."
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip

# ── 3. Install dependencies ──────────────────────────────────────────────────
info "Installing dependencies from requirements.txt ..."
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
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
