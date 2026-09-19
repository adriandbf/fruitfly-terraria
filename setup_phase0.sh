#!/usr/bin/env bash
# Phase 0: create a Python env, fetch the Shiu et al. fly brain model, run a smoke test.
# Run from a Terminal on the Mac:  cd <this folder> && bash setup_phase0.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== checking tools"
command -v git >/dev/null || { echo "git missing"; exit 1; }
command -v python3 >/dev/null || { echo "python3 missing"; exit 1; }
python3 --version
if [ "$(uname)" = "Darwin" ] && ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode command line tools missing (needed for Brian2's fast C++ mode). Run: xcode-select --install"; exit 1
fi

echo "== python env"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install brian2 numpy pandas pyarrow joblib cython setuptools
pip freeze > brain/requirements.lock.txt

echo "== fly brain model (about 370 MB, includes the connectome data)"
[ -d brain/Drosophila_brain_model ] || git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model brain/Drosophila_brain_model

echo "== smoke test"
python brain/smoke_test.py 2>&1 | tee logs/phase0_smoke_test.log
