#!/usr/bin/env bash
# Update an EXISTING submission repo with the latest files in this folder
# (the v0.22.0-baseline rebuild), commit the changes, and push.
#
# Usage:
#   1) cd into the folder you cloned earlier (the one containing baseline_colab.ipynb
#      and a .git directory).
#   2) Replace the OLD files with the new ones from this v2 bundle (i.e. copy
#      everything from this folder OVER your existing files — same filenames).
#   3) Run:
#        chmod +x update_and_push.sh
#        ./update_and_push.sh
#   4) When done, the script prints the NEW commit SHA. That's what you'll
#      eventually submit on Mercor's form (after you run Colab and re-push
#      with the real reward.json).
#
# This script does NOT take a remote URL — it uses the `origin` remote that
# was set by the original `push.sh` you ran yesterday.

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .git ]; then
  echo "ERROR: no .git directory here." >&2
  echo "Either you're in the wrong folder, or this is a fresh checkout — in that" >&2
  echo "case use push.sh, not update_and_push.sh." >&2
  exit 1
fi

# Make sure files are fresh on disk before we stage.
git status --short

echo
echo "Staging updated files..."
git add \
  baseline_colab.ipynb \
  candidate_colab.ipynb \
  tests.ipynb \
  README.md \
  SUBMISSION_GUIDE.md \
  reward.json \
  push.sh \
  update_and_push.sh \
  patches/_regionprops_fast.py \
  patches/apply_optimization.py \
  workloads/generate_workload.py \
  benchmark/benchmark_utils.py \
  docs/profiling_notes.md

echo
echo "About to create commit + push to:"
git remote -v | head -1
echo "Press Ctrl-C now if that's wrong; otherwise hit Enter."
read -r _

git commit -m "v2 — switch baseline to scikit-image v0.22.0 (PyPI wheel, no compile)"
git push origin main

SHA="$(git rev-parse HEAD)"
echo
echo "=================================================================="
echo " Push complete."
echo " NEW commit SHA: $SHA"
echo
echo " Next steps:"
echo "   1. Open each notebook in Colab from the latest commit and Run all."
echo "   2. tests.ipynb produces /content/reward.json — download it."
echo "   3. Replace the local reward.json placeholder, commit + push AGAIN."
echo "   4. Submit that FINAL SHA on Mercor's form."
echo "=================================================================="
