#!/usr/bin/env bash
# Push this submission to a fresh GitHub repo.
#
# Usage:
#   1) Create an empty public repo on github.com:
#        Suggested name: scikit-image-performance-optimization
#        Do NOT initialize with a README, .gitignore, or license — leave it empty.
#   2) Copy its clone URL (the SSH or HTTPS one).
#   3) Run:
#        ./push.sh git@github.com:YOUR_USER/YOUR_REPO.git
#      or
#        ./push.sh https://github.com/YOUR_USER/YOUR_REPO.git
#
# The script will:
#   * git init in this submission directory
#   * commit every file here on `main`
#   * add the remote and push.
# After the push completes, copy the printed commit SHA into Mercor's
# question 4 ("Commit SHA to fork at").

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <git remote URL>" >&2
  exit 2
fi
REMOTE="$1"

cd "$(dirname "$0")"

if [ -d .git ]; then
  echo "ERROR: this folder already has a .git directory. Refusing to re-init." >&2
  exit 1
fi

# Initial repo setup.
git init -q -b main

# Write .gitignore FIRST so __pycache__ etc. never get staged.
cat > .gitignore <<'GI'
outputs/
*.npz
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
GI

# Stage files explicitly (file-by-file) — never `git add .`, which could
# pull in editor backups or stray cache files.
git add \
  .gitignore \
  baseline_colab.ipynb \
  candidate_colab.ipynb \
  tests.ipynb \
  README.md \
  SUBMISSION_GUIDE.md \
  reward.json \
  push.sh \
  patches/_regionprops_fast.py \
  patches/apply_optimization.py \
  workloads/generate_workload.py \
  benchmark/benchmark_utils.py \
  docs/profiling_notes.md

git commit -q -m "Initial submission: scikit-image v0.18.3 regionprops_table vectorized fast path"

git remote add origin "$REMOTE"
echo
echo "About to push to: $REMOTE"
echo "Press Ctrl-C now if that's wrong; otherwise hit Enter."
read -r _

git push -u origin main

SHA="$(git rev-parse HEAD)"
echo
echo "=================================================================="
echo " Push complete."
echo " Commit SHA: $SHA"
echo " Use this SHA in Mercor's question 4 (\"Commit SHA to fork at\")."
echo "=================================================================="
