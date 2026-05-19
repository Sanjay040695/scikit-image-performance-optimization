# Submission guide — exact minimal manual steps

Everything in this folder is ready. You only need to do four things:

## Step 1. Create an empty public GitHub repo

Go to https://github.com/new and create a repo named (suggested)
`scikit-image-performance-optimization`. **Important**:

- Visibility: **Public**.
- Do NOT initialize with a README, .gitignore, or license. Leave it empty.

Copy the clone URL (HTTPS or SSH).

## Step 2. Push everything in this folder

Open a terminal in this folder (the one containing `push.sh`) and run:

```bash
chmod +x push.sh
./push.sh <YOUR_REPO_CLONE_URL>
```

Examples of `<YOUR_REPO_CLONE_URL>`:

- `git@github.com:harshityadav/scikit-image-performance-optimization.git`
- `https://github.com/harshityadav/scikit-image-performance-optimization.git`

The script will:

1. `git init` here
2. add all the right files (and exclude `outputs/`, `__pycache__`, etc. via `.gitignore`)
3. commit them on `main`
4. push to your remote

At the end it prints a **commit SHA**. Copy that — it's Mercor's question 4.

## Step 3. Run the three notebooks in Google Colab

For each of `baseline_colab.ipynb`, `candidate_colab.ipynb`, `tests.ipynb`:

1. On GitHub, open the notebook file.
2. Click "Open in Colab" (or paste the GitHub URL into <https://colab.research.google.com>).
3. **Runtime → Change runtime type → CPU**. High-RAM is preferred but
   not required.
4. In **cell 1.5 (the "Config" cell)**, edit `SUBMISSION_REPO_URL` to
   your GitHub repo URL — e.g.
   `https://github.com/harshityadav/scikit-image-performance-optimization.git`.
   (Or set the env var `SUBMISSION_REPO_URL` before running. Most
   evaluators won't bother with env vars, so editing the cell is fine.)
5. **Runtime → Run all**. Wait until it finishes.

Order matters:

1. `baseline_colab.ipynb` (produces `golden_output.npz`, `baseline_tests.json`)
2. `candidate_colab.ipynb` (applies patch, produces `candidate_output.npz`)
3. `tests.ipynb` — the source of truth. Writes `/content/reward.json`.

Total wall time: **~30–45 min** across all three.

## Step 4. Replace `reward.json` in your repo with the one tests.ipynb wrote

After `tests.ipynb` finishes:

1. In Colab, in the file browser on the left, find `/content/reward.json`.
   Right-click → Download.
2. Replace the placeholder `reward.json` in your local clone of the
   submission repo with this downloaded one.
3. Commit and push:
   ```bash
   git add reward.json
   git commit -m "Update reward.json with real Colab measurements"
   git push
   ```
4. Record the **new** commit SHA (the grader will fork at the latest
   SHA you give them).

## Step 5. Fill in Mercor's submission form

| Field | Value |
|-------|-------|
| Target repository | `scikit-image/scikit-image` |
| One-line description | `Vectorized regionprops_table fast path: scipy.ndimage.find_objects for bboxes, np.bincount for areas/centroids, scipy.ndimage.minimum/maximum for intensity extrema; replaces the per-region Python dispatch loop for common scalar properties.` |
| Submission GitHub URL | Your repo URL from Step 1 |
| Commit SHA | The SHA printed at the end of Step 4 |
| Hours spent | Your honest answer |

## Quick sanity checklist before submitting

- [ ] Repo is **public** and contains all three notebooks at top level.
- [ ] `reward.json` in the repo is the one Colab generated, NOT the
      placeholder (fields like `baseline_sha`, `median`, `speedup` are
      real numbers, not "TO_BE_FILLED_BY_TESTS_NOTEBOOK" / `null`).
- [ ] `correctness.existing_tests_pass == true` and
      `correctness.output_equivalent == true` in `reward.json`.
- [ ] `speedup` is not `null`.
- [ ] Submitted SHA points at the commit that contains the real
      `reward.json` (not the initial commit with the placeholder).

## If something goes wrong in Colab

| Symptom | Fix |
| --- | --- |
| "Module not found: generate_workload" | The Config cell didn't clone the repo. Verify `SUBMISSION_REPO_URL` is correct and re-run the Config cell. |
| `pip install -e .` fails on scikit-image | Colab sometimes has stale numpy. Restart runtime, re-run all cells. |
| `tests.ipynb` reports a regression | Open the report, look at which test failed. The patch falls back to the original implementation on any error, so this is unexpected — but if it happens, please share the failing test name. |
| Timings have very high IQR | Colab can be noisy. Re-run just the benchmark cell, or bump `n_measured` from 7 to 11 in `tests.ipynb`. |
| Out-of-memory | Switch to High-RAM runtime, or reduce `DEFAULT_N_DISKS` / image size in `workloads/generate_workload.py`. |
