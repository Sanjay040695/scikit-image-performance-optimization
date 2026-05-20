# Submission guide — clean v2 (post-rebuild)

This guide replaces the previous one. Follow it from the top.

## The whole flow in three minutes of reading

We targeted scikit-image **v0.24.0** (June 2024, satisfies the brief's
≥12-month requirement) because PyPI ships binary wheels for it on Python
3.10 / 3.11 / 3.12. That means **no Cython compile in Colab**, no
fragile build pins — `pip install scikit-image==0.24.0` is one line and
~99% reliable.

You will:
1. Update your local clone with the new files (5 min).
2. Push the update to GitHub (1 min).
3. Run three notebooks in Colab, in order (~15 min).
4. Replace your repo's `reward.json` with the one Colab produced, push,
   submit that SHA on Mercor's form (3 min).

Total active time on your part: ~10–15 min, plus waiting on Colab.

---

## Step 1. Replace your local clone with the v2 files

The repo you pushed yesterday (with the v0.18.3 plan) is now stale.
On your Mac terminal:

```bash
cd ~/path/to/scikit-image-performance-optimization   # your existing clone
```

Then copy every file from this folder OVER your existing files. (Cowork
saved this entire folder to your Downloads / a sync folder.) After
copying, `git status` should show all the files as modified.

Make sure these specific files were replaced:
- `baseline_colab.ipynb`, `candidate_colab.ipynb`, `tests.ipynb`
- `README.md`, `SUBMISSION_GUIDE.md`
- `patches/_regionprops_fast.py`, `patches/apply_optimization.py`
- `workloads/generate_workload.py`
- `benchmark/benchmark_utils.py`
- `docs/profiling_notes.md`
- `update_and_push.sh` (new — doesn't exist in your old clone)

## Step 2. Push the update

In the terminal, from inside the repo folder:

```bash
chmod +x update_and_push.sh
./update_and_push.sh
```

The script stages every changed file, commits with a clean message
("v2 — switch baseline to scikit-image v0.24.0"), and pushes to
`origin main`. You'll be asked for your GitHub username + token again
(or it may use the keychain from yesterday).

When done it prints a new commit SHA. **Don't submit this SHA to
Mercor yet** — it's the "v2 code, placeholder reward.json" SHA. The
final one comes after Colab.

## Step 3. Run the three notebooks in Colab

For each notebook, open it from your repo:

```
https://colab.research.google.com/github/Sanjay040695/scikit-image-performance-optimization/blob/main/baseline_colab.ipynb
```

```
https://colab.research.google.com/github/Sanjay040695/scikit-image-performance-optimization/blob/main/candidate_colab.ipynb
```

```
https://colab.research.google.com/github/Sanjay040695/scikit-image-performance-optimization/blob/main/tests.ipynb
```

For **each notebook**:

1. **Runtime → Change runtime type → CPU**. Runtime version: **Latest**
   is fine now (any Python from 3.10 to 3.12 works because we use the
   PyPI wheel, no compile).
2. **Runtime → Run all.**

That's it. No cell edits required this time — the repo URL is pre-baked.

If Colab pops a "Warning: notebook not authored by Google" dialog,
click **Run anyway**.

Order matters:
- Run **baseline_colab.ipynb** first (~5–8 min).
- Then **candidate_colab.ipynb** (~5–8 min).
- Then **tests.ipynb** (~8–12 min, this is the slow one — same-VM
  comparison + writes reward.json).

The last cell of `tests.ipynb` prints a banner like:
```
============================================================
BASELINE  median = X.XXXX s
CANDIDATE median = X.XXXX s
SPEEDUP = X.XXx
existing_tests_pass = True
output_equivalent   = True
============================================================
reward.json written to /content/reward.json
```

Three things MUST be true at the end:
- `existing_tests_pass = True`
- `output_equivalent   = True`
- `SPEEDUP = something other than null`

If any of those is wrong, copy the output to me before pushing further.

## Step 4. Download reward.json and push it back

After `tests.ipynb` finishes:

1. In Colab's left sidebar, click the folder icon → expand `/content/`
   → right-click `reward.json` → **Download**.
2. On your laptop, copy that downloaded file INTO your local repo,
   overwriting the placeholder.
3. In the terminal (inside the repo folder):
   ```bash
   git add reward.json
   git commit -m "Update reward.json with measured Colab numbers"
   git push
   git rev-parse HEAD
   ```
4. The last command prints the **final SHA**. THIS is what Mercor needs.

## Step 5. Mercor form

| Field | Value |
|---|---|
| Target repository | `scikit-image/scikit-image` |
| One-line description | `Vectorized regionprops_table fast path: scipy.ndimage.find_objects for bboxes, np.bincount for areas/centroids, scipy.ndimage.minimum/maximum for intensity extrema; replaces the per-region Python dispatch loop for common scalar properties.` |
| Submission GitHub URL | `https://github.com/Sanjay040695/scikit-image-performance-optimization` |
| Commit SHA | The SHA from step 4 (NOT the one from step 2) |
| Hours spent | Honest answer |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `update_and_push.sh: command not found` | Run `chmod +x update_and_push.sh` first |
| `git push` rejected with "non-fast-forward" | Pull first: `git pull --rebase origin main`, then push |
| Colab cell errors `AssertionError: Expected scikit-image 0.22.0` | Run a single cell first: `!pip install --upgrade scikit-image==0.24.0`. Then **Runtime → Restart session**, then Run all again. |
| `SPEEDUP = null` in the final output | One of correctness checks failed. Look above for `regressions:` and the `match=False` lines and share those |
| Colab times out / disconnects mid-run | Re-run from `Runtime → Run all`. Colab tab needs to stay open for the duration |
