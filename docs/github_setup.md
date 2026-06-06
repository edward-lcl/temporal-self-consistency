# Pushing this to GitHub

The repo is already initialized with one clean commit. To put it on GitHub:

## Option A — GitHub CLI (fastest)

```bash
cd tsct-temporal-calibration
gh repo create tsct-temporal-calibration --public --source=. --push
```

## Option B — Manual

1. Create an empty repo on github.com (no README, no .gitignore — this repo
   already has them).
2. Then:

```bash
cd tsct-temporal-calibration
git remote add origin https://github.com/<your-username>/tsct-temporal-calibration.git
git branch -M main
git push -u origin main
```

## Before pushing — a few things to confirm

- **Large data files**: `data/prep/` and `predictions/` are gitignored. The
  big `all_triples.jsonl` (207k records) is NOT included — host it on
  HuggingFace Datasets or Google Drive and link it in the README instead of
  committing it.
- **Model checkpoints**: never commit `.safetensors` / `.ckpt` (already
  gitignored). Keep them on HuggingFace.
- **W&B keys / HF tokens**: make sure none are hard-coded in any script before
  pushing a public repo.

## Recommended repo settings

- Add a topic tags: `nlp`, `llm`, `calibration`, `temporal-reasoning`
- Add the team as collaborators (Settings -> Collaborators)
- Turn on branch protection for `main` once others start pushing
- Point the README status badge at the CI workflow once it's running

## Suggested branch workflow for the team

- `main` — protected, reviewed merges only
- `eval/*` — Jason's evaluation work
- `train/*` — Tanvi's training work
- `data/*` — Aarav's data work
- `baselines/*` — Logan's baseline work

Each person works on their branch and opens a PR into `main`.
