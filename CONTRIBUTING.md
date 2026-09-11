# Contributing to MakerMods Lab

PRs welcome. This page covers setup, the dev loop, and the checks a PR has to pass.

## Setup

Do the editable install from the README's [Quick Start](README.md#quick-start), then add
the dev extra — it brings in ruff, pre-commit, and everything the test suite needs:

```bash
uv pip install -e ".[dev]"
pre-commit install
```

That second line is the one people skip. It wires the git hook, so every commit runs the
formatters and linters on the files you touched. Most problems get fixed in place —
re-`git add` and commit again. Skip it and CI tells you the same thing, slower.

Requires Python ≥ 3.12. Use the repo's `.venv` — pytest fails to collect under other
interpreters because of the pinned lerobot.

## Dev loop

```bash
makermodslab --dev
```

Vite on `:8080`, uvicorn `--reload` on `:8000` — frontend and backend edits reload live.

## Which branch to target

`main` is the release branch. `staging` is a permanent integration branch sitting in
front of it.

**Branch off `staging` and open your PR against `staging`.** `staging` is promoted to
`main` by its own PR when a batch is ready. Both branches run the same workflows, so a
green `staging` is what `main` will do.

Two things follow from that, and both bite if you get them wrong:

- Baseline your checks against `staging`, not `main` — a three-dot diff against `main`
  from a branch cut off `staging` lists the entire staging-vs-main delta.
- Never commit `frontend/dist/` by hand. CI rebuilds and commits it on a push to either
  branch, and `dist` is marked `-merge` in `frontend/.gitattributes`, so a bundle moved
  on both sides conflicts as a _binary_ file — which GitHub's web conflict editor
  refuses outright.
  [`sync_staging.yml`](.github/workflows/sync_staging.yml) merges `main` into `staging`
  after every push to `main` and resolves that automatically, so the promotion PR stays
  clean. Hand-committing dist is how you defeat it.

A hotfix straight into `main` is fine when it's genuinely urgent — the sync workflow
carries it back into `staging` for you.

## Before you open a PR

Run the same checks CI runs, and get them green:

```bash
pre-commit run --files $(git diff --name-only origin/staging...HEAD)   # lint, format, typos, security
pytest                                                                 # Python suite
cd frontend && npm run lint \
  && npx tsc --noEmit -p tsconfig.app.json \
  && npx tsc --noEmit -p tsconfig.node.json \
  && npm run build                                                     # frontend
```

Notes:

- The **first** `pre-commit` run downloads and builds every hook environment (mypy,
  bandit, gitleaks, zizmor) and takes a few minutes. Every run after that is seconds.
- Scope it to your own diff, as above, against **your PR's base** — `staging` for a
  feature branch, `main` for a promotion. `pre-commit run --all-files` is fine too —
  both branches are kept green — but if it ever does surface unrelated churn, don't
  sweep it into your PR; say so in the PR instead.
- The `-p` on `tsc` is not optional, and there are two projects. The root
  `tsconfig.json` is a solution file (`"files": []` plus references), so a bare
  `npx tsc --noEmit` checks zero files and always passes. `tsconfig.app.json` covers
  `src`; `tsconfig.node.json` covers the build configs.
- `Build` is the only required check — it blocks the merge. `Quality` and `Pytest` go
  red on failure but do not block, so nobody is stuck waiting on a formatting nit. Treat
  a red `Quality` as something to fix, not something to merge past.
- Some frontend type/lint errors pre-date any given change. Record the baseline before
  you start and compare against it; don't fix unrelated pre-existing errors in your PR.
- Leave `frontend/dist/` alone — CI rebuilds and commits it on a push to `main` or
  `staging`. A hand-built bundle turns a clean merge into a binary conflict.
- `frontend/package-lock.json` only changes when you deliberately change a dependency.
  An incidental diff there is local `npm install` churn that has broken CI before —
  `git checkout -- frontend/package-lock.json` and move on.

## Suppressing a check

If a hook flags something that is genuinely a false positive, suppress it narrowly and
say why in a comment (`# nosec Bxxx — reason`, `# zizmor: ignore[rule]`, or a scoped
`extend-ignore-re` entry in `pyproject.toml`). Don't widen a global skip list — the
skip lists are shared by the whole repo, and a false positive in one file is not a
reason to stop checking every other file.

Two paths are excluded from the hooks on purpose: `frontend/dist/` (generated, CI
commits it) and `makermodslab/vendor/` (third-party code kept byte-for-byte verbatim so
it stays diffable against upstream).

## License

By contributing you agree that your work is licensed under the
[Apache 2.0 License](LICENSE).
