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

## Before you open a PR

Run the same checks CI runs, and get them green:

```bash
pre-commit run --files $(git diff --name-only origin/main...HEAD)   # lint, format, typos, security
pytest                                                              # Python suite
cd frontend && npm run lint && npx tsc --noEmit && npm run build    # frontend
```

Notes:

- The **first** `pre-commit` run downloads and builds every hook environment (mypy,
  bandit, gitleaks, zizmor) and takes a few minutes. Every run after that is seconds.
- Scope it to your own diff, as above. `pre-commit run --all-files` is fine too — `main`
  is kept green — but if it ever does surface unrelated churn, don't sweep it into your
  PR; say so in the PR instead.
- `Build` is the only required check — it blocks the merge. `Quality` and `Pytest` go
  red on failure but do not block, so nobody is stuck waiting on a formatting nit. Treat
  a red `Quality` as something to fix, not something to merge past.
- Some frontend type/lint errors pre-date any given change. Record the baseline before
  you start and compare against it; don't fix unrelated pre-existing errors in your PR.
- Leave `frontend/dist/` alone — CI rebuilds and commits it on merge to `main`.
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
