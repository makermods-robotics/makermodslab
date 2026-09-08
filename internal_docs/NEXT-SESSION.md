# Handoff — written end of 2026-08-04 (marathon session)

Read alongside `logs/daily/2026-08-04.md` (full narrative) and `logs/PROJECT.md`. This doc is the actionable state: what's on disk, what's decided, what to do first, and the traps.

## State on disk

- Branch `rig` at `ad93026`, **10 commits ahead of its base today**, local-only:
  `e32e507` wandb removal · `e2d23df` RTC selector · `cee842c` gitignore · `d8accb6` optimizer passthrough (MT43) · `4096a2a` MT2/listing/naming/resume-guards · `6468584` UI gates + target lock · `8a6cb5a` DeployPanel checkpoint race · `8618ef3` ModelCard + naming convergence · `3c5976c` record-first fine-tune start · `ad93026` dist rebuild.
- **UNCOMMITTED (review first!): the MT44 phase-1 guard diff** — `makermodslab/jobs.py`, `makermodslab/datasets.py`, `tests/test_jobs.py`. Feature-space preflight on fine-tune: hard-400 on state/action-dim mismatch and on equal-count camera renames, EXCEPT checkpoint-side all-placeholder keys (`camera\d+` — smolvla_base et al.) which demote to warn (the canonical adapt-a-base flow; decision 2026-08-04, docstring documents it, phase 2's confirm UI may supersede). Missing/extra camera + resolution = warn-only. 165 `test_jobs` green, zero ruff-format drift. Commit message suggestion: `feat(jobs): feature-space preflight on fine-tune (MT44 phase 1)`.
- Test baselines (pre-existing, do NOT chase): 7 `test_models` failures (read the dev's real HF cache — MT35), `test_entry_points_target_correct_functions` (stale editable install; fix = rerun `uv pip install -e .`).
- Remote branches: `pr/remove-wandb` (PR **#52**, open, single commit, awaiting merge) · `pr/optimizer-passthrough` (`b99d264`, its PR #53 was opened-then-closed per user — it is the SEED of the combined training PR) · `origin/fix/training` (PR **#42**, open, will be partly superseded) · `origin/feat/jobs-models-split` (draft PR #26 — the ModelCard/JobsHistory presentation spec; read via `git show`, never checkout).

## Do first, in order

1. ~~**Review + commit the MT44 diff**~~ DONE 08-04 late: committed as `44b4246` (review clean).
2. ~~**Rebase `rig` onto `origin/main` (`743c240`)**~~ DONE 08-04 late: topology preserved, replay-dropped rename imports restored in `7912455` (line-level delta vs backup verified identical to upstream's; backup `rig-pre-rebase-0804`). Original brief: — 9 commits behind, and #46 ("Studio UI polish") rewrote `DeployPanel.tsx` (+145), `TrainPanel.tsx`, `StudioOverlay.tsx`, which rig also reworked. Resolution intent: keep rig's checkpoint-invalidate fix + ModelsLibrary/ModelCard integration, absorb #46's Import-skill button + `pr-9` trigger padding + DatasetPicker. Backend (#24/#28/#29 shutdown fixes) should be near-clean; light `server.py` friction vs wandb removal possible. Lessons from the 08-03 rebase redo: never `git add -A`, diff each replayed merge vs original, patch-id drops won't fire for the PR-carve copies (different diffs — expect manual skips for `e32e507`'s content if #52 merges first).
3. **Carve PRs off post-rebase main, in this order** (per corrected protocol: build in a worktree, show the user the full diff + PR body, push NOTHING until they approve that artifact; get explicit per-PR yes — this was violated once with #53):
   1. **Naming PR** (recommended first; user leaning yes but NOT yet confirmed): authored fresh from `utils/naming.py` + `models.py` enrichment/dedupe/`_run_identity_name` + `lib/modelNames.ts` + card title lines. Fixes main's defects A–E (audit in the 08-04 log; B is a correctness bug — Deploy picker attributes chain weights to the failed first run). Feeds the supervisor identity escalation.
   2. **Training-correctness PR** (manifest proposed, NOT yet confirmed by user): optimizer passthrough (seeded on `pr/optimizer-passthrough`) + the `4623711` resume-form-lock port + done-source/cross-runner refusals + compute-target lock + MT2/option-C + MT44 guards. May split further during carving; requires hunk surgery (rig commits bundle features per-file). Supersedes parts of #42 — needs the supervisor told.
   3. **ModelCard redesign PR**: depends on 1+2 (titles, ref-keyed dropdown, honest fine-tune gate). Never before the rebase (would revert #46's Import button).
4. **Task #5 — JobsHistory port** (queued, full brief in the session task list): PR #26's JobsHistory replaces JobCard/JobsLibrary grid; resume-from-step must survive on rows; only after the rebase.

## Decisions settled today (do not relitigate)

- Resuming a COMPLETED run is disallowed everywhere (LR schedule spent — SmolVLA cosine decays to 2.5e-6 floor over a fixed 30k horizon; fine-tune is the continuation path).
- Cross-runner resume refused + Compute pinned on resume (feature filed as **F7**, incl. cache policy: weights stay in shared HF cache, training_state/ is the GC candidate).
- MT2 = option C (materialize step refs; host-side for local, wrapper-side for cloud). MT3-as-written invalidated (root ≡ final checkpoint, byte-proven).
- Model naming: ALL current work is stopgap pending the **user's supervisor escalation on model identity** (shared-repo chains; three copies of the ranking rule exist — backend `_job_outranks`, `findJobForModel`, `ModelsLibrary.collapseByRepo` — all deletable if identity gets one record per repo). Title rule: task identity, namespace-free, policy/dataset in meta rows, renames verbatim, dedupe date suffixes flex-pinned.
- MT44 matrix: dim mismatch + rig-rename = hard 400; placeholder-base rename = warn; missing/extra camera + resolution = warn (phase 2 = confirm UI, with the pre-launch comparison panel).

## Open decisions (user's)

- Naming-PR-first confirm; training-PR manifest confirm; #42's fate messaging.
- Supervisor conversation: model identity design (options: child-repo-per-resume / README-update-on-completion / identity decoupled from repos), PR #52 merge, always-red Quality CI (uniform red since inception; fix = one cleanup pass + make it required).
- Small parked items: Deploy skill picker has NO policy cue since the prefix peel (muted chip = 2 lines); TrainPanel "Starting point" items lack truncate (defect F); foreign no-record repos still show raw repo ids (defect G residue); `InferenceModal` = ~720 lines dead code carrying an unfixed copy of the DeployPanel stale-list bug (delete in JobsHistory work); function-local `get_hub_status` import in jobs.py now redundant.

## Tests still owed (mt-test-plan scratch file dies with the old session — this replaces it)

- **Tier 2 manual (needs one 200-step local ACT parent, stop it ~150 → interrupted-with-checkpoints)**: MT5 A–D (locked fields render, prefill honesty, steps-gate, record honesty), MT34 (resume with "latest" — guard-skip check), tqdm rebasing (bar shows ~150/400 not 0/200), reload-survival (bar stays rebased after uvicorn reload). Then tier-1 leftovers on fresh UI: target-lock pin visual, MT31 no-resume-toggle, optimizer preset display.
- **Stub-harness (no HF spend, wandb-canary pattern)**: MT36 (cloud-Continue inherits parent timeout — capture run_job kwarg), MT42 (local→cloud continue wrapper args show no resume download = restart-from-0 proof; moot if target lock holds).
- **Hardware days**: RTC sync-vs-RTC A/B on the arm (committed `e2d23df`, never energized; fixed-seed N≥10 per side); MT41 checkpoint-upload race + cloud-resume machinery on next paid cloud run.
- Optional: root-fallback live check (fake cloud job.json pointing hf_repo_id at `lerobot/smolvla_base` → expect one `@root` checkpoint entry; Resume on it → plain-language refusal).

## Traps (all bit this session)

- **:8000 serves the committed dist; :8080 is live dev.** Third bite today. Every "the fix didn't take" was this. Check the URL bar first.
- `ps aux | grep lerobot-train` finds nothing — the module is `lerobot.scripts.lerobot_train` (underscore). Grep the flag: `ps aux | grep -o "\-\-policy.pretrained_path=[^ ]*"`.
- zsh: words starting with `=` (e.g. `echo ===`) trigger =cmd expansion and abort compound commands.
- `jobs.py` has TWO Hub-mock seams: module-level `hf_hub_download` vs function-local `snapshot_download` (kept so `tests/test_rollout.py`'s origin-patching works). Don't "fix" the function-local import without rewriting those tests.
- Local fine-tune Start now returns fast (record-first) — download progress is IN the job log/monitor. Cloud-side wrapper downloads pod-side.
- Agent process exits orphan in-flight subagent work silently (no completion record) — check `git status` + run the relevant test file before assuming an interrupted agent finished.

## Standing conventions (unchanged)

Deploy-agent skill for all code work (Opus, no-commit, fencing, stop-and-report); enumerate commits/pushes and get explicit yes — PR creation gets its own yes on the final artifact; never merge PRs; `gh` needs `--repo makermods-robotics/makermodslab`; ruff-format baseline-aware (jobs.py 11 / test_jobs.py 4 pre-existing hunks); no frontend test infra exists — don't build any without asking.
