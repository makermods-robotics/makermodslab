# Model Training Bug List

*Citations synced 2026-07-26 against `main` @ `ce42158`. **Only `file:line` coordinates were changed** — no prose, verdicts, severities, entry IDs or numbering were altered. References whose construct no longer exists at HEAD are marked inline `⚠ UNRESOLVED @ HEAD ce42158`.*

*Renumbered 2026-08-02 into one continuous per-module sequence: the original entries 1–24 now carry their long-used cross-reference IDs **MT1–MT24** in the heading, and the former appended `NEW-n` / `N<n>` series became **MT25–MT42** (NEW-1→MT25 … NEW-18→MT42). Every bug entry now carries a **First identified** date (and **Resolved**, where the entry attests a fix or a descope). Entries may also carry a **Fix:** line, directly under **First identified**, linking the pull request(s) that fix or attempt to fix them.*

> **Re-verified against `main` @ `c7d9f27` on 2026-07-23.** This list folds the 23-07-26 re-audit into the
> original. It **supersedes** the pre-reverification version, archived at
> [`archive/Model Training Bug List (pre-reverify).md`](archive/Model%20Training%20Bug%20List%20%28pre-reverify%29.md);
> the standalone re-audit is archived at
> [`archive/Model Training REVERIFIED 23-07-26.md`](archive/Model%20Training%20REVERIFIED%2023-07-26.md).
> Cross-module current-state overview: [`CURRENT 23-07-26.md`](CURRENT%2023-07-26.md).
>
> **Context you need for every verdict below (branch-topology warning):**
> - The original entries were audited **2026-07-14 against `518ca56`** (the **`andrew`** branch). Re-verification
>   target is **`main` @ `c7d9f27`** (the `redesign` line). **`518ca56` is NOT an ancestor of `main`** —
>   the two diverge by ~9.4k insertions / 13.4k deletions across 132 files, and neither is a superset. So
>   re-verifying "against `main`" *reopens* entries whose fix only ever lived on `andrew`.
> - **Fixes recorded as "in progress" in the original are ABSENT from `main`.** Entries **16 and 17** are
>   therefore **OPEN on `main`, not "in progress"** (`andrew`'s `_persist_collapsed`, `_reattach_disk_skip`,
>   `resume_total`/`log_freq` constructor are all absent). **MT24**'s fix ("uncommitted on `redesign`") did not
>   land on `main` either. Also: `andrew` renamed `--eval_freq` → `--env_eval_freq`; `main` still emits
>   `--eval_freq` — the two branches build **different trainer argv** and are not interchangeable when porting
>   fixes. *(Per CURRENT: `main`'s HEAD has since renamed the field to `env_eval_freq`, closing that specific
>   divergence; MT24 is about `--policy.tags`, still emitted at `train.py:268-270`.)*
> - **The frontend was restructured on `main`.** `Training.tsx` went 900+ → 105 lines; the form moved to
>   `components/training/TrainingConfigurator.tsx` + `components/studio/TrainPanel.tsx`; the monitor became
>   `components/training/TrainingJobDialog.tsx`. **Every frontend `file:line` in the original was stale**;
>   `main`-current cites are layered on.
> - **PRs #5–#11 are CLOSED with `merged=no`.** Nothing here is in flight.
> - **Numbering:** the original numbered entries **1–24**; the re-verification uses the same numbers as
>   **MT1–MT24**, and since the 2026-08-02 renumber the headings carry those IDs directly. New
>   findings used **NEW-n** — *renumbered 2026-08-02 to **MT25–MT42** (NEW-1→MT25 … NEW-18→MT42);
>   see the note at the top of this file.*
> - Read-only static audit — no HF Jobs submission, Hub call, training process, or hardware. No code modified.

This ledger tracks confirmed defects in MakerMods Lab's model-training pathways: configuration and preflight, local and Hugging Face cloud execution, resume and fine-tune, monitoring, checkpoints, model publication, deletion, and restart recovery.

Audited on July 14, 2026, against MakerMods Lab commit `518ca56`; re-verified against `main` @ `c7d9f27` on 2026-07-23.

## Status key

- **Open:** confirmed defect with no validated fix.
- **In progress:** implementation has started but is not fully validated. *(NOTE: on `main`, none of the original "in progress" fixes are present — see per-entry verdicts.)*
- **Fixed:** implementation and regression coverage are complete.
- **Design gap / Ruled out:** as labelled.

## Severity key

- **P0 — Critical:** breaks a primary training pathway or leaves remote compute running without correct local ownership.
- **P1 — High:** can train from the wrong state, lose or misrepresent an artifact, break recovery, or delete a resource in active use.
- **P2 — Moderate:** misleading failure, incomplete monitoring, avoidable operational risk, or a recoverable inconsistency.
- **P3 — Low:** limited hardening or usability defect with no demonstrated loss of training state.

## Re-verification verdict key (2026-07-23)

- **STILL REAL** — reproduces in `main`'s code as written.
- **FIXED** — the mechanism is gone on `main`.
- **PARTIALLY FIXED** — one leg closed, the rest reproduces.
- **NOT APPLICABLE — SUPERSEDED** — the code the finding described no longer exists in that shape.
- **CANNOT VERIFY STATICALLY** — depends on remote HF Jobs / Hub behavior.

## Confirmed findings

### MT1 · Ruled out — Hugging Face cloud jobs never reach a terminal state locally

**First identified:** 2026-07-14

**Severity:** P0 — Critical

> **Verdict on `main` (2026-07-23): Ruled out — unchanged.** Evidence:
> `hf_cloud.py:426`, `:744-749`. `inspect_job` returns `status.stage` as a plain uncoerced string so the
> terminal comparison works.
>
> ⚠ **Correction, 2026-07-26 (verified at HEAD `ce42158`): the "hardening still unimplemented" clause below
> and in the Priority line is STALE — the hardening is present.** `_TERMINAL_STAGES` is an explicit allowlist
> (`runners/hf_cloud.py:426` — `frozenset({"COMPLETED", "CANCELED", "ERROR", "DELETED"})`) and the poller
> coerces before comparing against it: `stage_str = str(stage).upper()` then `if stage_str in
> _TERMINAL_STAGES` (`:745-746`). **Verdict itself is unchanged — still NOT-A-BUG**; only the outstanding-work
> note was wrong. The definitive close-out is still one live cloud job run to terminal.

**Verdict:** NOT-A-BUG — inspect_job returns status.stage as a plain uncoerced string so the terminal comparison works — the original audit probed the enum in isolation (huggingface_hub/_jobs_api.py:92-98 JobStatus is an uncoerced dataclass, :265-266 builds it from raw JSON; makermodslab/runners/hf_cloud.py:745-746 therefore matches).

**Priority:** none — ruled out; recommended hardening: compare with 'stage in _TERMINAL_STAGES'; a live cloud job to terminal is the definitive close-out.

**Pathway:** Cloud start -> remote execution -> status polling -> terminal job/model handoff. The cloud runner converts the Hub status stage with `str(stage).upper()` and compares with bare strings; the installed Hub client's enum string is shaped like `JobStage.COMPLETED`. Evidence: `makermodslab/runners/hf_cloud.py:426`, `:730-776`.

### MT2 · Open — Fine-tuning a selected Hub checkpoint discards the selected revision

**First identified:** 2026-07-14

**Severity:** P1 — High

> **Verdict on `main` (2026-07-23): STILL REAL.** `_resolve_finetune_pretrained_path` still returns
> `chosen.ref.split("@", 1)[0]` (`jobs.py:731`), discarding `checkpoints/<step>`. The docstring (`jobs.py:727-730`)
> documents this as a known limitation, but the UI presents a step picker (`JobCard.tsx:366-385`,
> `TrainPanel.tsx:190-197`) and passes `finetune_from_step` through (`TrainingConfigurator.tsx:105`), so the
> user is told a step was honoured when it was not. `--policy.pretrained_path=<repo>` loads repo-root weights
> (`train.py:291-292`). **Fix cluster with MT3, MT12** (Hub-checkpoint listing semantics).

**Verdict:** CONFIRMED — `_resolve_finetune_pretrained_path` returns `chosen.ref.split("@", 1)[0]` (makermodslab/jobs.py:730-731), dropping the selected step; `--policy.pretrained_path=<repo>` then loads repo-root weights (makermodslab/train.py:291-292).

**Priority:** P1 — common fine-tune flow silently starts from repository-root weights instead of the user-selected step, with no signal to the user (verified 2026-07-14 against the current working tree)

**Pathway:** Job checkpoint selection -> fine-tune request -> pretrained model resolution -> trainer start. Hub checkpoints are `repo@checkpoints/<step>`, but fine-tune resolution strips everything after `@`. Evidence: `makermodslab/jobs.py:682-734`, `makermodslab/train.py:275-286`.

### MT3 · Open — The final Hub policy is hidden when periodic checkpoints exist

**First identified:** 2026-07-14

**Severity:** P1 — High

> **Verdict on `main` (2026-07-23): STILL REAL.** `_list_imported_hub` returns the root policy **only** when the
> tree scan is empty (`jobs.py:784-788`). Worse for tracked cloud runs: `_list_hub_checkpoints` (`jobs.py:792-800`)
> has **no** root fallback at all, so a completed cloud run with `save_checkpoint=false` publishes a root policy
> and its job card reports zero checkpoints (`checkpoint_count` via `jobs.py:1472` → `TrainingJobDialog.tsx:354-357`
> "No checkpoints yet"). `_hub_checkpoints_from_files` matches only `checkpoints/<step>/pretrained_model/config.json`
> (`jobs.py:736-757`). **Fix cluster with MT2, MT12.**

**Verdict:** CONFIRMED — `_list_imported_hub` returns the root config.json only when the checkpoints tree is empty (makermodslab/jobs.py:783-788), and `_hub_checkpoints_from_files` matches only `checkpoints/<step>/pretrained_model/config.json` (makermodslab/jobs.py:735, 738-756), so the final root policy is never listed alongside periodic checkpoints.

**Priority:** P1 — a completed cloud model can appear unusable from its job card, or default to an older periodic checkpoint instead of the final policy (verified 2026-07-14 against the current working tree)

**Pathway:** Cloud completion/import -> Hub model discovery -> model/checkpoint selection -> inference or continuation. LeRobot publishes the final policy at the root at end of training, which can be newer than the last periodic checkpoint. Cloud runs with checkpoint saving disabled can publish a root policy while their tracked job exposes no selectable checkpoints. Evidence: `makermodslab/jobs.py:736-800`, `makermodslab/models.py:150-178`, `makermodslab/train.py:261-318`.

### MT4 · Open — Switching a cloud resume request to Local launches an invalid resume

**First identified:** 2026-07-14

**Severity:** P1 — High

> **Verdict on `main` (2026-07-23): STILL REAL (new file:line).** `TargetCard.setRunner` (`TargetCard.tsx:42-50`)
> switches to `local` unconditionally; there is no resume lock on the target. The backend branches on the
> **source** runner, not the target (`jobs.py:1168`), so a cloud source sets only `resume_from_hub_repo`
> (`:1178-1179`) and never `config_path`. `build_training_command`'s resume branch requires `request.resume and
> request.config_path` (`train.py:251`), so the call falls through to the fresh branch and emits `--resume true`
> with no `--config_path` (`train.py:322`). Deterministic doomed run. **Fix cluster with MT5, MT6** (resume-form
> lock).

**Verdict:** CONFIRMED — TargetCard allows switching to local unconditionally (frontend/src/components/training/config/TargetCard.tsx:57-60); a cloud source sets only `resume_from_hub_repo`, never `config_path` (makermodslab/jobs.py:1163-1174), so `build_training_command` falls through its resume branch (makermodslab/train.py:251) and emits `--resume true` with no `--config_path` (makermodslab/train.py:322).

**Priority:** P1 — a valid-looking resume form deterministically launches a doomed local run (verified 2026-07-14 against the current working tree)

**Pathway:** Resume action -> resume form -> target switch -> local runner command construction. Evidence: `frontend/src/pages/Training.tsx:198-240`, `frontend/src/components/training/config/TargetCard.tsx:83-145`, `makermodslab/jobs.py:1157-1184`, `makermodslab/train.py:251-273`. ⚠ UNRESOLVED @ HEAD ce42158 — `frontend/src/pages/Training.tsx` is now a 106-line router wrapper (reduced in the redesign, 9e18b27); the cited lines are past EOF. Searched frontend/src/components/training/ (TrainingConfigurator.tsx, config/*) and studio/TrainPanel.tsx; construct not re-located.

### MT5 · Open — Editable resume controls do not match inherited trainer state

**First identified:** 2026-07-14
**Fix:** [PR #42](https://github.com/makermods-robotics/makermodslab/pull/42) (open) · branch `fix/training`

**Severity:** P1 — High → **P2 on `main`**

> **Verdict on `main` (2026-07-23): PARTIALLY FIXED — policy is now locked; target/batch/optimizer/W&B still
> editable.** `policyLocked` (`TrainingConfigurator.tsx:537` → `EssentialsCard.tsx:77`) closes the "record labels
> a different policy" leg the original led with. The cloud dependency-extra leg (`hf_cloud.py:555` still keyed on
> the edited `config.policy_type`) is now unreachable *from the UI* on resume, reachable only from non-UI callers.
> The remaining live leg is **target/batch/optimizer/W&B editability** while the resume branch restores those
> from `config_path` (`train.py:251-273`). Cites: `TrainingConfigurator.tsx:537`, `EssentialsCard.tsx:77`,
> `ConfigurationTab.tsx:27-39`. **Fix cluster with MT4, MT6.**

**Verdict:** CONFIRMED — the config cards stay fully editable on resume (ConfigurationTab passes no lock, frontend/src/components/training/ConfigurationTab.tsx:25-37) while the resume branch restores those fields from config_path (makermodslab/train.py:248-270); the record name and cloud dependency extra still use the edited policy_type (makermodslab/jobs.py:1192; makermodslab/runners/hf_cloud.py:555).

**Priority:** P2 — only bites when the user edits inherited fields; the default resume flow is unaffected (verified 2026-07-14 against the current working tree)

**Pathway:** Resume action -> configuration editing -> dependency preflight -> runner launch -> job history. Evidence: `frontend/src/pages/Training.tsx:539-553`, `frontend/src/components/training/config/EssentialsCard.tsx:86-119`, `makermodslab/train.py:247-273`, `makermodslab/jobs.py:1122-1243`. ⚠ UNRESOLVED @ HEAD ce42158 — `frontend/src/pages/Training.tsx` is now a 106-line router wrapper (reduced in the redesign, 9e18b27); the cited lines are past EOF. Searched frontend/src/components/training/ (TrainingConfigurator.tsx, config/*) and studio/TrainPanel.tsx; construct not re-located.

### MT6 · Open — Cloud W&B resume can omit the required secret

**First identified:** 2026-07-14

**Severity:** P1 — High

> **Verdict on `main` (2026-07-23): STILL REAL.** The resume form initializes `wandb_enable: false`
> (`TrainingConfigurator.tsx:188`); the resume branch emits no `--wandb.*` flags so the checkpoint's restored
> W&B config governs (`train.py:248-270`), but `WANDB_API_KEY` is forwarded only when the new request's
> `wandb_enable` is true (`hf_cloud.py:576-585`). **Fix cluster with MT4, MT5** (derive secret forwarding from
> the restored config, not the draft).

**Verdict:** CONFIRMED — the resume form initializes `wandb_enable: false` (frontend/src/pages/Training.tsx:233); the resume branch emits no `--wandb.*` flags so the checkpoint's restored W&B config governs (makermodslab/train.py:248-270), but `WANDB_API_KEY` is forwarded only when the new request's wandb_enable is true (makermodslab/runners/hf_cloud.py:576-585). ⚠ UNRESOLVED @ HEAD ce42158 — `frontend/src/pages/Training.tsx` is now a 106-line router wrapper (reduced in the redesign, 9e18b27); the cited lines are past EOF. Searched frontend/src/components/training/ (TrainingConfigurator.tsx, config/*) and studio/TrainPanel.tsx; construct not re-located.

**Priority:** P1 — no workaround in the form; a previously working W&B-enabled cloud run loses tracking or fails on resume (verified 2026-07-14 against the current working tree)

**Pathway:** W&B-enabled cloud checkpoint -> resume form -> cloud submission -> restored trainer config. Evidence: `frontend/src/pages/Training.tsx:224-235`, `makermodslab/train.py:248-270`, `makermodslab/runners/hf_cloud.py:573-592`. ⚠ UNRESOLVED @ HEAD ce42158 — `frontend/src/pages/Training.tsx` is now a 106-line router wrapper (reduced in the redesign, 9e18b27); the cited lines are past EOF. Searched frontend/src/components/training/ (TrainingConfigurator.tsx, config/*) and studio/TrainPanel.tsx; construct not re-located.

### MT7 · Open — Job deletion bypasses the active-inference model guard

**First identified:** 2026-07-14

**Severity:** P1 — High

> **Verdict on `main` (2026-07-23): STILL REAL.** `delete_job` calls `job_registry.delete` directly
> (`server.py:1664-1672`) and `JobRegistry.delete` guards only on running state (`jobs.py:1545-1557`); no
> `_model_in_use` check like the model-browser path performs (`models.py:1055-1059`). *(Same rmtree-under-live-
> rollout family as Inference I15.)*

**Verdict:** CONFIRMED — `delete_job` calls `job_registry.delete` directly (makermodslab/server.py:1664-1668) and `JobRegistry.delete` guards only on running state (makermodslab/jobs.py:1540-1552); no `_model_in_use` check like the model-browser path performs (makermodslab/models.py:1052-1054, guard at 959-984).

**Priority:** P1 — checkpoint files can be removed under a live inference subprocess; the safety guarantee depends on which UI card initiates deletion (verified 2026-07-14 against the current working tree)

**Pathway:** Completed training job -> start inference from checkpoint -> delete job from monitoring. Evidence: `makermodslab/models.py:962-987`, `:1040-1063`, `makermodslab/server.py:1664-1678`, `makermodslab/jobs.py:1545-1558`.

### MT8 · Open — Config-only and partial models are classified as usable

**First identified:** 2026-07-14

**Severity:** P1 — High → **P2**

> **Verdict on `main` (2026-07-23): STILL REAL.** `_resolve_pretrained_dir` accepts a bare root config.json with
> no weights (`models.py:329-337`), `_list_local_checkpoints` requires only `pretrained_model/config.json`
> (`jobs.py:562-565`), and `_cleanup_partial_model` preserves any dir that passes that probe (`models.py:1112-1117`).

**Verdict:** CONFIRMED — `_resolve_pretrained_dir` accepts a bare root config.json with no weights required (makermodslab/models.py:334-335), `_list_local_checkpoints` requires only pretrained_model/config.json (makermodslab/jobs.py:562-563), and `_cleanup_partial_model` preserves any dir that passes that probe (makermodslab/models.py:1107-1112).

**Priority:** P2 — the doc's P1 was downgraded for narrow trigger conditions (requires an interrupted download or a genuinely weightless source); failure surfaces later at inference/fine-tune (verified 2026-07-14 against the current working tree)

**Pathway:** Model download/import -> local validation -> listing -> inference or fine-tune. Evidence: `makermodslab/models.py:150-178`, `:318-337`, `:1090-1117`, `:1131-1145`.

### MT9 · Open — Cloud checkpoint upload can permanently publish a partial checkpoint

**First identified:** 2026-07-14
**Fix:** [PR #33](https://github.com/makermods-robotics/makermodslab/pull/33) (draft) · branch `fix/mt9-cloud-checkpoint-watcher-partial-upload` · [PR #42](https://github.com/makermods-robotics/makermodslab/pull/42) (open) · branch `fix/training`

**Severity:** P1 — High

> **Verdict on `main` (2026-07-23): STILL REAL.** `hf_cloud.py:340-343` gates the upload on
> `pretrained_model/config.json` alone and `:352` adds the step to `seen` on success, so a partial directory is
> sealed; the final pass (`:384`) skips seen steps. The recommended completeness marker
> (`training_state/scheduler_state.json`) is **not** implemented on `main`. The resume-side mitigation is intact
> (`jobs.py:633-637` verifies `training_state/training_step.json` on the Hub and fails cleanly), so *resume* is
> protected and *inference / fine-tune from the partial artifact* remain exposed — unchanged. The 2026-07-19
> production observation below stands (nothing on `main` changed the mechanism). Related concurrency bug: MT33.

**Verdict:** CONFIRMED — the wrapper uploads once `pretrained_model/config.json` exists and marks the step seen permanently (makermodslab/runners/hf_cloud.py:340-352, final scan skips seen steps at 382-386); note the training_state resume mitigation: `_resolve_cloud_resume` verifies `training_state/training_step.json` on the Hub (makermodslab/jobs.py:632-636), so resume from a partial checkpoint fails cleanly — inference/fine-tune from the partial artifact remain exposed.

**Priority:** P1 — permanent partial publication on a paid run; mitigated for resume only (verified 2026-07-14 against the current working tree)

**Pathway:** Remote checkpoint save -> wrapper scan -> Hub upload -> resume/inference. Evidence: `makermodslab/runners/hf_cloud.py:328-355`, `:358-386`, `makermodslab/jobs.py:582-639`.

**Observed in production (2026-07-19, `redesign`)**

- Confirmed live. Cloud run `makermods/smolvla_makermods_eraser_stack_20260718_175437_2026-07-18_20-39-35` (SmolVLA, 10k steps, ended failed) left an **intermittent** pattern on the Hub — steps 3000 and 5000 have `pretrained_model/` but **no `training_state/`**, while steps 1000, 2000, 4000 are complete. Intermittent subset = the 15s watcher landing inside the `pretrained_model` → `training_state` write window; SmolVLA's large save widens it.
- User-visible failure: resuming step 5000 hit `_resolve_cloud_resume`'s clean-fail guard. Resume from step 4000 (complete) then succeeded.
- Recommended fix (unimplemented): gate upload on `training_state/scheduler_state.json` (the last file LeRobot writes), and do not add the step to `seen` until a complete upload succeeds.

**Observed in production again (2026-07-28) — truncation point is EARLIER than 07-19 showed**

- Second recurrence, same dataset lineage: run `makermods/smolvla_makermods_eraser_stack_20260718_175437_2026-07-27_16-52-42` (HF job `6a67ef4b6026358f64018ef1`, `a10g-small`, SmolVLA, 15k steps, `save_freq=1000`), killed at step 4129 by an external SIGTERM unrelated to this bug.
- **Step 001000 on the Hub holds only `pretrained_model/config.json` and `pretrained_model/.tmpUKtcgO`** — no `model.safetensors`, no `train_config.json`, no processor files, no `training_state/`. The 07-19 observation had complete `pretrained_model/` trees missing only `training_state/`; this one landed *inside* `policy.save_pretrained`'s weight write, so the exposed window is the **whole save**, not just the `pretrained_model` → `training_state` gap. The `.tmp*` file is safetensors' atomic-write temp captured mid-rename, which is what pins the scan to that point.
- **Timing proof.** LeRobot logs `Checkpoint policy after step 2000` at `01:07:08`; at the run's steady 2.13 s/step the step-1000 save began ≈ `00:31:38`. The Hub commit `checkpoint 001000` is timestamped `00:31:43` — ≈5 s into the save. Commit lag for the healthy steps, for contrast: 002000 = 14 s, 003000 = 20 s, 004000 = 26 s.
- **The asymmetry that makes this P1 rather than cosmetic:** step 004000 hit the *same* race but the temp file vanished between `upload_folder`'s listing and read, raising `I/O error: No such file or directory (os error 2)`. The exception skipped `seen.add` (`hf_cloud.py:352`), so the next 15 s scan re-uploaded it complete. 001000's partial upload *succeeded*, so it was sealed permanently. **The bug only does lasting damage when the partial upload does not error** — which also means the user-visible log line appears in the harmless case and is silent in the harmful one.
- Resume-side mitigation held as documented: 004000 was complete and resumable. But `_list_hub_checkpoints` still offers 001000 in the resume picker and `_resolve_cloud_resume` (`jobs.py:633-637`) then rejects it — correct, but a dead option is surfaced to the user.
- **Refinement to the recommended fix.** `training_state/scheduler_state.json` is written only when a scheduler exists (`save_training_state`: `if scheduler is not None`), so that gate is conditional on the policy preset. A policy-agnostic alternative is LeRobot's own completion marker: `update_last_checkpoint()` creates the `checkpoints/last` symlink *after* `save_checkpoint` returns, so resolving `last` and uploading only what it has pointed at uses upstream's own ordering guarantee. Cheap independent hardening: pass `ignore_patterns=[".tmp*"]` to `upload_folder` (`:346`) — a temp file is precisely what landed here. *(LeRobot cites read from the locally installed package — `common/train_utils.py:145-164`, `scripts/lerobot_train.py:663` — not from the container's pinned spec; confirm the write order against the pin before relying on it.)*
- Cheapest verification of any fix: run with `save_freq` small on a large policy and diff `list_repo_files` per checkpoint against the expected file set; a fix is good only if every published step carries `training_state/training_step.json`.

**Base rate and final-checkpoint exposure (measured 2026-07-28)**

- Swept all 35 `makermods` model repos via `list_repo_files`: **7 carry at least one partial checkpoint** (~20% of runs). Affected: `smolvla_…sock_purple_green_orange_2026-07-08_01-47-15` (006000), `smolvla_…bimanual_test…2026-07-08_18-05-51` (000010), `smolvla_…sock_purple_green_orange_sort_2026-07-08_23-21-47` (007000), `act_…sock_2_only_merged_2026-07-15_19-32-14` (001000), `smolvla_…sock_2_only_more_orange_2026-07-17_15-52-35` (009000), plus both `eraser_stack` runs. 5 of the 7 are missing `model.safetensors` itself, so the scan landing inside the weight write is the common case, not the edge case.
- **The final checkpoint is equally exposed.** `is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps` (`lerobot_train.py:589`) routes the last step through the identical save→watch→upload path. The wrapper's post-`proc.wait()` final scan (`hf_cloud.py:378-386`) would normally catch it — the trainer has exited, so the tree is complete — but `if entry.name in seen: continue` (`:343`) makes it skip any step the 15 s watcher already published partially. **The final-scan safety net is defeated by exactly the bug it would otherwise cover.**
- Confirmed in the data: `smolvla_…eraser_stack…2026-07-18_20-39-35` has its **last checkpoint (005000) partial**, alongside 003000.
- **Severity splits by run outcome.** On a run that *completes*, LeRobot's own end-of-training `push_model_to_hub` (`lerobot_train.py:735-743`) writes the model to the **repo root**, bypassing the watcher entirely — all 5 completed runs above have an intact root `model.safetensors`, so a deployable model survives and only optimizer-state resume from that step is lost. On a run that *fails*, there is no root push (both `eraser_stack` runs show `root model.safetensors=False`), so the checkpoint tree is the entire salvage — and a partial last checkpoint costs the most steps precisely when it is least affordable.

### MT10 · Open — A reattached local training failure is recorded as success

**First identified:** 2026-07-14
**Fix:** [PR #34](https://github.com/makermods-robotics/makermodslab/pull/34) (draft) · branch `fix/mt10-tailing-runner-false-done-on-crash`

**Severity:** P1 — High

> **Verdict on `main` (2026-07-23): STILL REAL.** `TailingJobRunner.returncode()` returns `0` once the pid is
> gone (`jobs.py:478-485`, comment documents the choice), and the watchdog writes `record.state = "done" if
> rc == 0 else "failed"` (`jobs.py:1754`). Compounding: `list_local_models` admits a run only when `state ==
> "done"` (`models.py:252-258`), so a trainer that crashed after a MakerMods Lab restart is not merely mislabeled —
> its last partial checkpoint is **promoted into the models browser** as a completed model, and
> `_find_local_record` (`models.py:264-280`) then lets it be uploaded to the Hub. **Fix cluster with MT11 +
> MT25 + MT29** (runner finalization contract).

**Verdict:** CONFIRMED — `TailingJobRunner.returncode()` returns 0 once the PID is gone because it cannot reap a process from another session (makermodslab/jobs.py:477-484); the watchdog finalizes rc==0 as "done" (makermodslab/jobs.py:1749).

**Priority:** P1 — a trainer crash after a MakerMods Lab restart is presented as a completed model with downstream actions offered (verified 2026-07-14 against the current working tree)

**Pathway:** Local training -> MakerMods Lab restart -> PID/log reattachment -> trainer exit -> terminal state. Evidence: `makermodslab/jobs.py:429-485`, `:1565-1624`.

### MT11 · Open — Cloud stop reports cancellation before remote cancellation succeeds

**First identified:** 2026-07-14

**Severity:** P1 — High

> **Verdict on `main` (2026-07-23): STILL REAL — and worse than recorded (see MT25).** `stop()` pre-sets
> `CANCELED` (`hf_cloud.py:760`) *before* `cancel_job` (`:762`) and swallows every exception at `logger.info`
> (`:763-765`). `is_running()` then returns `False` (`:767-771`) and the watchdog finalises the record and drops
> the runner (`jobs.py:1768`). Compounding: **`cancel_job` appears exactly once in the entire repository** —
> inside this method — reachable only via `POST /jobs/{id}/stop` requiring a live runner for a `running` record.
> Once the record is finalised (by this bug, a watchdog tick, a restart marking it `interrupted`, or delete),
> **MakerMods Lab has no code path that can cancel the remote job** — see **MT25**. Correction to the original impact
> line ("no longer monitors"): `server.py:1364-1373` keeps a dismissed-but-active Hub job visible in `/jobs/hub`,
> so the gap is *actionability*, not visibility. **Fix cluster with MT10 + MT25 + MT29.**

**Verdict:** CONFIRMED — `stop()` pre-sets CANCELED before calling `cancel_job` and swallows every cancellation exception at info level (makermodslab/runners/hf_cloud.py:755-765); `is_running()` then returns False (767-771), so the runner is finalized and stops monitoring.

**Priority:** P1 — a failed cancel leaves the paid GPU burning while MakerMods Lab reports the job stopped and no longer monitors it (verified 2026-07-14 against the current working tree)

**Pathway:** Running cloud job -> Stop -> local terminal transition -> remote cancellation. Evidence: `makermodslab/runners/hf_cloud.py:755-776`, `makermodslab/jobs.py:1417-1498`.

### MT12 · Open — Cloud resume lineage cannot distinguish parent and child checkpoints

**First identified:** 2026-07-14

**Severity:** P1 — High → **P2**

> **Verdict on `main` (2026-07-23): STILL REAL (+ new UI symptom MT30).** Cloud resume reuses the parent's
> output repo (`config.policy_repo_id = config.resume_from_hub_repo`, `hf_cloud.py:536`), so parent and child
> records enumerate the same shared checkpoint tree via `_list_hub_checkpoints` (`jobs.py:792-800`). The observable
> half — duplicate checkpoint entries with identical `step` in the lineage dropdown — is **MT30** (`JobCard.tsx:187-216`).
> **Fix cluster with MT2, MT3.**

**Verdict:** CONFIRMED — cloud resume reuses the parent's output repo (`config.policy_repo_id = config.resume_from_hub_repo`, makermodslab/runners/hf_cloud.py:536), so parent and child records enumerate the same shared checkpoint tree via `_list_hub_checkpoints` (makermodslab/jobs.py:791-799).

**Priority:** P2 — the doc's P1 was downgraded for narrow trigger conditions; provenance ambiguity without artifact loss (verified 2026-07-14 against the current working tree)

**Pathway:** Cloud checkpoint -> resume child -> checkpoint listing -> later continuation. Evidence: `makermodslab/runners/hf_cloud.py:518-547`, `makermodslab/jobs.py:588-639`, `frontend/src/components/jobs/JobCard.tsx:116-211`, `frontend/src/components/jobs/JobsSection.tsx:353-375`. ⚠ UNRESOLVED @ HEAD ce42158 — `JobsSection.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/jobs/, construct not re-located.

### MT13 · Open — Invalid training request bodies return HTTP 500

**First identified:** 2026-07-14

**Severity:** P2 — Moderate

> **Verdict on `main` (2026-07-23): STILL REAL (both legs).** `from_legacy` runs before/outside the endpoint's
> try block (`server.py:1079-1080` vs `try:` at `:1146`), so a pydantic ValidationError escapes as a 500. Second
> clause survived the UI rewrite: the timeout field moved to `AdvancedCard.tsx:294-322` and its `timeoutInvalid`
> state (`:103-104`) is *still* absent from `startDisabled` (`TrainingConfigurator.tsx:477-485`).

**Verdict:** CONFIRMED — `from_legacy` runs before/outside the endpoint's try block (makermodslab/server.py:1079-1080; try begins at 1146), so a pydantic ValidationError escapes uncaught as a 500 — the `except ValueError` at 1155 cannot reach it.

**Priority:** P2 — internal-server-error response instead of an actionable 4xx (verified 2026-07-14 against the current working tree)

**Pathway:** Direct API or invalid UI state -> create training job -> schema validation -> error response. Evidence: `makermodslab/server.py:1077-1081`, `frontend/src/components/training/config/TargetCard.tsx:42-58`, `:150-190`, `frontend/src/pages/Training.tsx:503-535`. ⚠ UNRESOLVED @ HEAD ce42158 — `frontend/src/pages/Training.tsx` is now a 106-line router wrapper (reduced in the redesign, 9e18b27); the cited lines are past EOF. Searched frontend/src/components/training/ (TrainingConfigurator.tsx, config/*) and studio/TrainPanel.tsx; construct not re-located. ⚠ UNRESOLVED @ HEAD ce42158 — `TargetCard.tsx` is 140 lines at HEAD; `:150-190` is past EOF. Searched frontend/src/components/training/config/TargetCard.tsx; construct not re-located.

### MT14 · Open — Core numeric training parameters accept zero and negative values

**First identified:** 2026-07-14

**Severity:** P2 — Moderate

> **Verdict on `main` (2026-07-23): STILL REAL.** `TrainingRequest` carries a field validator only for
> `hf_job_timeout` (`train.py:206-222`); steps/batch_size/log_freq/save_freq have no positive-range validation
> (`train.py:115-130`), and the server only warns on freq > steps (`server.py:1086-1098`).

**Verdict:** CONFIRMED — `TrainingRequest` carries a field validator only for `hf_job_timeout` (makermodslab/train.py:203-219); steps/batch_size/log_freq/save_freq have no positive-range validation (makermodslab/train.py:115-130), and the server only warns on freq > steps (makermodslab/server.py:1086-1098).

**Priority:** P2 — local jobs fail only after process creation; cloud jobs incur startup cost before the deterministic config error (verified 2026-07-14 against the current working tree)

**Pathway:** Training configuration -> request validation -> local/cloud trainer start. Evidence: `makermodslab/train.py:104-220`, `frontend/src/components/training/config/EssentialsCard.tsx:53-149`, `AdvancedCard.tsx:185-315`.

### MT15 · Open — Terminal transition can omit final logs and checkpoints until reload

**First identified:** 2026-07-14

**Severity:** P2 — Moderate

> **Verdict on `main` (2026-07-23): STILL REAL (new file:line).** The checkpoint poll's interval returns early
> once state != running with no final tick, and the log poll fetches only while running — cites now
> `TrainingJobDialog.tsx:135-139`, `:154-164`, `:170-174`. Backend twin is MT28 (the persisted log is truncated
> on disk, not just in the live view).

**Verdict:** CONFIRMED — the checkpoint poll's interval returns early once state != running with no final tick (frontend/src/pages/Training.tsx:719-723), and the log poll fetches logs only while `next.state === "running"` (frontend/src/pages/Training.tsx:738-758), so the transition-observing tick skips the final drain. ⚠ UNRESOLVED @ HEAD ce42158 — `frontend/src/pages/Training.tsx` is now a 106-line router wrapper (reduced in the redesign, 9e18b27); the cited lines are past EOF. Searched frontend/src/components/training/ (TrainingConfigurator.tsx, config/*) and studio/TrainPanel.tsx; construct not re-located.

**Priority:** P2 — the final traceback/success line and a last-moment checkpoint are missing until manual reload (verified 2026-07-14 against the current working tree)

**Pathway:** Running job -> final trainer output/checkpoint -> terminal state poll -> monitoring UI. Evidence: `frontend/src/pages/Training.tsx:691-729`, `:730-760`. ⚠ UNRESOLVED @ HEAD ce42158 — `frontend/src/pages/Training.tsx` is now a 106-line router wrapper (reduced in the redesign, 9e18b27); the cited lines are past EOF. Searched frontend/src/components/training/ (TrainingConfigurator.tsx, config/*) and studio/TrainPanel.tsx; construct not re-located.

### MT16 · Open — Cloud resume progress is not rebased to the inherited step

**First identified:** 2026-07-14
**Fix:** [PR #42](https://github.com/makermods-robotics/makermodslab/pull/42) (open) · branch `fix/training`

**Severity:** P2 — Moderate

> **Verdict on `main` (2026-07-23): STILL REAL — the andrew "in progress" fix is NOT on `main` (REOPENED).** The
> cloud `_tail_loop` calls `parse_metrics_into(stripped, self._metrics)` with no resume offset (`hf_cloud.py:702`),
> unlike the local/tailing runners; `main`'s `HfCloudJobRunner.__init__` (`hf_cloud.py:462-467`) has no
> `resume_total`/`log_freq` params (cf. `andrew:hf_cloud.py:785`). The history endpoint does rebase
> (`jobs.py:1218`), so live and reconstructed views disagree. **Fix cluster with MT17** (cloud tail loop).

**Verdict:** CONFIRMED — the cloud `_tail_loop` calls `parse_metrics_into(stripped, self._metrics)` with no resume offset (makermodslab/runners/hf_cloud.py:702), unlike the local/tailing runners (makermodslab/jobs.py:403, 527-529); the history endpoint does rebase (makermodslab/jobs.py:1456), so live and reconstructed views disagree.

**Fix in progress (2026-07-14) — NOTE: this andrew-branch fix is NOT on `main`.** `HfCloudJobRunner` accepted `resume_total`/`log_freq` and passed them into `parse_metrics_into`; covered by `test_tail_loop_parse_applies_resume_offset`/`test_tail_loop_applies_log_freq_for_exact_step`. **Absent from `main`.**

**Priority:** P2 — cosmetic monitoring inconsistency (verified 2026-07-14 against the current working tree)

**Pathway:** Cloud resume -> remote logs -> metric parsing -> progress display. Evidence: `makermodslab/jobs.py:375-405`, `:528-530`, `makermodslab/runners/hf_cloud.py:678-708`.

### MT17 · Open — Cloud reattachment duplicates persistent logs

**First identified:** 2026-07-14

**Severity:** P2 — Moderate

> **Verdict on `main` (2026-07-23): STILL REAL — the andrew "in progress" fix is NOT on `main` (REOPENED).**
> `reattach` opens the log file in append mode (`hf_cloud.py:609`) with `_lines_processed` still 0 (init at
> `:494`), so the SSE stream's replayed historical prefix passes the `seen <= self._lines_processed` dedup
> (`:696-698`) and is re-appended in full. `main` has none of `andrew`'s `_persist_collapsed`/`_reattach_disk_skip`
> (cf. `andrew:hf_cloud.py:558,685`). **Fix cluster with MT16.**

**Verdict:** CONFIRMED — `reattach` opens the log file in append mode (makermodslab/runners/hf_cloud.py:609) with `_lines_processed` still 0 (init at 494), so the SSE stream's replayed historical prefix passes the `seen <= self._lines_processed` dedup (696) and is re-appended in full.

**Fix in progress (2026-07-14) — NOTE: this andrew-branch fix is NOT on `main`.** `_persist_collapsed` + `_reattach_disk_skip` seeded from `_count_existing_log_lines`; covered by four tests. **Absent from `main`.**

**Priority:** P2 — full log history duplicated on disk per server restart (verified 2026-07-14 against the current working tree)

**Pathway:** Cloud training -> MakerMods Lab restart -> remote-log reattachment -> local log persistence. Evidence: `makermodslab/runners/hf_cloud.py:494`, `:599-620`, `:678-710`.

### MT18 · Open — Manual model upload can partially succeed while reporting total failure

**First identified:** 2026-07-14

**Severity:** P2 — Moderate

> **Verdict on `main` (2026-07-23): STILL REAL.** A `metadata_update` failure raises after the public repo and
> weights already exist (`models.py:918-927`, `:938-944`); cache invalidation runs only after tagging succeeds
> (`:946-948`); the successful repo id is never written back to the source `JobRecord`. Related to MT32 (the
> same function wipes existing repo tags).

**Verdict:** CONFIRMED — a `metadata_update` failure raises `_upload_model_error` after the public repo and weights already exist (makermodslab/models.py:915-941); cache invalidation runs only after tagging succeeds (makermodslab/models.py:943); the successful repo id is never written back to the source JobRecord.

**Priority:** P2 — the UI reports failure after a public mutation succeeded, prompting retries and duplicate entries (verified 2026-07-14 against the current working tree)

**Pathway:** Completed local model -> Upload to Hub -> weights upload -> metadata tagging -> UI result. Evidence: `makermodslab/models.py:886-949`, `:950-954`, `makermodslab/jobs.py:65-108`.

### MT19 · Open — The frontend blocks cloud runs on host-only training dependencies

**First identified:** 2026-07-14

**Severity:** P2 — Moderate

> **Verdict on `main` (2026-07-23): STILL REAL (same correction: only the accelerate gate).** The policy-extra
> preflight is target-gated (`TrainingConfigurator.tsx:400`); only the accelerate gate blocks cloud —
> `trainingExtraAvailable === false` renders the gate page-wide regardless of target
> (`TrainingConfigurator.tsx:457-459`).

**Verdict:** CONFIRMED-WITH-CORRECTIONS — the policy-extra preflight is already target-gated (skipped for cloud, frontend/src/pages/Training.tsx:424); only the accelerate gate blocks cloud: `trainingExtraAvailable === false` renders TrainingExtraGate for the whole page regardless of target (frontend/src/pages/Training.tsx:486-494; probe at makermodslab/utils/system.py:203-211). No W&B host gate on Start exists. ⚠ UNRESOLVED @ HEAD ce42158 — `frontend/src/pages/Training.tsx` is now a 106-line router wrapper (reduced in the redesign, 9e18b27); the cited lines are past EOF. Searched frontend/src/components/training/ (TrainingConfigurator.tsx, config/*) and studio/TrainPanel.tsx; construct not re-located.

**Priority:** P2 — a valid cloud pathway is blocked by an unrelated host dependency (verified 2026-07-14 against the current working tree)

**Pathway:** Cloud configuration -> host dependency preflight -> cloud submission. Evidence: `frontend/src/pages/Training.tsx:255-285`, `:472-497`, `makermodslab/runners/hf_cloud.py:204-327`, `makermodslab/utils/system.py:203-265`. ⚠ UNRESOLVED @ HEAD ce42158 — `frontend/src/pages/Training.tsx` is now a 106-line router wrapper (reduced in the redesign, 9e18b27); the cited lines are past EOF. Searched frontend/src/components/training/ (TrainingConfigurator.tsx, config/*) and studio/TrainPanel.tsx; construct not re-located.

### MT20 · Open — All training HTTP 409 responses are presented as a local mutex conflict

**First identified:** 2026-07-14

**Severity:** P2 — Moderate

> **Verdict on `main` (2026-07-23): STILL REAL.** `startTrainingJob` rewrites every 409 to the mutex message
> (`jobsApi.ts:190-195`) while the backend also 409s `DatasetNotOnHubError` (`server.py:1150-1154`).

**Verdict:** CONFIRMED — `startTrainingJob` rewrites every 409 to the mutex message (frontend/src/lib/jobsApi.ts:190-195) while the backend also 409s `DatasetNotOnHubError` (makermodslab/server.py:1150-1154).

**Priority:** P2 — mitigated by the deliberate 400 rerouting of the offline-local guard (makermodslab/server.py:1132-1145) and the UI's upload-then-train flow; wrong remediation remains for non-UI callers and future 409s (verified 2026-07-14 against the current working tree)

**Pathway:** Training start -> backend domain conflict -> frontend error message. Evidence: `frontend/src/lib/jobsApi.ts:180-195`, `makermodslab/server.py:1100-1160`, `makermodslab/jobs.py:1210-1243`.

### MT21 · Open — Legacy job migration moves every directory under the legacy root

**First identified:** 2026-07-14

**Severity:** P2 — Moderate

> **Verdict on `main` (2026-07-23): STILL REAL.** `_migrate_legacy_cwd_jobs` moves every directory under the
> legacy root before best-effort metadata rewrite, never requiring a job.json (`jobs.py:1005-1029`).

**Verdict:** CONFIRMED — `_migrate_legacy_cwd_jobs` moves every directory under the legacy root before best-effort metadata rewrite, never requiring a job.json (makermodslab/jobs.py:1004-1024).

**Priority:** P2 — the doc's P1 was downgraded for narrow trigger conditions (one-shot, fires only on first boot under the new layout with a populated legacy dir) (verified 2026-07-14 against the current working tree)

**Pathway:** MakerMods Lab startup -> legacy output discovery -> migration to the current training root. Evidence: `makermodslab/jobs.py:991-1051`.

### MT22 · Open — Checkpoint ZIP download is unbounded in memory and can race active writes

**First identified:** 2026-07-14

**Severity:** P2 — Moderate

> **Verdict on `main` (2026-07-23): STILL REAL.** The full ZIP is built in `io.BytesIO` then copied again via
> `getvalue()` (`server.py:1605-1612`, `:1625`); the only guard is `runner != "local"` (`:1590`) — no
> running-state check, so files can change during traversal.

**Verdict:** CONFIRMED — the full ZIP is built in `io.BytesIO` then copied again via `getvalue()` (makermodslab/server.py:1605-1612, 1625); the only guard is `runner != "local"` (makermodslab/server.py:1590) — no running-state check, so files can change during traversal.

**Priority:** P2 — roughly 2x archive-size peak memory; a download during active training can capture an inconsistent snapshot (verified 2026-07-14 against the current working tree)

**Pathway:** Running/completed local job -> checkpoint download -> ZIP response. Evidence: `makermodslab/server.py:1576-1628`, `frontend/src/components/jobs/JobCard.tsx:212-290`.

### MT23 · Open — The job-registry lock spans remote network work and runner submission

**First identified:** 2026-07-14

**Severity:** P2 — Moderate

> **Verdict on `main` (2026-07-23): STILL REAL.** `JobRegistry.start` holds `self._lock` across the whole body
> including `runner.start()` (`jobs.py:1122` … `:1241` wrapping `:1221`), which for cloud runs performs the
> synchronous dataset upload and job submission (`hf_cloud.py:516-593`, `:641-676`); list/get/stop/delete take
> the same lock.

**Verdict:** CONFIRMED — `JobRegistry.start` holds `self._lock` across the whole body including `runner.start()` (makermodslab/jobs.py:1216), which for cloud runs performs the synchronous dataset upload and job submission (makermodslab/runners/hf_cloud.py:516-593); list/get/stop/delete take the same lock.

**Priority:** P2 — unrelated job operations block for the duration of network work; a slow Hub op freezes the job interface (verified 2026-07-14 against the current working tree)

**Pathway:** Cloud start/resume/fine-tune -> Hub resolution/upload/submission -> concurrent job operations. Evidence: `makermodslab/jobs.py:1100-1243`, `makermodslab/runners/hf_cloud.py:518-597`, `:641-676`.

### MT24 · Open — Resume re-passes `--policy.tags` in a form draccus rejects on the config-file path

**First identified:** 2026-07-14
**Fix:** [PR #42](https://github.com/makermods-robotics/makermodslab/pull/42) (open) · branch `fix/training`

**Severity:** P1 — High

> **Verdict on `main` (2026-07-23): STILL REAL — the fix never landed on `main`.** The resume branch still emits
> `_policy_hub_flags()` (`train.py:79`, called at `:268-270`), which builds the tag list as the unquoted token
> `[makermods,openbooth,MakerMods Lab]`. Fresh runs parse fine (argparse tolerates the bracket form); resume loads via
> `from_pretrained` → `draccus.parse(config_file, args=cli_args)`, whose list-override decoder rejects the
> unquoted string, killing the trainer at parse time. `main`'s `train.py:268-270` is byte-identical to the version
> that produced the 2026-07-19 traceback. *(Note: the `--eval_freq` → `env_eval_freq` field rename has since
> landed on `main`'s HEAD per CURRENT, but MT24 is about `--policy.tags`, which is unaffected.)*

**Verdict:** CONFIRMED (observed in production 2026-07-19, `redesign`) — the resume branch emits `_policy_hub_flags()`, which builds the tag list as the unquoted token `[makermods,openbooth,MakerMods Lab]` (makermodslab/train.py:79, called at :267). A fresh run parses fine (draccus builds the config from argparse). Resume does not: it loads via `from_pretrained` → `draccus.parse(config_file, args=cli_args)`, whose list-override decoder rejects the unquoted string. The trainer dies at parse time before any training happens.

**Priority:** P1 — every cloud resume with `push_to_hub` (the default for cloud runs) crashes deterministically at launch, after the checkpoint has already been downloaded. Fix implemented on `andrew`/`redesign` but **uncommitted, and NOT on `main`.**

**Pathway:** Resume action -> runner command construction -> trainer config parse (`from_pretrained`).

**Observed (2026-07-19, `redesign`):** Resuming step 4000 of `makermods/smolvla_makermods_eraser_stack_...20-39-35` downloaded the checkpoint, launched the trainer, and died: `draccus.utils.DecodingError: `policy.tags`: … value='[makermods,openbooth,MakerMods Lab]' is not of a valid input for a list type`. The checkpoint's own `train_config.json` already carries `tags`/`push_to_hub`/`repo_id`/`private`, so re-passing them on resume is redundant as well as breaking.

**Fix (implemented, uncommitted — NOT on `main`):** resume branch stops emitting `--policy.tags`, keeping `--policy.private false` only. Follow-up not done: the fresh-run branch still emits the same unquoted bracket form (works via argparse but latently fragile).

**Evidence:** `makermodslab/train.py:69-80` (`_policy_hub_flags`), `:262-273` (resume branch), `lerobot/configs/train.py` `from_pretrained`.

**Recurred and FIXED (2026-07-28, `feat/jobs-models-split` @ `d640daa`, uncommitted)**

- Third occurrence, identical signature. HF job `6a694d1115e81eca66a8d82e` — a cloud resume of `…eraser_stack…2026-07-27_16-52-42` from step 4000 — downloaded and reconstructed the checkpoint (1.61 GB), launched the trainer, and died after 54s with `draccus.utils.DecodingError: 'policy': … 'tags': … value='[makermods,openbooth,MakerMods Lab]' is not of a valid input for a list type`. Note the 2026-07-19 occurrence was on the *previous* run of this same dataset: **every cloud resume attempted so far has hit this**, which is the practical meaning of "deterministic at launch".
- Confirmed the redundancy claim directly rather than by inference: the checkpoint's `checkpoints/004000/pretrained_model/train_config.json` carries `policy.tags = ['makermods','openbooth','MakerMods Lab']`, `policy.private = False`, `policy.push_to_hub = True`, `policy.repo_id = …`. Dropping the CLI override loses nothing.
- **Fix applied** at `train.py:268-280`: the resume branch now emits `["--policy.private", "false"]` directly instead of `_policy_hub_flags()`, with a comment explaining the argparse-vs-config-file parse divergence. Generated resume argv verified to contain no `--policy.tags`.
- Test updated: `test_resume_push_to_hub_emits_public_and_tags` asserted the broken behaviour and is now `test_resume_push_to_hub_emits_public_but_never_tags`, asserting `"--policy.tags" not in cmd`. 122 tests pass across `test_train.py`, `test_runners_hf_cloud.py`, `test_jobs.py`.
- **Still open:** the fresh-run branch (`:312`) keeps emitting the unquoted bracket token. It works via argparse and was left alone deliberately — changing it risks fresh runs, which are the common path. Latent fragility, unchanged from the original entry's follow-up note.
- **Branch caution:** this fix is on `feat/jobs-models-split` and uncommitted, exactly like the two prior fixes that never reached `main`. That is the reason this bug has now been "fixed" three times and hit production three times. Port it before the next re-verification sweep.

### MT25 · P1 — There is no way to cancel a remote HF job once the local record is finalised *(formerly NEW-1)*

**First identified:** 2026-07-23

**Pathway:** cloud run → any local finalisation (failed cancel / watchdog / restart / delete) → paid GPU keeps running with no MakerMods Lab control.

`cancel_job` is called from exactly one place (`hf_cloud.py:762`, inside `HfCloudJobRunner.stop()`). `stop()` is reachable only through `JobRegistry.stop` (`jobs.py:1379-1395`), which raises `JobNotRunningError` unless `record.state == "running"` **and** `self._runners[job_id]` exists. Every finalisation path removes that possibility while the remote job may still run:

- MT11's optimistic `_set_terminal("CANCELED")` before a `cancel_job` that failed (`hf_cloud.py:760-765`);
- the watchdog popping the runner on any terminal transition (`jobs.py:1768`);
- a restart where the persisted record lacks `hf_job_id`/`hf_flavor` → marked `interrupted`, never reattached (`jobs.py:1618-1623`). Reachable: the record is persisted `state="running"` at `jobs.py:1212` **before** `runner.start()` submits the job, and `hf_job_id` is only written at `:1234-1240` — a crash in that window leaves a submitted, paid, permanently untracked job;
- `DELETE /jobs/{id}` (`server.py:1664-1668` → `jobs.py:1552-1553`) drops the runner outright.

The orphan does remain **visible** (`server.py:1373` keeps a dismissed id in `/jobs/hub` while its stage is active — a real mitigation), but `HubJobCard` offers only "View on Hub" and a trash button gated on `!isHubJobActive(job)` (`HubJobCard.tsx:84-119`) — **no cancel control and no cancel endpoint** exist. The user's only remedy is huggingface.co. Fix shape: a `POST /jobs/hub/jobs/{id}/cancel` calling `HfApi.cancel_job` independently of the registry, plus MT11's reordering. **Fix cluster with MT10, MT11, MT29.**

**Evidence:** `hf_cloud.py:755-765`, `:767-776`; `jobs.py:1379-1387`, `:1552-1553`, `:1618-1623`, `:1768`; `server.py:1364-1373`, `:1492-1505`, `:1664-1677`; `HubJobCard.tsx:84-119`.

### MT26 · P1 — Fine-tune from the jobs library launches with the WRONG policy type, and the UI locks it *(formerly NEW-2)*

**First identified:** 2026-07-23
**Fix:** [PR #42](https://github.com/makermods-robotics/makermodslab/pull/42) (open) · branch `fix/training`

**Pathway:** JobCard "Fine-tune" / SkillDetailDialog → studio Train prefill → policy select disabled → trainer launched with `--policy.type act` against a non-ACT checkpoint.

`TrainPanel` owns `policyType`, defaulting to `"act"` (`TrainPanel.tsx:157`). `resolveFinetune` resolves the base checkpoint's real policy **only on the Hub-repo branch** (`!jobId && opts.repoId`); the two prefill entry points both take the **`jobId`** branch, where `policy` stays `null`:

- `JobCard.tsx:378-384` — `openStudio("train", { train: { baseJobId, baseStep, baseName } })`;
- `SkillDetailDialog.tsx:125` — `{ baseJobId: model.id }`;
- consumed at `TrainPanel.tsx:267-273` → `resolveFinetune({ jobId, step, name })`.

So `setPolicyType` never fires (`TrainPanel.tsx:214`) and `policyType` stays at default/last value. Meanwhile `finetuneSeed != null` sets `policyLocked` (`TrainingConfigurator.tsx:537`), which **disables** the policy select (`EssentialsCard.tsx:77`) under the caption "Set by the base skill — the run trains the same architecture as its source checkpoint" (`:91-94`). The claim is false and the user cannot correct it. Resulting argv: `--policy.type act --policy.pretrained_path <base-checkpoint>` (`train.py:285`, `:291-292`). The manual path is correct (`handleBaseModelChange` does `if (model.policy_type) setPolicyType(...)`, `TrainPanel.tsx:246`) — a wiring omission on the prefill path. *(Not statically determinable whether lerobot hard-fails or mis-loads; either way the advertised fine-tune is not what runs.)*

**Evidence:** `TrainPanel.tsx:157`, `:178-214`, `:231-249`, `:262-285`, `:475-479`; `TrainingConfigurator.tsx:537`; `EssentialsCard.tsx:73-94`; `JobCard.tsx:375-385`; `SkillDetailDialog.tsx:125`; `train.py:285`, `:291-292`.

### MT27 · ~~P1~~ → P2 (narrow trigger) — Implicit cloud-run dataset upload decides visibility at its own site *(formerly NEW-3)*

**First identified:** 2026-07-23 · **Reframed:** 2026-07-26 (product decision)
**Fix:** [PR #8](https://github.com/makermods-robotics/makermodslab/pull/8) (closed) · branch `fix/dataset-cloud-notice-public-mismatch` · [PR #19](https://github.com/makermods-robotics/makermodslab/pull/19) (closed) · branch `fix/dataset-cloud-notice-public-mismatch`

> **The "publishes PUBLICLY" complaint is void — public is the intended default** (see CURRENT → "Descoped by
> product decision"; the record-flow sibling CURRENT P0-2 / Dataset D3 / Recording R17 is retired). What remains
> is that this site hardcodes `private=False` itself instead of reading one central policy constant — a
> **third** independent visibility site alongside Dataset D1 and Dataset D12. Open PR #19 introduces that
> constant (`DATASET_DEFAULT_PRIVATE`) and converts this call site; merging it closes this entry.

**Pathway:** cloud start → `_ensure_dataset_on_hub` → `push_to_hub(private=False)` with no per-call consent.

`hf_cloud.py:671`: `LeRobotDataset(repo_id).push_to_hub(tags=with_makermodslab_tag(None), private=False)`. The comment at `:665-670` states this is deliberate; the browser flow normally reaches the Hub via the explicit upload-then-train path (`TrainingConfigurator.tsx:429-443` + `LocalDatasetCloudNotice`), and `JobRegistry.start` preflight raises `DatasetNotOnHubError` first on a definitive `local_only` (`jobs.py:1116-1120`) — so this is not the common route. It is still reachable, because the preflight only blocks on a **definitive** `local_only` while `_ensure_dataset_on_hub` re-decides from `dataset_info`:

- `get_hub_status` memoizes definitive answers for the process lifetime (`datasets.py:190-193`, `:213-217`). A repo cached `on_hub` then deleted on the Hub yields `RepositoryNotFoundError` at `hf_cloud.py:650-653` → local copy found at `:656` → **public push**.
- `get_hub_status` returns `unknown` (uncached) on any transport error (`datasets.py:196-202`), which the preflight deliberately lets through (`jobs.py:1111-1114`).

Secondary defect regardless of trigger: this push does **not** call `invalidate_hub_status`/`invalidate_dataset_listing_cache`, unlike every other upload site, so the dataset card keeps showing stale locality after publication.

**Evidence:** `hf_cloud.py:641-676` (esp. `:650-653`, `:656`, `:671`); `jobs.py:1107-1120`; `datasets.py:83-88`, `:164-217`.

### MT28 · P2 — Cloud terminal transition truncates the persisted log (backend twin of MT15) *(formerly NEW-4)*

**First identified:** 2026-07-23

`_set_terminal` sets `_stop_event` (`hf_cloud.py:622-629`). `_tail_loop` checks it per line (`:692`) and in its outer wait (`:722-723`). The status poller reaches a terminal stage on a 5s cadence (`:421`, `:730-753`) independent of the SSE stream, so it routinely fires **before** the log stream has delivered the tail — the final trainer traceback, `[wrapper] uploaded checkpoint <step>`, and `[wrapper] trainer exited with rc=N` (`:353`, `:388`). Those lines are never fetched → never written to `log.jsonl` (`:707-712`), and there is no post-terminal drain. `read_persisted_logs` (`jobs.py:1406-1429`) and `read_metrics_history` (`:1431-1463`) are permanently missing the end of every cloud run — MT15 is only the *frontend* not draining; this destroys the record on disk. The single biggest reason a failed cloud run is undiagnosable after the fact. Fix: on terminal, stop reconnecting but let the current SSE iteration finish (or one bounded non-follow `fetch_job_logs` pass) before closing the file.

**Evidence:** `hf_cloud.py:421`, `:622-629`, `:678-728` (esp. `:687-692`, `:717-723`), `:730-753`.

### MT29 · P2 — A user-initiated cloud stop is recorded as `failed` with a synthetic exit-code message *(formerly NEW-5)*

**First identified:** 2026-07-23
**Fix:** [PR #42](https://github.com/makermods-robotics/makermodslab/pull/42) (open) · branch `fix/training`

`JobState` has no `canceled` member (`jobs.py:48`). `stop()` sets `_terminal_status = "CANCELED"` with no message (`hf_cloud.py:760`; `_set_terminal` stores `message` only when truthy, `:626-628`). `returncode()` returns `1` for anything not `COMPLETED` (`:773-776`), so the watchdog writes `state="failed"`, `exit_code=1`, `error_message="Subprocess exited with code 1"` (`jobs.py:1754-1767`). Downstream: the monitor renders `failed — Subprocess exited with code 1` for a deliberate stop (`TrainingJobDialog.tsx:324-327`); `endedBeforeTarget` treats `failed`/`interrupted` as resumable (`JobCard.tsx:319-329`), so a stopped run and a crashed run are indistinguishable; `list_local_models` excludes it (`models.py:253`), so a local run stopped at a good checkpoint never appears in the models browser. **Fix cluster with MT10, MT11, MT25.**

**Evidence:** `jobs.py:48`, `:1750-1767`; `hf_cloud.py:622-629`, `:755-765`, `:773-776`, `:796-803`.

### MT30 · P2 — MT12's UI symptom: duplicate checkpoint entries with identical `step` in the lineage dropdown *(formerly NEW-6)*

**First identified:** 2026-07-23
**Fix:** [PR #42](https://github.com/makermods-robotics/makermodslab/pull/42) (open) · branch `fix/training`

`JobCard` fetches `/jobs/{id}/checkpoints` for the run **and each ancestor** and flat-merges by step (`JobCard.tsx:187-216`, `:203`). For a cloud resume, parent and child share one output repo (`hf_cloud.py:536`) and both enumerate the same tree (`jobs.py:792-800`), so every parent checkpoint appears **twice** with the same `step`. `CheckpointDropdown` keys and values items on the raw step (`CheckpointDropdown.tsx:52-60`): duplicate React keys plus two `SelectItem`s sharing a `value`, and `lineageCheckpoints.find(c => c.ckpt.step === selectedStep)` (`JobCard.tsx:294-296`) silently picks whichever came first, which decides which run a subsequent Continue/Download/Inference targets. The observable half of MT12; fix with it (dedupe by `(repo, step)`).

**Evidence:** `JobCard.tsx:187-216`, `:292-304`, `:331-349`, `:394-400`; `CheckpointDropdown.tsx:52-60`; `hf_cloud.py:536`; `jobs.py:792-800`.

### MT31 · P2 — A seeded resume can be silently converted into a fresh full-length run *(formerly NEW-7)*

**First identified:** 2026-07-23
**Fix:** [PR #42](https://github.com/makermods-robotics/makermodslab/pull/42) (open) · branch `fix/training`

`AdvancedCard` exposes a raw "Resume from checkpoint" switch (`AdvancedCard.tsx:277-285`) not disabled on a resume seed. `TrainingConfigurator` seeds `resume: !!resumeSeed` but keeps `resume_from_job_id`/`resume_from_step` unconditionally (`:183-187`), and the "Continuing …" banner is keyed on `resumeSeed`, not `config.resume` (`:499-514`). Toggle it off and: `jobs.py:1161` skips resume resolution, `config_path` stays `None`; `train.py:248` falls through to fresh and emits `--resume false` with `--steps` still prefilled at `sourceSteps * 2` (`TrainingConfigurator.tsx:173`). A from-scratch run of double length launches while the UI says "Continuing X from step N". The record still carries `resume_from_job_id`, and `read_metrics_history` walks that lineage **regardless of `config.resume`** (`jobs.py:1449-1453`), so the chart concatenates the parent's curve with a child that restarted at step 0 and dedupes by step — the parent's history is overwritten.

**Evidence:** `AdvancedCard.tsx:282-290`; `TrainingConfigurator.tsx:173`, `:183-187`, `:499-514`; `jobs.py:1161-1189`, `:1431-1463`; `train.py:251`.

### MT32 · P2 — `upload_local_model` wipes an existing repo's tag metadata *(formerly NEW-8)*

**First identified:** 2026-07-23

`upload_local_model` accepts a caller-supplied `repo_id`, creates with `exist_ok=True` (`models.py:919`), and finishes with `metadata_update(..., {"tags": final_tags}, overwrite=True)` (`:939`). On a repo that already exists (e.g. one lerobot pushed, carrying `{"robotics","lerobot",<model_type>}`), `overwrite=True` **replaces** the tag list with `with_makermodslab_tag([policy_tag])` (`utils/config.py:90`, `:93-105`), dropping the generic `lerobot` tag — exactly what `_list_author_models`/`_hub_policy_type` key on (`models.py:102-123`, `:466-493`), so the repo can fall out of the `/jobs/hub` model listing. Called from `POST /jobs/{id}/upload` (`server.py:942-945`). Distinct from MT18 (partial-success reporting on the same function).

**Evidence:** `models.py:886-954` (esp. `:916`, `:919`, `:932-939`), `:102-123`, `:466-493`; `utils/config.py:85-105`.

### MT33 · P2 — The cloud wrapper's `seen` set is mutated from two threads without a lock *(formerly NEW-9)*

**First identified:** 2026-07-23

`_watch` runs `_scan_and_upload` on a 15s cadence in a daemon thread (`hf_cloud.py:358-368`), and the main thread runs a final `_scan_and_upload` in the `finally` of `proc.wait()` immediately after `stop_event.set()` (`:378-386`) — which does not join the watcher. `_scan_and_upload` does a check-then-add on `seen` around a slow `upload_folder` (`:343-352`), so the same step can be uploaded twice concurrently into the same repo. Self-healing in the common case, but it makes MT9's `seen` bookkeeping harder to reason about and produces spurious failure lines in the user-visible log.

**Evidence:** `hf_cloud.py:328-368`, `:378-389`.

### MT34 · P2 — The resume step guard is skipped when resuming "latest" *(formerly NEW-10)*

**First identified:** 2026-07-23

`server.py:1102` blocks a resume only when `cfg.resume_from_step is not None`. Resuming from the latest checkpoint (`resume_from_step = None`, an explicitly supported mode — `jobs.py:615-617`, `:658-659`, `ResumeSeed.step: number | null` at `TrainingConfigurator.tsx:39`) skips the guard, so a `steps` value at or below the actual latest checkpoint reaches the trainer and dies there. The UI is currently safe because `JobCard.goToResume` always sends a concrete step (`JobCard.tsx:337`), but the frontend mirror of the check has the same hole (`TrainingConfigurator.tsx:467-472`).

**Evidence:** `server.py:1099-1114`; `jobs.py:615-617`, `:658-659`; `TrainingConfigurator.tsx:39`, `:467-472`; `JobCard.tsx:331-349`.

---

### MT35 · P2 — `tests/test_models.py` reads the developer's real pin/hide files, so the model tests pass or fail depending on the machine *(formerly NEW-11)*

**First identified:** 2026-07-24

**Observed (2026-07-24):** after downloading `makermods/sock_2_only_merged` through the UI,
`test_list_all_models_local_type_wins_on_both_collapse` fails `assert 2 == 1` with that repo present in the
result; **7 tests in `tests/test_models.py`** fail this way. The same failures reproduce in two independent
checkouts of the same commit, so this is environmental, not a code regression.

**Root cause (static).** `list_all_models` folds the user's *pinned* Hub models into the listing —
`for repo_id in get_saved_custom_models():` (`models.py:628`, plus `get_hidden_models()` at `:658`). Those read
`SAVED_CUSTOM_MODELS_FILE` / `SAVED_HIDDEN_MODELS_FILE` (`utils/config.py:72`, `:80`), which point at the real
`~/.cache/huggingface/lerobot/saved_custom_models.json`.

The collection objects are deliberately built to be patchable — the comment at `config.py:821-822` says so
outright: *"The `*_FILE` constants are read through a lambda (not captured) so monkeypatching them in tests
reaches the collection"* (`_SAVED_CUSTOM_MODELS = _JsonRepoCollection(lambda: SAVED_CUSTOM_MODELS_FILE, …)`,
`:830-836`). **But the `tmp_lerobot_home` fixture never patches them.** `tests/conftest.py:36-76` redirects
`CALIBRATION_BASE_PATH_*`, `ROBOTS_PATH`, `LEADER/FOLLOWER_CONFIG_PATH`, `PORT_CONFIG_PATH`, the two port
files, `DISMISSED_HUB_JOBS_FILE` and `MAKERMODSLAB_BISO_STAGING_PATH` — and stops there. The four
`SAVED_CUSTOM_{DATASETS,MODELS}_FILE` / `SAVED_HIDDEN_{DATASETS,MODELS}_FILE` constants are **not** in the
list, so every test that reaches `list_all_models()` (or `list_all_datasets()`, which has the symmetric
`get_saved_custom_datasets` / `get_hidden_datasets` reads) sees the developer's real pinned and hidden ids.
`tmp_lerobot_home` is also opt-in, not autouse, so tests taking only the `registry` fixture are exposed twice
over.

This contradicts the file's own header contract — *"HF and the filesystem are MOCKED throughout"*
(`tests/test_models.py:16-18`) — and the fixture's own docstring, *"Redirect **every** persisted-state path
under `~/.cache/huggingface/lerobot/`"* (`conftest.py:37-38`).

**Suggested direction (not applied):** add the four `SAVED_*` constants to `tmp_lerobot_home`, and make the
fixture (or an autouse companion that patches just those four) apply to the model/dataset listing tests.
P2 rather than P1: no product behavior is wrong, but a suite that fails on a developer's own machine trains
people to ignore red — and the same unpatched reads mean a test *could* mutate the real pin file.

### MT36 · P2 — A cloud "Continue" silently drops the parent run's job timeout, falling back to 2h *(formerly NEW-12)*

**First identified:** 2026-07-28
**Fix:** [PR #42](https://github.com/makermods-robotics/makermodslab/pull/42) (open) · branch `fix/training`

`ResumeSeed` (`TrainingConfigurator.tsx:37-51`) deliberately carries the parent run's cadence and target settings forward — `sourceSteps`, `logFreq`, `saveFreq`, `runner`, `flavor`, each with a comment saying it exists "to preserve on resume". **`hf_job_timeout` is not among them.** The form field reads `config.hf_job_timeout ?? ""` (`AdvancedCard.tsx:102`), so on a Continue it renders blank; `buildRequest` then sends `undefined` for any blank value (`TrainingConfigurator.tsx:121-124`), and `resolve_job_timeout` falls back to `HF_JOB_TIMEOUT = "2h"` (`hf_cloud.py:413-415`).

Consequence: a resume of a long run silently gets a **2h ceiling** regardless of what the parent was configured with, and a resume is by definition the tail of a run that already proved it needs a long budget. The user must retype the timeout with no prompt indicating it was cleared — every other long-run-relevant setting persisted, so there is nothing to signal this one didn't.

**Concrete case (2026-07-28).** Parent run `…eraser_stack…2026-07-27_16-52-42` was configured `hf_job_timeout='6h'` (confirmed in its persisted `job.json`). Resuming it from step 4000 leaves 11,000 steps at the observed 2.13 s/step ≈ 6.5h; a Continue left at the default would be killed at 2h, around step 7400 — a second truncated paid run, for a reason invisible in the form.

**Severity:** P2 — wastes a paid run and its wall-clock, but publishes checkpoints normally along the way, so nothing is destroyed. Compare MT31 (seeded resume silently converted to a fresh full-length run), same class.

**Suggested direction (not applied):** add `hfJobTimeout?: string` to `ResumeSeed` and prefill it in both `goToResume` sites (`JobCard.tsx:331-349`, `ModelCard.tsx:271-308`). Failing that, surface the effective default in the field's placeholder so a blank field doesn't read as "no limit".

**Verification:** launch a cloud Continue, and before submitting compare the Advanced card's timeout field against the parent's `job.json` `hf_job_timeout`.

---

### MT37 · P1 — Cloud runs never set `save_checkpoint_to_hub`, so the full checkpoint is written to the container's disk and discarded; only the policy weights survive *(formerly NEW-13)*

**First identified:** 2026-07-28

**Observed live**, not statically inferred: deploying `makermods/smolvla_makermods_eraser_stack_20260718_175437_2026-07-27_16-52-42` @ `checkpoints/007000` died with `lerobot.processor.pipeline.ProcessorMigrationError: … Original error: Config file 'policy_preprocessor.json' not found`.

**Chain.** lerobot 0.6.0 moved normalization out of the policy into a serialized pre/post-processor pipeline. The trainer *does* save it — `scripts/lerobot_train.py:653-666` passes `preprocessor=` / `postprocessor=` into `save_checkpoint`. But the very next lines are:

```python
update_last_checkpoint(checkpoint_dir)
if cfg.save_checkpoint_to_hub:
    push_checkpoint_to_hub(checkpoint_dir, ...)
```

and that run's own config dump, recovered from its `log.jsonl`, reads:

```
save_checkpoint        = True
save_checkpoint_to_hub = False
push_to_hub            = True
```

So the complete checkpoint directory — processor configs, optimizer and scheduler state, `train_config.json` — was written to **container-local disk**, `push_checkpoint_to_hub` never ran, and the HF Job's filesystem was destroyed at job end. What reached the Hub is the separate `--policy.push_to_hub` output: the policy artifacts alone. The downloaded checkpoint contains exactly `config.json` and `model.safetensors`.

**Root cause in MakerMods Lab:** `save_checkpoint_to_hub` is **absent from `TrainingRequest` and from `build_training_command` entirely** — it never appears in `job.json`'s config, so it takes lerobot's default of `False`. That default is correct for a **local** run, where "save checkpoints to disk" means a disk that persists. On the **cloud** runner it means everything except the policy weights is written to a machine that is about to be deleted. `--save_checkpoint true` is passed, which makes the setting look handled when only half of it is.

**Symptoms this produces, all one bug:**
1. **Inference fails on any cloud-trained checkpoint** with `ProcessorMigrationError`, after the robot is already connected and the cameras opened (see the Inference list — the rollout subprocess dies post-`_prepare_robot`, so a deploy that could never work still energizes the arm first).
2. lerobot's suggested remedy is unusable as printed: it says `python src/lerobot/processor/migrate_policy_normalization.py`, a **source-tree path** that does not exist in a site-packages install. The script is really at `.venv/lib/python3.12/site-packages/lerobot/processor/migrate_policy_normalization.py`. Migrating there also writes into the **HF cache snapshot**, so it is undone by a re-download and fixes only the one step directory.
3. Any consumer expecting a complete checkpoint sees a partial one.

**Fix direction:** pass `--save_checkpoint_to_hub true` on the cloud runner (local runs should keep the current default). Weigh the cost — it uploads every periodic checkpoint rather than just the policy, so it interacts with `save_freq`.

*Open, needs a Hub call to settle:* the 2026-07-28 resume of this same repo **did** restore step and data order ("Resuming data order at epoch 43, sample 64"), which requires training state — so the `checkpoints/NNNNNN/` trees evidently hold more than the two files that cached locally. Listing the repo's files would establish exactly what is and isn't uploaded, and whether `push_checkpoint_to_hub` ran on some earlier attempt.

---

### MT38 · P1 — A cross-architecture fine-tune silently discards the base weights and trains a randomly-initialized policy that records itself as a fine-tune *(formerly NEW-14)*

**First identified:** 2026-07-29 — launch path fixed same day (`788565c` + a follow-up checkpoint guard); **the underlying silent-failure mode is NOT closed** (see "What remains").
**Fix:** [PR #42](https://github.com/makermods-robotics/makermodslab/pull/42) (open) · branch `fix/training`

Fine-tuning an imported `lerobot/smolvla_base` launched an **ACT** run. The base checkpoint's policy type was never propagated into the training form, and the form then *locked* the picker, so it could not be corrected by hand.

**Why it is P1 rather than a UI annoyance: it does not fail.** The written artifact is indistinguishable from a legitimate from-scratch run of the declared type, and the loss curve looks plausible. Measured against the pinned lerobot 0.6.0 with a real local smolvla checkpoint (safetensors header read only):

| | count |
|---|---|
| smolvla checkpoint tensors | 500 |
| fresh ACT policy parameters | 234 |
| `missing_keys` | 234 / 234 |
| `unexpected_keys` | 500 / 500 |
| **tensors actually loaded** | **0** |

The two key sets are **disjoint**, so inspecting `model.safetensors` can never settle whether a checkpoint is damaged. Detection must come from provenance or logs.

**Root cause is upstream, in two parts.** `policies/factory.py` `make_policy` passes `config=cfg` **explicitly**, so `from_pretrained` skips its `if config is None` branch and *never reads the checkpoint's own `config.json`*; and it never passes `strict`, which defaults to `False` (`policies/pretrained.py:174`). `policies/utils.py:90-93` logs the dropped keys at WARNING and continues.

**The silent window is exactly the cross-architecture case.** `safetensors.torch.load_model` delegates to `load_state_dict(..., strict=False)`, which **still raises on shape mismatch** (verified empirically). A *same*-architecture fine-tune onto a differently-shaped dataset therefore dies loudly at startup. Only disjoint key sets fail in silence.

**Detection signals, ranked** (implemented as pure functions in `makermodslab/finetune_audit.py`, currently unwired):

* **Provable** — the run's `train_config.json` `policy.type` vs. the architecture of the checkpoint named in `policy.pretrained_path`; a null `pretrained_path` (⇒ from scratch, defect cannot apply); `resume: true` (⇒ lerobot rebuilds from the checkpoint's own `train_config.json` via `--config_path`, so type and weights share one source and cannot disagree).
* **Corroborating** — `log.jsonl` containing both `Missing key(s)` and `Unexpected key(s)` (`LocalJobRunner` merges stderr via `stderr=subprocess.STDOUT`). Never invert it: a truncated log proves nothing.
* **Suspected only** — `job.json`'s own `config.policy_type`, which is MakerMods Lab bookkeeping that this very defect populated wrongly.

A null `policy_pretrained_path` in `job.json` is **proof**, not merely absence of evidence: `JobRegistry.start` mutates `config.policy_pretrained_path` and `_persist(record, force=True)`s it *before* the runner starts, so a launched fine-tune necessarily records one.

**Audit of this machine, 2026-07-29 — nothing damaged.** 18 `train_config.json` swept across the registry root, `makermodslab_models/` download caches, and `/Users/mokuroh54/models/`. Every `job.json`: `finetune_from_job_id: null` **and** `policy_pretrained_path: null`. No `--policy.pretrained_path` in any recorded launch argv (verified meaningful — 5 logs record a full argv and `--policy.type act` greps inside them). Zero `Missing key(s)`/`Unexpected key(s)` warnings anywhere. Two `resume: true` runs immune by the separate route above. **Near-miss:** `lerobot/smolvla_base` was imported 13:50, the fix landed 15:48 — the setup existed for under two hours and was never launched.

**What remains (why this entry stays open).** MakerMods Lab's guards are *validation before the spawn*, not prevention during the load — it shells out to `lerobot_train` and `make_policy` hardcodes the non-strict default with no CLI surface. Residual paths to a broken checkpoint:

1. **Unreadable checkpoint config** — `read_pretrained_policy_type` returns `None` on a missing/malformed `config.json`, a private or absent repo, **or no network**. The guard raises only on a *provable* contradiction, so "not established" proceeds. The offline case is not hypothetical.
2. **Anything bypassing `JobRegistry.start`** — driving lerobot directly gets no guard.
3. **The `register_imported` `"model"` placeholder** can enter the registry as a `policy_type` when a checkpoint's config is unreadable; both guards skip it by design.

**Structural fix, not attempted.** lerobot has a purpose-built idiom: `configs/train.py:167-170` `_resolve_pretrained_from_cli` reads `--policy.path`, derives the subclass from the checkpoint's own config, and `configs/parser.py:221` **hard-errors** if `--policy.type` is passed alongside it — making the mismatch impossible rather than checked. Cost: `--policy.path` loads the checkpoint's *entire* config (chunk_size, normalization mapping, optimizer presets) instead of the requested type's defaults, so it changes which hyperparameters a fine-tune inherits, and MakerMods Lab's policy overrides would have to route through `parser.get_cli_overrides("policy")`. Worth doing as its own change with its own testing.

**Upstream asks, in preference order:** (1) `make_policy` should compare `cfg.type` against the checkpoint's `config.json` `type` and raise on disagreement; (2) it should pass `strict=True` when `pretrained_path` was supplied for a fresh run, since a legitimate same-architecture load is strict-clean; (3) at minimum `log_model_loading_keys` should escalate to an error when `missing_keys` covers *every* model parameter — a 100%-miss load is never intentional.

**Related UI dead end (not fixed).** `TrainingConfigurator.tsx:537` still locks the picker unconditionally for any fine-tune seed, and `recordPolicyType()` returns `null` in two reachable cases (the `"model"` placeholder, or a failed `getJob`), leaving the picker disabled on the `act` default behind the label *"Set by the base skill…"*. With the new backend check that now yields a 400 the user cannot correct from the UI — better than silent corruption, but a dead end.

### MT39 · P2 — A user-requested stop is recorded as a failure, indistinguishable from a crash *(formerly NEW-15)*

**First identified:** 2026-07-29
**Fix:** [PR #42](https://github.com/makermods-robotics/makermodslab/pull/42) (open) · branch `fix/training`

`JobRegistry.stop(job_id)` (`jobs.py:1516`) calls `runner.stop()` — SIGTERM for the local runner (`jobs.py:473`) — and records **no intent anywhere**. Its own comment defers everything: *"The watchdog will finalise the record (state, ended_at, exit_code)."* The watchdog then has only the exit code to classify on (`jobs.py:1891`):

```python
record.state = "done" if rc == 0 else "failed"
...
record.error_message = reason or f"Subprocess exited with code {rc}"
```

SIGTERM ⇒ nonzero rc ⇒ **every deliberate stop lands as `failed`** behind a synthetic error message. Nothing distinguishes "the user pressed stop" from "the trainer crashed".

The semantically correct state already exists — `JobState = Literal["running", "done", "failed", "interrupted"]` (`jobs.py:49`) — but `"interrupted"` is only assigned at `jobs.py:1731` and `jobs.py:1757`, reconciling *stranded* records at startup (a record found `running` when no process exists after a restart). It is unreachable by intent:

| how a run ends | recorded as |
|---|---|
| finishes cleanly | `done` |
| trainer crashes | `failed` |
| **user presses stop** | **`failed`** ← wrong |
| makermodslab restarts mid-run | `interrupted` |

So the right state is reachable only by an accident of process lifetime.

**Observed impact.** `smolvla_makermods_eraser_stack_20260718_175437_2026-07-28_19-06-49` resumed from checkpoint 004000, trained to ~step 8250/15000, **successfully uploaded checkpoint 008000** (`[wrapper] uploaded checkpoint 008000` in its `log.jsonl`), and was then paused by the user. Its record reads `state: failed`, `Subprocess exited with code 1`, with no traceback anywhere in 333 unique log lines. The user reasonably concluded the model was damaged and asked which bug had broken their checkpoints. Nothing was wrong with it — the downloaded artifact at `~/models/smolvla_eraser_8k/checkpoints/008000` is complete (all 7 files, both processor pairs, 1.2 GB `model.safetensors`) and both its `config.json` and `train_config.json` agree on `smolvla`. **The run history lied, and the false signal cost real debugging time.**

**Why P2 and not P1:** nothing is destroyed or mis-trained, and the artifacts are intact. But it corrupts the one signal used to decide which runs are worth deploying — a genuinely failed run and a deliberately paused one are visually identical — and it actively misleads during incident triage, as it just did.

**Fix shape.** `stop()` must record intent before signalling; the watchdog must prefer `interrupted` over `failed` when a stop was requested. **The race is the hard part:** the subprocess can exit on its own — cleanly *or* by crashing — between the intent being recorded and the signal landing, or between the signal and the next watchdog tick. A run that genuinely finished before the stop took effect must not be relabelled, and a real crash that coincides with a stop must not be laundered into `interrupted`.

**Constraints for whoever fixes this.** Do NOT add a new `JobState` member without discussion — it widens into every persisted record on disk, the frontend's state rendering, and the status endpoints. Do NOT retroactively rewrite existing `job.json` records; the user's real training history is not ours to relabel. Check whether the frontend renders `interrupted` distinctly from `failed` before assuming the state change is visible.

**Open question — cloud runs.** `jobs.py:1075` notes hf_cloud jobs "always reattach and let the tail loop" finalise them, and `terminal_message` supplies runner-specific reasons (e.g. HF Jobs' "Job timeout"). Whether a user-initiated *cloud* cancel is distinguishable from a cloud failure at all is unresolved.

### MT40 · FIXED (pending review, 2026-08-07) — Weights & Biases support restored on both runners; a resume can no longer run wandb without an API key *(formerly NEW-16)*

**First identified:** 2026-07-29. **Fixed:** 2026-08-07, pending review. W&B support is back end-to-end, on a deliberately narrower footing than it had before.

#### The decisions that shaped the fix

**W&B is supported on BOTH runners.** Where the trainer executes changes only how the API key reaches it: a cloud job gets it forwarded into the pod as an HF Jobs secret, while a local trainer inherits `os.environ` and wandb can read `~/.netrc` itself, so nothing has to be forwarded. Nothing in `build_training_command` is runner-aware — these are lerobot's own trainer flags and they mean the same thing in a container and in a subprocess — and the frontend does not gate the W&B group on the Compute selection.

Consequently a cross-runner continuation (F7, either direction) keeps logging to the parent's W&B run: inheritance is runner-blind, because the run a continuation re-opens is identified by the checkpoint, not by where the trainer happens to execute.

**Resume state is INHERITED server-side, not form-decided — the constraint is lerobot's, not ours.** Verified against the installed lerobot v0.6.0: `WandBLogger.__init__` calls `wandb.init(resume="must")` with the run id taken from the checkpoint's `train_config.json` (`lerobot/common/wandb_utils.py:90-113`). A continuation therefore *always* re-opens the parent's W&B run; turning W&B **on** for a resume of a non-W&B parent is structurally impossible, and turning it **off** is the only other available lever. `JobRegistry.start` copies `wandb_enable` / `wandb_project` / `wandb_entity` off the parent record inside the resume block and ignores whatever the form sent. The form carries the parent's values too (`buildResumeSeed`), but strictly to render them read-only.

Consequently the old `wandb_run_id` request field was **not** restored: the id is checkpoint-inherited state, never a per-launch choice.

**The old install gate was NOT restored.** No `/system/wandb-extra*`, no `InstallManager("wandb")`, no `WandbInstallDialog.tsx`. It was dead code — `wandb` is a hard transitive dependency of the pinned lerobot's `training` extra, so it always answered "installed" while the question that actually blocks a run went unasked. It is replaced by `GET /system/wandb-credentials`, which reports (boolean only, never the key) whether a key is resolvable from `WANDB_API_KEY` or a `~/.netrc` entry for `api.wandb.ai`.

**`wandb_disable_artifact` now defaults to `true`** (backend field and frontend initial config): per-checkpoint model uploads to W&B are opt-in, not a side effect of enabling logging.

#### The original defect, and where it is now closed

`build_training_command`'s resume branch is a whitelist, so it emitted no `--wandb.*` at all and a resumed run inherited the **checkpoint's** wandb settings from `train_config.json`. A W&B-enabled parent therefore resumed with wandb **on** inside the container, while the `WANDB_API_KEY` secret was injected only when the *request* said `wandb_enable` — which a resume's form did not. Net effect: wandb active in-container with no API key, failing inside a paid GPU container rather than at submit time.

Closed at both ends:

- The resume branch now emits an explicit `--wandb.enable <bool>`, so the checkpoint's config never gets a silent vote.
- The credential check is a submit-time hard 400 that fires **before any side effect** — before a local subprocess is spawned, before `_ensure_dataset_on_hub` pushes a local-only dataset, and before the F7 `deferred_resume_upload` thread is spawned (that path returns 201 and uploads on a thread, so a later refusal would have become a failed job rather than a message on the button). It applies to **both runners**: a cloud job dies in a billed container without a key, and a local trainer is a non-tty subprocess in which `wandb.init` cannot prompt for a login, so it fails just as uselessly once the record already says `running`. Three layers: a fast half in `POST /jobs/training` for fresh runs, the authoritative runner-blind check in `JobRegistry.start` after inheritance, and a belt-and-braces repeat at the top of `HfCloudJobRunner.start` (above `_ensure_dataset_on_hub`, so a missing key cannot leave a freshly published dataset behind).
- The UI surfaces the missing-key state through the existing `PaneIssue` / `startDisabled` / `startTooltip` machinery, so Start is disabled with a named reason before the user clicks.

#### Also restored

`JobRecord.wandb_run_url`, the `JobRunner` protocol member, and the watchdog's pull-while-running / snapshot / terminal capture, plus the "View on W&B ↗" link in `TrainingJobDialog.tsx`. All three real runners scrape: `LocalJobRunner` from the subprocess's stdout, `TailingJobRunner` by replaying the log from offset 0 when it re-attaches after a `--reload` (the URL is printed once near the start, so tailing from EOF would lose it permanently), and `HfCloudJobRunner` from the SSE stream. Only `PreparingJobRunner` returns a constant `None`, because no trainer has run yet. **Corrected from the pre-removal implementation:** the URL comes from lerobot's own `logging.info("Track this run --> <coloured url>")`, *not* from wandb's `wandb: 🚀 View run at …` banner — lerobot sets `WANDB_SILENT=True` before importing wandb, which suppresses that banner entirely. The pattern therefore matches the URL's shape anywhere in the line so it survives the ANSI wrapper `termcolor` adds. A missing URL is normal (W&B off, offline/disabled mode, self-hosted W&B on another host) and is never treated as an error.

#### Two later additions (2026-08-08)

**Fresh runs default W&B ON when a key exists.** A UI-level default only: once `GET /system/wandb-credentials` reports a resolvable key, a FRESH run's form initialises `wandb_enable` to true. The backend field still defaults `false`, so non-UI/API callers keep opt-in semantics and the submit-time preflight still protects them. Because the probe answers asynchronously, the default is applied at most ONCE per mounted form and never overwrites a toggle the user has already touched — otherwise a late "yes, there's a key" would silently reverse a decision made while it was in flight. No key, or a probe that failed, leaves it off. A resume is excluded entirely: its W&B state is inherited, not defaulted.

**A resume now displays the parent's FULL W&B settings, and they are real.** `wandb_mode` / `wandb_notes` / `wandb_disable_artifact` were previously hidden on a resume on the grounds that the resume branch emits no flags for them. That reasoning was wrong: it emits none *because none are needed*. Verified against the installed lerobot v0.6.0 by reloading a real checkpoint `train_config.json` through the resume path (`TrainPipelineConfig.from_pretrained(config_path, cli_args=["--wandb.enable", …])`): mode, notes and disable_artifact all come back as the PARENT's values, each distinct from the `WandBConfig` class default, and `--wandb.enable` overrides only itself. `WandBLogger` reads exactly those fields (`notes=self.cfg.notes`, `mode=self.cfg.mode`, and `disable_artifact` in `log_policy`). So the parent's values are what a continuation genuinely runs with; they are now carried onto the resume seed and rendered read-only with the rest of the group. No new `--wandb.*` flags were added to the resume argv — the checkpoint already supplies them, and the argv tests pin that minimalism.

Legacy parents need no special case: a record predating the W&B fields deserialises with `wandb_enable` false, and the five detail fields only render when it is true — so nothing is fabricated for them. The one residual ambiguity is a PRE-REMOVAL record (before `017f385`) that had W&B on: `wandb_disable_artifact`'s default flipped `false` → `true` across the restore, so such a record would display "don't upload checkpoints" where it originally uploaded them. Absent-vs-explicit is not recoverable from the persisted JSON on the client, and the window is narrow.

#### Closed by that verification

An earlier draft of this entry flagged a possible gap: that a W&B resume depends on `wandb.run_id` being in the checkpoint's `train_config.json`, with lerobot's fallback (`get_wandb_run_id_from_filesystem`) globbing the NEW run's output dir and raising. Checking a real checkpoint written by this app settles it — the persisted `wandb` block carries `"run_id": "03zcjuoy"` alongside enable/project/entity/notes/mode/disable_artifact, so the fallback is never reached on a resume from a MakerMods Lab-produced checkpoint. Not a gap.

**The concrete defect.** `build_training_command`'s resume branch emits no `--wandb.*` flags (it is a whitelist that returns at `train.py:283`), so a resumed run inherits the **checkpoint's** wandb settings from `train_config.json`. A wandb-enabled parent therefore resumes with wandb **on** inside the container. But `hf_cloud.py:584` injects the `WANDB_API_KEY` secret only when `config.wandb_enable` is true, and — until `c28dd9c` — the form defaulted `wandb_enable` to `false` on a resume. Net effect on a cloud resume of a wandb-enabled parent: **wandb active in-container with no API key**, failing inside a paid GPU container rather than at submit time.

Note `c28dd9c` (resume settings inheritance) deliberately did **not** prefill `wandb_enable`, so this remains open.

**Fix shape and its tradeoff.** Prefilling `wandb_enable` from the parent record closes it, but converts a missing local key into a hard 400 at `hf_cloud.py:588` on submit. That is the better failure (fail at submit, not 20 minutes into a billed run), but it *is* a submit-time behaviour change and should be a deliberate choice, not incidental.

**The broader ask.** The user's framing was that "wandb support needs to be back" — treat this as a **restore-wandb-support** item rather than only the resume-secret gap. Before fixing, audit the whole path end to end: the `wandb_*` fields in `TrainingRequest`, whether the fresh-run branch emits the flags it should, the `WANDB_API_KEY` secret plumbing in `runners/hf_cloud.py`, whether `wandb_run_url` is captured and surfaced (`JobRecord.wandb_run_url` exists and `runner.wandb_run_url()` is polled in the watchdog), and what the UI exposes. It is likely more than one gap.

### MT41 · P1 — The cloud checkpoint watcher uploads mid-save snapshots and marks them done forever: Hub checkpoints can be permanently incomplete (unresumable) *(formerly NEW-17)*

**First identified:** 2026-07-31 — hit live: `checkpoints/005000` of a SmolVLA cloud run resumed into
`FileNotFoundError: .../training_state/optimizer_state.safetensors`, `[wrapper] trainer exited with rc=1`.
**Fix:** [PR #33](https://github.com/makermods-robotics/makermodslab/pull/33) (draft) · branch `fix/mt9-cloud-checkpoint-watcher-partial-upload` · [PR #42](https://github.com/makermods-robotics/makermodslab/pull/42) (open) · branch `fix/training`

The in-container sidecar (`runners/hf_cloud.py` `_scan_and_upload`, 15s poll) treats a checkpoint directory as
ready the moment `pretrained_model/config.json` exists — the FIRST artifact of lerobot's multi-second,
multi-file save sequence (weights first, `training_state/` last, with the large `optimizer_state.safetensors`
landing latest). A poll tick inside the save window uploads a partial directory, and `seen.add(entry.name)`
then permanently retires it — the files that finish writing seconds later are never re-uploaded. Larger
policies (SmolVLA) widen the race window; the failure is silent until a resume attempt.

Compounding gap: the resume guards check only `training_state/training_step.json` (Hub) / `training_state/`
dir existence (local, `jobs.py` ~731-822), so a partial checkpoint passes validation and dies inside the
trainer instead of at the API with a clear message.

Fix direction (three small parts): (1) readiness = full expected file set including
`training_state/optimizer_state.safetensors`, plus a two-poll stability check on the directory; (2) only add
to `seen` after a verified-complete upload — re-scan and re-upload otherwise; (3) resume guards verify the
full training_state set and refuse with a named remedy (resume an earlier checkpoint / fine-tune weights-only).

### MT42 · P1 — Switching a local Continue to Hugging Face Cloud silently restarts from step 0 wearing a resume label *(formerly NEW-18)*

**First identified:** 2026-07-31 (surfaced during the resume-locked-fields UI work).

`localize_config_for_cloud` (`runners/hf_cloud.py:182`) clears `config.config_path` when there is no
`resume_from_hub_repo` — which is exactly what a LOCAL Continue switched to the cloud runner looks like.
`build_training_command`'s resume branch then fails its `request.config_path` guard (`train.py:252`) and falls
through to the FRESH-RUN argv while still carrying `--resume true`: a full from-scratch run (no checkpoint
state loaded) that presents as a resume in the UI and the job record. Silent, and billed at GPU rates.

Fix direction: guard the runner toggle on a local resume (block the local→cloud switch with an explanation,
since a local checkpoint isn't reachable from the container), or make the switch upload/translate the local
checkpoint first (bigger). Related: MT40 (the W&B resume gap in the same area — additional evidence from
2026-07-31: `ResumeSeed` doesn't carry the parent's `wandb_enable`, so a wandb-enabled parent resumes on cloud
with the toggle false, no secret injected, and the container's checkpoint config still enabling wandb —
expected to fail at wandb init; conversely enabling the toggle on resume can 400-block the launch while never
enabling logging, because the resume argv passes no --wandb.* flags).

## Design gaps and hardening risks

These are source-backed weaknesses; the audit did not establish a single unambiguous intended behavior or safely reproduce an end-to-end failure. *(Re-verification: all still present on `main`; the cited helpers are unchanged in substance.)*

### Process identity and child-process ownership

**First identified:** 2026-07-14

- Local restart recovery tests only whether a PID exists; PID reuse could attach to an unrelated process.
- Local stop signals the recorded parent PID rather than an owned process group; LeRobot/data-loader children may survive.

Evidence: `makermodslab/jobs.py:131-139`, `:330-385`, `:470-485`.

### Dependency-environment mutation

**First identified:** 2026-07-14

- Separate install managers can mutate the same Python environment concurrently.
- Policy extras are expressed as `lerobot[extra]` rather than tying the extra installation to the repo's pinned LeRobot Git revision.

Evidence: `makermodslab/utils/system.py:113-203`, `:204-265`.

### Artifact provenance and compatibility

**First identified:** 2026-07-14

- Job records do not retain immutable dataset/model Hub commit SHAs, so an external repo update can change what a later retry/continuation resolves.
- Fine-tune preflight does not establish that the selected source policy's inputs/outputs are compatible with the selected dataset before launching compute.
- Hub checkpoint listing converts broad API failures into an empty list, conflating no checkpoints with auth/network failure.

Evidence: `makermodslab/jobs.py:65-108`, `:682-734`, `:736-800`, `:811-848`.

### Publication contract

**First identified:** 2026-07-14

- Training and upload controls do not consistently state that implicit dataset publication and model upload are public operations. *(See MT27.)*
- Custom model-upload metadata can omit the generic `lerobot` discovery tag. *(See MT32.)*

Evidence: `makermodslab/runners/hf_cloud.py:641-671`, `makermodslab/models.py:886-949`, `makermodslab/utils/config.py:89-106`.

## Fix clusters (verification note — still valid on `main`, with re-verification additions)

- **Entries 2, 3, 12 — Hub-checkpoint listing semantics.** One "list the root policy alongside the tree + disambiguate lineage/step refs" change addresses all three. Add **MT30** (the UI symptom of 12).
- **Entries 4, 5, 6 — resume-form lock.** One resume-mode read-only/target-lock pass (plus deriving W&B secret forwarding from the restored config) addresses all three.
- **Entries 10, 11 — runner finalization contract.** A runner that cannot know (or was told to stop) reporting a definitive `returncode()` the watchdog converts into an irreversible terminal state. **Re-verification: absorb MT25 and MT29** — all four are the same contract violation.
- **Entries 16, 17 — cloud tail loop.** Both in `HfCloudJobRunner`'s tail path (missing resume offset; `_lines_processed` reset on reattach); both `andrew` fixes are absent from `main` and should be re-implemented together.

## Context: contradictions and corrections to the re-verification brief

1. **`main` is not a superset of the audited tree.** `518ca56` is not an ancestor of `main`; MT16/MT17 fixes exist only on `andrew`, MT24's on neither. Re-verifying "against `main`" reopens three entries.
2. **MT11's "no longer monitors" is imprecise.** `server.py:1364-1373` keeps a dismissed-but-active Hub job visible; the real gap is *actionability* — see MT25.
3. **MT5 → PARTIALLY FIXED** (`policyLocked` closes the policy-label leg).
4. **MT13's second clause survived the UI rewrite** — the timeout field moved to `AdvancedCard.tsx:294-322`, `timeoutInvalid` still absent from `startDisabled`.
5. **`andrew`-vs-`main` argv divergence (`--eval_freq` vs `--env_eval_freq`)** is material to any fix port and belongs to the coordinator. *(Per CURRENT, `main`'s HEAD has since adopted `env_eval_freq`.)*

### MT43 · Fixed in working tree — The training form's optimizer controls were display-only: the preset overwrites every `--optimizer.*` flag

**First identified:** 2026-08-04

The form always emitted `--use_policy_training_preset true` alongside `--optimizer.type/lr/weight_decay/grad_clip_norm` — but lerobot's `TrainPipelineConfig.validate()` unconditionally replaces the whole optimizer with `active_cfg.get_optimizer_preset()` when the preset flag is on (and resume is false), so every `--optimizer.*` flag was dead on arrival. Every run trained with the policy preset (AdamW for act/smolvla/pi0, Adam for diffusion/vqbet/tdmpc) regardless of what the user set; the editable "Adam" dropdown was actively misleading. The preset builds from POLICY-config fields, so the working knobs are `--policy.optimizer_lr` / `--policy.optimizer_weight_decay` / `--policy.optimizer_grad_clip_norm` — with per-policy availability (weight_decay: all except tdmpc; grad_clip_norm: only smolvla/pi0/pi0_fast; a flag the policy config lacks fails at draccus parse).

**Fix (in working tree, 2026-08-04):** `_policy_optimizer_flags` in train.py routes the three knobs through the policy config gated by a per-policy capability table; no `--optimizer.*` flag is emitted anywhere; AdvancedCard shows the policy's actual preset type as a non-editable display and hides unsupported knobs. Resume branch unchanged (optimizer state comes from the checkpoint; lerobot skips the preset overwrite on resume).

**Evidence:** `makermodslab/train.py` (`_policy_optimizer_flags`, the optimizer block comments), lerobot `configs/train.py:249-253`, policy configs' `get_optimizer_preset` (ACT/SmolVLA → AdamWConfig; diffusion/vqbet/tdmpc → AdamConfig).

### MT44 · Open — Fine-tune has no feature-space preflight: a mismatched dataset silently trains garbage (SmolVLA dof case) or dies late and unreadably (ACT)

**First identified:** 2026-08-04 (investigated same day against the pinned lerobot; per-class verdicts experiment- or source-proven)

MakerMods launches fine-tunes as `--policy.type <T> --policy.pretrained_path <dir>`; lerobot's checkpoint-config load keys off `--policy.path` only, so on our path the checkpoint's config.json is never read for architecture — the policy is built FROM THE NEW DATASET's features and weights load with `strict=False`. lerobot's `validate_visual_features_consistency` is a tautology on this path (compares the dataset to itself). Consequences by mismatch class: state/action-dim mismatch is LOUD-but-late for ACT (raw size-mismatch stack after the dataset download) and **SILENT-WRONG for SmolVLA/pi0/pi05** (dofs padded to 32 — a 6-dof checkpoint loads cleanly into a 12-dof run and trains garbage recorded as a fine-tune); renamed cameras (the bimanual `left_` prefix case) load silently; missing/extra cameras load silently (architecturally sound for ACT's shared backbone, but silently changes the sensor suite); resolution changes are handled sanely; normalization stats are consistent-by-construction today (dataset-derived features + dataset stats), latent hazard only if `--policy.path`/`--rename_map` is ever adopted.

**Guard design (decided from the verdicts):** extend `read_pretrained_policy_type` to also return `input_features`/`output_features` (same config.json fetch, zero extra network); new `_check_pretrained_feature_space` beside `_check_pretrained_policy_type` in `JobRegistry.start`. Matrix: state/action-dim mismatch → hard 400 (reuse `rollout._arm_count_mismatch` phrasing); renamed cameras → hard 400 (never intentional in this UI); missing/extra camera → warn + explicit confirm (legitimate but must be chosen); resolution → allow + note (changes ACT token count/VRAM); stats → allow, one-line comment guarding the future path switch. Keep the "silent when unreadable" discipline. Note: switching the launch to `--policy.path` would revive lerobot's own camera guard but make the checkpoint's stale `output_features` authoritative — not a free win; recorded here so the tradeoff isn't relitigated blind. Inference already has the full preflight (`get_policy_config_summary`, `_arm_count_mismatch`, DeployPanel camera binding) — this brings training to parity.

**Adjacent, noted:** `missing_checkpoint_files` doesn't require `policy_preprocessor.json`, but `make_pre_post_processors` loads it on every fine-tune — a flat import lacking it passes the completeness check and fails in the trainer. Possibly deliberate (documented as a resume completeness test); verify before filing separately.

**Hole found during live verification (2026-08-05, user-spotted):** the rename refusal requires EQUAL camera counts (`jobs.py:1395`), so a dataset whose camera set is fully DISJOINT from the checkpoint's but differently sized — e.g. 1 cam named `left` vs a `{wrist, front}` checkpoint — falls into the benign count-change branch and is warn-allowed, despite being the "different rig" case the rename rule exists to catch (proven live: fixture `eraser_place_1cam_renamed_test`, guard accepted with drop-everything/from-scratch warnings). Fix: refuse when the intersection is empty (both sides non-empty), any counts — and the placeholder exemption becomes load-bearing there, since `smolvla_base`'s `camera1/2/3` vs any real rig is also disjoint-with-unequal-counts and must stay allowed (the canonical base fine-tune). Partial overlap keeps the count-change warn/confirm path.

**Phase-2 design note — hard vs soft camera requirements (2026-08-05):** hardness is provenance, not architecture (shared backbones tolerate any camera set). Signal hierarchy: (1) all-placeholder checkpoint names (`_is_placeholder_camera_set`) → soft, extend the exemption from the rename branch to the count branch; (2) registry lineage — a checkpoint from a run we know has an exact parent dataset, diff against the true rig instead of inferring from names (imported checkpoints fall back to signal 1); (3) the phase-2 confirm dialog supplies the intent bit when neither resolves it. Target matrix: dim mismatch = always hard · disjoint real-named sets = hard · rename/count vs placeholder base = soft, proceed · count change with overlap, real names = confirm.

**Evidence:** lerobot `configs/parser.py:35,173-176`, `configs/train.py:159-170`, `policies/factory.py:580-615,651`, `policies/pretrained.py:174`, `policies/utils.py:83-93,226-249`, `modeling_smolvla.py:487,492,583-587`; makermodslab `train.py:354`, `jobs.py:982,1011-1099,1697,2128-2167`, `rollout.py:768-805`.

### MT45 · Open — Resume lets steps extend past a fixed-horizon LR schedule: the added steps train at floor LR with no warning

**First identified:** 2026-08-05 (spotted while deciding whether `steps` should stay editable on the resume form — it should, and does; this is the unguarded edge of that choice)

The resume form deliberately keeps `steps` editable (the resume branch passes `--steps`; raising it is the point of a continuation), but nothing bounds the raise against the parent's LR-schedule horizon. On resume lerobot skips the preset rebuild (`use_policy_training_preset and not self.resume`, `configs/train.py:249-253`) and takes optimizer/scheduler config from the checkpoint's `train_config.json` — so a SmolVLA/pi0 run's `CosineDecayWithWarmupSchedulerConfig` keeps the parent's `scheduler_decay_steps` horizon no matter what the new `--steps` says. Steps past that horizon train at the decay floor (`scheduler_decay_lr`, 2.5e-6 for SmolVLA): loss flattens, reads as convergence, run is barely learning. This is the partial-case sibling of the done-source resume refusal (which handles the total case, where the ENTIRE continuation would sit at floor). Per-policy reality is uneven, which is why this is a design pass and not a one-liner:

- **smolvla / pi0 / pi0_fast** — fixed horizon `scheduler_decay_steps` (a config field, independent of `steps` even on fresh runs), inherited from the checkpoint on resume. The affected case.
- **act / tdmpc** — `get_scheduler_preset() → None`, flat LR; extension is genuinely fine, no warning wanted.
- **diffusion / vqbet** — scheduler configs exist but are built at runtime (`cfg.scheduler.build(optimizer, cfg.steps)` takes the NEW steps; diffusion's warmup-only `DiffuserSchedulerConfig`, vqbet's vqvae-step split), then restored state applies. Verify actual stretch-vs-floor behavior at implementation time before deciding whether they warn.

**Guard design (proposed):** warn + confirm, never refuse — resuming up to the parent's horizon is fully legitimate, and even a past-horizon extension might be wanted. Condition: resume with `steps > parent's schedule horizon` on a fixed-horizon policy. Needs a third per-policy capability table (scheduler kind + horizon field, alongside the two optimizer-knob tables in train.py/AdvancedCard) and the warn/confirm surface — **batch with MT44 phase 2**, which builds exactly that surface for camera-count/resolution soft mismatches. Same fetch discipline as MT44: horizon comes from the checkpoint's `train_config.json`, already read on the resume path.

**Evidence:** lerobot `configs/train.py:249-253`, `policies/smolvla/configuration_smolvla.py` / `pi0/configuration_pi0.py` (`get_scheduler_preset` → `CosineDecayWithWarmupSchedulerConfig(num_decay_steps=self.scheduler_decay_steps)`), `act/configuration_act.py` + `tdmpc/configuration_tdmpc.py` (→ `None`); makermodslab `train.py:300,316` (resume branch passes `--steps`), `frontend/.../EssentialsCard.tsx` (steps-stays-editable rationale), `JobCard.tsx:350-364` (done-source refusal — the total-case twin).

### MT46 · Open — No WebSocket heartbeat: a half-open socket stays silently "connected", losing jobs_changed signals

**First identified:** 2026-08-05 (during the "new job needs a page reload to appear" diagnosis; the reload symptom itself was fixed by refetch-on-start + unknown-id self-heal in JobsDataContext/ActivityStrip — this entry is the remaining substrate)

The `/ws/joint-data` server loop never pings and `useJobsChangedSignal` has no client keepalive, so a half-open TCP connection (sleep/wake, network blip) keeps `readyState === OPEN` indefinitely without delivering messages and without firing `onclose` — the client never reconnects and silently misses `jobs_changed` (fire-and-forget, no replay: `server.py:390-400` no-ops unless `is_running and active_connections`, and `_send_to_all_connections` drops a connection on any send failure). The jobs surfaces now self-heal via the 1 Hz `job_progress` unknown-id reconcile, but any OTHER consumer of the socket (joint-data streaming itself) still trusts a connection that may be dead. Fix shape: server-side periodic ping (or app-level heartbeat message) + client-side idle-timeout reconnect in the WS hooks. Low urgency post-self-heal; batch with the next WS/broadcast touch.

**Evidence:** makermodslab `server.py:390-400` (`notify_jobs_changed`, `_send_to_all_connections`), `frontend/src/hooks/useJobsChangedSignal.ts` (no heartbeat/reconnect), `JobsDataContext.tsx`/`ActivityStrip.tsx` (post-fix self-heal that masks this for jobs only).

### MT47 · Fixed (pending review, 2026-08-07) — The HF Cloud log tail can go permanently silent mid-run: progress and charts freeze while the job trains on to completion

**First identified:** 2026-08-07, from a live run the user reported as "the logs didn't show up". **Root cause introduced:** `a08ab035` — *fix(hf-cloud): decouple job liveness from log stream*, Nicolas Rabault, 2026-05-12 (authored on the pre-rename `lelab/` tree; an ancestor of `main`).

**The symptom, as seen.** Cloud run `2_blue_cube_orange_box_2026-08-07_21-20-36` (t4-small, ACT, 10,000 steps) finished normally — `state: done`, `exit_code: 0`, three checkpoints on the Hub including step 10,000 — while the app showed **3,650 / 10,000 (36.5%)** with a live-looking **00:53:05 remaining**, and the loss/LR charts stopped at ~3.6k. Verified on disk: the run spanned 21:20:36 → 23:46:24, but its `log.jsonl` holds 275 lines ending abruptly mid-tqdm-burst at **21:56:17** — nothing for the last 110 minutes. `metrics` froze on the last frame it ever parsed (`current_step 3650`, `eta_seconds 3185` = 00:53:05), which is exactly what the UI rendered.

Three surfaces of one record disagreed because they have three different sources: progress/ETA and the charts come from the log tail (`JobRecord.metrics`, and `read_metrics_history` re-parsing `log.jsonl`), while the checkpoint dropdown lists the **Hub** repo (`_list_cloud_cached` → `_list_hub_checkpoints`) and liveness comes from `_status_poll_loop`'s `inspect_job`. Only the log-derived pair was wrong.

**Two independent defects, both in `_tail_loop`, both reproduced deterministically** (`tests/test_runners_hf_cloud.py`, fake `fetch_job_logs`, no network):

1. **Reconnect accounting that assumes a full replay.** `a08ab035` de-duplicated a reconnect's replayed prefix with a *per-connection* counter compared against a *cross-connection* total: `seen = 0` at the top of each connection, `if seen <= self._lines_processed: continue`. That is only correct if every reconnect replays the whole log from line 1. When a reconnect replays **less** than that — a tail window, or nothing at all because `follow=True` resumes from "now" — `seen` never catches up to `_lines_processed` and **every subsequent line is skipped, silently, forever**. Repro: 6 lines, drop, reconnect replaying 2 + 3 genuinely new ⇒ the 3 new lines are lost permanently. The arithmetic fits the live case: 275 lines processed, and the remaining 110 minutes at the observed ~25 s/line cadence ≈ 264 more lines — fewer than the 275 deficit, so nothing would ever have appeared again.
2. **No timeout on a silent connection.** `fetch_job_logs(follow=True)` can block inside a single read indefinitely (half-open socket: NAT eviction, laptop sleep, proxy idle timeout) — no line, no `StopIteration`, no exception. A plain `for` over that iterator cannot even observe `_stop_event`, because the loop body never runs again. Repro: the tail thread sits in the generator forever, no reconnect is attempted, and nothing anywhere notices.

Either defect alone produces exactly the observed on-disk state, and they cannot be told apart after the fact from the log — so both are fixed. The same freeze signature appears in older records that predate the W&B work entirely, confirming it is long-standing and not a regression from it: `smolvla_makermods_eraser_place_2026-07-30_00…` (`done`, frozen at 12,650/15,000, log 91.9 min short of the end) and `smolvla_…_unblurry_real` on 07-31 (`failed`, 5,500/20,000, 67.4 min short). Both postdate `a08ab035` (2026-05-12), as expected.

**Fix (working tree, 2026-08-07).**

- **Content-based replay de-dupe** replaces the positional counter: `_is_replayed()` keeps a bounded window (`_TAIL_DEDUPE_WINDOW = 1000`) of recently-emitted lines. Correct whether a reconnect replays everything, some of it, or none. The tradeoff is deliberately inverted — a line repeated inside the window is dropped, and a replay longer than the window is re-emitted, both cosmetic — because the scheme it replaces failed the other way and went mute.
- **A silence timeout** (`_TAIL_SILENCE_TIMEOUT_S = 600`) via `_iter_job_logs()`: the blocking iteration runs on a reader thread and is consumed through a queue with a timeout, so a stalled connection raises and reconnects through the path that already existed. The stalled reader is abandoned rather than joined — it is stuck in the read we gave up on — but it is a daemon, and this is not a new leak: before the fix, a stalled read stranded the entire tail loop permanently. 600 s is generous on purpose (a QUEUED/BUILDING job is legitimately silent for minutes) and a spurious reconnect is now cheap.
- **Mitigation, independent of the tail:** `_settle_terminal_metrics()` runs when the watchdog finalises a run — a `done` run's `current_step` snaps to `total_steps` (only when a target is known; never for `failed`/`interrupted`, where the last frame is the truth and inventing progress would also poison the resume flow), and `eta_seconds` is cleared for every terminal state. The frontend gates the ETA label on `training_active` as well, so a finished run cannot render a countdown.

**Known gap:** the fix is forward-looking. Records already on disk in this frozen state — including the run above — keep their stale `current_step`, because `_settle_terminal_metrics` only runs on the live terminal transition. The frontend ETA gate does clean up their phantom countdown. Repairing the persisted step would mean either settling terminal records on registry load (mutates history) or deriving it at `list`/`get` time; deferred as a deliberate product call rather than done in passing.

**Evidence:** `makermodslab/runners/hf_cloud.py` (`_is_replayed`, `_iter_job_logs`, `_tail_loop`, the two new constants), `makermodslab/jobs.py` (`_settle_terminal_metrics` and its call in the watchdog's finalise block), `frontend/src/components/training/monitoring/MonitoringStats.tsx` (ETA gate), `tests/test_runners_hf_cloud.py` (five MT47 tests: the two repros, the no-premature-reconnect guard, and the de-dupe window), `tests/test_jobs.py` (four `_settle_terminal_metrics` tests). Introduced by `a08ab035`; the live record is `~/.cache/huggingface/lerobot/outputs/train/2_blue_cube_orange_box_2026-08-07_21-20-36/`.
