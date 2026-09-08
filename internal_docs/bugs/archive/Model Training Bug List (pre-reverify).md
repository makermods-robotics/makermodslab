# Model Training Bug List

This ledger tracks confirmed defects in MakerMods Lab's model-training pathways: configuration and preflight, local and Hugging Face cloud execution, resume and fine-tune, monitoring, checkpoints, model publication, deletion, and restart recovery. It is separate from the neutral current-behavior inventory in `complete functionalities/Model Training Interface Pathways.md`.

Audited on July 14, 2026, against MakerMods Lab commit `518ca56`.

## Status key

- **Open:** confirmed defect with no validated fix.
- **In progress:** implementation has started but is not fully validated.
- **Fixed:** implementation and regression coverage are complete.
- **Design gap:** desired safety or product behavior is not yet represented by a complete implementation contract.
- **Ruled out:** verified not a defect — do not re-flag.

## Severity key

- **P0 — Critical:** breaks a primary training pathway or leaves remote compute running without correct local ownership.
- **P1 — High:** can train from the wrong state, lose or misrepresent an artifact, break recovery, or delete a resource in active use.
- **P2 — Moderate:** causes a misleading failure, incomplete monitoring, avoidable operational risk, or a recoverable inconsistency.
- **P3 — Low:** limited hardening or usability defect with no demonstrated loss of training state.

## Confirmed findings

### 1. Ruled out — Hugging Face cloud jobs never reach a terminal state locally

**Severity:** P0 — Critical

**Verdict:** NOT-A-BUG — inspect_job returns status.stage as a plain uncoerced string so the terminal comparison works — the original audit probed the enum in isolation (huggingface_hub/_jobs_api.py:92-98 JobStatus is an uncoerced dataclass, :265-266 builds it from raw JSON; makermodslab/runners/hf_cloud.py:745-746 therefore matches).

**Priority:** none — ruled out; recommended hardening: compare with 'stage in _TERMINAL_STAGES' (survives future enum coercion); a live cloud job to terminal is the definitive close-out.

**Pathway:** Cloud start -> remote execution -> status polling -> terminal job/model handoff

**Current behavior**

The cloud runner converts the Hub status stage with `str(stage).upper()` and compares it with bare strings such as `COMPLETED`. The installed Hub client's enum string is shaped like `JobStage.COMPLETED`, so the comparison never matches. `is_running()` remains true and the registry watchdog cannot finalize the job.

**Impact**

- A completed, canceled, or failed cloud job remains shown as running.
- Final model discovery, continuation actions, deletion, and terminal UI behavior remain blocked.
- Remote-job ownership is misleading after completion.

**Evidence**

- `makermodslab/runners/hf_cloud.py:426`
- `makermodslab/runners/hf_cloud.py:730-776`

### 2. Open — Fine-tuning a selected Hub checkpoint discards the selected revision

**Severity:** P1 — High

**Verdict:** CONFIRMED — `_resolve_finetune_pretrained_path` returns `chosen.ref.split("@", 1)[0]` (makermodslab/jobs.py:730-731), dropping the selected step; `--policy.pretrained_path=<repo>` then loads repo-root weights (makermodslab/train.py:288-289).

**Priority:** P1 — common fine-tune flow silently starts from repository-root weights instead of the user-selected step, with no signal to the user (verified 2026-07-14 against the current working tree)

**Pathway:** Job checkpoint selection -> fine-tune request -> pretrained model resolution -> trainer start

**Current behavior**

Hub checkpoints are represented as `repo@checkpoints/<step>`, but fine-tune resolution strips everything after `@` and returns only the repository ID. Selecting a specific checkpoint therefore does not reliably supply those checkpoint weights to LeRobot.

**Impact**

- Fine-tuning can silently start from repository-root weights instead of the user-selected step.
- The UI and persisted continuation metadata describe a different starting point from the actual trainer input.

**Evidence**

- `makermodslab/jobs.py:682-734`
- `makermodslab/train.py:272-283`

### 3. Open — The final Hub policy is hidden when periodic checkpoints exist

**Severity:** P1 — High

**Verdict:** CONFIRMED — `_list_imported_hub` returns the root config.json only when the checkpoints tree is empty (makermodslab/jobs.py:783-788), and `_hub_checkpoints_from_files` matches only `checkpoints/<step>/pretrained_model/config.json` (makermodslab/jobs.py:735, 738-756), so the final root policy is never listed alongside periodic checkpoints.

**Priority:** P1 — a completed cloud model can appear unusable from its job card, or default to an older periodic checkpoint instead of the final policy (verified 2026-07-14 against the current working tree)

**Pathway:** Cloud completion/import -> Hub model discovery -> model/checkpoint selection -> inference or continuation

**Current behavior**

When a Hub repository contains a `checkpoints/<step>/pretrained_model` tree, imported-model listing returns that tree and omits the repository-root policy. LeRobot publishes the final policy at the root at the end of training, which can be newer than the last periodic checkpoint.

Cloud runs with checkpoint saving disabled can also publish a root policy while their tracked job exposes no selectable checkpoints.

**Impact**

- MakerMods Lab can default to an older periodic checkpoint rather than the final trained policy.
- A valid final cloud model can appear unusable from its tracked job card.
- Inference, download, and fine-tune pathways can disagree about which artifact represents the completed run.

**Evidence**

- `makermodslab/jobs.py:736-800`
- `makermodslab/models.py:150-178`
- `makermodslab/train.py:258-315`

### 4. Open — Switching a cloud resume request to Local launches an invalid resume

**Severity:** P1 — High

**Verdict:** CONFIRMED — TargetCard allows switching to local unconditionally (frontend/src/components/training/config/TargetCard.tsx:57-60); a cloud source sets only `resume_from_hub_repo`, never `config_path` (makermodslab/jobs.py:1163-1174), so `build_training_command` falls through its resume branch (makermodslab/train.py:248) and emits `--resume true` with no `--config_path` (makermodslab/train.py:319).

**Priority:** P1 — a valid-looking resume form deterministically launches a doomed local run (verified 2026-07-14 against the current working tree)

**Pathway:** Resume action -> resume form -> target switch -> local runner command construction

**Current behavior**

The resume form permits changing the inherited cloud target to Local. Registry resolution assigns `resume_from_hub_repo` only for cloud execution; it does not create a host-local `config_path` for this cross-target case. The local command then receives `--resume true` without the configuration path LeRobot requires.

**Impact**

- A valid-looking resume form launches a predictably failing local process.
- The job record is created before the trainer reports the missing resume configuration.

**Evidence**

- `frontend/src/pages/Training.tsx:198-240`
- `frontend/src/components/training/config/TargetCard.tsx:83-145`
- `makermodslab/jobs.py:1157-1184`
- `makermodslab/train.py:248-270`

### 5. Open — Editable resume controls do not match inherited trainer state

**Severity:** P1 — High

**Verdict:** CONFIRMED — the config cards stay fully editable on resume (ConfigurationTab passes no lock, frontend/src/components/training/ConfigurationTab.tsx:25-37) while the resume branch restores those fields from config_path (makermodslab/train.py:248-270); the record name and cloud dependency extra still use the edited policy_type (makermodslab/jobs.py:1192; makermodslab/runners/hf_cloud.py:555).

**Priority:** P2 — only bites when the user edits inherited fields; the default resume flow is unaffected (verified 2026-07-14 against the current working tree)

**Pathway:** Resume action -> configuration editing -> dependency preflight -> runner launch -> job history

**Current behavior**

The interface says dataset, policy, batch size, and optimizer are inherited, but leaves policy, target, optimizer, batch, W&B, and related controls editable. The resume command intentionally restores most of those values from `train_config.json`, while registry naming and cloud dependency selection still use the newly edited request.

**Impact**

- Job records can label a different policy from the one actually resumed.
- The cloud wrapper can install the wrong policy dependency extra.
- Users can spend time adjusting controls that the trainer ignores.

**Evidence**

- `frontend/src/pages/Training.tsx:539-553`
- `frontend/src/components/training/config/EssentialsCard.tsx:91-130`
- `makermodslab/train.py:244-270`
- `makermodslab/jobs.py:1122-1243`

### 6. Open — Cloud W&B resume can omit the required secret

**Severity:** P1 — High

**Verdict:** CONFIRMED — the resume form initializes `wandb_enable: false` (frontend/src/pages/Training.tsx:233); the resume branch emits no `--wandb.*` flags so the checkpoint's restored W&B config governs (makermodslab/train.py:248-270), but `WANDB_API_KEY` is forwarded only when the new request's wandb_enable is true (makermodslab/runners/hf_cloud.py:576-585).

**Priority:** P1 — no workaround in the form; a previously working W&B-enabled cloud run loses tracking or fails on resume (verified 2026-07-14 against the current working tree)

**Pathway:** W&B-enabled cloud checkpoint -> resume form -> cloud submission -> restored trainer config

**Current behavior**

The resume form initializes W&B as disabled. The resumed LeRobot configuration can restore W&B as enabled from the checkpoint, but the cloud runner forwards `WANDB_API_KEY` only when the new request's `wandb_enable` value is true.

**Impact**

- A previously working W&B-enabled cloud run can fail or lose tracking on resume.
- Secret forwarding is decided from UI draft state rather than the effective restored training configuration.

**Evidence**

- `frontend/src/pages/Training.tsx:224-235`
- `makermodslab/train.py:248-270`
- `makermodslab/runners/hf_cloud.py:573-592`

### 7. Open — Job deletion bypasses the active-inference model guard

**Severity:** P1 — High

**Verdict:** CONFIRMED — `delete_job` calls `job_registry.delete` directly (makermodslab/server.py:1664-1668) and `JobRegistry.delete` guards only on running state (makermodslab/jobs.py:1540-1552); no `_model_in_use` check like the model-browser path performs (makermodslab/models.py:1052-1054, guard at 959-984).

**Priority:** P1 — checkpoint files can be removed under a live inference subprocess; the safety guarantee depends on which UI card initiates deletion (verified 2026-07-14 against the current working tree)

**Pathway:** Completed training job -> start inference from checkpoint -> delete job from monitoring

**Current behavior**

The model-browser deletion pathway checks whether a checkpoint is being used by inference. The generic job deletion endpoint calls `job_registry.delete()` directly and performs no equivalent check.

**Impact**

- A user can remove checkpoint files while an inference subprocess is reading them.
- The same model has different safety guarantees depending on which UI card initiates deletion.

**Evidence**

- `makermodslab/models.py:962-987`
- `makermodslab/models.py:1040-1063`
- `makermodslab/server.py:1664-1678`
- `makermodslab/jobs.py:1545-1558`

### 8. Open — Config-only and partial models are classified as usable

**Severity:** P1 — High

**Verdict:** CONFIRMED — `_resolve_pretrained_dir` accepts a bare root config.json with no weights required (makermodslab/models.py:334-335), `_list_local_checkpoints` requires only pretrained_model/config.json (makermodslab/jobs.py:562-563), and `_cleanup_partial_model` preserves any dir that passes that probe (makermodslab/models.py:1107-1112).

**Priority:** P2 — the doc's P1 was downgraded for narrow trigger conditions (requires an interrupted download or a genuinely weightless source); failure surfaces later at inference/fine-tune (verified 2026-07-14 against the current working tree)

**Pathway:** Model download/import -> local validation -> listing -> inference or fine-tune

**Current behavior**

Model validation treats the presence of `config.json` as sufficient. It does not require weights, processors, or other checkpoint artifacts. Failed-download cleanup also preserves a partial directory once that config file exists.

**Impact**

- Interrupted downloads can remain listed as local or both.
- Import accepts incomplete checkpoints.
- Later inference or fine-tuning fails after the user has already selected an apparently usable model.

**Evidence**

- `makermodslab/models.py:150-178`
- `makermodslab/models.py:318-337`
- `makermodslab/models.py:1090-1117`
- `makermodslab/models.py:1131-1145`

### 9. Open — Cloud checkpoint upload can permanently publish a partial checkpoint

**Severity:** P1 — High

**Verdict:** CONFIRMED — the wrapper uploads once `pretrained_model/config.json` exists and marks the step seen permanently (makermodslab/runners/hf_cloud.py:340-352, final scan skips seen steps at 382-386); note the training_state resume mitigation: `_resolve_cloud_resume` verifies `training_state/training_step.json` on the Hub (makermodslab/jobs.py:632-636), so resume from a partial checkpoint fails cleanly rather than silently — inference/fine-tune from the partial artifact remain exposed.

**Priority:** P1 — permanent partial publication on a paid run; mitigated for resume only (verified 2026-07-14 against the current working tree)

**Pathway:** Remote checkpoint save -> wrapper scan -> Hub upload -> resume/inference

**Current behavior**

The wrapper considers a checkpoint ready as soon as `pretrained_model/config.json` exists. It uploads the directory while the trainer may still be writing weights or `training_state`, then records the step in `seen`. Later scans, including the final scan, skip that step permanently.

**Impact**

- The Hub can contain a checkpoint missing weights or optimizer/training-step state.
- Cloud resume can become impossible even though MakerMods Lab reports that the checkpoint was uploaded.
- Inference or fine-tune can select a corrupt remote artifact.

**Evidence**

- `makermodslab/runners/hf_cloud.py:328-355`
- `makermodslab/runners/hf_cloud.py:358-386`
- `makermodslab/jobs.py:582-639`

**Observed in production (2026-07-19, `redesign`)**

- Confirmed live, not just by code reading. Cloud run `makermods/smolvla_makermods_eraser_stack_20260718_175437_2026-07-18_20-39-35` (SmolVLA, 10k steps, ended failed) left an **intermittent** pattern on the Hub — steps 3000 and 5000 have `pretrained_model/` but **no `training_state/`**, while steps 1000, 2000, and 4000 are complete. An intermittent subset (not all/none) is the signature of the 15s watcher landing inside the `pretrained_model` → `training_state` write window; SmolVLA's large `pretrained_model` save widens that window.
- User-visible failure: resuming the latest checkpoint (5000) hit `_resolve_cloud_resume`'s clean-fail guard — `Start training failed: Checkpoint at step 5000 has no optimizer/step state (training_state/) on the Hub, so it can't be resumed`. Resume from step 4000 (complete) then succeeded.
- Recommended fix (unimplemented): gate the upload on a completeness marker — the **last** file LeRobot writes is `training_state/scheduler_state.json` (order inside `save_training_state`: `training_step.json` → rng → `optimizer_state.safetensors` → `scheduler_state.json`). Skip any step whose `training_state/scheduler_state.json` is absent, and do **not** add the step to `seen` until a complete upload succeeds, so partial checkpoints are retried on the next scan / the final scan instead of sealed.

### 10. Open — A reattached local training failure is recorded as success

**Severity:** P1 — High

**Verdict:** CONFIRMED — `TailingJobRunner.returncode()` returns 0 once the PID is gone because it cannot reap a process from another session (makermodslab/jobs.py:477-484); the watchdog finalizes rc==0 as "done" (makermodslab/jobs.py:1749).

**Priority:** P1 — a trainer crash after a MakerMods Lab restart is presented as a completed model with downstream actions offered (verified 2026-07-14 against the current working tree)

**Pathway:** Local training -> MakerMods Lab restart -> PID/log reattachment -> trainer exit -> terminal state

**Current behavior**

After reattachment, MakerMods Lab cannot reap the trainer process's real exit status. Once the PID disappears, `TailingJobRunner.returncode()` always returns zero and the watchdog finalizes the run as done.

**Impact**

- A trainer crash after a MakerMods Lab restart is presented as successful completion.
- Model cards and downstream actions can be offered for a failed or incomplete run.

**Evidence**

- `makermodslab/jobs.py:429-485`
- `makermodslab/jobs.py:1565-1624`

### 11. Open — Cloud stop reports cancellation before remote cancellation succeeds

**Severity:** P1 — High

**Verdict:** CONFIRMED — `stop()` pre-sets CANCELED before calling `cancel_job` and swallows every cancellation exception at info level (makermodslab/runners/hf_cloud.py:755-765); `is_running()` then returns False (767-771), so the runner is finalized and stops monitoring.

**Priority:** P1 — a failed cancel leaves the paid GPU burning while MakerMods Lab reports the job stopped and no longer monitors it (verified 2026-07-14 against the current working tree)

**Pathway:** Running cloud job -> Stop -> local terminal transition -> remote cancellation

**Current behavior**

`stop()` sets the local terminal state to `CANCELED` before calling the Hub cancellation API, then suppresses every cancellation exception.

**Impact**

- A paid remote job can continue after MakerMods Lab reports it stopped.
- A job that completed during the race can be recorded as canceled/failed rather than done.
- The local runner is finalized and no longer monitors the still-running remote job.

**Evidence**

- `makermodslab/runners/hf_cloud.py:755-776`
- `makermodslab/jobs.py:1417-1498`

### 12. Open — Cloud resume lineage cannot distinguish parent and child checkpoints

**Severity:** P1 — High

**Verdict:** CONFIRMED — cloud resume reuses the parent's output repo (`config.policy_repo_id = config.resume_from_hub_repo`, makermodslab/runners/hf_cloud.py:536), so parent and child records enumerate the same shared checkpoint tree via `_list_hub_checkpoints` (makermodslab/jobs.py:791-799).

**Priority:** P2 — the doc's P1 was downgraded for narrow trigger conditions; provenance ambiguity without artifact loss (verified 2026-07-14 against the current working tree)

**Pathway:** Cloud checkpoint -> resume child -> checkpoint listing -> later continuation

**Current behavior**

Cloud resume publishes the child into the same model repository as its parent. Both job records therefore enumerate the same repository checkpoint tree. The frontend combines child and ancestor checkpoints using numeric step values.

**Impact**

- A child can appear to own checkpoints before producing any new artifact.
- Parent and child provenance is ambiguous.
- A later "latest" resume can select an old higher-numbered parent checkpoint rather than the intended child trajectory.

**Evidence**

- `makermodslab/runners/hf_cloud.py:518-547`
- `makermodslab/jobs.py:588-639`
- `frontend/src/components/jobs/JobCard.tsx:116-211`
- `frontend/src/components/jobs/JobsSection.tsx:353-375`

### 13. Open — Invalid training request bodies return HTTP 500

**Severity:** P2 — Moderate

**Verdict:** CONFIRMED — `from_legacy` runs before/outside the endpoint's try block (makermodslab/server.py:1079-1080; try begins at 1146), so a pydantic ValidationError escapes uncaught as a 500 — the `except ValueError` at 1155 cannot reach it.

**Priority:** P2 — internal-server-error response instead of an actionable 4xx (verified 2026-07-14 against the current working tree)

**Pathway:** Direct API or invalid UI state -> create training job -> schema validation -> error response

**Current behavior**

The endpoint reads raw JSON and invokes `StartTrainingBody.from_legacy()` inside the route instead of using FastAPI body validation. Pydantic validation errors are not mapped to a client error response.

**Impact**

- Missing or malformed configuration returns an internal-server-error response instead of an actionable 4xx response.
- Invalid cloud timeout input can reach this path because the timeout field's local error state is not part of the parent form's start-disable calculation.

**Evidence**

- `makermodslab/server.py:1077-1081`
- `frontend/src/components/training/config/TargetCard.tsx:42-58`
- `frontend/src/components/training/config/TargetCard.tsx:150-190`
- `frontend/src/pages/Training.tsx:503-535`

### 14. Open — Core numeric training parameters accept zero and negative values

**Severity:** P2 — Moderate

**Verdict:** CONFIRMED — `TrainingRequest` carries a field validator only for `hf_job_timeout` (makermodslab/train.py:203-219); steps/batch_size/log_freq/save_freq have no positive-range validation (makermodslab/train.py:115-127), and the server only warns on freq > steps (makermodslab/server.py:1086-1098).

**Priority:** P2 — local jobs fail only after process creation; cloud jobs incur startup cost before the deterministic config error (verified 2026-07-14 against the current working tree)

**Pathway:** Training configuration -> request validation -> local/cloud trainer start

**Current behavior**

Steps, batch size, log frequency, and save frequency have no positive-range validation in the request model. Direct API requests can therefore create invalid jobs, and zero frequencies can reach LeRobot's frequency arithmetic.

**Impact**

- Local jobs fail only after process creation.
- Cloud jobs can incur startup time or cost before failing on a deterministic configuration error.

**Evidence**

- `makermodslab/train.py:104-217`
- `frontend/src/components/training/config/EssentialsCard.tsx:53-160`
- `frontend/src/components/training/config/AdvancedCard.tsx:180-310`

### 15. Open — Terminal transition can omit final logs and checkpoints until reload

**Severity:** P2 — Moderate

**Verdict:** CONFIRMED — the checkpoint poll's interval returns early once state != running with no final tick (frontend/src/pages/Training.tsx:719-723), and the log poll fetches logs only while `next.state === "running"` (frontend/src/pages/Training.tsx:738-758), so the transition-observing tick skips the final drain.

**Priority:** P2 — the final traceback/success line and a last-moment checkpoint are missing until manual reload (verified 2026-07-14 against the current working tree)

**Pathway:** Running job -> final trainer output/checkpoint -> terminal state poll -> monitoring UI

**Current behavior**

The frontend stops both log and checkpoint polling as soon as its current job state is no longer `running`. It does not perform a final drain or checkpoint refresh after observing the terminal state.

**Impact**

- The final traceback, wrapper exit message, or success line can be absent from the live view.
- A checkpoint created or uploaded after the preceding checkpoint poll can be missing until the page reloads.

**Evidence**

- `frontend/src/pages/Training.tsx:691-729`
- `frontend/src/pages/Training.tsx:730-760`

### 16. In progress — Cloud resume progress is not rebased to the inherited step

**Severity:** P2 — Moderate

**Verdict:** CONFIRMED — the cloud `_tail_loop` calls `parse_metrics_into(stripped, self._metrics)` with no resume offset (makermodslab/runners/hf_cloud.py:702), unlike the local/tailing runners (makermodslab/jobs.py:403, 527-529); the history endpoint does rebase (makermodslab/jobs.py:1456), so live and reconstructed views disagree.

**Fix in progress (2026-07-14):** `HfCloudJobRunner` now accepts `resume_total` and `log_freq` (constructed from `_resume_total_steps(config)` / `config.log_freq` at both the start and reattach sites in makermodslab/jobs.py), and its `_tail_loop` passes them into `parse_metrics_into` exactly like the local runners — so live cloud-resume progress is rebased to the global step and agrees with `read_metrics_history`. `parse_metrics_into` also gained `log_freq` so the rounded INFO step ("8K") snaps to the exact log_freq multiple. Covered by `test_tail_loop_parse_applies_resume_offset` and `test_tail_loop_applies_log_freq_for_exact_step` (tests/test_runners_hf_cloud.py). Not yet validated end-to-end against a live cloud resume.

**Priority:** P2 — cosmetic monitoring inconsistency (verified 2026-07-14 against the current working tree)

**Pathway:** Cloud resume -> remote logs -> metric parsing -> progress display

**Current behavior**

The local runner passes a resume-total offset into metric parsing. The cloud log tail calls the parser without that offset, even though the resumed run inherits an existing training step.

**Impact**

- Live cloud-resume progress reports the remaining-run step window rather than the global training step.
- Metrics can jump or disagree after reload and historical reconstruction.

**Evidence**

- `makermodslab/jobs.py:375-405`
- `makermodslab/jobs.py:528-530`
- `makermodslab/runners/hf_cloud.py:678-708`

### 17. In progress — Cloud reattachment duplicates persistent logs

**Severity:** P2 — Moderate

**Verdict:** CONFIRMED — `reattach` opens the log file in append mode (makermodslab/runners/hf_cloud.py:609) with `_lines_processed` still 0 (init at 494), so the SSE stream's replayed historical prefix passes the `seen <= self._lines_processed` dedup (696) and is re-appended in full.

**Fix in progress (2026-07-14):** the persisted cloud log now collapses consecutive tqdm bar-render lines to the last of each run (`_persist_collapsed`), so its line count no longer tracks the SSE index — seeding `_lines_processed` from the file would undercount and re-append most metric lines. Instead the physical write is gated by a separate `_reattach_disk_skip`, seeded on reattach from `_count_existing_log_lines` (existing persisted lines, excluding host-injected `[upload]` lines). `_lines_processed` is deliberately left at 0 on reattach so parse+queue still reprocess the history to rebuild live state; only the append to disk is suppressed for lines already present, which the deterministic re-collapse of the replayed stream reproduces exactly. Covered by `test_reattach_does_not_duplicate_persisted_lines`, `test_tail_loop_collapses_consecutive_bar_lines`, `test_tail_loop_flushes_trailing_bar`, and `test_count_existing_log_lines_excludes_host_upload_lines` (tests/test_runners_hf_cloud.py). Not yet validated end-to-end against a live cloud restart.

**Priority:** P2 — full log history duplicated on disk per server restart; metric reconstruction grows progressively noisier (verified 2026-07-14 against the current working tree)

**Pathway:** Cloud training -> MakerMods Lab restart -> remote-log reattachment -> local log persistence

**Current behavior**

Reattachment opens the existing log file in append mode, but `_lines_processed` starts at zero. The Hub log stream replays its historical prefix, which is appended again before reconnect de-duplication has any prior count.

**Impact**

- Every server restart can duplicate the full cloud log history on disk.
- Monitoring and metric reconstruction become progressively noisier and more expensive.

**Evidence**

- `makermodslab/runners/hf_cloud.py:494`
- `makermodslab/runners/hf_cloud.py:599-620`
- `makermodslab/runners/hf_cloud.py:678-710`

### 18. Open — Manual model upload can partially succeed while reporting total failure

**Severity:** P2 — Moderate

**Verdict:** CONFIRMED — a `metadata_update` failure raises `_upload_model_error` after the public repo and weights already exist (makermodslab/models.py:915-941); cache invalidation runs only after tagging succeeds (makermodslab/models.py:943); the successful repo id is never written back to the source JobRecord.

**Priority:** P2 — the UI reports failure after a public mutation succeeded, prompting retries and duplicate entries (verified 2026-07-14 against the current working tree)

**Pathway:** Completed local model -> Upload to Hub -> weights upload -> metadata tagging -> UI result

**Current behavior**

The implementation creates the public repository and uploads weights before updating metadata. If metadata update fails, it raises an upload error even though the model repository already exists. Cache invalidation occurs only after tagging succeeds.

The successful repository ID is also not persisted into the source `JobRecord`, so the local and Hub copies may continue to appear as unrelated models.

**Impact**

- The UI says upload failed after an external public mutation has succeeded.
- Immediate model listing can remain stale.
- The user can retry and create duplicate or confusing model entries.

**Evidence**

- `makermodslab/models.py:886-949`
- `makermodslab/models.py:950-954`
- `makermodslab/jobs.py:65-108`

### 19. Open — The frontend blocks cloud runs on host-only training dependencies

**Severity:** P2 — Moderate

**Verdict:** CONFIRMED-WITH-CORRECTIONS — the policy-extra preflight is already target-gated (skipped for cloud, frontend/src/pages/Training.tsx:424); only the accelerate gate blocks cloud: `trainingExtraAvailable === false` renders TrainingExtraGate for the whole page regardless of target (frontend/src/pages/Training.tsx:486-494; probe at makermodslab/utils/system.py:203-211). No W&B host gate on Start exists.

**Priority:** P2 — a valid cloud pathway is blocked by an unrelated host dependency (verified 2026-07-14 against the current working tree)

**Pathway:** Cloud configuration -> host dependency preflight -> cloud submission

**Current behavior**

Training preflight checks and installs host `accelerate`, W&B, and policy extras before target-specific execution. A cloud job can therefore be blocked by the workstation environment even though the cloud wrapper constructs and installs its own pinned environment.

**Impact**

- A valid cloud pathway is unavailable until unrelated local dependencies are installed.
- The UI may prompt environment mutations that are unnecessary for the selected target.

**Evidence**

- `frontend/src/pages/Training.tsx:255-285`
- `frontend/src/pages/Training.tsx:472-497`
- `makermodslab/runners/hf_cloud.py:204-327`
- `makermodslab/utils/system.py:203-265`

### 20. Open — All training HTTP 409 responses are presented as a local mutex conflict

**Severity:** P2 — Moderate

**Verdict:** CONFIRMED — `startTrainingJob` rewrites every 409 to the mutex message (frontend/src/lib/jobsApi.ts:190-195) while the backend also 409s `DatasetNotOnHubError` (makermodslab/server.py:1150-1154).

**Priority:** P2 — mitigated by the deliberate 400 rerouting of the offline-local guard (makermodslab/server.py:1132-1145) and the UI's upload-then-train flow; wrong remediation remains for non-UI callers and future 409s (verified 2026-07-14 against the current working tree)

**Pathway:** Training start -> backend domain conflict -> frontend error message

**Current behavior**

The frontend rewrites every 409 response to "Another training is already running." The backend also uses 409 for other conditions, including a cloud target whose selected dataset is not available on the Hub.

**Impact**

- Users receive the wrong remediation for dataset publication and other domain conflicts.
- A cloud start failure appears to be local job contention even when no job is running.

**Evidence**

- `frontend/src/lib/jobsApi.ts:180-195`
- `makermodslab/server.py:1100-1160`
- `makermodslab/jobs.py:1210-1243`

### 21. Open — Legacy job migration moves every directory under the legacy root

**Severity:** P2 — Moderate

**Verdict:** CONFIRMED — `_migrate_legacy_cwd_jobs` moves every directory under the legacy root before best-effort metadata rewrite, never requiring a job.json (makermodslab/jobs.py:1004-1024).

**Priority:** P2 — the doc's P1 was downgraded for narrow trigger conditions (one-shot, fires only on first boot under the new layout with a populated legacy dir) (verified 2026-07-14 against the current working tree)

**Pathway:** MakerMods Lab startup -> legacy output discovery -> migration to the current training root

**Current behavior**

Migration treats every directory under the legacy `outputs/train` root as a job and moves it before best-effort metadata rewriting. It does not first require a valid MakerMods Lab job record.

**Impact**

- Unrelated user directories under that location can be relocated.
- A malformed or non-job folder becomes part of training-state migration without an explicit user action.

**Evidence**

- `makermodslab/jobs.py:991-1051`

### 22. Open — Checkpoint ZIP download is unbounded in memory and can race active writes

**Severity:** P2 — Moderate

**Verdict:** CONFIRMED — the full ZIP is built in `io.BytesIO` then copied again via `getvalue()` (makermodslab/server.py:1605-1612, 1625); the only guard is `runner != "local"` (makermodslab/server.py:1590) — no running-state check, so files can change during traversal.

**Priority:** P2 — roughly 2x archive-size peak memory; a download during active training can capture an inconsistent snapshot (verified 2026-07-14 against the current working tree)

**Pathway:** Running/completed local job -> checkpoint download -> ZIP response

**Current behavior**

The endpoint builds the complete uncompressed ZIP in `BytesIO` and then copies it again with `getvalue()`. Downloads are allowed while training is active, so files can change during traversal and archiving.

**Impact**

- Large policies can create substantial peak server memory use.
- A downloaded archive can contain an inconsistent checkpoint snapshot.

**Evidence**

- `makermodslab/server.py:1576-1628`
- `frontend/src/components/jobs/JobCard.tsx:212-290`

### 23. Open — The job-registry lock spans remote network work and runner submission

**Severity:** P2 — Moderate

**Verdict:** CONFIRMED — `JobRegistry.start` holds `self._lock` across the whole body including `runner.start()` (makermodslab/jobs.py:1216), which for cloud runs performs the synchronous dataset upload and job submission (makermodslab/runners/hf_cloud.py:516-593); list/get/stop/delete take the same lock.

**Priority:** P2 — unrelated job operations block for the duration of network work; a slow Hub op freezes the job interface (verified 2026-07-14 against the current working tree)

**Pathway:** Cloud start/resume/fine-tune -> Hub resolution/upload/submission -> concurrent job operations

**Current behavior**

`JobRegistry.start()` holds the registry's global lock while resolving remote resume/fine-tune state and while `runner.start()` can inspect the Hub, upload a dataset, and submit a cloud job.

**Impact**

- Listing, reading, stopping, and deleting unrelated jobs can block for the duration of network calls or a dataset upload.
- A slow Hub operation can make the entire job interface appear unresponsive.

**Evidence**

- `makermodslab/jobs.py:1100-1243`
- `makermodslab/runners/hf_cloud.py:518-597`
- `makermodslab/runners/hf_cloud.py:641-676`

### 24. In progress — Resume re-passes `--policy.tags` in a form draccus rejects on the config-file path

**Severity:** P1 — High

**Verdict:** CONFIRMED (observed in production 2026-07-19, `redesign`) — the resume branch emits `_policy_hub_flags()`, which builds the tag list as the unquoted token `[makermods,openbooth,MakerMods Lab]` (makermodslab/train.py:79, called at :267). A fresh run parses fine because draccus builds the config from argparse, which tolerates that bracket form. Resume does **not**: it loads the config via `from_pretrained` → `draccus.parse(config_file, args=cli_args)`, whose list-**override** decoder (`draccus/parsers/decoding.py:_decode_list`) rejects the unquoted string. The trainer dies at parse time before any training happens.

**Priority:** P1 — every cloud resume with `push_to_hub` (the default for cloud runs) crashes deterministically at launch, after the checkpoint has already been downloaded. Fix implemented this session but uncommitted and not yet validated by a completed live resume.

**Pathway:** Resume action -> runner command construction -> trainer config parse (`from_pretrained`)

**Current behavior**

`build_training_command`'s resume branch re-passes `--policy.private false --policy.tags [makermods,openbooth,MakerMods Lab]`. The tags value is a single unquoted bracketed token. On the resume/`from_pretrained` code path draccus decodes CLI values as typed overrides and cannot coerce that token into `list[str]`, raising `DecodingError: `policy.tags`` and exiting rc=1.

**Impact**

- Cloud resume is impossible for any run pushing to the Hub — the trainer exits immediately after downloading the (correct) checkpoint.
- The failure surfaces as an opaque draccus traceback, not a MakerMods Lab-level message.

**Observed (2026-07-19, `redesign`)**

- Resuming step 4000 of `makermods/smolvla_makermods_eraser_stack_...20-39-35` downloaded the checkpoint, launched the trainer, and died: `draccus.utils.DecodingError: `policy.tags`: ... The given value='[makermods,openbooth,MakerMods Lab]' is not of a valid input for a list type`.
- The checkpoint's own `train_config.json` already carries `tags`, `push_to_hub`, `repo_id`, and `private`, so re-passing them on resume is redundant as well as breaking.

**Fix (implemented, uncommitted on `redesign`)**

- Resume branch no longer emits `--policy.tags`; it keeps `--policy.private false` only (makermodslab/train.py:265). Tags persist via the checkpoint's `train_config.json`, so Hub pushes stay tagged.
- `tests/test_train.py::test_resume_push_to_hub_public_without_tags` updated to assert `--policy.tags` is absent on resume (was previously asserting the buggy value). Full `tests/test_train.py` green (40 passed).
- Follow-up (not done): the **fresh-run** branch still emits the same unquoted bracket form (works today via the argparse path, but latently fragile); consider a draccus-safe encoding for both paths.

**Evidence**

- `makermodslab/train.py:69-80` (`_policy_hub_flags`)
- `makermodslab/train.py:259-270` (resume branch)
- `lerobot/configs/train.py` `from_pretrained` → `draccus.parse(config_file, args=cli_args)`

## Design gaps and hardening risks

These are source-backed weaknesses, but the audit did not establish a single unambiguous intended behavior or safely reproduce an end-to-end failure. They should be specified before implementation.

### Process identity and child-process ownership

- Local restart recovery tests only whether a PID exists; it does not retain a process start time or executable identity. PID reuse could attach to an unrelated process.
- Local stop signals the recorded parent PID rather than an owned process group. LeRobot or data-loader children may survive.

Evidence: `makermodslab/jobs.py:131-139`, `makermodslab/jobs.py:330-385`, `makermodslab/jobs.py:470-485`.

### Dependency-environment mutation

- Separate install managers can mutate the same Python environment concurrently.
- Policy extras are expressed as `lerobot[extra]` rather than explicitly tying the extra installation to the repository's pinned LeRobot Git revision.

Evidence: `makermodslab/utils/system.py:113-203`, `makermodslab/utils/system.py:204-265`.

### Artifact provenance and compatibility

- Job records do not retain immutable dataset and model Hub commit SHAs, so an external repository update can change what a later retry or continuation resolves.
- Fine-tune preflight does not establish that the selected source policy's inputs and outputs are compatible with the selected dataset before launching paid or local compute.
- Hub checkpoint listing converts broad API failures into an empty checkpoint list, conflating no checkpoints with authentication or network failure.

Evidence: `makermodslab/jobs.py:65-108`, `makermodslab/jobs.py:682-734`, `makermodslab/jobs.py:736-800`, `makermodslab/jobs.py:811-848`.

### Publication contract

- Training and upload controls do not consistently state that implicit dataset publication and model upload are public operations.
- Custom model-upload metadata can omit the generic `lerobot` discovery tag, so repositories with nonstandard names may not remain automatically discoverable after their local copy is removed.

Evidence: `makermodslab/runners/hf_cloud.py:641-671`, `makermodslab/models.py:886-949`, `makermodslab/utils/config.py:89-106`.

## Validation performed

- Ran the focused backend suite:
  - `tests/test_train.py`
  - `tests/test_jobs.py`
  - `tests/test_runners_hf_cloud.py`
  - `tests/test_models.py`
  - `tests/test_training_preflight.py`
- Result: **209 passed**, with 5 warnings.
- Used temporary or mocked probes to verify:
  - the installed Hub `JobStage.COMPLETED` string shape does not match `_TERMINAL_STAGES`;
  - a selected Hub step resolves to the repository root for fine-tuning;
  - a cloud-to-local resume lacks `config_path`;
  - reattached local jobs return zero when their PID disappears;
  - a directory containing only `config.json` is classified as a usable local model;
  - Hub checkpoint listing omits a repository-root policy when a checkpoint tree exists;
  - cloud reattachment appends a replayed log prefix to existing persistent logs.
- Inspected frontend request construction, polling, continuation controls, and error mapping against the corresponding backend schemas and runners.
- No real hardware, credentials, user datasets, local caches, external jobs, or Hub repositories were read or mutated by validation.

## Coverage gaps

The focused suite does not currently exercise these important boundaries:

- Real `huggingface_hub.JobStage` enum values through the cloud status poller.
- Natural, failed, canceled, and cancellation-error terminal transitions for a cloud job.
- Fine-tune command construction from an explicitly selected Hub checkpoint.
- Cross-target resume attempts and policy/dependency inheritance during resume.
- W&B-enabled cloud resume secret forwarding.
- Server restart followed by a detached local trainer failure.
- Server restart followed by cloud-log replay.
- A checkpoint being deleted through the job endpoint while inference uses it.
- Partial model downloads/imports missing weights or processors.
- A checkpoint directory still being written while the cloud uploader scans it.
- Final log draining and checkpoint refresh after the frontend observes a terminal state.
- Partial-success Hub publication where weights upload succeeds but metadata update fails.
- Registry responsiveness while cloud submission or dataset upload is blocked.

End-to-end behavior with actual Hugging Face Jobs, real Hub repository revisions, browser timing, multi-process trainer children, large checkpoint downloads, and crash recovery remains outside this safe audit boundary.

## Fix clusters (verification note, 2026-07-14)

These groupings change how fixes should be scoped — each cluster shares one root mechanism, so fixing entries individually risks three partial patches to the same code path.

- **Entries 2, 3, 12 — Hub-checkpoint listing semantics.** All stem from the listing/resolution helpers in `makermodslab/jobs.py` (`_hub_checkpoints_from_files`, `_list_imported_hub`, `_resolve_finetune_pretrained_path`): the tree scan never surfaces the repo-root policy, and hub refs collapse to the bare repo id. A single "list the root policy alongside the tree + disambiguate lineage/step refs" change addresses all three.
- **Entries 4, 5, 6 — resume-form lock.** The resume form renders `ConfigurationTab`/`TargetCard`/`EssentialsCard` fully editable with no resume lock, while `build_training_command`'s resume branch (makermodslab/train.py:248-270) ignores most edited fields. One resume-mode read-only/target-lock pass (plus deriving W&B secret forwarding from the restored config rather than the draft) addresses all three.
- **Entries 10, 11 — runner finalization contract; entries 16, 17 — cloud tail loop.** 10 and 11 share the `returncode()`/terminal-state contract between runners and the registry watchdog (a runner that cannot know the real outcome should not report a definitive one). 16 and 17 are both in `HfCloudJobRunner`'s tail path (missing resume offset; `_lines_processed` reset on reattach) and are naturally fixed together.
