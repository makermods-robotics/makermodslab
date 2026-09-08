# Model Training Bug List — RE-VERIFIED against `main`

Re-audit of `internal_docs/bugs/Model Training Bug List.md` (audited 2026-07-14 against commit
`518ca56`, i.e. the **`andrew`** branch) against **`main` @ `c7d9f27`** ("Merge pull request #4 from
makermods-robotics/redesign"), on 2026-07-23.

Read-only audit: no code, git state, hardware, credentials, Hub calls, HF Jobs submissions, or
training processes were touched. Every verdict below is static (source reading + `git diff`).

## Branch-topology warning — read this first

`518ca56` is **NOT an ancestor of `main`** (`git merge-base --is-ancestor 518ca56 HEAD` → false).
The original list was verified against `andrew`; `main` is the `redesign` line. The two diverge by
~9.4k insertions / 13.4k deletions across 132 files, and the divergence is *not* a superset in
either direction:

- **Fixes recorded as "in progress" in the original list exist only on `andrew` and are ABSENT from
  `main`.** Confirmed by `git show andrew:makermodslab/runners/hf_cloud.py` — `andrew` has
  `_persist_collapsed` (:812), `_reattach_disk_skip` (:558), and a `resume_total`/`log_freq`
  constructor (:512). `main`'s `HfCloudJobRunner.__init__` (hf_cloud.py:462-494) has none of them.
  **Entries 16 and 17 are therefore OPEN on `main`, not "in progress".**
- The MT24 fix ("implemented, uncommitted on `redesign`") **did not land on `main` either** —
  `train.py:265-267` still emits `_policy_hub_flags()` on the resume branch.
- `andrew` also renames `eval_freq` → `env_eval_freq` (`git diff main andrew -- makermodslab/train.py`);
  `main` still emits `--eval_freq`. Whichever is right depends on the lerobot pin — out of scope
  here, but it means **the two branches build different trainer argv** and cannot be assumed
  interchangeable when porting fixes.
- The frontend was **restructured** on `main`: `Training.tsx` went 900+ → 105 lines and the form
  moved to `components/training/TrainingConfigurator.tsx` + `components/studio/TrainPanel.tsx`; the
  monitor page became `components/training/TrainingJobDialog.tsx`. **Every frontend `file:line` in
  the original list is stale.** All frontend evidence below is re-derived from `main`.

## Status key

Same keys as the source list. Verdicts added here:

- **STILL REAL** — the defect reproduces in `main`'s code as written.
- **FIXED** — the mechanism is gone on `main`.
- **PARTIALLY FIXED** — one leg of the finding is closed, the rest reproduces.
- **NOT APPLICABLE — SUPERSEDED** — the code the finding described no longer exists in that shape.
- **CANNOT VERIFY STATICALLY** — depends on remote HF Jobs / Hub behavior this audit may not exercise.

---

## Part 1 — Verdict table

| # | Original title | Sev (orig) | Verdict on `main` | Key evidence on `main` |
|---|---|---|---|---|
| 1 | Cloud jobs never reach a terminal state locally | P0 (ruled out) | **Ruled out — unchanged** (hardening still unimplemented) | `hf_cloud.py:426`, `:744-749` |
| 2 | Fine-tune discards the selected Hub revision | P1 | **STILL REAL** | `jobs.py:727-732` (`chosen.ref.split("@", 1)[0]`), `train.py:288-289` |
| 3 | Final Hub policy hidden when periodic checkpoints exist | P1 | **STILL REAL** | `jobs.py:774-789`, `:792-800`, `:736-757` |
| 4 | Cloud resume switched to Local launches an invalid resume | P1 | **STILL REAL** (new file:line) | `TargetCard.tsx:42-50`, `jobs.py:1168-1183`, `train.py:248`, `:319` |
| 5 | Editable resume controls don't match inherited trainer state | P1→P2 | **PARTIALLY FIXED** — policy is now locked; target/batch/optimizer/W&B still editable | `TrainingConfigurator.tsx:537`, `EssentialsCard.tsx:77`, `ConfigurationTab.tsx:27-39`, `train.py:248-270` |
| 6 | Cloud W&B resume can omit the required secret | P1 | **STILL REAL** | `TrainingConfigurator.tsx:188`, `train.py:248-270`, `hf_cloud.py:576-585` |
| 7 | Job deletion bypasses the active-inference model guard | P1 | **STILL REAL** | `server.py:1664-1672`, `jobs.py:1545-1557`, vs `models.py:1055-1059` |
| 8 | Config-only / partial models classified as usable | P1→P2 | **STILL REAL** | `models.py:329-337`, `jobs.py:562-565`, `models.py:1112-1117` |
| 9 | Cloud upload can permanently publish a partial checkpoint | P1 | **STILL REAL** | `hf_cloud.py:340-352`, `:382-386`; mitigation intact at `jobs.py:633-637` |
| 10 | Reattached local training failure recorded as success | P1 | **STILL REAL** | `jobs.py:478-485`, `:1750-1767` |
| 11 | Cloud stop reports cancellation before remote cancel succeeds | P1 | **STILL REAL** — and worse than recorded (see NEW-1) | `hf_cloud.py:755-765`, `:767-776` |
| 12 | Cloud resume lineage can't distinguish parent/child | P1→P2 | **STILL REAL** (+ new UI symptom, NEW-6) | `hf_cloud.py:536`, `jobs.py:792-800`, `JobCard.tsx:187-216` |
| 13 | Invalid training request bodies return HTTP 500 | P2 | **STILL REAL** (both legs) | `server.py:1079-1080` vs `try:` at `:1146`; `AdvancedCard.tsx:101-104` not in `TrainingConfigurator.tsx:477-485` |
| 14 | Core numeric params accept zero and negative values | P2 | **STILL REAL** | `train.py:115-127`, `:203-219`; `server.py:1086-1098` warns only |
| 15 | Terminal transition omits final logs and checkpoints | P2 | **STILL REAL** (new file:line) | `TrainingJobDialog.tsx:135-139`, `:154-164`, `:170-174` |
| 16 | Cloud resume progress not rebased to inherited step | P2 (in progress) | **STILL REAL — fix is NOT on `main`** | `hf_cloud.py:702` (no offset), `:462-467`, `jobs.py:1218`; cf. `andrew:hf_cloud.py:785` |
| 17 | Cloud reattachment duplicates persistent logs | P2 (in progress) | **STILL REAL — fix is NOT on `main`** | `hf_cloud.py:494`, `:609`, `:696-698`; cf. `andrew:hf_cloud.py:558,685` |
| 18 | Manual model upload partially succeeds while reporting failure | P2 | **STILL REAL** | `models.py:918-927`, `:938-944`, `:946-948`; no write-back to `JobRecord` |
| 19 | Frontend blocks cloud runs on host-only training deps | P2 | **STILL REAL** (same correction: only the accelerate gate) | `TrainingConfigurator.tsx:457-459` (page-wide gate) vs `:400` (policy extra correctly target-gated) |
| 20 | All training 409s presented as a local mutex conflict | P2 | **STILL REAL** | `jobsApi.ts:190-195`, `server.py:1150-1154` |
| 21 | Legacy job migration moves every directory under legacy root | P2 | **STILL REAL** | `jobs.py:1005-1029` |
| 22 | Checkpoint ZIP download unbounded in memory, races writes | P2 | **STILL REAL** | `server.py:1605-1612`, `:1625`; only guard `:1590` |
| 23 | Registry lock spans remote network work and submission | P2 | **STILL REAL** | `jobs.py:1122` … `:1241` wrapping `:1221 runner.start()`; `hf_cloud.py:516-593`, `:641-676` |
| 24 | Resume re-passes `--policy.tags` in a form draccus rejects | P1 | **STILL REAL — fix never landed on `main`** | `train.py:69-80`, `:265-267` |

### Priority entries — detail

**MT11 (money, P1) — STILL REAL, and the exposure is larger than the original entry states.**
`stop()` pre-sets `CANCELED` (`hf_cloud.py:760`) *before* `cancel_job` (`:762`) and swallows every
exception at `logger.info` (`:763-765`). `is_running()` then returns `False` (`:767-771`) and the
watchdog finalises the record and drops the runner (`jobs.py:1768`). The compounding fact this audit
adds: **`cancel_job` appears exactly once in the entire repository** — inside `HfCloudJobRunner.stop`
— and the only route to it is `POST /jobs/{id}/stop`, which requires the registry to still hold a
live runner for a record whose state is `running` (`jobs.py:1379-1387`). Once the record is
finalised (by this bug, by a watchdog tick, by a restart that marks it `interrupted` at
`jobs.py:1618-1623`, or by delete at `jobs.py:1553`), **MakerMods Lab has no code path that can cancel the
remote job.** See NEW-1.

**MT9 (data, P1) — STILL REAL.** `hf_cloud.py:340-343` gates the upload on
`pretrained_model/config.json` alone and `:352` adds the step to `seen` on success, so a partial
directory is sealed; the final pass (`:384`) skips seen steps. The recommended completeness marker
(`training_state/scheduler_state.json`) is **not** implemented on `main`. The resume-side mitigation
the original entry credits is intact (`jobs.py:633-637` verifies
`checkpoints/<step>/training_state/training_step.json` on the Hub and fails cleanly), so *resume* is
protected and *inference / fine-tune from the partial artifact* remain exposed — unchanged.

**MT10 (recovery, P1) — STILL REAL.** `TailingJobRunner.returncode()` returns `0` once the pid is
gone (`jobs.py:478-485`, comment explicitly documents the choice), and the watchdog writes
`record.state = "done" if rc == 0 else "failed"` (`jobs.py:1754`). Compounding: `list_local_models`
admits a run only when `state == "done"` (`models.py:252-258`), so a trainer that crashed after a
MakerMods Lab restart is not merely mislabeled — its last partial checkpoint is promoted into the
**models browser** as a completed model, and `_find_local_record` (`models.py:264-280`) then lets it
be uploaded to the Hub.

**MT4 (P1) — STILL REAL.** `TargetCard.setRunner` (`TargetCard.tsx:42-50`) switches to `local`
unconditionally; there is no resume lock on the target. The backend branches on the **source**
runner, not the target (`jobs.py:1168`), so a cloud source sets only `resume_from_hub_repo`
(`:1178-1179`) and never `config_path`. `build_training_command`'s resume branch requires
`request.resume and request.config_path` (`train.py:248`), so the call falls through to the fresh
branch and emits `--resume true` with no `--config_path` (`train.py:319`). Deterministic doomed run.

**MT2 (P1) — STILL REAL.** `_resolve_finetune_pretrained_path` still returns
`chosen.ref.split("@", 1)[0]` (`jobs.py:731`), discarding `checkpoints/<step>`. The docstring
(`jobs.py:727-730`) documents this as a known limitation, but the UI presents a step picker
(`JobCard.tsx:366-385`, `TrainPanel.tsx:190-197`) and passes `finetune_from_step` through
(`TrainingConfigurator.tsx:105`), so the user is told a step was honoured when it was not.

**MT3 (P1) — STILL REAL.** `_list_imported_hub` returns the root policy **only** when the tree scan
is empty (`jobs.py:784-788`). Worse for tracked cloud runs: `_list_hub_checkpoints`
(`jobs.py:792-800`) has **no** root fallback at all, so a completed cloud run with
`save_checkpoint=false` publishes a root policy and its job card reports zero checkpoints
(`checkpoint_count` via `jobs.py:1472` → `TrainingJobDialog.tsx:354-357` "No checkpoints yet").

---

## Part 2 — New findings

Ranked P0/P1/P2 per the brief (P0 = data loss/corruption, silent private-data publication, or main
path completely broken; unmonitored paid-GPU burn ≥ P1).

### NEW-1 — P1 — There is no way to cancel a remote HF job once the local record is finalised

**Pathway:** cloud run → any local finalisation (failed cancel / watchdog / restart / delete) → paid
GPU keeps running with no MakerMods Lab control

`cancel_job` is called from exactly one place in the repository:

- `makermodslab/runners/hf_cloud.py:762` — inside `HfCloudJobRunner.stop()`.

`stop()` is reachable only through `JobRegistry.stop` (`jobs.py:1379-1395`), which raises
`JobNotRunningError` unless `record.state == "running"` **and** `self._runners[job_id]` exists.
Every one of these finalisation paths removes that possibility while the remote job may still run:

- MT11's optimistic `_set_terminal("CANCELED")` before a `cancel_job` that failed (`hf_cloud.py:760-765`);
- the watchdog popping the runner on any terminal transition (`jobs.py:1768`);
- a restart where the persisted record lacks `hf_job_id`/`hf_flavor` → marked `interrupted`, never
  reattached (`jobs.py:1618-1623`). Reachable: the record is persisted with `state="running"` at
  `jobs.py:1212`, **before** `runner.start()` submits the job, and `hf_job_id` is only written at
  `:1234-1240` — a crash in that window leaves a submitted, paid, permanently untracked job;
- `DELETE /jobs/{id}` (`server.py:1664-1668` → `jobs.py:1552-1553`) drops the runner outright.

The orphan does remain **visible** — `server.py:1373` deliberately keeps a dismissed id in the
`/jobs/hub` listing while its stage is in `_HUB_ACTIVE_STAGES`, which is a real mitigation and
contradicts a plausible reading of MT11's "no longer monitors". But `HubJobCard` offers only
"View on Hub" and a trash button gated on `!isHubJobActive(job)` (`HubJobCard.tsx:84-119`) — there
is **no cancel control and no cancel endpoint** (`grep -n "cancel_job\|/cancel" makermodslab/server.py
frontend/src/lib/jobsApi.ts` → no matches). The user's only remedy is huggingface.co.

**Fix shape:** a `POST /jobs/hub/jobs/{id}/cancel` that calls `HfApi.cancel_job` independently of the
registry, surfaced on `HubJobCard` for active stages; plus MT11's reordering (cancel first, verify,
*then* set terminal).

**Evidence:** `hf_cloud.py:755-765`, `:767-776`; `jobs.py:1379-1387`, `:1552-1553`, `:1618-1623`,
`:1768`; `server.py:1364-1373`, `:1492-1505`, `:1664-1677`; `HubJobCard.tsx:84-119`.

---

### NEW-2 — P1 — Fine-tune from the jobs library launches with the WRONG policy type, and the UI locks it

**Pathway:** JobCard "Fine-tune" / SkillDetailDialog → studio Train prefill → policy select disabled
→ trainer launched with `--policy.type act` against a non-ACT checkpoint

`TrainPanel` owns `policyType`, defaulting to `"act"` (`TrainPanel.tsx:157`). `resolveFinetune`
resolves the base checkpoint's real policy **only on the Hub-repo branch**:

```
if (!jobId && opts.repoId) { const rec = await importModel(...); policy = rec.config?.policy_type ?? null; }
...
setFinetuneSeed({ jobId, step: latest, name: ..., policyType: policy ?? "act" });
if (policy) setPolicyType(policy);                          // TrainPanel.tsx:214
```

The two prefill entry points both take the **`jobId`** branch, where `policy` stays `null`:

- `JobCard.tsx:378-384` — `openStudio("train", { train: { baseJobId, baseStep, baseName } })`;
- `SkillDetailDialog.tsx:117` — `{ baseJobId: model.id }`;
- consumed at `TrainPanel.tsx:267-273` → `resolveFinetune({ jobId, step, name })`.

So `setPolicyType` never fires and `policyType` stays at its default/last value. Meanwhile
`finetuneSeed != null` sets `policyLocked` (`TrainingConfigurator.tsx:537`), which **disables** the
policy select (`EssentialsCard.tsx:77`) under the caption "Set by the base skill — the run trains the
same architecture as its source checkpoint" (`EssentialsCard.tsx:102-104`). The claim is false and
the user cannot correct it.

The resulting argv is `--policy.type act --policy.pretrained_path <base-checkpoint>`
(`train.py:282`, `:288-289`). Note the manual path is correct — `handleBaseModelChange` does
`if (model.policy_type) setPolicyType(model.policy_type)` (`TrainPanel.tsx:246`) — which is what
makes this a wiring omission on the prefill path rather than a design choice.

**Not statically determinable:** whether lerobot hard-fails on the type/weights mismatch or loads
something wrong. Either way the advertised fine-tune is not what runs.

**Evidence:** `TrainPanel.tsx:157`, `:178-214`, `:231-249`, `:262-285`, `:475-479`;
`TrainingConfigurator.tsx:537`; `EssentialsCard.tsx:73-105`; `JobCard.tsx:375-385`;
`SkillDetailDialog.tsx:117`; `train.py:282`, `:288-289`.

---

### NEW-3 — P1 (narrow trigger) — Implicit cloud-run dataset upload publishes the user's data PUBLICLY

**Pathway:** cloud start → `_ensure_dataset_on_hub` → `push_to_hub(private=False)` with no per-call consent

`hf_cloud.py:671`:

```
LeRobotDataset(repo_id).push_to_hub(tags=with_makermodslab_tag(None), private=False)
```

The comment at `:665-670` states this is deliberate ("This intentionally reverses the earlier private
default"), and the browser flow normally reaches the Hub through the explicit upload-then-train path
(`TrainingConfigurator.tsx:429-443` + `LocalDatasetCloudNotice`), so this is not the common route.
The `JobRegistry.start` preflight also narrows it: a definitive `local_only` raises
`DatasetNotOnHubError` first (`jobs.py:1116-1120`).

It is still reachable, because the preflight only blocks on a **definitive** `local_only` while
`_ensure_dataset_on_hub` re-decides from `dataset_info`:

- `get_hub_status` memoizes definitive answers for the process lifetime (`datasets.py:190-193`,
  `:213-217`). A repo cached `on_hub` that is subsequently deleted on the Hub yields
  `RepositoryNotFoundError` at `hf_cloud.py:650-653` → local copy found at `:656` → **public push**.
- `get_hub_status` returns `unknown` (uncached) on any transport error (`datasets.py:196-202`), which
  the preflight deliberately lets through (`jobs.py:1111-1114`).

Secondary defect regardless of trigger: this push does **not** call `invalidate_hub_status(repo_id)`
/ `invalidate_dataset_listing_cache()`, unlike every other upload site (`record.py:1112-1114`,
`datasets.py:299`, `:323`, `:800-801`, `:1021`, `:1119`), so the dataset card keeps showing the stale
locality after MakerMods Lab has published it.

**Evidence:** `hf_cloud.py:641-676` (esp. `:650-653`, `:656`, `:671`); `jobs.py:1107-1120`;
`datasets.py:83-88`, `:164-217`.

---

### NEW-4 — P2 — Cloud terminal transition truncates the persisted log (backend twin of MT15)

`_set_terminal` sets `_stop_event` (`hf_cloud.py:622-629`). `_tail_loop` checks it **per line**
(`:692 if self._stop_event.is_set(): return`) and again in its outer wait (`:722-723`). The status
poller reaches a terminal stage on a 5s cadence (`:421`, `:730-753`) and is independent of the SSE
stream, so it routinely fires **before** the log stream has delivered the tail of the job — the final
trainer traceback, `[wrapper] uploaded checkpoint <step>`, and `[wrapper] trainer exited with rc=N`
(wrapper source `:353`, `:388`).

Those lines are never fetched, so they are never written to `log.jsonl` (`:707-712`), and there is no
post-terminal drain. `read_persisted_logs` (`jobs.py:1406-1429`) and `read_metrics_history`
(`:1431-1463`) are therefore permanently missing the end of every cloud run — MT15 is only about the
*frontend* not draining; this one destroys the record on disk. It is the single biggest reason a
failed cloud run is undiagnosable after the fact.

**Fix shape:** on terminal, stop *reconnecting* but let the current SSE iteration finish (or do one
bounded non-follow `fetch_job_logs` pass) before closing the file.

**Evidence:** `hf_cloud.py:421`, `:622-629`, `:678-728` (esp. `:687-692`, `:717-723`), `:730-753`.

---

### NEW-5 — P2 — A user-initiated cloud stop is recorded as `failed` with a synthetic exit-code message

`JobState` has no `canceled` member (`jobs.py:48`). `stop()` sets `_terminal_status = "CANCELED"`
with **no message** (`hf_cloud.py:760` — `_set_terminal(status)` only stores `message` when truthy,
`:626-628`). `returncode()` then returns `1` for anything that isn't `COMPLETED`
(`hf_cloud.py:773-776`), so the watchdog writes `state="failed"`, `exit_code=1`,
`error_message="Subprocess exited with code 1"` (`jobs.py:1754-1767`, `terminal_message()` returns
`None`).

Downstream consequences, not merely cosmetic:

- the monitor renders `failed — Subprocess exited with code 1` for a deliberate stop
  (`TrainingJobDialog.tsx:324-327`);
- `endedBeforeTarget` treats `failed`/`interrupted` as resumable (`JobCard.tsx:319-329`), so a
  stopped run and a crashed run are indistinguishable in the resume affordance;
- `list_local_models` excludes it (`models.py:253`), so a local run stopped at a good checkpoint
  never appears in the models browser.

**Evidence:** `jobs.py:48`, `:1750-1767`; `hf_cloud.py:622-629`, `:755-765`, `:773-776`, `:796-803`.

---

### NEW-6 — P2 — MT12's UI symptom: duplicate checkpoint entries with identical `step` in the lineage dropdown

`JobCard` fetches `/jobs/{id}/checkpoints` for the run **and each ancestor** and flat-merges by step
(`JobCard.tsx:187-216`, `:203`). For a cloud resume, parent and child share one output repo
(`hf_cloud.py:536`) and both enumerate the same tree (`jobs.py:792-800`), so every parent checkpoint
appears **twice** with the same `step`. `CheckpointDropdown` keys and values items on the raw step
(`CheckpointDropdown.tsx:52-60`): duplicate React keys plus two `SelectItem`s sharing a `value`, and
`lineageCheckpoints.find(c => c.ckpt.step === selectedStep)` (`JobCard.tsx:294-296`) silently picks
whichever came first, which decides which run a subsequent Continue/Download/Inference targets
(`:303`, `:336`, `:399`).

This is the observable half of MT12 and should be fixed with it (dedupe by `(repo, step)` and/or
carry lineage in the ref).

**Evidence:** `JobCard.tsx:187-216`, `:292-304`, `:331-349`, `:394-400`; `CheckpointDropdown.tsx:52-60`;
`hf_cloud.py:536`; `jobs.py:792-800`.

---

### NEW-7 — P2 — A seeded resume can be silently converted into a fresh full-length run

`AdvancedCard` exposes a raw "Resume from checkpoint" switch (`AdvancedCard.tsx:277-285`) that is not
disabled on a resume seed. `TrainingConfigurator` seeds `resume: !!resumeSeed` but keeps
`resume_from_job_id` / `resume_from_step` unconditionally (`:183-187`), and the "Continuing …"
banner is keyed on `resumeSeed`, not on `config.resume` (`:499-514`).

Toggle it off and: `jobs.py:1161` skips the entire resume resolution, so `config_path` stays `None`;
`train.py:248` falls through to the fresh branch and emits `--resume false` with `--steps` still
prefilled at `sourceSteps * 2` (`TrainingConfigurator.tsx:173`). A from-scratch run of double length
is launched while the UI says "Continuing X from step N".

The record still carries `resume_from_job_id`, and `read_metrics_history` walks that lineage
**regardless of `config.resume`** (`jobs.py:1449-1453`), so the chart concatenates the parent's curve
with a child that restarted at step 0 and dedupes by step (`:1458-1463`) — the parent's history is
overwritten by the fresh run's early steps.

**Evidence:** `AdvancedCard.tsx:277-285`; `TrainingConfigurator.tsx:173`, `:183-187`, `:499-514`;
`jobs.py:1161-1189`, `:1431-1463`; `train.py:248`.

---

### NEW-8 — P2 — `upload_local_model` wipes an existing repo's tag metadata

`upload_local_model` accepts a caller-supplied `repo_id`, creates with `exist_ok=True`
(`models.py:919`), and finishes with `metadata_update(..., {"tags": final_tags}, overwrite=True)`
(`models.py:939`). On a repo that already exists — e.g. one lerobot pushed, carrying
`{"robotics", "lerobot", <model_type>}` — `overwrite=True` **replaces** the tag list with
`with_makermodslab_tag([policy_tag])` (`utils/config.py:90`, `:93-105`), dropping the generic `lerobot`
tag. That tag is exactly what `_list_author_models` and `_hub_policy_type` key on
(`models.py:102-123`, `:466-493`), so the repo can fall out of the `/jobs/hub` model listing.

Called from `POST /jobs/{id}/upload` (`server.py:942-945`). Related to, but distinct from, MT18 (that
one is about the partial-success reporting on the same function).

**Evidence:** `models.py:886-954` (esp. `:916`, `:919`, `:932-939`), `:102-123`, `:466-493`;
`utils/config.py:85-105`.

---

### NEW-9 — P2 — The cloud wrapper's `seen` set is mutated from two threads without a lock

`_watch` runs `_scan_and_upload` on a 15s cadence in a daemon thread (`hf_cloud.py:358-368`), and the
main thread runs a final `_scan_and_upload` in the `finally` of `proc.wait()` **immediately after**
`stop_event.set()` (`:378-386`) — which does not join the watcher. `_scan_and_upload` does a
check-then-add on `seen` around a slow `upload_folder` (`:343-352`), so the same step can be uploaded
twice concurrently into the same repo. Self-healing in the common case (one commit wins, the loser
prints `upload failed` and is retried), but it makes MT9's `seen` bookkeeping harder to reason about
and produces spurious failure lines in the log the user reads.

**Evidence:** `hf_cloud.py:328-368`, `:378-389`.

---

### NEW-10 — P2 — The resume step guard is skipped when resuming "latest"

`server.py:1102` blocks a resume only when `cfg.resume_from_step is not None`. Resuming from the
latest checkpoint (`resume_from_step = None`, an explicitly supported mode — `jobs.py:615-617`,
`:658-659`, `ResumeSeed.step: number | null` at `TrainingConfigurator.tsx:39`) skips the guard, so a
`steps` value at or below the actual latest checkpoint reaches the trainer and dies there. The UI is
currently safe because `JobCard.goToResume` always sends a concrete step (`JobCard.tsx:337`), but the
frontend mirror of the check has the same hole (`TrainingConfigurator.tsx:467-472`).

**Evidence:** `server.py:1099-1114`; `jobs.py:615-617`, `:658-659`; `TrainingConfigurator.tsx:39`,
`:467-472`; `JobCard.tsx:331-349`.

---

## Part 3 — Contradictions and corrections to this brief

1. **The brief's premise that `main` is a superset of the audited tree is wrong.** `518ca56` is not
   an ancestor of `main`. Two fixes the source list records as landed-or-in-progress (MT16, MT17)
   exist **only on `andrew`**, and MT24's fix exists on **neither**. Re-verifying "against `main`"
   therefore *reopens* three entries rather than closing them.
2. **MT11's stated impact "the local runner … no longer monitors the still-running remote job" is
   incomplete but its opposite is also not true.** `server.py:1364-1373` deliberately keeps a
   dismissed-but-active Hub job visible in `/jobs/hub`, so the job is not invisible. The real gap is
   *actionability*, not visibility — see NEW-1. Worth correcting in the source list so a fix isn't
   scoped to the wrong problem.
3. **MT5 should be downgraded to PARTIALLY FIXED.** `policyLocked` (`TrainingConfigurator.tsx:537` →
   `EssentialsCard.tsx:77`) closes the "record labels a different policy" leg the original entry led
   with. The cloud dependency-extra leg (`hf_cloud.py:555` still keyed on the edited
   `config.policy_type`) is now unreachable *from the UI* on resume and reachable only from non-UI
   callers. The remaining live leg is target/batch/optimizer/W&B editability.
4. **MT13's second clause survived a UI rewrite.** The original cited `TargetCard.tsx:150-190` for
   the timeout input; on `main` the field moved to `AdvancedCard.tsx:289-317` and its `timeoutInvalid`
   state (`:103-104`) is *still* absent from `startDisabled` (`TrainingConfigurator.tsx:477-485`).
   Same defect, different file — re-point the evidence.
5. **The fix clusters in the source list still hold on `main`**, with one addition: cluster
   "10, 11 — runner finalization contract" should absorb NEW-1 and NEW-5, since all four are the same
   contract violation (a runner that cannot know, or was told to stop, reporting a definitive
   `returncode()` the watchdog converts into an irreversible terminal state).

---

## Part 4 — What could not be verified, and why

Purely static audit; no HF Jobs submission, no Hub call, no training process, no hardware.

| Question | Why unverifiable here |
|---|---|
| MT1's close-out (`str(JobStage.COMPLETED)` shape through a real terminal poll) | Needs a live cloud job reaching terminal. The recommended `stage in _TERMINAL_STAGES` hardening is still unimplemented (`hf_cloud.py:744-749`), so the close-out rests on the installed hub version's dataclass staying uncoerced. |
| MT9's exact partial-upload window | Reproducing needs the 15s watcher to land inside lerobot's `pretrained_model` → `training_state` write. The production observation of 2026-07-19 stands as the evidence; nothing on `main` changed the mechanism. |
| MT11's cancel-failure path | Requires `cancel_job` to actually raise against the live Hub. The code path (`hf_cloud.py:761-765`) is unambiguous; the *frequency* is not knowable statically. |
| MT24's draccus rejection | Depends on the pinned lerobot/draccus `_decode_list` behavior on the `from_pretrained` path. The production traceback of 2026-07-19 is the evidence; `main`'s `train.py:265-267` is byte-identical to the version that produced it. |
| NEW-2's downstream failure mode | Whether lerobot hard-fails or mis-loads on `--policy.type act` + a SmolVLA `pretrained_path` needs the trainer. The *wiring* defect is fully established statically. |
| Race between the wrapper's checkpoint watcher and lerobot's own end-of-training `push_to_hub` into the same repo | Both write commits to the same model repo, and `stop_event.set()` only fires *after* `proc.wait()` returns (`hf_cloud.py:378-381`) — i.e. the watcher is live during the trainer's final push. Whether a commit conflict can lose the final root policy needs a live run. Flagging as an open question, not a finding. |
| `andrew`-vs-`main` argv divergence (`--eval_freq` vs `--env_eval_freq`) | Which is correct depends on the lerobot pin. Both branches ship a `TrainingRequest` field and a CLI flag; only a live trainer settles it. **Material to any fix port and belongs to the coordinator.** |
| Frontend runtime behavior (polling races, portal/remount timing, duplicate-key rendering) | Per house rules, no browser was driven. All frontend verdicts are source-derived. |
