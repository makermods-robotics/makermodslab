# Inference Bug List — RE-VERIFIED against `main`

**Re-verification date:** July 23, 2026
**Baseline audit:** `internal_docs/bugs/Inference Bug List.md` (July 14, 2026, commit `518ca56`, **`andrew` branch**)
**Re-verified against:** `main` @ `c7d9f27` (working tree; no code, index, or branch state was touched)
**Scope:** every entry of the original list re-assessed against `main`'s code, plus a fresh hunt across
`makermodslab/rollout.py`, the inference routes in `makermodslab/server.py`, and the four inference frontend surfaces.

**Structural change since the audit.** `main` is post-redesign: **`frontend/src/pages/Inference.tsx` no longer
exists**. The inference page became a dialog hosted above the router
(`frontend/src/contexts/InferenceSessionContext.tsx`), and its polling / stop / exit-guard logic was ported
essentially verbatim into `frontend/src/components/inference/InferenceSessionDialog.tsx`. There is also now a
**second** launch+stop surface, `frontend/src/components/studio/DeployPanel.tsx`, which re-implements the
InferenceModal logic ("ported VERBATIM", `DeployPanel.tsx:69-74`). Every frontend finding below therefore has to
be checked on **two** files, and several now reproduce on both.

**Open PRs against `main` (not merged).** #9 = I1, #10 = I5, #11 = I4. All three defects are verified present in
`main`'s code below. The PRs were not read and are not assumed correct or complete — the overlap is flagged per
entry so the coordinator can decide whether the PR actually closes what is described here.

**Excluded per brief (context only, not re-investigated):** the arm-identity warn-and-proceed severity
(rows 3/4), and the lerobot 0.6.0-checkpoint-on-0.5.2-runtime `draccus`/`pretrained_revision` constraint.

## Severity key

- **P0:** hardware-safety — an arm can be driven, or left energized, with no working stop path.
- **P1:** high-impact lifecycle / mutual-exclusion / stop-path failure.
- **P2:** major pathway, validation, or reporting failure.
- **P3:** lower-impact correctness or robustness defect.

---

## Part 1 — Verdicts on the original list

| # | Original finding | Verdict on `main` | Key evidence on `main` |
|---|---|---|---|
| I1 | Stopped startup worker keeps touching hardware / corrupts a newer session | **STILL REAL** (one sub-case fixed) | `rollout.py:733-741`, `:786-805`, `:1020-1033` |
| I2 | Stop reports success and loses the handle even when termination fails | **STILL REAL** | `rollout.py:1035-1052` |
| I3 | A failed explicit Stop permanently disarms the leave guard | **STILL REAL** (relocated file) | `InferenceSessionDialog.tsx:235-246`, `useSessionExitGuard.ts:87-89` |
| I4 | Robot-driving exclusion not atomic; calibration omitted | **STILL REAL** (worse than audited) | `rollout.py:917-935`, `calibrate.py:235`, `auto_calibrate.py:210,518-519` |
| I5 | FastAPI shutdown does not stop the inference subprocess | **STILL REAL** | `server.py:2549-2558`, `rollout.py:829-835` |
| I6 | A model downloaded for offline use is still launched through the Hub | **STILL REAL** (now on 2 surfaces) | `inferenceLaunch.ts:47-49`, `DeployPanel.tsx:321`, `jobs.py:774,1317` |
| I7 | A selected local model can be deleted during active startup | **STILL REAL** | `rollout.py:945`, `:862-876`, `:996-999`, `models.py:975-987` |
| I8 | Overlapping status polls consume and erase a terminal failure | **FIXED** (backend); residuals remain | `rollout.py:97-104`, `:1112-1113`, `:1138-1154`, `:948` |
| I9 | External bimanual `right_*` cameras mapped to wrong feature keys | **STILL REAL** (now on 2 surfaces) | `InferenceModal.tsx:129-150`, `DeployPanel.tsx:141-158`, `rollout.py:628-648` |
| I10 | A real mid-run motor overload mislabeled as a cleanup-only warning | **STILL REAL** (documented tradeoff) | `rollout.py:573-586`, `utils/errors.py:33,36-46` |
| I11 | Active startup can display the previous run's log | **STILL REAL** | `rollout.py:811-814`, `:1060-1077`, `InferenceSessionDialog.tsx:144-149` |
| I12 | Hub checkpoint cache conflates imported vs cloud listing rules | **STILL REAL** | `jobs.py:1465-1472`, `:1486-1499` |
| G1 | Required language task is not enforced | **STILL REAL** (now on 2 surfaces) | `InferenceModal.tsx:353-360`, `DeployPanel.tsx:491-499`, `rollout.py:69` |
| G2 | Backend duration has no positive lower bound | **STILL REAL — severity understated** | `rollout.py:71,604`, `ui/number-input.tsx:39-47` |
| G3 | Multiple roles may bind the same physical camera | **STILL REAL** (now on 2 surfaces) | `InferenceModal.tsx:347-349`, `DeployPanel.tsx:482-484` |
| G4 | Server arm-count guard loses the frontend's action-dim fallback | **STILL REAL** (now on 2 surfaces) | `InferenceModal.tsx:330 vs :425`, `DeployPanel.tsx:470 vs :554`, `rollout.py:378-379` |

### I1 — STILL REAL (one sub-case fixed; a false invariant is now written into a comment)

Everything session-keying-related survives. State is still global and un-keyed
(`rollout.py:92-114`); `_set_phase` writes into whatever meta is present (`rollout.py:158-166`);
`_pump_stdout` writes the module-global `_inference_rollout_started_at` with no session check
(`rollout.py:189-196`); `_report_download_progress` mutates whichever meta exists (`rollout.py:283-290`);
`_fail_startup` carries **no session token** and clears whichever session is currently active — it only
checks `if not inference_active: return` (`rollout.py:733-741`), so a stale worker A that raises after B
has claimed the slot still fails B.

Cancellation is still checked only **after** `_prepare_robot` returns (`rollout.py:786-788` after the
download, `:803-805` after the whole preflight). Nothing inside `_prepare_robot` (`rollout.py:651-718`)
consults the cancel event, so a stop landing there lets the worker keep opening follower buses and
rewriting `Torque_Limit`/`Goal_Velocity` (`rollout.py:695-696`, `:712-716`). Stop with no process still
clears state and returns success without joining the worker (`rollout.py:1020-1033`).

**Fixed sub-case:** the subprocess commit is now performed under `_state_lock` with a re-check of both the
cancel event and `inference_active` (`rollout.py:852-888`). A stale worker can therefore no longer overwrite
a *newer* session's `_inference_proc` — the P0-adjacent untracked-child variant the audit called out is
closed on that path. (It is replaced by a different untracked-child path — see **N2**.)

**New, sharper evidence:** `rollout.py:1021-1027` now states as an invariant that
*"There's no process to terminate and no policy has driven the robot … A download-first ordering guarantees
'no robot touched' here."* That is false whenever the stop lands during phase 2 (`_prepare_robot`), which is
the second of the two pre-subprocess phases. The comment will read as a guarantee to the next maintainer.

*Overlaps PR #9.*

### I2 — STILL REAL

`rollout.py:1035-1052`: `terminate()` / `wait(timeout=5)` / `kill()` / `wait()` are wrapped in a single
`try/except Exception` that only logs (`:1043-1044`); the state is then cleared unconditionally
(`:1046-1051`) and `{"success": True, …}` returned (`:1052`) without verifying the child exited. The only
process handle is discarded.

### I3 — STILL REAL, relocated

The audited `frontend/src/pages/Inference.tsx` is gone; the logic lives in
`InferenceSessionDialog.tsx`. `markHandled()` is still called **before** the awaited stop request
(`InferenceSessionDialog.tsx:235`, request at `:237`) and the catch path (`:239-246`) re-arms nothing.
`useSessionExitGuard`'s latch is re-armed only by an `active` false→true transition
(`useSessionExitGuard.ts:87-89`), which never happens while the run stays active. Every leave vector
short-circuits on the latch: `beforeunload` (`:98`), `pagehide` beacon (`:104-105`), popstate confirm
(`:139`), unmount cleanup (`:161`). The hung-run watchdog's one-shot `stopRequestedRef` is likewise set
before the request (`InferenceSessionDialog.tsx:200`, call at `:208`) and never reset when
`stopIfHung` swallows the failure (`:118-124`).

### I4 — STILL REAL, and broader than audited

Inference reads the other features' flags while holding only its own `_state_lock`
(`rollout.py:917-935`); teleoperation and recording each do the same under their own locks. **Inference
never mentions calibration at all** (no `calibration_active` reference anywhere in `rollout.py`).

**Correction / extension to the original entry:** the audit scoped the calibration case to manual
`calibrate.py:235` (checks only `self.status.calibration_active`). `main` shows **auto-calibration has the
same hole**: `auto_calibrate.py:210` and `:518-519` guard only against another auto-calibration. Auto-cal is
the sharper case — per `CLAUDE.md` it drives the arm under torque and **writes servo EEPROM** — and it can be
admitted on the same follower port as a live inference run.

*Overlaps PR #11.*

### I5 — STILL REAL

`server.py:2549-2558`: the shutdown handler stops only the broadcast thread; there is no inference cleanup
and no reference to `handle_stop_inference`. The child is still spawned without `start_new_session`
(`rollout.py:829-835`), so a terminal Ctrl-C reaches it via the shared process group, but a
`--reload` worker restart, a SIGTERM to the uvicorn pid, or a programmatic shutdown orphans it.

*Overlaps PR #10.*

### I6 — STILL REAL, now reachable from a second surface

`importSourceForModel` still returns `model.hf_repo_id ?? model.path ?? model.id`
(`inferenceLaunch.ts:47-49`), preferring the Hub id over the downloaded local checkpoint. The offline
promise is still made in the UI (`AddModelFromHubDialog.tsx:30`, `:109`). A non-directory source registers
as Hub-backed and lists checkpoints through the Hub API (`jobs.py:1317`, `:774`). New on `main`: the studio
Deploy panel takes the same path (`DeployPanel.tsx:321`), so the defect now has two entry points, not one.

### I7 — STILL REAL

The claim window is unchanged: the meta seeded at claim time carries only `phase` + `policy_ref`
(`rollout.py:945`), and `policy_path` is written only inside the subprocess-commit block
(`rollout.py:862-876`, key at `:867`). `inference_in_use_path()` returns that field
(`rollout.py:996-999`), so it is `None` for the whole download + preflight window, and
`models._model_in_use` treats a missing path as permission to delete (`models.py:977-979`). See **N3** for a
second, unguarded deletion route that makes this materially worse.

### I8 — FIXED (backend); two residuals

The one-shot terminal payload is gone. `_last_result` is now explicitly documented as idempotent and is
*kept* until the next start claims the slot (`rollout.py:97-104`); the idle branch returns a copy without
clearing (`rollout.py:1112-1113`), the finalisation branch stores and returns a copy
(`rollout.py:1138-1154`), and only `handle_start_inference` clears it (`rollout.py:948`). The declaration
comment names this exact bug as the reason. **Verdict: FIXED.**

Residuals, both now cosmetic rather than error-losing:
- the frontend still has no in-flight guard on its 1 Hz interval (`InferenceSessionDialog.tsx:220-221`) and
  still sets `doneRef` only after two awaits (`:130`, `:145`, then `:157`) — with an idempotent payload the
  worst outcome is a duplicated failure toast, not a swallowed error;
- the secondary claim **STILL REAL**: finalisation mines the log (`rollout.py:1136`) without joining the
  stdout pump thread (`rollout.py:892-897`), so a traceback still being drained can be read partially. The
  pump's per-line `flush()` (`rollout.py:181-183`) keeps this narrow.

### I9 — STILL REAL, now duplicated

`cameraMappings` still strips a unique `left_`/`right_` prefix down to the bare name
(`InferenceModal.tsx:143-149`), and the backend still attaches **every** camera to the BiSO **left** arm
(`rollout.py:647`, rationale at `:636-638`) — so a checkpoint's `right_wrist` becomes `left_wrist`. The
collision branch keeping full names still yields `left_left_x` / `left_right_x`. The whole function is
duplicated verbatim in the new studio panel (`DeployPanel.tsx:141-158`), so any fix must land twice.

### I10 — STILL REAL (acknowledged tradeoff)

`_classify_outcome` still downgrades any post-marker non-zero exit whose mined text matches the
`CLEANUP_MARKERS` (`rollout.py:584-586`; markers `("overload", "torque_enable")` at `utils/errors.py:33`,
matcher at `:36-46`) with no teardown-vs-active evidence. `utils/errors.py:62-77` still explicitly owns this
as rollout's log-tail-only fallback, so it remains a documented limitation rather than an oversight.

### I11 — STILL REAL

The log file is created only immediately before the spawn (`rollout.py:811-814`), i.e. after download and
preflight. `_resolve_inference_log_path` still falls back to the newest historical `*.log` whenever the
active meta has no usable path (`rollout.py:1066-1077`), and the dialog fetches the log on every 1 s tick
(`InferenceSessionDialog.tsx:144-149`).

### I12 — STILL REAL

Imported records fetch with `_list_imported_hub` and cloud records with the tree-only default
(`jobs.py:1465-1472`), but `_list_cloud_cached` keys its 30 s TTL cache on `repo_id` alone with no scanner
mode in the key (`jobs.py:1493-1499`).

### G1 — STILL REAL, now on both surfaces

`canStart` has no task check on either surface (`InferenceModal.tsx:353-360`,
`DeployPanel.tsx:491-499`); the task input renders for `requires_task` policies but an empty value is
submittable (`InferenceModal.tsx:543-559`, `DeployPanel.tsx:754-769`); the backend defaults `task: str = ""`
(`rollout.py:69`) and forwards it verbatim (`rollout.py:603`). Whether a blank task degrades or fails
remains policy-dependent — **cannot verify statically**.

### G2 — STILL REAL, and the audit understated the reach

`duration_s: int = 60` is unvalidated (`rollout.py:71`) and passed verbatim as `--duration=` (`rollout.py:604`);
the installed lerobot prints `"infinite"` for a non-positive duration
(`.venv/…/lerobot/scripts/lerobot_rollout.py:226`).

**Correction:** the audit priced this as *"reachable only by a direct API caller"* because the UI sets
`min={1}`. That is wrong on `main`. `NumberInput` never clamps to `min` — it only rejects non-finite input
(`frontend/src/components/ui/number-input.tsx:39-47`), and both call sites pass the value straight through
(`InferenceModal.tsx:568-570`, `DeployPanel.tsx:778-780`). A user typing `0` into "Max duration (seconds)"
starts an **unbounded autonomous run from the normal UI**. Severity should rise accordingly.

### G3 — STILL REAL, now on both surfaces

`allCamerasBound` enforces only that every role resolves to a live camera, never device uniqueness
(`InferenceModal.tsx:347-349`, `DeployPanel.tsx:482-484`). Whether the second open fails or duplicates
frames remains device-dependent — **cannot verify statically**.

### G4 — STILL REAL, now on both surfaces

Both surfaces compute the arm count from `state_dim ?? action_dim`
(`InferenceModal.tsx:330`, `DeployPanel.tsx:469-470`) but forward only `state_dim`
(`InferenceModal.tsx:425`, `DeployPanel.tsx:554`), and the server's `_arm_count_mismatch` defers to the
subprocess when `checkpoint_state_dim` is `None` (`rollout.py:378-379`). An action-only checkpoint keeps the
UI guard and loses the authoritative backend one.

---

## Part 2 — New findings

### N1 — P0 — A Stop that overruns the 5 s budget SIGKILLs the rollout mid-teardown, leaving follower torque enabled; MakerMods Lab has no fallback release on any inference exit path

**Status:** Confirmed structurally; the timing threshold needs one hardware measurement.

`handle_stop_inference` gives the child a **hard 5-second budget**, then SIGKILLs it:

```
rollout.py:1036-1042
    proc.terminate()
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("Inference did not exit in 5s; killing")
        proc.kill(); proc.wait()
```

What that 5 seconds has to cover, in the pinned lerobot (`.venv/…/lerobot/rollout/strategies/core.py:119-146`):

1. `self._engine.stop()` — for the RTC engine this joins a background thread (`inference/rtc.py:196-207`).
2. `_return_to_initial_position(hw)` — signature `duration_s: float = 3.0, fps: int = 50`
   (`core.py:141`), i.e. **150 iterations** of `robot.send_action(interp)` followed by
   `precise_sleep(1/50)` (`core.py:147-154`). `precise_sleep` is a plain fixed wait, not a
   deadline-aware one (`.venv/…/lerobot/utils/robot_utils.py:19-32`), so the **3.0 s of sleeps is a floor**
   and the 150 serial round-trips are **additive** on top of it.
3. Only *after* that does `robot.disconnect()` run (`core.py:133-134`) — and `disconnect()` is what
   **disables torque** and releases the cameras.

So the teardown that ends with torque release cannot finish in under 3.0 s even in the best case, and on a
real rig (150 serial writes at SO-101 bus latency, plus a bimanual second bus, plus OpenCV/AVFoundation
camera release) plausibly exceeds 5 s. When it does, MakerMods Lab SIGKILLs the process **before**
`robot.disconnect()` — and:

**MakerMods Lab has no fallback torque release for inference.** `rollout.py`'s import block
(`rollout.py:42-53`) does not import `.torque`; the only consumer of `force_disable_bus_torque`
(`torque.py:30`) anywhere in the package is `auto_calibrate.py:39,179`. There is no
`force_disable_bus_torque` call on the stop path, the abandon path, the `_fail_startup` path, or the
status-finalisation path. Inference is the **only** robot-driving feature whose teardown lives in a
separate process that MakerMods Lab can forcibly sever, and it is the one without a fallback.

This directly undercuts a contract the repo pins on purpose:
- `rollout.py:605-611` — `--return_to_initial_position=true` is set explicitly *"so the contract is ours,
  not upstream's"*;
- `InferenceSessionDialog.tsx:228-230` — *"The follower eases back to its start pose and releases torque."*

Aggravating: `DeployPanel.tsx:584-585` toasts *"Stopping inference / The rollout is winding down"* on any
HTTP 200, including the SIGKILL case, and `handle_stop_inference` returns 200 there (see **I2**).

**Impact.** After a Stop, the follower can be left energized, holding a mid-motion pose indefinitely, with
the backend and both UIs reporting idle. Nothing in MakerMods Lab will release it; the operator's only recovery
is a power cycle.

**Not verifiable statically:** whether teardown actually crosses 5 s on this hardware. The static facts —
a 3.0 s floor plus N serial writes plus camera release against a 5 s hard kill, and zero fallback torque
release — do not depend on that measurement.

### N2 — P0/P1 — The abandoned-after-spawn path can leak a live, completely untracked rollout subprocess

**Status:** Confirmed.

When a stop races the spawn, the worker takes the abandon branch:

```
rollout.py:878-888
    logger.info("Inference startup abandoned after spawn (stop requested); killing subprocess")
    with contextlib.suppress(Exception): proc.terminate()
    with contextlib.suppress(Exception): proc.wait(timeout=5)
    with contextlib.suppress(Exception): log_handle.close()
    return
```

- There is **no `kill()` fallback** here (unlike `handle_stop_inference`). A `TimeoutExpired` from
  `wait(timeout=5)` is swallowed by `contextlib.suppress`.
- The stdout pump is deliberately never started on this path (`rollout.py:890-897` runs only after the
  commit), so nothing is reading the pipe.
- `proc` is a local variable that goes out of scope at `return`, and it was never assigned to
  `_inference_proc` (the commit block skipped it, `rollout.py:853-854`). **No handle survives anywhere.**
- Meanwhile the SIGTERM is only observed through lerobot's `shutdown_event`, which the *run loop* polls.
  During `strategy.setup(ctx)` — policy load and `robot.connect()` — nothing checks it
  (`.venv/…/lerobot/scripts/lerobot_rollout.py:232-238`). A multi-GB policy load can easily outlast the
  5 s window.

**Broken pathway.** Stop is pressed just as the subprocess spawns → the worker terminates and waits 5 s →
the child is still loading the policy → the worker gives up silently and returns → the child finishes
loading, **connects and energizes the follower**, enters the loop, and only then observes the shutdown
event. During that window the backend reports idle, the mutex is open, and no handle exists to stop it.

Distinct from **I2** (which is the terminate-fails path *with* a retained handle) and from **I1**.

### N3 — P1 — `DELETE /jobs/{job_id}` bypasses the inference in-use guard and `rmtree`s a checkpoint a live rollout is reading

**Status:** Confirmed.

`server.py:1664-1677` deletes a job with **no** `_model_in_use` / `inference_in_use_path()` check:

```
@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    record = job_registry.get(job_id)
    job_registry.delete(job_id)
```

`JobRegistry.delete` removes the directory outright — `shutil.rmtree(_job_dir(self._output_root, job_id))`
(`jobs.py:1555-1556`). Its only guard is `record.state == "running"` (`jobs.py:1550-1551`), which is about
the **training** job, not about an inference run reading its checkpoints.

The in-use guard exists only on the models route: `delete_local_model` calls `_model_in_use`
(`models.py:1019-1021` for imported/downloaded dirs, `:1057-1059` for run dirs), which consults
`rollout.inference_in_use_path()` (`models.py:975-987`). Two delete routes, one guard.

Reachable from the UI, not just the API: `deleteJob` (`frontend/src/lib/jobsApi.ts:247`) is called from
`frontend/src/components/jobs/JobsDataContext.tsx` and
`frontend/src/components/training/TrainingJobDialog.tsx`.

**Impact.** A local training run's output dir can be deleted while a rollout subprocess is running that
checkpoint. This is the harm **I7** describes, but with the guard bypassed entirely rather than merely
un-armed during a startup window — so it applies for the *whole* run, not just the pre-spawn window.

### N4 — P1 — During a status-endpoint outage the dialog shows no Stop control and the exit guard is disarmed

**Status:** Confirmed.

`InferenceSessionDialog.tsx:338-342`: while `status` is `null` the entire body — including the Stop button
at `:437-447` — is replaced by a *"Connecting to inference…"* spinner. The tick's catch (`:210-218`) only
toasts *"Lost connection to backend"* and **never sets `status`**, so a status-endpoint outage keeps
`status` null for its whole duration.

In that state:
- the exit guard's `active` is `status?.inference_active === true` (`:107`) → **false** → no `beforeunload`
  prompt, no `pagehide` stop beacon, no popstate confirm, no unmount stop
  (`useSessionExitGuard.ts:94-95`, `:134-135` both early-return on `!active`);
- but `live` is `status == null` → **true** (`:252`), so ESC, outside-click and the X are all blocked
  (`:315-329`) — the operator is pinned on a dialog with no stop control;
- the run itself is unaffected and continues toward (or through) the arm.

This directly contradicts the file's own stated contract at `:100-105`: *"While a session is active (any
phase, INCLUDING downloading_model), an unintentional exit stops the run."* That holds only after the first
successful poll. The same null-status hole covers the entire pre-first-poll window on mount.

Partial mitigation: `DeployPanel`'s independent Stop button (`DeployPanel.tsx:874-882`) still works, but
only if the studio panel is open (`DeployPanel.tsx:442-443` gates the poll on `open`) and its own status
poll is succeeding.

### N5 — P2 — The subprocess is reaped only by a `/inference-status` poll; with nobody polling, the slot stays latched active forever

**Status:** Confirmed.

`handle_inference_status` is the **only** place `proc.poll()` is consulted and the only writer of the
terminal `_last_result` (`rollout.py:1114-1154`). There is no background reaper: the startup handler runs
only `warn_if_cuda_mismatch()` (`server.py:2543-2546`) and the shutdown handler touches only the broadcast
thread (`server.py:2549-2558`).

If the last polling surface goes away — the tab is closed and the `pagehide` beacon fails or is dropped
(`useSessionExitGuard.ts:114-119`, best-effort with `.catch(() => {})`), the studio panel is closed
(`DeployPanel.tsx:442-443`) — a run that ends on its own `--duration` is never reaped:
`inference_active` stays `True` and `_inference_proc` stays set.

**Impact.** The feature mutex stays claimed: inference (`rollout.py:930-935`), teleoperation and recording
all 409 indefinitely, and `inference_in_use_path()` (`rollout.py:996-999`) keeps blocking model deletion
against a process that no longer exists. Recoverable by pressing Stop or restarting the server, but the
error messages ("Inference is already active. Stop it first.") give no clue that the run already ended.

### N6 — P2 — Bimanual right-arm fields are unvalidated, so a bad request opens and primes the LEFT arm before failing

**Status:** Confirmed.

`handle_start_inference`'s synchronous pre-spawn checks are only the mutex (`rollout.py:917-935`), the
arm-count guard (`:962-965`) and the policy-ref shape check (`:969-975`). It never validates
`follower_port`, `follower_config`, or — when `mode == "bimanual"` — `right_follower_port` /
`right_follower_config`. `mode` itself is taken verbatim from the client (`rollout.py:75`) and never
cross-checked against the named robot record.

`_prepare_robot` then works left-then-right (`rollout.py:672-696`), so a blank or wrong right-arm port
surfaces only **after** the left follower's bus has been opened and its `Torque_Limit` / `Goal_Velocity`
rewritten (`rollout.py:688-695`).

The comment at `rollout.py:958-961` — the pre-spawn guards exist so a mismatch is rejected *"BEFORE spawning
the worker"* — holds only for the arm-count case. A bimanual request-shape error still touches one arm.

### N7 — P2 — Camera dict keys are interpolated into a draccus dict literal with no escaping

**Status:** Confirmed (robustness/legibility, not a shell-injection).

`_format_cameras_arg` (`rollout.py:500-520`) builds lerobot's CLI dict by f-string:

```
rollout.py:518-520
    body = ", ".join(f"{k}: {v}" for k, v in remapped.items())
    parts.append(f"{name}: {{{body}}}")
return "{" + ", ".join(parts) + "}"
```

`name` is the checkpoint's `observation.images.*` feature key, forwarded verbatim by both frontends as
`m.requestKey` (`InferenceModal.tsx:387-400`, `DeployPanel.tsx:528-541`). A feature name containing `,`,
`:`, `{` or `}` — possible in an externally produced checkpoint, which the bimanual-collision branch already
treats as a real case (`InferenceModal.tsx:122-127`) — corrupts the argument and fails deep inside the
subprocess with an unrelated parse error rather than a legible pre-spawn rejection. No shell is involved
(`subprocess.Popen` with an argv list, `rollout.py:829-835`), so this is not an injection vector.

### N8 — P2 — Camera bindings are frozen cv2 indices; a replug that reuses an index silently rebinds a role to a different physical camera

**Status:** Confirmed statically; the physical outcome needs a hardware check.

Both surfaces key their bindings on the **cv2 index string** — `cameraKey = (cam) => String(cam.index)`
(`InferenceModal.tsx:50`, `DeployPanel.tsx:80`) — and send `camera_index: live.index`
(`InferenceModal.tsx:395`, `DeployPanel.tsx:536`), which the backend forwards verbatim
(`rollout.py:513-515`) with no re-resolution.

`useAvailableCameras` does refresh on USB hotplug, and there is a "drop a binding whose camera vanished"
effect on both surfaces (`InferenceModal.tsx:301-314`, `DeployPanel.tsx:426-439`) — but it tests
`liveCameraByKey(key)`, i.e. *"does index N still exist?"*. Per this repo's own documented macOS gotcha
(`CLAUDE.md`; `/available-cameras` returns a stable `unique_id`, `server.py:2180`), removing one device
**renumbers the rest**. After such a replug, index `1` usually still exists — pointing at a *different*
physical camera — so the stale-binding effect does not fire and the role is silently rebound.

The stable identity is available and discarded: `AvailableCamera` carries only
`index / name / deviceId / available` (`frontend/src/hooks/useAvailableCameras.ts:4-9`); the backend's
`unique_id` is not surfaced. The auto-bind path does prefer the stored `device_id`
(`InferenceModal.tsx:289-291`), but only for the *initial* bind — retention and submission are index-only.

**Impact.** A policy can be handed the wrong camera on each observation key: a silently wrong run rather
than an error. This is the inference-side analogue of the camera-identity re-anchoring the recording flow
received (`c4e5198 feat(cameras): identity re-anchoring`); no equivalent exists in `rollout.py`.

### N9 — P3 — The final `proc.wait()` on the stop path has no timeout

`rollout.py:1042`: after `proc.kill()`, `proc.wait()` is called with no timeout, so an unkillable child
(e.g. blocked in an uninterruptible serial ioctl) hangs the `/stop-inference` request thread indefinitely.
The UI has no timeout of its own — `handleStop` awaits the request (`InferenceSessionDialog.tsx:237`) and
leaves the button pinned at "Stopping…" (`:445`).

### N10 — P3 — Two independent Stop surfaces can double-signal the child, turning a clean user stop into a reported failure

The terminate/wait sequence runs **outside** `_state_lock` while `inference_active` is still `True`
(`rollout.py:1035-1044` — the lock is released at `:1034` and re-taken at `:1046`), so a second
`/stop-inference` arriving in that window passes the `if not inference_active` check (`:1007`) and issues a
second `proc.terminate()`. Two Stop surfaces are on screen simultaneously — the dialog
(`InferenceSessionDialog.tsx:231-247`) and the studio panel underneath it
(`DeployPanel.tsx:581-595`) — plus the hung-run watchdog (`InferenceSessionDialog.tsx:192-209`); each has
its own independent in-flight guard, so they do not see each other.

lerobot's handler escalates on the second signal: `if self._counter > 1: sys.exit(1)`
(`.venv/…/lerobot/utils/process.py:57-71`). **Steelmanned:** `sys.exit` raises `SystemExit` in the main
thread, and `finally: strategy.teardown(ctx)` (`.venv/…/lerobot/scripts/lerobot_rollout.py:236`) still runs
— so this is **not** a torque-loss path. The consequence is reporting: the process exits `rc=1`, and
`_classify_outcome` (`rollout.py:573-586`) reports a clean, user-requested stop as `failed` (or
`ran_with_warning` if the mined text happens to match a cleanup marker).

---

## Contradictions to the brief

1. **`frontend/src/pages/Inference.tsx` does not exist on `main`.** The brief's frontend file list is
   correct for `main` (`InferenceSessionDialog.tsx`, `InferenceModal.tsx`, `DeployPanel.tsx`,
   `inferenceApi.ts`), but the *original audit's* frontend line references (I3, I8, I11) point at a deleted
   file. Verdicts above are re-anchored to `InferenceSessionDialog.tsx`.

2. **One original finding is FIXED, not merely re-ranked.** I8's backend half (the consume-once terminal
   payload) is genuinely closed on `main` by the `_last_result` idempotence rework, with a code comment
   naming the bug it fixes. The brief's framing ("re-assess every entry") anticipated this; flagging it
   explicitly because it is the only FIXED verdict.

3. **G2's stated severity is wrong on `main`.** The audit's "reachable only by a direct API caller" no
   longer holds — `NumberInput` does not clamp, so an unbounded run is reachable by typing `0` in the normal
   UI. This raises rather than lowers the priority.

4. **The brief's "arm-count validation before hardware is touched" holds; the general claim does not.**
   `_arm_count_mismatch` genuinely runs before the worker thread starts (`rollout.py:962-965`). But the
   *request-shape* validation it sits beside does not cover the bimanual right-arm fields, so "validated
   before hardware is touched" is true of arm count only — see **N6**.

## What could not be verified

- **Whether lerobot's teardown actually exceeds the 5 s SIGKILL budget (N1).** Requires timing a real stop
  on the SO-101 with cameras attached — explicitly out of scope (live hardware). Only the structural facts
  (3.0 s sleep floor + 150 serial writes + camera release, ordered *before* the torque-disabling
  `disconnect()`; no fallback release anywhere in `makermodslab`) are established here.
- **Whether a blank task degrades or hard-fails a language-conditioned policy (G1).** Policy-dependent;
  needs a live run.
- **Whether two roles bound to the same cv2 index fail to open or duplicate frames (G3).** Device- and
  backend-dependent.
- **The physical outcome of the index-reuse rebind (N8).** The static path is confirmed; confirming that a
  role actually receives the wrong stream needs a replug on real hardware.
- **Whether PRs #9 / #10 / #11 fully close I1 / I5 / I4.** The PR branches were not read (read-only brief,
  no branch switching). All three defects are confirmed present in `main`'s code; note in particular that
  I1 has a sub-case already fixed on `main` (`rollout.py:852-888`) and a *new* untracked-child path (**N2**)
  that a PR written against the older shape may not cover, and that I4 is broader than audited
  (auto-calibration, `auto_calibrate.py:210,518-519`).
- **Runtime interleavings generally.** No tests were run and no server was started, per the brief. Every
  concurrency verdict above is a code reading of the lock/ordering structure, not an observed race.

## Suggested regression scenarios (additions to the original list's coverage gaps)

- Stop where the child needs >5 s to tear down: assert torque is released (or a fallback fires) rather than
  SIGKILL-and-forget.
- Stop racing the spawn: assert no untracked process survives the abandon branch.
- `DELETE /jobs/{id}` while an inference is running that job's checkpoint: assert 409.
- Status endpoint returning 5xx for N ticks while a run is live: assert a Stop control is still reachable
  and the exit guard is armed.
- Run ends by `--duration` with no status poller: assert the slot is released.
- `duration_s = 0` submitted from the UI's number field: assert rejection.
- Camera replug that reuses a cv2 index between picker-open and Start: assert the binding is invalidated.
