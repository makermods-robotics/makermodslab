> ## 📦 HISTORICAL RECORD — content folded into [`../CURRENT 23-07-26.md`](../CURRENT%2023-07-26.md) (2026-07-26)
>
> **Do not triage from this file.** This ranking's substance is already synthesized throughout
> [`../CURRENT 23-07-26.md`](../CURRENT%2023-07-26.md) — its P0s, its per-module P1 ordering and its
> cross-cutting fix clusters. The two items that had no home there were folded in on 2026-07-26 as
> **"Folded in from `archive/RANKING.md`"**: the ruled-out **Training #1** ("cloud jobs never reach terminal
> state" — NOT A BUG) and the **stale-venv testing caveat** attached to Recording #1. CURRENT is the single
> place bugs are organized.
>
> This file is **retained as the historical record of its method and date**: a **2026-07-14 forensic,
> file:line-traced ranking verified against `andrew` (pre-redesign)**, HEAD `257580f` + the uncommitted 0.6.0
> bump. Read it for *how and when* the backlog was ranked, not for current state — `main` is the redesign line
> and is **not** a descendant of what was audited here, so CURRENT reopens rather than closes these entries.
>
> Two corrections to the header this file used to carry: the two P0s below are **not** simply "addressed by the
> merge" on `main`, and the private-vs-public upload default is **no longer an open decision** — it is
> **DECIDED: public** (2026-07-26), which retired CURRENT's P0-2. See CURRENT → "Descoped by product decision".
> Recording #1's finding below is also **OBE** — CURRENT records it as RESOLVED by the lerobot 0.6.0 port; only
> its venv caveat survives.
>
> *Note: this file was moved into `archive/` after it was written; relative links in the body below were
> authored against the old top-level path and may not resolve from here. They are left as authored.*

---

# Consolidated P0/P1/P2 Ranking — all six module bug lists  *(pre-redesign backlog)*

Verified 2026-07-14 against the working tree on `andrew` (HEAD `257580f` + the uncommitted
lerobot v0.6.0 bump/refactor). Every entry below was independently re-traced in code by a
verification agent (file:line evidence in each module's list); priorities are
coordinator-ratified. 79 items assessed: **2 P0 · 27 P1 · 48 P2 · 1 ruled out · 1 already
fixed in tree**.

Rubric: **P0** = hardware safety, dataset loss/corruption, silent publication of private
data, or main path completely broken. **P1** = wrong behavior in common flows with real
consequences or no workaround. **P2** = edge cases, cosmetic, polish.

## P0 — fix before anything else

> **Status 2026-07-14 (evening): both P0s are FIXED in the uncommitted tree** (unit-tested,
> full suite 866 green; live validation pending with the rest of the tree). #1: visibility
> toggle default-private + hf_cloud fallback private=True. #2: fixed together with R5 —
> resume discard deletes nothing, terminal status retains saved_episodes. Details in each
> list's entry ("In progress").

1. **[Dataset #1] Cloud-training notice promises a PRIVATE upload; both upload paths publish PUBLIC.**
   `LocalDatasetCloudNotice.tsx:77-83` vs `Training.tsx:456` → `record.py:1104` and
   `hf_cloud.py:671`. Deliberate backend public-default policy that the frontend notice
   was never updated to match. Silent publication of camera footage; GFW makes it hard to
   notice or undo. Fix must reconcile BOTH push sites + `set_dataset_visibility`.
2. **[Recording #6] Resume + "Discard & exit" deletes the ENTIRE dataset (latent).**
   `Recording.tsx:520-535` → `record.py:849-885` has no resume-awareness; removes
   pre-existing episodes, not just the current take. Currently unreachable only because
   Recording #5 keeps its gate false. ⚠ **Coupled: fixing #5 unmasks this. Fix #6 with or
   before #5, in the same change.**

## P1 — by module

**Teleoperation** (top-ranked first)
- T7 Failed startup can strand an energized follower AND drops the "torque may still be
  enabled — unplug" warning (startup path lacks the per-motor hardening; F.1 precedent).
- T1 Done/page-leave latches "stopped" on a stop *attempt*; failure never re-arms the guard.
- T2 Rest-return failures are discarded; session reports `outcome=ok`.
- T3 Timed-out second stop orphans a live worker; a new session can be admitted and corrupted.
- T4 Cross-feature hardware ownership not atomic; calibration/wiggle entirely unguarded
  (see cross-cutting: the c737a30 mutex never shipped on this branch).

**Recording**
- R5 Terminal status erases `saved_episodes` (hides upload path; also gates R6/P0).
- R2 Logical start-rejection returns HTTP 200; frontend proceeds as success.
- R3 Setup failure leaks serial port + cameras (resources only — torque verified OFF).
- R4 Episode-duration limit silently discards a completed take and retries.
- R7 Done/Quit disarm the leave-safety before the stop is acknowledged.

**Model Training**
- MT11 Cloud stop reports canceled before/regardless of remote cancel — paid GPU can keep burning, unmonitored.
- MT9 Cloud watcher can permanently publish a partial checkpoint (resume path mitigated; inference/fine-tune not).
- MT10 Trainer crash after MakerMods Lab restart is finalized as "done" (reattached returncode=0).
- MT4 Cloud→Local resume switch deterministically launches a doomed run (no config_path).
- MT6 Cloud W&B resume omits the API key (form always initializes wandb off).
- MT2 Fine-tune silently drops the selected Hub checkpoint step (loads repo root).
- MT3 Final root policy hidden when periodic checkpoints exist (completed model unusable from card).
- MT7 Generic job delete bypasses the active-inference guard that the model card enforces.

**Inference**
- I1 Stopped startup worker keeps touching hardware / can clobber a newer session's state
  (P0-adjacent subcase: untracked child = live policy with no stop handle; no torque enabled).
- I2 Stop reports success and loses the process handle when termination fails.
- I3 Failed explicit Stop permanently disarms the page-leave guard (manual retry still works).
- I4 Non-atomic feature exclusion + calibration omitted (same family as T4).
- I5 FastAPI shutdown doesn't stop the inference child (orphan on --reload/pid-kill — common in dev).

**Configuration Setup**
- C1 Failed/stopped follower auto-recalibration deletes the previous valid calibration file.
- C2 Canceled manual calibration leaves servo homing offsets diverged → trips the next
  session's arm-identity check (stronger than the list stated).
- C3 Grossly incomplete ranges are saveable (backend rejects only min==max).
- C4 Manual-calibration singleton admits overlapping workers (unlocked check; 5s join timeout).

## P2 — 48 items (see each module list for annotations)

Config 9 (C5-C10, D1-D3) · Teleop 6 (T5, T6, T8-T11) · Recording 8 (R8-R15) ·
Training 14 (MT5, MT8, MT12-MT23) · Inference 11 (I6-I12, DG-a-DG-d).
Notable deliberate-design downgrades: T5/R12 fail-open identity check (documented, tested —
escalate as a product decision, not a code fix); T6/R10 motor-power warnings (toasted,
degraded-but-safe); I10 overload classification (documented subprocess limitation).

## Ruled out / already fixed

- **[Training #1] "Cloud jobs never reach terminal state" — NOT A BUG.** The installed hub
  returns `status.stage` as a plain string; the original audit probed the enum in isolation.
  Recommended hardening: `stage in _TERMINAL_STAGES`. Definitive close-out: one live cloud job.
- **[Recording #1] "v0.6 migration breaks recording" — INVERTED.** The uncommitted
  `rgb_encoder`/`depth_encoder` migration is CORRECT for lerobot 0.6.0; committed HEAD is the
  side that would crash. The audit ran against a stale venv. ⚠ Reinstall the venv
  (`pip install -e .`) before hardware testing.

## Cross-cutting fix clusters (fix once, close many)

1. **Hardware-session arbiter** — T4, I4, C4, R9, config D1, plus the missing calibration↔
   session mutex (PROJECT.md documents c737a30 as shipped; it is NOT on this branch — lost in
   the rebrand). One server-owned session token + single lock closes the family.
2. **Stop-path integrity** — T1, T3, T7, R7, I2, I3, MT11: every feature has some variant of
   "report stopped before verified stopped." A shared stop-contract (verify, then report;
   never latch a guard before the ack) addresses all seven.
3. **Recording R5+R6 must land together** (P0 unmasking dependency).
4. **Hub checkpoint listing semantics** — MT2, MT3, MT12 share `_hub_checkpoints_from_files`
   / `_list_imported_hub` / `_resolve_finetune_pretrained_path`.
5. **Resume form lock** — MT4, MT5, MT6: resume renders fully-editable controls whose edits
   are ignored (or worse, honored only by the wrong layer).
6. **Cloud runner finalization/tail** — MT10+MT11 (returncode contract), MT16+MT17 (tail loop
   offset + dedup).
