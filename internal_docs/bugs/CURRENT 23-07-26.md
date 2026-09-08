# CURRENT — consolidated current-state bug doc (23-07-26)

> **2026-08-02 note — the per-module lists have been RENUMBERED; the IDs used below are stale.** This rollup stays frozen as of 07-26 and its body is left exactly as written. Each module list now uses **one continuous letter-prefixed sequence** and the `N` / `NEW-n` series no longer exists. Translate every ID below as follows: **Recording** N-k → R(15+k) (N1→R16 … N15→R30), R1–R15 and the D1–D5 design gaps unchanged; **Inference** N-k → I(12+k) (N1→I13 … N13→I25), I1–I12 and G1–G4 unchanged; **Teleoperation** N-k → T(11+k) (N1→T12 … N11→T22), T1–T11 unchanged; **Dataset** entries 1 and 2 → D1 and D2, N-k → D(2+k) (N1→D3 … N24→D26); **Configuration Setup** the ten confirmed bugs → their RANKING numbers C1–C10, NEW-k → C(10+k) (NEW-1→C11 … NEW-9→C19), RANKING D1–D3 design gaps unchanged; **Model Training** originals n → MTn (unchanged), NEW-k → MT(24+k) (NEW-1→MT25 … NEW-18→MT42). The lists also now carry a `**First identified:**` date on every entry, plus `**Resolved:**` where the entry attests a fix or a descope.

*Citations synced 2026-07-26 against `main` @ `ce42158`. **Only `file:line` coordinates were changed** — no prose, verdicts, severities, entry IDs or numbering were altered. References whose construct no longer exists at HEAD are marked inline `⚠ UNRESOLVED @ HEAD ce42158`.*

*Merge 2026-07-26: the unique content of `archive/CONFIRMED 22-07-26.md` and `archive/RANKING.md` was folded
into this file (two new sections, marked as such). **Additive only** — no existing entry was reworded,
re-verdicted, renumbered or re-sequenced, and no synced citation or `UNRESOLVED` mark was touched.*

**This is the single file to act from.** It supersedes [`archive/CONFIRMED 22-07-26.md`](archive/CONFIRMED%2022-07-26.md)
and [`archive/RANKING.md`](archive/RANKING.md) as the statement of
*what is broken now* — and, as of 2026-07-26, it **contains the unique content of the first two**: the
2026-07-22 bench run's findings and non-bug asks (see "Bench-run findings" below) and RANKING's ruled-out entry
and testing caveat (see "Folded in from `archive/RANKING.md`"). Those two files stay in `archive/` as historical
records of their method and date; there is no longer any reason to triage from them.

It is otherwise a synthesis of the six per-module **Bug List** files (linked in the module index below), each of
which now folds its 23-07-26 re-audit in place — the primary evidence, every file:line traced there.

## How to read this / where everything lives

- **Primary evidence = the six merged Bug List files** (module index at the bottom). Each was re-verified on
  **branch `main` at commit `c7d9f27`** ("Merge pull request #4 from makermods-robotics/redesign") and now
  carries its original entries plus the `main`-current verdicts and new findings in one place. This doc
  summarises them; it does **not** replace them. When you go to fix something, open the module Bug List for the
  full trace. (The standalone re-audit docs are archived under [`archive/`](archive/).)
- **Numbering is preserved, not reassigned.** Every ID keeps the identifier its source doc gave it, always
  with a module prefix so cross-module references stay unambiguous. New-finding `N`/`NEW` numbers are
  **per-module** — "Recording N2", "Dataset N1", "Inference N1" and "Config NEW-1" are four different bugs.
  Where two source docs number the *same* bug differently, both numbers are carried and the collision is
  called out (see "Numbering collisions" below). Nothing here is renumbered or re-sequenced.
- **Severity rubric** (unchanged across all docs): **P0** = hardware safety, dataset loss/corruption, silent
  private-data publication, or main path completely broken. **P1** = wrong behavior in a common flow, real
  consequence, no workaround. **P2** = edge/cosmetic/polish.
- **Branch/commit note.** The REVERIFIED docs were written at `c7d9f27`. `main`'s HEAD has since advanced
  (through `f8ca339`) with the **lerobot 0.6.0 port** (`c6d97e7`) plus UI-copy polish. That port is what
  resolves the "recording dead on arrival" finding — see **Resolved since the reports** below. It does *not*
  touch any of the open P0s.

---

## Why the whole backlog is live (read before triaging)

Two structural facts, together, mean **there is no "already handled" pile** — the entire pre-redesign backlog
plus every REVERIFIED finding is live on `main`:

1. **Branch lineage: `main` is the redesign line; the fixes were verified against `andrew`; neither branch is
   a superset of the other.** The original six Bug Lists (and `RANKING.md`) were forensically verified
   2026-07-14 against the **`andrew`** branch. `main` is the **`redesign`** line. `518ca56` (the audited
   commit) is **not an ancestor of `main`** (Training REVERIFIED confirms `git merge-base --is-ancestor` →
   false; the two diverge by ~9.4k insertions / 13.4k deletions across 132 files). So re-verifying "against
   `main`" **reopens** entries rather than closing them: any fix that only ever lived on `andrew` is simply
   absent from `main`. Concretely — Configuration Setup: **13/13 existing entries STILL REAL, 0 fixed**;
   Model Training: **nothing fixed, and MT16/MT17/MT24 reopened** (their `andrew` fixes are not on `main`).
2. **Nothing is in flight. PRs #5–#11 are CLOSED with `merged=no`** (`mergedAt: null` on every one, verified
   across three module docs). Several targeted specific entries — #5=Teleop T7, #6=Teleop T3, #7=Teleop T4
   (cross-feature mutex incl. calibration/auto-cal/wiggle), #8=Dataset #1 visibility, #9=Inference I1,
   #10=Inference I5, #11=Inference I4 / Config entry 4. **None merged, none on track.** `main` still contains
   every one of those bugs.

Combined: the redesign dropped UI wiring (the `CONFIRMED 22-07-26` regressions), the `andrew` fixes never
merged, and no PR is open to change that. **Treat the whole set as unfixed on `main`.**

---

## The open P0s — fix these before anything else

Ranked. Each cites the REVERIFIED doc it comes from; open that doc for the full trace.

> **Numbering note.** **P0-2 is gone** — the record flow's public-by-default upload is **intended product
> behavior** (user decision, 2026-07-26), not a bug. See "Descoped by product decision" below. Per this doc's
> no-renumbering rule the identifier is **retired, not reused**: the open set is **P0-1, P0-3, P0-4, P0-5**.

### P0-1 · Auto-calibration wipes servo EEPROM before measuring, with no restore on Stop/exception

**Source:** Configuration Setup REVERIFIED **NEW-1** (P0), compounding existing **Config entry 1** (= RANKING
**C1**). **This is the primary user-facing calibration button** (`RobotConfigDialog.tsx:1431-1438` — "Auto-calibrate"
is the default action; "Calibrate manually" is the secondary).

- Stage 0 of the vendored script destroys the arm's persistent calibration **before any measurement**: it
  writes `("Homing_Offset", 0)` to every motor (`makermodslab/vendor/feetech_autocal/auto_calibrate_script.py:726`,
  written at `:744`) and `bus.write_position_limits(m, 0, 4095)` (`:751`), both with `Lock=1` — i.e. into
  **EEPROM** — for all six motors (`:728-760`). *(Verified directly: the register table at `:715-727` and the
  `write_position_limits(m, 0, 4095)` / `Lock=1` writes are present.)*
- The calibration is written back only at the **natural end of a fully successful run** (`:906-925`). Neither
  terminal failure path restores anything: `KeyboardInterrupt` (what a Stop sends) → `_graceful_stop` +
  `safe_disable_all`, `return 130` (`:684-690`); `Exception` → `safe_disable_all`, `return 1` (`:691-699`).
  There is **no snapshot** of pre-run `Homing_Offset` / `Min/Max_Position_Limit` and nothing in MakerMods Lab
  restores them.
- Nothing re-writes calibration into the servos on a later session: MakerMods Lab connects with `calibrate=False`
  (`record.py:1283,1342`) and pinned lerobot's `MotorsBus.connect()` does not write calibration. So every
  reading and every goal position is silently shifted by the destroyed offset for the whole next session, and
  the firmware travel clamp is gone (`(0,4095)`).
- **Combined with Config entry 1 / C1** ("auto-recal deletes the previous valid calibration *file*" — STILL
  REAL, the `andrew` staging-dir fix is absent; `auto_calibrate.py:112-138`, called on all three non-success
  paths `:291-293`, `:306-307`, `:438-440`): **a single Stop mid-autocal destroys both the calibration file
  and the servo state.** The excluded arm-identity guard is the only thing that notices — and it
  warns-and-proceeds.
- *Cannot verify statically:* exact post-cancel register values on hardware (bench check: read
  `Homing_Offset` + limits, Stop mid-run, re-read). Control flow is unambiguous.

### P0-3 · Inference Stop can leave the follower torque-enabled; a second path leaks a live untracked subprocess

**Source:** Inference REVERIFIED **N1** (P0) and **N2** (P0/P1). Both about torque left on with the backend
reporting idle.

- **N1 — SIGKILL mid-teardown.** `handle_stop_inference` gives the child a hard 5-second budget then SIGKILLs
  it (`rollout.py:1036-1042` — `terminate()` / `wait(timeout=5)` / `kill()`; *verified*). The teardown that
  disables torque cannot finish that fast: pinned lerobot's `_return_to_initial_position` is
  `duration_s=3.0, fps=50` = **150 serial round-trips on a 3.0 s sleep floor**, and only *after* it does
  `robot.disconnect()` (which disables torque) run. On a real rig (plus a bimanual second bus + camera
  release) this plausibly exceeds 5 s. **MakerMods Lab has no fallback torque release for inference**: `rollout.py`
  does **not** import `torque.py` — *verified: `rollout.py` imports `motor_power` (`reset_torque_limit`,
  `clear_goal_velocity`) but never `force_disable_bus_torque`; the only consumer of that helper in the package
  is `auto_calibrate.py`.* No fallback on the stop path, abandon path, `_fail_startup`, or status
  finalisation. After a Stop the follower can be left energized indefinitely with both UIs reporting idle; the
  stop still returns HTTP 200 (I2), and `DeployPanel.tsx:584-585` toasts "winding down".
- **N2 — untracked live subprocess.** When a stop races the spawn, the abandon branch
  (`rollout.py:878-888`) does `terminate()` + `wait(timeout=5)` with **no `kill()` fallback**, suppresses the
  timeout, never started the stdout pump, and `proc` was never assigned to `_inference_proc` — **no handle
  survives anywhere**. During `strategy.setup(ctx)` (policy load + `robot.connect()`) nothing polls the
  shutdown event, so a multi-GB load outlasts the 5 s window: the child then connects and **energizes the
  follower**, with the backend reporting idle, the mutex open, and no way to stop it.
- Related un-ranked-here but same family: Inference **N3** (P1) — `DELETE /jobs/{job_id}` bypasses the
  in-use guard and `rmtree`s a checkpoint a live rollout is reading (`server.py:1664-1677`, `jobs.py:1555-1556`);
  and **I1** (STILL REAL) — cancellation checked only after `_prepare_robot` returns, so a stop landing inside
  it still opens buses and rewrites `Torque_Limit`.

### P0-4 · Recording R5/R6 dataset-loss coupling is live — fixing R5 unmasks whole-dataset deletion

**Source:** Recording REVERIFIED **R5** and **R6** (R6 was the original latent P0 in `RANKING`). Both
`andrew`-side fixes are **absent from `main`**. **They must ship as one change.**

- **R5 (P1, STILL REAL).** The worker's `finally` zeroes the only counter (`saved_episodes = 0`,
  `record.py:654`), and the count is emitted only inside `if recording_active and recording_config:`
  (`record.py:833-836`) — but `recording_active` is set False at `:650` *before* the zeroing, so no terminal
  poll can ever observe the true count. There is no `last_session_saved_episodes` on `main` (the `andrew`
  global and its tests are absent). Consequence: every clean session reports "0 episodes"; on a
  failed/warning finish `keptSomething` is permanently false, so "Keep episodes & continue" / "Discard &
  exit" never render — recovery is blocked and the count is false.
- **R6 (P0 if R5 is fixed without R6; STILL REAL, STILL LATENT).** `discardAndExit()`
  (`RecordingSessionDialog.tsx:512-527`) posts to `/delete-dataset` with **no `resume` check** even though
  it has `resume` in scope; `handle_delete_dataset` (`record.py:859-897`) `shutil.rmtree`s the whole dataset
  directory (`:887`), resume-unaware. The backend's *worker-side* discard **is** resume-safe and tested
  (`_discard_session_dataset` / `_discard_empty_dataset` early-return on resume) — the frontend button
  bypasses it. The button renders only when `keptSomething` is true, which R5 makes permanently false — so
  **fixing R5 unmasks R6 and arms whole-dataset deletion of pre-existing episodes.** Additional current gate:
  `CollectPanel.tsx:247` hardcodes `resume: false`, so a resume session can today only be created by a direct
  `POST /start-recording` (the UI resume entry point was dropped — this is `CONFIRMED 22-07-26` #5).
- **Phantom-dialog aggravator (Recording R2, escalates here):** a logically-rejected `/start-recording`
  returns **HTTP 200** (`server.py:838-841`; `record.py:474-496`), and the dialog trusts `response.ok`
  (`RecordingSessionDialog.tsx:343`), so a phantom dialog arms the page-leave guard with
  `/stop-recording?discard=true` against the **real live session** — deleting its fresh dataset wholesale.
  Reachable via the second-tab "Use this tab" flow (`SingleTabGuard.tsx:120-140`).

---

## Added after this report — 2026-07-24 working session (against `main` @ `ce42158`)

This doc's title date and the P0s above are the 23-07-26 state and are **left as written** (except that P0-2
was later descoped — see "Descoped by product decision"). A working session on **2026-07-24** added eight
entries across four module lists; one of them is a **further P0**, numbered **P0-5**. Every
entry in every list now carries a `**First identified:**` field (2026-07-14 = original forensic audit against
`andrew` @ `518ca56`; 2026-07-23 = the re-audit; 2026-07-24 = this session).

### P0-5 · A bare dataset name lands at the same level as MakerMods Lab's state dirs, and delete `rmtree`s whatever is there

**Source:** **Dataset N20** (P0), with **Dataset N21** / **Recording N10** as the enablers.

- `validate_dataset_repo_id` (`utils/config.py:989-1005`) accepts a **single-segment** id and reserves
  nothing, so a bare name resolves to `<HF_LEROBOT_HOME>/<name>` — the level that holds `calibration/`,
  `robots/`, `ports/`, `outputs/`, `hub/`, `makermodslab_biso/`, `makermodslab_models/`, `merge_logs/`,
  `inference_logs/`.
- `handle_delete_dataset` (`record.py:859-897`) guards **only** traversal (`target == root or root not in
  target.parents`, `:870`) plus the `_dataset_in_use` busy-check, then `shutil.rmtree(target)` (`:887`). It
  never asks whether the target is a LeRobot dataset (`_is_dataset_dir` is not consulted). Deleting a
  "dataset" named `calibration` removes **every calibration profile**; `robots` removes **every robot
  record**; `outputs` removes the local job registry. Two-segment ids are unaffected.
- Bare names arise in ordinary use: disk import defaults to the source folder's basename with no namespace
  (`datasets.py:1091`, dialog placeholder *"Defaults to the folder name"*), and Collect falls back to a bare
  name whenever the user is not logged in (`CollectPanel.tsx:180-183`) — i.e. on an offline station.
- Same-family as P0-4 (dataset loss via an unguarded `rmtree`) and shares the delete path with **R6**; the two
  sibling discard paths (`record.py:936`, `:1002`) carry the same traversal-only guard.

### The rest of the 2026-07-24 set

- **Dataset N21** (P1) neither import nor the logged-out record path prefills the Hub namespace → bare names
  (the P0-5 precondition) and datasets that cannot be pushed without a rename. **Dataset N22** (P2)
  `import_local_dataset` copies synchronously inside the request (acknowledged in the docstring).
  **Dataset N23** (P2) `DatasetCard` `truncate`s the dataset name — real repos differing only by suffix render
  identically — and repeats the same string twice more on the card. **Dataset N24** (P2) the Launchpad library
  sheet lists datasets with no search, no filter and no `CappedGrid`, while the studio's `DatasetLibraryList`
  has all three over the same data.
- **Recording N10** (P2) the logged-out Collect fallback that mints the bare name (cross-filed as Dataset N21).
- **Model Training NEW-11** (P2) `tests/test_models.py` reads the developer's real
  `saved_custom_models.json` / `hidden_models.json` — `tmp_lerobot_home` patches every other state path but not
  those four constants — so 7 tests pass or fail depending on what the developer has pinned locally.
- **Configuration Setup NEW-8** (P2) the `lelab_biso` → `makermodslab_biso` staging rename shipped without a
  migration, unlike `lelab_models` (`models.py:295-308`) and legacy `outputs/train` (`jobs.py:986-1035`).

### Upstream leLab `308c7c3` (2026-07-23) — checked, nothing filed

Upstream fixed three things; none produced a new entry here. **(1) `repo_id` containment in
`dataset_repair.py`: N/A** — this fork has no `dataset_repair.py` and no repair flow at all (`grep -rn repair
makermodslab/` is empty). **(2) Device setup outside the try whose `finally` disconnects: already filed** as
**Recording R3** — still real on this fork (setup at `record.py:1292-1408` sits outside the `try:` at `:1430`),
so R3 now has a known upstream fix to port rather than being a new finding. **(3) Dropped episodes leaving
orphaned frames in the data shards + a stale `stats.json`: N/A** — the trimming that stranded those rows lives
in the repair flow this fork does not have. *Not filed, needs a call:* this fork never calls
`dataset.finalize()` anywhere (upstream added it in `cb80520` "finalize recorded datasets so uploads stop
404ing"); the pinned lerobot has `DatasetWriter.__del__` as a safety net, so whether that is a live defect here
needs a runtime check.

---

## Bench-run findings — folded in from `archive/CONFIRMED 22-07-26.md` (2026-07-22 live run)

**Provenance — why these have no equivalent in the six module lists.** Everything in this section was found by
a **human driving the app** against the merged tree during the [22-07-26 Full Test](../tests/22-07-26%20Full%20Test/Pathway%20Test%20Checklist.md)
pathway run — *not* by reading code. That is exactly why the module audits (which are code-forensic) could not
surface them: a code audit sees the backend that survived the redesign merge, it cannot see that the **UI entry
point was dropped**. Most of the P1s below are that shape — backend present, UI door missing.

IDs are the bench run's own numbers, prefixed **`Bench #n`** so they never collide with the module `N`/`NEW`
sequences. Nothing is renumbered. Gaps in the sequence are deliberate — see "deliberately not carried over"
at the end of this section.

### Bench #1 · Arm-identity guard is completely non-functional — zero protection against a bad calibration↔arm pair *(P0 as filed)*

Run: Phase 1 "Hardware safety" all `[-]` / *"Complete fail"*; user note Big #0. The guard
(`makermodslab/arm_identity.py`) is meant to fire before torque-enable and block a swapped leader/follower or a
wrong-calibration arm. It isn't guarding at all → **an energized arm can run with the wrong calibration**
(violent motion / hardware damage / injury). Highest priority as filed; nothing else matters if arms can be
energized wrong. Related pre-redesign entries: Teleop **T4** (hardware-ownership mutex "never shipped") and the
cross-cutting `c737a30` mutex. The guard's severity *design* (block-with-override vs warn) is itself open — see
the stop/identity design discussion.

> ⚠ **Open coordinator decision — severity of Bench #1 vs Teleop T5 / Recording R12. Not resolved here.**
> The bench run rates the identity guard **P0, "not guarding at all"** (observed on hardware). CURRENT carries
> identity as a documented **fail-open** whose severity was **excluded by the re-audit's brief** (Teleop **T5**,
> Recording **R12**). Those are *different claims*: "excluded by brief" is not "found working", and neither
> supersedes the other. Bench #1 is therefore merged **as filed, with its P0 rating intact**, and is
> deliberately **not** folded into the numbered open-P0 set — it is **not** P0-6, nothing is renumbered, and it
> is **not** downgraded to match T5/R12. A coordinator call is required to reconcile the two.

### Bench #2 · Teleop camera visualization gone *(P1 — redesign regression)*

Run: Phase 2 ⭐ line `[-]`; Big #3. Andrew had live camera previews during teleop; the redesign teleop UI shows
none. Ledger parity item (camera previews). **Root cause is now known** and is recorded in "Corrections"
below: the `/teleoperation` route is **orphaned** — it exists but nothing navigates to it, so its camera panel
is never mounted; the live teleop window renders no cameras. The backend feed was never the problem. Xref:
Teleoperation **T10** (camera-preview `mediaDevices` throw — *not reachable from teleop* precisely because the
panel is orphaned).

### Bench #4 · Calibration "download config" control dropped in the redesign *(P1 — redesign regression)*

Run: Phase 1 line `[-]` ("No download"); Big #4. The calibration-JSON export/download control present before the
redesign is absent from the redesign calibration UI. Xref: Configuration Setup module list (no existing entry
covers the missing export surface — the module audit read backend calibration handling, not the dropped control).

### Bench #6 · Dataset library ops dead: create / add-Hub / download / import-from-disk / rename / delete *(P1 — redesign regression)*

Run: Phase 4 line `[-]`; Smaller #6. The whole dataset-library action set is non-functional from the redesign UI.
Xref: Dataset **N4** — note that N4 covers the **delete** surface *only* ("no UI at all to delete a local-only
dataset"); create / add-from-Hub / download / import-from-disk / rename are additional and unfiled elsewhere.

### Bench #7 · Model library ops dead: add-Hub / download / import-checkpoint / upload / delete *(P1 — redesign regression)*

Run: Phase 5 "Model library" line `[-]`. Same class as Bench #6, for models. Xref: Model Training module list.

### Bench #8 · 0.6.0-trained models can't deploy on a 0.5.2 runtime *(P1 — compat constraint, not a code bug)*

Run: Phase 6 ⭐ Deploy `[-]`; Big #2. `draccus` hard-rejects the `pretrained_revision` field that a 0.6.0 policy
config carries. This is an **expected forward-incompatibility across the version boundary** — per the entry's own
text it is documented as a **compat constraint, not a merged-code bug**: deploy those models on the 0.6.0 runtime,
not on a 0.5.2 one. (Confirmed during the bench session. Note for readers arriving after the fact: `main` has
since pulled the 0.6.0 port — see "Resolved since the reports" — which changes which side of the boundary `main`
sits on, but not the constraint itself.)

### Bench #9 · `makermodslab --stop` can't stop a normally-started instance *(P1)*

Serve line `[✗]`. Prod mode runs uvicorn **in-process**, so the launcher's cmdline lacks `makermodslab.server` and the
identity match misses it — only `--dev` orphans and manual uvicorn invocations are caught. Fix: also recognise
the `makermodslab` / `makermodslab-station` entry-point process that is listening on our port. (Confirmed during the
bench session.) *Citations checked @ HEAD `ce42158`, entry unchanged:* `_run_prod` builds `uvicorn.Config` /
`uvicorn.Server` in the main process (`makermodslab/scripts/makermodslab.py:322-330`, `server.run()` at `:358`);
`_identity_reason` (`:174-190`) matches only `"makermodslab.server" in cmdline` or an orphaned
`multiprocessing.spawn` reload worker whose cwd is this checkout; `_find_makermodslab_pids` (`:205-230`) therefore
files a prod-mode instance under `strangers`, which are **reported but never terminated** (`:229`).

### Bench #10 · Intermittent "failed to write lock position" (motor comm) *(P2)*

Big #1. Usually resolves on a second try — motor-bus write flakiness. Investigate whether the bounded-retry path
can absorb it.

### Bench #11 · Upload-to-Hub card: wrong placement + wrong episode count *(P2)*

Run: Phase 3 `[/]`; Smaller #2. Should live on the "+ New Skill" page, and it displays the wrong number of
episodes.

### Bench #12 · Model download progress doesn't display correctly *(P2)*

Run: Phase 6 `[/]`; Smaller #8.

### Bench #13 · Dataset info card is hard to read / needs more info *(P2)*

Run: Phase 4 `[/]`; Smaller #4.

### Bench #14 · No dropdown to select a dataset *(P2)*

Smaller #5.

### Bench #16 · End-of-recording: follower returns to start pose too quickly *(P2)*

Smaller #1. See the stop-behavior design discussion (gentle-return speed; whether recording should return at
all). **Xref, same family — not merged with these:** Recording **R11** (rest-pose failures reported clean) and
Recording **N3** (post-session release grace invisible; arm moving under power while the UI says "SESSION
COMPLETE"). All three concern the end-of-recording rest-return, but they are distinct findings — Bench #16 is
about the *speed* of a return that works, R11 about a return that *fails silently*, N3 about the *UI during* it.
Keep them separate.

### Deliberately not carried over from the bench run (do NOT re-merge)

The gaps above are intentional; each is already handled elsewhere in this doc, and re-merging them would
resurrect a finding this doc deliberately corrected.

- **Bench #3** "Torque / motor-power slider gone" — **DOES NOT REPRODUCE**; see "Corrections" #1 below.
- **Bench #5** "Cannot resume runs (recording AND training)" — already carried **inside P0-4** (the
  `CollectPanel.tsx` `resume: false` hardcode / dropped resume entry point).
- **Bench #15** "Power telemetry not implemented" — **IMPRECISE** (backend works and broadcasts; the frontend
  consumer count is zero); see "Corrections" #2 below.
- The bench run's **"Open decisions"** item (cloud-upload private vs public) is **decided and already recorded
  here** — see "Descoped by product decision" below. It does not need to move again.

### Feature requests / design asks from the bench run — tracked, NOT bugs

These are asks, not defects. They are recorded here so the bench run's non-bug output has a home; do not triage
them against the P0/P1/P2 rubric.

- Pause/exit for **Detect & Wiggle** (currently no way out but waiting it out). (Big Feature #1)
- Rethink saving robot configs — **explicit save vs autosave**. (Big Feature #2 / Convenience #2)
- **"+ New Calibration" doesn't fit multi-arm** — either per-arm subprocess (concurrent) or remove it.
  (Convenience #1)
- **When should the user be allowed to name a calibration file?** (Convenience #3)
- **Batch delete.** (Convenience #4)
- **More detailed logging** — e.g. "waiting for HF to allocate a GPU". (Smaller #3)
- **Rebuild the frontend on every PR accept?** (Extra Notes #0)

---

## Corrections to `CONFIRMED 22-07-26.md` (act on these, not on the originals)

`CONFIRMED 22-07-26.md` is the record of the 2026-07-22 bench run and is otherwise useful, but three of its
entries are wrong or imprecise. A banner now sits at the top of that file; the corrections in full:

1. **`CONFIRMED` #3 "Torque / motor-power slider gone" — DOES NOT REPRODUCE.** On `main` the slider is present
   and wired: `RobotConfigDialog.tsx:1215-1276` (percent↔raw conversion, `DEFAULT_MOTOR_POWER = 38`, dirty
   tracking, `patch.motor_power`). Either the tier-1 UX commits (`fc85b40` / `e7f7d18`) fixed it or #3 referred
   to a different surface. Source: Teleoperation REVERIFIED, T6 note + contradiction #3. Do not re-act on it.
2. **`CONFIRMED` #15 "Power telemetry not implemented" — IMPRECISE. The backend works; the frontend consumer
   count is zero.** Backend computes and broadcasts `follower_currents_ma` (`teleoperate.py:141-178`,
   `:732-735`) and logs the session summary at INFO — but a repo-wide grep finds **no frontend consumer**. So
   "not implemented" is true of the UI only; the fix is a frontend wiring job, not a backend one. Source:
   Teleoperation REVERIFIED contradiction #5.
3. **`CONFIRMED` #2 "Teleop camera visualization gone" — REAL, root cause now known.** The `/teleoperation`
   route still exists (`App.tsx:41`) but **nothing in the app navigates to it** — it is orphaned; its camera
   panel (`pages/Teleoperation.tsx:211` → `TeleopCameraPanel`) is mounted nowhere the live surface reaches.
   The live teleop window (`TeleopDialog.tsx`) renders no cameras. The backend feed was never the problem.
   Source: Teleoperation REVERIFIED "Structural change" + contradiction #4.

---

## Descoped by product decision — 2026-07-26 (do NOT re-file)

- **P0-2 · "Recording auto-publishes footage to a PUBLIC Hub repo" — NOT A BUG. Public-by-default upload is
  intended.** Filed independently as Recording **N2** and Dataset **N1**, both now descoped. The chain the
  entry traced is real and unchanged (`StudioContext.tsx:42` `pushToHub: true` → collapsed "Advanced
  parameters" → `CollectHandoff.tsx` `start([], false)` on mount → `{private: false}` → public repo) — but
  the outcome it treated as a defect is the product's intent: recorded datasets are meant to be
  discoverable, published without a per-session confirmation step. The rubric line it was ranked under
  ("silent publication of private data") does not apply to a publication the product means to perform.
  **Both IDs are retired, not reused. Do not re-file this chain from a future audit.**
- **Knock-on: the visibility question is now CLOSED, and closed on the "public" side.** Every doc that called
  the dataset-upload default an *open product decision* — Dataset entry 1 / RANKING P0 #1, Dataset N10,
  Training NEW-3, `CONFIRMED 22-07-26`'s "Open decisions" — is answered: **public is the intended default at
  every push site.** What survives is narrower and non-P0:
  - **Dataset entry 1 stays open as a COPY bug, not a policy bug.** `LocalDatasetCloudNotice` still tells the
    user the upload will be *private* while every path publishes public. The fix is now unambiguous — correct
    the notice to match the policy, rather than the reverse.
  - **Centralization stays open.** `main` still decides visibility at three independent sites. Open
    [PR #19](https://github.com/makermods-robotics/makermodslab/pull/19) already implements exactly this: one
    `DATASET_DEFAULT_PRIVATE = False` constant in `utils/config.py` that every push site reads, with explicit
    user choices overriding it. That PR is now the right shape for the settled policy — merging it closes the
    cross-cutting item below.
  - **Optional, unfiled:** the "Push to Hugging Face Hub" helper copy still never says *public*. Under the
    settled policy that is a disclosure nicety, not a defect; noted here, deliberately not filed.

## Resolved since the reports (do NOT carry forward as open)

- **Recording N4 / R1 — "recording is dead on arrival" — RESOLVED by the lerobot 0.6.0 port.** The Recording
  REVERIFIED doc (written at `c7d9f27`) flagged an environment/pin mismatch: the checkout passed the lerobot
  0.5.2 `vcodec=` API while the installed package was 0.6.0, so recording would raise
  `AttributeError: 'DatasetRecordConfig' object has no attribute 'vcodec'`. `main` has since pulled the port
  (`c6d97e7`): `pyproject.toml` now pins `lerobot ... @ v0.6.0`, and `record.py` imports `RGBEncoderConfig`
  and uses `rgb_encoder=RGBEncoderConfig(vcodec="auto")` for create and `rgb_encoder=cfg.dataset.rgb_encoder`
  / `depth_encoder=cfg.dataset.depth_encoder` for create+resume (`record.py:28`, `:405-418`, `:1211-1217`,
  `:1244-1250`). *Verified.* **Resolved-by-pull.** Note this is the only "resolved" item — everything else in
  the reports stands.

### Verdicts that may need re-checking because they were computed against the pre-port checkout

The Recording and Model Training REVERIFIED docs were verified at `c7d9f27`, **before** the 0.6.0 port. Their
verdicts that turn on the lerobot pin should be re-read against HEAD:

- **Recording N1 (P1) — re-record during reset discards the next episode — now MORE reliable, not less.**
  N1's own caveat said it depends on lerobot's `record_loop` never clearing `rerecord_episode`, read from the
  *installed* 0.6.0 package, and asked to "confirm against the pinned source" because `main` then pinned a
  different commit (`82dffde`). With the port, `main`'s pin **is** 0.6.0 — the pinned source now matches what
  the agent read, so the caveat resolves in the direction that **keeps N1 live**. Still worth a bench confirm,
  but no longer a pin mismatch.
- **Model Training — the `--eval_freq` → `--env_eval_freq` argv divergence is now partly OBE.** The Training
  REVERIFIED "branch-topology" note said `main` still emits `--eval_freq` while `andrew` renamed it. HEAD has
  since renamed the field/flag to `env_eval_freq` (`train.py:123-130`, `:314-317`), matching the 0.6.0 CLI.
  So that specific divergence is closed; the MT entries that don't touch the pin are unaffected. Also the
  `sac` → `gaussian_actor` policy-type rename landed (`models.py`, `server.py`, `hf_cloud.py`,
  `utils/system.py`) — cosmetic to the bug set but relevant if any MT/inference verdict keyed on `"sac"`.

---

## Folded in from `archive/RANKING.md` — the residue with no other home

`RANKING.md`'s rankings are already synthesized throughout this doc (its P0s, its per-module P1 ordering, its
cross-cutting clusters). Only two things in it had no home here; both are durable, so they are carried:

### Ruled out — "Cloud jobs never reach terminal state" is NOT A BUG (do NOT re-file)

**[RANKING Training #1 = MT1.]** The installed `huggingface_hub` returns `status.stage` as a **plain string**;
the original audit probed the enum in isolation and concluded the terminal comparison could never match.
Recommended hardening was `stage in _TERMINAL_STAGES`. Definitive close-out remains **one live cloud job**.
This is the same ruled-out item the Model Training summary below records as "MT1 ruled-out (unchanged)"; the
reasoning is kept here so a future audit does not re-file it from scratch.
*Citations checked @ HEAD `ce42158`:* the recommended hardening is present — `_TERMINAL_STAGES` is an explicit
allowlist (`runners/hf_cloud.py:426`) and the poller coerces before comparing
(`stage_str = str(stage).upper()` → `if stage_str in _TERMINAL_STAGES`, `:743-746`). Verdict unchanged: not a bug.

### Testing caveat — reinstall the venv before hardware testing

**[Carried from RANKING Recording #1.]** That audit ran against a **stale venv**, which is what produced its
inverted verdict. ⚠ **Reinstall the environment (`pip install -e .`) before any hardware test or re-audit** —
otherwise the installed lerobot can differ from the pinned one and verdicts will be computed against a package
the checkout does not actually use. **Only the caveat is carried:** Recording #1's *substance* is OBE — this doc
records Recording **N4 / R1** as **RESOLVED** by the 0.6.0 port (see "Resolved since the reports" above). Do not
re-open the finding; do keep the methodology warning.

---

## Remaining findings by module (summaries — open the linked doc before acting)

Densities below are enough to triage without opening all six files. Every ID is the source doc's own; nothing
is renumbered. "STILL REAL" = reproduces on `main` as written.

### Configuration Setup → [Bug List](Configuration%20Setup%20Bug%20List.md)
Existing entries **1–13 all STILL REAL** (0 fixed). See the numbering-collision note below: entries **1–4** =
RANKING **C1–C4**, entries **5–13** = RANKING **C5–C10 + D1–D3**.
- **1 / C1** (P1) auto-recal deletes the previous valid follower calibration file — feeds **P0-1**.
- **2 / C2** (P1) canceling manual calibration doesn't restore servo state (no snapshot/rollback).
- **3 / C3** (P1) grossly incomplete ranges are saveable (backend rejects only `min==max`).
- **4 / C4** (P1) manual-calibration singleton admits overlapping workers (unlocked check; 5 s join). Overlaps
  closed-unmerged **PR #11**.
- **5–13 / C5–C10 + D1–D3** (P2) concurrent robot-record write races; auto-cal reports success with a missing
  file; imported-calibration validation accepts unusable profiles; nav discards unsaved drafts (narrowed);
  duplicate camera names collapse a camera; offline-station mislabels cached auth; **no global
  hardware-session exclusion** (entry 11, "WORSE THAN STATED", overlaps closed-unmerged **PR #7**);
  library-mutation unguarded during a live calibration; macOS camera-enum has no fallback.
- New: **Config NEW-1** (P0) → **P0-1**. **NEW-2** (P1) start teleop/record/inference while arms auto-calibrate
  under torque (bimanual concurrent-drive; overlaps PR #7). **NEW-3** (P1) calibration write-back bypasses the
  duplicate-port guard yet the record still reports `is_clean: true`. **NEW-4** (P1) camera identity binds to
  browser `deviceId` via fuzzy name-match; the backend's stable `unique_id` is discarded. **NEW-5..NEW-7** (P2)
  `clamp_motor_power` raises on NaN/Inf and can hide every robot; leader auto-cal leaves a permanent scratch
  copy; dead `ports/*_port.txt` persistence + unreferenced `/robot-port` endpoint.
- 2026-07-24: **NEW-8** (P2) the `lelab_biso` → `makermodslab_biso` staging rename shipped with no migration,
  unlike `lelab_models` and legacy `outputs/train`.

### Teleoperation → [Bug List](Teleoperation%20Bug%20List.md)
- **T1** (P1) Done/page-leave latch "stopped" on a stop *attempt*; now duplicated into the live dialog.
- **T2** (P1) rest-return failures discarded, session reports `outcome=ok`.
- **T3** (P1) timed-out second stop orphans a live worker (overlaps closed-unmerged **PR #6**; reachability
  narrowed to API-only, blast radius unchanged).
- **T4** (P1) cross-feature ownership incomplete/non-atomic — **strengthened** by the non-modal window
  (overlaps closed-unmerged **PR #7**).
- **T5** identity fail-open (code fact; severity **excluded by brief**). **T6** motor-power apply — **N/A
  SUPERSEDED** (`apply_motor_power` removed; teleop runs stock torque). ⚠ T6 note contradicts `CONFIRMED` #3
  (slider present) — see corrections.
- **T7** (P1, top-ranked teleop) startup cleanup discards "torque may still be enabled" (overlaps
  closed-unmerged **PR #5**).
- **T8** raised **P2→P1**: no second-stop "release now" control; Done now closes the whole window.
- **T9** (P2) "Live Robot Data" with no session. **T10** (P2) camera-preview throws when `mediaDevices`
  absent — **not reachable from teleop** (panel orphaned), re-file against Recording/Deploy/Inference.
  **T11** (P2) no immutable session identity; robot dropdown clickable mid-session.
- New: **Teleop N1** (P1) teleop is the only live-hardware surface with **no page-leave guard**. **N2** (P1)
  Done closes the only window while the arm is still energized and moving. **N3** (P1)
  `finish_pending_release()` failure return ignored by both start paths (record→teleop direction unsafe).
  **N4** (P1) non-modal window leaves the whole app interactive during a session (UI half of T4). **N5–N11**
  (P2) duplicated state machine + `/teleoperation` footgun; 13 s cleanup check reads process-global state;
  unbounded rest-return join; stop path mutates shared state without `_state_lock`; unbounded WS queue with a
  dead overflow branch; `strict=False` telemetry/bus mismatch; failed start leaves no status trace.

### Recording → [Bug List](Recording%20Bug%20List.md)
- **R5** + **R6** → **P0-4**. **R1** → resolved (see above). **R2** (P1) logical rejection returns HTTP 200
  → phantom dialog deletes the real session (feeds P0-4). **R3** (P1) setup failure after follower connect
  leaks port/cameras. **R4** (P1) episode-duration timeout discards the take and can retry forever (UI now
  mislabels it "auto-advance"). **R7** (P1) Done/Quit disarm the leave guard before the backend acks.
- **R8** (P2) resume can mutate an in-use dataset (reach reduced to API-only). **R9** (P2) hardware ownership
  non-atomic. **R10** (P2, SUPERSEDED as framed) narrower warning-drop remains. **R11** (P2) rest-pose
  failures reported clean. **R12** (P2) arm-identity read-failure fail-open (severity excluded). **R13** (P2)
  ended-session deletion ignores server result. **R14** (P2, under-rated → see N1) re-record accepted outside
  the recording phase. **R15** (P2) backend accepts out-of-contract params. **D1–D5** design gaps (durable
  session identity, episode review, frame-health observability, process-local recovery, single owner).
- New: **Recording N1** (P1) re-record during reset silently discards the *next* full episode (see re-check
  note above — now reliable). **N2** — **DESCOPED** (public auto-push is intended; see "Descoped by product
  decision"). **N3** (P1) post-session release grace is invisible and
  its controls report actions that never happen (arm moving under power while UI says "SESSION COMPLETE").
  **N4** → resolved. **N5** (P2) every recording control returns HTTP 200 on logical rejection. **N6** (P2)
  Quit re-opens browser camera previews while the backend still owns the cameras. **N7** (P2) reload/tab-close
  destroys every saved episode behind a generic browser prompt (coordinator may want higher). **N8** (P2) dead
  `test_mode` safety-shaped flag. **N9** (P2 minor) terminal status logs+prints every poll.
- 2026-07-24: **N10** (P2) a logged-out Collect session records into a **bare** dataset name — the
  precondition for Dataset N20 / **P0-5**. Also: **R3** is what upstream leLab `308c7c3` fixes (device setup
  moved inside the try whose `finally` disconnects) — a fix to port, not a new finding.

### Dataset → [Bug List](Dataset%20Bug%20List.md)
- **Dataset entry 1** (STILL REAL, **reframed** 2026-07-26) cloud-training notice says "private", both upload
  paths publish **public**; `main` moved *further* to a deliberate public-default policy
  (`hf_cloud.py:665-671`). **The policy question is now CLOSED — public is intended** (see "Descoped by
  product decision"), so this is a **copy bug, not a policy bug**: correct the notice to match the behavior,
  not the reverse. Superseded PR #8 → now open **PR #19**. This is also RANKING **P0 #1**, whose P0 framing no
  longer applies.
- **Dataset entry 2** (STILL REAL) in-use guard excludes cloud runs, uses raw string equality, and caps the
  registry scan at 200 (a long-running old job is invisible).
- New: **Dataset N1** — **DESCOPED** (public auto-push is intended; see "Descoped by product decision").
  **N2** (P1) aborted Hub download leaves a partial dataset that passes
  every completeness check. **N3** (P1) local delete doesn't invalidate the Hub-status cache (siblings do).
  **N4** (P1) **no UI at all to delete a local-only dataset** on `main` (multi-GB recordings accumulate).
  **N5** (P1) "remove local copy" of a `both` dataset can discard episodes the Hub copy lacks. **N6–N19** (P2)
  merge-source not guarded; merge/record/login don't invalidate listing caches; process-lifetime Hub-status
  cache pins visibility and feeds the public-push gate (N10); upload path may download first; TOCTOU on the
  in-use guard; per-author Hub listing truncated at 200; `useDatasets` wipes the list on any fetch error;
  stale "private-by-default" docstrings; pinned custom Hub datasets always render public.
- 2026-07-24: **N20** (P0) → **P0-5** — bare name + traversal-only delete guard = `rmtree` of a MakerMods Lab state
  dir. **N21** (P1) import/logged-out record don't prefill the Hub namespace (the P0-5 precondition).
  **N22** (P2) synchronous `copytree` inside the import request. **N23** (P2) `DatasetCard` truncates the name
  users disambiguate by, then repeats it twice. **N24** (P2) the Launchpad library sheet has no search, filter
  or `CappedGrid` while the studio list has all three.

### Model Training → [Bug List](Model%20Training%20Bug%20List.md)
- **MT1** ruled-out (unchanged). **MT2** (P1) fine-tune discards the selected Hub step. **MT3** (P1) final
  root policy hidden when periodic checkpoints exist. **MT4** (P1) Cloud→Local resume launches an invalid
  resume (`--resume true`, no `--config_path`). **MT5** (P1→P2, **PARTIALLY FIXED** — policy now locked;
  target/batch/optimizer/W&B still editable). **MT6** (P1) cloud W&B resume omits the secret. **MT7** (P1) job
  delete bypasses the active-inference guard. **MT8** (P1→P2) config-only/partial models classified usable.
  **MT9** (P1) cloud upload can publish a partial checkpoint. **MT10** (P1) reattached training failure
  recorded as success (and promoted into the models browser). **MT11** (P1, money) cloud stop reports
  canceled before/regardless of remote cancel — worse than recorded, see NEW-1.
- **MT12** (P1→P2, + NEW-6) cloud resume lineage can't distinguish parent/child. **MT13–MT23** (P2) HTTP-500
  on bad bodies; zero/negative numeric params; terminal transition omits final logs/checkpoints; **MT16 &
  MT17 reopened** (their fixes are `andrew`-only, absent from `main`); manual upload partial-success
  mis-reported; host-only-dep gate too broad; 409s all shown as local mutex; legacy migration moves every
  dir; ZIP download unbounded/races; registry lock spans network + submission. **MT24** (P1) resume re-passes
  `--policy.tags` that draccus rejects — **fix never landed on `main`** (note: the *field* rename to
  `env_eval_freq` did land, but MT24 is about `--policy.tags`, still emitted at `train.py:268-270`).
- New: **Training NEW-1** (P1) no way to cancel a remote HF job once the local record is finalised (paid GPU
  burns). **NEW-2** (P1) fine-tune from the jobs library launches with the WRONG policy type and the UI
  **locks** it. **NEW-3** (P1→P2, narrow trigger) implicit cloud-run dataset upload publishes **public** —
  now **policy-conformant** (public is intended), so what remains is that it decides visibility at its own
  site instead of reading the central constant; reconcile with Dataset entry 1 via PR #19.
  **NEW-4..NEW-10** (P2) cloud terminal transition
  truncates the persisted log; user stop recorded as `failed rc=1`; duplicate lineage checkpoints; seeded
  resume silently converted to a fresh double-length run; `upload_local_model` wipes existing repo tags;
  unlocked `seen` set; resume step-guard skipped when resuming "latest".
- 2026-07-24: **NEW-11** (P2) `tests/test_models.py` reads the developer's real pin/hide files
  (`tmp_lerobot_home` patches every other state path but not the four `SAVED_*` constants), so 7 tests fail
  or pass depending on the machine.

### Inference → [Bug List](Inference%20Bug%20List.md)
- **I1** (STILL REAL, one sub-case fixed) stopped startup worker keeps touching hardware; a *new*
  untracked-child path replaces the fixed one → **Inference N2**. **I2** (P1) stop reports success and loses
  the handle when termination fails. **I3** (P1) failed explicit Stop permanently disarms the leave guard.
  **I4** (P1, broader than audited — auto-cal has the same hole) non-atomic exclusion, calibration omitted
  (overlaps closed-unmerged **PR #11**). **I5** (P1) FastAPI shutdown doesn't stop the child (overlaps
  closed-unmerged **PR #10**). **I6–I12** local model launched through the Hub (2 surfaces); local model
  deletable during startup; I8 **FIXED** (backend `_last_result` idempotence) with cosmetic residuals;
  bimanual `right_*` cameras mapped to wrong keys (2 surfaces); mid-run overload mislabeled;
  active-startup shows previous log; Hub-checkpoint cache conflates imported vs cloud.
- **G1** (STILL REAL, 2 surfaces) required language task not enforced. **G2** (STILL REAL, **severity raised**)
  duration has no positive lower bound — `NumberInput` doesn't clamp, so typing `0` starts an **unbounded
  autonomous run from the normal UI**. **G3** (2 surfaces) multiple roles may bind one camera. **G4** (2
  surfaces) server arm-count guard loses the UI's action-dim fallback.
- New: **Inference N1** (P0) + **N2** (P0/P1) → **P0-3**. **N3** (P1) `DELETE /jobs/{id}` bypasses the in-use
  guard and rmtrees a live checkpoint. **N4** (P1) during a status-endpoint outage the dialog shows no Stop
  control and the exit guard is disarmed. **N5–N10** (P2/P3) subprocess reaped only by a poll (slot latches
  active forever with nobody polling); bimanual right-arm fields unvalidated (opens/primes the LEFT arm
  first); camera-key draccus interpolation unescaped; frozen cv2-index bindings silently rebind on replug;
  no-timeout final `wait()`; two Stop surfaces double-signal → clean stop reported as failure.

---

## Cross-cutting (fix once, close many) — carried from RANKING, still valid on `main`

- **Visibility policy — settled as PUBLIC (2026-07-26); what remains is centralization + copy.** The record
  auto-push is intended behavior and no longer part of this cluster (see "Descoped by product decision").
  Still live: Dataset entry 1 (the cloud-training notice says *private* and must be corrected to match), the
  cloud fallback `hf_cloud.py:671` (Training NEW-3), and the cache-driven route Dataset N10 — three sites
  deciding visibility independently, with **no single centralized publisher**. Open PR #19 introduces that
  publisher (`DATASET_DEFAULT_PRIVATE = False`); merging it closes the cluster.
- **Hardware-session arbiter:** Teleop T4/N3/N4, Inference I4/N6, Config entry 4 (C4)/NEW-2, Recording R9 —
  plus the `c737a30` calibration↔session mutex that `logs/PROJECT.md` documents as shipped but is **not on
  this branch**. One server-owned session token + single lock closes the family. (Closed-unmerged PR #7 / #11
  targeted parts of it.)
- **Stop-path integrity:** Teleop T1/T3/T7, Recording R7, Inference I2/I3, Training MT11 — every feature has a
  "report stopped before verified stopped" variant. Shared contract: verify, then report; never latch a guard
  before the ack.
- **Runner finalization contract:** Training MT10 + MT11 + NEW-1 + NEW-5 — a runner that cannot know (or was
  told to stop) reporting a definitive `returncode()` the watchdog converts into an irreversible terminal
  state.

---

## Numbering collisions (kept both; not resolved here)

- **Configuration Setup: two schemes for the same bugs.** The Config REVERIFIED doc numbers its existing
  entries **1–13**; `RANKING.md` numbers the same bugs **C1–C10 + D1–D3**. They map in order — entries **1–4
  = C1–C4**, entries **5–13 = C5–C10 + D1–D3** — but the two docs never state the mapping, so both identifiers
  are carried above. Config's *new* findings use the **NEW-n** prefix (NEW-1..NEW-7).
- **The `N`/`NEW` new-finding prefix is per-module and collides across modules.** Teleop **N1–N11**, Recording
  **N1–N9**, Dataset **N1–N19**, Inference **N1–N10**, and Config/Training **NEW-1..NEW-n** are *independent*
  sequences. A bare "N1" is ambiguous — always read the module prefix ("Recording N1" ≠ "Inference N1" ≠
  "Dataset N1"). This doc always prefixes them.
- **`D`-prefix is reused.** `RANKING`/Config use **D1–D3** for Configuration P2 gaps; Recording REVERIFIED
  uses **D1–D5** for its *design gaps*. Different modules, same letters — module prefix disambiguates.
- **One bug, two module IDs, on purpose:** the retired **P0-2** was **Recording N2** *and* **Dataset N1** (two
  agents found the same chain independently). Both IDs are retained as **descoped** — retired, not reused, and
  not to be re-filed.

---

## Module index — the six primary sources (merged current-state bug lists)

Each per-module **Bug List** now folds its 23-07-26 re-audit in place (original entry + `main`-current verdict +
new findings), so it is the single self-contained current source. The standalone `REVERIFIED` docs and the
pre-reverification originals are retained under [`archive/`](archive/).

| Module | Merged Bug List (primary source) | Archived re-audit |
|---|---|---|
| Configuration Setup | [Configuration Setup Bug List.md](Configuration%20Setup%20Bug%20List.md) | [archive/Configuration Setup REVERIFIED 23-07-26.md](archive/Configuration%20Setup%20REVERIFIED%2023-07-26.md) |
| Teleoperation | [Teleoperation Bug List.md](Teleoperation%20Bug%20List.md) | [archive/Teleoperation REVERIFIED 23-07-26.md](archive/Teleoperation%20REVERIFIED%2023-07-26.md) |
| Recording | [Recording Bug List.md](Recording%20Bug%20List.md) | [archive/Recording REVERIFIED 23-07-26.md](archive/Recording%20REVERIFIED%2023-07-26.md) |
| Dataset | [Dataset Bug List.md](Dataset%20Bug%20List.md) | [archive/Dataset REVERIFIED 23-07-26.md](archive/Dataset%20REVERIFIED%2023-07-26.md) |
| Model Training | [Model Training Bug List.md](Model%20Training%20Bug%20List.md) | [archive/Model Training REVERIFIED 23-07-26.md](archive/Model%20Training%20REVERIFIED%2023-07-26.md) |
| Inference | [Inference Bug List.md](Inference%20Bug%20List.md) | [archive/Inference REVERIFIED 23-07-26.md](archive/Inference%20REVERIFIED%2023-07-26.md) |

Related (historical records only — their unique content is folded into this doc):
[archive/CONFIRMED 22-07-26.md](archive/CONFIRMED%2022-07-26.md) (2026-07-22 bench run) ·
[archive/RANKING.md](archive/RANKING.md) (pre-redesign backlog).
