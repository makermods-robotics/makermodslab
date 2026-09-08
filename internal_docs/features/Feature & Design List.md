# Feature & Design List

Sibling to [`../bugs/`](../bugs/): the bug lists track **defects** (something behaves wrongly today); this list
tracks **structural and feature work** — changes where nothing is "broken" per se, but the design is generating
bugs, friction, or risk faster than point fixes retire them. Entries follow the bug-list conventions where they
apply: `### F<n> · <effort S/M/L>` headers, a **First identified** date, and links to the bugs/PRs that
motivate them. Unlike bugs, entries here carry a **Window** (when it's sane to do the work) — most of these are
deliberately parked until after the first release and the `rig` merge, and doing them early would be worse than
waiting.

---

### F1 · L — Hardware session lifecycle (ownership, cancellation, fail-safe)

**First identified:** 2026-08-02 (synthesized from PRs #23/#25/#27/#28/#29 and the 2026-07-28 shutdown audit)

**Window:** post-first-release, post-`rig`-merge. One coherent design discussion first; implementation likely
lands as 3–4 separate PRs.

The I1–I8 fix lineage patched the same underlying flaw four times, once per feature, and its own follow-up
notes say so. Root causes, named — each maps to a workstream below:

1. **No lifecycle-ownership primitive.** "Who holds the bus and how do I stop them safely" exists only as an
   N×N mesh of hand-written reciprocal flag checks (~8 of the ~20 needed edges exist; auto-calibration has
   **zero** — bug list D1) plus a second hand-maintained enumeration of the same knowledge in
   `shutdown_event()`'s five-way `asyncio.gather`. Both go stale silently when a feature is added.
2. **No cooperative cancellation in blocking hardware paths.** `_prepare_robot` has no checkpoint at all
   (why I6 could only make the orphaned-worker window *observable*, not eliminate it); the control loops are
   interruptible only in whole-iteration steps; the vendored autocal script only at stage boundaries. Every
   bounded-wait / escalation ladder in the I-series exists to work around this.
3. **Fail-unsafe by construction.** The arm is safe only if cleanup code successfully runs — torque stays
   enabled absent an explicit register write. The entire "make sure cleanup runs" PR genre (I5/I7/I8) hardens
   that assumption but can never cover SIGKILL or a server crash. A fail-safe design inverts the default.
4. **Destructive operations without undo.** Autocal Stage 0 wipes servo EEPROM before measuring, with no
   snapshot and no restore on stop (P0-1). Cost a real calibration loss on 2026-07-28 (both arms; interrupted
   test runs wiped, accidental completions were the only recovery).

**Workstreams:**

- **(a) Resource-ownership registry.** A single `claim(resources, owner, stop_hook)` / `release()` primitive
  every hardware-touching path goes through. Fail-fast refusal with the holder's name — *never* a lock held
  across a hardware operation, so the deadlock concern that motivated the current deliberately-not-a-shared-lock
  design does not apply (this distinction is the crux; the existing code comments defend against the wrong
  alternative). Kills the mesh (exclusion becomes structural, missing edges impossible), kills the shutdown
  enumeration (`shutdown_event` iterates live claims and calls stop hooks), represents the I6 shutting-down
  window natively (claim held until the worker actually exits), and enables per-port granularity the global
  booleans can't express (e.g. calibrate the leader while the follower idles). Resource key design question:
  serial ports only, or cameras too — the Chrome-camera-hold history says cameras contend just as hard.
- **(b) Cancellation checkpoints at the source.** Add cancel-event checks inside `_prepare_robot` (between bus
  open / preflight reads / camera checks) and at finer grain in the vendored script's stages. Shrinks the
  uninterruptible windows from seconds-to-indefinite down to fractions of a second; retires most of the
  bounded-wait machinery to a rarely-hit backstop. Note the hand-mirrored-twin constraint on the vendored
  script (`rest_pose.py` docs) and C20's constant-drift hazard — deriving the escalation budgets from imported
  constants should ride along.
- **(c) Fail-safe direction decision.** Threads die with the server (arm frozen energized, no release —
  SIGKILL/crash is uncoverable); subprocesses survive it (arm actively driven by an orphan — the I5 bug) *unless*
  given a deadman watchdog (child watches stdin-pipe EOF, runs its own return-to-rest + torque release —
  works on macOS, covers SIGKILL and server crashes, the only design that does). Decide whether teleop/record
  stay threads (status quo: acceptable while an operator stands at the arm with a power switch) or become
  watchdogged subprocesses (the escalation path if arms ever run less supervised). This is a cost/benefit
  decision, not a bug fix — the IPC rewrite of `record_with_web_events` + joint streaming is the price.
- **(d) Snapshot/restore for destructive EEPROM operations.** Fix P0-1 at the root: snapshot all six motors'
  `Homing_Offset` + limits before Stage 0 touches them; restore on any stop/interrupt path. Converts the
  entire accident class from "calibration destroyed" to "no-op." Smallest workstream; could land before the
  others as an ordinary bug-fix PR if prioritized.

**Evidence file:** PR #23 (I1), #25 (I5), #27 (I6), #28 (I7), #29 (I8) and their follow-up notes; bug list
entries D1, C20, P0-1; the 2026-07-28 calibration-loss incident. The `stop_and_wait` implementations and
escalation ladders from the I-series are **kept**, not discarded — they become the registry's stop hooks.

---

### F2 · M — Split `server.py` into per-feature routers

**First identified:** 2026-08-02 (long-standing; sized during the refactor survey)

**Window:** post-`rig`-merge — refactoring 106 endpoint definitions under an 8k-line in-flight branch invites
conflict hell.

`server.py` is ~2,650 lines: 106 endpoints, 18 inline request models, one file. Mechanical split into FastAPI
`APIRouter`s per feature module, request models moved next to their handlers. Highest-value structural cleanup
per unit of risk; mostly mechanical; delivers most of what a framework migration would promise at ~5% of the
cost. Pairs naturally with F1(a): the routers and the registry claim sites end up feature-aligned.

### F3 · S — Frontend consumer for the `shutting_down` status flag

**First identified:** 2026-08-02 (from PR #27 review)

**Window:** after #27 merges.

PR #27 ships `shutting_down` in `/inference-status` with no UI consumer. Add one: while true, the Stop button
becomes a disabled "Stopping…" state (visible state change, **not** a bare greyed-out button — in the
wedged-bus case the flag stays true indefinitely and an unexplained disabled button next to an idle-looking
page is its own confusion). Closes the "status says idle / start says busy / stop says nothing to stop"
operator trap end-to-end. The second-stop endpoint behavior stays as the safety net for out-of-band callers.

### F4 · S — Dev-mode (`--dev`) hardening

**First identified:** 2026-08-02 (from the PR #25 review discussion)

**Window:** any time; independent of release.

Three cheap changes, no custom reloader (keeping `--dev` a thin stock-uvicorn + stock-Vite wrapper is the
maintenance strategy):
1. Scope the reload watcher (`--reload-dir makermodslab`, exclude `makermodslab/vendor`) — today any `*.py` save
   anywhere in the repo (tests, `debug/`, vendored sources) restarts the backend and kills whatever robot
   session is live, despite none of that code being loaded by the server.
2. Loud logging when a reload's shutdown hook actually stops an active session ("reload terminated active
   inference session"), so the developer whose arm just stopped sees why.
3. A visible "dev" badge in the UI when served from Vite — retires the :8080-vs-:8000 stale-tab confusion
   class (bit twice already).

### F5 · M — Frontend oversized-component decomposition

**First identified:** 2026-08-02 (refactor survey)

**Window:** post-`rig`-merge; opportunistic, per-component.

`RobotConfigDialog.tsx` is 2,470 lines (nearly server.py-sized); then a band of 500–1,100-line components:
`DatasetInfoCard` (1,097), `RecordingSessionDialog` (919), `DeployPanel` (883), `JobCard` (755),
`InferenceModal` (676). Classic dialog-that-grew shape; each splits into subcomponents plus a hook. No urgency,
no user-visible change — do them opportunistically when a feature change already forces a visit, starting with
`RobotConfigDialog`. Also check whether `mockHub.ts` (520 lines) is still load-bearing.

### F6 · M — Hub model downloads: two disjoint stores, full-repo fetches, per-revision re-pulls

**First identified:** 2026-08-03 (user noticed repeat downloads on inference; disk audit confirmed)

**Window:** post-`rig`-merge; the cleanup half (delete duplicates/stale revisions) can happen any time.

The same model weights land on disk 2–3× because three mechanisms don't share:

1. **Two disjoint stores.** Inference (`rollout._resolve_policy_path`) snapshots hub refs into the shared HF
   hub cache (`~/.cache/huggingface/hub/`). The Models-page download (`models._fetch_model_snapshot`) uses
   `snapshot_download(local_dir=<makerlab_models>/<repo_id>)` — and `local_dir` mode in huggingface_hub ≥0.23
   bypasses the shared cache in both directions (no reuse, no population). Observed: `lerobot/smolvla_base`
   fully present in both stores, 872 MB each. Running inference on a hub ref never consults the local copy,
   and vice versa. (`datasets.py` has the same `local_dir` pattern for dataset downloads.)
2. **Models-page download pulls the whole repo.** `_fetch_model_snapshot` has no `ignore_patterns`, so it
   pulls every `checkpoints/<step>/` including `training_state/` (394 MB of optimizer state per step).
   Observed: one local model dir at 5.6 GB of which ~4.9 GB is checkpoints/training_state inference never
   reads. `rollout.py`'s `@root` path already excludes exactly these (`checkpoints/**`, `training_state/**`)
   for exactly this reason — the two call sites disagree.
3. **Per-revision re-pulls.** Each inference on a hub ref resolves `main`; every push to the same repo (e.g.
   re-training to the same name) re-downloads all changed files (~700 MB of weights each time). Observed: one
   repo with 3 snapshots / 2.2 GB blobs. Legitimate hub-cache semantics, but it compounds the "downloading
   again" feeling, and old revisions are never GC'd.

Fixes, roughly independent: (a) add `ignore_patterns=["*/training_state/**", "training_state/**"]` (and
possibly non-selected checkpoint steps) to `_fetch_model_snapshot`; (b) make `_resolve_policy_path` check the
local models store before hitting the hub (or drop `local_dir` from the Models-page download so both paths
share the hub cache and the local "copy" becomes a resolve); (c) surface/automate `hf cache` GC of stale
revisions. One-off cleanup available today: delete the duplicated `smolvla_base` local copy, the stale
2 revisions in the eraser-stack repo, and the training_state trees (~3.5 GB total).

### F7 · M — Cross-runner resume (continue a run on a different compute target)

**First identified:** 2026-08-04 (design review of the resume-form lock; MT4/MT42 showed both directions broken)

**Window:** after the resume-UX PR (cluster 2) lands; the interim lock ships with that PR.

Today a resume can only correctly continue on the parent's runner, but the form lets the user flip Compute —
and both cross-runner directions are broken: cloud-parent → Local launches an invalid resume that dies at
startup (MT4: the parent's checkpoints live on the Hub, and the local resume path points config_path at a
host directory that doesn't exist), and local-parent → Cloud silently restarts from step 0 while recording
itself as a resume (MT42, P1: the local checkpoint never reaches the pod, so the wrapper has nothing to
download). **Interim decision (2026-08-04, user):** lock the compute target on resume — inherited from the
parent, rendered disabled with a "continues on the parent's runner" note, plus a server-side 400 on
cross-runner resume requests as defense in depth. That closes MT4/MT42's front door without building the
feature.

The actual feature, when wanted:

1. **cloud-parent → Local**: resolve resume_from_hub_repo/step by downloading the checkpoint dir
   (incl. training_state/ — optimizer state is required for a true resume, unlike fine-tune) to the host,
   reconstructing lerobot's output-dir layout, and pointing config_path at it. Cleanup policy (decided
   2026-08-04): the weights half stays in the shared HF cache (reused by later deploys of the same
   checkpoint; see F6 — no per-feature deletion), but the training_state/ half is single-purpose and is a
   GC candidate once the resumed run reaches a terminal state (~394 MB/step, nothing else ever reads it).
   Note cloud→cloud resume needs no such policy at all — its download is pod-side and dies with the pod. The in-container wrapper
   already implements exactly this reconstruction pod-side — port its logic host-side.
2. **local-parent → Cloud**: requires the parent checkpoint on the Hub first. Given the standing
   public-upload concern, this must be an explicit, consented upload step (and ideally private-by-default),
   not an automatic side effect of clicking Continue.

Both directions share the step-ref → materialized-directory machinery the MT2 option-C fix introduces;
build on it rather than duplicating.
