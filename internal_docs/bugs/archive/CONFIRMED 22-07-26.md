> ## 📦 HISTORICAL RECORD — content folded into [`../CURRENT 23-07-26.md`](../CURRENT%2023-07-26.md) (2026-07-26)
>
> **Do not triage from this file.** Everything below that is still actionable now lives in
> [`../CURRENT 23-07-26.md`](../CURRENT%2023-07-26.md) — the bench findings as the **"Bench-run findings"**
> section (`Bench #1`, `#2`, `#4`, `#6`–`#14`, `#16`, IDs preserved), the non-bug asks as that section's
> **"Feature requests / design asks"** list, and the "Open decisions" upload-visibility item as
> **"Descoped by product decision"** (decided: public). CURRENT is the single place bugs are organized.
>
> This file is **retained as the historical record of its method and date**: the log of the **2026-07-22 live
> bench run** — a human driving the merged tree, which is why it caught dropped UI entry points that a
> code-forensic audit structurally cannot see. Read it for *how and when* these were found, not for current state.
>
> **Three entries below did NOT move, deliberately — do NOT re-act on them or re-merge them:**
> - **#3 "Torque / motor-power slider gone" — does NOT reproduce.** The slider is present and wired on `main`
>   (`RobotConfigDialog.tsx:1215-1276`). See CURRENT → "Corrections".
> - **#15 "Power telemetry not implemented" — imprecise.** The **backend works and broadcasts**
>   `follower_currents_ma`; the **frontend consumer count is zero**. It is a UI wiring gap, not a missing
>   feature. See CURRENT → "Corrections".
> - **#5 "Cannot resume runs"** — already carried inside CURRENT's **P0-4**.
>
> One clarification carried into CURRENT: **#2 "Teleop camera visualization gone" is real**, and the root
> cause is now known — the `/teleoperation` route is **orphaned** (the route exists but nothing navigates to
> it), so its camera panel is never mounted.
>
> *Note: this file was moved into `archive/` after it was written; relative links in the body below were
> authored against the old top-level path and may not resolve from here. They are left as authored.*

# Confirmed Bugs — 22-07-26 bench run (post-redesign / `integrate`)

**This is the current authoritative confirmed-bug list.** Sourced from the live [22-07-26 Full Test](../tests/22-07-26%20Full%20Test/Pathway%20Test%20Checklist.md) pathway run on the merged `integrate/andrew-into-redesign` tree — i.e. **post-redesign**, unlike [RANKING.md](RANKING.md) + the six module lists, which were forensic-verified on **andrew (pre-redesign, 2026-07-14)** and are now the *backlog to re-verify*.

**Headline finding:** most confirmed fails are **redesign regressions** — a capability whose backend survived the merge but whose **UI entry point was dropped** when we took redesign's frontend wholesale. This is precisely the parity risk flagged in the [Andrew Merge Ledger](../Andrew%20Merge%20Ledger.md); these are the boxes that didn't get ticked.

Rubric (same as RANKING): **P0** = hardware safety, data loss/corruption, silent private-data publication, or main path completely broken. **P1** = wrong behavior in a common flow, real consequence, no workaround. **P2** = edge/cosmetic/polish.

---

## P0 — safety / main path broken

1. **Arm-identity guard is completely non-functional — zero protection against a bad calibration↔arm pair.**
   Run: Phase 1 "Hardware safety" all `[-]` / *"Complete fail"*; user note Big #0. The guard (`makermodslab/arm_identity.py`) is meant to fire before torque-enable and block swapped leader/follower or wrong-calibration arms. It isn't guarding at all → **an energized arm can run with the wrong calibration** (violent motion / hardware damage / injury). Highest priority; nothing else matters if arms can be energized wrong.
   → Related pre-redesign entries: Teleop T4 (hardware-ownership mutex "never shipped"), the cross-cutting `c737a30` mutex. Also revisit the guard's severity design (block-with-override vs warn) — see the stop/identity design discussion.

## P1 — redesign regressions (backend survived, UI wiring lost)

2. **Teleop camera visualization gone.** Run: Phase 2 ⭐ line `[-]`; Big #3. Andrew had live camera previews during teleop; the redesign teleop UI doesn't show them. Ledger parity item (camera previews) — verify the backend feed still exists and re-wire.
3. **Torque / motor-power slider gone.** Run: Phase 1 line `[-]`; Big #5. The per-robot motor-power control is absent from the redesign robot config — even though the merge **kept the backend** (`motor_power` per-robot, `clamp_motor_power`, `torque_limit_from_percent`). Backend present, UI door missing.
4. **Calibration "download config" gone.** Run: Phase 1 line `[-]` ("No download"); Big #4. Calibration JSON export/download control dropped in the redesign calibration UI.
5. **Cannot resume runs (recording AND training).** Run: Phase 3 ⭐ Resume `[-]`, Phase 5 Continue-from-checkpoint `[-]`; Smaller #7. Both resume entry points are broken/absent. (Backend resume logic exists — the merge kept resume-safe discard etc.)
6. **Dataset management ops don't work: Create / add-Hub / download / import-from-disk / rename / delete.** Run: Phase 4 line `[-]`; Smaller #6. The whole dataset-library action set is non-functional from the redesign UI.
7. **Model-library ops don't work: add-Hub / download / import-checkpoint / upload / delete.** Run: Phase 5 "Model library" line `[-]`. Same class as #6 for models.

## P1 — version / launcher correctness

8. **0.6.0-trained models can't deploy on a 0.5.2 runtime.** Run: Phase 6 ⭐ Deploy `[-]`; Big #2. `draccus` hard-rejects the `pretrained_revision` field a 0.6.0 policy config carries. Expected forward-incompatibility across the branch/version boundary — **deploy those models on the 0.6.0 (`integrate`) branch, not on `main`/0.5.2.** Document as a compat constraint; not a merged-code bug. (Confirmed this session.)
9. **`makermodslab --stop` can't stop a normally-started instance.** Serve line `[✗]`. Prod-mode runs uvicorn in-process, so the launcher's cmdline lacks `makermodslab.server` and the identity match misses it — only `--dev` orphans / manual uvicorn are caught. Fix: also recognize the `makermodslab`/`makermodslab-station` entry-point process listening on our port. (Confirmed this session.)

## P2 — polish / display / flakiness

10. **Intermittent "failed to write lock position" (motor comm).** Big #1. Usually resolves on a second try — motor-bus write flakiness; investigate whether the bounded-retry path can absorb it.
11. **Upload-to-Hub card: wrong placement + wrong episode count.** Run: Phase 3 `[/]`; Smaller #2. Should live on the "+ New Skill" page and displays the wrong # of episodes.
12. **Model download progress doesn't display correctly.** Run: Phase 6 `[/]`; Smaller #8.
13. **Dataset info card is hard to read / needs more info.** Run: Phase 4 `[/]`; Smaller #4.
14. **No dropdown to select a dataset.** Smaller #5.
15. **Power telemetry (peak/avg) not implemented.** Run: Phase 2 line `[-]` "Not implemented". (Feature-not-done rather than a regression.)
16. **End-of-recording: follower returns to start pose too quickly.** Smaller #1. See the stop-behavior design discussion (gentle-return speed / whether recording should return at all).

---

## Open decisions (not bugs — need a call)

- ✅ **DECIDED 2026-07-26 — public.** *(Was: "Cloud-training upload default: private vs public." The merge set it **private**; the run annotated *"[SHOULD BE PUBLIC]"* (Phase 5); the two conflicted.)* **Public is the intended default at every push site**, and the record flow's default-on auto-push is intended behavior — which retires what CURRENT filed as **P0-2** (Recording N2 / Dataset N1). What remains is to make notice + push sites agree *on public*: update `LocalDatasetCloudNotice` (it still promises private) and route every site through one policy constant — open [PR #19](https://github.com/makermods-robotics/makermodslab/pull/19) does exactly that. See CURRENT → "Descoped by product decision".

## Feature requests / design (from the run — tracked, not bugs)

- Pause/exit for Detect & Wiggle (currently no way out but waiting it out). (Big Feature #1)
- Rethink saving robot configs — explicit save vs autosave. (Big Feature #2, Convenience #2)
- "+ New Calibration" doesn't fit multi-arm — either per-arm subprocess (concurrent) or remove. (Convenience #1)
- When should the user be allowed to name a calibration file? (Convenience #3)
- Batch delete. (Convenience #4)
- More detailed logging (e.g. "waiting for HF to allocate a GPU"). (Smaller #3)
- Rebuild frontend on every PR accept? (Extra Notes #0)

---

## Reconciliation with the pre-redesign lists

The six module lists + RANKING remain valid as **forensic analysis of andrew's backend** (many entries are deep, file:line-traced logic bugs unlikely to have moved). But:
- The **P0s in RANKING are addressed** by the merge (private-default upload + resume-safe discard) — modulo the open public/private decision above.
- The **redesign regressions here (#2–#7) are NEW** — the old lists analyzed code, not dropped UI, so they couldn't surface them. Cross-check against the [Andrew Merge Ledger](../Andrew%20Merge%20Ledger.md) parity table.
- **Action:** treat this file as current truth; re-verify the pre-redesign P1/P2 backlog against the redesign UI before spending on it.
