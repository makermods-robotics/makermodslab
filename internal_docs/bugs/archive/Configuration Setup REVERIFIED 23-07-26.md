# Configuration Setup — Re-verification + new findings

Re-verification date: July 23, 2026
Re-verified against: branch `main`, commit `c7d9f27` ("Merge pull request #4 from makermods-robotics/redesign"), working tree clean apart from the untracked `internal_docs/`, `debug/`, `.agents/` and the `frontend/dist` bundle swap.

Source list: `internal_docs/bugs/Configuration Setup Bug List.md`, audited 2026-07-14 at `518ca56` on `min_stable` and later annotated with fixes made on the `andrew` branch.

**Two structural facts govern this re-verification:**

1. **The `andrew`-branch fixes are NOT on `main`.** The source list's two "Fix implemented 2026-07-14" status updates (auto-calibration staging dir, manual-calibration snapshot-and-rollback) describe code that does not exist on `main`. Both defects are re-confirmed below against `main`'s current source.
2. **The redesign moved the whole configuration surface.** `frontend/src/pages/Calibration.tsx` and `frontend/src/pages/Landing.tsx` no longer exist. Configuration now lives in `frontend/src/components/dialogs/RobotConfigDialog.tsx` (a modal window), `frontend/src/components/calibration/CalibrationLibrary.tsx`, and `frontend/src/components/studio/CollectPanel.tsx`. Every frontend citation in the source list is stale; the verdicts below re-cite `main`.

**Open-PR overlap.** PRs #5–#11 on `makermods-robotics/makermodslab` are all **CLOSED WITHOUT MERGE** (`mergedAt: null`; `git merge-base --is-ancestor origin/fix/... origin/main` → NO for both branches checked). Two of them target entries in this list — **#7 (T4, cross-feature mutex incl. calibration/auto-calibration/wiggle)** and **#11 (I4, atomic check-and-claim in `start_calibration` + exit guard)**. Neither is on `main`, so `main` still contains both bugs.

**Excluded by brief:** the arm-identity guard severity (`makermodslab/arm_identity.py` warn-and-proceed). Not reported; referenced only where it is the *detector* for another defect.

---

## Part 1 — Verdicts on the existing entries

| # | Entry (source list) | Verdict on `main` | Key evidence |
|---|---|---|---|
| 1 | P1 — Failed/stopped follower auto-recalibration deletes the previous valid calibration | **STILL REAL** (andrew fix absent) | `makermodslab/auto_calibrate.py:112-138`, call sites `:292`, `:307`, `:440`; `makermodslab/utils/config.py:32`; `frontend/.../RobotConfigDialog.tsx:951-956` |
| 2 | P1 — Canceling manual calibration does not restore servo calibration state | **STILL REAL** (andrew fix absent) | `makermodslab/calibrate.py:417-429`, `:367-386`, `:655-659` |
| 3 | P1 — Manual calibration permits saving grossly incomplete ranges | **STILL REAL** (restructured frontend, same behavior) | `makermodslab/calibrate.py:554-575`; `frontend/.../RobotConfigDialog.tsx:1701-1744`, disable at `:1718` |
| 4 | P1 — Manual calibration lifecycle can overlap worker threads on one singleton | **STILL REAL** — overlaps **PR #11 (I4), closed unmerged** | `makermodslab/calibrate.py:235`, `:252-276`, `:306-331` |
| 5 | P2 — Concurrent robot-record updates lose fields / collide on the temp file | **STILL REAL** | `makermodslab/utils/config.py:387-427`, `:109-117`; writers at `calibrate.py:651`, `auto_calibrate.py:342`, `config.py:1102`, `:1133` |
| 6 | P2 — Auto-calibration reports success and assigns a missing calibration file | **STILL REAL** | `makermodslab/auto_calibrate.py:274-311`, `:313-348`; codified by `tests/test_auto_calibrate.py:123-152` |
| 7 | P2 — Imported calibration validation accepts unusable SO-101 profiles | **STILL REAL** | `makermodslab/utils/config.py:1008-1027`, `:474-516`; `tests/test_utils_config.py:259-274`, `:493-501` |
| 8 | P2 — Browser navigation silently discards unsaved configuration drafts | **STILL REAL, NARROWED** | `frontend/.../RobotConfigDialog.tsx:1270`, `:1338-1348`, guard armed only at `:595-606`; `frontend/src/hooks/useSessionExitGuard.ts:94-128` |
| 9 | P2 — Duplicate camera names make a configured camera disappear from recording | **STILL REAL** (moved file) | `frontend/.../CameraConfiguration.tsx:141-155`, `:185-191`; `frontend/.../CollectPanel.tsx:198-210` |
| 10 | P2 (was P3) — Offline station mode labels cached auth as "HF not configured" | **STILL REAL** | `makermodslab/scripts/makermodslab.py:499-506`, `:481-489`; `makermodslab/utils/hf_auth.py:116-124`; `HfAuthContext.tsx:12-22`, `:37-64`; `HfAuthDialog.tsx:51` |
| 11 | Gap — No global hardware-session exclusion across calibration managers | **STILL REAL, WORSE THAN STATED** — overlaps **PR #7 (T4), closed unmerged** | `makermodslab/server.py:1804-1871`; `teleoperate.py:567-581`; `record.py:461-472`; `rollout.py:918-930` |
| 12 | Gap — Calibration-library mutation not guarded during an active calibration | **STILL REAL** (same mitigations) | `makermodslab/server.py:1911-1964`, `:2016-2048`, `:2051-2073`; mitigations `config.py:1062-1105`, `:1108-1135` |
| 13 | Gap — macOS camera enumeration has no fallback after AVFoundation-helper failure | **STILL REAL (static)** / **CANNOT VERIFY STATICALLY (manifestation)** | `makermodslab/server.py:2332-2336`, `:2198-2213` |

### Detail per entry

#### 1. Failed/stopped follower auto-recalibration deletes the previous valid calibration — **STILL REAL**

The staging-dir fix described in the source list's status update is **not present on `main`**. `_subprocess_output_path("robot", stem)` still resolves to `CALIBRATE_BASE_PATH_ROBOTS/so_follower/<stem>.json` (`makermodslab/auto_calibrate.py:112-122`), which is byte-identical to `FOLLOWER_CONFIG_PATH` (`makermodslab/utils/config.py:32`) — the real library. `_remove_stray_calibration_file` (`:125-138`) `os.remove`s that path with no way to tell a newly written stray from a pre-existing profile, and is called on **all three** non-success terminal paths:

- post-processing failure — `makermodslab/auto_calibrate.py:291-293`
- nonzero exit / dropped connection — `:306-307`
- stop — `:438-440`

The frontend still reuses the arm's assigned config name and always sends `overwrite: true` (`frontend/src/components/dialogs/RobotConfigDialog.tsx:931-938` picks `robot[s.cfgField]` as the name; `:951-956` sends `overwrite: true`), so a routine "recalibrate this arm" cancel deletes the arm's last working profile.

**Refinement over the source list:** this is **follower-only**. For a leader (`device_type == "teleop"`) the subprocess writes to the scratch dir `robots/so_leader/`, and `_remove_stray_calibration_file` deletes only that scratch file — the real leader library lives under `teleoperators/so_leader` and is untouched (`:141-149`, `:323-329`).

The existing tests still codify the destructive contract: `tests/test_auto_calibrate.py:253-277` plants a follower file *before* the run and asserts it is deleted on nonzero exit; same at `:280-311` (post-processing) and `:313-328` (stop).

#### 2. Canceling manual calibration does not restore persistent servo calibration state — **STILL REAL**

No snapshot/rollback exists on `main`. `_step_homing` disables torque, calls `reset_calibration()` (`makermodslab/calibrate.py:421`) and writes a new `Homing_Offset` to every servo (`:428-429`) before range recording. Every cancel/error path (`:367-370`, `:375-378`, `:383-386`, `:394-402`) goes to `_cleanup_and_finish` (`:655-659`), which only disconnects and flips status — no register restore, no `warning` field.

#### 3. Manual calibration permits saving grossly incomplete ranges — **STILL REAL**

Backend unchanged: only exact `min == max` is rejected (`makermodslab/calibrate.py:554-560`); a range under 100 steps produces two `logger.warning` lines and is saved anyway (`:562-578`). Frontend moved but behaves identically: `allComplete` (computed from `isMotorRangeComplete`, `frontend/src/lib/calibrationTargets.ts:34-45`) only changes the "Save calibration" button's colour/icon; the button is disabled solely on `!calibrationStatus.calibration_active` (`RobotConfigDialog.tsx:1701-1744`, disable at `:1718`).

#### 4. Manual calibration lifecycle can overlap worker threads — **STILL REAL** (PR #11 / I4 not merged)

`start_calibration` reads `self.status.calibration_active` **outside** `_status_lock` (`makermodslab/calibrate.py:235`), then resets the shared singleton fields and starts a worker (`:252-276`). `stop_calibration_process` joins for 5 s, logs a warning if the thread is still alive, and then runs `_cleanup_and_finish(..., status="idle")` anyway (`:306-331`) — reporting idle while the stale worker may still hold the device. PR #11 (`fix/i4-calibration-atomic-exclusion-and-exit-guard`) makes the check-and-claim atomic; it is **closed unmerged**, so `main` retains the race.

#### 5. Concurrent robot-record updates — **STILL REAL**

`save_robot_record` is still an unlocked read-modify-write of the whole record (`makermodslab/utils/config.py:387-427`, read at `:402`, write at `:425`) and `_atomic_write_text` still uses the single fixed `<path>.tmp` (`:109-117`). Background writers that race the config dialog's Save: `calibrate.py:649-651`, `auto_calibrate.py:341-348`, plus the per-record loops in `rename_calibration_config` (`config.py:1099-1102`) and `clear_config_references` (`:1130-1133`).

#### 6. Auto-calibration reports success and assigns a missing calibration file — **STILL REAL**

A zero exit calls `_finalize_success()` and unconditionally sets `status="completed"` (`makermodslab/auto_calibrate.py:277-283`). `_finalize_success` guards the *leader copy* on `os.path.exists(src)` (`:326`) but never checks that a follower file was produced, and the robot-record write-back runs regardless (`:333-348`). `tests/test_auto_calibrate.py:123-152` is a zero-exit fake process that writes no file and still asserts `completed`.

#### 7. Imported calibration validation accepts unusable profiles — **STILL REAL**

`validate_calibration_data` still accepts any non-empty dict whose *present* motors each carry the five integer fields (`makermodslab/utils/config.py:1008-1027`) — no six-motor requirement, no ID uniqueness, no `range_min < range_max`, no register bounds. `is_robot_record_clean` still checks file *existence* only (`:474-516`). Tests still bless a one-motor profile (`tests/test_utils_config.py:259-274`) and `{}` calibration files as sufficient for readiness (`:493-501`).

#### 8. Browser navigation discards unsaved drafts — **STILL REAL, NARROWED**

The window now funnels **every in-window close vector** (Quit button, X, Esc, overlay click) through `requestClose` → discard prompt (`RobotConfigDialog.tsx:1338-1348`, `:2417-2437`), which is an improvement. What remains unguarded is browser-level and route-level leave: `useSessionExitGuard` is armed only on `manualCalibLive` (`:595-606`), and its `beforeunload`/`popstate`/unmount handlers are all gated on `active` (`useSessionExitGuard.ts:94-128`, `:134-152`, `:159-166`). A reload, tab close, or in-app route change away from the Launchpad while `isDirty` (`:1270`) silently drops port/camera/torque drafts.

#### 9. Duplicate camera names — **STILL REAL**

`addCamera` blocks duplicates by cv2 index **or** browser `deviceId` only (`CameraConfiguration.tsx:141-155`); `updateCamera` (`:185-191`) applies a renamed camera with no uniqueness check at all. `CollectPanel.tsx:198-210` reduces the list into a dict keyed by `cam.name`, so a later duplicate silently replaces the earlier entry, and `:249` sends that dict as the recording's `cameras`.

#### 10. Offline station mode mislabels cached auth — **STILL REAL**

`makermodslab-station` still injects `--offline` (`makermodslab/scripts/makermodslab.py:499-506`), which sets `HF_HUB_OFFLINE=1` before the server imports (`:481-489`). `handle_hf_auth_status` still collapses `LocalTokenNotFoundError`, `HfHubHTTPError` and `OSError` into one unauthenticated response with no offline field (`makermodslab/utils/hf_auth.py:116-124`), and the frontend state model is still three-valued (`HfAuthContext.tsx:12-22`, `:37-64`) with the "Hugging Face CLI not configured" copy at `HfAuthDialog.tsx:51`.

#### 11. No global hardware-session exclusion — **STILL REAL, and the impact is larger than the source list states** (PR #7 / T4 not merged)

Confirmed: three independent managers behind independent routes with no cross-guard (`makermodslab/server.py:1804-1871`), and the legacy single-arm auto path is still routed (`:1834-1849`).

The source list judged this P2 because "the normal UI drives one path at a time". On `main` that reasoning no longer holds, because the batch auto-calibration subprocess **deliberately survives closing the settings window** (`RobotConfigDialog.tsx:571-576`). The reciprocal-flag mutex covers only teleop ↔ record ↔ inference:

- `teleoperate.py:567-581` — checks `recording_active` and `inference_active` only
- `record.py:461-472` — checks teleop and inference only
- `rollout.py:918-930` — checks teleop and recording only

None of them consults `calibration_manager`, `auto_calibration_manager`, or `auto_calibration_batch_manager`. See NEW-2 below for the reachable consequence.

#### 12. Calibration-library mutation unguarded during an active calibration — **STILL REAL**

Delete (`server.py:1911-1964`), upload (`:2016-2048`) and rename (`:2051-2073`) still perform no calibration-activity check. The mitigations noted in the source list are present: delete unassigns referencing records via `clear_config_references` (`config.py:1108-1135`) and rename repoints them (`:1062-1105`). The library controls are also still enabled in the UI during a live session — `CalibrationLibrary.tsx` has no `disabled` wiring for calibration state, and `RobotConfigDialog.tsx:2184-2196` renders it unconditionally.

#### 13. macOS camera enumeration fallback — **STILL REAL (static) / CANNOT VERIFY STATICALLY (manifestation)**

The Darwin branch returns the AVFoundation list directly with no OpenCV retry (`server.py:2332-2336`); a helper failure is swallowed inside `_avfoundation_cameras_in_cv2_order` and returns `[]` (`:2206-2213`), so the endpoint answers `status: "success", cameras: []`. **What would settle it:** a packaged (non-editable, non-repo-venv) install on macOS, running `curl localhost:8000/available-cameras` with a camera plugged in, plus the server log line "AVFoundation enumeration subprocess failed". Not attempted — requires running the app against live USB cameras.

---

## Part 2 — New findings

Ranked by the source list's rubric (P0 = hardware safety, data loss/corruption, silent private-data publication, or main path completely broken).

### NEW-1 · **P0** — A stopped or failed auto-calibration leaves the servos' persistent calibration registers wiped, with no rollback

**Files:** `makermodslab/vendor/feetech_autocal/auto_calibrate_script.py`

Stage 0 of the vendored script destroys the arm's existing persistent calibration **before any measurement begins**:

- `_init_checks()` includes `("Homing_Offset", 0)` (`auto_calibrate_script.py:726`, table at `:713-726`), written to every motor at `:744`.
- `bus.write_position_limits(m, 0, 4095)` at `:751` clears the servo's soft travel limits.
- Both writes happen with `Lock=1` (`:717`) — i.e. into EEPROM — for all six motors, in `_run_init` (`:728-760`).

The calibration is only written back at the *natural end* of a fully successful run (`:906-925`). Neither terminal failure path restores anything:

- `KeyboardInterrupt` (what a Stop sends): `_graceful_stop` + `safe_disable_all`, then `return 130` (`:684-690`) — freeze/return-to-pose and torque release only.
- `Exception`: `safe_disable_all`, `return 1` (`:691-699`).

There is no snapshot of the pre-run `Homing_Offset` / `Min_Position_Limit` / `Max_Position_Limit`, and nothing in MakerMods Lab restores them afterwards.

**Why this is P0 rather than a duplicate of entry 2:**

1. Auto-calibration is now the **primary** calibration action on `main` (`RobotConfigDialog.tsx:1431-1438` — "Auto-calibrate" is the default button; "Calibrate manually" is the outline secondary), so this is the main path, not the fallback.
2. Nothing re-writes the file's calibration into the servos on a later session. MakerMods Lab connects with `calibrate=False` (`makermodslab/record.py:1265`, `:1324`), and pinned lerobot's `MotorsBus.connect()` does **not** write calibration (`.venv/.../lerobot/motors/motors_bus.py:513-527`). `_normalize` applies only the file's `range_min`/`range_max` to the raw `Present_Position` that the servo has already offset by its own `Homing_Offset` register (`motors_bus.py:850-877`). So after a cancelled auto-cal, every normalized reading and every `_unnormalize`d goal position is shifted by the destroyed offset — silently, for the whole session.
3. `Min/Max_Position_Limit` left at `(0, 4095)` removes the firmware clamp that stops the follower being commanded outside its safe travel. That is a hardware-safety regression produced by an ordinary "Stop".
4. Combined with entry 1 (which deletes the follower's calibration *file* on the very same stop), a single cancelled recalibration can destroy **both** the file and the servo state.

The excluded arm-identity guard is the only thing that notices — and it warns and proceeds.

**Cannot verify statically:** the exact post-cancel register values on hardware. The control flow above is unambiguous; a bench check (read `Homing_Offset` and position limits after a Stop mid-autocal) would confirm magnitude.

### NEW-2 · **P1** — Nothing prevents starting teleoperation, recording, or inference while arms are auto-calibrating under torque

**Files:** `makermodslab/server.py`, `makermodslab/teleoperate.py`, `makermodslab/record.py`, `makermodslab/rollout.py`, `frontend/src/components/dialogs/RobotConfigDialog.tsx`

This is the reachable consequence of entry 11, and it is reachable **through the normal UI**, not only by a direct API caller:

1. The batch auto-calibration is designed to survive closing the settings window (`RobotConfigDialog.tsx:566-576`: "The batch auto-calibration subprocess is deliberately designed to SURVIVE close … it is intentionally NOT aborted on close").
2. Once the window is closed, the Launchpad/studio offers Teleoperate / Collect / Deploy as usual — no calibration-activity check anywhere in their start paths (`teleoperate.py:567-581`, `record.py:461-472`, `rollout.py:918-930`).
3. Serial-port exclusivity only saves the case where the *same* ports are involved. On a **bimanual** rig the user can auto-calibrate the right pair while teleoperating the left pair: different ports, both start, and two features drive parts of the same machine simultaneously — one of them moving on its own under torque with the operator's hands on the leader.

PR #7 (`fix/t4-cross-feature-mutex`, T4) adds calibration, auto-calibration and wiggle to the mutex. It is **closed unmerged**, so `main` has no such guard.

### NEW-3 · **P1** — Calibration write-back bypasses the duplicate-port guard, and the resulting record still reports `is_clean: true`

**Files:** `makermodslab/calibrate.py`, `makermodslab/auto_calibrate.py`, `makermodslab/utils/config.py`, `frontend/src/components/dialogs/RobotConfigDialog.tsx`

`port_slot_conflict` and `config_slot_conflict` are enforced **only** inside the `/robots/{name}` HTTP handler (`makermodslab/server.py:2450-2486`, definitions at `config.py:519-537` and `:647-664`). Both calibration write-backs call `save_robot_record` directly (`calibrate.py:649-651`, `auto_calibrate.py:341-348`), and `save_robot_record` performs no conflict check (`config.py:387-427`).

The port swap/take flow makes this reachable in the ordinary first-run sequence "Detect the port, then Calibrate":

1. `handleDetect` / `handleSelectPort` stage a swap into the **local draft only** — `persistPorts({[releasedField]: swapPort ?? "", [portField]: detected})` (`RobotConfigDialog.tsx:745-781`, `:1201-1213`). Nothing is written to the record yet.
2. `handleStartCalibration` sends `port` (which is the drafted value, synced at `:1169-1173`) to `/start-calibration` (`:1018-1029`). The batch path does the same via `slotPort` → `draftPort` (`:813-829`, `:931-938`).
3. On success the backend writes back **only the calibrated slot's** port (`calibrate.py:642-651`). The other half of the swap — clearing or reassigning the released slot — never reaches disk.

Result: the record now names the **same serial port on two arms**, a state the API would have rejected with 409. It is silent, because `is_robot_record_clean` checks only field-non-emptiness and calibration-file existence (`config.py:474-516`) — it never calls `port_slot_conflict`. The robot therefore shows `is_clean: true` / "All changes saved", and the failure only surfaces later as an opaque busy-port error when teleop tries to open one bus twice.

Secondary defect in the same path: the write-back violates the window's own stated contract. The Quit dialog says "Closing now discards them — **nothing was written to the robot**" (`RobotConfigDialog.tsx:2421-2425`), but a calibration run has already persisted the drafted port, and Discard cannot undo it.

### NEW-4 · **P1** — Camera identity binds to the browser `deviceId` via fuzzy name matching; the backend's stable `unique_id` is discarded

**Files:** `makermodslab/server.py`, `frontend/src/hooks/useAvailableCameras.ts`, `frontend/src/components/recording/CameraConfiguration.tsx`

The backend already computes the stable macOS camera identity and returns it: `/available-cameras` emits `{"index", "name", "unique_id"}` per device (`server.py:2179-2182`, `:2332-2336`), where `unique_id` is AVFoundation's `uniqueID` — the only field that distinguishes two identical camera models.

The frontend **never reads it**. `useAvailableCameras` types the backend payload as `{index, name?, available}` (`useAvailableCameras.ts:80-85`) and binds each backend index to a browser `deviceId` by **fuzzy label matching** — exact, then prefix, then either-contains — with a first-come-first-served `used` set (`:87-105`). `grep -rn "unique_id" frontend/src/` returns no hits. Two cameras of the same model have the *same* `localizedName`, so all three tiers tie and the pairing is decided by array order: backend cameras are `uniqueID`-sorted, browser devices come in `enumerateDevices()` order, and the two orders are unrelated.

Downstream, the (possibly mis-bound) `device_id` is persisted on the camera record and used as the authority to **rewrite `camera_index`** on every enumeration change:

```
const match = availableCameras.find((m) => m.deviceId === cam.device_id);
if (match && match.index !== cam.camera_index) { changed = true; return { ...cam, camera_index: match.index }; }
```
(`CameraConfiguration.tsx:93-109`)

Consequences: the operator names a camera from a preview driven by `deviceId` while the recorder opens `camera_index`, so "wrist" and "front" can be swapped silently, and the binding can flip between sessions. The recorded dataset's image keys then describe the wrong physical camera — a data-correctness failure that survives into training. This is precisely the rig documented as three same-model USB cameras.

Windows and Linux are worse in one respect and better in another: `_windows_cameras` and `_linux_cameras` return no `unique_id` at all (`server.py:2230-2248`, `:2278-2310`), so there is no stable identity to fall back on — but their names (DirectShow FriendlyName / V4L2 card name) are equally non-unique across identical models.

**Cannot verify statically:** the actual mis-binding, which needs two same-model cameras on a live machine. The absence of any `unique_id` consumer in `frontend/src/` is verifiable and confirmed (`grep -rn "unique_id" frontend/src/` → no hits).

### NEW-5 · **P2** — `clamp_motor_power` raises on NaN/Infinity, contradicting its documented "never raises" contract, and can hide every robot from the UI

**File:** `makermodslab/utils/config.py:300-309`

The docstring promises "Anything non-numeric … falls back to `DEFAULT_MOTOR_POWER` rather than raising, so a corrupted record can never block a session start." But the guard is `isinstance(value, (int, float))`, which NaN and Infinity satisfy, and `int(value)` is evaluated **first** inside the clamp expression: `int(float("nan"))` raises `ValueError`, `int(float("inf"))` raises `OverflowError`.

Python's `json.load` accepts the bare literals `NaN`, `Infinity`, `-Infinity`, so a record on disk carrying one makes `get_robot_record` raise past its `except (json.JSONDecodeError, OSError)` handler (`:340-369`), which propagates through `list_robot_records` (`:372-384`) to `get_robots`, whose blanket `except Exception` returns `{"status": "error", "robots": []}` (`server.py:2382-2390`). **One malformed record hides every robot in the UI**, not just its own.

Reachability is low — MakerMods Lab's own writers can't produce it (`save_robot_record` would raise before writing) so it takes a hand-edited or externally-synced file — hence P2 rather than P1. The fix is a one-line reorder plus a `math.isfinite` check.

### NEW-6 · **P2** — Auto-calibrated leader runs leave a permanent scratch copy in `robots/so_leader/`

**File:** `makermodslab/auto_calibrate.py:313-329`

On a successful leader auto-calibration, `_finalize_success` copies `robots/so_leader/<stem>.json` → `teleoperators/so_leader/<stem>.json` (`:323-329`) but never removes the source. The scratch dir therefore accumulates stale leader calibrations that are invisible to the library UI (`get_calibration_configs` reads only `LEADER_CONFIG_PATH`/`FOLLOWER_CONFIG_PATH`, `server.py:1874-1908`) but *are* readable by anything that walks `calibration/robots/`. Worse, they go stale: a later library rename/delete of the leader config (`config.py:1062-1135`) never touches the scratch copy, so an out-of-date profile persists under the old name indefinitely.

### NEW-7 · **P2** — `ports/{leader,follower}_port.txt` is dead persistence, and `/robot-port/{robot_type}` has no caller

**Files:** `makermodslab/utils/config.py:34-37`, `:246-265`; `makermodslab/server.py:2356-2361`

Nothing in `makermodslab/` writes `LEADER_PORT_FILE` or `FOLLOWER_PORT_FILE` — the only references are the constants and the reader `get_saved_robot_port` (grep over `makermodslab/` returns `config.py:120-124`, `:248` and nothing else). `get_default_robot_port` therefore always falls through to the hardcoded `"COM3"` / `"/dev/ttyUSB0"` (`:258-265`), and `grep -rn "robot-port" frontend/src/` has no hits, so the endpoint is unreachable from the UI. Ports live on the robot record now. This is stale surface area rather than a live defect, but `MakerLab/CLAUDE.md` still documents these files as "last-used serial ports", which will mislead the next reader.

---

## (c) Contradictions to the brief

1. **PR state.** The brief describes #5–#11 as "OPEN". They are **CLOSED without merge** (all seven verified via `gh pr view … --json state,mergedAt`; `mergedAt: null` for every one). The operative conclusion is unchanged and stronger: not merged, and no longer on track to be — `main` contains the bugs, and the fixes are not queued.
2. **Audited branch.** The brief says the source list was verified against `andrew`. The document itself records "branch `min_stable`, commit `518ca56`" (line 256), with the two fix write-ups apparently describing `andrew` work. Either way the fixes are absent from `main`; verdicts above are against `main` only.
3. **File paths in the brief's Part-2 reading list.** `frontend/src/components/calibration/` contains only `CalibrationLibrary.tsx` and `ImportCalibrationButton.tsx` — the calibration *flow* lives in `RobotConfigDialog.tsx`. `makermodslab/wiggle.py` is documented in `CLAUDE.md` as legacy, but it is live: `/wiggle` (`server.py:2124-2127`) is wired to the "Wiggle" button (`RobotConfigDialog.tsx:2034-2051`) and it does drive the gripper.

## (d) What could not be verified statically

| Claim | Why it needs more than static reading | What would settle it |
|---|---|---|
| Entry 13 — AVFoundation helper actually failing | Depends on PyObjC availability in a *packaged* install, not the repo venv | Packaged macOS install; `curl :8000/available-cameras`; watch for the "AVFoundation enumeration subprocess failed" warning |
| NEW-1 — post-cancel register values | Control flow is unambiguous; the magnitude of the offset error is not | Bench: read `Homing_Offset` + `Min/Max_Position_Limit` on all six motors before an autocal, Stop mid-run, re-read |
| NEW-4 — actual camera mis-binding | Needs two same-model USB cameras and a real `enumerateDevices()` ordering | Plug two identical cameras; compare `/available-cameras` `unique_id` ordering against `navigator.mediaDevices.enumerateDevices()` labels/ids across two page loads |
| NEW-2 — bimanual concurrent-drive | Requires a bimanual rig with four ports | Start a batch autocal on the right pair, close the window, start teleop on the left pair |
| Entries 4, 5 — race windows | Timing-dependent | Threaded unit tests (a mocked worker outliving the 5 s join; two threads paused between read and write in `save_robot_record`) |

No hardware was touched, no serial port or camera was opened, no server was started, no network or Hub call was made, and no repository file was modified other than this report.
