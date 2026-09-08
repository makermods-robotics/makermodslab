# P0 Reproduction — executable harness + bench procedures

Companion to [`CURRENT 23-07-26.md`](CURRENT%2023-07-26.md). That doc says *what* is broken; this one says
**how to make it happen on demand**, so a fix can be shown to change something.

Two halves:

- **`tests/repro/`** — the automated harness. Everything reachable without hardware, without a network write,
  and without touching the real `~/.cache/huggingface/lerobot`.
- **This document, below** — the bench procedures for the halves that terminate in real servo EEPROM writes or
  a real energized arm. Those were deliberately **not executed**.

> **P0-2 is out of scope.** Public-by-default dataset upload was ruled INTENDED product behaviour, so it is
> neither reproduced here nor listed below. The remaining four keep their original numbers — nothing is
> renumbered.

---

## Triage table

| P0 | What it is | Reproduced in code | Needs the bench | Needs a throwaway HF account |
|---|---|---|---|---|
| **P0-1** | Auto-cal wipes servo EEPROM at Stage 0, no restore on Stop | ✅ Stage-0 write ordering, absent snapshot, KeyboardInterrupt path leaving `Homing_Offset=0` / limits `(0,4095)`, and the calibration-*file* deletion (C1) — all against a fake bus | ⚠️ **Yes, for the physical consequence only**: that the servos really hold the wiped values after a UI Stop, and that the next session's readings are shifted | No |
| **P0-3** | Inference Stop can leave the follower torque-enabled; abandon path leaks a live untracked child | ✅ No `kill()` fallback, no surviving handle, no `force_disable_bus_torque` anywhere in `rollout.py`, stop returns success after SIGKILL | ⚠️ **Yes, for the timing premise only**: that a real teardown exceeds the hard 5 s budget, and that the follower is still energized afterwards | No |
| **P0-4** | R5 zeroed counter ⇄ R6 wholesale dataset delete | ✅ Both halves. R5 through the real recording worker; R6 through the real delete handler | ❌ No | No |
| **P0-5** | Bare dataset name + traversal-only delete guard = `rmtree` of a MakerMods Lab state dir | ✅ Fully, including over HTTP via `POST /delete-dataset` | ❌ No | No |

**Nothing in the four remaining P0s needs a throwaway Hugging Face account.** That column existed for P0-2 and
is retained only to record that the answer is now "no" across the board.

---

## The automated harness

```
.venv/bin/python -m pytest tests/repro/ -x -q
```

**Every defect assertion is `@pytest.mark.xfail(strict=True)`.** Read the result like this:

| Outcome | Meaning |
|---|---|
| `xfailed` | **The bug still reproduces.** Expected on today's `main`. |
| `XPASS` (reported as a failure, because `strict=True`) | **The bug is gone.** Promote the test into the real suite as a regression test and delete the marker. Do not "fix" it by loosening the assertion. |

So a **green run means "every documented P0 still reproduces exactly as filed"**, and CI stays green either
way — `pytest tests` collects `tests/repro/` (`testpaths = ["tests"]`), and an `xfailed` test is not a failure.

A handful of tests are deliberately **not** xfail. They are labelled "descriptive control" in their docstrings
and assert the *premises* the bug reports rest on — that Stage 0's writes really are EEPROM-persistent
(`Lock=1` first), that the worker-side discard really is resume-safe, that two-segment dataset ids really are
unaffected. If one of those ever starts failing, the corresponding entry in `CURRENT 23-07-26.md` needs
revisiting, not the test.

### Safety properties built into the harness

These are load-bearing; preserve them if you extend `tests/repro/`.

- `tests/repro/conftest.py::repro_lerobot_home` redirects the cache root **through all three mechanisms that
  actually read it** and then *verifies the redirect from the perspective of the code under test* before any
  test body runs. The three are not interchangeable:
  - `HF_LEROBOT_HOME` env var → `makermodslab.datasets._lerobot_cache_root()` (`datasets.py:329-330`);
  - `lerobot.utils.constants.HF_LEROBOT_HOME` module attribute → `handle_delete_dataset` imports it *inside
    the function body* (`record.py:862-864`), so setting the env var afterwards changes nothing;
  - `makermodslab.auto_calibrate.CALIBRATION_BASE_PATH_ROBOTS` → bound at import (`auto_calibrate.py:41`), so the
    shared `tests/conftest.py::tmp_lerobot_home` fixture (which patches `makermodslab.utils.config` only) does
    **not** redirect it.
- `assert_not_real_cache()` is called again immediately before every destructive handler invocation, so the
  guard sits next to the `rmtree` rather than only in a fixture.
- Autouse fixtures make a real socket and a real `serial.Serial` raise. Real SO-101 arms may be attached to
  the machine running the suite; a fake bus is not enough on its own.
- Module globals in `record.py` / `rollout.py` are snapshotted and restored, so a repro can never leave the
  hardware mutex latched.

---

## Bench procedures

For a person standing next to the arms with the MakerMods Lab UI open. Read the whole procedure before starting.

### Before you start — general

- **Work on a throwaway calibration name.** Do not use the calibration you rely on day to day. P0-1's whole
  point is that the run destroys it.
- **Back up the calibration directory first**, and confirm the backup is readable:
  ```bash
  cp -R ~/.cache/huggingface/lerobot/calibration ~/calibration-backup-$(date +%Y%m%d-%H%M)
  ls -R ~/calibration-backup-*
  ```
- **Clear the arm's workspace.** P0-1 leaves the firmware travel clamp at `(0, 4095)` — the end-stop
  protection is gone for the rest of the session, so a subsequent move can drive a joint into its mechanical
  limit.
- Keep the arm's power switch within reach. It is the reliable stop for both P0-1 and P0-3.

### Reading servo registers (used by P0-1, before and after)

Read-only. Run from the repo root with the arm connected and **no MakerMods Lab session running** — the serial port
takes one owner at a time, so stop teleop/record/inference and quit any auto-calibration first.

```python
# scratch_read_registers.py — READ ONLY. Writes nothing.
from makermodslab.vendor.feetech_autocal.bus import AutoCalBus
from makermodslab.vendor.feetech_autocal.calibration_defaults import MOTOR_NAMES, SO_FOLLOWER_MOTORS

PORT = "/dev/tty.usbmodem…"  # the follower port shown in the UI

bus = AutoCalBus(port=PORT, motors=SO_FOLLOWER_MOTORS.copy())
bus.connect(handshake=False)
try:
    for m in MOTOR_NAMES:
        print(f"{m:16s} Homing_Offset={bus.read('Homing_Offset', m, normalize=False):6d}  "
              f"limits={bus.read_position_limits(m)}")
finally:
    bus.disconnect(disable_torque=False)
```

Record the six lines verbatim. That table is your before/after evidence.

---

### P0-1 · Auto-calibration wipes servo EEPROM before measuring

**What the bench adds over the harness:** the harness proves the control flow — Stage 0 writes
`Homing_Offset=0` and `write_position_limits(m, 0, 4095)` behind `Lock=1` with no prior read, and the
`KeyboardInterrupt` path returns 130 without restoring either. The bench confirms the servos really persist
those values across the stop and a power cycle, and that the next session's readings are shifted.

**Steps**

1. On an arm with a **known-good calibration**, run the register read above. Expect nonzero `Homing_Offset`
   values and limits that are a real travel range (something like `(512, 3600)`), **not** `(0, 4095)`.
2. Note the calibration file that arm uses:
   `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<name>.json`. Copy it somewhere safe and
   `cat` it — you will check for its return later.
3. In the UI: open the robot's config dialog → **Auto-calibrate** (the default action; "Calibrate manually" is
   the secondary). When it warns the name already exists, choose to overwrite — that is the ordinary
   re-calibrate flow and the one that arms C1.
4. **Where to interrupt:** as soon as Stage 0 is visibly underway — the log pane shows
   `Stage 0: Initialization` and `Configuring servo: …` lines, and the arm audibly energizes — press **Stop**.
   Do this *before* the arm starts its unfold motion. Stage 0 is only a few seconds; if you miss it, let the
   run reach the unfold and stop there instead (the same failure path applies, `KeyboardInterrupt` → rc 130).
5. Wait for the UI to report the run stopped. The arm should return toward its start pose and go limp
   (`_graceful_stop` + `safe_disable_all`). **That part is expected to work** — it is not what is being
   tested.
6. **Power-cycle the arm.** This is what proves the writes were EEPROM, not RAM.
7. Re-run the register read.

**What confirms the bug**

- Every motor now reads `Homing_Offset=0`. ❗
- Every motor now reads limits `(0, 4095)`. ❗
- The calibration file from step 2 is **gone** from
  `calibration/robots/so_follower/` (this is Config C1, deleted by
  `_remove_stray_calibration_file` on the stop path).
- Both survive the power cycle.

Taken together: one Stop destroyed both the servo state and the file, and nothing in MakerMods Lab restores either.

**Secondary check (the silent-shift consequence, optional).** Without re-calibrating, start a teleop session
and note the follower's joint readings against the arm's visible pose — MakerMods Lab connects with
`calibrate=False` (`record.py:1265,1324`) and the pinned lerobot's `MotorsBus.connect()` does not write
calibration, so the offsets stay destroyed and every reading is shifted for the whole session.

**How to restore the arm afterwards**

1. Run a **full auto-calibration to natural completion** — do not stop it. The success path writes real
   calibration into EEPROM (`auto_calibrate_script.py:906-925`) and re-writes the calibration file.
2. Verify with the register read: `Homing_Offset` back to sensible nonzero values, limits back to a real
   travel range.
3. If the file did not come back, restore it from your step-2 copy (or the directory backup) and restart the
   MakerMods Lab process so the library re-reads it.
4. Do **not** leave the arm at limits `(0, 4095)` — that is the unprotected state.

---

### P0-3 · Inference Stop can leave the follower torque-enabled

**What the bench adds over the harness:** the harness proves there is no fallback release anywhere in
`rollout.py` and that `handle_stop_inference` returns `{"success": true}` after SIGKILLing a child that never
ran its teardown. The bench confirms the timing premise — that a real teardown does not fit in the hard 5 s
budget — and the physical consequence.

⚠️ **This procedure ends with a possibly-energized arm.** Keep a hand on the power switch. Do not leave the
room. Clear the workspace first.

**Steps — Inference I13 (SIGKILL mid-teardown)**

1. Start an inference run from the UI on a short duration. Let it reach `running` and actually drive the arm.
2. Press **Stop**. Start a stopwatch at the press.
3. Watch the backend log for `Inference did not exit in 5s; killing` (`rollout.py:1039`). Its presence is the
   primary signal: the child was SIGKILLed mid-teardown.
4. The moment the UI reports idle, **check the arm physically**: try to backdrive a joint by hand.
   - Limp / freely backdrivable → torque was released, I13 did not bite on this run.
   - Still stiff / holding position → ❗ **torque is still enabled with the backend reporting idle.**
5. Bimanual makes this far more likely (a second bus plus camera release inside the same 5 s). Repeat on a
   bimanual rig if you have one.
6. If the arm is left energized: power-cycle it. There is no in-app release — that absence *is* the bug.

**Steps — Inference I14 (abandoned spawn leaks a live child)**

This one is a race, so expect to need several attempts.

1. Pick a **large, not-yet-cached Hub checkpoint** so the download phase is long, and give the follower a
   valid port.
2. Start inference, then press **Stop** in the narrow window *after* the model download finishes and while
   the subprocess is starting (the UI is around `starting` / `loading_policy`).
3. The UI will report idle immediately, and the mutex will be released.
4. **Now check for an orphan:**
   ```bash
   pgrep -fl lerobot.scripts.lerobot_rollout
   ```
   ❗ A surviving PID here with the backend reporting idle **is** I14: `_inference_proc` was never assigned, so
   nothing in the app can stop it. Watch the arm — the child continues into `robot.connect()` and energizes
   the follower.
5. Kill the orphan by PID and power-cycle the arm.

**What confirms the bug**

- I13: `did not exit in 5s; killing` in the log **and** a stiff follower after the UI says idle.
- I14: a live `lerobot_rollout` PID after a stop the UI reported as complete.

**How to restore**

- Power-cycle the follower. Confirm it is backdrivable before doing anything else.
- `pgrep -fl lerobot.scripts.lerobot_rollout` again and kill any survivor.
- Restart the MakerMods Lab process before the next session — inference state is process-global.
- No calibration damage from P0-3; no restore needed beyond the above.

---

### P0-4 and P0-5 — no bench needed

Both are fully reproduced in `tests/repro/`, including P0-5 over the real HTTP boundary. **Do not exercise
them by hand against a real cache**: P0-5's delete path removes `~/.cache/huggingface/lerobot/calibration`
outright, which destroys every calibration profile on the machine. If you want to see it interactively, do it
with `HF_LEROBOT_HOME` pointed at a scratch directory **and** `lerobot.utils.constants.HF_LEROBOT_HOME`
patched — the env var alone is not enough, because `handle_delete_dataset` imports the module attribute
(`record.py:862-864`). The harness fixture is the worked example.
