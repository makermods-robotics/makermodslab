#!/usr/bin/env python
"""Bench tool: find a leader-arm torque that holds its own weight but yields to a hand.

WHY THIS EXISTS
---------------
During a coaching takeover, lerobot's `_apply_transition` drives the leader to
the follower's pose and then calls `teleop.disable_torque()` — so the arm goes
from actuated to completely limp in one step, and the operator catches a
falling arm. What we want instead is a leader that is *slightly energized*:
enough to carry its own weight where you left it, weak enough that grabbing it
never fights back hard enough to hurt anyone or strain a servo.

This script is a bench rig for finding that number by feel. It is deliberately
standalone — it does not import the app, does not touch the UI, and changes
nothing persistent. Run it, grab the arm, turn the dials, write down what felt
right. Wiring the winning value into the runner is a separate job.

HOW COMPLIANCE IS ACHIEVED
--------------------------
Two independent dials, and it matters that they are different things:

  * `Torque_Limit` (RAM, address 48) caps how hard a servo may ever pull. This
    is the safety dial: it bounds the worst case when a human and the servo
    disagree.
  * `P_Coefficient` (address 21) sets how hard it pulls *per unit of error*.
    This is the feel dial: high P with a low cap snaps to the limit and stays
    there; low P eases in.

...plus a third, which is this script's own idea rather than a register:

  * DEADBAND. Each tick the goal position is re-pinned to wherever the arm
    actually is, but ONLY once it has moved further than the deadband. Inside
    the deadband the servo is holding a fixed target, so it carries the arm's
    weight. Push past it and the target follows your hand, so the arm yields
    and — this is the point — STAYS where you put it instead of springing back.

    Deadband 0 makes the arm nearly limp (the goal chases every wobble).
    A large deadband makes it feel stiff and springy. The sweet spot is small.

SAFETY
------
  * `Max_Torque_Limit` (EEPROM, address 16) is NEVER written. This project has
    been burned by persistent EEPROM state; everything here is RAM and dies
    with a power cycle.
  * The original `Torque_Limit` and `P_Coefficient` of every motor are read
    before anything is changed and restored on the way out, including on
    Ctrl-C and on an unhandled exception.
  * Torque starts OFF and at a low cap, and the arm is never commanded to a
    pose it is not already in — the goal is seeded from the present position,
    so energizing cannot make it jump.
  * Nothing else may hold the serial port. Stop the app first
    (`makermodslab --stop`) or this will fail to connect.

USAGE
-----
    python tools/leader_compliance.py --list
    python tools/leader_compliance.py --robot my_robot
    python tools/leader_compliance.py --port /dev/tty.usbmodem58FA0827721

Then type commands at the prompt (`?` for help). Suggested first session:
energize with `on`, then walk `t` up from 100 in steps of 50 until the arm
stops sagging, then back off until a firm push moves it without drama.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

ROBOTS_PATH = Path(os.path.expanduser("~/.cache/huggingface/lerobot/robots"))

# Feetech STS3215 register ranges. Torque_Limit is 0-1000 in the servo's own
# units (it is a permille of Max_Torque_Limit, not a percentage).
TORQUE_LIMIT_MAX = 1000
P_COEFF_MAX = 254

# Where a session starts. Low on purpose: it is much nicer to walk a weak arm
# up until it holds than to meet one that is already stiff.
DEFAULT_TORQUE = 150
DEFAULT_DEADBAND = 12  # raw encoder ticks (4096 per turn)
DEFAULT_HZ = 50.0

# Refuse to go above this without --max-torque. At full scale an SO-101 servo
# can hurt a finger and stall against a hard stop.
DEFAULT_TORQUE_CEILING = 500

# STS3215 firmware default. lerobot writes 16 on the FOLLOWER (so_follower.py:165,
# "to avoid shakiness") but never touches the LEADER, so a leader's stock value
# is the firmware's own.
STOCK_P_COEFFICIENT = 32


def restore_stock(leader, *, reset_p: bool) -> None:
    """Put a leader back to stock and release it. The panic button.

    Needed because two of the registers this tool writes outlive the process:

      * `Torque_Limit` is SRAM, so it survives until the servo loses power. A
        run that is killed, or whose restore writes fail, leaves the arm capped
        wherever it was last set — which reads as "the leader is suddenly far
        too weak to reach its position".
      * `Torque_Enable` likewise: a killed run leaves the arm ENERGIZED and
        stiff, so it cannot be repositioned by hand.
      * `P_Coefficient` is worse — address 21 is in the servo's EPROM block, so
        it survives a POWER CYCLE. This is the persistent-state trap
        makermodslab/motor_power.py warns about, and this tool should not have
        been writing it without saying so.

    Torque_Limit is restored from `Max_Torque_Limit`, the servo's own power-on
    source, exactly as `motor_power.reset_torque_limit` does.
    """
    bus = leader.bus
    print("Restoring stock values...")
    for motor in bus.motors:
        try:
            stock = bus.read("Max_Torque_Limit", motor, normalize=False)
            bus.write("Torque_Limit", motor, stock, normalize=False, num_retry=2)
            print(f"  {motor:<16} Torque_Limit -> {stock} (from Max_Torque_Limit)")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {motor}: could not restore Torque_Limit: {exc}")
        if reset_p:
            try:
                bus.write("P_Coefficient", motor, STOCK_P_COEFFICIENT, normalize=False, num_retry=2)
                print(f"  {motor:<16} P_Coefficient -> {STOCK_P_COEFFICIENT} (firmware default)")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {motor}: could not restore P_Coefficient: {exc}")
    try:
        leader.disable_torque()
        print("  torque released — the arm should move freely by hand now")
    except Exception as exc:  # noqa: BLE001
        print(f"  ! disable_torque failed: {exc}")


def find_robot_records() -> list[dict]:
    records = []
    if not ROBOTS_PATH.is_dir():
        return records
    for path in sorted(ROBOTS_PATH.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        data["_name"] = path.stem
        records.append(data)
    return records


def resolve_leader(args) -> tuple[str, str | None]:
    """Return (port, calibration_id) for the leader arm to drive."""
    if args.port:
        return args.port, args.id
    records = find_robot_records()
    if args.robot:
        for r in records:
            if r["_name"] == args.robot:
                port = r.get("leader_port")
                if not port:
                    sys.exit(f"Robot {args.robot!r} has no leader_port set.")
                return port, args.id or r.get("leader_config")
        sys.exit(f"No robot record named {args.robot!r}. Try --list.")
    with_leader = [r for r in records if r.get("leader_port")]
    if len(with_leader) == 1:
        r = with_leader[0]
        print(f"Using the only robot with a leader arm: {r['_name']}")
        return r["leader_port"], args.id or r.get("leader_config")
    sys.exit("Pass --robot NAME or --port. Use --list to see what's configured.")


class ComplianceRig:
    """Owns the bus, the hold loop, and the promise to put things back."""

    def __init__(self, leader, torque: int, p_coeff: int | None, deadband: int, hz: float, ceiling: int):
        self.leader = leader
        self.bus = leader.bus
        self.motors = list(self.bus.motors)
        self.torque = torque
        self.p_coeff = p_coeff
        self.deadband = deadband
        self.period = 1.0 / hz
        self.ceiling = ceiling

        # "hold" pins the goal where you left it: the servo fights to stay
        # there, so this is where you actually FEEL the torque cap and can
        # judge "does it carry its own weight". It springs back when released,
        # gently, bounded by Torque_Limit.
        #
        # "follow" re-pins the goal past the deadband, so the arm yields and
        # STAYS where you put it — the end behaviour we want during a takeover,
        # but by construction it offers almost no resistance (the goal chases
        # your hand within a degree), so it is the wrong mode for choosing a
        # number. Pick the torque in hold, then confirm the feel in follow.
        self.mode = "hold"
        self.energized = False
        self._stop = threading.Event()
        # ONE lock, and it guards the SERIAL PORT as well as the goal dict.
        #
        # The Feetech SDK's port handler is not thread-safe: it carries an
        # `is_using` flag and rejects a second caller with "port is in use"
        # while a packet transaction is open. The hold loop transacts at 50 Hz,
        # so any REPL command that touched the bus raced it — the write was
        # refused, the transaction was left half-done, and every later write
        # landed in a confused bus. Reentrant because the outer methods take it
        # and then call helpers that take it again.
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._goal: dict[str, int] = {}
        self._last_load: dict[str, int] = {}
        # Everything we change, as we found it.
        self._original: dict[str, dict[str, int]] = {}

    # -- register plumbing --------------------------------------------------

    def snapshot_originals(self) -> None:
        with self._lock:
            self._snapshot_originals_locked()

    def _snapshot_originals_locked(self) -> None:
        for motor in self.motors:
            entry = {}
            for reg in ("Torque_Limit", "P_Coefficient"):
                try:
                    entry[reg] = self.bus.read(reg, motor, normalize=False)
                except Exception as exc:  # noqa: BLE001 - bench tool, report and continue
                    print(f"  ! could not read {reg} on {motor}: {exc}")
            self._original[motor] = entry

    def restore_originals(self) -> None:
        with self._lock:
            self._restore_originals_locked()

    def _restore_originals_locked(self) -> None:
        print("\nRestoring the registers this script changed...")
        for motor, entry in self._original.items():
            for reg, value in entry.items():
                try:
                    self.bus.write(reg, motor, value, normalize=False, num_retry=2)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! could not restore {reg} on {motor}: {exc}")
        print("  restored (RAM only — Max_Torque_Limit was never touched).")

    def warn_if_cold(self) -> None:
        if not self.energized:
            print("  NOTE: torque is OFF — the arm is limp and you will feel nothing.")
            print("        Type `on` to energize, then try again.")

    def apply_torque_limit(self, value: int) -> None:
        value = max(0, min(int(value), self.ceiling))
        self.torque = value
        with self._lock:
            for motor in self.motors:
                try:
                    self.bus.write("Torque_Limit", motor, value, normalize=False, num_retry=2)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! Torque_Limit write failed on {motor}: {exc}")
        print(f"  Torque_Limit = {value} / {TORQUE_LIMIT_MAX} (ceiling {self.ceiling})")
        self.warn_if_cold()

    def apply_p_coeff(self, value: int) -> None:
        value = max(0, min(int(value), P_COEFF_MAX))
        self.p_coeff = value
        with self._lock:
            for motor in self.motors:
                try:
                    self.bus.write("P_Coefficient", motor, value, normalize=False, num_retry=2)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! P_Coefficient write failed on {motor}: {exc}")
        print(f"  P_Coefficient = {value}")
        print("  NOTE: P_Coefficient lives in EPROM — this SURVIVES A POWER CYCLE.")
        print(f"        Stock is {STOCK_P_COEFFICIENT}; `--restore --reset-p` puts it back.")
        self.warn_if_cold()

    # -- the hold loop ------------------------------------------------------

    def _read_positions(self) -> dict[str, int]:
        with self._lock:
            return self.bus.sync_read("Present_Position", normalize=False, num_retry=1)

    def seed_goal(self) -> None:
        """Pin the target to where the arm already is, so nothing jumps."""
        with self._lock:
            self._goal = dict(self._read_positions())

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                present = self._read_positions()
                with self._lock:
                    deadband = self.deadband
                    following = self.mode == "follow"
                    for motor, pos in present.items():
                        target = self._goal.get(motor)
                        if target is not None and not following:
                            continue  # hold: the target never moves
                        # Outside the deadband the human is clearly moving it:
                        # follow, so the arm yields and stays put rather than
                        # springing back. Inside it, hold — that is what carries
                        # the arm's own weight.
                        if target is None or abs(pos - target) > deadband:
                            self._goal[motor] = pos
                    goal = dict(self._goal)
                if self.energized:
                    with self._lock:
                        self.bus.sync_write("Goal_Position", goal, normalize=False, num_retry=1)
                        # Load is diagnostics only; never break the hold loop.
                        with contextlib.suppress(Exception):
                            self._last_load = self.bus.sync_read("Present_Load", normalize=False)
            except Exception as exc:  # noqa: BLE001
                print(f"\n  ! hold loop: {exc}")
            time.sleep(max(0.0, self.period - (time.perf_counter() - started)))

    def start(self) -> None:
        self.seed_goal()
        self._thread = threading.Thread(target=self._loop, name="hold", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    # -- energize / release -------------------------------------------------

    def energize(self) -> None:
        if self.energized:
            print("  already on")
            return
        # Seed FIRST. Enabling torque with a stale goal is how you get a jump.
        self.seed_goal()
        self.apply_torque_limit(self.torque)
        if self.p_coeff is not None:
            self.apply_p_coeff(self.p_coeff)
        with self._lock:
            self.leader.enable_torque()
        # enable_torque may reset the cap on some firmware; re-assert it.
        self.apply_torque_limit(self.torque)
        self.energized = True
        print("  ENERGIZED — grab the arm and see how it feels.")

    def release(self) -> None:
        self.energized = False
        try:
            with self._lock:
                self.leader.disable_torque()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! disable_torque failed: {exc}")
        print("  released (limp)")

    def status(self) -> None:
        try:
            present = self._read_positions()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not read positions: {exc}")
            return
        print(
            f"\n  torque={self.torque}  P={self.p_coeff}  deadband={self.deadband} ticks  "
            f"mode={self.mode}  {'ON' if self.energized else 'OFF'}"
        )
        # Read the registers back rather than trusting our own bookkeeping: if
        # a write silently failed, or firmware reset a value on torque enable,
        # this is the only place that would show it.
        for reg in ("Torque_Enable", "Torque_Limit"):
            try:
                with self._lock:
                    actual = self.bus.sync_read(reg, normalize=False)
                shown = ", ".join(f"{m}={v}" for m, v in actual.items())
                print(f"  {reg:<14} {shown}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! could not read {reg} back: {exc}")
        print(f"  {'motor':<16}{'present':>9}{'goal':>9}{'err':>7}{'load':>8}")
        with self._lock:
            goal = dict(self._goal)
        for motor in self.motors:
            pos = present.get(motor)
            tgt = goal.get(motor)
            err = "" if pos is None or tgt is None else str(pos - tgt)
            load = self._last_load.get(motor, "")
            print(
                f"  {motor:<16}{pos if pos is not None else '':>9}"
                f"{tgt if tgt is not None else '':>9}{err:>7}{load:>8}"
            )


HELP = """
  on              energize (hold current pose, weakly)
  off             release torque — arm goes limp
  t <0-1000>      Torque_Limit: the SAFETY cap, how hard it may ever pull
  p <0-254>       P_Coefficient: the FEEL, how hard it pulls per unit of error
  d <ticks>       deadband: how far your hand must move it before it yields
  mode hold       goal stays put — you FEEL the torque; pick your number here
  mode follow     goal yields past the deadband — arm stays where you put it
  sync            re-pin the goal to where the arm is right now
  s               status (positions, error, measured load)
  ?               this help
  q               quit (restores registers, releases torque)
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", help="name of a saved robot record to take the leader port from")
    ap.add_argument("--port", help="serial port of the leader arm (overrides --robot)")
    ap.add_argument("--id", help="calibration id; defaults to the record's leader_config")
    ap.add_argument("--list", action="store_true", help="list saved robot records and exit")
    ap.add_argument(
        "--restore",
        action="store_true",
        help="RECOVERY: release torque and put Torque_Limit back to stock, then exit. "
        "Use after a run that was killed or errored and left the arm weak or stiff.",
    )
    ap.add_argument(
        "--reset-p",
        action="store_true",
        help="with --restore, also reset P_Coefficient to the firmware default (32). "
        "P lives in EPROM and survives a power cycle, so this is the only way back.",
    )
    ap.add_argument(
        "--torque", type=int, default=DEFAULT_TORQUE, help=f"starting Torque_Limit (default {DEFAULT_TORQUE})"
    )
    ap.add_argument("--p", type=int, default=None, help="starting P_Coefficient (default: leave as found)")
    ap.add_argument(
        "--deadband", type=int, default=DEFAULT_DEADBAND, help=f"ticks (default {DEFAULT_DEADBAND})"
    )
    ap.add_argument("--hz", type=float, default=DEFAULT_HZ, help=f"hold-loop rate (default {DEFAULT_HZ})")
    ap.add_argument(
        "--max-torque",
        type=int,
        default=DEFAULT_TORQUE_CEILING,
        help=f"safety ceiling for `t` (default {DEFAULT_TORQUE_CEILING}, max {TORQUE_LIMIT_MAX})",
    )
    args = ap.parse_args()

    if args.list:
        records = find_robot_records()
        if not records:
            print(f"No robot records under {ROBOTS_PATH}")
            return 0
        for r in records:
            leader = r.get("leader_port") or "-"
            print(
                f"  {r['_name']:<24} leader_port={leader:<28} leader_config={r.get('leader_config') or '-'}"
            )
        return 0

    ceiling = max(0, min(args.max_torque, TORQUE_LIMIT_MAX))
    port, calib_id = resolve_leader(args)

    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    print(f"Leader port : {port}")
    print(f"Calibration : {calib_id or '(none — running uncalibrated, raw ticks)'}")
    print(f"Ceiling     : Torque_Limit <= {ceiling}")
    print("\nNothing else may hold this port — stop the app first if it is running.")
    if input("Connect? [y/N] ").strip().lower() not in {"y", "yes"}:
        return 1

    cfg = SO101LeaderConfig(port=port, id=calib_id) if calib_id else SO101LeaderConfig(port=port)
    leader = SO101Leader(cfg)
    if args.restore:
        leader.connect(calibrate=False)
        try:
            restore_stock(leader, reset_p=args.reset_p)
        finally:
            with contextlib.suppress(Exception):
                leader.disconnect()
        print("Done. Power-cycling the arm is still the belt-and-braces option.")
        return 0
    # calibrate=False: this is a bench tool and must never launch the
    # interactive range-of-motion flow. Raw ticks are what we work in anyway.
    leader.connect(calibrate=False)
    print("Connected.")

    rig = ComplianceRig(leader, args.torque, args.p, args.deadband, args.hz, ceiling)
    rig.snapshot_originals()
    print(f"Captured original registers for {len(rig._original)} motors.")
    rig.start()

    def shutdown(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)

    print(HELP)
    try:
        while True:
            try:
                state = "ON " if rig.energized else "OFF"
                raw = input(f"leader[{state} t={rig.torque} {rig.mode}]> ").strip()
            except EOFError:
                break
            if not raw:
                continue
            cmd, _, arg = raw.partition(" ")
            cmd = cmd.lower()
            if cmd == "q":
                break
            # One bad bus transaction must not tear down the session: that
            # would leave the prompt gone with the arm still energized and no
            # way to type `off`. Report and stay put.
            try:
                if cmd == "?":
                    print(HELP)
                elif cmd == "on":
                    rig.energize()
                elif cmd == "off":
                    rig.release()
                elif cmd == "s":
                    rig.status()
                elif cmd == "mode":
                    want = arg.strip().lower()
                    if want in {"hold", "follow"}:
                        with rig._lock:
                            rig.mode = want
                        rig.seed_goal()
                        print(f"  mode = {want}")
                        rig.warn_if_cold()
                    else:
                        print("  usage: mode hold | mode follow")
                elif cmd == "sync":
                    rig.seed_goal()
                    print("  goal re-pinned to the current pose")
                elif cmd == "t":
                    try:
                        rig.apply_torque_limit(int(arg))
                    except ValueError:
                        print("  usage: t <0-1000>")
                elif cmd == "p":
                    try:
                        rig.apply_p_coeff(int(arg))
                    except ValueError:
                        print("  usage: p <0-254>")
                elif cmd == "d":
                    try:
                        with rig._lock:
                            rig.deadband = max(0, int(arg))
                        print(f"  deadband = {rig.deadband} ticks")
                    except ValueError:
                        print("  usage: d <ticks>")
                else:
                    print("  ? for help")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {cmd}: {exc}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        # Order matters: stop commanding, then unpower, then put the registers
        # back, then let go of the port. Every step is best-effort so one
        # failure cannot strand the arm energized.
        rig.stop()
        try:
            rig.release()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! release failed: {exc}")
        try:
            rig.restore_originals()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! restore failed: {exc}")
        try:
            leader.disconnect()
            print("Disconnected.")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! disconnect failed: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
