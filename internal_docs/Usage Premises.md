# Usage Premises

Compiled 2026-07-09. The intended operating envelope MakerMods Lab is currently built for. These sit above [Assumptions.md](Assumptions.md) (the `[premise]` tags there derive from these) and [Core Functions.md](Core%20Functions.md) (whose "bare necessity" verdicts hold only inside this envelope). When a premise changes, re-scope deliberately — the camera excision (`8d7634d`) was exactly this: the "local Mac" premise made backend previews pure downside.

## 1. Everything on one machine (local Mac)

Backend, browser, cameras, and arm serial all on a single Mac; UI at `localhost:8000`. This is what makes getUserMedia previews sufficient, HTTPS unnecessary, and backend cv2 previews harmful (a second consumer fighting the recorder).

**If it changes** (Jetson un-pauses, LAN serving, office server): tiles vanish on insecure origins, server-side frames are needed again (snapshot endpoint, not MJPEG streamer), secure-context bugs (#34) become live, JETSON_SETUP.md/INSTALL.md paths reactivate.

## 2. A supervised bench, one operator, one session

A human is physically at the arms whenever they're powered. One browser tab (SingleTabGuard), one hardware session at a time (mutexes; the calibration one pending on `full-chain`), no concurrent users, no auth. The policy never drives the arm unwatched — leaving the inference page stops it; leaving recording quits it. Emergency recovery may assume terminal access (`curl -X POST /stop-recording`).

**If it changes** (multi-user, remote operation, unattended runs): the whole safety model needs redesign, not patches — torque-release guarantees, exit-guard semantics, and the unenforced feature exclusivity all assume an operator who notices.

## 3. Hardware is SO-101, in a fixed, labeled topology

Exactly `so101_leader`/`so101_follower` (one or two pairs; bimanual is first-class). Hobby-class arms — the tolerable failure mode is "operator grabs it", not industrial safety. USB topology is FROZEN: camera identity is the port path, same-model cameras are told apart only by port, serial adapters get their own chain/power rail, ports are labeled, roles re-verified by eye after any recabling. Sustained 3-camera recording waits on a powered hub (2-camera interim).

**If it changes** (new robot type): every feature module's config construction is touched (search `SO101LeaderConfig`). (Plugs shuffled): silent camera-role swaps — the accepted cost of index-based binding.

## 4. Offline-capable core; Hub is opportunistic

The bench loop (calibrate → teleop → record) must work with zero internet (`HF_HUB_OFFLINE=1`, bare dataset names). Hub access — uploads, cloud training, model downloads — is best-effort and bounded (every HF call gets our own timeout; HF's client has none). Dev happens behind the GFW: the Mac's network view is tunneled and unrepresentative; raw-LAN devices lose GitHub/HF outright. The eventual deployment target is outside the GFW, but every resilience mechanism stays (it's general robustness).

**If it changes** (always-online assumption creeps in): offline recording breaks in ways only a logged-out/blocked machine reveals — test that path on purpose.

## 5. The user is a MakerMods bench tester, not the public

Semi-technical teammates following the bench checklists; the web UI replaces LeRobot's CLI + keyboard flows so no terminal is needed for normal operation. Errors must be legible at the bench (outcome/error/hint taxonomy, actionable toasts). It is NOT hardened for untrusted users or exposed networks; the upstream HF Space is a separate runtime with its own build.

**If it changes** (external users, demos-at-events as a product): input validation, auth, and destructive-action guards (e.g. the Hub-repo-deleting trash can, #37) need a real pass.

## 6. The workflow is data-collection-first, episodic, table-top

Teleoperated demonstrations → episodic datasets (~30 Hz, minutes-long episodes, LeRobot format) → ACT locally / SmolVLA on HF Jobs → deploy the checkpoint back on the same rig. Recording quality is verified by humans (aim by eye, freeze-audit suspect sessions) — the recorder itself doesn't judge content. Dataset/checkpoint compatibility is rig-bound: camera role names, arm count, and state dims must match between record and inference (preflights enforce the checkable parts).

**If it changes** (long-horizon tasks, mobile base, non-episodic streams): episode semantics, return-to-rest, and the dataset tooling all assume the current shape.

## 7. One process, local state, pinned dependencies

A single uvicorn process owns all state (module globals + locks); persistence lives under `~/.cache/huggingface/lerobot/` via `utils/config.py`. `lerobot` is pinned to a specific SHA whose internals we've verified and depend on ([Assumptions.md](Assumptions.md) #9). Run from the repo's editable venv (`.venv/bin/makermodslab --dev`); the committed dist is trusted only on `main`.

**If it changes** (workers > 1, systemd multi-instance, pin bump): state corruption is silent; a pin bump re-opens every `[verified-in-source]` assumption — re-verify the list before trusting the bench.

## 8. Development co-exists with hardware use — carefully

Agents and editors work in the same tree the server runs from, so: no hardware sessions while files are being edited under `--dev` (reload kills recordings), commit finished work promptly (dirty-tree branch switches stash it silently), and bench results only count against a verified-fresh server + bundle (stale-server and stale-dist traps).

**If it changes** (dedicated test rig / CI hardware): most of the process rules in [Regression Checks.md](Regression%20Checks.md) §0 relax.

---

*Summary sentence: MakerMods Lab is a locally-served, human-supervised bench tool for one SO-101 rig on one Mac in a frozen USB topology, collecting episodic demonstrations offline-first, operated by teammates, built on a pinned LeRobot, with the Hub as an optional convenience — and every "why don't we just…" feature question should be checked against which of these eight premises it quietly changes.*
