# Split-host software validation record

Date: 2026-09-01

Scope: contributor-executed software evidence only. This record is not the
SO-101 secured-arm acceptance receipt and does not claim physical torque-off,
gravity behavior, device identity, real bus timing, or two physical laptops.

## Proven in the contribution checkout

| Evidence                        | Result                                                                                                                                                                                                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Full Python suite               | 2,436 tests passed on the final code tree, with 17 existing deprecation/serializer warnings                                                                                                                                                                                     |
| Full frontend unit suite        | 24 files, 156 tests passed                                                                                                                                                                                                                                                      |
| Frontend type checks            | `tsconfig.app.json` and `tsconfig.node.json` passed                                                                                                                                                                                                                             |
| Frontend production build       | Passed; bundle-size warning only                                                                                                                                                                                                                                                |
| Frontend lint                   | No changed-file error; the repository workflow still reports its four documented pre-existing errors and keeps lint non-blocking                                                                                                                                                |
| Python quality gates            | Ruff check/format, MyPy, Bandit, PyUpgrade, generated OpenAPI, Gitleaks, and repository hooks other than the separately-run Prettier hook passed                                                                                                                                |
| Formatting                      | Prettier 3.6.2 was run directly because the isolated hook environment could not download Node through the host certificate chain                                                                                                                                                |
| Dependency advisory gate        | Patched all reported high-severity frontend transitive advisories; `npm audit --audit-level=high` passes with zero high/critical findings. Two moderate React Router 6 notices remain documented below                                                                          |
| Two-instance protocol matrix    | Two clean subprocesses passed pinned TLS, one-time pairing, authenticated UDP, loss/reorder/duplicate, stale/future actions, clock drift, duplicate sessions, acknowledged STOP, browser/operator/control/network loss, and restart rejection                                   |
| Real runtime-service loss proof | Separate `RemoteRobotService` and `RemoteOperatorService` subprocesses passed abrupt operator loss with robot-local dispatch halt, stop/close, simulated torque evidence, and registry release                                                                                  |
| Open-handle identity boundary   | Linux derives the unique follower USB binding from descriptor device number plus sysfs ancestry; macOS uses descriptor device number plus a twice-verified IOKit serial/USB registry chain; swaps, duplicates, disconnects, failed close, and unsupported platforms fail closed |
| Default dormancy                | Tests prove application/configuration/restart do not open a listener or arm device                                                                                                                                                                                              |
| Central ownership               | Pairwise barrier races cover every registered arm feature kind; static checks require every live arm feature to use the central registry                                                                                                                                        |
| Secret/path hygiene             | Gitleaks and targeted contributor-path/private-key scans passed                                                                                                                                                                                                                 |

## PR gate status

| Gate                      | Contributor result                                                                                                                             |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| PR 0 — ownership/executor | Passed deterministic mutual-exclusion, startup unwind, unresolved-latch, executor, and full-regression gates                                   |
| PR 1 — follower adapter   | Passed fake lifecycle, static boundary, dormant-start, blocking-call, and fault-lockout gates; physical secured-arm adapter proof remains open |
| PR 2 — leader adapter     | Passed leader-only construction, identity/schema validation, raw action encoding, and cleanup gates                                            |
| PR 3 — control/auth/clock | Passed two-process pinned TLS, one-time pairing, credential, clock, heartbeat, STOP, and control-loss gates                                    |
| PR 4 — UDP/observability  | Passed spoof/size/rate/sequence/stale/future/loss/reorder/duplicate/watchdog and bounded-recording gates                                       |
| PR 5 — UI/configuration   | Passed typed API, secret-clearing, role/configuration, commissioning/recovery, status, STOP, and full-page UI tests                            |
| PR 6 — field package      | Clean install, software fault matrix, documentation, and rollback rehearsal passed; two supervised physical SO-101 sessions remain open        |

The one-command gate in
[`two-laptop-quickstart.md`](two-laptop-quickstart.md) was executed in a fresh,
isolated macOS arm64 copy. It installed Python 3.12.13, resolved 132 Python and
510 Node packages, built the production UI, and passed all three authenticated
two-process smoke tests in 84.40 seconds. Those tests cover the UDP
receiver/session startup race, the TLS/UDP fault matrix, and robot-local stop
after abrupt operator-process loss. The script opened only ephemeral loopback
test sockets, not a MakerMods application listener or arm. Dependency download
time is additional and network-dependent.

The lockfile update clears the reported `brace-expansion`, `js-yaml`, `nanoid`,
and `postcss` advisories. `npm audit` still reports two moderate React Router 6
advisories whose patched release is the breaking React Router 7 line. This
client-only Vite SPA does not use server-side hydration or
`deserializeErrors()`, and every `useNavigate()` target in the current source
is a source-controlled literal rather than remote or user input. The
single-PR gate therefore rejects high/critical advisories while recording the
Router 7 migration as separate dependency work.

The rollback procedure was rehearsed in that isolated copy: credentials and
runtime state were removed or preserved as documented, no listeners, devices,
threads, or leases remained, and calibration/firmware sentinels were
byte-identical before and after.

Two independent source-only adversarial reviews then examined the complete PR
packet. Confirmed findings were repaired and regression-tested: incomplete
follower STOP attempts no longer default to successful hardware completion;
commissioning and listener enablement fail before device/listener creation
without a stable unique USB binding; an active pairing window cannot be
silently replaced; pre-authentication control errors are generic; and the UI
uses the documented `7443`/`7444` defaults. The reviewers' proposed
credential-revocation race was not reproducible: session publication and
revocation share the credential lock, the protocol rechecks authorization
after the blocking open, and the blocked-open race regression passes. The full
gate above was rerun after these repairs.

The macOS robot-host implementation now loads and performs a read-only IOKit
inventory on Apple Silicon, and its injected fault tests cover descriptor
mapping, identity duplication, disappearance, device-number ambiguity, and
same-identity or changed-identity unplug/replug. No follower adapter was
attached during that software proof, so it does not replace the physical
secured-arm acceptance below. Linux and native Apple Silicon macOS are both
eligible robot hosts for that remaining trial.

## Still required for live acceptance

The seven-PR definition of done remains open until maintainers or a contributor
with the hardware completes and attaches a redacted
[`commissioning-worksheet.md`](commissioning-worksheet.md). That trial must use
two physical laptops, one SO-101 leader, and one physically secured SO-101
follower, then prove:

1. exact local leader/follower calibration and device identities;
2. first motion within the minimum envelope and configured limits;
3. local stop on action, control, browser, operator-process, and network loss;
4. independent Feetech torque-off readback and truthful fault lockout;
5. a full two-process restart followed by a second critical session; and
6. maintainer reproduction without contributor machines, credentials, paths,
   or network access.

Until that receipt exists, describe the contribution as software-complete and
hardware-acceptance-pending, not as a completed live SO-101 rollout.
