# Frontend map — where things live (for `/ui-iterate`)

Anchors are function names, component names or marker comments — grep for them;
line numbers drift and are deliberately absent. Update this file at the end of
any batch that moves something. Untracked (`internal_docs/` is gitignored).

**Lint baseline** (`npm run lint`, from `frontend/`): 38 problems (4 errors,
34 warnings), all pre-existing, measured 2026-09-03 on `feat/drtc-session`
and re-measured unchanged after S3.9. Re-measure only if this looks stale.
Both tsc projects clean; vitest 30 files / 335 tests.

## Studio → Deploy panel (`components/studio/DeployPanel.tsx`, ~2500 lines)

Top of file: constants `MAX_EVAL_EPISODES`, `MAX_COACHING_CORRECTIONS`,
`DEFAULT_TEMPORAL_ENSEMBLE_COEFF`, `FIELD_TRIGGER`, `UNBOUNDED_DURATION_S`,
`TRANSPORT_REPROBE_MS`; `cameraMappings()` (BiSO `left_`/`right_` prefix
round-trip); `OPERATOR_TABS` (single | coach).

Inside `DeployPanel` (S3.9 — TWO AXES):
- Axis state: `runsOn` ("local" | "remote") + `operatorMode` ("single" |
  "coach") + `scoring` (prefill-only eval). Derived: `remote`,
  `effectiveOperator` (coach collapses to single while remote), `coaching`,
  and `runMode: DeployRunMode` — still what the guards and the launch speak.

### 2026-09-04 UI batch (4fa7f09a on feat/drtc-session)
- Precision · GPU · Flow steps selects live on `GpuLaunchSection` (the
  "Policy server on Modal" card), one `grid gap-2 sm:grid-cols-2` row under
  Modal profile / Environment, reading `effectiveGpuKnobs` + `knobSupport`
  (S3.8f's disabled-with-reason kept). `RemoteAdvancedSection` = transport
  knobs only (horizon/s_min/fps/codec); no `knobs` prop, no `{{extra}}`.
- `CameraRoleBindings` renders INSIDE the cameras block, under
  `SessionCameraList`, above the unmatched-camera `Alert` (A8 is gone from
  its old slot after the task field). S3.8g's "Add a camera role" unchanged.
- ACT: task block gated `{!isAct ? … : null}`; engine slot is
  `{!isAct ? <engine> : !remote ? <temporal section> : null}`; the
  panel-foot `AdvancedSection` + `advancedOpen` state are gone.
- Hub policy id: field only (hint paragraph removed).
- Dropped keys: remoteInference.form.{transportGroupHint,gpuGroupHint,
  precisionHint,flowStepsHint,hubIdHint,hubIdInherited},
  remoteInference.cameraRoles.{hint,identityNote},
  studio.deploy.tabs.coachNeedsLocal, studio.deploy.advanced.summary.
  `remoteInference.form.gpuHint` is dead-but-kept (never rendered).
- Remote wiring: `useRemoteInferenceStatus(open)`,
  `useRemoteInferenceTransport(open && remote)`, `useGpuLauncher(open &&
  remote)` + `useGpuTargets(...)` (the GPU is owned HERE, not in the dialog),
  `remoteActive`, `runActive`; two re-probe effects (GPU state/phase
  transitions; a 15 s timer while idle).
- Engine: ONE `engine` = `remoteConfig.engine`, with `setEngine` re-seeding the
  horizon; ONE fact — `rtcSupported` = `policySupportsRtc(policyConfig)` —
  disables the rtc option, drives the fallback effect, the single hint
  (`studio.deploy.engine.rtcUnavailable`) and `remoteEngineSupported`. Fail-
  closed on BOTH paths: `rtcAvailable` and the remote-only warn paragraph are
  gone (SLICE3 "Follow-ups the same evening").
- Cameras: `cameraMap` → `cameraBindings` (name match) → `boundCameraBindings`
  (role picks layered over, EVERY mode) → `unmatchedCameras` /
  `disconnectedCameras` / `mismatchedCameras` / `allCamerasReady`.
- Guards: `canStartAnyMode`, `blockedReasonKey(mode)` → `deployBlockedReason`
  (pure, `deployGuards.ts`, key-returning; `durationValid` is S3.9's).
- Launch: `handleStart(mode)` — forks on `mode === "remote"` (session kind
  `remote_inference`) else `inference` (+ coaching options); BOTH now call
  `openInferenceSession`, remote with kind `"remote_inference"`.
- JSX order (mockup codes in the comments): `PanelHeader` → `Collapsible`
  (never forced open any more) → `PanelEntryControl` → intro → `RobotStatus` →
  policy picker + `CheckpointDropdown` → arm-mismatch `Alert` → A6 Hub policy
  id (remote only) → A7 task (marker: "Run parameters — flat") → A8
  `CameraRoleBindings` → A9 engine → A10 duration → A5 "Runs on" segmented
  control → R6 `GpuLaunchSection` + R7 `RemoteManualSection` + A13
  `RemoteAdvancedSection` (remote only) → A11 `Tabs` (Run / Human in the loop)
  + A12 what/commitment + the coach-blocked warn → per-tab content →
  A14 `SessionCameraList` + A15 camera `Alert` → ACT `AdvancedSection` →
  `MilestoneReveal` → A16 Start + the blocked / transport-summary line →
  `LibrarySection` / `ModelsLibrary` → `PolicyExtraDialog`.

## Remote inference (`components/remote-inference/`)

No mount point any more — the panel owns the ORDER and renders these directly
(S3.9 deleted `RemoteInferenceBlock`, `RemoteRunFields`, `TransportSection` and
`RemoteInferenceStatusPanel`).

- `RemoteManualSection.tsx` (R7) — "Run it yourself instead": `ModalRunLine`
  plus the crib-sheet rows (GPU address, room, key id, key file) that survived
  the Transport section.
- `RemoteAdvancedSection.tsx` (A13) — the transport knobs under
  `AdvancedSection`, summary line "Transport: horizon … · … fps · … [·
  minimum budget …]" (`remoteInference.form.advancedSummary{,Rtc}`).
- `RemoteSessionBody.tsx` — the remote run INSIDE the session dialog: local
  `StatusPill` / `SessionTimer` / `PhaseLine` over `inference/sessionFrame.ts`,
  the D4 GPU card (holds RATE not counter, lead-vs-margin bar, `Sparkline`
  under e2e p50 / rtt / holds), outcome card, `LogPanel` holding `log_path`.
- `transportSummary.ts` — pure `summarizeTransport()`, the sentence under Start.
  `SFU_OFF_SUMMARY_KEY` marks the one case (SFU off AND nothing else
  configured) under which the panel also prints the start command + the
  backend's install hint.
- `GpuLaunchSection.tsx` — Start/Stop GPU, phases, idle-stop countdown,
  "this is billing". S3.8b adds profile/environment selects. THE ONLY GPU
  control anywhere: the dialog reads the launcher for one billing line.
- `remoteRunConfig.ts` — `RemoteRunConfig` (no `durationS` since S3.9),
  `DEFAULT_HORIZON`, `DEFAULT_S_MIN`, `FLOW_POLICY_TYPES` (fallback only),
  `policySupportsRtc(policyConfig)`, `defaultEngineForPolicy`,
  `horizonForEngine`, `armSupportsRemoteInference` (the ONE place the remote
  arm guard lives — S3.10 touches it).
- `modalCommand.ts` — `MODAL_WRAPPERS`, the generated `modal run` line.
- Hooks: `hooks/useRemoteInferenceStatus.ts` (1 Hz poll while live,
  `perSecondRate`), `useRemoteInferenceTransport.ts` (`transportIsReady`),
  `useGpuLauncher.ts`, `useRemoteCameraRoles.ts` (S3.7b localStorage picks).

## Shared primitives

- `components/ui/` — shadcn: button, input, label, number-input, select,
  switch, collapsible, alert, dialog, popover, tooltip, badge, tabs (S3.9,
  over `@radix-ui/react-tabs`).
- `components/studio/panel/primitives.tsx` — `PanelHeader`,
  `PanelEntryControl`, `AdvancedSection` (title/summary/open), `RobotStatus`,
  `LibrarySection`, `SLIDE`, `useEyebrowClass`.
- `components/recording/CameraConfiguration.tsx` — `SessionCameraList`.
- Contexts: `StudioContext` (`deployPrefill`), `InferenceSessionContext`
  (`openInferenceSession(id, lineage?, kind?)`, `sessionOpen`; the `kind`
  picks `InferenceSessionDialog` vs `RemoteSessionDialog`), `ApiContext`.

## Session dialogs (`components/inference/`)

- `sessionFrame.ts` — `PHASE_DOT` / `PHASE_TEXT` / `PILL_BG` / `formatTime`,
  shared by both dialogs so they cannot drift cosmetically. Values only.
- `InferenceSessionDialog.tsx` (~2320 lines) — the LOCAL run: plain, scored
  and coached. Untouched by S3.9 except the import above.
- `RemoteSessionDialog.tsx` — the remote run's shell: Dialog + heartbeat +
  unload warning + stop (with a `getCurrentSession` fallback), rendering
  `RemoteSessionBody`. A sibling rather than a branch — the local dialog's
  hooks are all about a local rollout, and its lease is gated on
  `inference_active`.

## i18n

`i18n/locales/{en,zh-CN}/` — one file per namespace: `studio.ts`
(`studio.deploy.*`: runMode, runsOn, tabs, coaching, engine, duration,
cameras, advanced, blocked, actions, toast, milestone), `remoteInference.ts`
(form incl. `advancedSummary*`, engine hints, cameraRoles, modalRun, gpu,
transport — now only the crib-sheet labels, `source.*` and `summary.*` —
phase, outcome, status incl. the dialog's own copy, toast), `shared.ts`.
Rules: `frontend/docs/localization.md`.

## Shelved

`internal_docs/drtc/s3.9-ui-tabs.patch` — HISTORY as of S3.9. The shipped
design (two axes, no inline status panel, the run in the session dialog)
diverges from it; the mockups in session 984594ee's scratchpad are the record.

## S3.8e (2026-09-04, a943c061) — GPU launch knobs
- `RemoteAdvancedSection.tsx` (A13) also carries two GPU-side selects, Precision
  (`model_dtype`) and GPU (`gpu`), in their own divided group below the
  transport knobs; both ride the Advanced summary line's `{{extra}}` suffix.
- `useGpuLauncher.ts` exports a third hook, `useGpuKnobs()` (localStorage
  `makermodslab.gpuModelDtype` / `.gpuType`; `MODEL_DTYPES`, `GPU_TYPES`,
  `DEFAULT_GPU = "A100"`). Start body + drift warning + `modalCommand.ts`
  (`--model-dtype X`, `DRTC_GPU=X ` prefix) all read from it.

## S3.8f/S3.8g (2026-09-04, efa598b0) — Flow steps, knob guards, extra camera roles
- `RemoteAdvancedSection.tsx` (A13): Flow steps select beside Precision; both disable
  with a reason via `gpuKnobSupport` (from the policy-config `supports_*` fields).
- `CameraRoleBindings.tsx` (A8): "Add a camera role" / remove per extra role, gated
  on `supports_extra_image_roles` && remote; `useRemoteCameraRoles` storage is now
  `{roles, extra}` (reads the old bare map).
- `useGpuLauncher.ts`: `FLOW_STEPS`, `useGpuKnobs().flowSteps`, `gpuKnobSupport`,
  `effectiveGpuKnobs`. Start body carries `flow_steps` + `extra_image_roles`.
- NOTE: the other session's stashed UI batch (stash@{0}) moves Precision/GPU into
  `GpuLaunchSection.tsx`; Flow steps should follow them there when it lands.
