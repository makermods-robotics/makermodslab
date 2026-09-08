# Model Training Interface Pathways

**Audit date:** July 14, 2026  
**Audited commit:** `518ca56`

This document maps MakerMods Lab's implemented model-training journeys from entry and configuration through execution, persistence, monitoring, continuation, publication, model discovery, and handoff to inference. It describes current behavior neutrally and is based on the frontend, backend, and focused tests at the audited commit.

## System overview

Model training is supported by two related systems:

- `JobRegistry` tracks training runs, imported-model pointers, logs, metrics, checkpoints, runner state, and resume lineage.
- The model browser aggregates completed local runs, copied or downloaded checkpoints, pinned Hub repositories, and discovered Hub models for selection and inference.

The broad journey is:

```text
Selected dataset
  -> Create model / choose policy
  -> Configure local or HF Cloud target
  -> Environment and dataset preflight
  -> POST /jobs/training
  -> Persist JobRecord
  -> Local subprocess OR HF Jobs wrapper
  -> Logs, metrics, and checkpoints
  -> Done / failed / interrupted
  -> Inference, continue/resume, fine-tune, export, upload, or delete
```

## 1. Entering training

### Fresh training

The ordinary fresh-training journey starts from the landing-page Models panel:

1. The user opens **Add model**.
2. They choose **Train a model**.
3. MakerMods Lab opens the policy grid at `/create-model`.
4. A dataset must already be selected on the home page; otherwise all policy buttons are disabled.
5. Selecting a policy navigates to `/training` with the policy type in router state.

ACT and SmolVLA are labelled as tested in MakerMods Lab. Other selectable policies are explicitly labelled untested. Backend policy availability is derived by attempting to construct each policy using the installed, pinned LeRobot version; unavailable policy types are disabled once that answer arrives.

Source evidence:

- `frontend/src/components/landing/ModelsPanel.tsx:229-260`
- `frontend/src/pages/CreateModel.tsx:13-65`
- `frontend/src/pages/CreateModel.tsx:66-137`
- `frontend/src/components/training/types.ts:67-95`
- `makermodslab/server.py:414-501`

### Other training entry points

The training form can also be opened by:

- **Continue** on a local training checkpoint.
- **Resume** on an unfinished cloud run with a saved checkpoint.
- **Fine-tune** on an imported model.
- Direct navigation to `/training`.

Router state carries resume or fine-tune provenance. Because router state disappears on a hard refresh, the page stores the last resolved policy type in `sessionStorage`; direct visits fall back to that value and then to ACT.

Source evidence:

- `frontend/src/pages/Training.tsx:60-101`
- `frontend/src/pages/Training.tsx:173-254`
- `frontend/src/components/jobs/JobCard.tsx:303-380`

## 2. Configuration journey

The training page is divided into compute target, essentials, and advanced settings.

### Compute target

The user chooses either:

- **Local -- your machine**
- **Hugging Face Cloud**

For a local run, the device options are:

- **Automatic**, allowing LeRobot/PyTorch to use CUDA or MPS when available.
- **CPU**, explicitly forcing CPU.

For a cloud run, the UI loads the available HF Jobs hardware flavors, including accelerator/CPU description and hourly price. The user must select a flavor. A cloud-only timeout field accepts friendly duration forms such as `2h`, `45m`, and `3h30m`; leaving it blank uses the backend's two-hour default. The UI also presents a computed suggestion based on policy, steps, flavor, and dataset size, but does not apply it automatically.

Source evidence:

- `frontend/src/components/training/config/TargetCard.tsx:28-200`
- `frontend/src/lib/jobTimeout.ts`
- `makermodslab/server.py:1680-1718`
- `makermodslab/train.py:24-67`

### Essentials

The visible essentials are:

- The dataset selected on the home page.
- Optional run display name.
- Policy type.
- Total training steps.
- Batch size.
- Weights & Biases enablement and settings.

Although a policy is normally chosen before reaching the form, the policy remains editable in the training form.

W&B is supported on BOTH runners. Where the trainer executes changes only how
the API key reaches it — forwarded into the pod as an HF Jobs secret for a cloud
job, inherited through `os.environ` (or read from `~/.netrc` by wandb itself) for
a local subprocess — so the form does not gate the group on the runner.

What IS enforced on both runners is that a key resolves at all before the trainer
starts; see the credential preflight below.

The group lives in the **Run** pane (`config/RunPane.tsx`), behind the same
Advanced disclosure `OptimizerPane` uses — W&B is part of how loudly a run
reports, the same question `log_freq` answers. When W&B is enabled, the form
exposes:

- Project name.
- Optional entity.
- Optional notes.
- Mode: online, offline, or disabled.
- Disable-artifacts toggle (defaults to ON, i.e. artifacts off — per-checkpoint
  model uploads to W&B are opt-in).

On a RESUME the whole group renders read-only and shows the PARENT run's values
(carried by `buildResumeSeed`). It is not a display choice: lerobot resumes with
`wandb.init(resume="must")` using the run id stored in the checkpoint's
`train_config.json` (`lerobot/common/wandb_utils.py:90-113`), so a continuation
always re-opens the parent's W&B run. Enabling W&B on a resume of a non-W&B
parent is structurally impossible, and `JobRegistry.start` copies
`wandb_enable` / `wandb_project` / `wandb_entity` off the parent record rather
than trusting the request.

Source evidence:

- `frontend/src/components/training/config/EssentialsCard.tsx:18-267`

### Advanced configuration

The advanced panel exposes:

- Automatic mixed precision.
- Random seed.
- Data-loader worker count.
- Optimizer type.
- Optional learning rate override.
- Optional weight-decay override.
- Optional gradient-clipping override.
- Log frequency.
- Checkpoint save frequency.
- Save-checkpoints toggle.
- Resume toggle.
- Use-policy-training-preset toggle.

Optimizer placeholders are populated from the selected policy's actual LeRobot optimizer preset rather than from one generic hardcoded default. User-supplied optimizer fields are emitted as explicit CLI overrides.

If `log_freq` or `save_freq` exceeds the total step count, the frontend shows a warning explaining that the corresponding event will never occur. The backend repeats this as a log warning but does not block the run.

Source evidence:

- `frontend/src/components/training/config/AdvancedCard.tsx:47-337`
- `makermodslab/server.py:454-501`
- `makermodslab/server.py:1082-1098`

### Fresh-run defaults

A new training form starts with:

| Field | Default |
|---|---:|
| Target | Local |
| Policy | Router/session value, then ACT |
| Steps | 10,000 |
| Batch size | 8 |
| Seed | 1,000 |
| Workers | 4 |
| Log frequency | 50 |
| Save frequency | 1,000 |
| Save checkpoints | Yes |
| W&B | Disabled |
| Device | Automatic |
| AMP | Disabled |
| Optimizer | Adam |
| Policy training preset | Enabled |

Source evidence: `frontend/src/pages/Training.tsx:198-240`.

### Backend-only configuration fields

`TrainingRequest` supports fields that the normal frontend does not expose, including:

- Dataset revision.
- Dataset root.
- Dataset episode subset.
- Evaluation frequency and evaluation parameters.
- Environment type and task.
- Direct `config_path`.
- Direct `policy_pretrained_path`.

These remain available to API callers and persisted historical records.

Source evidence: `makermodslab/train.py:104-217`.

## 3. Environment and launch preflight

### Training package gate

Before the form is shown, the frontend checks `/system/training-extra` for `accelerate`. If it is missing, MakerMods Lab shows an installation gate instead of the form. This UI gate occurs before local/cloud selection, so it currently participates in both journeys.

Package installation prefers:

```text
uv pip install --python <MakerMods Lab interpreter> <package>
```

and falls back to:

```text
<MakerMods Lab interpreter> -m pip install <package>
```

Installation runs in the background, streams logs through a pollable status endpoint, and invalidates Python import caches after success so the new package can be detected without restarting MakerMods Lab.

Source evidence:

- `frontend/src/pages/Training.tsx:255-285`
- `frontend/src/pages/Training.tsx:472-497`
- `frontend/src/components/training/TrainingExtraGate.tsx`
- `makermodslab/utils/system.py:55-234`
- `makermodslab/server.py:1732-1765`

### W&B credential preflight

There is no W&B *package* gate. The old `/system/wandb-extra` install flow was dead code: `wandb` is a hard transitive dependency of the pinned lerobot's `training` extra, so the probe always answered yes, while the thing that actually blocks a run — whether a credential exists — went unasked until the job had been submitted.

`GET /system/wandb-credentials` reports whether a W&B API key is resolvable on the backend's host (`WANDB_API_KEY`, then a `~/.netrc` entry for `api.wandb.ai`). It returns a boolean and a login hint, never the key. It answers for BOTH runners: a cloud job needs the key forwarded as a secret, and a local trainer is a non-tty subprocess in which `wandb.init` cannot prompt for a login, so a missing key is a launch-time refusal either way rather than a run that dies once the record already says `running`. `useTrainingEnvironment` polls it once; when W&B is on and the answer is an explicit `false`, `TrainingConfigurator` disables Start through the ordinary `PaneIssue` / `startDisabled` / `startTooltip` machinery, marks the Run segment, and opens the W&B disclosure so the named control is visible. A probe that never answered leaves the state `null` and blocks nothing — the backend preflight is the belt-and-braces catch.

Source evidence:

- `frontend/src/components/training/config/RunPane.tsx`
- `frontend/src/hooks/useTrainingEnvironment.ts`
- `frontend/src/components/training/TrainingConfigurator.tsx`
- `makermodslab/runners/hf_cloud.py` (`resolve_wandb_api_key`, `handle_get_wandb_credentials`)

### Local policy extras

Immediately before a local launch, the frontend asks `/system/policy-extra/{policy_type}` whether that policy needs an optional package and whether it is currently importable.

The current mappings are:

| Policy | Probe | Install target |
|---|---|---|
| SmolVLA | `transformers` | `lerobot[smolvla]` |
| PI0 | `transformers` | `lerobot[pi]` |
| PI0 Fast | `transformers` | `lerobot[pi]` |
| Diffusion | `diffusers` | `lerobot[diffusion]` |

Cloud jobs skip this host-side policy-extra check because the remote container assembles its own policy dependencies.

Source evidence:

- `frontend/src/pages/Training.tsx:410-445`
- `frontend/src/components/training/PolicyExtraDialog.tsx`
- `makermodslab/utils/system.py:238-313`

### Dataset handoff preflight

For local training:

- The dataset repo ID must be present.
- If HF offline mode is enabled and the dataset is not available locally, the backend rejects the run before spawning the trainer.
- Otherwise LeRobot may use the local cache or resolve the repo from the Hub.

For cloud training:

- A dataset known to the frontend as local-only is uploaded first.
- The upload completes before the job is submitted; upload failure launches no job.
- The current frontend upload request uses MakerMods Lab's public-upload default.
- A Hub or `both` dataset goes directly to job launch.
- A dataset absent from the current listing is left for backend/runner validation.
- The registry rejects a definitive `local_only` Hub-status result for direct/non-UI cloud calls.
- The cloud runner contains a second synchronous ensure-on-Hub path before submission.

Source evidence:

- `frontend/src/pages/Training.tsx:323-466`
- `makermodslab/server.py:1115-1146`
- `makermodslab/jobs.py:1108-1120`
- `makermodslab/runners/hf_cloud.py:641-676`

### Runner and resume guards

The frontend and backend also enforce:

- Only one local training job at a time.
- Multiple cloud jobs may run concurrently.
- Cloud target requires authentication and a selected flavor.
- Offline mode blocks a cloud run when its local-only dataset would require upload.
- Resume total steps must be strictly greater than the selected checkpoint step.
- Resume and fine-tune cannot both be requested.
- A bare resume toggle without a source is rejected unless an API caller supplies a valid `config_path`.

Source evidence:

- `frontend/src/pages/Training.tsx:287-317`
- `frontend/src/pages/Training.tsx:503-535`
- `makermodslab/server.py:1099-1114`
- `makermodslab/jobs.py:1125-1190`

## 4. Request creation and job persistence

The frontend converts its form state into a `TrainingRequest`, separates the optional `target`, and calls `POST /jobs/training` using:

```json
{
  "config": {},
  "target": {
    "runner": "local",
    "flavor": null
  }
}
```

The endpoint also retains compatibility with the older shape in which training fields appear directly at the top level.

Source evidence:

- `frontend/src/pages/Training.tsx:130-164`
- `frontend/src/lib/jobsApi.ts:120-143`
- `makermodslab/server.py:208-229`

### Job identity and on-disk layout

The registry generates an ID from policy type, dataset slug, and timestamp. Same-second collisions receive `-2`, `-3`, and later suffixes.

The default job layout is:

```text
~/.cache/huggingface/lerobot/outputs/train/<job-id>/
|-- job.json
|-- log.jsonl
`-- run/
    `-- checkpoints/...
```

`MAKERMODSLAB_OUTPUT_ROOT` can override the root.

The registry creates and atomically persists a running `JobRecord` before it launches the runner. The record includes:

- Immutable job ID and output directory.
- Original and optional display names.
- Full training configuration.
- State and timestamps.
- Exit code and error message.
- Current metrics.
- Runner type.
- Local PID or HF Job identifiers.
- HF hardware flavor.
- Hub model repo and job URL.
- Captured W&B run URL (both runners scrape it; `null` is the ordinary value).
- Derived checkpoint count.

Source evidence:

- `makermodslab/jobs.py:73-103`
- `makermodslab/jobs.py:858-865`
- `makermodslab/jobs.py:1100-1243`
- `makermodslab/jobs.py:1774-1797`

## 5. Local training runner

Local training uses the same Python executable as the running MakerMods Lab process:

```text
<sys.executable> -m lerobot.scripts.lerobot_train ...
```

This prevents PATH lookup from selecting an unrelated Python environment.

The subprocess is launched with:

- Stdout and stderr combined.
- Line buffering.
- `PYTHONUNBUFFERED=1`.
- A new process session, allowing it to survive a development-server reload.

Each nonblank output line is:

- Parsed into current metrics.
- Written as JSON to `log.jsonl`.
- Added to an in-memory queue for live consumers.

The queue is capped at 1,000 lines; the persistent file retains full history.

Source evidence: `makermodslab/jobs.py:284-426`.

### Local CLI assembly

A fresh local command may include:

- Dataset repo ID, revision, root, and episode subset.
- Policy type and optional pretrained path.
- Steps, batch size, workers, and seed.
- Concrete CUDA, MPS, or CPU device.
- AMP setting.
- Hub push explicitly disabled.
- Log, save, and evaluation frequencies.
- Checkpoint toggle.
- Output directory.
- W&B settings.
- Evaluation environment settings.
- Optimizer overrides.
- Policy-training-preset setting.

`auto` device is resolved to CUDA, then MPS, then CPU. Explicit `cuda`, `mps`, and `cpu` pass through.

Source evidence:

- `makermodslab/train.py:83-101`
- `makermodslab/train.py:222-365`

## 6. Hugging Face Cloud runner

Cloud launch follows this sequence:

1. Resolve the locally stored HF token.
2. Resolve the authenticated username.
3. Remove or reject host-only paths.
4. Pin the device to the selected flavor's backend.
5. Ensure the dataset exists on the Hub.
6. Enable policy push and assign the output model repo.
7. Build a wrapped trainer command.
8. Forward HF and, when W&B is enabled, W&B credentials as secrets.
9. Submit `HfApi.run_job` with image, command, flavor, and timeout.
10. Start independent log-tail and status-poll threads.

Fresh cloud jobs use:

```text
<username>/<unique-job-id>
```

as their output model repository. A cloud resume keeps using its parent run's existing model repository.

Source evidence:

- `makermodslab/runners/hf_cloud.py:459-597`
- `makermodslab/runners/hf_cloud.py:161-190`

### Cloud package assembly

The base image is:

```text
huggingface/lerobot-gpu:latest
```

Before training, the wrapper replaces the image's LeRobot installation with MakerMods Lab's exact pin. The pin is read from installed MakerMods Lab metadata, with `pyproject.toml` as a source-tree fallback.

At the audited commit, MakerMods Lab pins:

```text
lerobot[core_scripts,feetech,training] @
git+https://github.com/huggingface/lerobot.git@82dffde7fad11cba91f7916b050fbe7d7eea35ab
```

For the cloud container:

- GitHub Git pins are converted to equivalent source-archive URLs.
- Host-only `feetech` is removed.
- The selected policy's LeRobot extra is added when needed.
- Installation prefers `uv`, falls back to pip, and finally tries `ensurepip` plus pip.

Source evidence:

- `pyproject.toml:13-22`
- `makermodslab/runners/hf_cloud.py:41-133`
- `makermodslab/runners/hf_cloud.py:136-158`

### Cloud wrapper behavior

The in-container wrapper:

1. Installs the pinned LeRobot requirement.
2. Creates the output model repo early.
3. Optionally reconstructs a cloud-resume checkpoint.
4. Launches the LeRobot trainer as an argv list.
5. Runs a checkpoint watcher every 15 seconds.
6. Uploads new checkpoint trees to `checkpoints/<step>/`.
7. Performs a final upload scan after the trainer exits.
8. Returns the trainer's exit code.

Cloud checkpoints include both `pretrained_model/` and `training_state/`, enabling later true resume when those files are present.

Source evidence: `makermodslab/runners/hf_cloud.py:217-390`.

### Cloud publication and credentials

The generated cloud training command enables policy push, supplies the model repo ID, explicitly requests a public model, and passes MakerMods Lab's required Hub tags. LeRobot contributes its own model-card metadata during final push.

`HF_TOKEN` is sent through HF Jobs `secrets`. If W&B is enabled, MakerMods Lab resolves `WANDB_API_KEY` first from the environment and then from `~/.netrc`, and forwards it on the same channel.

A missing key is a hard 400 at SUBMIT time, raised before anything with a cost happens (MT40). Three checks, innermost last:

1. `POST /jobs/training` refuses a fresh run on EITHER runner whose request has `wandb_enable` set and no resolvable key — the fast half, skipped for resumes because a resume's W&B state is not the request's to state.
2. `JobRegistry.start` re-asks — runner-blind — once the resume block has copied the parent's `wandb_enable` onto the config, and before it creates a record, spawns a local subprocess, pushes a dataset, or spawns the F7 local→cloud upload thread. That thread is why the check cannot live any deeper: it returns 201 first and uploads on a thread, so a later refusal would surface as a failed job instead of a message on the button. This check is the authority.
3. `HfCloudJobRunner.start` repeats it belt-and-braces, above `_ensure_dataset_on_hub` so a missing key can never leave a freshly published dataset behind for a job that never ran.

Source evidence:

- `makermodslab/train.py:69-81`
- `makermodslab/train.py:295-310`
- `makermodslab/runners/hf_cloud.py:427-456`
- `makermodslab/runners/hf_cloud.py:565-590`

## 7. Monitoring, logs, metrics, and checkpoints

### Dedicated monitoring page

After job creation, the frontend navigates to `/training/<job-id>`.

The page:

- Seeds its log panel from the persistent log file.
- Polls job state and new live logs every second while running.
- Polls checkpoints every five seconds while running.
- Caps displayed log history at 5,000 lines.
- Shows progress, ETA, loss, and learning-rate charts.
- Links to W&B when a run URL is detected.
- Links completed cloud runs to their Hub model repo.
- Offers inference from any visible checkpoint.

Source evidence: `frontend/src/pages/Training.tsx:656-967`.

### Metric extraction

The backend reads two complementary trainer output formats:

- tqdm progress lines provide current step, total steps, and ETA.
- LeRobot's `step: ... loss: ... lr: ... grdn: ...` lines provide loss, learning rate, and gradient norm at `log_freq` cadence.

Resumed local runs rebase tqdm's remaining-window count onto the global target step. Log lines already carry global steps.

Source evidence: `makermodslab/jobs.py:155-235`.

### Persistent metric history

`GET /jobs/{id}/metrics-history` reparses `log.jsonl`. For resumed runs it follows `resume_from_job_id` through the ancestry chain, reads oldest-to-newest, deduplicates points by step, and returns a continuous series.

The frontend seeds charts from this history and then appends live points, keeping charts useful across navigation, page reloads, and MakerMods Lab restarts.

Source evidence:

- `makermodslab/jobs.py:238-281`
- `makermodslab/jobs.py:1431-1463`
- `frontend/src/components/training/monitoring/MonitoringStats.tsx:42-132`

### Jobs-dashboard updates

The jobs dashboard uses the shared `/ws/joint-data` socket for:

- `jobs_changed`: refetch local and Hub listings after registry changes.
- `job_progress`: apply compact running-job snapshots without repeatedly refetching the full listing.

The dashboard also refreshes when the browser tab becomes visible or regains focus, covering state changes originating elsewhere.

Source evidence:

- `frontend/src/hooks/useJobsChangedSignal.ts`
- `frontend/src/components/jobs/JobsSection.tsx:67-136`
- `makermodslab/jobs.py:976-985`
- `makermodslab/jobs.py:1716-1769`

### Checkpoint discovery

MakerMods Lab recognizes:

- Local training trees: `checkpoints/<numeric-step>/pretrained_model/config.json`.
- Hub training trees with the same shape.
- Flat imported model dirs containing root `config.json`, represented as step 0 and labelled latest in the UI.
- Imported training-output trees.

Checkpoint lists are sorted by numeric step. Hub checkpoint results are cached for 30 seconds.

Source evidence:

- `makermodslab/jobs.py:546-575`
- `makermodslab/jobs.py:736-800`
- `makermodslab/jobs.py:1465-1503`
- `frontend/src/components/jobs/CheckpointDropdown.tsx`

## 8. Stop, completion, failure, and restart recovery

### Watchdog finalization

The registry watchdog runs approximately once per second.

While a runner is live, it:

- Captures the first W&B URL found in output. All three real runners scrape it — `LocalJobRunner` from the subprocess's stdout, `TailingJobRunner` by replaying the log from offset 0 after a `--reload` (the URL is printed once, near the start, so tailing from EOF would lose it), and `HfCloudJobRunner` from the SSE stream. Only `PreparingJobRunner` returns a constant `None`. The URL comes from lerobot's own `Track this run --> <url>` log line, not from wandb's banner — lerobot sets `WANDB_SILENT=True` — and the pattern tolerates the ANSI colour wrapper `termcolor` adds. No URL is the ordinary outcome and never an error.
- Persists metric snapshots at most once per second.
- Broadcasts progress and checkpoint counts.

When a runner stops, it records:

- `done` for return code 0.
- `failed` for nonzero return code.
- End time.
- Exit code.
- A runner-supplied terminal message when available, otherwise a synthetic exit-code message.

Source evidence: `makermodslab/jobs.py:1702-1769`.

### Stop behavior

For local jobs, MakerMods Lab terminates the subprocess, waits up to ten seconds, and then kills it if necessary.

For cloud jobs, MakerMods Lab calls the HF Jobs cancellation API.

`JobRegistry.stop` then waits briefly for the watchdog to publish the terminal state. The dedicated monitoring page exposes Stop for running jobs. A tracked running cloud card with an HF Job URL steers the user toward its Hub job page, while the monitoring page still has the API-backed Stop action.

Source evidence:

- `makermodslab/jobs.py:357-371`
- `makermodslab/jobs.py:1379-1395`
- `makermodslab/runners/hf_cloud.py:755-773`
- `frontend/src/pages/Training.tsx:786-800`
- `frontend/src/components/jobs/JobCard.tsx:448-487`

### Restart recovery

On registry initialization, MakerMods Lab:

1. Migrates eligible legacy `<cwd>/outputs/train/` job dirs to the cache-root layout.
2. Loads valid `job.json` files.
3. Reattaches to a running local PID with a `TailingJobRunner` when that PID remains alive.
4. Marks a missing local process as `interrupted`.
5. Reattaches running cloud records using their HF Job IDs and flavors.
6. Marks malformed running records as `interrupted`.

The local tailing runner replays existing persisted logs into its metrics parser and continues following new lines. The cloud runner reopens the local log file and reconnects both remote log and status monitoring.

Source evidence:

- `makermodslab/jobs.py:991-1051`
- `makermodslab/jobs.py:429-543`
- `makermodslab/jobs.py:1565-1624`
- `makermodslab/runners/hf_cloud.py:599-621`

### Recoverable failed runs

A failed or interrupted run can remain useful when it saved checkpoints:

- Its checkpoints remain selectable for inference.
- A local checkpoint may be continued if resume state is present.
- An unfinished cloud run may be resumed if the Hub checkpoint contains training state.
- A failed local job whose policy extra is still absent exposes an install action on its card.

Jobs that are running or have runnable checkpoints are treated as active in the dashboard. Other local/cloud remnants are grouped under **Untracked**.

Source evidence:

- `frontend/src/components/jobs/JobsSection.tsx:35-39`
- `frontend/src/components/jobs/JobsSection.tsx:391-426`
- `frontend/src/components/jobs/JobCard.tsx:213-257`

## 9. Continue and resume journeys

### Local Continue

The local Continue journey is:

1. Select a checkpoint on a local job card.
2. Click **Continue**.
3. Navigate to `/training` with source job ID, checkpoint step, dataset, policy, target steps, and log/save cadence.
4. Prefill the new total-step target to twice the source run's configured total.
5. Submit a new job with `resume=true` and the source identifiers.
6. Resolve the source checkpoint's `train_config.json`.
7. Verify that `training_state/` exists.
8. Launch a new local job with a new output directory.

LeRobot reconstructs policy, dataset, optimizer, batch size, and other inherited settings from `train_config.json`. MakerMods Lab's resume command passes only the resume essentials plus continuation settings such as output directory, target steps, log/save frequencies, checkpoint toggle, and optional job name.

The new record retains `resume_from_job_id`, which drives nested dashboard display and stitched metrics history.

Source evidence:

- `frontend/src/components/jobs/JobCard.tsx:303-350`
- `frontend/src/pages/Training.tsx:198-231`
- `frontend/src/pages/Training.tsx:539-553`
- `makermodslab/jobs.py:641-680`
- `makermodslab/jobs.py:1161-1189`
- `makermodslab/train.py:248-270`

### Cloud Resume

The cloud Resume journey is offered for a failed or interrupted cloud run that ended before its configured target and still has a checkpoint.

The flow is:

1. Select the Hub checkpoint.
2. Click **Resume**.
3. Reopen the training form with the source's dataset, policy, cloud flavor, and cadence.
4. Verify the Hub checkpoint exists.
5. Verify `training_state/training_step.json` exists.
6. Start a new HF Job.
7. Download the selected checkpoint tree inside the container.
8. Reconstruct `checkpoints/<step>/pretrained_model` and `training_state`.
9. Resume through LeRobot's `config_path` pathway.
10. Continue uploading into the parent model repository.

The new job remains a distinct JobRecord and retains its resume lineage even though the Hub model artifacts stay in one repository.

Source evidence:

- `frontend/src/components/jobs/JobCard.tsx:309-355`
- `makermodslab/jobs.py:588-639`
- `makermodslab/jobs.py:1168-1180`
- `makermodslab/runners/hf_cloud.py:260-302`
- `makermodslab/runners/hf_cloud.py:528-547`

## 10. Fine-tuning journey

Fine-tuning is distinct from resume:

- It starts a fresh run at step 0.
- It uses a fresh optimizer.
- It initializes policy weights and processors from an existing model.
- The user selects a new dataset and configures normal fresh-run parameters.

The ordinary UI exposes **Fine-tune** for imported models. An untracked Hub model is first lazily registered as an imported pseudo-job, then routed through the same fine-tune flow.

Backend source resolution supports:

- Imported local flat model.
- Imported local checkpoint tree.
- Normal local training checkpoint.
- Imported Hub model.
- Cloud training repository.

A local source resolves to an absolute `pretrained_model` directory. A Hub source resolves to a repo ID. Cloud localization refuses an absolute host-local pretrained path because it cannot exist in the container.

For a Hub-backed checkpoint tree, MakerMods Lab validates the selected step but currently passes the plain repo ID to LeRobot because `policy.pretrained_path` accepts the model repository rather than a checkpoint subdirectory reference. The current Hub fine-tune handoff therefore uses the repo-root model.

Source evidence:

- `frontend/src/components/jobs/JobCard.tsx:357-380`
- `frontend/src/components/jobs/JobsSection.tsx:205-251`
- `frontend/src/pages/Training.tsx:555-571`
- `makermodslab/jobs.py:682-734`
- `makermodslab/jobs.py:1144-1155`
- `makermodslab/runners/hf_cloud.py:174-183`
- `makermodslab/train.py:279-289`

## 11. Model publication

### Automatic cloud publication

Cloud training enables `policy.push_to_hub` and assigns a repository before launch. Intermediate checkpoint trees are uploaded by the wrapper during training. LeRobot performs its normal final policy push at completion.

Fresh cloud runs use a per-job repo under the authenticated username. Resumed cloud runs reuse their parent repo so the lineage's checkpoints remain together.

The generated policy flags explicitly request public visibility and MakerMods Lab's required tags.

Source evidence:

- `makermodslab/runners/hf_cloud.py:518-547`
- `makermodslab/train.py:295-310`
- `makermodslab/utils/config.py:89-106`

### Manual upload of a local model

A completed local run appears as a model when it has at least one valid checkpoint. Its highest numeric checkpoint is considered final.

Manual upload from the model card:

1. Resolves the completed local run and final checkpoint.
2. Chooses an explicit repo ID, an existing recorded repo ID, or `<username>/<job-id>`.
3. Creates or updates a public model repo.
4. Uploads the final `pretrained_model` directory.
5. Stamps required MakerMods Lab tags plus the recognized policy type.
6. Invalidates model listing and info caches.

This upload is synchronous. The local run remains on disk.

Source evidence:

- `makermodslab/models.py:161-282`
- `makermodslab/models.py:875-958`
- `frontend/src/components/landing/ModelInfoCard.tsx:183-217`

## 12. Model discovery and browsing

### Merged Models listing

`GET /models` combines:

- Completed local training jobs with checkpoints.
- Downloaded/imported checkpoints under the local models cache.
- The authenticated user's Hub models.
- Models owned by authenticated organizations.
- Explicitly pinned custom Hub repo IDs.

Hub discovery accepts models that either:

- Carry the `lerobot` tag, or
- Match MakerMods Lab's timestamped cloud-run repo naming pattern.

This second condition keeps early-created or incomplete cloud repos visible even if they never received final model tags.

Rows are classified as:

- `local`
- `hub`
- `both`

Local checkpoint metadata wins when a local and Hub row collapse. Hidden model IDs are filtered after all merge and pin steps. The merged listing is cached for 45 seconds, with local mutations invalidating it.

Source evidence:

- `makermodslab/models.py:240-282`
- `makermodslab/models.py:413-464`
- `makermodslab/models.py:466-524`
- `makermodslab/models.py:526-670`

### Hub model details

The detail card obtains:

- Policy type.
- Base dataset when present in model-card metadata.
- Step count when recoverable.
- Local checkpoint path or Hub repo ID.
- Size.
- Last-modified time.
- Private/public indicator.

For Hub models, MakerMods Lab first uses expanded `model_info` metadata and falls back to inspecting the repo's checkpoint/config structure when needed.

Source evidence:

- `makermodslab/models.py:689-847`
- `frontend/src/components/landing/ModelInfoCard.tsx`

### Jobs and models dashboard

The Jobs section separately combines:

- Locally tracked jobs.
- Tracked cloud jobs.
- Untracked HF Jobs.
- Imported pseudo-jobs.
- Untracked Hub model repos.

A tracked HF Job ID suppresses the duplicate plain Hub-job card. A tracked cloud/imported repo ID suppresses the duplicate plain Hub-model card.

Resume successors can nest their source jobs. The dashboard fetches missing ancestors outside its ordinary ten-job page so old resume parents can still render.

Source evidence: `frontend/src/components/jobs/JobsSection.tsx:35-426`.

## 13. Import and download journeys

MakerMods Lab has two intentionally different model-import mechanisms.

### Import from disk into the Models browser

Landing-page **Import from disk** calls `/models/import`.

The backend:

- Accepts a root `pretrained_model` dir with `config.json`, or a training-output checkpoint tree.
- Validates the target name.
- Copies the entire source folder into:

  ```text
  ~/.cache/huggingface/lerobot/makermodslab_models/<name>/
  ```

- Leaves the source untouched.
- Removes a partial destination if the copy fails.
- Adds the result to the local model listing.

Source evidence:

- `frontend/src/components/landing/ImportModelFromDiskDialog.tsx`
- `makermodslab/models.py:1131-1193`
- `makermodslab/server.py:1057-1074`

### Register an imported pointer in Jobs

Jobs-section **Import model** calls `/jobs/import` with a local path or Hub repo ID.

The backend:

- Normalizes pasted Hub URLs and repo IDs.
- Recognizes flat models and checkpoint trees.
- Reads policy type best-effort from the selected checkpoint.
- Creates a `runner="imported"`, `state="done"` pseudo-job.
- Stores a path or Hub repo ID without copying model files.
- Returns an existing record when the source was already imported.

Deleting this record removes only MakerMods Lab's pointer metadata; it does not delete the source model.

Source evidence:

- `frontend/src/components/jobs/ImportModelModal.tsx`
- `makermodslab/jobs.py:1258-1351`
- `makermodslab/server.py:1166-1184`

### Add or download from the Hub

Landing-page **Add from Hugging Face** validates a `namespace/name` ID, pins and selects it, and optionally starts a background download.

Without download, inference resolves the Hub model on demand. With download, MakerMods Lab snapshots the repo into the local model cache. The downloader:

- Runs one model download at a time.
- Exposes `idle`, `running`, `done`, and `error` status.
- Survives frontend navigation through polling/reattachment.
- Accepts flat and checkpoint-tree layouts.
- Cleans an unusable partial directory on failure.
- Changes a matching listing row from `hub` to `both` on success.

Source evidence:

- `frontend/src/components/landing/AddModelFromHubDialog.tsx`
- `frontend/src/components/landing/ModelsPanel.tsx:152-175`
- `frontend/src/hooks/useModelDownload.ts`
- `makermodslab/models.py:1090-1123`
- `makermodslab/server.py:1027-1054`

## 14. Stop, delete, hide, and dismiss semantics

### Job deletion

Job deletion is permitted only after the record stops running.

Its meaning depends on the record:

- **Local training job:** removes the job directory, checkpoints, logs, and registry record.
- **Cloud training job:** removes the local record and logs; the Hub model repo remains.
- **Imported pseudo-job:** removes the pointer record; the source folder or Hub repo remains.

Deleting a tracked cloud record also adds its HF Job ID to the local dismissed set so the same compute job does not immediately return as an untracked card. The HF Jobs API itself provides no delete operation for historical compute jobs.

Source evidence:

- `makermodslab/jobs.py:1545-1558`
- `makermodslab/server.py:1664-1678`
- `frontend/src/components/jobs/JobCard.tsx:260-285`

### Model-browser removal

The landing Models browser resolves one of four meanings:

- **Local-only:** delete the local training run or copied/downloaded checkpoint.
- **Both:** remove the local copy and leave the Hub model listed.
- **Pinned custom Hub model:** unpin it.
- **Owned/discovered Hub-only model:** hide it from MakerMods Lab's listing.

These normal model-browser removals do not delete the Hub repository.

Local deletion is path-sandboxed. Running training jobs cannot be deleted, and a checkpoint currently being used by inference is protected.

Source evidence:

- `frontend/src/components/landing/ModelsPanel.tsx:72-147`
- `frontend/src/lib/deleteSemantics.ts`
- `makermodslab/models.py:962-1082`

### Permanent Hub deletion and untracked-job dismissal

The Jobs/Models dashboard has separate cleanup actions:

- A Hub model repo can be permanently deleted through `/jobs/hub/models/{repo_id}` only when it is under the authenticated user's own namespace.
- An untracked HF Job is locally dismissed because historical HF Jobs cannot be deleted through the API.
- Active jobs remain visible even if their ID was previously dismissed; dismissal applies once they become terminal.

Source evidence:

- `makermodslab/server.py:1427-1490`
- `makermodslab/server.py:1492-1512`
- `frontend/src/components/jobs/HubModelCard.tsx`
- `frontend/src/components/jobs/HubJobCard.tsx`

## 15. Handoff to inference

Training and model entry points converge on the same inference launch mechanism.

The handoff is based on:

```text
job id + checkpoint step
```

Checkpoint refs are opaque to the frontend and can represent:

- A local absolute `pretrained_model` directory.
- `repo@checkpoints/<step>` for a Hub checkpoint tree.
- `repo@root` for a flat Hub model.

Before inference starts, MakerMods Lab can read the selected checkpoint's policy config and return:

- Policy type.
- Required image/camera names and resolutions.
- Whether the policy requires a language task.
- State and action dimensions used for single-arm/bimanual compatibility checks.

Source evidence:

- `makermodslab/jobs.py:105-115`
- `makermodslab/jobs.py:810-837`
- `makermodslab/jobs.py:1505-1543`
- `makermodslab/server.py:1560-1573`

### From the monitoring page or a job card

The user selects a checkpoint and opens `InferenceModal` with that job ID and step. The modal owns robot/camera binding and the final inference request.

Source evidence:

- `frontend/src/pages/Training.tsx:929-965`
- `frontend/src/components/jobs/JobCard.tsx:287-300`

### From the landing Models browser

The Deploy footer first maps the selected model to an existing job:

- Local-run model IDs directly match their registry job.
- Existing cloud/imported records match by Hub repo ID.
- An untracked Hub or downloaded model has no job yet.

If no job covers the model, MakerMods Lab lazily registers the Hub repo or local checkpoint path through `/jobs/import`, then opens the same `InferenceModal`. Mere model selection does not mutate the registry; lazy import happens only when the user clicks Run inference.

Hub checkpoint refs are resolved to local directories by the inference backend before its robot subprocess starts.

Source evidence:

- `frontend/src/components/landing/ModelLaunchFooter.tsx`
- `frontend/src/lib/inferenceLaunch.ts`
- `frontend/src/hooks/useInferenceLaunch.tsx`
- `makermodslab/rollout.py:299-355`

## 16. API summary

### Training and job lifecycle

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/jobs/training` | Start local or cloud training |
| GET | `/jobs` | List tracked jobs/imported pointers |
| GET | `/jobs/{id}` | Get one job |
| GET | `/jobs/{id}/logs` | Drain new live log lines |
| GET | `/jobs/{id}/log-file` | Read complete persisted logs |
| GET | `/jobs/{id}/metrics-history` | Read reconstructed metric series |
| GET | `/jobs/{id}/checkpoints` | List available checkpoints |
| GET | `/jobs/{id}/checkpoints/{step}/policy-config` | Inspect inference-relevant policy config |
| GET | `/jobs/{id}/checkpoints/{step}/download` | Download a local pretrained model as ZIP |
| POST | `/jobs/{id}/stop` | Stop/cancel a running job |
| POST | `/jobs/{id}/rename` | Set a display-only alias |
| DELETE | `/jobs/{id}` | Delete a non-running tracked record |
| POST | `/jobs/import` | Register an external model pointer |

### Cloud discovery and target selection

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/jobs/runners/hardware` | Auth/offline state and HF hardware flavors |
| GET | `/jobs/hub` | List HF Jobs and relevant model repos |
| POST | `/jobs/hub/jobs/{id}/dismiss` | Persistently hide a terminal HF Job locally |
| DELETE | `/jobs/hub/models/{repo_id}` | Permanently delete an owned Hub model repo |

### Environment checks and installers

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/policy-optimizer-defaults` | Policy availability and optimizer presets |
| GET | `/system/training-extra` | Probe `accelerate` |
| POST | `/system/training-extra/install` | Start training-extra installation |
| GET | `/system/training-extra/install-status` | Poll installation |
| GET | `/system/wandb-credentials` | Probe for a resolvable W&B API key |
| GET | `/system/policy-extra/{type}` | Probe policy dependency |
| POST | `/system/policy-extra/{type}/install` | Install policy extra |
| GET | `/system/policy-extra/{type}/install-status` | Poll installation |

### Model browser

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/models` | Merged local/Hub model listing |
| GET | `/models/info?id=...` | Model details |
| POST | `/models/upload` | Upload a completed local model |
| POST | `/models/delete` | Delete local model files/run |
| POST | `/models/custom` | Pin a custom Hub model ID |
| DELETE | `/models/custom` | Unpin a custom ID |
| POST | `/models/hide` | Hide a model from the local listing |
| DELETE | `/models/hide` | Unhide a model |
| POST | `/models/download` | Start background Hub download |
| GET | `/models/download-status` | Poll model download |
| POST | `/models/import` | Copy a local model into MakerMods Lab's model cache |

## 17. Validation scope

The focused validation suites were:

```text
tests/test_train.py
tests/test_jobs.py
tests/test_runners_hf_cloud.py
tests/test_models.py
tests/test_training_preflight.py
```

At the audited commit, these produced:

```text
209 passed
```

They cover request and command assembly, duration parsing, runner localization, mocked cloud-wrapper behavior, checkpoint parsing, resume/fine-tune resolution, job persistence helpers, model discovery/import/upload/delete semantics, and training preflight.

They do not submit real HF Jobs, drive robot hardware, or run a real LeRobot training subprocess end to end.
