# Dataset Bug List

This ledger tracks defects specifically in MakerMods Lab's dataset acquisition, storage, publication, synchronization, and training pathways. Broader robot, recording, model, job-runner, and hardware issues remain in the main `Bug Ledger.md` unless their primary failure is dataset behavior.

Audited on July 14, 2026, against MakerMods Lab commit `518ca56`.

## Status key

- **Open:** confirmed defect with no validated fix.
- **In progress:** implementation has started but is not fully validated.
- **Fixed:** implementation and regression coverage are complete.
- **Design gap:** required behavior is documented but not yet implemented.

## Bugs

### 1. In progress — Local dataset cloud-training notice says private, but upload is public

**Severity:** High

**Verdict:** CONFIRMED — notice promises private (LocalDatasetCloudNotice.tsx:77-83) while both upload paths push public (Training.tsx:456 → record.py:1104; hf_cloud.py:671).

**Priority:** P0 — silent publication of user camera footage under an explicit private promise; coordinator-ratified (verified 2026-07-14 against the current working tree).

**Area:** Local-only dataset -> Hugging Face Jobs

**Current behavior**

When the selected dataset exists only in the local LeRobot cache and the user chooses Hugging Face cloud training:

1. `LocalDatasetCloudNotice.tsx` tells the user that the dataset will be uploaded as a **private** dataset.
2. `Training.tsx` starts the upload with `private=false`.
3. The HF cloud runner's fallback also calls `push_to_hub(..., private=False)` and logs that it is pushing the dataset publicly.

The resulting Hub repository is public even though the confirmation text says private.

**Impact**

- A user can unintentionally publish camera footage, robot observations, actions, task descriptions, or other locally collected data.
- The interface gives false assurance immediately before an external data mutation.
- The frontend upload path and backend fallback encode the same public behavior separately, increasing the chance that future fixes only correct one path.

**Intended behavior**

- **Upload & start training** must show the real Hub destination, approximate size, and visibility before uploading.
- Visibility must be an explicit choice and should default to private for an upload initiated only because cloud compute needs access.
- The selected visibility must flow through one centralized dataset publisher used by both the frontend chain and backend safety path.
- The upload must add the required Hub metadata tags and create or update the matching dataset-format revision only after the upload succeeds.
- The HF Job must start only after the remote dataset and revision are verified.

**Acceptance criteria**

- The notice, confirmation control, upload request, backend logs, and resulting Hub repository all agree on visibility.
- Private is the default for the cloud-training-triggered upload, with an explicit public option.
- Canceling the confirmation starts neither an upload nor a job.
- Both direct upload-and-start and backend fallback use the same visibility value and publication service.
- Automated tests cover private and public choices and assert the final `push_to_hub(private=...)` argument.
- A regression test prevents UI copy from claiming a fixed visibility independently of the actual request.

**Evidence**

- `frontend/src/components/training/config/LocalDatasetCloudNotice.tsx:77-82`
- `frontend/src/pages/Training.tsx:447-456`
- `makermodslab/runners/hf_cloud.py:641-671`

**Related design**

See **Local-only dataset to Hugging Face Jobs -> Intended upload-and-start behavior** in `complete functionalities/Dataset Interface Pathways.md`.

**Progress (2026-07-14)**

- Fixed the visibility mismatch on both paths: the cloud-training-triggered upload now defaults to **private**. `LocalDatasetCloudNotice.tsx` gained a "Make this dataset public" toggle (default off) whose state the notice text reflects ("private"/"public"); `Training.tsx` lifts that state (`uploadMakePublic`, default false) and passes `startUpload([], !uploadMakePublic)` so the request's `private` flag matches the UI. The backend safety fallback `hf_cloud.py:_ensure_dataset_on_hub` now calls `push_to_hub(..., private=True)` with the log line and policy comment updated to explain private-by-default (upload exists only so cloud compute can read the dataset; user opts into public from the UI). Regression coverage added in `tests/test_runners_hf_cloud.py` asserting the fallback pushes `private=True` (and that no push happens when the dataset already resolves on the Hub).
- Still open (full acceptance criteria not implemented): a single centralized dataset publisher shared by the frontend chain and backend fallback (the two paths still encode visibility separately); the upload-size preview shown before uploading; and the tag/revision sequencing (add Hub metadata tags + create/verify the dataset-format revision, and only start the HF Job after the remote dataset + revision are verified).

### 2. Open — Dataset in-use guard does not cover training jobs

**Severity:** High

**Verdict:** CONFIRMED-LIVE (user bench observation, 2026-07-14)

**Priority:** P1 — in-use-guard family, wastes paid GPU time, delayed failure mode (coordinator-ratified)

**Area:** Dataset deletion during training

**Current behavior**

Deleting a dataset while a training run is using it succeeds — LIVE-CONFIRMED on the Jetson bench 2026-07-14: the user deleted a dataset during a training run's warmup and the delete went through. What the guard actually checks (`_dataset_in_use`, `makermodslab/datasets.py:697-745`, called by the delete path at `makermodslab/record.py:900-904`): an active recording session (stamped or base repo id), a running Hub upload, a running merge output, and running **local** training jobs (`record.state == "running" and record.runner == "local"` with exact `config.dataset_repo_id` equality; registry scan capped at `limit=200`). Cloud (HF Jobs) runs are excluded outright by the `runner == "local"` filter — even during warmup, when the runner may still be uploading or reading the local dataset — and any repo-id form mismatch (bare vs namespaced) defeats the local-job check. (During the fix, also verify the bench build against this tree: `record.py:896-899` notes the delete path only recently began reusing the full guard in place of an upload-only check, so the live event may additionally reflect a pre-guard build.)

**Impact**

- A mid-training delete wastes the run (local) or paid GPU time (cloud).
- On Linux, open file handles can keep the trainer running until a later re-open, so the failure can surface long after the delete.

**Intended behavior**

The guard consults the job registry for running/starting jobs — local **and** cloud — whose config references the dataset (normalized repo-id comparison), and delete/rename is refused with a legible reason while any such job exists.

**Acceptance criteria**

- Delete and rename are refused for a local run's whole lifetime, including warmup.
- Delete and rename are refused while a running cloud job's config references the dataset.
- Bare and namespaced forms of the same repo id both match the guard.
- A regression test reproduces the live bench scenario (delete during warmup) and asserts refusal.

**Evidence**

- `makermodslab/datasets.py:697-745` (`_dataset_in_use` — what it checks and the `runner == "local"` exclusion at line 740)
- `makermodslab/record.py:884-920` (`handle_delete_dataset` — the delete path and its guard call)
- User bench observation, Jetson, 2026-07-14 (delete during training warmup succeeded)
