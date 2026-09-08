# Dataset Bug List

*Citations synced 2026-07-26 against `main` @ `ce42158`. **Only `file:line` coordinates were changed** — no prose, verdicts, severities, entry IDs or numbering were altered. References whose construct no longer exists at HEAD are marked inline `⚠ UNRESOLVED @ HEAD ce42158`.*

> **Re-verified against `main` @ `c7d9f27` on 2026-07-23.** This list folds the 23-07-26 re-audit
> into the original. It **supersedes** the pre-reverification version, archived at
> [`archive/Dataset Bug List (pre-reverify).md`](archive/Dataset%20Bug%20List%20%28pre-reverify%29.md);
> the standalone re-audit is archived at
> [`archive/Dataset REVERIFIED 23-07-26.md`](archive/Dataset%20REVERIFIED%2023-07-26.md).
> Cross-module current-state overview: [`CURRENT 23-07-26.md`](CURRENT%2023-07-26.md).
>
> **Context you need for every verdict below:**
> - The original entries were forensically verified **2026-07-14 against the `andrew` branch** (commit
>   `518ca56`). The re-verification target is **`main` @ `c7d9f27`** ("Merge pull request #4 from
>   makermods-robotics/redesign"), working tree of 2026-07-23. `518ca56` is **not an ancestor of `main`** —
>   the two lines diverge, and **`andrew`-branch fixes are NOT on `main`.** Re-verifying "against `main`"
>   therefore *reopens* entries whose fix only ever lived on `andrew`.
> - **The redesign moved files.** Old citations that named `Training.tsx:456`,
>   `frontend/src/components/landing/DatasetsPanel.tsx`, etc. no longer resolve; the upload chain moved to
>   `TrainingConfigurator.tsx`. Updated `main`-current cites are layered onto each entry.
> - **PRs #5–#11 are CLOSED with `merged=no`.** PR #8 (Dataset #1 visibility) targeted this backlog and did
>   not merge. Nothing here is in flight.
> - Analysis is **purely static** — no Hub call, no app run, no hardware. No code was modified.

This ledger tracks defects specifically in MakerMods Lab's dataset acquisition, storage, publication, synchronization, and training pathways. Broader robot, recording, model, job-runner, and hardware issues live in the sibling module Bug Lists unless their primary failure is dataset behavior. *(This sentence originally pointed at a `Bug Ledger.md`; that file does not exist in the repo — reference dropped 2026-07-27.)*

Audited on July 14, 2026, against MakerMods Lab commit `518ca56` (original); re-verified against `main` @ `c7d9f27` on 2026-07-23.

## Status key

- **Open:** confirmed defect with no validated fix.
- **In progress:** implementation has started but is not fully validated.
- **Fixed:** implementation and regression coverage are complete.
- **Design gap:** required behavior is documented but not yet implemented.

## Verdict key (re-verification, 2026-07-23)

- **STILL REAL** — the defect reproduces in `main`'s code as written.
- **FIXED** — `main` no longer has the defect.
- **N/A-SUPERSEDED** — the code the entry described no longer exists in that form.
- **CANNOT VERIFY STATICALLY** — needs a runtime/Hub observation.

## Bugs

### 1. In progress — Local dataset cloud-training notice says private, but upload is public

**First identified:** 2026-07-14

**Severity:** High

**Verdict:** CONFIRMED — notice promises private (LocalDatasetCloudNotice.tsx:77-83) while both upload paths push public (Training.tsx:456 → record.py:1116; hf_cloud.py:671).

**Priority:** ~~P0~~ → **P1, reframed 2026-07-26.** The policy question is CLOSED: **public-by-default upload
is intended** (product decision; see CURRENT → "Descoped by product decision", which retired the record-flow
sibling N1 / Recording N2). So this is no longer *silent publication of private data* — it is a **copy bug**:
the notice makes a promise the product never intended to keep. **Fix direction is now unambiguous — correct
the notice to say public; do not change the upload behavior.** Original P0 rating (coordinator-ratified,
verified 2026-07-14) applied under the pre-decision reading.

> **Verdict on `main` (2026-07-23): STILL REAL — and the andrew-branch fix is NOT on `main`; `main` moved
> further the other way.** *(The "moved further the other way" framing is 2026-07-23's; as of 2026-07-26
> `main`'s public-default policy comment is the **correct** policy, and it is the notice that is out of step.)*
> Formerly shared with the record-flow auto-push as **CURRENT P0-2** (now retired) and **RANKING P0 #1**. Evidence on `main`: `LocalDatasetCloudNotice.tsx:77-83`; `TrainingConfigurator.tsx:432`;
> `hf_cloud.py:661,665-671`.
>
> **What `main` does today, at all three sites:**
>
> 1. **Notice text** — `LocalDatasetCloudNotice.tsx:77-83` hardcodes the word `private` as a literal. There
>    is **no visibility toggle** in this component on `main` — the props interface (`:4-17`) has no
>    visibility field. The `uploadMakePublic` / "Make this dataset public" toggle described in the
>    andrew-branch progress note below **does not exist here.**
> 2. **Frontend upload call** — the `Training.tsx:456` call site is gone (the redesign reduced `Training.tsx`
>    to a 105-line router wrapper). The upload chain now lives in `TrainingConfigurator.tsx:429-432`, which
>    calls `startUpload([], false /* public ... */)` → `POST /upload-dataset` with `private: false`
>    (`useDatasetUpload.ts:30,111-120`; `replayApi.ts:289`) → `record.py:1116`
>    `dataset.push_to_hub(tags=tags, private=request.private)`. **Result: PUBLIC.**
> 3. **Backend fallback** — `hf_cloud.py:641-676` (`_ensure_dataset_on_hub`) logs "pushing local copy
>    (public)" and calls `push_to_hub(..., private=False)`, with a policy comment at `:665-670` that
>    **explicitly reverses the earlier private default** ("an implicit upload of a local-only dataset is now
>    public").
>
> The two *upload paths* now agree with each other (both public). The *notice text* is the one site still
> promising "private", and it is the only thing the user reads before the upload fires. So the mismatch the
> entry describes still exists, but its shape changed: from "one path forgot to be private" to **"the code
> committed to public-by-default and the UI copy was never updated."**
>
> Corroborating: `remotes/origin/fix/dataset-cloud-notice-public-mismatch` is unmerged; **PR #8 is
> CLOSED-unmerged**. The regression test the progress note claims (`tests/test_runners_hf_cloud.py`
> asserting `private=True`) is **not present**.
>
> **The correct end-state (public vs private default) is an OPEN PRODUCT DECISION, deliberately not resolved
> here.** Options, stated neutrally:
> - **(A) Private default** — `TrainingConfigurator.tsx:432` → `true`, `hf_cloud.py:671` → `private=True`,
>   keep the notice. Contradicts the explicit policy comment at `hf_cloud.py:665-670`.
> - **(B) Public default** — change the notice text at `LocalDatasetCloudNotice.tsx:81` from "private" to
>   "public", ideally with the camera-footage warning `UploadDatasetDialog.tsx:105` already uses.
> - **(C) Explicit choice** — add the toggle to the notice (what PR #8 targeted); no surprising default.
>
> Whichever is chosen, the acceptance criterion "notice, request, backend logs and the resulting repo all
> agree" is currently **unmet on `main`**, and there is still **no single centralized publisher** —
> `TrainingConfigurator.tsx:432`, `CollectHandoff.tsx:194` and `hf_cloud.py:671` each encode visibility
> separately (see **N1**, a fourth, worse path). *Cannot verify statically:* the live repo's actual
> visibility (no Hub call) — inferred from `private=False` reaching `create_repo(private=False)`.

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

**Progress (2026-07-14) — NOTE: this describes andrew-branch work that is NOT on `main` (see verdict above)**

- Fixed the visibility mismatch on both paths: the cloud-training-triggered upload now defaults to **private**. `LocalDatasetCloudNotice.tsx` gained a "Make this dataset public" toggle (default off) whose state the notice text reflects ("private"/"public"); `Training.tsx` lifts that state (`uploadMakePublic`, default false) and passes `startUpload([], !uploadMakePublic)` so the request's `private` flag matches the UI. The backend safety fallback `hf_cloud.py:_ensure_dataset_on_hub` now calls `push_to_hub(..., private=True)` with the log line and policy comment updated to explain private-by-default (upload exists only so cloud compute can read the dataset; user opts into public from the UI). Regression coverage added in `tests/test_runners_hf_cloud.py` asserting the fallback pushes `private=True` (and that no push happens when the dataset already resolves on the Hub).
- Still open (full acceptance criteria not implemented): a single centralized dataset publisher shared by the frontend chain and backend fallback (the two paths still encode visibility separately); the upload-size preview shown before uploading; and the tag/revision sequencing (add Hub metadata tags + create/verify the dataset-format revision, and only start the HF Job after the remote dataset + revision are verified).

### 2. Open — Dataset in-use guard does not cover training jobs

**First identified:** 2026-07-14

**Severity:** High

**Verdict:** CONFIRMED-LIVE (user bench observation, 2026-07-14)

**Priority:** P1 — in-use-guard family, wastes paid GPU time, delayed failure mode (coordinator-ratified)

> **Verdict on `main` (2026-07-23): STILL REAL — unchanged.** `datasets.py:696-746` is byte-for-byte the
> behaviour the entry describes:
>
> ```python
> for record in job_registry.list(limit=200):
>     if (
>         record.state == "running"
>         and record.runner == "local"                       # datasets.py:741
>         and record.config.dataset_repo_id == repo_id       # datasets.py:742
>     ):
>         return "A local training run is using this dataset. Stop it first."
> ```
>
> - Cloud (`runner == "hf_cloud"`) runs remain excluded (`:741`).
> - Comparison is still raw string equality — no repo-id normalization (`:742`).
> - The registry scan is still capped (`job_registry.list(limit=200)`, `:738`); `JobRegistry.list` sorts by
>   `started_at` descending and truncates (`jobs.py:1083-1089`), so a **long-running job older than the 200
>   most recent records is invisible to the guard** — a second, previously unnoted fail-open.
> - The delete path does reuse the full guard (`record.py:865-869`), so the original "pre-guard build"
>   caveat is resolved — `main` has the guard, it is just incomplete.
> - Additional gap not in the original entry: `_dataset_in_use` protects the merge **output**
>   (`datasets.py:731-733`) but **not the merge sources** — see **N6**.

**Area:** Dataset deletion during training

**Current behavior**

Deleting a dataset while a training run is using it succeeds — LIVE-CONFIRMED on the Jetson bench 2026-07-14: the user deleted a dataset during a training run's warmup and the delete went through. What the guard actually checks (`_dataset_in_use`, `makermodslab/datasets.py:697-745`, called by the delete path at `makermodslab/record.py:912-916`): an active recording session (stamped or base repo id), a running Hub upload, a running merge output, and running **local** training jobs (`record.state == "running" and record.runner == "local"` with exact `config.dataset_repo_id` equality; registry scan capped at `limit=200`). Cloud (HF Jobs) runs are excluded outright by the `runner == "local"` filter — even during warmup, when the runner may still be uploading or reading the local dataset — and any repo-id form mismatch (bare vs namespaced) defeats the local-job check. (During the fix, also verify the bench build against this tree: `record.py:908-911` notes the delete path only recently began reusing the full guard in place of an upload-only check, so the live event may additionally reflect a pre-guard build.)

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
- `makermodslab/record.py:896-932` (`handle_delete_dataset` — the delete path and its guard call)
- User bench observation, Jetson, 2026-07-14 (delete during training warmup succeeded)

---

## Appended findings (N-series)

One consolidated, append-only list: every finding since the original audit gets the next `N` here, newest last. IDs are permanent and per-module — "Recording N1" is distinct from "Dataset N1" / "Inference N1". N1–N19 date from the 2026-07-23 re-verification, N20–N24 from the 2026-07-24 working session; severities were folded into each header from the former P0/P1/P2 group headings. *(Consolidated 2026-08-02 from the former "New findings" sections and later in-place appends; no prose, verdicts or severities were changed.)*

### N1 · ~~P0~~ — ~~Recording auto-pushes the just-recorded dataset to the Hub as PUBLIC~~ — **DESCOPED, not a bug**

**First identified:** 2026-07-23 · **Descoped:** 2026-07-26 (product decision)

> Was **CURRENT P0-2**, found independently as **Recording N2**. **Both IDs are retired, not reused.**
> **Do not re-file this chain from a future audit.**

The traced chain — `StudioContext.tsx:42` (`pushToHub: true` default) → `RecordingForm.tsx:237-255` (checkbox inside a collapsed "advanced" disclosure whose copy never mentions visibility) → `CollectHandoff.tsx:109` (`autoStart`) → `CollectHandoff.tsx:188-202` (`start([], false)` on mount, no dialog, no confirmation; docstring says *"no tags, public"*) → `useDatasetUpload.ts:112-120` → `record.py:1116` → **public repo** — is accurate and still present on `main`.

**The verdict was wrong, not the trace.** Public-by-default dataset upload is intended: recorded datasets are meant to be discoverable, published without a per-session confirmation step. This was ranked P0 under *silent publication of private data*, which does not apply to a publication the product means to perform. Nothing to fix.

Knock-on for this module: **Dataset #1 survives as a copy bug** (the cloud notice promises *private* while every path publishes public — correct the notice, not the behavior) and **Dataset N10** as a centralization issue. See CURRENT → "Descoped by product decision".

### N2 · P1 — A failed/aborted Hub download can leave a **partial dataset that passes every "is it complete?" check**

**First identified:** 2026-07-23

`makermodslab/datasets.py:1028-1033`:

```python
def _cleanup_partial_dataset(repo_id: str) -> None:
    target = _lerobot_cache_root() / repo_id
    if target.exists() and not _is_dataset_dir(target):
        shutil.rmtree(target, ignore_errors=True)
```

`_is_dataset_dir` is `(<dir>/meta/info.json).is_file()` (`datasets.py:333-338`). `snapshot_download` (`datasets.py:1020`, `local_dir=` flat layout) fetches many files concurrently; `meta/info.json` is a few hundred bytes and will typically land early. If the download dies **after** `meta/info.json` but **before** the parquet/video payload:

- the cleanup guard sees a "valid" dataset dir and **skips the rmtree**;
- `is_dataset_available_locally` returns True (`datasets.py:368-369`);
- `list_local_datasets` lists it, because `_dataset_has_episodes` reads `total_episodes` out of the **Hub's** `info.json` and gets a non-zero value (`datasets.py:397-406,442-453`);
- `get_hub_status` will happily answer `on_hub`/`local_only`; the info card renders full episode/frame counts from the truncated metadata.

The user sees a complete-looking local dataset. The failure surfaces much later — in training or in a merge — as a `FileNotFoundError` on a missing data file. There is no "re-download" repair path either: `datasets_download` → `DownloadManager.start` will just `snapshot_download` over the top, which may or may not repair it.

The same class of residue is produced by the merge subprocess's `_ensure_local_source` (`merge.py:576-592`) — it `snapshot_download`s a Hub source into the flat cache and has **no cleanup at all** on failure; `_run_cli` only cleans the *output* (`merge.py:619-637`).

**Suggested direction (not applied):** make the cleanup unconditional for a download this manager started (mirror `_run_cli`'s `output_pre_existed` pattern), or verify the snapshot against `info.json`'s file manifest before declaring success.

**Evidence:** `datasets.py:1010-1036`, `datasets.py:333-338`, `datasets.py:397-406`, `merge.py:576-592`, `merge.py:619-637`. *Cannot fully verify statically:* whether a partial `snapshot_download` actually lands `meta/info.json` first depends on huggingface_hub's thread-pool scheduling; the guard logic is unambiguously wrong, but the hit rate needs a live interrupted download. Static verdict: real; frequency unknown.

### N3 · P1 — Local dataset delete does **not** invalidate the Hub-status cache — and the two sibling discard paths do

**First identified:** 2026-07-23

`makermodslab/record.py:886-897` (`handle_delete_dataset`):

```python
shutil.rmtree(target)
...
invalidate_dataset_listing_cache()          # record.py:894
logger.info(f"Deleted dataset directory {target}")
```

Compare its two siblings, which get it right:

- `_discard_empty_dataset` — `record.py:960-961`: `invalidate_hub_status(repo_id)` **and** `invalidate_dataset_listing_cache()`
- `_discard_session_dataset` — `record.py:1015-1016`: same pair
- `rename_local_dataset` — `datasets.py:800-802`: invalidates hub-status for **both** ids

`_HUB_STATUS_CACHE` memoizes `local_only` **for the process lifetime** (`datasets.py:213-217`). So after the user deletes a local-only dataset, `/datasets/hub-status` keeps answering `local_only` until the server restarts. Downstream consequences:

- `DatasetInfoCard`'s `NotDownloadedView` takes the `local_only` branch and tells the user *"This dataset is on this machine, but its details couldn't be read — the local copy looks incomplete or corrupt. Re-record or re-download it."* (`DatasetInfoCard.tsx:834-846`) for a dataset that simply **does not exist**, and offers an "Upload to Hub" button for it.
- Pressing that button runs `LeRobotDataset(repo_id)` (`record.py:1111`) with no `root=`, which for a non-local repo id **downloads the whole dataset from the Hub** before "uploading" it. On a deleted local-only dataset it just fails, confusingly.
- `jobs.py:1119-1120`'s cloud gate keeps refusing cloud runs on the (now absent) id.

One-line asymmetry; the fix is to add `invalidate_hub_status(repo_id)` next to `record.py:894`.

### N4 · P1 — On `main` there is **no UI at all** to delete a local-only dataset

**First identified:** 2026-07-23

- The only dataset-delete entry point wired into the app is `ManageCachesDialog` (`LibrarySheet.tsx:29,366,435`), and it filters to **`source === "both"` only** (`ManageCachesDialog.tsx:44`: `datasets.filter((d) => d.source === "both")`).
- `DatasetPicker.tsx` — which carries the per-row trash affordance (`:95-120`) — has **no importer** anywhere in `frontend/src` (verified by grep). It is dead code after the redesign.
- `DatasetInfoCard`'s `canDelete` / `onDelete` props (`:886-891`) are never passed: its only consumer is `DatasetDetailDialog.tsx:12`, which supplies neither.
- Consequently `resolveDeleteAction`'s `"delete-local"` branch (`lib/deleteSemantics.ts:88-94`) is unreachable for datasets, and its `LOCAL_DELETE_DESCRIPTION` copy (`:39-46`) is dead.

**Impact:** a recorded dataset that was never pushed to the Hub (i.e. the user unchecked the default-on `pushToHub`) can only be removed by hand from `~/.cache/huggingface/lerobot`. Multi-GB recordings accumulate with no in-app cleanup. `POST /delete-dataset` still works — this is purely a missing frontend surface.

This interacts badly with **N1**: the *only* datasets the app can clean up are the ones it has already published. *(Whether `DatasetPicker.tsx` is retained-for-reinstatement vs. accidentally orphaned is an author question; per the steelman rule this flags the user-visible consequence, not that the file is a mistake.)*

### N5 · P1 — "Remove local copy" of a `both` dataset can silently discard episodes the Hub copy does not have

**First identified:** 2026-07-23

`lib/deleteSemantics.ts:54-64` promises: *"This removes the local copy from disk — the Hub copy stays, and it remains listed as a Hub dataset."* `ManageCachesDialog`'s header repeats it (*"The Hub copy stays — clearing only removes the local copy"*), and the action is a hard `shutil.rmtree` via `POST /delete-dataset` (`ManageCachesDialog.tsx:97` → `record.py:887`).

But `source: "both"` is decided purely by **repo-id set intersection** — `list_all_datasets` (`datasets.py:870-883`) marks a row `both` when the same id appears in the Hub listing and in the local scan. It never compares content, episode counts, or timestamps. So:

> record `user/pick` (10 eps) → upload → resume/record 10 more episodes locally (local = 20 eps, Hub = 10) → "Manage cached datasets" → "Clear" → **10 episodes are permanently gone**, under a UI promise that the Hub copy is equivalent.

`ManageCachesDialog` fetches `/datasets/info` per row for the **size** (`:71-83`) but never compares `total_episodes` against the Hub's. `get_hub_dataset_info` (`datasets.py:636-682`) already fetches the Hub's `meta/info.json` and would give the comparison cheaply.

**Suggested direction (not applied):** compare local vs Hub `total_episodes` before offering "Clear", and downgrade the copy to a warning when the local copy is ahead. *(Real-world frequency depends on whether users routinely resume-record into an already-uploaded dataset — code path unambiguous, exposure not.)*

### N6 · P2 — `_dataset_in_use` does not protect merge **sources**

**First identified:** 2026-07-23

`datasets.py:731-733` only matches `merge_manager.output_repo_id`. A source dataset can therefore be deleted or renamed while a merge is reading it. The merge then fails partway; `_run_cli` cleans the partial *output* (`merge.py:635-636`) and `_cli_friendly_error` produces a "looks incomplete or corrupt" message that **blames the wrong thing** (`merge.py:498-507`) — it says the source is corrupt when the user deleted it. `MergeRequest.source_repo_ids` is available on the manager, so the guard extension is mechanical.

### N7 · P2 — Merge completion never invalidates the dataset listing cache

**First identified:** 2026-07-23

`makermodslab/merge.py` contains **zero** calls to `invalidate_dataset_listing_cache` / `invalidate_hub_status` (verified by grep across `makermodslab/`). `_LISTING_CACHE_TTL_S = 45.0` (`datasets.py:97`), and `MergeDatasetsDialog`'s `onMerged` → `refreshDatasets` (`LibrarySheet.tsx:402`) just re-GETs `/datasets`, which returns the cached pre-merge list. The merge output can be invisible for up to 45 s after the dialog says "Created …". The listing-cache docstring (`datasets.py:102-106`) claims *"Called after any mutation that changes the listing"* — merge is the counterexample. The merge subprocess's `_ensure_local_source` downloads (`merge.py:576-592`) have the same problem.

### N8 · P2 — A completed recording session never invalidates the dataset listing cache either

**First identified:** 2026-07-23

Same 45 s window: `record.py` invalidates on delete (`:894`), discard (`:960-961`, `:1015-1016`) and upload (`:1112-1114`), but there is no invalidation on the **success** path of a recording session. The just-recorded dataset may not appear in the library for up to 45 s. (`CollectHandoff` masks this by preselecting the repo id directly, so it only bites the library/picker views.)

### N9 · P2 — Logging in does not invalidate the dataset listing cache or the Hub-status cache

**First identified:** 2026-07-23

`handle_hf_login` (`utils/hf_auth.py:129-152`) calls `invalidate_whoami_cache()` and nothing else. Two consequences:

- `/datasets` keeps returning the pre-login (Hub-less) listing for up to 45 s (`datasets.py:862-865`).
- Worse and unbounded: `get_hub_status` caches `local_only` **for the process lifetime** (`datasets.py:213-217`). `HfApi.repo_exists` returns `False` for a **private** repo when no token is present. So any dataset whose status was probed while logged out is permanently pinned to "Local only", even though it exists privately on the Hub. The info card then offers "Upload to Hub" (`DatasetInfoCard.tsx:581-595`) and the upload overwrites the existing private repo's contents. (`push_to_hub` uses `create_repo(..., exist_ok=True)`, which does **not** flip an existing repo's visibility — so this is content clobbering, not a visibility leak.)

`handle_hf_login` should call `invalidate_dataset_listing_cache()` and clear `_HUB_STATUS_CACHE`.

### N10 · P2 — `get_hub_status`'s `on_hub` answer is cached for the process lifetime and is load-bearing for the cloud gate

**First identified:** 2026-07-23

`datasets.py:213-217` memoizes `on_hub` forever. `jobs.py:1116-1120` uses it as the *only* gate preventing a cloud job on a local-only dataset. If the repo is deleted on the Hub (or moved) after being cached, the gate passes and control lands in `_ensure_dataset_on_hub`, which pushes the local copy **publicly** (`hf_cloud.py:671`). This is a second, cache-driven route into the N1/#1 visibility problem. Nothing outside MakerMods Lab invalidates the entry. *(Staleness window needs a running server plus an out-of-band Hub deletion to observe.)*

### N11 · P2 — `_ensure_dataset_on_hub` catches only `RepositoryNotFoundError`

**First identified:** 2026-07-23

`hf_cloud.py:649-653`:

```python
try:
    self._api.dataset_info(repo_id)
    return
except RepositoryNotFoundError:
    pass
```

A transient transport error, a rate limit, or a 403 on a repo the token can read but not `dataset_info` propagates out of `start()` and fails the job with a raw exception, even though the dataset is fine. The `jobs.py` gate comment (`:1111-1113`) explicitly defers the "unknown" (offline) case to this fallback, which then cannot handle it.

### N12 · P2 — The Hub-upload path "uploads" datasets it may first have to download

**First identified:** 2026-07-23

`record.py:1111` does `LeRobotDataset(repo_id)` with no `root=`. For a repo id that is **not** in the local flat cache, lerobot's constructor downloads the dataset from the Hub before the "upload" runs (`lerobot/datasets/lerobot_dataset.py:634-655`). `DatasetInfoCard.tsx:579-580` deliberately offers the Upload button for `status === "unknown"` ("the endpoint is a safe upsert"), so a user behind a flaky link can trigger a multi-GB download by pressing *Upload*. Progress is reported as "Uploading…" the whole time.

### N13 · P2 — `UploadManager` / delete / rename share no lock — TOCTOU on the in-use guard

**First identified:** 2026-07-23

`handle_delete_dataset` checks `_dataset_in_use` (`record.py:879`) and then `rmtree`s (`record.py:887`) with nothing held. `UploadManager.start` checks the same guard under **its own** `self._lock` (`record.py:1065-1080`). A delete that passes the guard microseconds before an upload starts will rmtree the directory the upload is about to read. Low probability, but the guard's whole purpose is to prevent exactly this.

### N14 · P2 — `_dataset_in_use` does a filesystem scan per registry record on every delete/rename/upload

**First identified:** 2026-07-23

`JobRegistry.list` calls `_count_checkpoints(r)` for each returned record (`jobs.py:1087-1089`), and `_dataset_in_use` asks for `limit=200` (`datasets.py:738`). Every delete, rename, and upload start therefore walks up to 200 checkpoint trees. Cosmetic today; it will not stay cosmetic.

### N15 · P2 — `MergeManager.get_status` drains the log queue destructively

**First identified:** 2026-07-23

`merge.py:381-392` drains `log_queue` on every poll. Two clients polling `/datasets/merge/status` (two tabs, or the dialog plus a stray poller) split the log lines between them; each sees a partial log. The persisted `merge_logs/<ts>.log` (`merge.py:428-440`) is the only complete record.

### N16 · P2 — `useDatasets` wipes the list to empty on any fetch failure

**First identified:** 2026-07-23

`frontend/src/hooks/useDatasets.ts:14`: `.catch(() => setDatasets([]))`. A single transient `/datasets` failure makes the library render *"No datasets yet. Record your first one above."* (`DatasetLibrary.tsx:244-255`) — indistinguishable from actual data loss. The hook has no error state at all.

### N17 · P2 — Per-author Hub listing is silently truncated at 200

**First identified:** 2026-07-23

`datasets.py:824`: `api.list_datasets(author=author, limit=200)`. A user (or org) with more than 200 datasets gets a silently short list with no indication. Same cap, undisclosed, in `models.py`.

### N18 · P2 — Stale docstrings claim "private by default" where the code is public by default

**First identified:** 2026-07-23

- `UploadDatasetDialog.tsx:17-18` — *"Private-by-default toggle (with the camera-footage note)"*, while `:38` is `useState(false)` → **Public** is the preselected side of `VisibilityToggle` (`VisibilityToggle.tsx:12-13`: `value` is the *private* flag).
- `DatasetInfoCard.tsx:458` repeats *"a confirm popover (private-by-default toggle + optional tags)"*.

The rendered dialog is **honest** — with Public selected it says *"Anyone can see this dataset — recordings include your camera footage."* (`UploadDatasetDialog.tsx:103-106`) — so this is a comment bug, not a user-facing one. Flagged because it is exactly the kind of stale comment that makes a future reviewer conclude the visibility problem is already fixed.

### N19 · P2 — Pinned custom Hub datasets always render as public in the library

**First identified:** 2026-07-23

`datasets.py:896-902` seeds a pinned row with `"private": False` and the listing never backfills it (`/datasets/hub-status` returns only existence, not visibility). `DatasetLibrary.tsx:137-145` gates the lock chip on `item.private`, so a **private** pinned dataset displays with no privacy indicator. Cosmetic, but it is a visibility signal that reads as authoritative.

---

### N20 · P0 — A bare dataset name resolves to the same level as MakerMods Lab's own state dirs, and delete `rmtree`s whatever is there

**First identified:** 2026-07-24

`validate_dataset_repo_id` (`makermodslab/utils/config.py:989-1005`) accepts a **single-segment** id — the
`len(parts) == 2` branch is optional, and a bare name falls through to `validate_dataset_name` (`:966-986`),
which enforces only character class, length and the `.`/`..` cases. **Nothing is reserved.** A bare name
therefore resolves to `<HF_LEROBOT_HOME>/<name>` — the same directory level MakerMods Lab uses for its own
persistent state:

| dir | constant / owner |
| --- | --- |
| `calibration/` | `CALIBRATION_BASE_PATH_TELEOP` / `_ROBOTS` (`config.py:29-30`) |
| `robots/` | `ROBOTS_PATH` (`config.py:40`) |
| `ports/` | `PORT_CONFIG_PATH` (`config.py:35`) |
| `makermodslab_biso/` | `MAKERMODSLAB_BISO_STAGING_PATH` (`config.py:51`) |
| `makermodslab_models/` | `_local_models_root()` (`models.py:295-308`) |
| `outputs/`, `hub/`, `merge_logs/`, `inference_logs/` | job registry / HF cache / merge + rollout logs |

`handle_delete_dataset` (`makermodslab/record.py:859-897`) guards **only** path traversal —

```python
root = Path(HF_LEROBOT_HOME).resolve()
target = (root / repo_id).resolve()
if target == root or root not in target.parents:      # record.py:870
    return {"success": False, "message": "Invalid dataset path"}
...
shutil.rmtree(target)                                  # record.py:887
```

— plus the `_dataset_in_use` busy-check. A `repo_id` of `calibration`, `robots`, `ports`,
`makermodslab_models`, `outputs` … passes both: it is strictly inside the root, so the traversal test is
satisfied, and `_dataset_in_use` knows nothing about state dirs. **There is no check that the target is
actually a LeRobot dataset** (`_is_dataset_dir`, `datasets.py:333-338`, is never consulted here) and no
reserved-name list anywhere. Deleting a "dataset" named `calibration` removes **every calibration profile**;
`robots` removes **every robot record**; `outputs` removes the whole local training job registry.

Two-segment (`namespace/name`) ids are unaffected — they land one level deeper.

**Why bare names occur:** the record path only namespaces when the user is logged in
(`CollectPanel.tsx:180-183` falls back to the bare `datasetName` when `auth.status !== "authenticated"`),
and the disk-import path defaults to the source folder's basename with no namespace at all (see **N21**).

**Adjacent, same root cause:** `import_local_dataset` refuses a target that already exists
(`datasets.py:1106-1107` → HTTP 409), so an import cannot merge *into* a state dir. The recorder has no such
guard, so a recording named after a state dir writes its dataset files *inside* that directory.

**Suggested direction (not applied):** a reserved-name set rejected in `validate_dataset_repo_id`, plus an
`_is_dataset_dir(target)` precondition in `handle_delete_dataset` (and in the two sibling discard paths at
`record.py:936` / `:1002`, which carry the same traversal-only guard). *Cannot verify statically:* the
destructive outcome itself — establishing it would require actually deleting real calibration data, which
this audit does not do. The control flow above is unambiguous.

### N21 · P1 — Neither import nor the logged-out record path prefills the Hub namespace, so both mint bare names

**First identified:** 2026-07-24

`import_local_dataset` defaults the target id to the **source folder's basename** — `raw = (name or "").strip()
or src.name` (`datasets.py:1091`) — and the dialog reinforces it: `ImportDatasetFromDiskDialog.tsx:115-124`
labels the field "Name (optional)" with placeholder *"Defaults to the folder name"*, and the component never
imports `useHfAuth` (verified by grep). The backend already knows the account — `cached_whoami`
(`makermodslab/utils/hf_auth.py`) — and the frontend already has `auth.username` in context (`HfAuthContext`,
used two files away in `LibrarySheet.tsx:159`).

Consequences:

1. Bare-named datasets, which are the precondition for **N20**.
2. A dataset that cannot be pushed without a rename: every publisher needs `namespace/name`
   (`CollectHandoff.tsx:109` gates auto-push on `repoId.includes("/")`).

**Record path — partially handled, one gap.** Contrary to a first reading, the Collect flow *does* prefill:
`RecordingForm.tsx:133-138` previews *"Will be saved as `{auth.username}/{datasetName}`"*, and
`CollectPanel.tsx:180-183` builds the id accordingly. But the else-branch is a **bare name**:

```tsx
const datasetRepoId =
  auth.status === "authenticated" ? `${auth.username}/${datasetName}` : datasetName;
```

So a logged-out / offline station (the documented deployment mode — see Config entry 10) records straight
into a bare-named directory. Cross-filed as **Recording N10**.

### N22 · P2 — `import_local_dataset` copies synchronously inside the request

**First identified:** 2026-07-24

`shutil.copytree(src, dst)` (`datasets.py:1111`) runs inline in the route (`server.py:789`), so a multi-GB
import blocks the HTTP request for its whole duration with only a frontend spinner. The code already
acknowledges this and names the fix: *"the copy runs SYNCHRONOUSLY … a background manager (like
DownloadManager) would be the follow-up if that becomes a pain point"* (`datasets.py:1074-1078`). Filed so the
acknowledged follow-up is tracked rather than living only in a docstring; the tradeoff (local disk copy ≪
network fetch) is reasonable as stated.

### N23 · P2 — `DatasetCard` truncates the one field users disambiguate by, then repeats it twice more

**First identified:** 2026-07-24

`frontend/src/components/library/DatasetLibrary.tsx:155-168` renders the dataset name with `truncate`
(single-line clip, `:156`) inside a `CappedGrid` (`:280`) that is now two cards per row. Real repo names differ
only by suffix — `sock_purple_green_orange` vs `sock_purple_green_orange_sort`,
`sock_2_only_merged` vs `sock_2_only_more_orange` — so **distinct datasets render as visually identical
cards**, and the disambiguating tail is exactly what the clip removes. The `title=` tooltip
(`:157`) is the only recovery, and it needs a hover.

Compounding clutter on the same card:

- the full `repo_id` is rendered directly beneath the name (`:163-168`) — the same string plus the namespace,
  also `truncate`d, so it clips at the same end;
- a footer **"Select"** button (`:170-191`) duplicates both the whole-card `onClick` (`:126`) and the
  top-right check mark (`:147-153`) — three affordances for one action.

Suggested direction (not applied): wrap/clamp on two lines, or middle-ellipsis so the suffix survives; drop
one of the three select affordances. UX severity — no data is at risk, but selecting the wrong dataset for a
training run is a real, silent misfire.

### N24 · P2 — The Launchpad library sheet lists datasets unfiltered, unsearchable and uncapped

**First identified:** 2026-07-24

`LibrarySheet.tsx:292` renders `myDatasets.map(...)` into a plain flex column: **no search box, no
local/hub filter, no `CappedGrid`** — the sheet imports none of them. The studio surface over the *same* data,
`DatasetLibraryList` (`DatasetLibrary.tsx:210-292`), has all three: a `query` state and `LibraryToolbar`
(`:216, :258-265`), a `local`/`hub` filter (`:217-226`) and `CappedGrid` two-row capping (`:280`). One data
source, two surfaces, inconsistent behaviour; the sheet degrades linearly with dataset count.

(`myDatasets` at `LibrarySheet.tsx:163-171` is an ownership filter, not a user-facing one.)

---

## Context: the `create_tag` / `codebase_version` gotcha — HANDLED on `main`

The documented failure mode (raw `HfApi.upload_folder` of a LeRobot dataset without a matching `create_tag`, so fresh downloads 404 on `meta/info.json` while cached copies mask it) **does not apply to any dataset path on `main`**:

- Every dataset publication goes through lerobot's `LeRobotDataset.push_to_hub`: `record.py:1116` (`UploadManager._worker`, the UI + cloud-chain path) and `hf_cloud.py:671` (`_ensure_dataset_on_hub`, the backend fallback). `record.py:1655` is the in-recorder push (unused — the frontend sends `push_to_hub: false`, `CollectPanel.tsx:246`).
- The installed lerobot's `push_to_hub` tags automatically (`lerobot/datasets/lerobot_dataset.py:620-623`): `tag_version` defaults to `True` and neither call site overrides it; the delete-then-create makes re-pushes idempotent.
- The only `upload_folder` calls in the repo are **model**-typed: `models.py:920` and `hf_cloud.py:346`. Neither touches a dataset repo.

**One adjacent caveat, P2, not the documented bug:** `set_dataset_tags` uses `metadata_update(...)` (`datasets.py:318`) which commits to the repo's `main` branch, but the `CODEBASE_VERSION` tag still points at the commit `push_to_hub` created. lerobot's `_download` resolves at `revision=self.revision` (the codebase-version tag), so a tag edit made in MakerMods Lab's "Visibility & tags" editor is **not visible to a fresh lerobot fetch**. Dataset-card metadata only (no data loss, repo-level visibility unaffected) — a discoverability wart rather than a correctness bug.

## Context: contradictions to the original brief (resolved during re-verification)

1. **`frontend/src/components/landing/DatasetsPanel.tsx` does not exist** on `main`. The dataset surfaces are `DatasetPicker.tsx` (now unreferenced dead code — see N4), `DatasetInfoCard.tsx`, `DatasetLibrary.tsx` (`components/library/`), `ManageCachesDialog.tsx`, `MergeDatasetsDialog.tsx`, `UploadDatasetDialog.tsx`, `DatasetDetailDialog.tsx` (`components/dialogs/`) and `CollectHandoff.tsx` / `CollectPanel.tsx` (`components/studio/`). That set was audited instead.
2. **The `Training.tsx:456` upload call site in Dataset #1 no longer exists.** `Training.tsx` on `main` is a 105-line router wrapper; the upload chain moved to `TrainingConfigurator.tsx:429-432`. The defect survived the move — same `private=false`.
3. **Dataset #1's "In progress / Progress (2026-07-14)" section describes andrew-branch work that is not on `main`.** No visibility toggle in `LocalDatasetCloudNotice.tsx`, no `uploadMakePublic` state, no `private=True` in `hf_cloud.py`, no `tests/test_runners_hf_cloud.py` visibility assertion. On `main` the P0 is not fixed at all, and `hf_cloud.py:669-670` documents a deliberate move in the opposite direction.
4. Dataset #1's framing "notice promises private while both upload paths publish public" is accurate for `main`; the backend carries an explicit **public-by-default policy comment**. *(2026-07-26: the policy is now confirmed intended, so this is a pure copy divergence — the notice is the side that is wrong. The record-flow auto-push originally logged here as a fourth public path (N1) is **descoped**, not a defect.)*
