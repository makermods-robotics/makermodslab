# Dataset Bug List — RE-VERIFIED against `main`

Re-audit of `internal_docs/bugs/Dataset Bug List.md` (originally verified **2026-07-14 against the
`andrew` branch**, commit `518ca56`) against **`main` @ `c7d9f27`** ("Merge pull request #4 from
makermods-robotics/redesign"), working tree as of **2026-07-23**.

Scope: `makermodslab/datasets.py`, `makermodslab/merge.py`, `makermodslab/record.py` (dataset paths),
`makermodslab/runners/hf_cloud.py`, dataset endpoints in `makermodslab/server.py`, and the dataset frontend
(`frontend/src/components/landing/*`, `frontend/src/components/library/DatasetLibrary.tsx`,
`frontend/src/components/studio/CollectHandoff.tsx`,
`frontend/src/components/training/**`, `frontend/src/hooks/useDataset*`, `frontend/src/lib/*`).

Analysis is **purely static**. No Hub call, no app run, no hardware. No code was modified.

## Verdict key

- **STILL REAL** — the defect reproduces in `main`'s code as written.
- **FIXED** — `main` no longer has the defect.
- **N/A-SUPERSEDED** — the code the entry described no longer exists in that form.
- **CANNOT VERIFY STATICALLY** — needs a runtime/Hub observation.

---

## Part 1 — Verdicts on the existing entries

| # | Entry | Verdict on `main` | Key evidence |
|---|---|---|---|
| 1 | Local dataset cloud-training notice says private, but upload is public | **STILL REAL — and the andrew-branch fix is NOT on `main`; `main` moved further the other way** | `LocalDatasetCloudNotice.tsx:77-83`; `TrainingConfigurator.tsx:432`; `hf_cloud.py:661,665-671` |
| 2 | Dataset in-use guard does not cover training jobs | **STILL REAL — unchanged** | `datasets.py:736-744` (`runner == "local"` at 741, exact `==` at 742, `limit=200` at 738); `record.py:865-869` |

### 1. Cloud-training notice vs. actual upload visibility — STILL REAL

**What `main` does today, at all three sites:**

1. **Notice text** — `frontend/src/components/training/config/LocalDatasetCloudNotice.tsx:77-83`:

   ```
   Hugging Face Cloud trains from the Hub, so {repoId} (~{size}) will be uploaded as a
   <span className="font-medium">private</span> dataset before training starts.
   ```

   The word `private` is a **hardcoded literal**. There is **no visibility toggle** in this component
   on `main` — the props interface (`:4-17`) has no visibility field at all. The
   `uploadMakePublic` / "Make this dataset public" toggle described in the andrew-branch progress note
   does not exist here.

2. **Frontend upload call** — the `Training.tsx` call site from the old audit is gone (the redesign
   reduced `Training.tsx` to a 105-line router wrapper). The upload chain now lives in
   `frontend/src/components/training/TrainingConfigurator.tsx:429-432`:

   ```tsx
   if (needsUpload) {
     setUploadError(null);
     setIsStarting(true);
     const err = await startUpload([], false /* public: MakerMods Lab uploads are public by default */);
   ```

   `startUpload(tags, isPrivate)` (`hooks/useDatasetUpload.ts:30,111-120`) → `POST /upload-dataset`
   with `private: false` (`lib/replayApi.ts:289`) → `record.py:1104`
   `dataset.push_to_hub(tags=tags, private=request.private)`. **Result: PUBLIC.**

3. **Backend fallback** — `makermodslab/runners/hf_cloud.py:641-676`, `_ensure_dataset_on_hub`:

   ```python
   self._log_line(f"[upload] dataset {repo_id} not on Hub; pushing local copy (public)...")
   ...
   # Public by default: MakerMods Lab's global policy is that datasets it pushes
   # to the Hub are public ... (This intentionally reverses the earlier private
   # default — an implicit upload of a local-only dataset is now public.)
   LeRobotDataset(repo_id).push_to_hub(tags=with_makermodslab_tag(None), private=False)
   ```

**Agreement analysis:** the two *upload paths* now agree with each other (both public, both
deliberately commented as such). The *notice text* is the one site still promising "private", and it
is the only thing the user reads before the upload fires. So the mismatch the entry describes still
exists, but its shape changed: it is no longer "one path forgot to be private", it is **"the code
committed to public-by-default and the UI copy was never updated"** — see `hf_cloud.py:669-670`,
which explicitly narrates the reversal.

Corroborating facts:

- The andrew-branch fix never merged. `git branch -a` shows
  `remotes/origin/fix/dataset-cloud-notice-public-mismatch` unmerged; PR #8 is open, not merged.
- The regression test the progress note claims (`tests/test_runners_hf_cloud.py` asserting
  `private=True`) is **not present** — `grep private tests/test_runners_hf_cloud.py` returns nothing.
- `git log --oneline -- makermodslab/runners/hf_cloud.py frontend/.../LocalDatasetCloudNotice.tsx` on
  `main` shows only redesign/rename commits after the original `9b95755` feature commit.

**The correct end-state (public vs private default) is an OPEN PRODUCT DECISION and is deliberately
not resolved here.** The options, stated neutrally:

- **(A) Private default** — change `TrainingConfigurator.tsx:432` to `true`, `hf_cloud.py:671` to
  `private=True`, keep the notice text. Contradicts the explicit policy comment at
  `hf_cloud.py:665-670` and the "discoverable MakerMods Lab datasets" rationale.
- **(B) Public default** — change the notice text at `LocalDatasetCloudNotice.tsx:81` from "private"
  to "public", ideally with the same camera-footage warning `UploadDatasetDialog.tsx:105` already
  uses. Keeps the stated policy; the risk moves to "did the user understand".
- **(C) Explicit choice** — add the toggle to the notice (what PR #8 targets), no default that can
  surprise.

Whichever is chosen, the acceptance criterion "notice, request, backend logs and the resulting repo
all agree" is currently **unmet on `main`**, and there is still **no single centralized publisher** —
`TrainingConfigurator.tsx:432`, `CollectHandoff.tsx:194` and `hf_cloud.py:671` each encode visibility
separately (see **N1** below, which adds a fourth, worse, path).

### 2. In-use guard does not cover training jobs — STILL REAL

`makermodslab/datasets.py:696-746` is byte-for-byte the behaviour the entry describes:

```python
for record in job_registry.list(limit=200):
    if (
        record.state == "running"
        and record.runner == "local"                       # datasets.py:741
        and record.config.dataset_repo_id == repo_id       # datasets.py:742
    ):
        return "A local training run is using this dataset. Stop it first."
```

- Cloud (`runner == "hf_cloud"`) runs remain excluded (`:741`).
- Comparison is still raw string equality — no repo-id normalization (`:742`).
- The registry scan is still capped (`job_registry.list(limit=200)`, `:738`); `JobRegistry.list`
  sorts by `started_at` descending and truncates (`jobs.py:1083-1089`), so a **long-running job that
  is older than the 200 most recent records is invisible to the guard**. That is a second, previously
  unnoted way this guard fails open.
- The delete path does reuse the full guard: `record.py:865-869`. So the "pre-guard build" caveat in
  the original entry is resolved — `main` has the guard, it is just incomplete.

Additional gap not in the original entry: `_dataset_in_use` protects the merge **output**
(`datasets.py:731-733`) but **not the merge sources** — see **N6**.

---

## Part 2 — New findings

### P0

#### N1. Recording auto-pushes the just-recorded dataset to the Hub as **PUBLIC**, on by default, with no visibility choice anywhere in the flow

The most consequential path is not the cloud-training one at all — it is the ordinary record flow.

- `frontend/src/contexts/StudioContext.tsx:42` — `pushToHub: true` is the **default** of the Collect
  form.
- `frontend/src/components/studio/RecordingForm.tsx:232-250` — the checkbox lives inside an
  **advanced Collapsible**. Its help text reads: *"Uploads the dataset to your Hugging Face account
  in the background once the session ends."* **Visibility is never mentioned.**
- `frontend/src/components/studio/CollectHandoff.tsx:109` — after the session ends the handoff banner
  renders `<UploadToHubAction autoStart={collectForm.pushToHub && repoId.includes("/")} />`.
- `frontend/src/components/studio/CollectHandoff.tsx:188-202` — the auto-push fires **on mount**,
  with no dialog and no confirmation:

  ```tsx
  useEffect(() => {
    if (!autoStart || autoPushed.has(repoId)) return;
    autoPushed.add(repoId);
    start([], false).then((error) => { ... });   // ← isPrivate = false → PUBLIC
  ```

  The component's own docstring (`:139`) states this plainly: *"the upload kicks off on mount (no
  tags, public — the dialog's own defaults)"*.

**Impact:** by default, every recorded SO-101 session — including all camera footage — is published
**publicly** to the user's Hub namespace, automatically, with the only opt-out being a default-on
checkbox behind an "advanced" disclosure whose copy says nothing about visibility. The user is never
shown a Public/Private choice on this path. This is the same failure class as Dataset #1 (silent
public publication of camera footage) but on the highest-traffic path, and with **no** on-screen
claim at all rather than a wrong one.

`record.py` itself does not push (`CollectPanel.tsx:246` sends `push_to_hub: false`), so this is
purely the frontend auto-push. Fixing it is one line + copy, and it must be resolved together with
Dataset #1 so all four visibility sites land on one policy.

**Evidence:** `StudioContext.tsx:42`, `RecordingForm.tsx:232-250`, `CollectHandoff.tsx:109,139,194`,
`useDatasetUpload.ts:112-120`, `record.py:1104`.

### P1

#### N2. A failed/aborted Hub download can leave a **partial dataset that passes every "is it complete?" check**

`makermodslab/datasets.py:1028-1033`:

```python
def _cleanup_partial_dataset(repo_id: str) -> None:
    target = _lerobot_cache_root() / repo_id
    if target.exists() and not _is_dataset_dir(target):
        shutil.rmtree(target, ignore_errors=True)
```

`_is_dataset_dir` is `(<dir>/meta/info.json).is_file()` (`datasets.py:333-338`). `snapshot_download`
(`datasets.py:1020`, `local_dir=` flat layout) fetches many files concurrently; `meta/info.json` is a
few hundred bytes and will typically land early. If the download dies **after** `meta/info.json` but
**before** the parquet/video payload:

- the cleanup guard sees a "valid" dataset dir and **skips the rmtree**;
- `is_dataset_available_locally` returns True (`datasets.py:368-369`);
- `list_local_datasets` lists it, because `_dataset_has_episodes` reads `total_episodes` out of the
  **Hub's** `info.json` and gets a non-zero value (`datasets.py:397-406,442-453`);
- `get_hub_status` will happily answer `on_hub`/`local_only`; the info card renders full episode/frame
  counts from the truncated metadata.

The user sees a complete-looking local dataset. The failure surfaces much later — in training or in a
merge — as a `FileNotFoundError` on a missing data file. There is no "re-download" repair path
either: `datasets_download` → `DownloadManager.start` will just `snapshot_download` over the top,
which may or may not repair it.

The same class of residue is produced by the merge subprocess's `_ensure_local_source`
(`merge.py:576-592`) — it `snapshot_download`s a Hub source into the flat cache and has **no cleanup
at all** on failure; `_run_cli` only cleans the *output* (`merge.py:619-637`).

**Suggested direction (not applied):** make the cleanup unconditional for a download this manager
started (mirror `_run_cli`'s `output_pre_existed` pattern), or verify the snapshot against
`info.json`'s file manifest before declaring success.

**Evidence:** `datasets.py:1010-1036`, `datasets.py:333-338`, `datasets.py:397-406`,
`merge.py:576-592`, `merge.py:619-637`.

#### N3. Local dataset delete does **not** invalidate the Hub-status cache — and the two sibling discard paths do

`makermodslab/record.py:874-885` (`handle_delete_dataset`):

```python
shutil.rmtree(target)
...
invalidate_dataset_listing_cache()          # record.py:882
logger.info(f"Deleted dataset directory {target}")
```

Compare its two siblings, which get it right:

- `_discard_empty_dataset` — `record.py:948-949`: `invalidate_hub_status(repo_id)` **and**
  `invalidate_dataset_listing_cache()`
- `_discard_session_dataset` — `record.py:1003-1004`: same pair
- `rename_local_dataset` — `datasets.py:800-802`: invalidates hub-status for **both** ids

`_HUB_STATUS_CACHE` memoizes `local_only` **for the process lifetime** (`datasets.py:213-217`). So
after the user deletes a local-only dataset, `/datasets/hub-status` keeps answering `local_only`
until the server restarts. Downstream consequences:

- `DatasetInfoCard`'s `NotDownloadedView` takes the `local_only` branch and tells the user
  *"This dataset is on this machine, but its details couldn't be read — the local copy looks
  incomplete or corrupt. Re-record or re-download it."* (`DatasetInfoCard.tsx:834-846`) for a dataset
  that simply **does not exist**, and offers an "Upload to Hub" button for it.
- Pressing that button runs `LeRobotDataset(repo_id)` (`record.py:1099`) with no `root=`, which for a
  non-local repo id **downloads the whole dataset from the Hub** before "uploading" it. On a deleted
  local-only dataset it just fails, confusingly.
- `jobs.py:1119-1120`'s cloud gate keeps refusing cloud runs on the (now absent) id.

One-line asymmetry; the fix is to add `invalidate_hub_status(repo_id)` next to `record.py:882`.

#### N4. On `main` there is **no UI at all** to delete a local-only dataset

- The only dataset-delete entry point wired into the app is `ManageCachesDialog`
  (`LibrarySheet.tsx:29,366,435`), and it filters to **`source === "both"` only**
  (`ManageCachesDialog.tsx:44`: `datasets.filter((d) => d.source === "both")`).
- `DatasetPicker.tsx` — which carries the per-row trash affordance (`:95-120`) — has **no importer**
  anywhere in `frontend/src` (verified by grep). It is dead code after the redesign.
- `DatasetInfoCard`'s `canDelete` / `onDelete` props (`:886-891`) are never passed: its only consumer
  is `DatasetDetailDialog.tsx:12`, which supplies neither.
- Consequently `resolveDeleteAction`'s `"delete-local"` branch (`lib/deleteSemantics.ts:88-94`) is
  unreachable for datasets, and its `LOCAL_DELETE_DESCRIPTION` copy (`:39-46`) is dead.

**Impact:** a recorded dataset that was never pushed to the Hub (i.e. the user unchecked the
default-on `pushToHub`) can only be removed by hand from `~/.cache/huggingface/lerobot`. Multi-GB
recordings accumulate with no in-app cleanup. `POST /delete-dataset` still works — this is purely a
missing frontend surface.

This interacts badly with **N1**: the *only* datasets the app can clean up are the ones it has
already published.

#### N5. "Remove local copy" of a `both` dataset can silently discard episodes the Hub copy does not have

`lib/deleteSemantics.ts:54-64` promises: *"This removes the local copy from disk — the Hub copy
stays, and it remains listed as a Hub dataset."* `ManageCachesDialog`'s header repeats it
(*"The Hub copy stays — clearing only removes the local copy"*), and the action is a hard
`shutil.rmtree` via `POST /delete-dataset` (`ManageCachesDialog.tsx:97` → `record.py:875`).

But `source: "both"` is decided purely by **repo-id set intersection** — `list_all_datasets`
(`datasets.py:870-883`) marks a row `both` when the same id appears in the Hub listing and in the
local scan. It never compares content, episode counts, or timestamps. So:

> record `user/pick` (10 eps) → upload → resume/record 10 more episodes locally (local = 20 eps, Hub =
> 10) → "Manage cached datasets" → "Clear" → **10 episodes are permanently gone**, under a UI promise
> that the Hub copy is equivalent.

`ManageCachesDialog` fetches `/datasets/info` per row for the **size** (`:71-83`) but never compares
`total_episodes` against the Hub's. `get_hub_dataset_info` (`datasets.py:636-682`) already fetches the
Hub's `meta/info.json` and would give the comparison cheaply.

**Suggested direction (not applied):** compare local vs Hub `total_episodes` before offering
"Clear", and downgrade the copy to a warning when the local copy is ahead.

### P2

#### N6. `_dataset_in_use` does not protect merge **sources**

`datasets.py:731-733` only matches `merge_manager.output_repo_id`. A source dataset can therefore be
deleted or renamed while a merge is reading it. The merge then fails partway; `_run_cli` cleans the
partial *output* (`merge.py:635-636`) and `_cli_friendly_error` produces a "looks incomplete or
corrupt" message that **blames the wrong thing** (`merge.py:498-507`) — it says the source is corrupt
when the user deleted it. `MergeRequest.source_repo_ids` is available on the manager, so the guard
extension is mechanical.

#### N7. Merge completion never invalidates the dataset listing cache

`makermodslab/merge.py` contains **zero** calls to `invalidate_dataset_listing_cache` /
`invalidate_hub_status` (verified by grep across `makermodslab/`). `_LISTING_CACHE_TTL_S = 45.0`
(`datasets.py:97`), and `MergeDatasetsDialog`'s `onMerged` → `refreshDatasets` (`LibrarySheet.tsx:402`)
just re-GETs `/datasets`, which returns the cached pre-merge list. The merge output can be invisible
for up to 45 s after the dialog says "Created …". The listing-cache docstring
(`datasets.py:102-106`) claims *"Called after any mutation that changes the listing"* — merge is the
counterexample. The merge subprocess's `_ensure_local_source` downloads (`merge.py:576-592`) have the
same problem.

#### N8. A completed recording session never invalidates the dataset listing cache either

Same 45 s window: `record.py` invalidates on delete (`:882`), discard (`:948-949`, `:1003-1004`) and
upload (`:1112-1114`), but there is no invalidation on the **success** path of a recording session.
The just-recorded dataset may not appear in the library for up to 45 s. (`CollectHandoff` masks this
by preselecting the repo id directly, so it only bites the library/picker views.)

#### N9. Logging in does not invalidate the dataset listing cache or the Hub-status cache

`handle_hf_login` (`utils/hf_auth.py:129-152`) calls `invalidate_whoami_cache()` and nothing else.
Two consequences:

- `/datasets` keeps returning the pre-login (Hub-less) listing for up to 45 s
  (`datasets.py:862-865`).
- Worse and unbounded: `get_hub_status` caches `local_only` **for the process lifetime**
  (`datasets.py:213-217`). `HfApi.repo_exists` returns `False` for a **private** repo when no token is
  present (huggingface_hub returns False on `RepositoryNotFoundError`, which is what a 401 becomes).
  So any dataset whose status was probed while logged out is permanently pinned to "Local only", even
  though it exists privately on the Hub. The info card then offers "Upload to Hub"
  (`DatasetInfoCard.tsx:581-595`) and the upload overwrites the existing private repo's contents.
  (`push_to_hub` uses `create_repo(..., exist_ok=True)`, which does **not** flip an existing repo's
  visibility — so this is content clobbering, not a visibility leak.)

`handle_hf_login` should call `invalidate_dataset_listing_cache()` and clear `_HUB_STATUS_CACHE`.

#### N10. `get_hub_status`'s `on_hub` answer is cached for the process lifetime and is load-bearing for the cloud gate

`datasets.py:213-217` memoizes `on_hub` forever. `jobs.py:1116-1120` uses it as the *only* gate
preventing a cloud job on a local-only dataset. If the repo is deleted on the Hub (or moved) after
being cached, the gate passes and control lands in `_ensure_dataset_on_hub`, which pushes the local
copy **publicly** (`hf_cloud.py:671`). This is a second, cache-driven route into the N1/#1 visibility
problem. Nothing outside MakerMods Lab invalidates the entry.

#### N11. `_ensure_dataset_on_hub` catches only `RepositoryNotFoundError`

`hf_cloud.py:649-653`:

```python
try:
    self._api.dataset_info(repo_id)
    return
except RepositoryNotFoundError:
    pass
```

A transient transport error, a rate limit, or a 403 on a repo the token can read but not `dataset_info`
propagates out of `start()` and fails the job with a raw exception, even though the dataset is fine.
The `jobs.py` gate comment (`:1111-1113`) explicitly defers the "unknown" (offline) case to this
fallback, which then cannot handle it.

#### N12. The Hub-upload path "uploads" datasets it may first have to download

`record.py:1099` does `LeRobotDataset(repo_id)` with no `root=`. For a repo id that is **not** in the
local flat cache, lerobot's constructor downloads the dataset from the Hub before the "upload" runs
(`lerobot/datasets/lerobot_dataset.py:634-655`). `DatasetInfoCard.tsx:579-580` deliberately offers the
Upload button for `status === "unknown"` ("the endpoint is a safe upsert"), so a user behind a flaky
link can trigger a multi-GB download by pressing *Upload*. Progress is reported as "Uploading…" the
whole time.

#### N13. `UploadManager` / delete / rename share no lock — TOCTOU on the in-use guard

`handle_delete_dataset` checks `_dataset_in_use` (`record.py:867`) and then `rmtree`s
(`record.py:875`) with nothing held. `UploadManager.start` checks the same guard under **its own**
`self._lock` (`record.py:1053-1068`). A delete that passes the guard microseconds before an upload
starts will rmtree the directory the upload is about to read. Low probability, but the guard's whole
purpose is to prevent exactly this.

#### N14. `_dataset_in_use` does a filesystem scan per registry record on every delete/rename/upload

`JobRegistry.list` calls `_count_checkpoints(r)` for each returned record (`jobs.py:1087-1089`), and
`_dataset_in_use` asks for `limit=200` (`datasets.py:738`). Every delete, rename, and upload start
therefore walks up to 200 checkpoint trees. Cosmetic today; it will not stay cosmetic.

#### N15. `MergeManager.get_status` drains the log queue destructively

`merge.py:381-392` drains `log_queue` on every poll. Two clients polling `/datasets/merge/status`
(two tabs, or the dialog plus a stray poller) split the log lines between them; each sees a
partial log. The persisted `merge_logs/<ts>.log` (`merge.py:428-440`) is the only complete record.

#### N16. `useDatasets` wipes the list to empty on any fetch failure

`frontend/src/hooks/useDatasets.ts:14`: `.catch(() => setDatasets([]))`. A single transient
`/datasets` failure makes the library render *"No datasets yet. Record your first one above."*
(`DatasetLibrary.tsx:244-255`) — indistinguishable from actual data loss. The hook has no error state
at all.

#### N17. Per-author Hub listing is silently truncated at 200

`datasets.py:824`: `api.list_datasets(author=author, limit=200)`. A user (or org) with more than 200
datasets gets a silently short list with no indication. Same cap, undisclosed, in `models.py`.

#### N18. Stale docstrings claim "private by default" where the code is public by default

- `UploadDatasetDialog.tsx:17-18` — *"Private-by-default toggle (with the camera-footage note)"*,
  while `:38` is `useState(false)` → **Public** is the preselected side of `VisibilityToggle`
  (`VisibilityToggle.tsx:12-13`: `value` is the *private* flag).
- `DatasetInfoCard.tsx:458` repeats *"a confirm popover (private-by-default toggle + optional tags)"*.

The rendered dialog is **honest** — with Public selected it says *"Anyone can see this dataset —
recordings include your camera footage."* (`UploadDatasetDialog.tsx:103-106`) — so this is a comment
bug, not a user-facing one. Flagged because it is exactly the kind of stale comment that makes a
future reviewer conclude the visibility problem is already fixed.

#### N19. Pinned custom Hub datasets always render as public in the library

`datasets.py:896-902` seeds a pinned row with `"private": False` and the listing never backfills it
(`/datasets/hub-status` returns only existence, not visibility). `DatasetLibrary.tsx:137-145` gates
the lock chip on `item.private`, so a **private** pinned dataset displays with no privacy indicator.
Cosmetic, but it is a visibility signal that reads as authoritative.

---

## The `create_tag` / `codebase_version` gotcha — HANDLED on `main`

The documented failure mode (raw `HfApi.upload_folder` of a LeRobot dataset without a matching
`create_tag`, so fresh downloads 404 on `meta/info.json` while cached copies mask it) **does not apply
to any dataset path on `main`**:

- Every dataset publication goes through lerobot's `LeRobotDataset.push_to_hub`:
  `record.py:1104` (`UploadManager._worker`, the UI + cloud-chain path) and
  `hf_cloud.py:671` (`_ensure_dataset_on_hub`, the backend fallback). `record.py:1637` is the
  in-recorder push (unused — the frontend sends `push_to_hub: false`, `CollectPanel.tsx:246`).
- The installed lerobot's `push_to_hub` tags automatically
  (`.venv/lib/python3.12/site-packages/lerobot/datasets/lerobot_dataset.py:620-623`):

  ```python
  if tag_version:
      with contextlib.suppress(RevisionNotFoundError):
          hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
      hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")
  ```

  `tag_version` defaults to `True` and neither call site overrides it. The delete-then-create makes
  re-pushes idempotent.
- The only `upload_folder` calls in the repo are **model**-typed: `models.py:920` and
  `hf_cloud.py:346` (the in-container checkpoint watcher, `repo_type="model"`). Neither touches a
  dataset repo.

**One adjacent caveat, P2, not the documented bug:** `set_dataset_tags` uses `metadata_update(...)`
(`datasets.py:318`) and `set_dataset_visibility` uses `update_repo_settings` (`datasets.py:294`).
`metadata_update` commits to the repo's `main` branch, but the `CODEBASE_VERSION` tag still points at
the commit `push_to_hub` created. lerobot's `_download` resolves at `revision=self.revision`
(the codebase-version tag), so a tag edit made in MakerMods Lab's "Visibility & tags" editor is **not
visible to a fresh lerobot fetch**. This affects dataset-card metadata only (no data loss, and
repo-level visibility is unaffected), so it is a discoverability wart rather than a correctness bug.

---

## (c) Contradictions to the brief

1. **`frontend/src/components/landing/DatasetsPanel.tsx` does not exist** on `main`. The dataset
   surfaces are `DatasetPicker.tsx` (now unreferenced dead code — see N4),
   `DatasetInfoCard.tsx`, `DatasetLibrary.tsx` (`components/library/`),
   `ManageCachesDialog.tsx`, `MergeDatasetsDialog.tsx`, `UploadDatasetDialog.tsx`,
   `DatasetDetailDialog.tsx` (`components/dialogs/`) and `CollectHandoff.tsx` /
   `CollectPanel.tsx` (`components/studio/`). I audited that set instead.
2. **The `Training.tsx:456` upload call site in Dataset #1 no longer exists.** `Training.tsx` on
   `main` is a 105-line router wrapper; the upload chain moved to
   `TrainingConfigurator.tsx:429-432`. The defect survived the move — same `private=false`.
3. **Dataset #1's "In progress / Progress (2026-07-14)" section describes work that is not on
   `main`.** No visibility toggle in `LocalDatasetCloudNotice.tsx`, no `uploadMakePublic` state, no
   `private=True` in `hf_cloud.py`, no `tests/test_runners_hf_cloud.py` visibility assertion. Anyone
   reading the ledger would conclude the P0 is half-fixed; on `main` it is not fixed at all, and
   `hf_cloud.py:669-670` documents a *deliberate move in the opposite direction*. The ledger's
   `andrew`-branch scoping needs to be stated at the top of the file, not inferred.
4. The brief frames Dataset #1 as "notice promises private while both upload paths publish public".
   That is accurate for `main`, but the framing understates it: the backend now carries an explicit
   **public-by-default policy comment**, so this is a policy/copy divergence rather than a one-sided
   oversight — and the record-flow auto-push (N1) is a fourth, unconfirmed, public path the entry
   never covered.

## (d) What I could not verify, and why

- **Whether a partial `snapshot_download` actually lands `meta/info.json` first (N2).** The ordering
  depends on huggingface_hub's thread-pool scheduling of the repo file list. The *guard logic* is
  unambiguously wrong (`_is_dataset_dir` cannot distinguish "complete" from "has metadata"), but the
  hit rate needs a live interrupted download. **Static verdict: real; frequency unknown.**
- **The live Hub-side outcome of any upload.** Per the safety rules I made no Hub call, so
  "the resulting repo is public" is inferred from `private=False` reaching
  `LeRobotDataset.push_to_hub` → `create_repo(private=False)`. Confirming the repo's actual
  visibility needs a Hub read.
- **N5's real-world frequency.** Whether users routinely resume-record into an already-uploaded
  dataset is a usage question. The code path is unambiguous; the exposure is not.
- **N10's staleness window.** Requires a running server plus an out-of-band Hub deletion.
- **Whether `DatasetPicker.tsx` is intentionally-retained-for-reinstatement vs. accidentally
  orphaned (N4).** Grep proves it has no importer; intent is a coordinator/author question. Per the
  steelman rule I am flagging the *user-visible consequence* (no local-delete UI), not asserting the
  file is a mistake.
- **`record.py:1637`'s in-recorder `push_to_hub`.** Reachable only if some caller sends
  `push_to_hub: true`; the only frontend caller sends `false` (`CollectPanel.tsx:246`). A non-UI
  caller could reach it — it honours `cfg.dataset.private`, which defaults to `False`
  (`record.py:287,300`), i.e. public. Not counted as a live path.
- **Ranking calibration against the other six agents' lists.** I ranked within the dataset domain
  only; cross-domain P0 ordering is the coordinator's call.
