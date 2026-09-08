# Dataset Interface Pathways

This document inventories how people can acquire, inspect, use, publish, and remove datasets through MakerMods Lab. It separates the pathways implemented in the current interface from important missing or proposed pathways.

This inventory was audited on July 14, 2026, against MakerMods Lab commit `518ca56`.

## Core dataset state model

MakerMods Lab treats a dataset as being in one of three storage states. The state determines which actions the interface offers.

| State | Meaning | Principal actions |
| --- | --- | --- |
| **Local only** | A complete dataset exists in the local LeRobot cache but is not confirmed on the Hugging Face Hub. | Inspect full metadata, record more episodes, rename locally, train locally, upload, merge, or delete permanently. A cloud-training request uploads it first. |
| **Hub only** | The dataset exists on the Hub but has no complete local copy. | Inspect the available Hub summary, train from the Hub, download it, edit Hub settings when authorized, or remove it from the local listing. Recording more episodes requires a download first. |
| **Both** | A local copy and a Hub copy share the same repository ID. | Use all applicable local and Hub actions, or clear only the local copy while retaining the Hub dataset. |

Dataset selection on the landing page is persistent and acts as the application's dataset context. The training and fine-tuning flows read that selection.

## Dataset acquisition pathways

### 1. Record a new dataset

Entry point: **Add dataset -> Record a dataset**.

1. The user names the dataset.
2. MakerMods Lab opens the recording configuration with the selected robot and its saved cameras.
3. The user supplies the task, number and duration of episodes, reset time, and encoding settings.
4. Recording creates a local dataset. When the user is logged into Hugging Face, the new ID is namespaced with the username; otherwise MakerMods Lab permits a bare local name.
5. After recording, the result is saved locally and preselected on the home page.
6. The user may inspect it, upload it, record more episodes, or begin training.

During a recording session, the user can complete an episode, advance early, re-record the current episode, finish with completed episodes, or quit and discard the in-progress work. An empty fresh session is removed automatically. If teardown fails after episodes were saved, MakerMods Lab distinguishes that warning from a failed recording and lets the user keep or discard the saved result.

### 2. Record more episodes into an existing dataset

Entry point: select a local dataset and click **Record more episodes**.

MakerMods Lab reuses the existing repository ID verbatim and resumes the on-disk dataset. The interface warns about likely differences in robot type, cameras, and FPS; LeRobot's compatibility check is the final gate. A Hub-only dataset must first be downloaded because resume recording mutates a local directory.

### 3. Add a Hub dataset without downloading it

Entry point: **Add dataset -> Add from Hugging Face**.

The user enters a `namespace/name` repository ID and leaves **Download to this machine now** disabled. MakerMods Lab pins and selects the repository so it persists in the dataset list. It remains Hub-only, and training can fetch it from the Hub on demand.

This pathway is useful when the user wants cloud training or does not need local episode-level data.

### 4. Add and download a Hub dataset

Entry point: **Add dataset -> Add from Hugging Face**, with **Download to this machine now** enabled.

MakerMods Lab pins the repository, starts a background download, and stores the result in the flat LeRobot cache layout. The download survives navigation through status polling. When complete, the dataset state changes from Hub-only to Both.

A Hub-only dataset can also be downloaded later from its information card.

### 5. Automatically discover owned Hub datasets

When the user is logged into Hugging Face, MakerMods Lab lists compatible datasets in the user's namespace and their organizations. The listing is merged with local datasets, pinned external datasets, and hidden-list preferences.

Automatic discovery is filtered using the Hub's `LeRobot` metadata tag. A valid LeRobot dataset without this searchable metadata tag will not appear automatically, although it can still be added explicitly by repository ID.

### 6. Import a dataset from disk

Entry point: **Add dataset -> Import from disk**.

The user provides a server-local folder containing a LeRobot dataset. MakerMods Lab validates it and copies it into the managed LeRobot cache; this is an import, not an external-path reference. The user may keep the folder's name or provide a new repository ID.

### 7. Create a dataset by merging existing datasets

Entry point: **Merge datasets...**.

The user chooses two or more sources and an output ID. MakerMods Lab can download Hub-only sources, then checks source availability and compatibility. Sources are expected to share the same FPS, cameras, feature keys, and feature shapes. The merge runs in the background, exposes logs, prevents output overwrite, and cleans up a partial output after failure.

The result is a new local dataset. It is not automatically uploaded.

## Browsing and inspection pathways

The landing-page dataset picker provides:

- Search by full repository ID.
- Separate Hugging Face and local sections.
- Namespace-aware sorting that prioritizes the logged-in user's datasets.
- Persistent selection for downstream training.
- Removal controls whose meaning depends on dataset state.

The selected dataset's information card can show:

- Episode and frame counts.
- Duration and FPS.
- Camera feature names.
- Robot type.
- Tasks and per-task episode counts.
- Local disk size.
- Local and Hub availability.
- A direct link to the Hub repository.

The card warns about an empty dataset or a local dataset without camera data. It also distinguishes a genuine Hub-only dataset, an incomplete or corrupt local copy, and a stale selection that no longer exists locally or on the Hub.

## Hub publishing and management pathways

### Upload a local dataset

A local dataset's information card offers **Upload to Hub**. Before upload, the user can choose public or private visibility and add optional tags. Upload runs in the background and reports completion or an actionable error.

For a local-only dataset selected for Hugging Face Jobs training, MakerMods Lab can chain this automatically: upload first, then start the cloud job.

### Edit Hub settings

When the authenticated user can write to the dataset namespace, the information card allows them to:

- Change public/private visibility.
- Replace the dataset-card tags.
- Open the dataset on Hugging Face.

MakerMods Lab preserves its required product tags during a tag edit.

### Pin, hide, and unpin Hub datasets

- An explicitly entered external repository is **pinned** into the list.
- Removing a pinned Hub-only dataset **unpins** it.
- Removing an automatically discovered Hub-only dataset **hides** it locally.
- Re-adding a hidden repository ID unhides it.

These actions do not delete or mutate the Hub repository.

### Hub operations intentionally not performed

The dataset interface does not delete Hub repositories. A local rename also does not rename the Hub repository; it only moves the local directory and changes the local ID.

## Training pathways

### Local dataset to local training

The selected dataset is passed directly to the local LeRobot trainer. Local training requires the dataset to be usable from the local machine. Dataset mutation is blocked while local training is using it.

### Hub dataset to local training

A Hub-only selection may be passed to LeRobot, which downloads the required snapshot when training starts. Explicitly downloading first gives the user local details and avoids repeating network work.

### Hub dataset to Hugging Face Jobs

The remote job receives the repository ID and fetches the dataset from the Hub. A local download does not get transferred to the cloud container; the Hub copy remains the source of truth for the job.

### Local-only dataset to Hugging Face Jobs

MakerMods Lab detects that the remote job cannot access the host cache. The Start action becomes an upload-and-start flow: upload the dataset, wait for success, then launch training.

#### Intended upload-and-start behavior

A cloud-training request must treat the upload as an explicit publishing transaction, not as an invisible implementation detail. The intended sequence is:

1. Validate the local dataset, including `meta/info.json`, its `codebase_version`, episode metadata, and required data files.
2. Resolve a complete Hub repository ID. A bare local name must be mapped to a namespace the authenticated user can write to.
3. Show the destination, approximate upload size, and visibility. Visibility must be an explicit choice and default to private because selecting cloud training does not by itself express an intent to publish the data publicly.
4. Add the locked Hub metadata tags `LeRobot`, `makermods`, `openbooth`, and `MakerMods Lab`, while preserving optional user tags.
5. Upload through the same centralized publisher used by the dataset information card.
6. After the files and dataset card have uploaded successfully, create or update the Git revision matching `meta/info.json`, currently `v3.0`.
7. Verify that the remote `meta/info.json` and dataset files can be resolved through that revision.
8. Resolve the uploaded commit SHA and launch the job against that immutable commit, rather than relying only on the movable `v3.0` tag.
9. Store the Hub repository ID and exact dataset commit with the job record for reproducibility.

The job must not start if validation, upload, revision creation, or remote verification fails. Dataset tags are assigned during the upload; starting the compute job should not mutate dataset metadata again.

If a repository with the same ID already exists, existence alone is insufficient evidence that it matches the local copy. MakerMods Lab should distinguish:

- **Synchronized:** launch against the verified existing commit.
- **Local changes not uploaded:** offer to synchronize the local copy or deliberately use the existing Hub copy.
- **Hub changes not downloaded:** offer to download, use the Hub copy, or publish the local dataset under a new ID.
- **Unknown relationship:** require an explicit choice rather than silently preferring one side.

An automatically uploaded training dataset should remain on the Hub after the job so the run remains reproducible. A separately labeled temporary-upload workflow may offer later cleanup, but deletion must be explicit and must explain that it removes part of the run's provenance.

### Dataset selection for fine-tuning

Fine-tuning an imported or existing model is a fresh training run initialized from model weights. It uses the currently selected dataset, just like a normal new training run.

Once a job starts, its dataset ID, policy, and training configuration are stored with the job record rather than following later landing-page selection changes.

## Storage, rename, and removal pathways

The trash action never deletes a Hub repository. Its effect is state-dependent:

| Dataset state | Removal behavior | Selection afterward |
| --- | --- | --- |
| Local only | Permanently deletes the local dataset, including episodes and videos. | Cleared. |
| Both | Removes only the local copy. The Hub row remains. | Preserved. |
| Hub only, explicitly pinned | Unpins it from MakerMods Lab. | Cleared. |
| Hub only, automatically discovered | Hides it from MakerMods Lab's listing. | Cleared. |

The **Manage cached datasets** dialog lists datasets in the Both state, shows their local sizes when available, and can clear one or all local copies while retaining the Hub versions. It warns when offline mode would prevent a subsequent re-download.

A local dataset can be renamed when no active operation is using it. Only the final name segment is editable; an existing namespace prefix is retained. The Hub copy, if any, is unaffected.

Recording, upload, merge, and local training protect datasets they are actively using from conflicting mutation.

## Availability and error-recovery pathways

The interface distinguishes several failure states:

- **Hub-only:** offer a download and explain that training can fetch on demand.
- **Local but unreadable:** identify the copy as incomplete or corrupt and suggest re-recording or re-downloading.
- **Absent:** explain that the repository may have been deleted or renamed.
- **Hub unreachable or authentication unknown:** preserve useful local behavior and surface Hub actions cautiously.
- **Merge incompatibility:** identify FPS, camera, feature, or missing-file differences.
- **Offline mode:** prevent Hub-only actions that cannot work and warn before clearing recoverable caches.

## Current missing or incomplete pathways

The following interactions are important candidates but are not complete in the current interface.

### Dataset version and revision management

MakerMods Lab does not currently preflight or repair the Git revision tag that LeRobot uses to select a compatible dataset snapshot. A Hub metadata tag such as `LeRobot` affects search, while a Git tag such as `v3.0` names the versioned snapshot; these are different mechanisms.

The desired pathway is:

1. Read `meta/info.json` and its `codebase_version`.
2. Compare it with MakerMods Lab's pinned LeRobot dataset format.
3. Confirm that the matching Hub Git tag exists and resolves.
4. Offer to create or repair the tag when the user has write permission.
5. Block training with a clear explanation when formats are incompatible.

The interface also does not let the user choose a branch, tag, or commit for training, nor does it show the exact dataset revision recorded by a run.

For cloud training, the desired revision model has two layers: the format tag such as `v3.0` proves which LeRobot schema the repository exposes, while an immutable commit SHA records the exact episodes used by one job. Multiple jobs may therefore share the `v3.0` format while deliberately recording different commit SHAs.

### Dataset migration

There is no guided conversion between LeRobot dataset formats or releases. A future flow should detect the source format, explain whether the current pin can read it, make a non-destructive converted copy, validate it, and optionally publish it under a new revision or repository.

### Episode review and editing

There is no first-class UI to:

- Browse and play individual episodes.
- Scrub synchronized camera streams and action/state traces.
- Mark episodes good or bad.
- Delete or trim an episode.
- Change task labels.
- Repair metadata or recompute statistics.
- Compare episodes or datasets.

The existing `/edit-dataset` page is only an under-construction placeholder.

### Replay discrepancy

The README and internal feature inventory advertise episode replay, but the current React source has no replay route or visible replay control. The backend only contains a stale comment saying replay is rendered by an embedded `lerobot/visualize_dataset` Space. The product should either restore a concrete replay pathway or correct the documentation.

### Clone, fork, and export

There is no interface for:

- Cloning a dataset under a new local name.
- Forking a Hub dataset into the user's namespace.
- Exporting a dataset as an archive.
- Pushing a local dataset to a differently named Hub repository.
- Managing dataset lineage between source, merged, converted, and cleaned derivatives.

### Hub repository lifecycle

The current safety rule intentionally avoids Hub deletion and rename. If those actions are ever added, they should be separately labeled, permission-gated, and strongly confirmed rather than sharing the local trash control.

## Example: `makermods/cube_grab`

After being downloaded, `makermods/cube_grab` is in the Both state. A user can:

- Select and inspect its 31 episodes, 14,604 frames, two camera streams, and seven-dimensional actions.
- Train locally using the downloaded files.
- Train on Hugging Face Jobs using the Hub repository.
- Include it in a compatible dataset merge.
- Edit its Hub visibility or metadata tags when authorized.
- Clear its local copy while keeping the Hub version.

The dataset declares LeRobot format `v3.0`, which MakerMods Lab's pinned LeRobot 0.5.2 can load. Cloud loading also requires the Hub repository's `v3.0` Git tag; that revision tag is distinct from the searchable `LeRobot`, `MakerMods Lab`, `makermods`, and `openbooth` metadata tags.

Although the information card can offer **Record more episodes** for any local dataset, resuming this dataset should only proceed when the selected robot, camera names, FPS, and feature layout pass the compatibility checks. Its recorded robot type is `metal_follower`, so a normal MakerMods Lab SO-101 profile may not be compatible without an intentional migration or matching hardware configuration.

## Product model

The intended dataset lifecycle can be summarized as:

> acquire -> validate -> inspect -> improve -> publish -> train -> retain or clean up

MakerMods Lab already covers most acquisition, publishing, training, and storage-management steps. The largest remaining gap is the middle of the lifecycle: explicit validation, version management, episode review, editing, and replay.
