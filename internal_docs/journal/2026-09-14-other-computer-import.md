# Other-computer source import

- Audited incoming flat files and nested copied project against current canonical files and common base `a031174e`: 213 non-cache files were identical, eight were stale base-only copies, and exactly five contained real changes. Each flat copy matched its nested counterpart. Coordinator removed the user-authorized untracked nested copy; flat originals remain preserved.
- Integrated only the five canonical deltas: preview requests MJPG before dimensions, recording shares its existing MJPG default with preview, and the selected gs_usb interface detaches an active Linux kernel driver before claiming it. Imported mocked tests cover format ordering and Linux detach ordering/skip cases. Applied only two lines to current `record.py`, preserving dataset v2.1 and replay changes.
- Incoming camera comments report measurements from another computer; those physical measurements have not been independently verified here. No hardware, services, experiments, or real datasets are authorized or used for this import.
- Validation pending: focused mocked pytest, Ruff, whitespace checks, and static OpenAPI regeneration to remove the incoming stale snapshot rollback while retaining current replay transport and dataset-format schemas. No worker staging, commit, push, or merge.

## Validation completed

- Repository `.venv` mocked regression checks passed: `tests/test_camera_preview.py`, `tests/test_record.py`, and `tests/test_gs_usb_transport.py`: 210 passed in 11.83 s (11 dependency/API deprecation warnings). No real capture or USB operations.
- Ruff lint and format checks passed for all five canonical files; `git diff --check` passed.
- Static OpenAPI exporter completed with temporary `MAKERMODSLAB_HOME`. Regenerated snapshot is identical to current HEAD, removing the incoming stale rollback; current dataset-format and replay-transport schema entries remain present. Export emitted duplicate AVFoundation class warnings from existing av/cv2 dependencies but exited successfully.
- Removed unverified measurement detail from the source comment, retaining the format/bandwidth mechanism. Current `record.py` diff remains exactly the two intended default-sharing edits. No frontend distribution changes were made by this worker.
- Implementation and checks complete; coordinator review and authorized commit pending. No worker staging, commit, push, merge, hardware, or service operation.
