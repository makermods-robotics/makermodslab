# Copyright 2026 RLSOK contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export a selected Metal record and calibration copies for offline review.

Run this file directly, with Python's standard library only. Do not import
utils.config: importing the application may migrate state, and opening a
Metal bus even to probe it can energize a motor. This exporter never imports
MakerModsLab/LeRobot, opens hardware, starts a session or contacts RLSOK.

Output is an optional RLSOK saved-configuration manifest, not execution
authorization. The files can also be reviewed without RLSOK. Only explicit
Metal records with a Star or Metal leader are supported initially; unknown or
implicit variants are refused instead of guessed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

MAX_BYTES = 8 * 1024 * 1024
SOURCE_FILES = (
    "makermodslab/utils/config.py",
    "makermodslab/utils/robot_factory.py",
    "makermodslab/arms/metal.py",
    "makermodslab/arms/can_common.py",
    "makermodslab/arms/registry.py",
    "makermodslab/maker_can.py",
    "makermodslab/gs_usb_transport.py",
    "pyproject.toml",
)


def _bytes(path: Path) -> bytes:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Expected a regular, non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_BYTES:
            raise ValueError(f"Expected a regular file no larger than 8 MiB: {path}")
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"File grew beyond 8 MiB: {path}")
    return data


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result or key in {"__proto__", "constructor", "prototype"}:
            raise ValueError(f"Duplicate or reserved JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _object(data: bytes, label: str) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected a nonempty JSON object: {label}")
    return value


def _choice(record: dict[str, Any], field: str, choices: set[str]) -> str:
    value = record.get(field)
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"Explicit {field} required; expected one of {sorted(choices)}")
    return value


def _string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Explicit {field} required")
    return value


def export_saved_setup(
    record_path: Path, calibration_root: Path, source: Path, source_commit: str, output: Path
) -> dict[str, Any]:
    """Copy explicitly selected files without resolving live application state."""
    if output.exists():
        raise ValueError("Output already exists; select a new directory")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Supply the selected checkout's full 40-character source commit")
    record_bytes = _bytes(record_path)
    record = _object(record_bytes, "robot record")
    _choice(record, "arm_type", {"metal"})
    mode = _choice(record, "mode", {"single", "bimanual"})
    arms = _choice(record, "arms", {"leader", "follower", "both"})
    leader_kind = _choice(record, "leader_kind", {"star", "metal"})
    robot_name = _string(record, "name")
    if not isinstance(record.get("cameras"), list):
        raise ValueError("Explicit cameras array required (an empty array is valid)")
    camera_names = []
    for camera in record["cameras"]:
        if not isinstance(camera, dict):
            raise ValueError("Each saved camera must be an object")
        camera_names.append(_string(camera, "name"))
    if len(set(camera_names)) != len(camera_names):
        raise ValueError("Camera names must be unique")

    files = [{"id": "record", "path": "record.json", "format": "json"}]
    copies = {"record.json": record_bytes}
    ports: set[str] = set()
    assignments: set[Path] = set()
    for prefix in (("", "right_") if mode == "bimanual" else ("",)):
        for side in (("leader", "follower") if arms == "both" else (arms,)):
            slot = prefix + side
            port = _string(record, slot + "_port")
            if port in ports:
                raise ValueError("Two active arm slots share the same port")
            ports.add(port)
            name = _string(record, slot + "_config").removesuffix(".json")
            if not name or "/" in name or "\\" in name or ".." in name:
                raise ValueError(f"Unsafe selected calibration name for {slot}")
            if side == "follower":
                relative = Path("robots") / "metal_follower" / (name + ".json")
            else:
                library = "metal_leader" if leader_kind == "metal" else "rebot_102_leader"
                relative = Path("teleoperators") / library / (name + ".json")
            selected = calibration_root / relative
            if selected in assignments:
                raise ValueError("Two active slots share the same selected calibration")
            assignments.add(selected)
            data = _bytes(selected)
            _object(data, slot + " calibration")
            filename = slot + "-calibration.json"
            copies[filename] = data
            files.append({"id": slot + "_calibration", "path": filename, "format": "json"})

    for index, relative_source in enumerate(SOURCE_FILES):
        filename = f"source-{index}.txt"
        copies[filename] = _bytes(source / relative_source)
        files.append({"id": relative_source, "path": filename, "format": "text"})
    manifest = {
        "schemaVersion": 1,
        "id": "makermodslab:" + robot_name,
        "source": {"repository": "makermods-robotics/makermodslab", "commit": source_commit},
        "scope": "saved-configuration-only",
        "files": files,
        "bindings": [],
    }
    # All selected inputs are validated before creating output. The exporter
    # neither scans unrelated calibration libraries nor guesses absent files.
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for filename, data in copies.items():
        descriptor = os.open(output / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
    descriptor = os.open(output / "manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def main() -> None:
    """Run the explicitly requested saved-file export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_saved_setup(
        args.record, args.calibration_root, args.source, args.source_commit, args.output
    )
    print(f"Exported {len(manifest['files'])} selected files to {args.output}")
    print("Saved configuration only; no device opened, session started or data uploaded.")
    print("Ports are compared literally. Calibration equality does not identify a physical arm.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RecursionError) as exc:
        print(f"Saved setup export refused: {exc}", file=sys.stderr)
        sys.exit(2)
