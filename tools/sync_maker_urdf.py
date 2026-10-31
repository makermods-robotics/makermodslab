#!/usr/bin/env python3
# Copyright 2026 MakerMods. All rights reserved.
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
"""Copy the Maker Arm URDF from maker-arm-sdk at SDK_COMMIT into frontend/public/maker-urdf/.

    python tools/sync_maker_urdf.py           # update the copy
    python tools/sync_maker_urdf.py --check   # exit 1 if the copy differs (CI)

To take a new SDK revision, set SDK_COMMIT and run it.
"""

import argparse
import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

SDK_COMMIT = "4517780b71664a07590947e6b68b3ebb406edc51"
SDK_DIR = "urdf/maker_arm/"
DEST = Path(__file__).resolve().parents[1] / "frontend/public/maker-urdf"
FILES = ("robot.urdf", "revision_report.json")


def sdk_files():
    url = f"https://codeload.github.com/makermods-robotics/maker-arm-sdk/tar.gz/{SDK_COMMIT}"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310  # nosec B310
        data = response.read()
    out = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            path = member.name.split("/", 1)[1] if "/" in member.name else ""
            if member.isfile() and path.startswith(SDK_DIR):
                name = path[len(SDK_DIR) :]
                if name in FILES or name.startswith("meshes/"):
                    out[name] = archive.extractfile(member).read()
    if "robot.urdf" not in out:
        sys.exit(f"no {SDK_DIR}robot.urdf at {SDK_COMMIT}")
    return out


def digest(data):
    """sha256 of the content; a Git LFS pointer (checkout without LFS) carries it as its oid."""
    if data.startswith(b"version https://git-lfs"):
        return next(
            line.split(b":")[1].decode() for line in data.splitlines() if line.startswith(b"oid sha256:")
        )
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    want = sdk_files()
    have = {str(p.relative_to(DEST)) for p in DEST.rglob("*") if p.is_file() and p.parent.name == "meshes"}
    have |= {f for f in FILES if (DEST / f).exists()}
    stale = sorted(have - want.keys())
    changed = sorted(
        n for n, d in want.items() if not (DEST / n).exists() or digest((DEST / n).read_bytes()) != digest(d)
    )
    if args.check:
        for name in changed + stale:
            print(f"{DEST / name} differs from maker-arm-sdk@{SDK_COMMIT[:7]}; run tools/sync_maker_urdf.py")
        sys.exit(1 if changed or stale else 0)
    for name in changed:
        (DEST / name).parent.mkdir(parents=True, exist_ok=True)
        (DEST / name).write_bytes(want[name])
    for name in stale:
        (DEST / name).unlink()
    print(f"maker-arm-sdk@{SDK_COMMIT[:7]}: {len(changed)} updated, {len(stale)} removed")


if __name__ == "__main__":
    main()
