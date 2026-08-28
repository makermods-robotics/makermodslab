# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Export the app's OpenAPI spec to the committed snapshot at docs/api/openapi.json.

The snapshot is the reviewable record of the API surface: any change to routes,
request models, or (once they exist) response models shows up as a diff in the
commit that caused it. A pre-commit hook regenerates it whenever backend code
changes, and tests/test_api_contract.py fails if the committed copy drifts from
the live app.

Run manually with:

    uv run python -m makermodslab.scripts.export_openapi          # write
    uv run python -m makermodslab.scripts.export_openapi --check  # exit 1 on drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = REPO_ROOT / "docs" / "api" / "openapi.json"


def generate_spec() -> dict[str, Any]:
    """The app's OpenAPI document, as FastAPI generates it."""
    from makermodslab.server import app

    return app.openapi()


def render_spec(spec: dict[str, Any]) -> str:
    """Canonical serialization: sorted keys, 2-space indent, trailing newline.

    Sorted keys make the output deterministic regardless of route registration
    order, so snapshot diffs only ever show real surface changes.
    """
    return json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Don't write; exit 1 if the committed snapshot differs from the live app",
    )
    args = parser.parse_args(argv)

    rendered = render_spec(generate_spec())
    committed = SNAPSHOT_PATH.read_text() if SNAPSHOT_PATH.exists() else None

    if args.check:
        if committed != rendered:
            print(f"{SNAPSHOT_PATH} is stale; regenerate with:")
            print("    uv run python -m makermodslab.scripts.export_openapi")
            return 1
        return 0

    if committed == rendered:
        print(f"{SNAPSHOT_PATH} already up to date")
        return 0
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(rendered)
    print(f"wrote {SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
