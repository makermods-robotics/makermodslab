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
#
"""The camera-coverage preflight every inference start runs (SPEC §5).

The server reads an empty ``camera_bindings`` as a camera-less policy, so a
vision policy started without them energizes the arm and then dies on its
first action (``KeyError: 'observation.images.top'``). Before the start
request, this reads the policy's config (``jobs.policy_config``) and:

* refuses (``CameraBindingError``) when the bindings leave any camera the
  policy reads unbound — fail CLOSED, nothing has been sent;
* warns (``UnverifiedCamerasWarning``) and lets the start go ahead when the
  config is UNKNOWN (unreadable, or a server that predates the route) —
  fail OPEN, because blocking every run on an offline Hub or an older server
  would be worse than the pre-check-era behaviour it adds to;
* resolves ``camera_bindings="auto"`` to the identity map over EXACT,
  case-sensitive name matches, and fails CLOSED on anything it cannot
  resolve — the caller asked the SDK to decide, and it cannot decide blind.

Why "auto" never matches fuzzily (case-folded, by index, by the only
camera): a matching name is the one thing the operator wrote down on both
sides, and even that is not proof the record's ``top`` is the viewpoint the
policy trained on. Anything looser would be the SDK asserting a physical
fact it cannot observe; the explicit map stays the way to say otherwise.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import TYPE_CHECKING, Any, Literal

from makermodslab_sdk.errors import (
    ApiError,
    CameraBindingError,
    MakerModsError,
    ServerTooOldError,
    UnverifiedCamerasWarning,
)

if TYPE_CHECKING:
    from makermodslab_sdk._transport import Transport
    from makermodslab_sdk.resources.jobs import CheckpointPolicyConfig

AUTO = "auto"

CameraBindings = dict[str, str] | Literal["auto"] | None

# A bare Hub repo id — what remote inference takes as-is (the GPU loads it)
# but the policy-config route addresses as "<owner>/<repo>@root", which reads
# the same config.json the server's own remote camera-role check reads.
_BARE_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def _config_lookup_ref(policy_ref: str, *, remote: bool) -> str:
    if remote and _BARE_REPO_RE.match(policy_ref):
        return f"{policy_ref}@root"
    return policy_ref


def _record_camera_names(transport: Transport, robot: str, *, required: bool) -> tuple[str, ...] | None:
    """The robot record's camera names, in record order. ``required=False``
    turns any failure into None (the refusal message just omits them)."""
    from makermodslab_sdk.resources.robots import RobotsResource

    try:
        record = RobotsResource(transport).get(robot)
    except MakerModsError:
        if required:
            raise
        return None
    names: list[str] = []
    for entry in record.cameras or []:
        if isinstance(entry, dict):
            name = str(entry.get("name") or "").strip()
            if name:
                names.append(name)
    return tuple(names)


def _render(mapping: dict[str, Any]) -> str:
    return json.dumps(mapping)


def _call(method: str, robot: str, policy_ref: str, bindings_literal: str) -> str:
    return f"{method}({json.dumps(robot)}, policy_ref={json.dumps(policy_ref)}, camera_bindings={bindings_literal}, ...)"


def _uncovered_error(
    *,
    method: str,
    robot: str,
    policy_ref: str,
    expected: tuple[str, ...],
    missing: tuple[str, ...],
    given: dict[str, str],
    record: tuple[str, ...] | None,
    auto: bool,
) -> CameraBindingError:
    lines = [
        f"Policy {policy_ref!r} reads cameras {list(expected)}, but "
        + (
            f'camera_bindings="auto" found no robot-record camera named exactly {list(missing)} '
            "(auto binds by exact, case-sensitive name only)"
            if auto
            else f"camera_bindings leaves {list(missing)} unbound"
        )
        + ". Started this way the arm would energize and the policy would fail on its first action "
        f"(KeyError: 'observation.images.{missing[0]}'). Nothing was started.",
    ]
    if record is None:
        lines.append(
            f'Robot {robot!r}\'s cameras could not be read — client.robots.get("{robot}").cameras lists them.'
        )
    elif not record:
        lines.append(f"Robot {robot!r} has no cameras; add them to the record before running this policy.")
    else:
        lines.append(f"Robot {robot!r} has cameras {list(record)}.")
    if not auto and record is not None and set(expected) <= set(record):
        suggestion = _call(method, robot, policy_ref, '"auto"') + (
            f"  # binds {_render({name: name for name in expected})} by identical name"
        )
    else:
        proposal = {name: given.get(name, "<record camera name>") for name in expected}
        suggestion = _call(method, robot, policy_ref, _render(proposal))
        suggestion += "  # each value a robot-record camera name"
    lines.append(f"Next step: {suggestion}")
    return CameraBindingError(
        "\n".join(lines),
        policy_ref=policy_ref,
        expected=expected,
        missing=missing,
        record_cameras=record,
        suggestion=suggestion,
    )


def _unknown(
    reason: str,
    *,
    method: str,
    robot: str,
    policy_ref: str,
    auto: bool,
) -> None:
    """The config is unknown: warn and go ahead, or — for "auto" — refuse."""
    if auto:
        suggestion = _call(method, robot, policy_ref, '{"<policy camera>": "<record camera>", ...}')
        raise CameraBindingError(
            f'camera_bindings="auto" needs the policy\'s camera list, but {reason}. Nothing was started.\n'
            f"Next step: {suggestion}  # bind explicitly",
            policy_ref=policy_ref,
            expected=(),
            missing=(),
            record_cameras=None,
            suggestion=suggestion,
        )
    warnings.warn(
        f"Camera bindings for {policy_ref!r} could not be verified ({reason}); starting anyway. "
        "If this is a vision policy and the bindings miss a camera it reads, the arm energizes and "
        "the run fails on its first action. Pass verify_cameras=False to skip the check deliberately.",
        UnverifiedCamerasWarning,
        stacklevel=4,
    )


def resolve_camera_bindings(
    transport: Transport,
    *,
    method: str,
    robot: str,
    policy_ref: str,
    camera_bindings: CameraBindings,
    camera_dims: dict[str, dict[str, int]] | None,
    remote: bool,
) -> tuple[dict[str, str] | None, dict[str, dict[str, int]] | None]:
    """Check (or, for ``"auto"``, decide) camera coverage before a start.

    Returns the ``(camera_bindings, camera_dims)`` to send — the caller's own
    values unchanged, except that ``"auto"`` is replaced by the resolved map
    (never sent literally) and fills ``camera_dims`` from the config when the
    caller gave none. ``method`` is the literal call named in the next step.

    ``checkpoint.invalid_ref`` propagates for local inference — its start
    refuses the same ref shapes, and nothing has started yet — but counts as
    unknown for remote inference, whose start takes any ref (a bare repo id
    included) and reads it on the GPU side.
    """
    from makermodslab_sdk.resources.jobs import JobsResource

    auto = camera_bindings == AUTO
    given: dict[str, str] = {} if auto else dict(camera_bindings or {})  # type: ignore[arg-type]
    lookup = _config_lookup_ref(policy_ref, remote=remote)
    context = {"method": method, "robot": robot, "policy_ref": policy_ref, "auto": auto}

    config: CheckpointPolicyConfig
    try:
        config = JobsResource(transport).policy_config(lookup)
    except ServerTooOldError:
        _unknown("this server predates GET /api/v1/policy-config — update it to verify", **context)
        return camera_bindings, camera_dims  # type: ignore[return-value]  # never "auto": that raised
    except ApiError as exc:
        if exc.code == "checkpoint.invalid_ref" and not remote:
            raise
        reason = f"the server answered {exc.status}{', ' + exc.code if exc.code else ''}: {exc.detail}"
        _unknown(reason, **context)
        return camera_bindings, camera_dims  # type: ignore[return-value]  # never "auto": that raised

    expected = tuple(config.image_features)
    if not auto:
        missing = tuple(name for name in expected if name not in given)
        if missing:
            record = _record_camera_names(transport, robot, required=False)
            raise _uncovered_error(expected=expected, missing=missing, given=given, record=record, **context)
        return camera_bindings, camera_dims  # type: ignore[return-value]

    if not expected:
        return None, camera_dims
    record = _record_camera_names(transport, robot, required=True)
    assert record is not None
    missing = tuple(name for name in expected if name not in record)
    if missing:
        raise _uncovered_error(expected=expected, missing=missing, given={}, record=record, **context)
    bindings = {name: name for name in expected}
    if camera_dims is None:
        camera_dims = {
            name: {"width": config.image_features[name].width, "height": config.image_features[name].height}
            for name in expected
        }
    return bindings, camera_dims
