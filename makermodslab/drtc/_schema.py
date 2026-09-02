"""The wire contract shared by `robot_sync.py` and `policy.py`.

Portal fingerprints the state/action/chunk schema and *silently drops* any
packet whose schema doesn't match on both peers (same field names, same
order, same dtype; same chunk name + horizon; same video track names). So the
two sides must independently arrive at an identical schema.

We make that schema **canonical and index-based** so neither side has to know
the other's field naming:

  * state fields   -> ``s0 .. s{D-1}``   (D = state dimension)
  * action fields  -> ``a0 .. a{A-1}``   (A = action dimension)
  * video tracks   -> the camera names   (must match on both sides by name)

The policy side derives ``D``, ``A`` and the camera names from the loaded
checkpoint (:func:`policy_wire_schema`). The robot side derives the same
numbers from its real hardware features (:func:`robot_wire_schema`). When the
robot is the one the policy was trained on, the two agree automatically and
inference just works. When they don't, Portal drops the packets and you see
"0 chunks / 0 observations" at runtime — see the README troubleshooting note.

The *ordering* of state and action is the load-bearing assumption: the robot
serializes its joints in ``observation_features`` / ``action_features`` order,
and the policy consumes ``observation.state`` / emits ``action`` in the order
it was trained on. For the same robot these are the same order.
"""

from __future__ import annotations

from dataclasses import dataclass

from livekit.portal import DType

IMAGE_PREFIX = "observation.images."
CHUNK_NAME = "act"


@dataclass
class WireSchema:
    """Canonical Portal schema, agreed by both peers."""

    state_dim: int
    action_dim: int
    cameras: list[str]  # camera/track names, order-insensitive (keyed by name)

    def state_fields(self) -> list[tuple[str, DType]]:
        return [(f"s{i}", DType.F64) for i in range(self.state_dim)]

    def action_fields(self) -> list[tuple[str, DType]]:
        return [(f"a{i}", DType.F64) for i in range(self.action_dim)]


def robot_wire_schema(robot) -> tuple[WireSchema, list[str], list[str]]:
    """Derive the wire schema from a connected lerobot ``Robot``.

    Returns ``(schema, state_keys, action_keys)`` where ``state_keys`` /
    ``action_keys`` are the robot's own feature names in order, so the caller
    can map ``s{i}`` <-> the real joint and ``a{i}`` <-> the real motor.

    lerobot convention: ``robot.observation_features`` maps scalar keys (e.g.
    ``"shoulder_pan.pos"``) to a python type and camera keys (e.g. ``"up"``) to
    a shape tuple. ``robot.action_features`` maps motor keys to a type.
    """
    obs_features = dict(robot.observation_features)
    state_keys = [k for k, v in obs_features.items() if not isinstance(v, (tuple, list))]
    cameras = [k for k, v in obs_features.items() if isinstance(v, (tuple, list))]
    action_keys = list(robot.action_features.keys())

    schema = WireSchema(
        state_dim=len(state_keys),
        action_dim=len(action_keys),
        cameras=cameras,
    )
    return schema, state_keys, action_keys


def policy_wire_schema(policy) -> tuple[WireSchema, dict]:
    """Derive the wire schema from a loaded lerobot policy.

    Returns ``(schema, image_features)`` where ``image_features`` is
    ``policy.config.image_features`` (full ``observation.images.*`` keys ->
    ``PolicyFeature`` with a ``(C, H, W)`` shape), used to resize incoming
    frames to what the model expects.
    """
    cfg = policy.config
    state_feature = cfg.robot_state_feature
    action_feature = cfg.action_feature
    if state_feature is None or action_feature is None:
        raise RuntimeError(
            "policy config is missing a robot_state_feature or action_feature; "
            "this example targets state+image policies (ACT, diffusion, pi0, ...)"
        )
    image_features = dict(cfg.image_features)
    cameras = [k[len(IMAGE_PREFIX) :] if k.startswith(IMAGE_PREFIX) else k for k in image_features]

    schema = WireSchema(
        state_dim=int(state_feature.shape[0]),
        action_dim=int(action_feature.shape[0]),
        cameras=cameras,
    )
    return schema, image_features
