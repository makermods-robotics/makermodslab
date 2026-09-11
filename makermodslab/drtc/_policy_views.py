"""``--extra-image-roles``: giving a checkpoint a camera it was not published with.

Shared by ``policy.py`` and ``policy_rtc.py``, which are otherwise deliberate
hand-mirrors of each other. This one is NOT mirrored, and the reason is the
processor override below: it is the sort of change whose failure mode is a third
camera that connects, streams, and is silently never looked at. Two copies of
that would drift, and the drift would be invisible on both sides.

The problem it solves
---------------------
``lerobot/MolmoAct2-SO100_101-LeRobot`` declares exactly two image inputs,
``observation.images.cam0`` and ``cam1`` (3x224x224, VISUAL/IDENTITY). The
allenai model underneath takes a LIST of images and was fine-tuned on community
datasets carrying one to three cameras; the lerobot wrapper simply fixed the
list at two. Forking the repo to add a third is not an option — its
``model.safetensors`` is 21.8 GB — so the third view has to be a RUNTIME
override, applied to the config before the policy and its pre/post-processors
are built, exactly like ``--model-dtype`` and ``--flow-steps``.

Why declaring the feature is enough for the MODEL (verified against the pinned
fork's own ``policies/molmoact2/``, not from memory):

* ``MolmoAct2PackInputsProcessorStep._extract_images``
  (``processor_molmoact2.py:814``) iterates whatever image keys it resolves and
  appends one array per key — no count is hardcoded. ``_build_robot_text``
  joins ``Image {i}<|image|>`` over ``range(num_images)`` (``:377``) and
  ``infer_molmoact2_max_sequence_length`` budgets
  ``num_images * MOLMOACT2_IMAGE_TOKENS_PER_IMAGE`` (196, ``:100``).
  ``MOLMOACT2_DEFAULT_NUM_IMAGES = 2`` (``:67``) exists only as the fallback for
  a ``num_images < 1`` call. So three declared views really are three pictures.
* NO dataset statistics are needed. ``normalization_mapping["VISUAL"]`` is
  ``NormalizationMode.IDENTITY`` (``configuration_molmoact2.py:123``) and
  lerobot's ``NormalizerProcessorStep._apply_transform`` returns the tensor
  untouched for IDENTITY *and* for any key absent from its stats
  (``processor/normalize_processor.py:329-330``); ``_normalize_observation``
  (``:278``) only ever touches keys it carries a feature for, so an added image
  key that the SAVED normalizer step has never heard of simply passes through.
  That is the whole reason this override is cheap.

The pin, and the thing that is NOT what it looks like
-----------------------------------------------------
``policy.py`` builds its processors with
``make_pre_post_processors(policy.config, pretrained_path=policy_path, …)``, and
with a ``pretrained_path`` that factory does **not** call
``make_molmoact2_pre_post_processors`` at all — it LOADS the pipeline the
checkpoint saved (``policies/factory.py:180-209``,
``PolicyProcessorPipeline.from_pretrained``). So the pack step's ``image_keys``
come from the checkpoint's own ``preprocessor_config.json``
(``MolmoAct2PackInputsProcessorStep.get_config`` serializes it,
``processor_molmoact2.py:748``), NOT from ``config.input_features``. Setting
``config.image_keys`` alone would therefore add the feature to the wire, resize
the third frame, hand it to the pipeline — and have the pack step drop it on the
floor with nothing in the log, because ``_resolve_image_keys`` (``:800``) takes
its own saved list first.

Hence :data:`_PACK_STEP_BY_TYPE`: the roles are ALSO pushed into that step as a
constructor override, which ``from_pretrained`` merges over the saved config
per step (``processor/pipeline.py:1243``, ``{**saved_cfg, **step_overrides}``).
``config.image_keys`` is still written, for the no-``pretrained_path`` path and
so the config the policy carries agrees with the pipeline that reads it.

Two consequences worth knowing:

* ``_validate_overrides_used`` (``processor/pipeline.py:1336``) RAISES when an
  override names a step the saved pipeline does not have. That is the right
  direction — a checkpoint saved by a lerobot old enough to lack the pack step
  fails loudly at load instead of quietly serving two views — but it means the
  override may only be emitted when roles were actually added.
* only ``image_keys`` is overridden, never ``allow_image_key_fallback``. The
  fallback is reached only when a declared key is MISSING from the observation,
  and ``Inference._build_batch`` already raises on a camera the robot did not
  send — so the flag cannot change an outcome, and overriding a field a
  differently-saved checkpoint might not carry only adds a way to fail.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..utils.system import (
    MAX_EXTRA_IMAGE_ROLES,
    MOLMOACT2,
    is_valid_image_role,
    policy_supports_extra_image_roles,
)

# lerobot's own key prefix for a camera input feature. Spelled here rather than
# imported from `._schema` DELIBERATELY: that module imports `livekit.portal`,
# the FFI dylib behind the optional `[drtc]` extra, and this one has to stay
# importable by a plain `pytest` on an install that does not carry it. Held
# equal to `_schema.IMAGE_PREFIX` by a test rather than by an import.
IMAGE_PREFIX = "observation.images."

# {policy type: the REGISTRY NAME of the pre-processor step that decides which
# image keys get packed}. See the module docstring: with a `pretrained_path` the
# processor pipeline is loaded from the checkpoint, so this is where an added
# view has to be declared a SECOND time or it is silently never looked at.
#
# Keyed by type and not assumed, because the step name is per family. A family
# in `VARIABLE_VIEW_POLICY_TYPES` with no entry here would add the feature and
# lose it in the pipeline, so :func:`add_extra_image_roles` refuses that pairing
# rather than shipping the silent version.
_PACK_STEP_BY_TYPE: dict[str, str] = {MOLMOACT2: "molmoact2_pack_inputs"}

EXTRA_IMAGE_ROLES_HELP = (
    "Comma-separated EXTRA camera roles to declare on the checkpoint before the "
    "weights load, e.g. `cam2` or `cam2,cam3`. Each becomes an "
    "`observation.images.<role>` input feature copied from the checkpoint's first "
    "image feature (same shape, same type, same normalization), so the policy "
    "expects one more video track and the robot side publishes it. OPT-IN and "
    "REFUSED unless the checkpoint's policy type is one whose view count is a "
    "property of its wrapper rather than its architecture "
    "(utils.system.VARIABLE_VIEW_POLICY_TYPES). More views is more image tokens "
    "per prefill and therefore more latency, and the checkpoint's own authors did "
    "not test it — measure before trusting it."
)


def parse_extra_image_roles(value: str) -> list[str]:
    """``"cam2, cam3"`` -> ``["cam2", "cam3"]``. Empty in, empty out.

    Only splits and strips — every rule about what a role may BE is enforced in
    :func:`add_extra_image_roles`, so the CLI and the Modal wrapper (which
    forwards the raw string) cannot end up with two different notions of valid.
    """
    return [part.strip() for part in value.split(",") if part.strip()]


def add_extra_image_roles(policy_cfg, roles: list[str]) -> dict[str, Any]:
    """Declare ``roles`` as extra image inputs on ``policy_cfg``. Mutates it.

    Returns the ``preprocessor_overrides`` the caller must merge into its
    ``make_pre_post_processors`` call — EMPTY when nothing was added, which is
    also the caller's signal that ``config=`` need not be handed to
    ``from_pretrained``. It is not optional: see the module docstring, without it
    the added view is decoded, resized and dropped in silence.

    Raises ``SystemExit`` — the CLI-level refusal both policy servers use — for
    every way this can be the wrong thing to ask for. All of them are checked
    BEFORE the weights load, which is the entire point: the first checkpoint
    this matters for pulls 21.8 GB, and a refusal that lands after the download
    is worth nothing.
    """
    if not roles:
        return {}

    policy_type = getattr(policy_cfg, "type", "")
    if not policy_supports_extra_image_roles(policy_type):
        raise SystemExit(
            f"--extra-image-roles={','.join(roles)} was passed, but a '{policy_type}' "
            "checkpoint's image views are fixed by its architecture, not by its wrapper. "
            "Adding one would reach the model as a shape it was never built for. Drop the "
            "flag; if this family really does take a variable number of views, add it to "
            "utils.system.VARIABLE_VIEW_POLICY_TYPES after reading its processor."
        )

    pack_step = _PACK_STEP_BY_TYPE.get(policy_type)
    if pack_step is None:  # pragma: no cover — unreachable while the two maps agree
        raise SystemExit(
            f"--extra-image-roles: '{policy_type}' is listed as a variable-view family but no "
            "pre-processor step is registered for it in _policy_views._PACK_STEP_BY_TYPE, so an "
            "added view would be declared on the config and then dropped by the checkpoint's own "
            "saved processor pipeline. Add the step's registry name there first."
        )

    if len(roles) > MAX_EXTRA_IMAGE_ROLES:
        raise SystemExit(
            f"--extra-image-roles takes at most {MAX_EXTRA_IMAGE_ROLES} roles; got "
            f"{len(roles)} ({','.join(roles)}). Every view is another ~196 image tokens "
            "through the prefill, and this is a latency ceiling rather than a model one."
        )

    features = getattr(policy_cfg, "input_features", None)
    if not isinstance(features, dict):
        raise SystemExit(
            f"--extra-image-roles={','.join(roles)} was passed, but "
            f"{type(policy_cfg).__name__} has no input_features to add to."
        )

    # The TEMPLATE: the checkpoint's own first image feature, copied whole. Its
    # `type` is what carries the normalization (the norm map is keyed by
    # FeatureType, not by name), and its `shape` is what the GPU side resizes
    # incoming frames to — so copying the feature is copying all three of the
    # things the flag's contract promises.
    template_key = next((key for key, feat in features.items() if _is_visual(feat)), None)
    if template_key is None:
        raise SystemExit(
            f"--extra-image-roles={','.join(roles)} was passed, but this checkpoint declares "
            "no image features at all, so there is nothing to copy a role's shape and "
            "normalization from. It is not a camera policy."
        )

    seen: set[str] = set()
    for role in roles:
        if not is_valid_image_role(role):
            raise SystemExit(
                f"--extra-image-roles: {role!r} is not a usable camera role. Use lowercase "
                "letters, digits and underscores, starting with a letter, at most 32 "
                "characters (e.g. `cam2`). The role becomes a policy feature key, a video "
                "track name and a `--robot.cameras` key, and anything else breaks one of "
                "those silently."
            )
        if role in seen:
            raise SystemExit(f"--extra-image-roles: {role!r} is listed twice.")
        seen.add(role)
        key = f"{IMAGE_PREFIX}{role}"
        if key in features:
            raise SystemExit(
                f"--extra-image-roles: this checkpoint already declares {key!r}, so adding it "
                "would overwrite a view the policy was trained with. Bind that role to a "
                "camera instead of adding it."
            )
        features[key] = replace(features[template_key])
        print(
            f"[policy] --extra-image-roles: ADDING {key} to the checkpoint's input_features, "
            f"copied from {template_key} (shape={tuple(features[template_key].shape)}), before "
            "weights load (operator opt-in; the checkpoint's authors did not train or test "
            "this view — it is another ~196 image tokens per prefill)"
        )

    # Insertion order, so the checkpoint's own views stay Image 1 / Image 2 in
    # the prompt and an added one is appended: `_extract_images` packs in
    # exactly this order.
    pinned = [key for key, feat in features.items() if _is_visual(feat)]
    # Written on the config for the no-`pretrained_path` factory path (and so
    # the config the policy carries agrees with the pipeline reading it) …
    if hasattr(policy_cfg, "image_keys"):
        policy_cfg.image_keys = list(pinned)
    # … and, load-bearingly, onto the SAVED pack step, which is what actually
    # runs. See the module docstring.
    print(
        f"[policy] --extra-image-roles: pinning the {pack_step} step's image_keys={pinned} "
        "so the added view is packed rather than silently dropped"
    )
    return {pack_step: {"image_keys": list(pinned)}}


def _is_visual(feature: object) -> bool:
    """Whether a ``PolicyFeature`` is an image one.

    By the enum member's NAME rather than by importing ``FeatureType``: this
    module is imported by tests on an install without the `[drtc]` extra, and
    the name is the same string the checkpoint's own config.json spells.
    """
    return getattr(getattr(feature, "type", None), "name", "") == "VISUAL"
