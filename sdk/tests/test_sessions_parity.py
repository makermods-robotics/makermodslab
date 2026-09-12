"""The sessions parity tripwire: kinds and per-kind options can't drift.

Operations are snapshot-visible (the coverage ratchet catches those), but
session KINDS and their per-kind OPTIONS hide inside POST /api/v1/sessions'
body — nothing else fails when the server grows a kind or an option field.
These tests import the server's own registries (test-only dependency, like
the TrainingOptions parity test in test_jobs.py) and equality-assert both
directions, so the failure NAMES exactly what to implement: a new sugar
method + SUGAR_BY_KIND entry, or new kwargs on an existing one.

Pure introspection — no HTTP, no hardware, nothing started.
"""

from __future__ import annotations

import inspect

from makermodslab_sdk.resources.sessions import SUGAR_BY_KIND, SessionsResource

# Sugar parameters that are CLIENT-side machinery, not kind options:
# `robot` is the positional record name (rides at the body's top level, not in
# options), `owner` attaches the lease, `lease_timeout_s` tunes it — all three
# are SessionStartBody-level fields the server reads outside `options`.
CONTROL_PARAMS = frozenset({"robot", "owner", "lease_timeout_s"})

# Per-kind server option fields the SDK deliberately does NOT expose as sugar
# kwargs, each with its reason. Equality-guarded below so entries can only be
# added or removed consciously. Empty today — full parity.
EXCLUDED_OPTIONS: dict[str, dict[str, str]] = {}


def server_registries():
    """The server's kind registry and kind -> options-model map.

    _OPTIONS_MODELS is private; importing it here is deliberate — this file
    is the tripwire, and a rename over there SHOULD fail loudly over here.
    """
    from makermodslab.sessions import _OPTIONS_MODELS, STARTABLE_KINDS

    return set(STARTABLE_KINDS), dict(_OPTIONS_MODELS)


def sugar_option_params(method_name: str) -> set[str]:
    signature = inspect.signature(getattr(SessionsResource, method_name))
    params = {name for name in signature.parameters if name != "self"}
    return params - CONTROL_PARAMS


def test_every_startable_kind_has_sugar():
    kinds, options_models = server_registries()
    assert set(SUGAR_BY_KIND) == kinds, (
        f"kind drift.\n"
        f"  new server kinds needing a sugar method + SUGAR_BY_KIND entry: "
        f"{sorted(kinds - set(SUGAR_BY_KIND))}\n"
        f"  SDK kinds the server no longer starts: {sorted(set(SUGAR_BY_KIND) - kinds)}"
    )
    assert set(options_models) == kinds  # server-internal consistency, cheap to pin
    for kind, method_name in SUGAR_BY_KIND.items():
        method = getattr(SessionsResource, method_name, None)
        assert callable(method), f"SUGAR_BY_KIND names missing method {method_name!r} for {kind!r}"


def test_every_kind_options_field_is_a_sugar_kwarg():
    _, options_models = server_registries()
    for kind, method_name in sorted(SUGAR_BY_KIND.items()):
        server_fields = set(options_models[kind].model_fields)
        excluded = set(EXCLUDED_OPTIONS.get(kind, {}))
        sdk_params = sugar_option_params(method_name)
        assert sdk_params == server_fields - excluded, (
            f"{kind}: options drift between schemas/sessions.py and sessions.{method_name}().\n"
            f"  server fields missing from the sugar (add kwargs, thread into _options()): "
            f"{sorted(server_fields - excluded - sdk_params)}\n"
            f"  sugar kwargs the server model doesn't have (typo, or a removed field): "
            f"{sorted(sdk_params - server_fields)}"
        )


def test_registers_hold_no_stale_entries():
    kinds, options_models = server_registries()
    for kind, exclusions in EXCLUDED_OPTIONS.items():
        assert kind in kinds, f"EXCLUDED_OPTIONS names unknown kind {kind!r}"
        for field, reason in exclusions.items():
            assert field in options_models[kind].model_fields, (
                f"EXCLUDED_OPTIONS[{kind!r}] names field {field!r} the server no longer has"
            )
            assert reason.strip(), f"EXCLUDED_OPTIONS[{kind!r}][{field!r}] needs a real reason"
    # Every control param actually appears on every sugar method — a rename
    # there would silently widen the compared set otherwise.
    for method_name in SUGAR_BY_KIND.values():
        signature = inspect.signature(getattr(SessionsResource, method_name))
        missing = CONTROL_PARAMS - set(signature.parameters)
        assert missing == set(), f"{method_name}: control params not in signature: {sorted(missing)}"
