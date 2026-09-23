"""Software test policy. These values do not change motor firmware protection.

135 C is the user's manufacturer-reported reference, not a verified readback of
this arm's shutdown setting. CAN sensor meaning/hardware revision must be checked
before interpreting winding limits as limits on a control-board sensor.
"""

STOP_AT_C = 100
CRITICAL_AT_C = 135
DEFAULT_TEST_DURATION_S = 1800
MAX_TEST_DURATION_S = 1800


def thermal_policy() -> dict:
    return {
        "stop_at_c": STOP_AT_C,
        "critical_at_c": CRITICAL_AT_C,
        "threshold_comparison": ">=",
        "critical_reference_source": "user-reported manufacturer guidance; firmware setting not read back",
    }
