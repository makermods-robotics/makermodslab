"""Software test policy. These values do not change motor firmware protection.

135 C is the user's manufacturer-reported reference, not a verified readback of
this arm's shutdown setting. The RS02 July 2026 manual identifies the MIT
feedback field as winding temperature; the legacy driver calls it temp_mos.
"""

STOP_AT_C = 110
CRITICAL_AT_C = 135
DEFAULT_TEST_DURATION_S = 1800
MAX_TEST_DURATION_S = 1800


def thermal_policy() -> dict:
    return {
        "temperature_source": "MIT winding-temperature feedback (legacy driver key: temp_mos)",
        "board_temperature_available": False,
        "stop_at_c": STOP_AT_C,
        "critical_at_c": CRITICAL_AT_C,
        "threshold_comparison": ">=",
        "critical_reference_source": "user-reported manufacturer guidance; firmware setting not read back",
    }
