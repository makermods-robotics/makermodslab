"""Remote lerobot inference over LiveKit Portal (DRTC), ported from `livekit-drtc`.

The robot half of a split inference loop: this machine owns the arm and the
cameras, a GPU elsewhere (today: Modal — see `modal_policy.py` /
`modal_policy_rtc.py` in this package) owns the policy, and LiveKit Portal
carries observations one way and action chunks the other.

Two regimes, picked by policy type — see `robot_sync` / `robot_rtc`:

  * **adaptive-sync** (`robot_sync`) — ANY policy, especially non-inpainting
    ones (ACT). Plays each chunk to completion, one seam per boundary, and
    adapts *when* it prefetches the next chunk to the measured round trip.
  * **full DRTC + inpainting** (`robot_rtc`) — flow/diffusion policies
    (smolvla, pi0, pi05). Ships the still-to-execute prefix so the server can
    guide denoising, making overlapping chunks dynamically consistent.

Both are `python -m` entrypoints so a Lab feature module can spawn them as a
subprocess the way `rollout.py` spawns `lerobot-rollout`:

    python -m makermodslab.drtc.robot_sync --robot.type=so101_follower ...

This package is only importable with the optional `drtc` extra installed
(`livekit-portal`, `livekit-api`, `python-dotenv`); nothing in the Lab imports
it at startup. The one exception is `_env`, which needs python-dotenv alone so
the credential-precedence rules stay unit-testable without LiveKit.
"""
