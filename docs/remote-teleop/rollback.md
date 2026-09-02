# Rollback and uninstall

Rollback is a stop-and-disable operation. It never writes firmware, servo IDs,
baud rate, EEPROM, or calibration, and it never requires deleting those assets.

## Rehearsal

Perform this once during secured-arm commissioning and record the before/after
calibration and firmware identifiers in the worksheet.

1. Press robot-local **STOP** and wait for the full receipt.
2. If torque-off is false or unknown, remove physical motor power and retain
   fault lockout.
3. Stop the operator action loop and close the leader.
4. Press **Disable** on both roles. Confirm listener stopped, devices closed,
   and runtime disabled.
5. Revoke the paired credential on the robot; confirm the operator can no
   longer authenticate.
6. Press **Remove saved remote config** on both laptops. This removes only the
   dormant remote-role file; it does not alter robot records or calibrations.
   The equivalent local API command is:

   ```bash
   curl -fsS -X DELETE http://127.0.0.1:8000/api/v1/arms/remote-teleoperation/configuration
   ```

7. Quit both MakerMods Lab processes with `Ctrl+C`.
8. Relaunch only a known prior/local-only build that includes the central
   hardware registry, then run normal local teleoperation. If the intended
   prior build predates that registry, keep motor power removed and do not open
   the arm with it; use a registry-capable build to reconcile any durable
   lockout first.
9. Compare calibration and firmware identifiers with the worksheet. They must
   be unchanged.

## Preserve evidence and disable configuration

Use the application UI to export/redact any required receipts. Keep the private
TLS key and role/credential files out of the repository. To take the generated
trial TLS identity out of service without deleting it, move its sibling
directory to a dated offline backup:

```bash
REMOTE_TLS_DIR="$(cd .. && pwd)/makermodslab-remote-private"
REMOTE_TLS_BACKUP="$(cd .. && pwd)/makermodslab-remote-private.disabled"
mv "$REMOTE_TLS_DIR" "$REMOTE_TLS_BACKUP"
chmod -R go-rwx "$REMOTE_TLS_BACKUP"
```

Do this only after both roles are disabled, the credential is revoked, and the
saved remote configuration is removed. The move is recoverable and touches no
MakerMods robot or calibration record.

## Remove the contribution build

For an editable checkout, stop its processes and switch to the known prior
revision using the team's normal Git workflow. Do not use `git reset --hard`
on a dirty checkout. Recreate the virtual environment only if the prior
revision requires it.

For a separately installed `uv tool` copy:

```bash
uv tool uninstall makermodslab
```

Then install the prior approved revision through the same mechanism used
before the trial. Do not delete application data as an uninstall shortcut:
robot definitions and calibrations share that data root. Remove only an
optional remote-specific autostart/service entry that the tester knowingly
created; this guide creates none.

The owner-private hardware safety journal and recovery target map are safety
state, not disposable remote configuration. An unresolved journal deliberately
survives uninstall and restart. Clear it only through a compatible local
recovery path that verifies torque-off and device close; never remove it as a
rollback shortcut.

## Rollback acceptance

- Remote runtime is disabled after a full process restart.
- TLS control and UDP action ports no longer listen.
- Leader and follower devices are closed.
- Any unresolved torque receipt remains a visible fault lockout.
- Local teleoperation can claim/release the central hardware registry.
- Firmware, servo IDs, baud, EEPROM, and calibration IDs/digests match the
  pre-trial worksheet.
