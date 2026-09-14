# Export a saved Metal setup for offline comparison

When reusing a table setup, compare the selected robot record and calibration
files before interpreting a later run. The optional exporter below produces
copies for human review or RLSOK's saved-configuration comparison workflow.
RLSOK is not required to export or inspect the files.

## Use the native-record review command

With RLSOK's saved-setup CLI installed, this wrapper exports the selected native
record/calibrations and creates the observation in one step:

```sh
python makermodslab/scripts/review_saved_setup.py \
  --record /path/to/copied/robots/my-metal.json \
  --calibration-root /path/to/copied/calibration \
  --source-commit YOUR_CHECKOUT_FULL_40_CHARACTER_COMMIT \
  --capture-only --output setup-01
```

Read `setup-01/record.json`, the selected calibration copies and observation.
Only then create your own baseline:

```sh
rlsok profile approve-setup --observation setup-01/observation.json \
  --actor YOUR_NAME --output approved-01.json
```

For the next saved configuration, run the same wrapper with
`--baseline approved-01.json --output setup-02` in place of `--capture-only`.
It leaves the old baseline untouched, captures the new selected native files,
and writes `setup-02/review/report.json` and `report.md`. For an offline change
example, use a separate copy of the record/calibration directories and alter
one selected calibration value in that copy only. The changed report identifies
the selected calibration file and value path. Never use that deliberately
changed copy in a robot session.

The wrapper invokes the installed `rlsok` executable with argument arrays and
never a shell command. `--rlsok /path/to/rlsok` selects an explicit executable.
Exit 0 means a capture or unchanged comparison completed, 1 means review is
required, and 2 means the workflow failed. It never creates approval
automatically. It does not add a runtime interception gate or prove bench use.

## Export without running RLSOK

Run the script directly using Python 3.12; it uses only the standard library:

```sh
python makermodslab/scripts/export_saved_setup.py \
  --record /path/to/copied/robots/my-metal.json \
  --calibration-root /path/to/copied/calibration \
  --source /path/to/makermodslab \
  --source-commit YOUR_CHECKOUT_FULL_40_CHARACTER_COMMIT \
  --output /path/to/new/setup-01
```

The paths are explicit; the exporter neither chooses an installation root nor
imports the live configuration module. It does not migrate application state,
connect to a bus, probe a motor, enable torque, start a session, run a policy,
or upload anything. In particular, a Metal bus handshake must not be treated
as an inert identification read.

The copied record must explicitly contain `arm_type: metal`, `mode`
(`single`/`bimanual`), `arms` (`leader`/`follower`/`both`), `leader_kind`
(`star`/`metal`), `name`, and a `cameras` array (empty is valid). For older
records, review and save a copy with explicit fields rather than letting this
exporter infer runtime defaults. Other arm families or leader variants are
refused in this initial exporter.

Only the calibration files assigned to active slots are read, with the same
library separation as the selected family:

- Metal follower: `robots/metal_follower/NAME.json`.
- Metal leader: `teleoperators/metal_leader/NAME.json`.
- Star leader using the Metal preset: `teleoperators/rebot_102_leader/NAME.json`.

The selected family/factory source, staging's CAN transport helpers, and `pyproject.toml` dependency pin are
copied too. A Maker preset is never substituted for Metal simply because
the Star calibration directory is shared. Source commit is supplied by the
operator; source bytes are included so local edits can also be compared.
Installed dependencies and active application state are not verified.

Duplicate JSON keys, missing selected files, repeated active ports, repeated
selected calibration files and ambiguous camera names are refused. The
output must be a new directory. Keep it private: it contains the selected
configuration and local device information. A partially failed filesystem
write can leave an incomplete directory; use a new output path after fixing
the error.

## Optional RLSOK comparison

Using RLSOK's `1.5.0-shadow.10` saved-setup workflow:

```sh
rlsok profile capture-setup --manifest setup-01/manifest.json --output observation-01.json
# Inspect the saved files and observation before recording your own review.
rlsok profile approve-setup --observation observation-01.json \
  --actor YOUR_NAME --output baseline-01.json

# Export again to setup-02 after selecting a later configuration, then:
rlsok profile capture-setup --manifest setup-02/manifest.json --output observation-02.json
rlsok profile review-setup --baseline baseline-01.json \
  --observation observation-02.json --output comparison-02
```

The baseline is never overwritten automatically. Changes to the mode, leader
kind, left/right assignments, selected calibration, cameras or copied source
appear in the report. Ports are compared literally; this export intentionally
does not claim a physical device identity. An operator may separately provide
reviewed device selectors/inventory through RLSOK's saved-setup workflow.

An unchanged report says only that selected saved inputs match. Metal
calibration files may have identical zero offsets across units: matching bytes
cannot identify which physical arm is attached. This tool does not install a
runtime gate, approve motion, demonstrate bench compatibility, or establish a
successful robot trial. It leaves all existing session controls unchanged.

See [RLSOK's saved-setup guide](https://github.com/realitywarden/rlsok/blob/v1.5.0-shadow.10/docs/saved-setup-review.md)
for optional identity resolution and the precise comparison scope.
