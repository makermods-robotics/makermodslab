# Copyright 2026 RLSOK contributors. Licensed under the Apache License, Version 2.0.
"""Review native Metal record/calibration copies with an installed RLSOK CLI.

Run this file directly. Only the neighboring stdlib exporter is imported;
no application, configuration migration, driver or robot session is loaded.
The result is saved configuration review, never permission to move a robot.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from export_saved_setup import export_saved_setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--calibration-root', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rlsok', default='rlsok', help='Installed rlsok executable; no shell command')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture-only', action='store_true', help='Prepare an observation for manual review')
    mode.add_argument('--baseline', type=Path, help='Previously approved baseline; never refreshed here')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    if args.baseline is not None and not args.baseline.is_file():
        parser.error('Previously approved baseline must be a file')
    export_saved_setup(args.record, args.calibration_root, args.source,
                       args.source_commit, args.output)
    observation = args.output / 'observation.json'

    def rlsok(*arguments):
        return subprocess.run([args.rlsok, 'profile', *map(str, arguments)],
                              check=False, timeout=60, shell=False).returncode

    status = rlsok('capture-setup', '--manifest', args.output / 'manifest.json', '--output', observation)
    if status != 0:
        raise ValueError('RLSOK capture failed; exported files are retained for inspection')
    if args.capture_only:
        print(f'Review {observation} and the copied record/calibrations first.')
        print('Then use rlsok profile approve-setup with your explicit --actor and a new --output file.')
        return 0
    status = rlsok('review-setup', '--baseline', args.baseline, '--observation', observation,
                   '--output', args.output / 'review')
    if status not in (0, 1):
        raise ValueError('RLSOK comparison failed; no matching result is claimed')
    print(f'Report: {args.output / "review"}. No bus opened and no session started.')
    return status


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'Saved setup review incomplete: {exc}', file=sys.stderr)
        sys.exit(2)
