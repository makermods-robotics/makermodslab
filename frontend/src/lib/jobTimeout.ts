// Pure, dependency-free helpers for the HF Jobs "Job timeout" field: a rough
// suggested-duration estimator plus the parse/validate logic that mirrors the
// backend rule (makermodslab/train.py::parse_hf_duration).
//
// IMPORTANT: every constant in the estimator below is a deliberately coarse
// RULE OF THUMB, not a benchmark. They exist to put a sensible default in
// front of the user (who can always overtype it), never to precisely predict
// wall-clock time. Tune them freely as we learn real per-flavor throughput —
// nothing downstream depends on their exact values.

// ---------------------------------------------------------------------------
// Duration parsing / validation (mirror of the backend validator)
// ---------------------------------------------------------------------------

// One-or-more <number><unit> segments and nothing else. Number may be integer
// or decimal ("1.5h"); units are s/m/h/d. Anchored so "2h30" (bare number) or
// "2x" (bad unit) are rejected — identical to _DURATION_FULL_RE in train.py.
const DURATION_FULL_RE = /^(?:\d+(?:\.\d+)?[smhd])+$/;
const DURATION_SEGMENT_RE = /(\d+(?:\.\d+)?)([smhd])/g;
const UNIT_SECONDS: Record<string, number> = {
  s: 1,
  m: 60,
  h: 3600,
  d: 86400,
};

/**
 * Parse an HF-Jobs duration string ("2h", "45m", "3h30m", "1.5h") into a
 * positive integer of seconds. Returns null for anything malformed or that
 * resolves to <= 0 — matching parse_hf_duration's ValueError cases on the
 * backend, so the UI accepts exactly what the server will accept.
 */
export function parseTimeoutSeconds(value: string): number | null {
  const text = value.trim().toLowerCase();
  if (!text || !DURATION_FULL_RE.test(text)) return null;
  let total = 0;
  for (const [, num, unit] of text.matchAll(DURATION_SEGMENT_RE)) {
    total += parseFloat(num) * UNIT_SECONDS[unit];
  }
  const seconds = Math.round(total);
  return seconds > 0 ? seconds : null;
}

/** True when `value` is a well-formed, positive HF-Jobs duration string. */
export function isValidTimeout(value: string): boolean {
  return parseTimeoutSeconds(value) !== null;
}

// ---------------------------------------------------------------------------
// Suggested-timeout estimator (rules of thumb — see file header)
// ---------------------------------------------------------------------------

// Fixed container startup cost: image pull + the wrapper's pip-install of the
// pinned lerobot before the trainer even starts. ~5 min is generous-ish.
export const SETUP_OVERHEAD_SECONDS = 5 * 60;

// Assumed Hub download throughput for the dataset. Only applies when a dataset
// size is known; 20 MB/s is a conservative mid-range guess.
export const DATASET_DOWNLOAD_BYTES_PER_SEC = 20 * 1024 * 1024;

// Per-training-step wall-clock seconds by policy family, measured against a
// t4-small baseline (flavor multiplier 1.0). Coarse buckets: the light
// conv/transformer policies vs the heavier VLA families. NOT benchmarks.
export const PER_STEP_SECONDS: Record<string, number> = {
  act: 0.15,
  vqbet: 0.15,
  diffusion: 0.15,
  smolvla: 0.45,
  pi0: 0.45,
  pi0_fast: 0.45,
  pi05: 0.45,
};
// Fallback per-step cost for any policy type not in the table above.
export const UNKNOWN_PER_STEP_SECONDS = 0.3;

// Safety margin applied to the raw estimate before flooring/rounding. Better
// to over-provision the timeout than to have the platform kill a real run.
export const MARGIN = 1.7;

// Never suggest less than this, and never more than this.
export const FLOOR_SECONDS = 30 * 60; // 30m
export const CAP_SECONDS = 24 * 3600; // 24h

// Round-up granularity threshold: below 2h we snap to 15m units, above to 30m.
const FRIENDLY_BREAK_SECONDS = 2 * 3600;
const FINE_GRANULARITY_SECONDS = 15 * 60;
const COARSE_GRANULARITY_SECONDS = 30 * 60;

/**
 * Speed of a flavor relative to the t4-small baseline (smaller = faster, so a
 * smaller multiplier on per-step time). Keyed by prefix of the HF Jobs flavor
 * id (JobHardware: t4-small, t4-medium, l4x1, l4x4, l40sx1, a10g-small,
 * a10g-large, a100-large, a100x4, ...). Unknown / cpu flavors fall back to the
 * baseline 1.0 (conservative — never under-provision on an unrecognised id).
 *
 * NB: order matters — "l40sx1".startsWith("l4") is true, so the l40s check
 * must come before the l4 check.
 */
export function flavorMultiplier(flavor: string | null | undefined): number {
  if (!flavor) return 1.0;
  const f = flavor.toLowerCase();
  if (f.startsWith("t4")) return 1.0;
  if (f.startsWith("l40s")) return 0.35; // L40S — must precede the l4 check
  if (f.startsWith("l4")) return 0.7;
  if (f.startsWith("a10g")) return 0.6;
  if (f.startsWith("a100")) return 0.35;
  if (f.startsWith("h100")) return 0.35;
  return 1.0;
}

/** Per-step seconds for a policy type, falling back for unknown types. */
export function perStepSeconds(policyType: string): number {
  return PER_STEP_SECONDS[policyType] ?? UNKNOWN_PER_STEP_SECONDS;
}

/**
 * Round `seconds` UP to a friendly unit: 15-minute granularity below 2h,
 * 30-minute granularity at/above 2h. The threshold is evaluated on the input,
 * so a value just under 2h can round up to exactly 2h.
 */
export function roundUpToFriendly(seconds: number): number {
  const granularity =
    seconds < FRIENDLY_BREAK_SECONDS
      ? FINE_GRANULARITY_SECONDS
      : COARSE_GRANULARITY_SECONDS;
  return Math.ceil(seconds / granularity) * granularity;
}

export interface TimeoutEstimateInput {
  steps: number;
  policyType: string;
  // The selected HF Jobs flavor id (e.g. "t4-small"); may be undefined before
  // the user picks hardware, in which case we assume the t4 baseline.
  flavor?: string | null;
  // On-disk dataset size in bytes, when the Training flow knows it. Null/undefined
  // ⇒ the download term is dropped (degrade gracefully, don't guess).
  datasetSizeBytes?: number | null;
}

/**
 * Estimate a suggested job timeout, in seconds. Formula (all rules of thumb):
 *
 *   raw = (setup_overhead
 *          + dataset_download          // 0 when size unknown
 *          + steps * per_step * flavor_multiplier) * MARGIN
 *   result = min(cap, roundUpToFriendly(max(floor, raw)))
 */
export function estimateJobTimeoutSeconds(input: TimeoutEstimateInput): number {
  const steps =
    Number.isFinite(input.steps) && input.steps > 0 ? input.steps : 0;

  const download =
    input.datasetSizeBytes && input.datasetSizeBytes > 0
      ? input.datasetSizeBytes / DATASET_DOWNLOAD_BYTES_PER_SEC
      : 0;

  const training =
    steps * perStepSeconds(input.policyType) * flavorMultiplier(input.flavor);

  const raw = (SETUP_OVERHEAD_SECONDS + download + training) * MARGIN;
  const floored = Math.max(raw, FLOOR_SECONDS);
  return Math.min(roundUpToFriendly(floored), CAP_SECONDS);
}

/**
 * Format a whole-second duration as a compact HF-Jobs string ("30m", "2h",
 * "2h30m", "24h"). Used both for the "Suggested: ~2h30m" label and as the
 * exact value applied to the input on click — the output always re-parses via
 * parseTimeoutSeconds. Inputs here are estimator results (multiples of 15m,
 * capped at 24h), so hours+minutes covers every case (no days needed).
 *
 * NEVER LOCALIZE THIS. The `h`/`m`/`s` suffixes are wire format, not copy: the
 * string is written straight into the timeout input, re-parsed by
 * parseTimeoutSeconds against DURATION_FULL_RE, and validated again by
 * _DURATION_FULL_RE in makermodslab/train.py. Translating a unit letter would
 * make the field reject its own suggestion. Words AROUND the value ("Suggested:
 * ~", "(default)") are ordinary copy and live in the i18n catalogs.
 */
export function formatDurationShort(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.round((total % 3600) / 60);
  if (hours === 0) return `${minutes}m`;
  if (minutes === 0) return `${hours}h`;
  return `${hours}h${minutes}m`;
}

/** Estimator result packaged for the UI: seconds plus its display/apply label. */
export function suggestedTimeout(input: TimeoutEstimateInput): {
  seconds: number;
  label: string;
} {
  const seconds = estimateJobTimeoutSeconds(input);
  return { seconds, label: formatDurationShort(seconds) };
}
