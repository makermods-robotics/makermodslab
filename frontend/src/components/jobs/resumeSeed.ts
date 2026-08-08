import { JobRecord, jobDisplayName } from "@/lib/jobsApi";
import type { Fetcher } from "@/lib/apiClient";
import {
  JobCheckpoint,
  dedupeCheckpointEntries,
  listJobCheckpoints,
} from "@/lib/checkpointsApi";
// The panel rework's seeds.ts is not part of this branch; ResumeSeed keeps its
// original home on TrainingConfigurator.
import type { ResumeSeed } from "@/components/training/TrainingConfigurator";

/** One checkpoint together with the run that owns it. Ownership is what a
 * resume needs: the seed's `resume_from_job_id` must name the run the backend
 * should resolve the checkpoint out of, which on a lineage is often an
 * ANCESTOR of the run whose row was clicked. */
export interface LineageCheckpoint {
  job: JobRecord;
  ckpt: JobCheckpoint;
}

/** Load the checkpoints reachable from `leaf`, walking its OWN ancestor path:
 * the leaf's own list first, then each ancestor's, nearest first.
 *
 * Both resume entry points (the library row's one-click and the card's
 * step-picker) read this one list, so they can't offer different checkpoints
 * for the same run. Sorted newest-step-first, then deduped by `ref` — with
 * lineage order preserved for equal refs, the surviving entry of a duplicate
 * pair is attributed to the NEAREST run in the chain.
 *
 * The dedupe is not cosmetic: a cloud resume reuses its parent's Hub output
 * repo, so every run in a cloud chain enumerates the same shared tree and each
 * inherited step would otherwise appear once per run.
 *
 * The whole ancestor path is loaded even though only the leaf's OWN entries
 * are resumable (see `resumableCheckpoints`): the card LISTS this list, and
 * being able to look at what a chain saved — and to run inference on an
 * inherited checkpoint — is not the same affordance as continuing training
 * from it. Narrowing happens at the resume rule, not here.
 *
 * KNOWN LIMIT, cloud only: because that Hub repo is shared by the WHOLE tree,
 * a checkpoint written by a FORKED SIBLING is in the listing too, and there is
 * nothing in it to attribute a step to the run that wrote it — so an ancestor
 * walk can't exclude a sibling's checkpoint the way it excludes an unrelated
 * run's. Local runs have no such ambiguity (each owns its output dir).
 * `resumableCheckpoints` narrows this for the resume path with a step cap
 * (see `cloudSiblingStepCap`); what remains after it is a sibling that forked
 * EARLY, whose checkpoints fall inside the leaf's own step range. True per-run
 * attribution of Hub checkpoints is a backend/identity change.
 *
 * A per-job failure yields no entries rather than rejecting: one unreachable
 * ancestor must not blank the checkpoints the rest of the chain does have. */
export async function loadLineageCheckpoints(
  baseUrl: string,
  fetcher: Fetcher,
  leaf: JobRecord,
  ancestors: JobRecord[],
): Promise<LineageCheckpoint[]> {
  const chain = [leaf, ...ancestors].filter((j) => j.checkpoint_count > 0);
  if (chain.length === 0) return [];
  const results = await Promise.all(
    chain.map((job) =>
      listJobCheckpoints(baseUrl, fetcher, job.id)
        .then((cks) => cks.map((ckpt) => ({ job, ckpt })))
        .catch(() => [] as LineageCheckpoint[]),
    ),
  );
  return dedupeCheckpointEntries(
    results.flat().sort((a, b) => b.ckpt.step - a.ckpt.step),
  );
}

/** Whether a run is in a state a resume can continue from.
 *
 * Asked of the LEAF — the tip of a resume chain, which is the only thing the
 * jobs list renders a row for. A leaf that is `running` has nothing to
 * continue yet; a `done` one reached its target, and resuming it would restore
 * an LR schedule that is already spent (SmolVLA's preset cosine-decays to a
 * 2.5e-6 floor over a fixed horizon, so the continuation trains at floor LR
 * while its loss curve flattens and reads as convergence). Fine-tune is the
 * intended way to build on a finished run. `imported` records have no run to
 * continue at all. The backend enforces the same refusals — this is the UI
 * half, so the button is absent rather than erroring on click. */
export function isResumableLeaf(job: JobRecord): boolean {
  return (
    (job.runner === "local" || job.runner === "hf_cloud") &&
    (job.state === "failed" || job.state === "interrupted")
  );
}

/** The highest checkpoint step a CLOUD leaf can prove lies on its own path, or
 * null when no cap applies (a local leaf, or one that never reported a step).
 *
 * A cloud chain publishes to ONE shared Hub repo, so its listing also holds
 * whatever a forked SIBLING wrote, and a checkpoint carries nothing that names
 * the run that wrote it. One thing is still provable without that attribution:
 * the leaf never trained past its own furthest step, so a checkpoint STRICTLY
 * ABOVE that step cannot lie on the leaf's path and must belong to a sibling
 * (or a sibling's descendant). Capping there excludes exactly the
 * provably-foreign steps. The live example is exactly this shape: two children
 * forked off one parent, the sibling ran to the shared 30k target, and its
 * checkpoints were being offered as "the newest checkpoint" to a leaf that
 * died at 6k.
 *
 * It narrows the known limit rather than closing it — a sibling that forked
 * EARLY wrote its checkpoints inside the leaf's own step range, where nothing
 * here can tell them apart. Per-run attribution of Hub checkpoints is a
 * backend/identity change, not something this call site can fix.
 *
 * POST-F7 GAP, deliberately left as-is here: the contaminating listing belongs
 * to the checkpoint's OWNER, not to the leaf, and a LOCAL leaf resumed from a
 * cloud parent now enumerates that parent's shared Hub repo too — so it can be
 * offered a forked sibling's checkpoint with no cap applied. Re-keying this on
 * the owner (`entry.job.runner === "hf_cloud"`) rather than on the leaf would
 * close that, and would change nothing for a pure-local or pure-cloud chain;
 * it is left for review rather than changed in passing, because it also
 * decides whether an ancestor's checkpoints ABOVE the leaf's inherited step
 * stay offered. On the live cross-runner example both keyings agree.
 * REVIEWED and deliberately kept leaf-keyed: owner-keying would hide a
 * superseded ancestor's above-step checkpoints, which have no other access
 * point in the UI — a real loss, traded against a sibling case that is both
 * rarer and recoverable, and the simpler rule is the one a novice can predict.
 *
 * `metrics.current_step` is the right reading, and not merely "how far it got
 * before dying": a resumed run's metrics are SEEDED at the checkpoint step it
 * resumed from (jobs.py `_initial_metrics`), so even a leaf that died before
 * its first tqdm frame reports a step at or above its inherited floor — which
 * is what keeps this cap from excluding the ANCESTOR checkpoints the leaf is
 * most often resumed from. A 0 means nothing was ever reported (no resume
 * floor, no log line), i.e. unknown rather than "step zero", so no cap. */
function cloudSiblingStepCap(leaf: JobRecord): number | null {
  if (leaf.runner !== "hf_cloud") return null;
  const furthest = leaf.metrics?.current_step ?? 0;
  return furthest > 0 ? furthest : null;
}

/** The checkpoints `leaf` may actually be resumed from, newest-step-first.
 *
 * THE one resume rule. Both entry points call it, so the row's Resume button
 * and the card's can no longer disagree about whether a run is resumable:
 *
 *  - the leaf must be in a resumable state (above);
 *  - the checkpoint's OWNER must not be `done` — the backend refuses a
 *    finished run as a resume source, so offering one only buys a 400;
 *  - the checkpoint's OWNER must have no children — STICKS ONLY (user
 *    decision 2026-08-07), see below;
 *  - the checkpoint must sit strictly below the leaf's step target, or the
 *    continuation would start at (or past) the finish line. A target of 0
 *    means "unset", so nothing is excluded;
 *  - on a CLOUD leaf, the checkpoint must also sit at or below the leaf's own
 *    furthest step — above that it provably belongs to a sibling sharing the
 *    Hub repo (see `cloudSiblingStepCap`).
 *
 * STICKS ONLY, and what it costs. A resume lineage is now a CHAIN: jobs.py
 * refuses a resume whose source already has a child (`JobAlreadyContinuedError`
 * -> 409), so continuing an ANCESTOR — which by construction already has a
 * child, the run below it on the path — is a click that could only end in that
 * refusal. Excluding an owner with children therefore leaves exactly the LEAF's
 * own checkpoints resumable, and the UI stops offering what the backend would
 * reject. It is the same rule stated at both ends, not a second policy.
 *
 * The list still SHOWS the whole lineage (`loadLineageCheckpoints` is
 * unchanged, and the card's dropdown renders all of it) — inspection and
 * inference on an inherited checkpoint are untouched. Only the Resume button
 * narrows, which is the affordance the 409 governs.
 *
 * The cost, stated plainly because it is the thing a user will hit: a tip that
 * died before saving anything now has NOTHING to resume, where it used to
 * inherit its parent's checkpoints. That case is not a dead end — deleting the
 * empty tip frees the parent to be resumed again — and `noResumeReason`
 * returns "parent-continued" so the toast can say exactly that. Branching is
 * deferred, not abolished; legacy forks keep rendering as they always did.
 *
 * What is deliberately NOT a rule: the owner sharing the leaf's runner. A
 * continuation may cross runners in EITHER direction (F7) — jobs.py
 * materializes a cloud parent's Hub checkpoint onto this disk for a
 * cloud→local one, and uploads a local parent's to the Hub, with consent, for
 * a local→cloud one. So the owner's runner decides only where the checkpoint
 * has to be MOVED before the trainer can read it, and becomes the Train form's
 * DEFAULT Compute (which stays editable — see ComputePane). Filtering on it
 * hid every checkpoint from a local leaf whose parent ran on the cloud,
 * leaving that card with no resume row at all while the library row still
 * offered a resume it then refused.
 */
export function resumableCheckpoints(
  leaf: JobRecord,
  lineage: LineageCheckpoint[],
): LineageCheckpoint[] {
  if (!isResumableLeaf(leaf)) return [];
  const target = leaf.config?.steps ?? 0;
  const cap = cloudSiblingStepCap(leaf);
  return lineage
    .filter(
      (entry) =>
        entry.job.state !== "done" &&
        entry.job.child_ids.length === 0 &&
        (target === 0 || entry.ckpt.step < target) &&
        (cap === null || entry.ckpt.step <= cap),
    )
    .sort((a, b) => b.ckpt.step - a.ckpt.step);
}

/** The newest lineage checkpoint that only the sticks rule excludes — i.e. the
 * one the user would be resuming if they first deleted the run in front of it.
 *
 * Exists so the empty-result toast can NAME the parent and the step instead of
 * saying "delete something". Lives here rather than in the toast because it is
 * the sticks rule read backwards, and a caller re-deriving it would be free to
 * disagree with `resumableCheckpoints` about which entry that is — the exact
 * drift the one-rule consolidation was for.
 *
 * Every other filter still applies: an entry a `done` owner or the step target
 * would have dropped anyway is not "blocked by a continuation", and offering
 * it as the delete-first payoff would be a lie. Returns null when there is no
 * such entry, which is every case except the empty-handed tip.
 */
export function blockedByContinuedOwner(
  leaf: JobRecord,
  lineage: LineageCheckpoint[],
): LineageCheckpoint | null {
  if (!isResumableLeaf(leaf)) return null;
  const target = leaf.config?.steps ?? 0;
  const cap = cloudSiblingStepCap(leaf);
  return (
    lineage
      .filter(
        (entry) =>
          entry.job.state !== "done" &&
          entry.job.child_ids.length > 0 &&
          (target === 0 || entry.ckpt.step < target) &&
          (cap === null || entry.ckpt.step <= cap),
      )
      .sort((a, b) => b.ckpt.step - a.ckpt.step)[0] ?? null
  );
}

/** Which rule above emptied the list — for the one caller that has to explain
 * an empty result to the user instead of just hiding a button.
 *
 * Lives here, beside the rule, because the alternative is a caller
 * re-implementing the filters to guess at the cause, which is how the toast
 * came to blame the step target for an exclusion the runner filter had made:
 * the run's checkpoints sat BELOW its target, and the message said the
 * opposite. The filters are applied in the same order as above, and the first
 * one to empty a non-empty set is the answer. */
export type NoResumeReason =
  | "not-resumable" // the leaf's own state — no run to continue
  | "no-checkpoints" // nothing saved anywhere in the lineage
  | "owner-done" // every candidate belongs to a finished run
  | "parent-continued" // sticks: every candidate is owned by an already-continued run
  | "at-target" // every candidate is at or past the leaf's step target
  | "sibling-cap" // cloud: every candidate is above the leaf's own furthest step
  | "other";

export function noResumeReason(
  leaf: JobRecord,
  lineage: LineageCheckpoint[],
): NoResumeReason {
  if (!isResumableLeaf(leaf)) return "not-resumable";
  if (lineage.length === 0) return "no-checkpoints";
  const live = lineage.filter((entry) => entry.job.state !== "done");
  if (live.length === 0) return "owner-done";
  // Sticks, and deliberately AFTER `owner-done`: both can be true of a legacy
  // chain, and only one of them has a way out. Telling the user to delete this
  // attempt so they can resume a parent that has itself finished would send
  // them to a refusal ("its schedule is spent — fine-tune instead"), so the
  // finished-owner answer wins where they overlap. jobs.py orders its two
  // refusals the same way, for the same reason.
  const ownedByLeaf = live.filter((entry) => entry.job.child_ids.length === 0);
  if (ownedByLeaf.length === 0) return "parent-continued";
  const target = leaf.config?.steps ?? 0;
  const belowTarget = ownedByLeaf.filter(
    (entry) => target === 0 || entry.ckpt.step < target,
  );
  if (belowTarget.length === 0) return "at-target";
  const cap = cloudSiblingStepCap(leaf);
  if (cap !== null && belowTarget.every((entry) => entry.ckpt.step > cap))
    return "sibling-cap";
  // Unreachable while this mirrors `resumableCheckpoints` — a non-empty set
  // surviving every filter means the caller had a checkpoint to offer. Kept as
  // a total answer rather than a throw: a wrong-but-vague toast beats a crash.
  return "other";
}

/**
 * Build the seed that puts the Train panel's configurator into resume mode for
 * `job` continuing from `step`.
 *
 * THE one place a resume seed is assembled. Both entry points call it — the
 * job card's step-selectable Resume and the jobs library's row-level
 * quick-resume — because they previously built the same payload
 * independently, which is exactly the shape of duplication that drifts.
 *
 * Carries the parent run's configured shape forward. The registry already
 * holds it as `job.config` (the persisted TrainingRequest), so this needs no
 * extra fetch and no reading of the checkpoint's train_config.json.
 *
 * The runner is derived from the job, not passed in. `ResumeSeed.runner` is
 * the runner the PARENT ran on — F7 lets the continuation cross to the other
 * one, and the form needs the parent's side to know whether continuing on the
 * cloud has to upload a local checkpoint first. That fact lives on the record,
 * and both call sites already gate their buttons on `job.runner`, so a caller
 * supplying it could only ever disagree by mistake; which runner the
 * continuation actually launches on stays the form's choice. `imported`
 * records are not resumable and never reach here; they map to "local" for
 * exhaustiveness.
 *
 * Scope note: this fills exactly the fields `ResumeSeed` declares today —
 * identity, dataset, policy, the step budget and the log/save cadence, plus the
 * runner and its flavor. The parent's HF Jobs timeout and its hyperparameters
 * (batch size, seed, worker count, device/AMP, optimizer) are NOT carried,
 * because the seed type has nowhere to put them and the configurator has
 * nothing to read them from. Widening that is a training-side change, not a
 * presentation one — see the PR description's deferred list.
 */
export function buildResumeSeed(job: JobRecord, step: number): ResumeSeed {
  const parent = job.config;
  const runner = job.runner === "hf_cloud" ? "hf_cloud" : "local";
  return {
    jobId: job.id,
    step,
    name: jobDisplayName(job),
    datasetRepoId: parent.dataset_repo_id,
    policyType: parent.policy_type,
    sourceSteps: parent.steps,
    logFreq: parent.log_freq,
    saveFreq: parent.save_freq,
    runner,
    flavor: runner === "hf_cloud" ? (job.hf_flavor ?? undefined) : undefined,
  };
}
