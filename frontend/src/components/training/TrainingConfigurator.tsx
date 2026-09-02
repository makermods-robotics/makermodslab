import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { createPortal } from "react-dom";
import { useLocation, useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useStudio } from "@/contexts/StudioContext";

import { TrainingConfig } from "@/components/training/types";
import ConfigurationTab from "@/components/training/ConfigurationTab";
import TrainingExtraGate from "@/components/training/TrainingExtraGate";
import PolicyExtraDialog from "@/components/training/PolicyExtraDialog";
import HfAuthBanner from "@/components/landing/HfAuthBanner";
import LocalDatasetCloudNotice from "@/components/training/config/LocalDatasetCloudNotice";
import LocalCheckpointCloudNotice from "@/components/training/config/LocalCheckpointCloudNotice";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import { JobCheckpoint, listJobCheckpoints } from "@/lib/checkpointsApi";
import { Label } from "@/components/ui/label";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import { Loader2, Play } from "lucide-react";

import {
  TrainingRequest,
  listJobs,
  startTrainingJob,
  listRunnerHardware,
  RunnerFlavor,
} from "@/lib/jobsApi";
import { useDatasets } from "@/hooks/useDatasets";
import { useDatasetUpload } from "@/hooks/useDatasetUpload";
import {
  getNodePolicyExtra,
  listNodes,
  nodeDisplayName,
} from "@/lib/nodesApi";
import { getDatasetInfo } from "@/lib/replayApi";

// Passed by the "Continue" button on a completed local job, or the "Resume"
// button on a cloud run that ended before its step target.
// Policies whose lerobot preset returns NO LR scheduler — a constant learning
// rate, so changing the step total cannot produce a schedule seam and the
// warning below would be a plain falsehood. Taken from
// `get_scheduler_preset() -> None` across the pinned lerobot v0.6.0's
// `policies/*/configuration_*.py`. If lerobot adds another such policy this
// list goes stale in the safe-ish direction: a warning that does not apply.
const POLICIES_WITHOUT_LR_SCHEDULE = new Set([
  "act",
  "tdmpc",
  "fastwam",
  "gaussian_actor",
]);

export type ResumeSeed = {
  // The run being CONTINUED — the lineage edge. Always the leaf the user
  // clicked, never the checkpoint's owner, so chains stay linear (see
  // buildResumeSeed).
  jobId: string;
  step: number | null; // null ⇒ resume from the latest checkpoint
  // CHAIN REWIND provenance: the ancestor whose storage holds the chosen
  // checkpoint, when it isn't the leaf's own. Undefined ⇒ the leaf owns it.
  // Never the edge — only where the bytes are read from.
  checkpointJobId?: string;
  name: string;
  datasetRepoId: string;
  policyType: string;
  sourceSteps: number; // the source run's configured total, for a sane prefill
  logFreq?: number; // the source run's log cadence, to preserve on resume
  saveFreq?: number; // the source run's checkpoint cadence, to preserve on resume
  // The parent run's runner + flavor. A resume DEFAULTS to the parent's runner
  // but may cross to the other one (F7), so `runner` is the form's starting
  // point AND its record of where the parent actually ran — which is what tells
  // the form that continuing on the cloud has to upload a local checkpoint
  // first. Omitted ⇒ treated as "local". When "hf_cloud", the launched run
  // targets the same flavor and continues into the parent's Hub output repo;
  // `flavor` stays editable, since which GPU the continuation rents is a real
  // per-launch choice.
  runner?: "local" | "hf_cloud";
  flavor?: string;
  // Cloud resume: the parent run's HF Jobs timeout. Omitting it let the form
  // render blank, which `configToRequest` sends as undefined and the runner
  // resolves to HF_JOB_TIMEOUT ("24h") — silently capping the continuation of a
  // run that had already proved it needs a longer budget. See NEW-12.
  hfJobTimeout?: string;
  // The remaining hyperparameters. lerobot rebuilds these from the checkpoint's
  // train_config.json on resume (build_training_command's resume branch emits
  // none of them), so carrying them forward does NOT change what trains — it
  // stops the form from *displaying* fresh-run defaults for a continuation, and
  // keeps the new JobRecord's persisted config truthful about the run's shape.
  batchSize?: number;
  seed?: number;
  numWorkers?: number;
  policyDevice?: string;
  policyUseAmp?: boolean;
  optimizerType?: string;
  optimizerLr?: number;
  optimizerWeightDecay?: number;
  optimizerGradClipNorm?: number;
};

// Passed by the "Fine-tune" button on an imported model. A fine-tune is a
// FRESH run (fresh optimizer, step 0) whose weights are initialized from the
// source checkpoint — distinct from resume.
export type FinetuneSeed = {
  jobId: string;
  step: number | null; // null ⇒ latest checkpoint of the source
  name: string;
  policyType: string;
  // Where the CHOSEN checkpoint's bytes live — the checkpoint's own `source`
  // field, straight from the job's checkpoint listing. It is exactly the fact
  // the backend keys on: a "local" checkpoint resolves to an absolute path on
  // this machine, which a pod cannot read, so fine-tuning it on the cloud has
  // to stage its weights to the Hub first (and asks for that consent).
  // Undefined ⇒ unknown (a stale deep-link seed built before this field), which
  // shows no notice and leaves the backend's 400 as the backstop.
  checkpointSource?: "local" | "hub";
};

/** Which local→cloud transfer a launch needs, if any. "resume" moves the whole
 * checkpoint of the run being continued (weights AND optimizer state);
 * "finetune" moves only the base checkpoint's weights, since a fine-tune starts
 * a fresh optimizer and never reads the rest. */
export type CheckpointUploadKind = "resume" | "finetune" | null;

interface TrainingConfiguratorProps {
  /** Controlled policy type (chosen upstream — the panel policy grid or the
   * home-page/router state). EssentialsCard's dropdown edits it back through
   * onPolicyTypeChange. */
  policyType: string;
  onPolicyTypeChange: (policyType: string) => void;
  /** Controlled training dataset. Empty string ⇒ Start stays disabled. */
  datasetRepoId: string;
  /** Controlled episode subset for datasetRepoId (e.g. seeded from the
   * dataset viewer's exclude-from-training checkboxes). undefined ⇒ every
   * episode trains, same as the field being absent from the request. */
  episodeIndices?: number[];
  /** A "Continue"/"Resume" seed — inherits the source run's target + cadence. */
  resumeSeed?: ResumeSeed | null;
  /** A "Fine-tune" seed — fresh run initialized from a source checkpoint. */
  finetuneSeed?: FinetuneSeed | null;
  /** Report a fine-tune base checkpoint choice UP to the owner of
   * `finetuneSeed`. The chosen step must live in the seed, not in this
   * component's state: this form is remounted on a number of parent changes
   * (and by Fast Refresh in dev), and local state would silently revert the
   * pick to the source's latest checkpoint — training from weights the user
   * did not choose, with nothing on screen saying so. */
  onFinetuneCheckpointChange?: (ckpt: JobCheckpoint) => void;
  /** Fired with the new job id right before navigating to its monitor (e.g. so
   * the studio overlay can close). Navigation to /training/:jobId still runs. */
  onStarted?: (jobId: string) => void;
  /** Where the Start button renders. Omitted (undefined) ⇒ inline at the foot
   * of the form (the /training route). An element ⇒ portaled there — the
   * studio Train panel passes its pinned actions slot above the jobs library.
   * null ⇒ the slot isn't mounted yet; render nothing (avoids a one-frame
   * inline flash before the ref callback fires). */
  actionsContainer?: HTMLElement | null;
}

// PI0.5 is a 4B-parameter VLA — a real cloud run OOM'd an 80GB A100 on step 1
// at the standard batch_size=8 in full precision. Every other policy keeps the
// historical off default; only PI0.5 needs mixed precision to fit.
const POLICY_DEFAULT_USE_AMP: Record<string, boolean> = {
  pi05: true,
};
const defaultUseAmp = (policyType: string): boolean =>
  POLICY_DEFAULT_USE_AMP[policyType] ?? false;

function configToRequest(
  c: TrainingConfig,
  checkpointUploadKind: CheckpointUploadKind,
): TrainingRequest {
  // The backend's TrainingRequest has more optional fields; the form covers
  // the user-meaningful subset.
  return {
    target: c.target,
    dataset_repo_id: c.dataset_repo_id,
    dataset_episodes: c.dataset_episodes,
    policy_type: c.policy_type,
    job_name: c.job_name,
    steps: c.steps,
    batch_size: c.batch_size,
    seed: c.seed,
    num_workers: c.num_workers,
    log_freq: c.log_freq,
    save_freq: c.save_freq,
    save_checkpoint: c.save_checkpoint,
    resume: c.resume,
    resume_from_job_id: c.resume_from_job_id,
    resume_from_step: c.resume_from_step,
    resume_from_checkpoint_job_id: c.resume_from_checkpoint_job_id,
    // Consent, not a mode: sent only for the combinations that have to push
    // bytes to the Hub (a checkpoint only this machine has, needed by a run on
    // cloud compute), and the backend refuses those without it. Left undefined
    // otherwise so no other launch carries an upload permission it has no use
    // for. One field per MODE rather than one shared flag, because they consent
    // to different disclosures: the whole checkpoint of the run being continued,
    // versus the base model's weights.
    upload_resume_checkpoint:
      checkpointUploadKind === "resume" ? true : undefined,
    upload_finetune_checkpoint:
      checkpointUploadKind === "finetune" ? true : undefined,
    finetune_from_job_id: c.finetune_from_job_id,
    finetune_from_step: c.finetune_from_step,
    wandb_enable: c.wandb_enable,
    wandb_project: c.wandb_project,
    wandb_entity: c.wandb_entity,
    wandb_notes: c.wandb_notes,
    wandb_mode: c.wandb_mode,
    wandb_disable_artifact: c.wandb_disable_artifact,
    policy_device: c.policy_device,
    policy_use_amp: c.policy_use_amp,
    optimizer_type: c.optimizer_type,
    optimizer_lr: c.optimizer_lr,
    optimizer_weight_decay: c.optimizer_weight_decay,
    optimizer_grad_clip_norm: c.optimizer_grad_clip_norm,
    use_policy_training_preset: c.use_policy_training_preset,
    // Cloud-only; the backend validates the format and ignores it for local.
    // Send only a non-blank value so a stray "" doesn't reach the validator.
    hf_job_timeout:
      c.target.runner === "hf_cloud" && c.hf_job_timeout?.trim()
        ? c.hf_job_timeout.trim()
        : undefined,
  };
}

/**
 * The training configuration form — extracted verbatim from Training.tsx's
 * ConfigurationMode so the behavior (resume/fine-tune seeding, cloud
 * upload-first, single-local-run lock, auth/flavor gating, policy-extra install
 * gate) lives in exactly one place. Rendered by the transitional /training route
 * AND by the studio Train panel.
 *
 * `policy_type` and `dataset_repo_id` are controlled by the caller (which owns
 * the policy picker + dataset selection); everything else is internal form
 * state seeded once from resume/fine-tune. Callers that change the resume /
 * fine-tune seed must remount this component (change its React key) — a
 * different seed is a fundamentally different run.
 */
const TrainingConfigurator: React.FC<TrainingConfiguratorProps> = ({
  policyType,
  onPolicyTypeChange,
  datasetRepoId: controlledDatasetRepoId,
  episodeIndices: controlledEpisodeIndices,
  resumeSeed = null,
  finetuneSeed = null,
  onFinetuneCheckpointChange,
  onStarted,
  actionsContainer,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { auth } = useHfAuth();
  const { toast } = useToast();
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const { openJobMonitor } = useStudio();

  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    // A resume DEFAULTS to the parent run's runner — a cloud one so the
    // continuation runs on the same flavor and pushes into the same Hub repo, a
    // local one so it reads the parent's on-disk checkpoint. The Compute
    // control stays editable: either cross-runner direction works now (F7), it
    // just moves the checkpoint first — down from the Hub for a cloud→local
    // continuation, up to it for a local→cloud one. Everything else defaults to
    // a fresh local run.
    target:
      resumeSeed?.runner === "hf_cloud"
        ? { runner: "hf_cloud", flavor: resumeSeed.flavor }
        : { runner: "local" },
    // Controlled fields — overlaid from props below; placeholders here.
    dataset_repo_id: "",
    policy_type: "act",
    job_name: "",
    // On resume the fields below are PREFILLED from the parent run's persisted
    // config (via ResumeSeed) rather than left at fresh-run defaults — see the
    // ResumeSeed comments for which of them lerobot actually honours. Prefill,
    // don't lock: every one stays editable, since extending steps or raising the
    // timeout on a continuation is exactly what the form is for. `steps` is
    // inherited as-is like the rest, so resuming an interrupted run defaults to
    // FINISHING its original target (one that died at step 4,814 of 15,000
    // resumes toward 15,000); raise it by hand to train past that. Resuming a
    // run that already reached its target therefore prefills steps equal to the
    // checkpoint's step — `resumeStepError` below blocks Start until the user
    // raises it, as does the backend. Fine-tune is a fresh run, so it uses the
    // normal fresh defaults throughout.
    steps: resumeSeed ? resumeSeed.sourceSteps : 10000,
    batch_size: resumeSeed?.batchSize ?? 8,
    seed: resumeSeed?.seed ?? 1000,
    num_workers: resumeSeed?.numWorkers ?? 4,
    log_freq: resumeSeed?.logFreq ?? 50,
    save_freq: resumeSeed?.saveFreq ?? 1000,
    // Always on, no longer user-settable. Turning checkpointing off produced a
    // run whose only artifact was a log: with no checkpoint there is nothing to
    // continue, fine-tune, download or deploy, so every follow-up action on the
    // job card went dead. The request field stays — lerobot still needs the
    // flag — it just isn't a choice the form offers.
    save_checkpoint: true,
    // Derived from the entry point, never from a control: a ResumeSeed means
    // the user arrived via Continue / Resume on a job card, and that is the
    // only way a resume can be coherent. The backend resolves resume_from_* into
    // the checkpoint's config_path, and refuses `resume` without a source.
    // Fine-tune is NOT a resume — it's a fresh run whose weights are seeded from
    // the source checkpoint. resume stays false; the backend resolves
    // finetune_from_* into --policy.pretrained_path.
    resume: !!resumeSeed,
    resume_from_job_id: resumeSeed?.jobId,
    resume_from_step: resumeSeed?.step ?? undefined,
    resume_from_checkpoint_job_id: resumeSeed?.checkpointJobId,
    finetune_from_job_id: finetuneSeed?.jobId,
    finetune_from_step: finetuneSeed?.step ?? undefined,
    wandb_enable: false,
    wandb_mode: "online",
    wandb_disable_artifact: false,
    policy_device: resumeSeed?.policyDevice ?? "auto",
    // A resume prefills the parent run's own AMP setting (it may not match
    // today's per-policy default); a fresh run gets the per-policy default.
    policy_use_amp: resumeSeed?.policyUseAmp ?? defaultUseAmp(policyType),
    optimizer_type: resumeSeed?.optimizerType ?? "adam",
    optimizer_lr: resumeSeed?.optimizerLr,
    optimizer_weight_decay: resumeSeed?.optimizerWeightDecay,
    optimizer_grad_clip_norm: resumeSeed?.optimizerGradClipNorm,
    // Always on, no longer user-settable. lerobot rejects the config outright
    // when the preset is off and no scheduler is supplied ("Optimizer and
    // Scheduler must be set when the policy presets are not used") — and this
    // form never emits a --scheduler.* flag, so `false` could only ever produce
    // a run that died at config validation. The request field stays.
    use_policy_training_preset: true,
    // Cloud-only. Prefilled from the parent so a Continue keeps its budget
    // instead of silently falling back to the runner's 24h default; the field
    // stays editable so the user can raise it for a longer tail.
    hf_job_timeout: resumeSeed?.hfJobTimeout,
  });

  // The config the form actually reads: internal state overlaid with the
  // controlled policy type + dataset. Keeps EssentialsCard's frozen dataset
  // display and policy dropdown bound to the caller's selection.
  const config: TrainingConfig = useMemo(
    () => ({
      ...trainingConfig,
      policy_type: policyType,
      dataset_repo_id: controlledDatasetRepoId,
      dataset_episodes: controlledEpisodeIndices,
      // Read the fine-tune step from the SEED, not from the state seeded at
      // mount. This is what makes the picker authoritative: `trainingConfig`
      // froze the step this form opened with, so a launch after a pick (or
      // after any remount) would send the wrong checkpoint.
      ...(finetuneSeed
        ? { finetune_from_step: finetuneSeed.step ?? undefined }
        : {}),
    }),
    [
      trainingConfig,
      policyType,
      controlledDatasetRepoId,
      controlledEpisodeIndices,
      finetuneSeed,
    ],
  );

  const [trainingExtraAvailable, setTrainingExtraAvailable] = useState<
    boolean | null
  >(null);
  const [trainingExtraInstallHint, setTrainingExtraInstallHint] =
    useState<string>("pip install accelerate");
  const [localJobRunning, setLocalJobRunning] = useState<boolean>(false);
  const [isStarting, setIsStarting] = useState(false);
  const [policyExtra, setPolicyExtra] = useState<{
    policyType: string;
    packageName: string;
    installTarget: string;
    installHint: string;
    // Set when the missing extra is on a LAN NODE's environment (the chosen
    // compute target), not the local one — the dialog then installs there.
    node?: { instanceId: string; name: string };
  } | null>(null);
  const [authenticated, setAuthenticated] = useState<boolean>(false);
  const [flavors, setFlavors] = useState<RunnerFlavor[]>([]);
  const [hardwareLoading, setHardwareLoading] = useState(true);
  // HF_HUB_OFFLINE on the backend: Hub writes (incl. dataset upload) disabled.
  const [offline, setOffline] = useState<boolean>(false);

  // Whether the user has hand-toggled the AMP switch this session — once they
  // have, their choice wins over the per-policy default below, same as any
  // other form field the policy dropdown doesn't reset.
  const ampTouchedRef = useRef(false);

  // The policy dropdown lives inside this same mounted form (PolicyField ->
  // updateConfig("policy_type", ...) -> onPolicyTypeChange), so switching
  // architecture does not remount the component or re-run the useState
  // initializer above. Re-apply the per-policy AMP default whenever the
  // selection changes, unless the user already touched the switch by hand.
  // A resumeSeed's policyType is locked (see policyLocked below), so this
  // never fires from a real dropdown change during a resume — but effects
  // still run once on mount, which would otherwise stomp the parent run's own
  // prefilled AMP value with today's per-policy default the instant the form
  // opens. Skip entirely when resuming.
  useEffect(() => {
    if (ampTouchedRef.current || resumeSeed) return;
    setTrainingConfig((prev) => ({
      ...prev,
      policy_use_amp: defaultUseAmp(policyType),
    }));
  }, [policyType, resumeSeed]);

  useEffect(() => {
    fetchWithHeaders(`${baseUrl}/api/v1/system/training-extra`)
      .then((r) => r.json())
      .then((data: { available: boolean; install_hint: string }) => {
        setTrainingExtraAvailable(data.available);
        setTrainingExtraInstallHint(data.install_hint);
      })
      .catch(() => setTrainingExtraAvailable(true));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    // Only the local lock matters for the Start button; cloud jobs can stack.
    // Pull a generous slice so a running local isn't masked by newer cloud
    // jobs in the started_at-desc ordering.
    listJobs(baseUrl, fetchWithHeaders, 200)
      .then((j) =>
        setLocalJobRunning(
          j.some((r) => r.runner === "local" && r.state === "running"),
        ),
      )
      .catch(() => setLocalJobRunning(false));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    // Re-fetches when auth status flips (e.g. user pastes a token in
    // HfAuthBanner) so flavors unlock without a page reload.
    setHardwareLoading(true);
    listRunnerHardware(baseUrl, fetchWithHeaders)
      .then((data) => {
        setAuthenticated(data.authenticated);
        setFlavors(data.flavors);
        setOffline(!!data.offline);
      })
      .catch(() => {
        setAuthenticated(false);
        setFlavors([]);
        setOffline(false);
      })
      .finally(() => setHardwareLoading(false));
  }, [baseUrl, fetchWithHeaders, auth.status]);

  // ── Fine-tune base checkpoint picker ─────────────────────────────────────
  // The base's checkpoints, so the user can fine-tune from a step OTHER than
  // the latest. This lives INSIDE the configurator on purpose: TrainPanel keys
  // this component on `finetuneSeed.step`, so driving the choice from outside
  // would remount the form and discard everything already typed into it. Here
  // the pick is a live edit of `finetune_from_step` instead.
  const [baseCheckpoints, setBaseCheckpoints] = useState<JobCheckpoint[]>([]);
  // Held in a ref so the fetch effect can call the newest callback without
  // listing it as a dependency: both parents pass an inline arrow, so a real
  // dependency would re-fetch the checkpoint list on every render.
  const onFinetuneCheckpointChangeRef = useRef(onFinetuneCheckpointChange);
  useEffect(() => {
    onFinetuneCheckpointChangeRef.current = onFinetuneCheckpointChange;
  });

  const finetuneJobId = finetuneSeed?.jobId;
  const finetuneSeedStep = finetuneSeed?.step;
  useEffect(() => {
    if (!finetuneJobId) {
      setBaseCheckpoints([]);
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    void (async () => {
      try {
        const cks = await listJobCheckpoints(
          baseUrl,
          fetchWithHeaders,
          finetuneJobId,
          controller.signal,
        );
        if (cancelled) return;
        setBaseCheckpoints(cks);
        // Honour the step the seed arrived with (a job card's Fine-tune, a
        // studio prefill); fall back to the latest when that step is gone or
        // the seed named none. Mirrors resolveFinetune's own rule so the
        // dropdown never contradicts the banner it sits in.
        const pinned =
          finetuneSeedStep != null
            ? cks.find((c) => c.step === finetuneSeedStep)
            : undefined;
        // Normalise UPWARD when the seed names no step, or one this source no
        // longer has: the parent then holds a real step and the choice survives
        // a remount. Never overwrite a step that IS present — that would undo
        // the user's pick every time this effect re-ran.
        if (pinned === undefined) {
          const latest = cks.length > 0 ? cks[cks.length - 1] : null;
          if (latest) onFinetuneCheckpointChangeRef.current?.(latest);
        }
      } catch {
        // Listing failed (offline Hub, deleted run). Leave the picker hidden;
        // the seed's own step still drives the request and the backend's 400 is
        // the backstop.
        if (!cancelled) setBaseCheckpoints([]);
      }
    })();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [finetuneJobId, finetuneSeedStep, baseUrl, fetchWithHeaders]);

  // The step actually being sent — the SEED's, which the parent owns. Derived,
  // never stored here, so a remount cannot revert it.
  const effectiveFinetuneStep = finetuneSeed?.step ?? null;
  const baseCkpt =
    baseCheckpoints.find((c) => c.step === effectiveFinetuneStep) ?? null;
  // Where the CHOSEN checkpoint's bytes live — drives the cloud upload consent
  // notice. Must follow the live pick: switching from a hub step to a local one
  // changes whether staging is required at all.
  const effectiveCheckpointSource =
    baseCkpt?.source ?? finetuneSeed?.checkpointSource;
  // "makermods/smolvla_3cam_200ep" -> "…smolvla_3cam_200ep". The leading
  // ellipsis marks that a namespace was dropped; the full id stays available
  // as the heading's title attribute.
  const baseDisplayName = (() => {
    const name = finetuneSeed?.name ?? "";
    const slash = name.lastIndexOf("/");
    return slash === -1 ? name : `…${name.slice(slash + 1)}`;
  })();

  const updateConfig = <T extends keyof TrainingConfig>(
    key: T,
    value: TrainingConfig[T],
  ) => {
    // policy_type is controlled by the caller; route edits (EssentialsCard's
    // dropdown) back up. dataset_repo_id is controlled + display-only, but
    // guard it too so a stray write can't desync.
    if (key === "policy_type") {
      onPolicyTypeChange(value as string);
      return;
    }
    if (key === "dataset_repo_id") return;
    if (key === "policy_use_amp") ampTouchedRef.current = true;
    setTrainingConfig((prev) => ({ ...prev, [key]: value }));
  };

  // Cloud training runs from the Hub, so a dataset that only exists in this
  // machine's local cache must be uploaded first. We read the chosen dataset's
  // `source` from the /datasets listing (already carries it) and, for a
  // local-only + cloud combo, chain an upload before the job launches.
  const { datasets } = useDatasets();
  const datasetRepoId = config.dataset_repo_id.trim();
  const isCloud = config.target.runner === "hf_cloud";

  const selectedDatasetItem = datasets.find((d) => d.repo_id === datasetRepoId);
  // Only "local" needs uploading; "both"/"hub" already exist on the Hub, and an
  // unknown item (listing not yet loaded / dataset typed by hand) is left alone
  // — the backend preflight is the belt-and-braces catch for that case.
  const datasetLocalOnly = selectedDatasetItem?.source === "local";
  const needsUpload = isCloud && datasetLocalOnly;

  // A checkpoint this machine alone holds, needed by a run that will happen on
  // a pod that can't read this disk — so it has to be uploaded to a private Hub
  // repo before the job is submitted (F7). Two ways in, and which one it is
  // decides what moves and which consent field carries it:
  //
  //   * "resume" — the parent ran locally and the continuation was retargeted
  //     at the cloud. Derived from the SEED's runner rather than the config,
  //     because the config's `target` is the user's live choice while the seed
  //     is the only record of where the parent actually ran.
  //   * "finetune" — the chosen base checkpoint is a local one. Read off the
  //     seed's `checkpointSource`, which is the checkpoint's own `source` field
  //     — exactly what the backend keys on. Undefined (a stale deep-link seed)
  //     is treated as no-notice; the backend's 400 is the backstop.
  const checkpointUploadKind: CheckpointUploadKind =
    resumeSeed != null && resumeSeed.runner !== "hf_cloud" && isCloud
      ? "resume"
      : effectiveCheckpointSource === "local" && isCloud
        ? "finetune"
        : null;
  const needsCheckpointUpload = checkpointUploadKind != null;

  // Approximate on-disk size for the notice (cheap detail endpoint; local only).
  const [datasetSizeBytes, setDatasetSizeBytes] = useState<number | null>(null);
  useEffect(() => {
    if (!needsUpload || !datasetRepoId) {
      setDatasetSizeBytes(null);
      return;
    }
    let cancelled = false;
    getDatasetInfo(baseUrl, fetchWithHeaders, datasetRepoId)
      .then((info) => {
        if (!cancelled) setDatasetSizeBytes(info.size_bytes ?? null);
      })
      .catch(() => {
        if (!cancelled) setDatasetSizeBytes(null);
      });
    return () => {
      cancelled = true;
    };
  }, [needsUpload, datasetRepoId, baseUrl, fetchWithHeaders]);

  const [uploadError, setUploadError] = useState<string | null>(null);

  // The actual job launch, factored out so it can run either directly (dataset
  // already on the Hub) or as the upload's success continuation.
  const launchJob = useCallback(async () => {
    setIsStarting(true);
    try {
      const job = await startTrainingJob(
        baseUrl,
        fetchWithHeaders,
        configToRequest(config, checkpointUploadKind),
      );
      // The job's name is data — shown exactly as the backend returned it.
      // A busy local slot QUEUES the run rather than refusing (PR #83): say
      // so, with its 1-based position when the record carries one.
      if (job.state === "queued") {
        toast({
          title: t("training.configurator.toast.queuedTitle"),
          description:
            (job.queue_position ?? 0) > 0
              ? t("training.configurator.toast.queuedBody", {
                  name: job.name,
                  position: job.queue_position ?? 0,
                })
              : job.name,
        });
      } else {
        toast({
          title: t("training.configurator.toast.startedTitle"),
          description: job.name,
        });
      }
      onStarted?.(job.id);
      // The monitor is a dialog over the studio's Train panel, not a route.
      // openJobMonitor opens the studio; off-Launchpad callers (the
      // transitional /training route) must also land on "/" to see it.
      openJobMonitor(job.id);
      if (location.pathname !== "/") navigate("/");
    } catch (e) {
      // The backend's message, untranslated; only the title is ours.
      const msg = e instanceof Error ? e.message : String(e);
      toast({
        title: t("training.configurator.toast.errorTitle"),
        description: msg,
        variant: "destructive",
      });
      // If the failure was the 409 case, refresh our running-job knowledge.
      listJobs(baseUrl, fetchWithHeaders, 200)
        .then((j) =>
          setLocalJobRunning(
            j.some((r) => r.runner === "local" && r.state === "running"),
          ),
        )
        .catch(() => {});
    } finally {
      setIsStarting(false);
    }
  }, [
    baseUrl,
    fetchWithHeaders,
    config,
    checkpointUploadKind,
    toast,
    navigate,
    location.pathname,
    openJobMonitor,
    onStarted,
    t,
  ]);

  // Latest launchJob without re-subscribing the upload hook every render.
  const launchJobRef = useRef(launchJob);
  launchJobRef.current = launchJob;

  const { uploading, start: startUpload } = useDatasetUpload({
    repoId: datasetRepoId,
    onDone: () => {
      // Upload finished: launch the cloud job exactly as a Hub dataset would.
      setUploadError(null);
      launchJobRef.current();
    },
    onError: (message) => {
      // Upload failed: stop cleanly, surface the error, launch nothing (no
      // orphan output repo — the job was never submitted).
      setUploadError(message);
      setIsStarting(false);
      toast({
        title: t("training.configurator.toast.uploadFailedTitle"),
        description: message,
        variant: "destructive",
      });
    },
  });

  const handleStart = async () => {
    if (!datasetRepoId) {
      toast({
        title: t("training.configurator.toast.errorTitle"),
        description: t("training.configurator.toast.datasetRequired"),
        variant: "destructive",
      });
      return;
    }

    // Pre-flight: smolvla/pi0/pi0_fast/pi05/diffusion need an optional package
    // installed WHERE THE RUN EXECUTES — locally for a local run, on the peer
    // for a LAN-node run (read through the server-to-server proxy). Catch it
    // here with a one-click installer instead of a buried ImportError after
    // the job has already started. Cloud jobs run in their own container
    // environment, so neither answer applies — skip the check.
    if (config.target.runner === "local") {
      try {
        const r = await fetchWithHeaders(
          `${baseUrl}/api/v1/system/policy-extra/${config.policy_type}`,
        );
        if (r.ok) {
          const extra = await r.json();
          if (extra.needs_extra && !extra.available) {
            setPolicyExtra({
              policyType: config.policy_type,
              packageName: extra.package,
              installTarget: extra.install_target,
              installHint: extra.install_hint,
            });
            return;
          }
        }
      } catch {
        // Check failed (offline / older backend) — fall through and let the
        // job report any problem itself.
      }
    } else if (
      config.target.runner === "lan_node" &&
      config.target.node_instance_id
    ) {
      const instanceId = config.target.node_instance_id;
      try {
        const extra = await getNodePolicyExtra(
          baseUrl,
          fetchWithHeaders,
          instanceId,
          config.policy_type,
        );
        if (extra.needs_extra && !extra.available) {
          // The dialog names the node so the user knows WHERE the install
          // lands; resolve its display name from the registry, falling back
          // to a short instance id when the listing can't be read.
          let name = instanceId.slice(0, 8);
          try {
            const listing = await listNodes(baseUrl, fetchWithHeaders);
            const entry = listing.nodes.find(
              (n) => n.instance_id === instanceId,
            );
            if (entry) name = nodeDisplayName(entry);
          } catch {
            // Name lookup is cosmetic — the short id is enough.
          }
          setPolicyExtra({
            policyType: config.policy_type,
            packageName: extra.package,
            installTarget: extra.install_target,
            installHint: extra.install_hint,
            node: { instanceId, name },
          });
          return;
        }
      } catch {
        // Node unreachable or an older peer without the endpoint — fall
        // through and let the peer's own job validation report the problem.
      }
    }

    // Cloud run on a local-only dataset: upload first, then launch on success.
    // The Start button is disabled while `uploading`, so we never reach here
    // with an in-flight upload for this repo (the hook re-attaches on mount and
    // will fire onDone/onError for one already running). On success the hook
    // seeds a running status and its poll fires onDone → launchJob; a refusal
    // (409 / dataset busy) is a hard stop surfaced in the notice.
    if (needsUpload) {
      setUploadError(null);
      setIsStarting(true);
      const err = await startUpload([], false /* public: MakerMods Lab uploads are public by default */);
      if (err) {
        setUploadError(err);
        setIsStarting(false);
        toast({
          title: t("training.configurator.toast.uploadFailedTitle"),
          description: err,
          variant: "destructive",
        });
      }
      return;
    }

    await launchJob();
  };

  if (trainingExtraAvailable === null) {
    return (
      <div className="flex items-center justify-center py-24 text-muted-foreground">
        <Loader2 className="w-6 h-6 animate-spin mr-3" />
        {t("training.configurator.checkingEnvironment")}
      </div>
    );
  }

  if (trainingExtraAvailable === false) {
    return <TrainingExtraGate installHint={trainingExtraInstallHint} />;
  }

  const targetRequiresAuth = config.target.runner === "hf_cloud";
  const targetMissingFlavor =
    config.target.runner === "hf_cloud" && !config.target.flavor;
  // A running local job no longer blocks Start: the backend QUEUES the
  // submission (PR #83). `localJobRunning` survives only to make the button
  // say what the click will actually do.
  const willQueue = config.target.runner === "local" && localJobRunning;
  // When resuming, total steps must be strictly above the checkpoint's step:
  // equal trains nothing and lerobot requires --steps above the resumed step.
  const resumeStepError =
    resumeSeed != null &&
    resumeSeed.step != null &&
    config.steps <= resumeSeed.step
      ? t("training.configurator.resumeStepError", {
          // Pre-formatted; number formatting stays as it was.
          step: resumeSeed.step.toLocaleString(),
        })
      : null;
  // A local-only dataset on a cloud run is uploadable — unless the backend is
  // in offline mode, in which case uploads are impossible and Start is a hard
  // block. (needsUpload is already gated on isCloud.)
  const uploadBlockedOffline = needsUpload && offline;
  // A local run continued on the cloud has to push its checkpoint to the Hub
  // first, which offline mode makes impossible — a hard block, exactly like the
  // dataset case above.
  const checkpointUploadBlockedOffline = needsCheckpointUpload && offline;
  const startDisabled =
    isStarting ||
    uploading ||
    !datasetRepoId ||
    (targetRequiresAuth && !authenticated) ||
    targetMissingFlavor ||
    uploadBlockedOffline ||
    checkpointUploadBlockedOffline ||
    resumeStepError != null;
  const startTooltip =
    targetRequiresAuth && !authenticated
      ? t("training.configurator.tooltip.needAuth")
      : targetMissingFlavor
        ? t("training.configurator.tooltip.needFlavor")
        : uploadBlockedOffline
          ? t("training.configurator.tooltip.offlineDataset")
          : checkpointUploadBlockedOffline
            ? t("training.configurator.tooltip.offlineCheckpoint")
            : willQueue
              ? t("training.configurator.tooltip.willQueue")
              : undefined;

  return (
    <div className="w-full">
      <HfAuthBanner />
      {resumeSeed ? (
        <div className="mb-4 rounded-lg border border-primary/40 bg-primary/5 p-4 text-sm text-foreground">
          {/* The run name and both step numbers are data (the numbers keep
              their existing toLocaleString formatting and are interpolated
              pre-formatted). */}
          <div className="font-semibold">
            {resumeSeed.step != null
              ? t("training.configurator.resume.titleFromStep", {
                  name: resumeSeed.name,
                  step: resumeSeed.step.toLocaleString(),
                })
              : t("training.configurator.resume.titleFromLatest", {
                  name: resumeSeed.name,
                })}
          </div>
          {/* One complete sentence per runner — English spliced ", and the job
              timeout" into the middle of a list, which no translation can
              place the same way. */}
          <p className="mt-1 text-muted-foreground">
            <Trans
              i18nKey={
                isCloud
                  ? "training.configurator.resume.bodyCloud"
                  : "training.configurator.resume.bodyLocal"
              }
              values={{ steps: config.steps.toLocaleString() }}
              components={[<span key="0" className="font-medium" />]}
            />
          </p>
          {/* The LR-schedule seam. lerobot's schedulers are LambdaLR objects
              whose horizon is captured in a closure over `cfg.steps`, and
              LambdaLR.state_dict() stores `lr_lambdas: [None]` for plain
              functions — so resuming REBUILDS the schedule from the new total
              and restores only the step counter, never the horizon. Change the
              total and the learning rate jumps at the resume point instead of
              continuing to decay. (Measured on the SmolVLA preset: a parent
              targeting 1,500 resumed at 5,000 gives a 32x higher LR at the
              seam.) Keeping the parent's total is continuous, so this warns
              only when the number actually differs. */}
          {resumeSeed.sourceSteps > 0 &&
          config.steps !== resumeSeed.sourceSteps &&
          !POLICIES_WITHOUT_LR_SCHEDULE.has(resumeSeed.policyType) ? (
            <p className="mt-2 text-warn">
              {t("training.configurator.resume.lrSeam", {
                from: resumeSeed.sourceSteps.toLocaleString(),
                to: config.steps.toLocaleString(),
              })}
            </p>
          ) : null}
          {isCloud ? (
            <p className="mt-1 text-muted-foreground">
              {/* The timeout is an HF-Jobs duration string — wire format, so
                  it is rendered verbatim. */}
              <Trans
                i18nKey="training.configurator.resume.jobTimeout"
                values={{
                  timeout: config.hf_job_timeout?.trim()
                    ? config.hf_job_timeout
                    : t("training.configurator.resume.jobTimeoutDefault"),
                }}
                components={[<span key="0" className="font-medium" />]}
              />
            </p>
          ) : null}
        </div>
      ) : null}
      {finetuneSeed ? (
        <div className="mb-4 rounded-lg border border-border bg-muted/50 p-4 text-sm text-foreground">
          {/* Namespace dropped and the id truncated from the LEFT: the
              namespace is the user's own on every row, and the distinguishing
              part of these names is the tail (…orange_tray vs …orange_box), so
              right-truncation would make two different bases render
              identically. Full id on hover via the title attribute.

              The step is deliberately NOT repeated here when the picker below
              is shown — it was printed twice, once in this heading and again
              in the control that sets it. With a single checkpoint there is no
              picker, so the heading keeps naming the step itself. */}
          <div className="min-w-0">
            <div className="truncate font-semibold" title={finetuneSeed.name}>
              {baseCheckpoints.length > 1
                ? t("training.configurator.finetune.title", {
                    name: baseDisplayName,
                  })
                : effectiveFinetuneStep != null && effectiveFinetuneStep !== 0
                  ? t("training.configurator.finetune.titleWithStep", {
                      name: baseDisplayName,
                      step: effectiveFinetuneStep.toLocaleString(),
                    })
                  : t("training.configurator.finetune.titleLatest", {
                      name: baseDisplayName,
                    })}
            </div>
            {/* The full id, verbatim, on its own line — the house pattern for a
                long repo id (DatasetLibrary, HubModelCard, ModelCard all pair a
                shortened heading with the complete address underneath). The
                heading drops the namespace to stay readable, and this is where
                that dropped information comes back, so "which org / which of
                two similarly-named bases" is answerable without hovering.
                font-mono because it is an address, not prose. */}
            <div
              className="truncate font-mono text-[11px] leading-4 text-muted-foreground"
              title={finetuneSeed.name}
            >
              {finetuneSeed.name}
            </div>
          </div>
          {/* Step picker. Shown only when there is a real choice: a single
              checkpoint (every imported model, and any run that saved once)
              would be a one-option dropdown, and the banner above already
              names it. Lives here so switching checkpoints is a live edit
              rather than a remount that would clear the form. */}
          {baseCheckpoints.length > 1 ? (
            <div className="mt-2 flex flex-wrap items-center gap-2">
              {/* Matches the chip it labels: same size (text-xs), same family
                  (font-mono) and same colour (inherited foreground, not muted),
                  so the pair reads as one control rather than a caption sitting
                  next to a widget. */}
              <Label
                htmlFor="finetune-base-checkpoint"
                className="font-mono text-xs text-foreground"
              >
                {t("training.configurator.finetune.checkpointLabel")}
              </Label>
              <CheckpointDropdown
                id="finetune-base-checkpoint"
                checkpoints={baseCheckpoints}
                selectedRef={baseCkpt?.ref ?? null}
                onChange={(c) => onFinetuneCheckpointChange?.(c)}
                disabled={isStarting}
              />
            </div>
          ) : null}
          <p className="mt-2 text-muted-foreground">
            <Trans
              i18nKey="training.configurator.finetune.body"
              components={[<span key="0" className="font-medium" />]}
            />
          </p>
        </div>
      ) : null}
      <ConfigurationTab
        config={config}
        updateConfig={updateConfig}
        authenticated={authenticated}
        flavors={flavors}
        hardwareLoading={hardwareLoading}
        policyLocked={finetuneSeed != null || resumeSeed != null}
        // On a resume, lerobot rebuilds every hyperparameter but the
        // continuation essentials from the checkpoint's train_config.json, so
        // those controls render read-only rather than accepting edits the run
        // will discard. configToRequest still sends the form's values, keeping
        // the new JobRecord's persisted config truthful about what was asked.
        resumeLocked={resumeSeed != null}
      />
      {needsUpload ? (
        <div className="mt-6">
          <LocalDatasetCloudNotice
            repoId={datasetRepoId}
            sizeBytes={datasetSizeBytes}
            offline={offline}
            uploading={uploading}
            errorMessage={uploadError}
          />
        </div>
      ) : null}
      {/* Compute is editable on a resume now (F7), so the user can retarget a
          local run's continuation at the cloud. That direction moves the
          parent's checkpoint to the Hub first, which is a disclosure — hence a
          notice before the click, and a Start button that says "Upload". */}
      {checkpointUploadKind === "resume" && resumeSeed ? (
        <div className="mt-6">
          <LocalCheckpointCloudNotice
            mode="resume"
            runName={resumeSeed.name}
            step={resumeSeed.step}
            offline={offline}
          />
        </div>
      ) : checkpointUploadKind === "finetune" && finetuneSeed ? (
        <div className="mt-6">
          {/* The LIVE step, not the seed's. This notice names the checkpoint
              whose weights get staged to the Hub, and the staging follows
              `finetune_from_step` — which the picker edits. Reading the seed
              here would name the checkpoint the user arrived with while
              uploading the one they actually chose. */}
          <LocalCheckpointCloudNotice
            mode="finetune"
            runName={finetuneSeed.name}
            step={effectiveFinetuneStep}
            offline={offline}
          />
        </div>
      ) : null}
      {(() => {
        const actionsBlock = (
          <div
            className={
              actionsContainer !== undefined
                ? "flex flex-col gap-2"
                : "mt-6 flex flex-col gap-2"
            }
          >
        {resumeStepError ? (
          <p className="text-sm text-destructive">{resumeStepError}</p>
        ) : null}
        {(() => {
          const startButton = (
            <Button
              onClick={handleStart}
              disabled={startDisabled}
              className="w-full gap-2"
            >
              {/* Icon sizing/spacing is left to the Button's own `gap-2` and
                  `[&_svg]:size-4`, exactly as Collect's "Start recording" and
                  Run's "Start inference" do — an extra `mr-2` here doubled the
                  icon gap, so the label shifted the moment the studio panel
                  swapped its disabled stand-in for this button. */}
              {uploading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />{" "}
                  {t("training.configurator.button.uploading")}
                </>
              ) : isStarting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />{" "}
                  {t("training.configurator.button.starting")}
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />{" "}
                  {/* Sentence case, matching the disabled stand-in in
                      TrainPanel and Collect's / Run's Start buttons — these
                      used to flip to Title Case the moment the form opened. */}
                  {resumeSeed
                    ? needsCheckpointUpload
                      ? t("training.configurator.button.uploadAndContinue")
                      : t("training.configurator.button.continueTraining")
                    : finetuneSeed
                      ? // A local base fine-tuned on the cloud has to push its
                        // weights first, so the button says what the click
                        // actually does — the same promise the resume branch
                        // above makes.
                        needsCheckpointUpload
                        ? t("training.configurator.button.uploadAndStart")
                        : t("training.configurator.button.startFinetuning")
                      : needsUpload
                        ? t("training.configurator.button.uploadAndStart")
                        : willQueue
                          ? // The local slot is busy — the click ENQUEUES, and
                            // the button says so instead of promising a start.
                            t("training.configurator.button.queueTraining")
                          : t("training.configurator.button.startTraining")}
                </>
              )}
            </Button>
          );
          // Native `title` doesn't fire reliably on disabled buttons across
          // browsers — and since Radix's tooltip relies on pointer events
          // that a disabled button swallows, wrap in a span so the trigger
          // still receives hover/focus.
          if (!startTooltip) return startButton;
          return (
            <Tooltip>
              <TooltipTrigger asChild>
                <span tabIndex={0} className="block w-full">
                  {startButton}
                </span>
              </TooltipTrigger>
              <TooltipContent>{startTooltip}</TooltipContent>
            </Tooltip>
          );
        })()}
          </div>
        );
        // undefined ⇒ inline (the /training route). An element ⇒ portal into
        // the studio panel's pinned slot. null ⇒ slot not mounted yet.
        if (actionsContainer === undefined) return actionsBlock;
        if (actionsContainer === null) return null;
        return createPortal(actionsBlock, actionsContainer);
      })()}

      {policyExtra && (
        <PolicyExtraDialog
          open={!!policyExtra}
          onOpenChange={(o) => !o && setPolicyExtra(null)}
          policyType={policyExtra.policyType}
          packageName={policyExtra.packageName}
          installTarget={policyExtra.installTarget}
          installHint={policyExtra.installHint}
          purpose="training"
          node={policyExtra.node}
        />
      )}
    </div>
  );
};

export default TrainingConfigurator;
