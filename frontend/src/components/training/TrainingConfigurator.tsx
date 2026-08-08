import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { getDatasetInfo } from "@/lib/replayApi";
import { validateModelName } from "@/lib/datasetName";
import { proposeModelName } from "@/lib/modelNames";

// Passed by the "Continue" button on a completed local job, or the "Resume"
// button on a cloud run that ended before its step target.
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
  // resolves to HF_JOB_TIMEOUT (see makermodslab/runners/hf_cloud.py) —
  // silently capping the continuation of a run that had already proved it
  // needs a longer budget. See NEW-12.
  hfJobTimeout?: string;
  // The parent's dataloader worker count — a starting point, not a lock: the
  // resume branch DOES pass --num_workers, so an edit here really takes effect
  // (a continuation can run on a host with different capacity than the parent).
  numWorkers?: number;
  // The remaining hyperparameters. lerobot rebuilds these from the checkpoint's
  // train_config.json on resume (build_training_command's resume branch emits
  // none of them), so carrying them forward does NOT change what trains — it
  // stops the form from *displaying* fresh-run defaults for a continuation, and
  // keeps the new JobRecord's persisted config truthful about the run's shape.
  batchSize?: number;
  seed?: number;
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
  /** A "Continue"/"Resume" seed — inherits the source run's target + cadence. */
  resumeSeed?: ResumeSeed | null;
  /** A "Fine-tune" seed — fresh run initialized from a source checkpoint. */
  finetuneSeed?: FinetuneSeed | null;
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

function configToRequest(
  c: TrainingConfig,
  checkpointUploadKind: CheckpointUploadKind,
): TrainingRequest {
  // The backend's TrainingRequest has more optional fields; the form covers
  // the user-meaningful subset.
  return {
    target: c.target,
    dataset_repo_id: c.dataset_repo_id,
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
  resumeSeed = null,
  finetuneSeed = null,
  onStarted,
  actionsContainer,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { auth } = useHfAuth();
  const { toast } = useToast();
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
    // ResumeSeed comments for which of them lerobot actually honours. The ones
    // it does NOT honour render read-only on a resume (`resumeLocked` below);
    // the values are still prefilled and still sent, so JobRecord.config keeps
    // describing the run's real shape — the lock is a display-honesty change,
    // not a payload change. What stays editable is what a continuation can
    // genuinely change: steps, the log/save cadence, the dataloader worker
    // count, and the cloud budget.
    // `steps` is inherited as-is like the rest, so resuming an interrupted run
    // defaults to FINISHING its original target (one that died at step 4,814
    // of 15,000 resumes toward 15,000); raise it by hand to train past that.
    // Resuming a run that already reached its target therefore prefills steps
    // equal to the checkpoint's step — `resumeStepError` below blocks Start
    // until the user raises it, as does the backend. Fine-tune is a fresh run,
    // so it uses the normal fresh defaults throughout.
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
    policy_device: resumeSeed?.policyDevice ?? "auto",
    policy_use_amp: resumeSeed?.policyUseAmp ?? false,
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
    // instead of silently falling back to the runner's default; the field
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
    }),
    [trainingConfig, policyType, controlledDatasetRepoId],
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
  } | null>(null);
  const [authenticated, setAuthenticated] = useState<boolean>(false);
  const [flavors, setFlavors] = useState<RunnerFlavor[]>([]);
  const [hardwareLoading, setHardwareLoading] = useState(true);
  // HF_HUB_OFFLINE on the backend: Hub writes (incl. dataset upload) disabled.
  const [offline, setOffline] = useState<boolean>(false);

  useEffect(() => {
    fetchWithHeaders(`${baseUrl}/system/training-extra`)
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

  // Whether the user has touched the Model name. Once they have, the prefill
  // below stops writing — swapping the dataset must never overwrite a name
  // someone typed. A ref, not state: nothing renders from it.
  const nameEdited = useRef(false);

  const updateConfig = <T extends keyof TrainingConfig>(
    key: T,
    value: TrainingConfig[T],
  ) => {
    if (key === "job_name") nameEdited.current = true;
    // policy_type is controlled by the caller; route edits (EssentialsCard's
    // dropdown) back up. dataset_repo_id is controlled + display-only, but
    // guard it too so a stray write can't desync.
    if (key === "policy_type") {
      onPolicyTypeChange(value as string);
      return;
    }
    if (key === "dataset_repo_id") return;
    setTrainingConfig((prev) => ({ ...prev, [key]: value }));
  };

  // Prefill the Model name from what the run is ABOUT, so the required field
  // costs zero keystrokes when the obvious name is the right one: the dataset's
  // task stem for a fresh run ("makermods/eraser_stack_20260718_175437" →
  // "eraser_stack"), the source model's stem for a fine-tune (a continuation of
  // that skill on new data, so its name is the better starting point). Re-runs
  // whenever the selection changes — a name proposed for the dataset the user
  // has since swapped away from is worse than no proposal — but only while the
  // field is untouched. A resume never proposes: it keeps the parent run's
  // identity and the field is locked (see EssentialsCard).
  const finetuneSourceName = finetuneSeed?.name;
  useEffect(() => {
    if (resumeSeed || nameEdited.current) return;
    const proposed =
      (finetuneSourceName ? proposeModelName(finetuneSourceName) : "") ||
      proposeModelName(controlledDatasetRepoId);
    setTrainingConfig((prev) =>
      prev.job_name === proposed ? prev : { ...prev, job_name: proposed },
    );
  }, [controlledDatasetRepoId, finetuneSourceName, resumeSeed]);

  // Cloud training runs from the Hub, so a dataset that only exists in this
  // machine's local cache must be uploaded first. We read the chosen dataset's
  // `source` from the /datasets listing (already carries it) and, for a
  // local-only + cloud combo, chain an upload before the job launches.
  const { datasets } = useDatasets();
  const datasetRepoId = config.dataset_repo_id.trim();
  // null when the model name is usable; a message otherwise (incl. empty).
  // Mirrors the backend's validate_model_name — a name that fails here is a 400
  // there, so gating Start on it turns a server refusal into a visible field.
  // A resume has no name to choose (it keeps the parent's identity), so the
  // requirement doesn't apply.
  const modelNameError = resumeSeed
    ? null
    : validateModelName(config.job_name || "");
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
      : finetuneSeed?.checkpointSource === "local" && isCloud
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
      toast({ title: "Training Started", description: job.name });
      onStarted?.(job.id);
      // The monitor is a dialog over the studio's Train panel, not a route.
      // openJobMonitor opens the studio; off-Launchpad callers (the
      // transitional /training route) must also land on "/" to see it.
      openJobMonitor(job.id);
      if (location.pathname !== "/") navigate("/");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      toast({ title: "Error", description: msg, variant: "destructive" });
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
        title: "Upload failed",
        description: message,
        variant: "destructive",
      });
    },
  });

  const handleStart = async () => {
    if (!datasetRepoId) {
      toast({
        title: "Error",
        description: "Dataset repository ID is required",
        variant: "destructive",
      });
      return;
    }

    // The name is the run's identity, so it is required and must be a legal
    // repo-id segment — the same gate CollectPanel puts in front of a
    // recording. The Start button is already disabled on this condition; this
    // catches the keyboard/enter path and states the reason.
    if (modelNameError) {
      toast({
        title: "Invalid model name",
        description: modelNameError,
        variant: "destructive",
      });
      return;
    }

    // Pre-flight: smolvla/pi0/diffusion need an optional package installed
    // locally. Catch it here with a one-click installer instead of a buried
    // ImportError after the job has already started. Cloud jobs run in their
    // own environment, so the local package is irrelevant — skip the check.
    if (config.target.runner === "local") {
      try {
        const r = await fetchWithHeaders(
          `${baseUrl}/system/policy-extra/${config.policy_type}`,
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
          title: "Upload failed",
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
        Checking training environment…
      </div>
    );
  }

  if (trainingExtraAvailable === false) {
    return <TrainingExtraGate installHint={trainingExtraInstallHint} />;
  }

  const targetRequiresAuth = config.target.runner === "hf_cloud";
  const targetMissingFlavor =
    config.target.runner === "hf_cloud" && !config.target.flavor;
  const localBlocked = config.target.runner === "local" && localJobRunning;
  // When resuming, total steps must be strictly above the checkpoint's step:
  // equal trains nothing and lerobot requires --steps above the resumed step.
  const resumeStepError =
    resumeSeed != null &&
    resumeSeed.step != null &&
    config.steps <= resumeSeed.step
      ? `Total steps must be greater than the checkpoint's step (${resumeSeed.step.toLocaleString()}).`
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
    modelNameError != null ||
    localBlocked ||
    (targetRequiresAuth && !authenticated) ||
    targetMissingFlavor ||
    uploadBlockedOffline ||
    checkpointUploadBlockedOffline ||
    resumeStepError != null;
  const blockedTooltip = localBlocked
    ? "Another local training is already running"
    : targetRequiresAuth && !authenticated
      ? "Log in to Hugging Face to use cloud compute"
      : targetMissingFlavor
        ? "Select a hardware flavor"
        : uploadBlockedOffline
          ? "Offline mode is on — the dataset can't be uploaded to the Hub"
          : checkpointUploadBlockedOffline
            ? "Offline mode is on — the checkpoint can't be uploaded to the Hub"
            : undefined;
  // The name field can sit far above a portaled Start button (the studio panel
  // pins it at the panel's foot), so its reason travels with the button — and
  // it takes precedence: an unnamed run can't start whatever else is true.
  const startTooltip = modelNameError ?? blockedTooltip;

  return (
    <div className="w-full">
      <HfAuthBanner />
      {resumeSeed ? (
        <div className="mb-4 rounded-lg border border-primary/40 bg-primary/5 p-4 text-sm text-foreground">
          <div className="font-semibold">
            Continuing “{resumeSeed.name}”
            {resumeSeed.step != null
              ? ` from step ${resumeSeed.step.toLocaleString()}`
              : " from its latest checkpoint"}
          </div>
          <p className="mt-1 text-muted-foreground">
            {datasetRepoId ? (
              <>
                Training continues on{" "}
                <span className="font-mono text-xs">{datasetRepoId}</span> with
                the parent run's
              </>
            ) : (
              <>The parent run's dataset,</>
            )}{" "}
            policy, batch size, seed, device, optimizer and W&amp;B logging —
            lerobot rebuilds all of them from the checkpoint, so they're shown
            read-only below. A resume continues the same experiment; to train with
            different settings, fine-tune from this checkpoint instead.{" "}
            <span className="font-medium">Steps</span>, the log and checkpoint
            cadence, the dataloader worker count
            {isCloud ? ", the hardware, and the job timeout" : ""} do still
            apply: set Steps above the resumed step to train further (prefilled
            to {config.steps.toLocaleString()}).
          </p>
          {isCloud ? (
            <p className="mt-1 text-muted-foreground">
              Job timeout:{" "}
              <span className="font-medium">
                {config.hf_job_timeout?.trim()
                  ? config.hf_job_timeout
                  : "24h (default)"}
              </span>{" "}
              — a continuation needs at least as long as the tail it has left.
            </p>
          ) : null}
        </div>
      ) : null}
      {finetuneSeed ? (
        <div className="mb-4 rounded-lg border border-border bg-muted/50 p-4 text-sm text-foreground">
          <div className="font-semibold">
            Fine-tuning from “{finetuneSeed.name}”
            {finetuneSeed.step != null
              ? ` (step ${finetuneSeed.step.toLocaleString()})`
              : " (latest checkpoint)"}
          </div>
          <p className="mt-1 text-muted-foreground">
            This starts a <span className="font-medium">fresh run</span> (new
            optimizer, from step 0) with the policy weights initialized from that
            model. Pick a <span className="font-medium">dataset</span> to train
            on and set your training parameters as usual.
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
          <LocalCheckpointCloudNotice
            mode="finetune"
            runName={finetuneSeed.name}
            step={finetuneSeed.step}
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
                  <Loader2 className="h-4 w-4 animate-spin" /> Uploading…
                </>
              ) : isStarting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Starting…
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />{" "}
                  {/* Sentence case, matching the disabled stand-in in
                      TrainPanel and Collect's / Run's Start buttons — these
                      used to flip to Title Case the moment the form opened. */}
                  {resumeSeed
                    ? needsCheckpointUpload
                      ? "Upload & continue training"
                      : "Continue training"
                    : finetuneSeed
                      ? // A local base fine-tuned on the cloud has to push its
                        // weights first, so the button says what the click
                        // actually does — the same promise the resume branch
                        // above makes.
                        needsCheckpointUpload
                        ? "Upload & start training"
                        : "Start fine-tuning"
                      : needsUpload
                        ? "Upload & start training"
                        : "Start training"}
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
        />
      )}
    </div>
  );
};

export default TrainingConfigurator;
