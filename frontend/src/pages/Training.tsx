import React, { useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";

import TrainingConfigurator, {
  ResumeSeed,
  FinetuneSeed,
} from "@/components/training/TrainingConfigurator";

import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import { useStudio } from "@/contexts/StudioContext";
import { useOnboarding } from "@/contexts/OnboardingContext";
import { useOnceFlag } from "@/lib/onboarding/storage";
import { trainingRouteTour } from "@/lib/onboarding/tours";

// The policy type is chosen before landing here (studio Train panel, or
// inherited by the Continue/Fine-tune flows) and arrives via router state.
// Router state is lost on a hard refresh / direct visit, so mirror the last
// resolved type in sessionStorage and fall back to it — the Policy display then
// survives a refresh instead of silently reverting to "act".
const POLICY_TYPE_STORAGE_KEY = "makermodslab.training.policyType";

function readStoredPolicyType(): string | null {
  try {
    return sessionStorage.getItem(POLICY_TYPE_STORAGE_KEY);
  } catch {
    return null;
  }
}

// Transitional /training route. Config mode now lives inside the studio Train
// panel; this wrapper keeps the same entry contract for the Continue / Resume /
// Fine-tune buttons (which navigate here with router state) by feeding that
// state into the shared TrainingConfigurator. Removed at merge cleanup.
const ConfigurationMode: React.FC = () => {
  const location = useLocation();
  const navState = location.state as {
    resume?: ResumeSeed;
    finetune?: FinetuneSeed;
    // Set by callers that pick a policy up front so it arrives preselected.
    // Direct visits have no state and fall back to the default.
    policyType?: string;
  } | null;
  const resumeSource = navState?.resume ?? null;
  const finetuneSource = navState?.finetune ?? null;
  const preselectedPolicyType = navState?.policyType ?? null;

  // Dataset is chosen upstream (single source of truth); a resume inherits the
  // source run's dataset. A fine-tune trains on a NEW dataset (the imported
  // model has none), so it uses the selection like a normal fresh run.
  const { selectedDataset } = useSelectedDataset();
  const datasetRepoId = resumeSource?.datasetRepoId ?? selectedDataset ?? "";

  const [policyType, setPolicyType] = useState<string>(
    () =>
      resumeSource?.policyType ??
      finetuneSource?.policyType ??
      preselectedPolicyType ??
      readStoredPolicyType() ??
      "act",
  );

  const { seen: hasSeenTrainingRouteTour, markSeen: markTrainingRouteTourSeen } =
    useOnceFlag("makerlab:seen-training-tour");
  const { start: startTour } = useOnboarding();

  // First visit to this route, walk through the single-step reveal.
  useEffect(() => {
    if (!hasSeenTrainingRouteTour) {
      startTour(trainingRouteTour, markTrainingRouteTourSeen);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Mirror the resolved policy type so a refresh — which drops router state —
  // restores the same choice via readStoredPolicyType().
  useEffect(() => {
    try {
      sessionStorage.setItem(POLICY_TYPE_STORAGE_KEY, policyType);
    } catch {
      // storage unavailable (private mode) — refresh falls back to default
    }
  }, [policyType]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-3xl px-4 py-6" data-tour="training-configurator">
        <TrainingConfigurator
          // Same composite the studio's Train panel keys on: the seeds are read
          // once as initial state, so the step (and a resume's checkpoint
          // owner) has to be in the key too — a second seed for the same run
          // would otherwise update the banner and the step validation while the
          // payload kept the first pick.
          key={`${resumeSource?.jobId ?? ""}@${resumeSource?.step ?? ""}@${
            resumeSource?.checkpointJobId ?? ""
          }::${finetuneSource?.jobId ?? ""}@${finetuneSource?.step ?? ""}`}
          policyType={policyType}
          onPolicyTypeChange={setPolicyType}
          datasetRepoId={datasetRepoId}
          resumeSeed={resumeSource}
          finetuneSeed={finetuneSource}
        />
      </div>
    </div>
  );
};

// The job monitor is no longer a page — it's TrainingJobDialog over the
// studio's Train panel. Deep links / refreshes / stale bookmarks onto
// /training/:jobId re-open that dialog on the Launchpad instead.
const MonitorRedirect: React.FC<{ jobId: string }> = ({ jobId }) => {
  const navigate = useNavigate();
  const { openJobMonitor } = useStudio();

  useEffect(() => {
    openJobMonitor(jobId);
    navigate("/", { replace: true });
  }, [jobId, openJobMonitor, navigate]);

  return null;
};

const Training: React.FC = () => {
  const { jobId } = useParams<{ jobId?: string }>();
  return jobId ? <MonitorRedirect jobId={jobId} /> : <ConfigurationMode />;
};

export default Training;
