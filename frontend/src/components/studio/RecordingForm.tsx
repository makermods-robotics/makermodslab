import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { SessionCameraList } from "@/components/recording/CameraConfiguration";
import {
  AdvancedSection,
  RobotStatus,
} from "@/components/studio/panel/primitives";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { RobotRecord } from "@/hooks/useRobots";
import { formatRobotSetupGap } from "@/lib/robotSetupGap";
import {
  datasetNameIssue,
  formatDatasetNameIssue,
} from "@/lib/datasetName";

interface RecordingFormProps {
  robot: RobotRecord | null;
  datasetName: string;
  setDatasetName: (value: string) => void;
  singleTask: string;
  setSingleTask: (value: string) => void;
  numEpisodes: number;
  setNumEpisodes: (value: number) => void;
  episodeTimeS: number;
  setEpisodeTimeS: (value: number) => void;
  resetTimeS: number;
  setResetTimeS: (value: number) => void;
  streamingEncoding: boolean;
  setStreamingEncoding: (value: boolean) => void;
  pushToHub: boolean;
  setPushToHub: (value: boolean) => void;
  releaseStreamsRef?: React.MutableRefObject<(() => void) | null>;
}

/**
 * The recording configuration form — ported from the old landing
 * RecordingModal (its logic is preserved verbatim: name validation +
 * namespace-prefix hint, streaming-encoding toggle). Cameras are no longer
 * part of the form: the session records the selected robot's cameras, resolved
 * server-side from the robot record, so they are shown read-only here and
 * edited only in the robot settings dialog. Lifted
 * out of the dialog into the studio Collect panel and restyled to Layout D
 * tokens. Every session records a NEW dataset — appending to an existing one
 * was removed in favor of merging datasets. The Start button lives in
 * CollectPanel, pinned above the dataset library, not in this form.
 */
const RecordingForm: React.FC<RecordingFormProps> = ({
  robot,
  datasetName,
  setDatasetName,
  singleTask,
  setSingleTask,
  numEpisodes,
  setNumEpisodes,
  episodeTimeS,
  setEpisodeTimeS,
  resetTimeS,
  setResetTimeS,
  streamingEncoding,
  setStreamingEncoding,
  pushToHub,
  setPushToHub,
  releaseStreamsRef,
}) => {
  const { auth } = useHfAuth();
  const { t } = useTranslation();

  // null when the name is valid; a message otherwise (incl. empty). Mirrors the
  // backend, so Start can't fire a recording the recorder will reject.
  const nameIssue = datasetNameIssue(datasetName);
  const nameError = nameIssue ? formatDatasetNameIssue(t, nameIssue) : null;

  return (
    <div className="space-y-6">
      <p className="text-sm leading-relaxed text-muted-foreground">
        {t("studio.collect.form.intro")}
      </p>

      {/* Robot readiness — a warning, not a parameter, so no eyebrow. A ready
          robot renders nothing: the robot menu already names the selection. */}
      <RobotStatus ready={!!robot && robot.is_clean}>
        {!robot ? (
          t("studio.collect.form.noRobot")
        ) : (
          <Trans
            i18nKey="studio.collect.form.robotNotReady"
            values={{
              name: robot.name,
              gap: formatRobotSetupGap(t, robot),
            }}
            components={[<strong key="0" />]}
          />
        )}
      </RobotStatus>

      {/* Dataset parameters — flat: every control carries its own <Label>, so
          no category heading sits above them restating "Dataset". */}
      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="datasetName">{t("studio.collect.form.datasetName")}</Label>
          <Input
            id="datasetName"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            placeholder="my_dataset"
            aria-invalid={!!datasetName.trim() && nameError !== null}
            className="aria-[invalid=true]:border-destructive"
          />
          {datasetName.trim() && nameError ? (
            <p className="text-xs text-destructive">{nameError}</p>
          ) : (
            <p className="text-xs text-muted-foreground">
              <Trans
                i18nKey="studio.collect.form.nameHint"
                components={[<code key="0" />, <code key="1" />, <code key="2" />]}
              />
            </p>
          )}
          {datasetName &&
            (auth.status === "authenticated" ? (
              <p className="text-xs text-muted-foreground">
                {/* The repo id is DATA — interpolated verbatim, never split
                    across translated fragments. */}
                <Trans
                  i18nKey="studio.collect.form.savedAs"
                  values={{ repoId: `${auth.username}/${datasetName}` }}
                  components={[
                    <span key="0" className="font-mono text-foreground" />,
                  ]}
                />
              </p>
            ) : auth.status === "unauthenticated" ? (
              <p className="text-xs text-amber-700 dark:text-amber-300">
                {t("studio.collect.form.loginHint")}
              </p>
            ) : null)}
        </div>

        <div className="space-y-2">
          <Label htmlFor="singleTask">{t("studio.collect.form.task")}</Label>
          <Input
            id="singleTask"
            value={singleTask}
            onChange={(e) => setSingleTask(e.target.value)}
            placeholder={t("studio.collect.form.taskPlaceholder")}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="numEpisodes">{t("studio.collect.form.numEpisodes")}</Label>
          <NumberInput
            id="numEpisodes"
            min="1"
            max="100"
            value={numEpisodes}
            onChange={(v) => {
              if (v !== undefined) setNumEpisodes(v);
            }}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="episodeTimeS">
              {t("studio.collect.form.episodeTime")}
            </Label>
            <NumberInput
              id="episodeTimeS"
              min="1"
              value={episodeTimeS}
              onChange={(v) => {
                if (v !== undefined) setEpisodeTimeS(v);
              }}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="resetTimeS">
              {t("studio.collect.form.resetTime")}
            </Label>
            <NumberInput
              id="resetTimeS"
              min="1"
              value={resetTimeS}
              onChange={(v) => {
                if (v !== undefined) setResetTimeS(v);
              }}
            />
          </div>
        </div>
      </div>

      {/* Cameras — read-only. The session records the SELECTED ROBOT's
          cameras (the backend resolves them from the robot record), so this
          confirms what will be captured; editing happens in Robot settings. */}
      <SessionCameraList
        cameras={robot?.cameras ?? []}
        releaseStreamsRef={releaseStreamsRef}
        emptyLabel={
          robot
            ? t("studio.collect.form.camerasEmptyRobot")
            : t("studio.collect.form.camerasNoRobot")
        }
      />

      {/* Advanced */}
      <AdvancedSection summary={t("studio.collect.form.advancedSummary")}>
          <div className="flex items-start gap-3">
            <Checkbox
              id="streamingEncoding"
              checked={streamingEncoding}
              onCheckedChange={(value) => setStreamingEncoding(value === true)}
              className="mt-0.5"
            />
            <div className="space-y-1">
              <Label
                htmlFor="streamingEncoding"
                className="cursor-pointer font-medium"
              >
                {t("studio.collect.form.streamingLabel")}
              </Label>
              <p className="text-xs text-muted-foreground">
                {t("studio.collect.form.streamingHint")}
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <Checkbox
              id="pushToHub"
              checked={pushToHub}
              onCheckedChange={(value) => setPushToHub(value === true)}
              className="mt-0.5"
            />
            <div className="space-y-1">
              <Label
                htmlFor="pushToHub"
                className="cursor-pointer font-medium"
              >
                {t("studio.collect.form.pushToHubLabel")}
              </Label>
              <p className="text-xs text-muted-foreground">
                {t("studio.collect.form.pushToHubHint")}
              </p>
            </div>
          </div>
      </AdvancedSection>
    </div>
  );
};

export default RecordingForm;
