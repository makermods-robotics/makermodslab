import React from "react";
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
import { RobotRecord, robotSetupGap } from "@/hooks/useRobots";
import { validateDatasetName } from "@/lib/datasetName";

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

  // null when the name is valid; a message otherwise (incl. empty). Mirrors the
  // backend, so Start can't fire a recording the recorder will reject.
  const nameError = validateDatasetName(datasetName);

  return (
    <div className="space-y-6">
      <p className="text-sm leading-relaxed text-muted-foreground">
        Name the dataset and set the capture parameters, then start recording
        on the selected robot.
      </p>

      {/* Robot readiness — a warning, not a parameter, so no eyebrow. A ready
          robot renders nothing: the robot menu already names the selection. */}
      <RobotStatus ready={!!robot && robot.is_clean}>
        {!robot ? (
          <>
            Select or create a robot before recording — use the robot menu in
            the top-right corner of this window.
          </>
        ) : (
          <>
            <strong>{robot.name}</strong> {robotSetupGap(robot)}. Open Robot
            settings before recording.
          </>
        )}
      </RobotStatus>

      {/* Dataset parameters — flat: every control carries its own <Label>, so
          no category heading sits above them restating "Dataset". */}
      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="datasetName">Dataset name *</Label>
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
              Letters, numbers, <code>.</code> <code>_</code> <code>-</code>{" "}
              only; start and end with a letter or number.
            </p>
          )}
          {datasetName &&
            (auth.status === "authenticated" ? (
              <p className="text-xs text-muted-foreground">
                Will be saved as{" "}
                <span className="font-mono text-foreground">
                  {auth.username}/{datasetName}
                </span>
              </p>
            ) : auth.status === "unauthenticated" ? (
              <p className="text-xs text-amber-700 dark:text-amber-300">
                Log in to Hugging Face to set the repository owner.
              </p>
            ) : null)}
        </div>

        <div className="space-y-2">
          <Label htmlFor="singleTask">Task description *</Label>
          <Input
            id="singleTask"
            value={singleTask}
            onChange={(e) => setSingleTask(e.target.value)}
            placeholder="e.g., pick up the red block and place it on the blue square"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="numEpisodes">Number of episodes</Label>
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
            <Label htmlFor="episodeTimeS">Episode duration (s)</Label>
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
            <Label htmlFor="resetTimeS">Reset duration (s)</Label>
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
            ? "This robot has no cameras. Add them in Robot settings to record video."
            : "Select a robot to see its cameras."
        }
      />

      {/* Advanced */}
      <AdvancedSection summary="Streaming encoding, push to Hub">
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
                Streaming video encoding
              </Label>
              <p className="text-xs text-muted-foreground">
                Encodes frames in real time during capture so each episode saves
                almost instantly. Uncheck to fall back to the slower
                PNG-then-encode flow.
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
                Push to Hugging Face Hub
              </Label>
              <p className="text-xs text-muted-foreground">
                Uploads the dataset to your Hugging Face account in the
                background once the session ends. Uncheck to keep it local —
                you can still upload it later from the dataset library.
              </p>
            </div>
          </div>
      </AdvancedSection>
    </div>
  );
};

export default RecordingForm;
