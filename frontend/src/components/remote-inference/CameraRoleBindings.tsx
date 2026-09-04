import React from "react";
import { useTranslation } from "react-i18next";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

/**
 * One control per checkpoint camera ROLE that nothing on the robot answers to.
 *
 * The panel binds a checkpoint's cameras to the robot's BY NAME, and that stays
 * the default: a camera's name in the robot record is its identity, not a label
 * to be rewritten for whichever policy is being run today. A checkpoint's camera
 * name is a role — `cam0` / `cam1` on `lerobot/MolmoAct2-SO100_101-LeRobot`,
 * `observation.images.base_0_rgb` on a pi05 foundation checkpoint — and a role
 * that matches no camera by name is not an error in the record, it is a
 * question only the operator can answer.
 *
 * So: matched roles render NOTHING (there is no decision to make, and a
 * dropdown showing the only possible answer is a chance to get it wrong), and
 * when every role matches this component renders nothing at all.
 *
 * Presentation only, deliberately. It takes the slots and the options already
 * derived and hands back a name — no storage, no binding derivation, no run
 * mode. Only the remote run is wired to it today (S3.7b), but nothing here is
 * remote-specific, so the local rollout can adopt it as-is when its own
 * name-only binding needs the same escape hatch.
 */

/** A robot camera, as an option. */
export interface CameraRoleOption {
  /** The robot record's own spelling — the value the start request carries. */
  name: string;
  width: number;
  height: number;
  /** Enumerated right now. An unplugged camera is still offered (it is the
   * right answer once it is plugged back in) but says so. */
  connected: boolean;
}

/** One unmatched checkpoint camera. */
export interface CameraRoleSlot {
  /** The key the start request's `camera_bindings` uses. */
  requestKey: string;
  /** The checkpoint's own camera name. DATA — never translated. */
  display: string;
  /** What the checkpoint captures at, when its config says. */
  dims?: { width: number; height: number };
  /** The robot camera currently bound to this role, if any. */
  selected: string | null;
}

/** Radix rejects an empty option value, so "no camera" needs a sentinel. It is
 * mapped back to `null` before it can reach a request. */
const UNBOUND = "__unbound__";

const CameraRoleBindings: React.FC<{
  /** ONLY the roles with no name match — a matched role has no control. */
  slots: CameraRoleSlot[];
  cameras: CameraRoleOption[];
  /** `null` clears the binding. */
  onChange: (requestKey: string, cameraName: string | null) => void;
  /** How many roles bound themselves by name; drives the one-line note. */
  nameMatchedCount: number;
  disabled?: boolean;
}> = ({ slots, cameras, onChange, nameMatchedCount, disabled }) => {
  const { t } = useTranslation();
  if (slots.length === 0) return null;

  return (
    <div className="space-y-3 rounded-lg border border-border bg-muted/40 p-3">
      <p className="text-xs font-semibold text-foreground">
        {t("remoteInference.cameraRoles.title")}
      </p>
      <p className="text-xs leading-relaxed text-muted-foreground">
        {t("remoteInference.cameraRoles.hint")}
      </p>
      {nameMatchedCount > 0 ? (
        <p className="text-xs text-muted-foreground">
          {t("remoteInference.cameraRoles.nameMatched", {
            count: nameMatchedCount,
          })}
        </p>
      ) : null}
      {slots.map((slot) => {
        const bound = cameras.find((c) => c.name === slot.selected) ?? null;
        return (
          <div key={slot.requestKey} className="space-y-1.5">
            {/* The role name is the CHECKPOINT's camera key — data, shown
                verbatim in every language. */}
            <Label
              htmlFor={`remote-camera-role-${slot.requestKey}`}
              className="font-mono text-xs"
            >
              {slot.display}
            </Label>
            {slot.dims ? (
              <p className="text-xs text-muted-foreground">
                {t("remoteInference.cameraRoles.capturesAt", {
                  width: slot.dims.width,
                  height: slot.dims.height,
                })}
              </p>
            ) : null}
            <Select
              value={slot.selected ?? UNBOUND}
              disabled={disabled}
              onValueChange={(v) =>
                onChange(slot.requestKey, v === UNBOUND ? null : v)
              }
            >
              <SelectTrigger id={`remote-camera-role-${slot.requestKey}`}>
                <SelectValue
                  placeholder={t("remoteInference.cameraRoles.unbound")}
                />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={UNBOUND}>
                  {t("remoteInference.cameraRoles.unbound")}
                </SelectItem>
                {/* Camera NAMES are the robot record's own keys and the value
                    the request carries — data, not copy. */}
                {cameras.map((cam) => (
                  <SelectItem key={cam.name} value={cam.name}>
                    {cam.name} — {cam.width}×{cam.height}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {cameras.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                {t("remoteInference.cameraRoles.noCameras")}
              </p>
            ) : null}
            {bound && !bound.connected ? (
              <p className="text-xs text-destructive">
                {t("remoteInference.cameraRoles.disconnected")}
              </p>
            ) : null}
          </div>
        );
      })}
      <p className="text-xs leading-relaxed text-muted-foreground">
        {t("remoteInference.cameraRoles.identityNote")}
      </p>
    </div>
  );
};

export default CameraRoleBindings;
