import React from "react";
import { useTranslation } from "react-i18next";
import { Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
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
 *
 * S3.8g adds the other direction: a role the checkpoint does NOT declare. Some
 * checkpoints fix their view count in their lerobot wrapper rather than in the
 * model — `lerobot/MolmoAct2-SO100_101-LeRobot` declares two images where the
 * allenai model underneath takes a list of any length — and the GPU side can be
 * asked to declare another before the weights load. That control appears ONLY
 * when the server says this checkpoint's policy family can take one
 * (`canAddRoles`), and an added role is an ordinary slot in every other
 * respect: it needs a camera, and an unbound one blocks Start exactly as an
 * unmatched checkpoint role does.
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
  /** An operator-added view the checkpoint never declared (S3.8g). Renders a
   * remove control and the "untested" note; false for every checkpoint role. */
  extra?: boolean;
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
  /** The selected checkpoint's policy family can take extra views (S3.8g).
   * False — the default, and what a server too old to say gets — hides the add
   * control entirely; adding a view to a policy whose vision tower is fixed is
   * a shape error inside a container after a paid cold start, not a degraded
   * run. */
  canAddRoles?: boolean;
  /** Adds the next free `cam<N>`. Absent ⇒ no add control. */
  onAddRole?: () => void;
  /** Drops one added role (and its binding). Absent ⇒ no remove control. */
  onRemoveRole?: (role: string) => void;
  /** Already at `MAX_EXTRA_CAMERA_ROLES`: the control stays visible and
   * disabled, with the reason under it, rather than vanishing. */
  addRolesFull?: boolean;
  disabled?: boolean;
}> = ({
  slots,
  cameras,
  onChange,
  nameMatchedCount,
  canAddRoles,
  onAddRole,
  onRemoveRole,
  addRolesFull,
  disabled,
}) => {
  const { t } = useTranslation();
  const showAdd = Boolean(canAddRoles && onAddRole);
  // The add control is the one thing here that has to render when there is
  // nothing else to show: on a checkpoint whose every role matched by name,
  // "add a third camera" is still an offer worth making.
  if (slots.length === 0 && !showAdd) return null;

  return (
    <div className="space-y-3 rounded-lg border border-border bg-muted/40 p-3">
      <p className="text-xs font-semibold text-foreground">
        {t("remoteInference.cameraRoles.title")}
      </p>
      {slots.length > 0 ? (
        <p className="text-xs leading-relaxed text-muted-foreground">
          {t("remoteInference.cameraRoles.hint")}
        </p>
      ) : null}
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
            {/* The role name is the CHECKPOINT's camera key (or, for an added
                one, the name the panel minted) — data, shown verbatim in every
                language. */}
            <div className="flex items-center justify-between gap-2">
              <Label
                htmlFor={`remote-camera-role-${slot.requestKey}`}
                className="font-mono text-xs"
              >
                {slot.display}
              </Label>
              {slot.extra && onRemoveRole ? (
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  disabled={disabled}
                  onClick={() => onRemoveRole(slot.requestKey)}
                  className="h-6 gap-1 px-1.5 text-xs text-muted-foreground"
                  aria-label={t("remoteInference.cameraRoles.removeRole", {
                    role: slot.display,
                  })}
                >
                  <X className="h-3 w-3" />
                  {t("remoteInference.cameraRoles.remove")}
                </Button>
              ) : null}
            </div>
            {slot.extra ? (
              <p className="text-xs text-muted-foreground">
                {t("remoteInference.cameraRoles.extraBadge")}
              </p>
            ) : null}
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
      {showAdd ? (
        <div className="space-y-1.5 border-t border-border pt-3">
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={disabled || addRolesFull}
            onClick={onAddRole}
            className="h-7 gap-1.5 px-2 text-xs"
          >
            <Plus className="h-3 w-3" />
            {t("remoteInference.cameraRoles.addRole")}
          </Button>
          <p className="text-xs leading-relaxed text-muted-foreground">
            {addRolesFull
              ? t("remoteInference.cameraRoles.addRoleFull")
              : t("remoteInference.cameraRoles.addRoleHint")}
          </p>
        </div>
      ) : null}
      {slots.length > 0 ? (
        <p className="text-xs leading-relaxed text-muted-foreground">
          {t("remoteInference.cameraRoles.identityNote")}
        </p>
      ) : null}
    </div>
  );
};

export default CameraRoleBindings;
