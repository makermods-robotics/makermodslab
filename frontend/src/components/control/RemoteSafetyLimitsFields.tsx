import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export interface RemoteSafetyLimits {
  action_watchdog_ms: number;
  first_action_deadline_ms: number;
  control_deadline_ms: number;
  browser_deadline_ms: number;
  max_velocity_per_s: number;
  max_acceleration_per_s2: number;
}

interface NumericLimit {
  key: keyof RemoteSafetyLimits;
  label: string;
  min: number;
  max: number;
  step: number;
}

export interface RemoteSafetyLimitsFieldsProps {
  value: RemoteSafetyLimits;
  onChange: (value: RemoteSafetyLimits) => void;
  disabled?: boolean;
}

/** Explicit commissioned limits; the backend remains the authority on validity. */
export default function RemoteSafetyLimitsFields({
  value,
  onChange,
  disabled = false,
}: RemoteSafetyLimitsFieldsProps) {
  const { t } = useTranslation();
  const fields: NumericLimit[] = [
    {
      key: "action_watchdog_ms",
      label: t("remoteTeleop.configuration.actionWatchdog"),
      min: 20,
      max: 2000,
      step: 1,
    },
    {
      key: "first_action_deadline_ms",
      label: t("remoteTeleop.configuration.firstActionDeadline"),
      min: Math.max(20, value.action_watchdog_ms),
      max: 5000,
      step: 1,
    },
    {
      key: "control_deadline_ms",
      label: t("remoteTeleop.configuration.controlDeadline"),
      min: 100,
      max: 5000,
      step: 1,
    },
    {
      key: "browser_deadline_ms",
      label: t("remoteTeleop.configuration.browserDeadline"),
      min: Math.max(100, value.control_deadline_ms),
      max: 10000,
      step: 1,
    },
    {
      key: "max_velocity_per_s",
      label: t("remoteTeleop.configuration.maxVelocity"),
      min: 0.01,
      max: 10000,
      step: 0.01,
    },
    {
      key: "max_acceleration_per_s2", // gitleaks:allow -- field name, not a credential
      label: t("remoteTeleop.configuration.maxAcceleration"),
      min: 0.01,
      max: 100000,
      step: 0.01,
    },
  ];

  return (
    <fieldset className="space-y-3 rounded-lg border border-border p-3">
      <legend className="px-1 text-sm font-medium">
        {t("remoteTeleop.configuration.safetyLimits")}
      </legend>
      <p className="text-xs text-muted-foreground">
        {t("remoteTeleop.configuration.safetyLimitsDescription")}
      </p>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {fields.map((field) => (
          <div key={field.key} className="space-y-1.5">
            <Label htmlFor={`remote-${field.key}`}>{field.label}</Label>
            <Input
              id={`remote-${field.key}`}
              type="number"
              min={field.min}
              max={field.max}
              step={field.step}
              value={value[field.key]}
              onChange={(event) =>
                onChange({
                  ...value,
                  [field.key]: Number(event.target.value),
                })
              }
              disabled={disabled}
              required
            />
          </div>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        {t("remoteTeleop.configuration.deadlineRelationship")}
      </p>
    </fieldset>
  );
}
