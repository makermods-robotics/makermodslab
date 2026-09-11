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
import { ConfigComponentProps, POLICY_TYPE_OPTIONS } from "../types";

/**
 * The policy-architecture picker — the single place a policy type is chosen.
 * Extracted verbatim from EssentialsCard so it can render directly after the
 * Train panel's "Starting point": the two together answer "what is this run
 * built from", which the old ordering split across two eyebrow sections.
 *
 * `policyLocked` disables it when a starting point / resume seed fixes the
 * architecture.
 *
 * The option values are wire identifiers and their labels are product names
 * (ACT, SmolVLA, Diffusion Policy) — neither is translated.
 */
const PolicyField: React.FC<
  ConfigComponentProps & { policyLocked?: boolean }
> = ({ config, updateConfig, policyLocked }) => {
  const { t } = useTranslation();
  return (
    <div className="space-y-2">
      <Label htmlFor="policy_type">{t("training.policyField.label")}</Label>
      <Select
        value={config.policy_type || undefined}
        onValueChange={(value) => updateConfig("policy_type", value)}
        disabled={policyLocked}
      >
        <SelectTrigger id="policy_type">
          <SelectValue placeholder={t("training.policyField.placeholder")} />
        </SelectTrigger>
        <SelectContent>
          {POLICY_TYPE_OPTIONS.map((policy) => (
            <SelectItem key={policy.value} value={policy.value}>
              {policy.display}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        {policyLocked
          ? t("training.policyField.hintLocked")
          : t("training.policyField.hint")}
      </p>
    </div>
  );
};

export default PolicyField;
