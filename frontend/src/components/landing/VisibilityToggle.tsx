import React from "react";
import { useTranslation } from "react-i18next";
import { Globe, Lock } from "lucide-react";

/**
 * Segmented Public|Private visibility toggle (Globe/Lock, active side filled).
 * Shared by UploadDatasetDialog (upload-time visibility) and DatasetInfoCard's
 * post-upload editor so both render the identical control. `value` is the
 * PRIVATE flag (true = Private selected). `idBase` seeds the aria ids so several
 * toggles on one page stay distinct.
 */
const VisibilityToggle: React.FC<{
  value: boolean;
  onChange: (isPrivate: boolean) => void;
  idBase: string;
  disabled?: boolean;
}> = ({ value, onChange, idBase, disabled = false }) => {
  const { t } = useTranslation();
  return (
    <div
      role="radiogroup"
      aria-labelledby={idBase}
      className="flex rounded-md border border-border bg-muted p-0.5"
    >
      <button
        type="button"
        role="radio"
        aria-checked={!value}
        disabled={disabled}
        onClick={() => onChange(false)}
        className={`flex flex-1 items-center justify-center gap-1.5 rounded px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
          !value ? "bg-background text-foreground shadow-1" : "text-muted-foreground hover:text-foreground"
        }`}
      >
        <Globe className="h-3 w-3" />
        {t("landing.visibilityToggle.public")}
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={value}
        disabled={disabled}
        onClick={() => onChange(true)}
        className={`flex flex-1 items-center justify-center gap-1.5 rounded px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
          value ? "bg-background text-foreground shadow-1" : "text-muted-foreground hover:text-foreground"
        }`}
      >
        <Lock className="h-3 w-3" />
        {t("landing.visibilityToggle.private")}
      </button>
    </div>
  );
};

export default VisibilityToggle;
