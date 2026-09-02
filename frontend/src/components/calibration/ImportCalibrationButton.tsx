import React, { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import type { ArmType } from "@/hooks/useRobots";

interface ImportCalibrationButtonProps {
  /** API device vocabulary: "teleop" (leader) or "robot" (follower). */
  device: "teleop" | "robot";
  /** Which calibration library to upload into — the SO-101 and Maker pairs
   * keep separate directories, so a file imported for one arm type is
   * invisible to the other. */
  armType: ArmType;
  /** Called with the saved config name after a successful import. */
  onImported?: (name: string) => void;
  /**
   * When set, no trigger button renders; instead the file-picker opener is
   * written into this ref so the caller can trigger it from its own control
   * (a menu item). The hidden input and the name dialog stay mounted HERE —
   * inside a dropdown menu they would unmount the moment the menu closed,
   * which is exactly when the dialog needs to appear.
   */
  pickRef?: React.MutableRefObject<(() => void) | null>;
}

/**
 * Import a raw lerobot calibration JSON into a side's config library.
 * Reads + parses the file client-side, then POSTs {name, data} to the upload
 * endpoint which shape-validates and never overwrites (409 → rename prompt).
 */
const ImportCalibrationButton: React.FC<ImportCalibrationButtonProps> = ({
  device,
  armType,
  onImported,
  pickRef,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { t } = useTranslation();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [data, setData] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const pickFile = () => fileInputRef.current?.click();
  if (pickRef) pickRef.current = pickFile;

  const handleFileChosen = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    // Reset the input so re-choosing the same file fires onChange again.
    e.target.value = "";
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      setData(parsed);
      // Default the name to the file's stem; user can edit before importing.
      setName(file.name.replace(/\.json$/i, ""));
      setError(null);
      setOpen(true);
    } catch {
      toast({
        title: t("calibration.import.invalidJsonTitle"),
        description: t("calibration.import.invalidJsonDescription"),
        variant: "destructive",
      });
    }
  };

  const submit = async () => {
    const trimmed = name.trim();
    if (!trimmed) {
      setError(t("calibration.import.emptyName"));
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/api/v1/calibration-configs/${device}/upload?arm_type=${armType}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: trimmed, data }),
        },
      );
      const body = await res.json().catch(() => ({}));
      if (res.ok && body.success) {
        toast({
          title: t("calibration.import.importedTitle"),
          // `body.name` is the saved file name — data, rendered verbatim.
          description: t("calibration.import.imported", { name: body.name }),
        });
        setOpen(false);
        onImported?.(body.name);
        return;
      }
      // 409 (collision) and 400 (validation) keep the dialog open with the
      // message so the user can rename / fix and retry. The backend's message
      // is English prose we pass through; only the fallback is translated.
      setError(body.message || t("calibration.import.failed"));
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  // `device` stays the API vocabulary ("teleop" / "robot"); only the label
  // shown for the arm side is localized. Each side gets its own whole phrase
  // rather than a translated word spliced into a sentence.
  const isLeader = device === "teleop";
  const importLabel = isLeader
    ? t("calibration.import.labelLeader")
    : t("calibration.import.labelFollower");
  const importDescription = isLeader
    ? t("calibration.import.descriptionLeader")
    : t("calibration.import.descriptionFollower");

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,application/json"
        className="hidden"
        onChange={handleFileChosen}
      />
      {!pickRef && (
        <Button
          size="icon"
          variant="ghost"
          className="shrink-0 text-muted-foreground hover:text-foreground"
          onClick={pickFile}
          aria-label={importLabel}
          title={importLabel}
        >
          <Upload className="h-4 w-4" />
        </Button>
      )}

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{importLabel}</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              {importDescription}
            </DialogDescription>
          </DialogHeader>
          <Input
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              setError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void submit();
              }
            }}
            autoFocus
            placeholder={t("calibration.import.placeholder")}
          />
          {error && <p className="text-sm text-destructive">{error}</p>}
          <DialogFooter className="flex gap-2 justify-end">
            <Button variant="outline" onClick={() => setOpen(false)}>
              {t("common.cancel")}
            </Button>
            <Button disabled={busy || !name.trim()} onClick={submit}>
              {busy
                ? t("calibration.import.submitting")
                : t("calibration.import.submit")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default ImportCalibrationButton;
