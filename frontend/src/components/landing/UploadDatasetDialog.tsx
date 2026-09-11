import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Loader2, Upload as UploadIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import VisibilityToggle from "@/components/landing/VisibilityToggle";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useToast } from "@/hooks/use-toast";

/**
 * Confirm-and-upload popover for pushing a locally-cached dataset to the Hub.
 * Private-by-default toggle (with the camera-footage note) + optional
 * comma-separated tags. The upload runs in the background (see UploadManager /
 * useDatasetUpload): this dialog only kicks it off and closes — the caller's
 * card/row then shows the polled "Uploading…" state and fires the success /
 * error toast on completion. A refused start (409: already running, or the
 * dataset is busy being written) surfaces here as a destructive toast.
 *
 * Shared by DatasetInfoCard's HubSyncRow and the DatasetPicker rows so both
 * offer the identical dialog. The caller owns the upload hook and passes down
 * `start` (which returns an error message, or null on a clean start).
 */
const UploadDatasetDialog: React.FC<{
  repoId: string;
  /** Kicks off the background upload. Resolves to null on a clean start, or an
   * error message when the start was refused / unreachable. */
  start: (tags: string[], isPrivate: boolean) => Promise<string | null>;
  /** Popover trigger element (a button/icon). */
  children: React.ReactNode;
  align?: "start" | "center" | "end";
}> = ({ repoId, start, children, align = "end" }) => {
  const { t } = useTranslation();
  const { toast } = useToast();
  const [popoverOpen, setPopoverOpen] = useState(false);
  const [isPrivate, setIsPrivate] = useState(false);
  const [tagsInput, setTagsInput] = useState("");
  const [starting, setStarting] = useState(false);

  const handleUpload = async () => {
    setStarting(true);
    try {
      const tags = tagsInput
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);
      const error = await start(tags, isPrivate);
      if (error) {
        toast({
          // `error` is the start refusal — backend prose, kept verbatim.
          title: t("landing.uploadDataset.failedTitle"),
          description: error,
          variant: "destructive",
        });
        return;
      }
      // Clean start: close the popover. The caller's row shows the live
      // "Uploading…" state and toasts on completion.
      setPopoverOpen(false);
      toast({
        title: t("landing.uploadDataset.startedTitle"),
        description: t("landing.uploadDataset.startedBody", { repoId }),
      });
    } finally {
      setStarting(false);
    }
  };

  return (
    <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent
        align={align}
        className="w-72 text-xs"
        // This popover is portaled to the body, but React synthetic events
        // still bubble through the React tree to the cmdk CommandItem that
        // renders our trigger (see DatasetPicker). cmdk's CommandItem selects
        // the row on click, so a click on any control in here (the
        // Public/Private toggle buttons, the Tags input, the Upload button)
        // would otherwise select the dataset and close the picker. Stop the
        // click (and pointer-down) from reaching cmdk. We only stopPropagation
        // — never preventDefault — so focusing the input and the buttons' own
        // handlers keep working. Radix's own outside-click / Escape close is
        // driven by events *outside* this content, so it's unaffected.
        onClick={(e) => e.stopPropagation()}
        onPointerDown={(e) => e.stopPropagation()}
      >
        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label
              id={`hub-upload-visibility-${repoId}`}
              className="font-normal text-muted-foreground"
            >
              {t("landing.uploadDataset.visibility")}
            </Label>
            <VisibilityToggle
              value={isPrivate}
              onChange={setIsPrivate}
              idBase={`hub-upload-visibility-${repoId}`}
            />
            <p className="leading-snug text-muted-foreground">
              {isPrivate
                ? t("landing.uploadDataset.privateNote")
                : t("landing.uploadDataset.publicNote")}
            </p>
          </div>
          <div className="space-y-1">
            <Label
              htmlFor={`hub-upload-tags-${repoId}`}
              className="font-normal text-muted-foreground"
            >
              {t("landing.uploadDataset.tagsLabel")}
            </Label>
            <Input
              id={`hub-upload-tags-${repoId}`}
              value={tagsInput}
              onChange={(e) => setTagsInput(e.target.value)}
              placeholder="robotics, manipulation"
              className="h-7 text-xs"
            />
          </div>
          <Button
            size="sm"
            onClick={handleUpload}
            disabled={starting}
            className="h-7 w-full gap-1 text-xs"
          >
            {starting ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                {t("landing.uploadDataset.starting")}
              </>
            ) : (
              <>
                <UploadIcon className="h-3 w-3" />
                {t("landing.uploadDataset.submit")}
              </>
            )}
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default UploadDatasetDialog;
