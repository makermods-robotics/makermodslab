import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Loader2, Plus, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import RobotLayoutChip from "@/components/launchpad/RobotLayoutChip";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useStationStatus } from "@/hooks/useStationStatus";
import { useRobots } from "@/hooks/useRobots";
import { formatStationRefusal, setStationRobot } from "@/lib/remoteApi";
import { cn } from "@/lib/utils";

export interface StationRobotDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Opens the Robot settings window for `name` (the empty state's way out
   * when robots exist but none has its follower side set up). */
  onOpenSettings?: (name: string) => void;
  /** Opens the create-robot flow (the empty state when no robot exists). */
  onCreateRobot?: () => void;
}

/**
 * The station's hosted-robot picker: which of the saved robots this
 * `--host` station publishes for remote teleoperation. A station may boot
 * with nothing chosen (several hostable robots, or none yet) and the corner
 * chip says so; this dialog is where the choice is made and later changed.
 *
 * The list is the backend's `hostable` (saved robots whose follower side is
 * set up — SO-101 only in this release), rendered as radio rows with the
 * current choice pre-selected. "Host this robot" PUTs the choice — the
 * station remembers it and re-arms hosting within seconds, a parked,
 * unseated session of the previous robot yielding on its own; "Stop hosting"
 * PUTs null. An engaged/seated session is a held session and the server
 * refuses with session.held — rendered as "an operator is driving", never
 * by dropping them.
 *
 * The hosted robot is deliberately NOT the corner's selected robot: the
 * selection drives LOCAL flows on this machine, the hosted robot is what a
 * remote operator gets, and the two may differ.
 */
const StationRobotDialog: React.FC<StationRobotDialogProps> = ({
  open,
  onOpenChange,
  onOpenSettings,
  onCreateRobot,
}) => {
  const { t } = useTranslation();
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { records, selectedName } = useRobots();
  const { status, refresh } = useStationStatus({ enabled: open, intervalMs: 3000 });

  const [picked, setPicked] = useState<string | null>(null);
  // Once the user clicks a row, a later status poll must not overwrite it.
  const touchedRef = useRef(false);
  const [submitting, setSubmitting] = useState<"host" | "stop" | null>(null);

  // Fresh per open: pre-select the current choice once status lands.
  useEffect(() => {
    if (!open) return;
    touchedRef.current = false;
    setPicked(null);
    setSubmitting(null);
  }, [open]);
  useEffect(() => {
    if (!open || touchedRef.current || !status) return;
    setPicked(status.robot);
  }, [open, status]);

  const current = status?.robot ?? null;
  const hostable = status?.hostable ?? [];
  const hostedNow = status?.hosting_active ? current : null;
  const robotNames = Object.keys(records);

  const apply = useCallback(
    async (robot: string | null) => {
      if (submitting) return;
      setSubmitting(robot === null ? "stop" : "host");
      try {
        await setStationRobot(baseUrl, fetchWithHeaders, robot);
        await refresh();
        if (robot === null) {
          toast({
            title: t("dialogs.stationRobot.toast.stoppedTitle"),
            description: t("dialogs.stationRobot.toast.stoppedDescription"),
          });
        } else {
          toast({
            title: t("dialogs.stationRobot.toast.changedTitle"),
            description: t("dialogs.stationRobot.toast.changedDescription", {
              robot,
            }),
          });
        }
        onOpenChange(false);
      } catch (e) {
        const line = formatStationRefusal(t, e, t("robot.station.failedFallback"));
        if (line !== null) {
          toast({
            title: t("robot.station.failedTitle"),
            description: line,
            variant: "destructive",
          });
        } else {
          toast({
            title: t("common.connectionError.title"),
            description: t("common.connectionError.description"),
            variant: "destructive",
          });
        }
      } finally {
        setSubmitting(null);
      }
    },
    [submitting, baseUrl, fetchWithHeaders, refresh, toast, t, onOpenChange],
  );

  const loading = status === null;
  const canHost =
    picked !== null && picked !== current && hostable.includes(picked);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t("dialogs.stationRobot.title")}</DialogTitle>
          <DialogDescription>
            {t("dialogs.stationRobot.description")}
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-6 text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
          </div>
        ) : hostable.length === 0 ? (
          <div className="space-y-3 rounded-md border border-border bg-muted/40 p-3 text-sm text-muted-foreground">
            <p>{t("dialogs.stationRobot.empty")}</p>
            {robotNames.length > 0 && onOpenSettings ? (
              <Button
                type="button"
                size="sm"
                variant="secondary"
                className="gap-1.5"
                onClick={() => {
                  onOpenChange(false);
                  onOpenSettings(selectedName ?? robotNames[0]);
                }}
              >
                <Settings className="h-3.5 w-3.5" />
                {t("dialogs.stationRobot.openSettings")}
              </Button>
            ) : onCreateRobot ? (
              <Button
                type="button"
                size="sm"
                variant="secondary"
                className="gap-1.5"
                onClick={() => {
                  onOpenChange(false);
                  onCreateRobot();
                }}
              >
                <Plus className="h-3.5 w-3.5" />
                {t("dialogs.stationRobot.createRobot")}
              </Button>
            ) : null}
          </div>
        ) : (
          <div
            role="radiogroup"
            aria-label={t("dialogs.stationRobot.listLabel")}
            className="divide-y divide-border overflow-hidden rounded-md border border-border"
          >
            {hostable.map((name) => {
              const checked = picked === name;
              const rec = records[name];
              // Robot names are data — verbatim.
              return (
                <button
                  key={name}
                  type="button"
                  role="radio"
                  aria-checked={checked}
                  disabled={submitting !== null}
                  onClick={() => {
                    touchedRef.current = true;
                    setPicked(name);
                  }}
                  className={cn(
                    "flex w-full items-center gap-2.5 bg-background px-3 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50",
                    checked
                      ? "bg-accent/60"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  <span
                    aria-hidden
                    className={cn(
                      "relative h-3.5 w-3.5 shrink-0 rounded-full border",
                      checked ? "border-primary" : "border-muted-foreground/60",
                    )}
                  >
                    {checked ? (
                      <span className="absolute inset-[2.5px] rounded-full bg-primary" />
                    ) : null}
                  </span>
                  <span
                    className={cn(
                      "min-w-0 flex-1 truncate text-sm font-medium",
                      checked && "text-foreground",
                    )}
                  >
                    {name}
                  </span>
                  <RobotLayoutChip arms={rec?.arms} />
                  {name === hostedNow ? (
                    <span className="flex shrink-0 items-center gap-1.5 text-[11px] font-semibold text-ok">
                      <span aria-hidden className="h-2 w-2 rounded-full bg-ok" />
                      {t("dialogs.stationRobot.hostedNow")}
                    </span>
                  ) : name === current ? (
                    <span className="shrink-0 text-[11px] font-semibold text-muted-foreground">
                      {t("dialogs.stationRobot.chosen")}
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
        )}

        <DialogFooter className="gap-2 sm:justify-between">
          <Button
            type="button"
            variant="outline"
            disabled={current === null || submitting !== null}
            onClick={() => apply(null)}
          >
            {submitting === "stop" ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {t("dialogs.stationRobot.applying")}
              </>
            ) : (
              t("dialogs.stationRobot.stopHosting")
            )}
          </Button>
          <div className="flex gap-2">
            <Button
              type="button"
              variant="ghost"
              onClick={() => onOpenChange(false)}
            >
              {t("common.cancel")}
            </Button>
            <Button
              type="button"
              disabled={!canHost || submitting !== null}
              onClick={() => picked !== null && apply(picked)}
            >
              {submitting === "host" ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  {t("dialogs.stationRobot.applying")}
                </>
              ) : (
                t("dialogs.stationRobot.host")
              )}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default StationRobotDialog;
