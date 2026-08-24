import React from "react";
import { useTranslation } from "react-i18next";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { useEyebrowClass } from "@/components/studio/panel/primitives";
import { cn } from "@/lib/utils";
import { FileText } from "lucide-react";
import { LogEntry } from "../types";

interface TrainingLogsProps {
  logs: LogEntry[];
  logContainerRef: React.RefObject<HTMLDivElement>;
}

const TrainingLogs: React.FC<TrainingLogsProps> = ({
  logs,
  logContainerRef,
}) => {
  const { t } = useTranslation();
  // `.eyebrow`'s tracking over-spaces CJK; useEyebrowClass drops it there.
  const eyebrow = useEyebrowClass();
  return (
    <Card className="bg-card border-border rounded-md">
      <CardHeader className="pb-2">
        <h3 className={cn(eyebrow, "flex items-center gap-1.5")}>
          <FileText className="h-3.5 w-3.5" /> {t("training.monitoring.logsTitle")}
        </h3>
      </CardHeader>
      <CardContent>
        <div
          ref={logContainerRef}
          className="h-96 overflow-y-auto rounded-md border border-border bg-muted p-4 font-mono text-xs"
        >
          {logs.length === 0 ? (
            <div className="py-8 text-muted-foreground">
              {t("training.monitoring.logsEmpty")}
            </div>
          ) : (
            logs.map((log, index) => (
              <div
                key={index}
                className="text-foreground break-words whitespace-pre-wrap"
              >
                <span className="text-muted-foreground mr-2 select-none">
                  {new Date(log.timestamp * 1000).toLocaleTimeString()}
                </span>
                {log.message}
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default TrainingLogs;
