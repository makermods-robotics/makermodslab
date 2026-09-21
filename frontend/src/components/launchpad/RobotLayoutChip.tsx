import React from "react";
import { useTranslation } from "react-i18next";
import type { RobotArms } from "@/lib/robotSetupGap";
import { cn } from "@/lib/utils";

/**
 * The small layout tag beside a record's name wherever one is listed or
 * named: "Robot (follower only)" for a station, "Controller (leader only)"
 * for a leader that drives a remote robot. A leader+follower pair — the
 * default, and every record from before remote teleoperation — renders
 * nothing, so the common case stays exactly as it was.
 *
 * The `arms` value is data (the record's field on disk); only the label
 * localizes.
 */
const RobotLayoutChip: React.FC<{
  arms: RobotArms | undefined;
  className?: string;
}> = ({ arms, className }) => {
  const { t } = useTranslation();
  if (arms !== "follower" && arms !== "leader") return null;
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center whitespace-nowrap rounded border border-border px-1.5 py-px text-[10px] font-medium leading-4 text-muted-foreground",
        className,
      )}
    >
      {arms === "follower"
        ? t("robot.layout.followerOnly")
        : t("robot.layout.leaderOnly")}
    </span>
  );
};

export default RobotLayoutChip;
