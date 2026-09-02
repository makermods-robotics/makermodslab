import * as React from "react";
import { useTranslation } from "react-i18next";
import { Loader2, PowerOff, Zap } from "lucide-react";

import { Button, type ButtonProps } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ROBOT_ACTIONS,
  type EnergizingActionKey,
  type ReleaseActionKey,
} from "@/lib/robotActions";

/**
 * The two shared affordances for hardware controls.
 *
 * `RobotActionButton` marks a button that ENERGIZES an arm; the sibling
 * `ReleaseActionButton` marks one that ends with the arm de-energized. Both
 * pin the icon and the tooltip to the action, so a call site cannot ship one
 * without the other and two screens cannot drift into two different amber
 * buttons. Treatment and copy come from lib/robotActions.ts.
 *
 * The trigger is the Button itself (`asChild`, no wrapper element) so
 * swapping a plain `<Button>` for one of these changes no layout: every
 * `flex-1` / `w-full` className the call site already had keeps applying to
 * the same DOM node. The trade-off is that a DISABLED button shows no
 * tooltip — `disabled:pointer-events-none` in the Button base swallows the
 * hover — which is fine here: these tooltips describe what pressing would do
 * to the hardware, and call sites that need to explain WHY a control is
 * unavailable already have their own tooltip for that.
 *
 * Tooltip rendering needs a TooltipProvider above it; App.tsx mounts one at
 * the root, so every app surface is covered.
 */

type SharedProps = Omit<ButtonProps, "variant" | "asChild"> & {
  /** Which side the tooltip opens on. Radix's default is "top". */
  tooltipSide?: React.ComponentProps<typeof TooltipContent>["side"];
  /**
   * Swap the fixed icon for a spinner while the request is in flight. The
   * icon SLOT stays mandatory — this only changes which glyph fills it — so
   * call sites that used to render their own `<Loader2 className="animate-spin" />`
   * keep their feedback without being able to supply an arbitrary icon.
   */
  busy?: boolean;
};

export interface RobotActionButtonProps extends SharedProps {
  /** Picks the tooltip from ROBOT_ACTIONS. Energizing actions only. */
  action: EnergizingActionKey;
}

const RobotActionButton = React.forwardRef<
  HTMLButtonElement,
  RobotActionButtonProps
>(({ action, tooltipSide, busy, children, ...props }, ref) => {
  const { t } = useTranslation();
  // Static per-action key from the map, resolved during render (never at
  // import time) so a language switch repaints it.
  const tooltip = t(ROBOT_ACTIONS[action].tooltipKey as never);
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button ref={ref} variant="robot" {...props}>
          {busy ? <Loader2 className="animate-spin" aria-hidden /> : <Zap aria-hidden />}
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent side={tooltipSide}>{tooltip}</TooltipContent>
    </Tooltip>
  );
});
RobotActionButton.displayName = "RobotActionButton";

export interface ReleaseActionButtonProps extends SharedProps {
  /** Picks the tooltip from ROBOT_ACTIONS. De-energizing actions only. */
  action: ReleaseActionKey;
}

const ReleaseActionButton = React.forwardRef<
  HTMLButtonElement,
  ReleaseActionButtonProps
>(({ action, tooltipSide, busy, children, ...props }, ref) => {
  const { t } = useTranslation();
  const tooltip = t(ROBOT_ACTIONS[action].tooltipKey as never);
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button ref={ref} variant="destructive" {...props}>
          {busy ? (
            <Loader2 className="animate-spin" aria-hidden />
          ) : (
            <PowerOff aria-hidden />
          )}
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent side={tooltipSide}>{tooltip}</TooltipContent>
    </Tooltip>
  );
});
ReleaseActionButton.displayName = "ReleaseActionButton";

export { RobotActionButton, ReleaseActionButton };
