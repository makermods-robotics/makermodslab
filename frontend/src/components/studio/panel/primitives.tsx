import React from "react";
import { AlertTriangle, ChevronDown } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

/** Slide animation shared by every studio collapsible — the panels' entry
 * forms and all three libraries. The single home for this animation string;
 * don't copy it into components. */
export const SLIDE =
  "overflow-hidden data-[state=open]:animate-collapsible-down data-[state=closed]:animate-collapsible-up";

/** Numbered studio panel header: mono step digit + title, identical across
 * Collect, Train, and Deploy. */
export const PanelHeader: React.FC<{
  step: string;
  title: string;
  /** Optional trailing element (e.g. a resolving spinner). */
  children?: React.ReactNode;
}> = ({ step, title, children }) => (
  <div className="flex items-baseline gap-2">
    <span className="font-mono text-xs text-muted-foreground">{step}</span>
    <h2 className="text-base">{title}</h2>
    {children}
  </div>
);

/**
 * The skin every panel's entry control wears. Exported separately because
 * Deploy's entry is a real <Select> (picking a skill IS the value, it doesn't
 * just open a form), so its SelectTrigger has to wear this rather than render
 * a PanelEntryControl.
 */
export const PANEL_ENTRY_CLASS =
  "flex h-10 w-full items-center gap-2 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background transition-colors hover:border-muted-foreground/40 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2";

/** The dot accent each panel's entry control carries, so the three openers are
 * distinguishable at a glance while sharing one shape. */
export const PanelEntryDot: React.FC<{ className: string }> = ({
  className,
}) => (
  <span
    aria-hidden
    className={cn("h-2 w-2 shrink-0 rounded-full", className)}
  />
);

/**
 * The panels' shared entry control — a quiet select-style row (same height,
 * border, and radius as a shadcn SelectTrigger) so Collect's "Record new
 * dataset", Train's "Start a new training", and Deploy's real skill <Select>
 * read as the same control. Used as a CollapsibleTrigger (asChild) that
 * slides the panel's form open in place.
 */
export const PanelEntryControl = React.forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    open: boolean;
    /** Tiny status-dot accent (e.g. bg-red-500 for record). */
    dotClassName?: string;
  }
>(({ open, dotClassName, children, className, ...props }, ref) => (
  <button
    ref={ref}
    type="button"
    className={cn(PANEL_ENTRY_CLASS, className)}
    {...props}
  >
    {dotClassName ? <PanelEntryDot className={dotClassName} /> : null}
    <span className="truncate">{children}</span>
    <ChevronDown
      className={cn(
        "ml-auto h-4 w-4 shrink-0 text-muted-foreground transition-transform",
        open && "rotate-180",
      )}
    />
  </button>
));
PanelEntryControl.displayName = "PanelEntryControl";

/**
 * An eyebrow-headed block inside a panel form. Per the studio's copy rule the
 * eyebrow level is reserved for blocks that are NOT a single labelled control
 * — in practice Cameras and Advanced. Everything else is just <Label> +
 * control, so a category heading never sits directly above one field and
 * restates it.
 */
export const FormSection: React.FC<{
  title: string;
  className?: string;
  children: React.ReactNode;
}> = ({ title, className, children }) => (
  <section className={cn("space-y-3", className)}>
    <h3 className="eyebrow">{title}</h3>
    {children}
  </section>
);

/**
 * The one "Advanced parameters" collapsible, shared by the Collect form and
 * the training configurator. Its trigger is an `.eyebrow` so it sits at the
 * same level as FormSection instead of outranking it — the previous
 * `text-sm font-semibold` triggers were visually heavier than the category
 * headings above them, inverting the hierarchy.
 *
 * Uncontrolled by default; pass open/onOpenChange to drive it.
 */
export const AdvancedSection: React.FC<{
  /** Defaults to "Advanced parameters" — the label used in every panel. */
  title?: string;
  /** One line naming what's inside, so the block is skippable unopened. */
  summary?: React.ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: React.ReactNode;
}> = ({ title = "Advanced parameters", summary, open, onOpenChange, children }) => (
  <Collapsible
    open={open}
    onOpenChange={onOpenChange}
    className="group space-y-3"
  >
    <CollapsibleTrigger className="flex w-full items-start justify-between border-b border-border pb-2 text-left">
      <span>
        <span className="eyebrow block">{title}</span>
        {summary ? (
          <span className="mt-1 block text-xs text-muted-foreground">
            {summary}
          </span>
        ) : null}
      </span>
      <ChevronDown className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" />
    </CollapsibleTrigger>
    <CollapsibleContent className={SLIDE}>
      <div className="space-y-4 pt-1">{children}</div>
    </CollapsibleContent>
  </Collapsible>
);

/**
 * The robot-readiness warning that opens the Collect and Deploy forms. One
 * implementation of a block the two panels used to render differently (Collect
 * hand-rolled amber divs, Deploy used <Alert> with warn tokens) — the caller
 * owns the copy, this owns the look. Warnings only: a ready robot renders
 * nothing, because the robot menu already names the selection and a green
 * "ready" line just restates it. Rendered without an eyebrow: it reports
 * state, it isn't a parameter.
 */
export const RobotStatus: React.FC<{
  ready: boolean;
  children: React.ReactNode;
}> = ({ ready, children }) =>
  ready ? null : (
    <Alert className="border-warn/40 text-warn [&>svg]:text-warn">
      <AlertTriangle className="h-4 w-4" />
      <AlertDescription>{children}</AlertDescription>
    </Alert>
  );

/** Pins a panel's library to the bottom of its column behind a hairline, so
 * the three libraries sit at the same level across the studio grid. */
export const LibrarySection: React.FC<{
  className?: string;
  children: React.ReactNode;
}> = ({ className, children }) => (
  <div className={cn("mt-auto border-t border-border pt-5", className)}>
    {children}
  </div>
);
