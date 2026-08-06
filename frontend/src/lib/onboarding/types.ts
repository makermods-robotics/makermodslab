export type Placement = "top" | "bottom" | "left" | "right";

export interface TourStep {
  /** CSS selector for the target element, e.g. '[data-tour="launchpad-search"]'. */
  target: string;
  title: string;
  description: string;
  placement?: Placement;
}

export interface Tour {
  id: string;
  steps: TourStep[];
}