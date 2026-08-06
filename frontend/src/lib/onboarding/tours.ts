import type { Tour } from "@/lib/onboarding/types";

/**
 * First-run tour shown once on the Launchpad — the only screen a brand-new
 * user sees before doing anything else. Covers finding an existing skill,
 * building a new one, setting up a robot, and where saved items live.
 */
export const launchpadTour: Tour = {
  id: "launchpad",
  steps: [
    {
      target: "[data-tour=launchpad-search]",
      title: "Find a skill",
      description:
        "Search for a skill by name, or browse the ones below.",
    },
    {
      target: "[data-tour=launchpad-skills]",
      title: "Browse skills",
      description:
        "Skills others have trained and shared — run one, or use it as a starting point for your own.",
    },
    {
      target: "[data-tour=launchpad-new-skill]",
      title: "Build your own",
      description:
        "Collect a dataset, train a policy, and deploy it to your robot — all without leaving this page.",
    },
    {
      target: "[data-tour=launchpad-robot-corner]",
      title: "Set up your robot",
      description:
        "Add and configure the arm you'll record and run skills on.",
    },
    {
      target: "[data-tour=launchpad-library]",
      title: "Your library",
      description: "Datasets, models, and training jobs you've saved live here.",
    },
  ],
};
