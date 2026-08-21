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
      titleKey: "onboarding.launchpad.search.title",
      descriptionKey: "onboarding.launchpad.search.description",
    },
    {
      target: "[data-tour=launchpad-skills]",
      titleKey: "onboarding.launchpad.skills.title",
      descriptionKey: "onboarding.launchpad.skills.description",
    },
    {
      target: "[data-tour=launchpad-new-skill]",
      titleKey: "onboarding.launchpad.newSkill.title",
      descriptionKey: "onboarding.launchpad.newSkill.description",
    },
    {
      target: "[data-tour=launchpad-robot-corner]",
      titleKey: "onboarding.launchpad.robot.title",
      descriptionKey: "onboarding.launchpad.robot.description",
    },
    {
      target: "[data-tour=launchpad-library]",
      titleKey: "onboarding.launchpad.library.title",
      descriptionKey: "onboarding.launchpad.library.description",
    },
  ],
};

/**
 * First-run tour shown once the first time a user opens the Skill studio —
 * walks through its three panels (Collect, Train, Deploy).
 */
export const studioTour: Tour = {
  id: "studio",
  steps: [
    {
      target: "[data-tour=studio-collect]",
      titleKey: "onboarding.studio.collect.title",
      descriptionKey: "onboarding.studio.collect.description",
      placement: "right",
    },
    {
      target: "[data-tour=studio-train]",
      titleKey: "onboarding.studio.train.title",
      descriptionKey: "onboarding.studio.train.description",
      placement: "bottom",
    },
    {
      target: "[data-tour=studio-deploy]",
      titleKey: "onboarding.studio.deploy.title",
      descriptionKey: "onboarding.studio.deploy.description",
      placement: "left",
    },
  ],
};
