import React from "react";
import { useTranslation } from "react-i18next";
import { useStudio } from "@/contexts/StudioContext";

/** Keys, not copy: this array is built once at import time, so resolved
 * strings here would never follow a language change. */
const STEPS = [
  {
    labelKey: "launchpad.newSkill.steps.collect.label",
    subKey: "launchpad.newSkill.steps.collect.sub",
  },
  {
    labelKey: "launchpad.newSkill.steps.train.label",
    subKey: "launchpad.newSkill.steps.train.sub",
  },
  {
    labelKey: "launchpad.newSkill.steps.deploy.label",
    subKey: "launchpad.newSkill.steps.deploy.sub",
  },
] as const;

/**
 * The "＋ New Skill" workbench banner — Layout D's signature element. Resting, it
 * reads "Collect, train, deploy — without leaving this page."; on hover or
 * keyboard focus it reveals the 1·Collect → 2·Train → 3·Deploy pipeline with a
 * smooth height/opacity transition (grid-rows 0fr → 1fr). Click slides the
 * studio up on the Collect panel. Fully keyboard accessible (it's a button;
 * :focus-visible reveals the steps).
 */
const NewSkillBanner: React.FC = () => {
  const { openStudio } = useStudio();
  const { t } = useTranslation();

  return (
    <button
      type="button"
      data-tour="launchpad-new-skill"
      onClick={() => openStudio("collect")}
      className="group w-full rounded-lg border border-border bg-card px-6 py-7 text-left shadow-1 transition-colors hover:border-ring focus-visible:border-ring focus-visible:outline-none"
      aria-label={t("launchpad.newSkill.aria")}
    >
      <span className="font-display text-xl font-semibold tracking-tight">
        {t("launchpad.newSkill.title")}
      </span>

      {/* Resting subtitle: fades out as the steps expand in. */}
      <span className="mt-1 block text-sm text-muted-foreground transition-opacity duration-300 group-hover:opacity-0 group-focus-visible:opacity-0">
        {t("launchpad.newSkill.subtitle")}
      </span>

      {/* Steps: collapsed to zero height at rest, expanding on hover/focus. */}
      <div className="grid grid-rows-[0fr] transition-[grid-template-rows] duration-300 ease-out group-hover:grid-rows-[1fr] group-focus-visible:grid-rows-[1fr]">
        <div className="overflow-hidden">
          <div className="mt-3 flex flex-col gap-3 opacity-0 transition-opacity duration-300 group-hover:opacity-100 group-focus-visible:opacity-100 sm:flex-row sm:items-stretch">
            {STEPS.map((step, i) => (
              <React.Fragment key={step.labelKey}>
                <div className="flex flex-1 flex-col gap-0.5 rounded-md border border-border bg-background px-3 py-2">
                  <span className="font-mono text-xs font-semibold text-foreground">
                    {t(step.labelKey)}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {t(step.subKey)}
                  </span>
                </div>
                {i < STEPS.length - 1 && (
                  <span
                    aria-hidden
                    className="hidden items-center justify-center font-mono text-muted-foreground sm:flex"
                  >
                    →
                  </span>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
    </button>
  );
};

export default NewSkillBanner;
