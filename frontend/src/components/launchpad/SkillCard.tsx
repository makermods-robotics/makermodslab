import React from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { ModelItem } from "@/lib/modelsApi";
import { policyTypeDisplayName } from "@/components/training/types";
import { useLanguage } from "@/contexts/LanguageContext";
import { isCaselessScript } from "@/i18n/config";
import { cn } from "@/lib/utils";
import sockThumb from "@/assets/skill-sock-orange.jpg";
import bottleThumb from "@/assets/skill-bottle.jpg";
import towelThumb from "@/assets/skill-towel.jpg";
import stackCubesThumb from "@/assets/skill-stack-cubes.jpg";

/** Synthetic ids for skill previews that haven't been trained yet — no Hub
 * repo or job backs any of these, so none ever comes from `/models`. See WIP
 * handling in SkillSlider/SkillDetailDialog. */
export const WIP_SKILL_IDS = {
  bottleCap: "wip/bottle-cap-removal",
  towelFold: "wip/towel-fold",
  stackCubes: "wip/stack-cubes",
} as const;

const WIP_SKILL_ID_SET: ReadonlySet<string> = new Set(
  Object.values(WIP_SKILL_IDS),
);

/** True when `id` is one of the not-yet-trained WIP preview cards. */
export function isWipSkillId(id: string): boolean {
  return WIP_SKILL_ID_SET.has(id);
}

/** Curated preview media for featured skills, keyed by Hub repo id. */
const SKILL_THUMBNAILS: Record<string, string> = {
  "makermods/act_makermods_sock_2_only_more_orange_2026-07-16_22-14-55":
    sockThumb,
  [WIP_SKILL_IDS.bottleCap]: bottleThumb,
  [WIP_SKILL_IDS.towelFold]: towelThumb,
  [WIP_SKILL_IDS.stackCubes]: stackCubesThumb,
};

/** The curated preview image for a skill, or undefined when it has none. */
export function skillThumbnail(m: ModelItem): string | undefined {
  return SKILL_THUMBNAILS[m.hf_repo_id ?? m.id];
}

/** Curated display names for featured skills, keyed by Hub repo id. The repo
 * ids are DATA — they address a real repo, so they stay verbatim; only the
 * slug they map to is a translation key. */
const SKILL_NAME_KEYS: Record<string, string> = {
  "makermods/act_makermods_sock_2_only_more_orange_2026-07-16_22-14-55":
    "sortingSocks",
  [WIP_SKILL_IDS.bottleCap]: "openingBottleCaps",
  [WIP_SKILL_IDS.towelFold]: "foldingTowels",
  [WIP_SKILL_IDS.stackCubes]: "stackingCubes",
};

/** The marketplace provenance of a skill card. "wip" marks a preview card for
 * a skill that hasn't been trained yet — no repo/job backs it. */
export type SkillBadge = "mine" | "makermods" | "community" | "wip";

/** 16000 -> "16k", 950 -> "950" — matches the models card's compact form. */
export const formatCount = (n: number): string => {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(n);
};

/** The Hub namespace a skill lives under, or null for a bare local run id (no
 * "/"), which belongs to the logged-in user by construction. */
export function skillNamespace(m: ModelItem): string | null {
  const src = m.hf_repo_id ?? m.id;
  return src.includes("/") ? src.split("/")[0] : null;
}

/** The byline shown under a skill's title. A WIP preview has no author — its id
 * is synthetic, so the "wip/" prefix is a keying device and not an org, and
 * `skillNamespace` would otherwise surface it as one. */
export function skillAuthorLabel(m: ModelItem): string {
  if (isWipSkillId(m.id)) return "Coming soon";
  return skillNamespace(m) ?? "local checkpoint";
}

/** Translated author byline. Mirrors skillTitle/skillDisplayTitle: the plain
 * function stays English for non-React callers, this one is what users read.
 * A namespace is a Hub org/user handle — data, never translated. */
export function skillDisplayAuthorLabel(t: TFunction, m: ModelItem): string {
  if (isWipSkillId(m.id)) return t("launchpad.skills.comingSoon");
  return skillNamespace(m) ?? t("launchpad.skills.localCheckpoint");
}

/** MINE when the skill is in the user's namespace (or a bare local run they own);
 * MAKERMODS for the makermods org; COMMUNITY otherwise. Author == the namespace,
 * matched case-insensitively (mirrors DatasetInfoCard's useCanEditHub). */
export function classifySkill(
  m: ModelItem,
  username?: string | null,
): SkillBadge {
  const ns = skillNamespace(m);
  if (ns === null) return "mine";
  if (username && ns.toLowerCase() === username.toLowerCase()) return "mine";
  if (ns.toLowerCase() === "makermods") return "makermods";
  return "community";
}

/** True when the skill belongs in the user's library: any local checkpoint
 * (local/both) or a Hub repo in their own namespace. */
export function isMineSkill(m: ModelItem, username?: string | null): boolean {
  if (m.source === "local" || m.source === "both") return true;
  return classifySkill(m, username) === "mine";
}

/** The UNTRANSLATED title: a curated English display name when the skill has
 * one, otherwise the name segment only (Hub rows carry the full
 * "namespace/name" in `name`), never an empty string.
 *
 * Search matches against this so that typing "sock" keeps working no matter
 * which language the UI is in — see SkillSlider's filter. Use
 * `skillDisplayTitle` for anything the user actually reads.
 */
export function skillTitle(m: ModelItem): string {
  const slug = SKILL_NAME_KEYS[m.hf_repo_id ?? m.id];
  if (slug) return SKILL_NAME_EN[slug];
  const raw = m.name || m.id;
  return raw.includes("/") ? (raw.split("/").pop() ?? raw) : raw;
}

/** English copies of the curated names, so `skillTitle` stays a pure data
 * function usable outside React (search, sorting) without an i18n instance. */
const SKILL_NAME_EN: Record<string, string> = {
  sortingSocks: "Sorting socks",
  openingBottleCaps: "Opening bottle caps",
  foldingTowels: "Folding towels",
  stackingCubes: "Stacking cubes",
};

/** The title the user sees. Falls back to the raw name segment for any skill
 * without a curated entry — those are repo-derived data and never translated. */
export function skillDisplayTitle(t: TFunction, m: ModelItem): string {
  const slug = SKILL_NAME_KEYS[m.hf_repo_id ?? m.id];
  if (slug) return t(`launchpad.skillNames.${slug}` as never);
  const raw = m.name || m.id;
  return raw.includes("/") ? (raw.split("/").pop() ?? raw) : raw;
}

const BADGE_LABEL_KEY: Record<SkillBadge, string> = {
  mine: "launchpad.badge.mine",
  makermods: "launchpad.badge.makermods",
  community: "launchpad.badge.community",
  wip: "launchpad.badge.wip",
};

const BADGE_CLASS: Record<SkillBadge, string> = {
  mine: "border-transparent bg-primary text-primary-foreground",
  makermods: "border-ring bg-transparent text-foreground",
  community: "border-border bg-transparent text-muted-foreground",
  wip: "border-warn/40 bg-transparent text-warn",
};

/** Provenance pill (MINE / MAKERMODS / COMMUNITY) — token-styled, works in both
 * themes. */
export const SkillBadgePill: React.FC<{ badge: SkillBadge }> = ({ badge }) => {
  const { t } = useTranslation();
  const { language } = useLanguage();
  return (
    <span
      className={cn(
        "shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold",
        // `uppercase` does nothing to Chinese, but the letter-spacing that
        // pairs with it does — both come off together.
        isCaselessScript(language) ? "" : "uppercase tracking-[0.06em]",
        BADGE_CLASS[badge],
      )}
    >
      {t(BADGE_LABEL_KEY[badge] as never)}
    </span>
  );
};

/** A small mono stat chip (policy type · steps · private). */
const Stat: React.FC<{ children: React.ReactNode; tone?: "amber" }> = ({
  children,
  tone,
}) => (
  <span
    className={`rounded border border-border px-1.5 py-0.5 font-mono text-[10.5px] ${
      tone === "amber" ? "text-warn" : "text-muted-foreground"
    }`}
  >
    {children}
  </span>
);

export interface SkillCardProps {
  model: ModelItem;
  badge: SkillBadge;
  onOpen: (model: ModelItem) => void;
}

/**
 * One skill in the launchpad slider. Preview media slot + title + author +
 * whatever real stats the /models payload carries (policy type, step count,
 * private flag) — no fabricated likes/downloads (the API exposes neither).
 * Click opens the skill detail dialog.
 */
const SkillCard: React.FC<SkillCardProps> = ({ model, badge, onOpen }) => {
  const { t } = useTranslation();
  const author = skillDisplayAuthorLabel(t, model);
  const title = skillDisplayTitle(t, model);
  const policy = model.policy_type
    ? policyTypeDisplayName(model.policy_type)
    : null;
  const thumbnail = skillThumbnail(model);

  return (
    <button
      type="button"
      onClick={() => onOpen(model)}
      className="group flex w-64 shrink-0 snap-start flex-col overflow-hidden rounded-lg border border-border bg-card text-left shadow-1 transition-colors hover:border-ring focus-visible:border-ring focus-visible:outline-none"
      aria-label={t("launchpad.skills.open", { title })}
    >
      {thumbnail ? (
        <img
          src={thumbnail}
          alt={t("launchpad.skills.previewAlt", { title })}
          className="aspect-[4/3] w-full object-cover"
        />
      ) : (
        <div
          className="media-slot aspect-[4/3] w-full"
          data-label={t("launchpad.skills.previewPlaceholder")}
        />
      )}
      <div className="flex flex-1 flex-col gap-2 p-3">
        <div className="flex flex-col items-start gap-1.5">
          <span className="w-full min-w-0 truncate font-display font-semibold tracking-tight">
            {title}
          </span>
          <SkillBadgePill badge={badge} />
        </div>
        <span className="truncate font-mono text-[11px] text-muted-foreground">
          {author}
        </span>
        <div className="mt-auto flex flex-wrap gap-1.5 pt-1">
          {policy && <Stat>{policy}</Stat>}
          {model.steps != null && (
            <Stat>
              {t("launchpad.skills.steps", {
                steps: formatCount(model.steps),
              })}
            </Stat>
          )}
          {model.private && (
            <Stat tone="amber">{t("launchpad.skills.private")}</Stat>
          )}
        </div>
      </div>
    </button>
  );
};

export default SkillCard;
