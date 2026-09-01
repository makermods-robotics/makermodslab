import React, { useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useModels } from "@/hooks/useModels";
import { ModelItem } from "@/lib/modelsApi";
import PolicyCard, {
  PolicyBadge,
  WIP_POLICY_IDS,
  isWipPolicyId,
  policyDisplayTitle,
  policyNamespace,
  policyTitle,
} from "@/components/launchpad/PolicyCard";
import PolicyDetailDialog from "@/components/dialogs/PolicyDetailDialog";

export interface PolicySliderProps {
  /** Live search filter from the hero search box. */
  search: string;
}

/** The one curated, fully-trained policy shown on the launchpad. */
const FEATURED_POLICY_ID =
  "makermods/act_makermods_sock_2_only_more_orange_2026-07-16_22-14-55";

/** Rendered when /models doesn't carry the featured repo (e.g. logged-out or
 * offline) so the launchpad always shows the curated card. */
const FEATURED_FALLBACK: ModelItem = {
  id: FEATURED_POLICY_ID,
  name: FEATURED_POLICY_ID,
  policy_type: "act",
  dataset: null,
  steps: null,
  path: null,
  last_modified: null,
  hf_repo_id: FEATURED_POLICY_ID,
  source: "hub",
};

/** Static preview card for a policy that hasn't been trained yet — never comes
 * from /models, so it's a plain client-side entry (never enriched, never
 * runnable). */
const wipPolicy = (id: string): ModelItem => ({
  id,
  name: id,
  policy_type: null,
  dataset: null,
  steps: null,
  path: null,
  last_modified: null,
  hf_repo_id: null,
  source: "hub",
});

/** Static WIP preview cards shown alongside the real featured policy. */
const WIP_POLICIES: ModelItem[] = Object.values(WIP_POLICY_IDS).map(wipPolicy);

/** A single loading skeleton shaped like a PolicyCard. */
const CardSkeleton: React.FC = () => (
  <div className="flex w-64 shrink-0 flex-col overflow-hidden rounded-lg border border-border bg-card shadow-1">
    <div className="aspect-[4/3] w-full animate-pulse bg-muted" />
    <div className="flex flex-col gap-2 p-3">
      <div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
      <div className="h-3 w-1/2 animate-pulse rounded bg-muted" />
      <div className="h-3 w-2/3 animate-pulse rounded bg-muted" />
    </div>
  </div>
);

/** "wip" for a not-yet-trained preview card, "makermods" for every other
 * curated card on the launchpad. */
const badgeFor = (m: ModelItem): PolicyBadge =>
  isWipPolicyId(m.id) ? "wip" : "makermods";

/**
 * Horizontal policy slider — the launchpad shows the curated MakerMods-
 * supported policy (real /models row when available, static fallback
 * otherwise) plus static WIP preview cards for policies still in training.
 * Scroll-snap track with ‹ › arrow buttons; the hero search box filters live
 * by name/author. Card click opens the policy detail dialog.
 */
const PolicySlider: React.FC<PolicySliderProps> = ({ search }) => {
  const { models, loading } = useModels();
  const { t } = useTranslation();
  const trackRef = useRef<HTMLDivElement>(null);
  const [detail, setDetail] = useState<ModelItem | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);

  const filtered = useMemo(() => {
    const featured =
      models.find((m) => (m.hf_repo_id ?? m.id) === FEATURED_POLICY_ID) ??
      FEATURED_FALLBACK;
    const curated = [featured, ...WIP_POLICIES];
    const q = search.trim().toLowerCase();
    if (!q) return curated;
    return curated.filter((m) => {
      const ns = policyNamespace(m) ?? "";
      return (
        // The English title stays matchable in every language, so searching
        // "sock" keeps finding the sorting-socks card while the UI is in
        // Chinese. The translated title is an ADDITIONAL match, never a
        // replacement — this filter is a strict superset of the old one.
        policyTitle(m).toLowerCase().includes(q) ||
        policyDisplayTitle(t, m).toLowerCase().includes(q) ||
        m.id.toLowerCase().includes(q) ||
        ns.toLowerCase().includes(q)
      );
    });
  }, [models, search, t]);

  const scrollBy = (dir: 1 | -1) => {
    const el = trackRef.current;
    if (!el) return;
    el.scrollBy({ left: dir * Math.round(el.clientWidth * 0.8), behavior: "smooth" });
  };

  const openDetail = (model: ModelItem) => {
    setDetail(model);
    setDetailOpen(true);
  };

  return (
    <section
      data-tour="launchpad-policies"
      className="w-full"
      aria-label={t("launchpad.policies.sectionLabel")}
    >
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => scrollBy(-1)}
          aria-label={t("launchpad.policies.previous")}
          className="hidden h-9 w-9 shrink-0 items-center justify-center rounded-full border border-border bg-card text-muted-foreground shadow-1 transition-colors hover:border-ring hover:text-foreground sm:flex"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>

        <div
          ref={trackRef}
          className="no-scrollbar flex flex-1 snap-x snap-mandatory gap-4 overflow-x-auto scroll-smooth pb-2"
        >
          {loading ? (
            <CardSkeleton />
          ) : filtered.length === 0 ? (
            <div className="flex min-h-[13rem] w-full items-center justify-center rounded-lg border border-dashed border-border bg-card/50 px-6 py-10 text-center text-sm text-muted-foreground">
              {t("launchpad.policies.empty")}
            </div>
          ) : (
            filtered.map((model) => (
              <PolicyCard
                key={model.id}
                model={model}
                badge={badgeFor(model)}
                onOpen={openDetail}
              />
            ))
          )}
        </div>

        <button
          type="button"
          onClick={() => scrollBy(1)}
          aria-label={t("launchpad.policies.next")}
          className="hidden h-9 w-9 shrink-0 items-center justify-center rounded-full border border-border bg-card text-muted-foreground shadow-1 transition-colors hover:border-ring hover:text-foreground sm:flex"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>

      <PolicyDetailDialog
        model={detail}
        open={detailOpen}
        onOpenChange={setDetailOpen}
      />
    </section>
  );
};

export default PolicySlider;
