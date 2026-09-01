import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronUp, Library } from "lucide-react";
import { Button } from "@/components/ui/button";
import BrandMark from "@/components/BrandMark";
import LanguageSwitcher from "@/components/LanguageSwitcher";
import Footer from "@/components/Footer";
import HfAuthChip from "@/components/landing/HfAuthChip";
import UsageInstructionsModal from "@/components/landing/UsageInstructionsModal";
import Hero from "@/components/launchpad/Hero";
import SkillSlider from "@/components/launchpad/SkillSlider";
import NewSkillBanner from "@/components/launchpad/NewSkillBanner";
import ActivityStrip from "@/components/launchpad/ActivityStrip";
import LibrarySheet from "@/components/launchpad/LibrarySheet";
import RobotCorner from "@/components/launchpad/RobotCorner";
import CollectHandoff from "@/components/studio/CollectHandoff";
import StudioOverlay from "@/components/studio/StudioOverlay";
import { useStudio } from "@/contexts/StudioContext";
import { isHostedSpace } from "@/lib/isHostedSpace";
import { useOnboarding } from "@/contexts/OnboardingContext";
import { useOnceFlag } from "@/lib/onboarding/storage";
import { launchpadTour } from "@/lib/onboarding/tours";

const ON_SPACE = isHostedSpace();
const ONBOARDING_KEY = "makerlab:onboarding-completed";

/**
 * Layout D "Launchpad" — the single dashboard route. Marketplace-first hero
 * with the skill slider, the "+ New Skill" banner that slides the studio up,
 * and the always-visible robot corner. Config happens in dialogs; live
 * hardware sessions live on their own immersive routes.
 */
const Launchpad = () => {
  const [showUsageModal, setShowUsageModal] = useState(ON_SPACE);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [search, setSearch] = useState("");
  const { openStudio } = useStudio();
  const { t } = useTranslation();
  const { start } = useOnboarding();
  const { seen, markSeen } = useOnceFlag(ONBOARDING_KEY);

  useEffect(() => {
    if (!seen) start(launchpadTour, markSeen);
    // Start exactly once on first mount — re-running on every `seen`/`start`
    // identity change would restart the tour mid-flow.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="flex items-center justify-between gap-3 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <BrandMark />
          <HfAuthChip />
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            data-tour="launchpad-library"
            className="h-8 gap-1.5 rounded-full px-3"
            onClick={() => setLibraryOpen(true)}
          >
            <Library className="h-3.5 w-3.5" />
            {t("launchpad.header.myLibrary")}
          </Button>
          <LanguageSwitcher />
          {/* Wrapped (rather than tagging RobotCorner.tsx itself) since the
              same component also renders inside StudioOverlay's header —
              tagging it directly would give the tour two matching elements. */}
          <div data-tour="launchpad-robot-corner">
            <RobotCorner />
          </div>
        </div>
      </header>

      {/* justify-center holds the whole stack (hero → banner) in the middle
          of the viewport rather than hugging the header. */}
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col items-center justify-center gap-10 px-4 py-8 sm:px-6">
        <CollectHandoff />
        <Hero search={search} onSearchChange={setSearch} />
        <SkillSlider search={search} />
        <ActivityStrip />
        <div className="w-full">
          <NewSkillBanner />
        </div>
      </main>

      <Footer />

      {/* Pull up to open the skill studio — the studio's own header carries
          the mirrored "pull down" arrow back to here. Pinned to the
          viewport (not just after Footer in normal flow) so it's reachable
          without scrolling even when the hero/slider/banner stack above is
          taller than the screen; centered on the viewport width the same
          way, independent of anything else on the page. */}
      <button
        type="button"
        onClick={() => openStudio()}
        aria-label={t("launchpad.header.openStudio")}
        title={t("launchpad.header.openStudio")}
        className="fixed bottom-3 left-1/2 z-30 flex h-9 w-9 -translate-x-1/2 items-center justify-center rounded-full border border-border bg-background/90 text-muted-foreground shadow-1 backdrop-blur-sm transition-colors hover:text-foreground"
      >
        <ChevronUp className="h-5 w-5 animate-bounce" />
      </button>

      <UsageInstructionsModal
        open={showUsageModal}
        onOpenChange={setShowUsageModal}
        dismissible={!ON_SPACE}
      />
      <LibrarySheet open={libraryOpen} onOpenChange={setLibraryOpen} />
      <StudioOverlay />
    </div>
  );
};

export default Launchpad;
