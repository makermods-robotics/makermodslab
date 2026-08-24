import React from "react";
import { Check, Languages } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useLanguage } from "@/contexts/LanguageContext";
import { SUPPORTED_LANGUAGES } from "@/i18n/config";
import { cn } from "@/lib/utils";

/**
 * App-language picker. Mounted in BOTH the Launchpad header and the studio
 * overlay header — the studio is `fixed inset-0 z-40` and covers the viewport,
 * so a single Launchpad mount would vanish while the studio is open. Same
 * dual-mount pattern as HfAuthChip and RobotCorner.
 *
 * Language names are deliberately shown as endonyms ("简体中文", not
 * "Simplified Chinese") — a picker is the one place a language must not be
 * named in the language the user is trying to leave.
 */
const LanguageSwitcher: React.FC<{ className?: string }> = ({ className }) => {
  const { language, setLanguage } = useLanguage();
  const active = SUPPORTED_LANGUAGES.find((l) => l.code === language);

  return (
    <DropdownMenu>
      <Tooltip>
        <TooltipTrigger asChild>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              // Static English label: a user who has landed in a language they
              // can't read needs this control to still be identifiable, and
              // the icon plus the endonym in the menu carry the meaning.
              aria-label="Language"
              className={cn("h-8 gap-1.5 rounded-full px-2.5", className)}
            >
              <Languages className="h-3.5 w-3.5" />
              <span className="hidden text-xs sm:inline">{active?.label}</span>
            </Button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent side="bottom">Language</TooltipContent>
      </Tooltip>
      <DropdownMenuContent align="end" className="w-44">
        {SUPPORTED_LANGUAGES.map(({ code, label }) => (
          <DropdownMenuItem
            key={code}
            onSelect={() => setLanguage(code)}
            className="gap-2"
          >
            <Check
              className={cn(
                "h-4 w-4",
                code === language ? "opacity-100" : "opacity-0",
              )}
            />
            <span className="flex-1">{label}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default LanguageSwitcher;
