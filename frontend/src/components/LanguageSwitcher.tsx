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
 * App-language picker. Mounted ONCE, in the Launchpad footer beside the GitHub
 * / Documentation / Discord links. It used to sit in the Launchpad header and
 * again in the studio overlay header (the studio is `fixed inset-0 z-40` and
 * covers the viewport, so one mount would have vanished behind it); both are
 * gone. Language is a settle-in-once choice, not something to reach for while
 * a robot flow is running, and the footer is where the other
 * once-per-install links already live.
 *
 * Icon only. The active language is not printed on the trigger — the menu's
 * check mark carries it — but the `aria-label` and tooltip stay static English
 * so someone stranded in a script they cannot read can still find this.
 *
 * Language names are deliberately shown as endonyms ("简体中文", not
 * "Simplified Chinese") — a picker is the one place a language must not be
 * named in the language the user is trying to leave.
 */
const LanguageSwitcher: React.FC<{ className?: string }> = ({ className }) => {
  const { language, setLanguage } = useLanguage();

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
              className={cn("h-8 w-8 rounded-full p-0", className)}
            >
              <Languages className="h-3.5 w-3.5" />
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
