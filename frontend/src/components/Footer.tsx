import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { Github, BookOpen } from "lucide-react";
import hfLogo from "@/assets/hf-logo.svg";
import lerobotLogo from "@/assets/lerobot-logo.png";

const DiscordIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    role="img"
    viewBox="0 0 24 24"
    xmlns="http://www.w3.org/2000/svg"
    fill="currentColor"
    className={className}
  >
    <path d="M20.317 4.369A19.79 19.79 0 0 0 16.558 3.2a.07.07 0 0 0-.074.035c-.211.375-.444.864-.608 1.249a18.27 18.27 0 0 0-5.487 0 12.51 12.51 0 0 0-.617-1.249.077.077 0 0 0-.074-.035 19.736 19.736 0 0 0-3.76 1.169.07.07 0 0 0-.032.027C2.533 8.046 1.79 11.624 2.155 15.157a.082.082 0 0 0 .031.056 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.027c.462-.63.873-1.295 1.226-1.994a.076.076 0 0 0-.041-.105 13.13 13.13 0 0 1-1.873-.892.077.077 0 0 1-.008-.128c.126-.094.252-.192.372-.291a.074.074 0 0 1 .077-.01c3.927 1.793 8.18 1.793 12.061 0a.074.074 0 0 1 .078.009c.12.099.246.198.373.292a.077.077 0 0 1-.006.128 12.32 12.32 0 0 1-1.873.891.077.077 0 0 0-.04.106c.36.698.772 1.363 1.225 1.993a.076.076 0 0 0 .084.028 19.84 19.84 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-4.087-.838-7.636-3.548-10.787a.061.061 0 0 0-.031-.028zM8.02 12.997c-1.182 0-2.156-1.085-2.156-2.419 0-1.333.955-2.419 2.156-2.419 1.21 0 2.175 1.095 2.156 2.42 0 1.333-.955 2.418-2.156 2.418zm7.974 0c-1.182 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.175 1.095 2.156 2.42 0 1.333-.946 2.418-2.156 2.418z" />
  </svg>
);

/** "GitHub" and "Discord" are product names whose catalog entries are the same
 * in every language — keyed anyway so every link has one shape. */
const links = [
  {
    href: "https://github.com/makermods-robotics/makermodslab",
    labelKey: "launchpad.footer.github",
    Icon: Github,
  },
  {
    href: "https://github.com/makermods-robotics/makermodslab",
    labelKey: "launchpad.footer.documentation",
    Icon: BookOpen,
  },
  {
    href: "https://discord.gg/HpXj3ynhhF",
    labelKey: "launchpad.footer.discord",
    Icon: DiscordIcon,
  },
] as const;

const Footer: React.FC = () => {
  const { t } = useTranslation();

  return (
    <footer className="fixed inset-x-0 bottom-0 z-30 border-t border-border bg-background/95 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-3 px-4 py-4 text-sm text-muted-foreground sm:flex-row">
        <span className="flex items-center gap-2">
          <span>
            <Trans
              i18nKey="launchpad.footer.poweredBy"
              components={[
                <span key="0" />,
                <a
                  key="1"
                  href="https://github.com/huggingface/lerobot"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-foreground hover:text-foreground/80"
                />,
              ]}
            />
          </span>
          <a
            href="https://huggingface.co"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Hugging Face"
          >
            <img src={hfLogo} alt="Hugging Face" className="h-5 w-auto" />
          </a>
          <a
            href="https://github.com/huggingface/lerobot"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="LeRobot"
          >
            <img src={lerobotLogo} alt="LeRobot" className="h-5 w-auto rounded" />
          </a>
        </span>
        <nav className="flex items-center gap-4">
          {links.map(({ href, labelKey, Icon }) => (
            <a
              key={labelKey}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground"
            >
              <Icon className="h-4 w-4" />
              <span>{t(labelKey)}</span>
            </a>
          ))}
        </nav>
      </div>
    </footer>
  );
};

export default Footer;
