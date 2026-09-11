import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown, ChevronRight } from "lucide-react";

interface LogPanelProps {
  /** The full log text to display (newline-separated). */
  logs: string;
  /** Panel heading, e.g. "Inference log" or "Recording log". Callers pass
   * already-translated text; omitted, it falls back to a generic "Log". */
  title?: string;
  /** Start collapsed. Defaults to expanded. */
  defaultCollapsed?: boolean;
  /**
   * Wrap long lines (default). Pass false to keep each log line on one line so
   * the panel's natural width is the longest line — for containers that grow
   * to fit their content instead of wrapping.
   */
  wrap?: boolean;
  className?: string;
}

/**
 * A collapsible, monospace, auto-scrolling log viewer.
 *
 * Shared by the Inference and Record pages to surface backend log output that
 * previously only reached the server console. The parent polls its endpoint and
 * feeds the latest text in via `logs`; this component only renders and manages
 * scroll/collapse state. Auto-scrolls to the bottom on new content unless the
 * user has scrolled up (so reading history isn't yanked away by new lines).
 */
const LogPanel: React.FC<LogPanelProps> = ({
  logs,
  title,
  defaultCollapsed = false,
  wrap = true,
  className = "",
}) => {
  const { t } = useTranslation();
  const heading = title ?? t("shared.logPanel.defaultTitle");
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const scrollRef = useRef<HTMLDivElement>(null);
  // Whether the view is pinned to the bottom. Starts true; flips false when the
  // user scrolls up, back true when they scroll back to the bottom.
  const pinnedRef = useRef(true);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight;
    pinnedRef.current = distanceFromBottom < 24;
  };

  useEffect(() => {
    if (collapsed) return;
    const el = scrollRef.current;
    if (el && pinnedRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs, collapsed]);

  return (
    <div
      className={`rounded-lg border border-border bg-muted overflow-hidden ${className}`}
    >
      <button
        type="button"
        onClick={() => setCollapsed((c) => !c)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground hover:bg-accent"
      >
        {collapsed ? (
          <ChevronRight className="w-4 h-4" />
        ) : (
          <ChevronDown className="w-4 h-4" />
        )}
        {title}
      </button>
      {!collapsed && (
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className={`max-h-64 px-3 pb-3 font-mono text-xs leading-relaxed text-foreground ${
            wrap
              ? "overflow-y-auto whitespace-pre-wrap break-words"
              : "overflow-auto whitespace-pre"
          }`}
        >
          {logs ? (
            logs
          ) : (
            <span className="text-muted-foreground">
              {t("shared.logPanel.waiting")}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default LogPanel;
