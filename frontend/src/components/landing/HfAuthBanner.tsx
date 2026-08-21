import React, { useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { AlertCircle, ExternalLink, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useApi } from "@/contexts/ApiContext";
import { useHfAuth } from "@/contexts/HfAuthContext";

/** The token-settings link. The trailing external-link icon lives here rather
 * than in the catalog string: <Trans> resolves only top-level slots, so an
 * icon nested inside another slot is silently dropped. */
const TokenSettingsLink: React.FC<{ children?: React.ReactNode }> = ({
  children,
}) => (
  <a
    href="https://huggingface.co/settings/tokens"
    target="_blank"
    rel="noreferrer"
    className="underline hover:text-amber-900 dark:hover:text-amber-50 inline-flex items-center gap-1"
  >
    {children}
    <ExternalLink className="w-3 h-3" />
  </a>
);

const HfAuthBanner: React.FC = () => {
  const { t } = useTranslation();
  const { auth, refetch } = useHfAuth();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [token, setToken] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (auth.status === "authenticated" || auth.status === "loading") {
    return null;
  }

  const handleSave = async () => {
    const trimmed = token.trim();
    if (!trimmed) return;
    setSubmitting(true);
    setError(null);
    try {
      const r = await fetchWithHeaders(`${baseUrl}/hf-auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: trimmed }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${r.status}`);
      }
      setToken("");
      await refetch();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="border border-amber-500/40 bg-amber-500/10 rounded-lg p-4 mb-6">
      <div className="flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
        <div className="flex-1 space-y-3">
          <div>
            <p className="text-sm text-amber-800 dark:text-amber-100 font-medium">
              {t("landing.hfAuthBanner.title")}
            </p>
            <p className="text-xs text-amber-700/90 dark:text-amber-200/80 mt-1">
              {/* One sentence with an embedded link, icon, and mono span —
                  <Trans> so the translator controls the word order instead of
                  us concatenating fragments. */}
              <Trans
                i18nKey="landing.hfAuthBanner.tokenHint"
                components={[
                  <TokenSettingsLink key="0" />,
                  <span key="1" className="font-mono" />,
                ]}
              />
            </p>
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSave();
            }}
            className="flex gap-2"
          >
            <Input
              type="password"
              placeholder="hf_..."
              value={token}
              onChange={(e) => setToken(e.target.value)}
              className=""
              disabled={submitting}
              autoComplete="off"
            />
            <Button
              type="submit"
              disabled={submitting || !token.trim()}
              className=""
            >
              {submitting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  {t("landing.hfAuthBanner.saving")}
                </>
              ) : (
                t("landing.hfAuthBanner.save")
              )}
            </Button>
          </form>
          {error && (
            <p className="text-xs text-destructive">{error}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default HfAuthBanner;
