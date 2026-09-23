import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useApi } from "@/contexts/ApiContext";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { useToast } from "@/hooks/use-toast";
import { getCurrentSession, stopSession } from "@/lib/sessionApi";

function RecoveryToastBody({ sessionId, message, onReleased }: {
  sessionId: string;
  message: string;
  onReleased: () => void;
}) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { t } = useTranslation();
  const [supported, setSupported] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const release = async () => {
    if (!supported || busy) return;
    setBusy(true);
    setError(null);
    try {
      await stopSession(baseUrl, fetchWithHeaders, sessionId, true);
      onReleased();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3">
      <p className="break-words">{message}</p>
      <label className="flex items-start gap-2">
        <Checkbox checked={supported} onCheckedChange={(value) => setSupported(value === true)} />
        {t("dialogs.teleop.armSupported")}
      </label>
      <Button variant="outline" size="sm" disabled={!supported || busy} onClick={release}>
        {t("dialogs.teleop.releaseTorque")}
      </Button>
      {error && <p role="alert">{error}</p>}
    </div>
  );
}

/** Failed landing retains the hardware; use the app's shared error toaster. */
export default function TeleopRecoveryNotice() {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { t } = useTranslation();
  const { toast, toasts } = useToast();
  const visibleToasts = useRef(toasts);
  visibleToasts.current = toasts;
  const notice = useRef<{ sessionId: string; id: string; dismiss: () => void } | null>(null);

  useEffect(() => {
    let cancelled = false;
    let pending = false;
    const clear = () => {
      notice.current?.dismiss();
      notice.current = null;
    };
    const poll = async () => {
      if (pending) return;
      pending = true;
      try {
        const response = await fetchWithHeaders(`${baseUrl}/api/v1/teleoperation-status`);
        if (!response.ok) return;
        const status = await response.json();
        if (cancelled) return;
        if (!status.rest_failed) {
          clear();
          return;
        }
        const { session } = await getCurrentSession(baseUrl, fetchWithHeaders);
        if (cancelled) return;
        if (session?.kind !== "teleoperation" || session.phase !== "rest_failed") {
          clear();
          return;
        }
        // Reoffer recovery on the next poll if another toast replaced it or
        // it was dismissed. Never release hardware by dismissing a toast.
        if (notice.current?.sessionId === session.id &&
            visibleToasts.current.some((item) => item.id === notice.current?.id && item.open)) return;
        clear();
        const handle = toast({
          title: t("dialogs.teleop.restFailed"),
          variant: "destructive",
          duration: Infinity,
          description: <RecoveryToastBody sessionId={session.id} message={status.message} onReleased={clear} />,
        });
        notice.current = { sessionId: session.id, id: handle.id, dismiss: handle.dismiss };
      } catch {
        // Retain the recovery toast if the connection goes away.
      } finally {
        pending = false;
      }
    };
    void poll();
    const timer = setInterval(poll, 2000);
    return () => { cancelled = true; clearInterval(timer); clear(); };
  }, [baseUrl, fetchWithHeaders, t, toast]);

  return null;
}
