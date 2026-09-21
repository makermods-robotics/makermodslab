import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useApi } from "@/contexts/ApiContext";

/** Poll completed JPEG snapshots, never opening another capture device or
 * keeping one HTTP connection per camera occupied for the entire session. */
export default function RecordingCameraStream({ cameraName }: { cameraName: string }) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { t } = useTranslation();
  const [src, setSrc] = useState<string>();
  const [waiting, setWaiting] = useState(true);

  useEffect(() => {
    let disposed = false;
    let objectUrl: string | undefined;
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController;
    setSrc(undefined);
    setWaiting(true);
    const tick = async () => {
      controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 10000);
      try {
        const response = await fetchWithHeaders(
          `${baseUrl}/api/v1/recording-preview/${encodeURIComponent(cameraName)}`,
          { signal: controller.signal, cache: "no-store" },
        );
        if (!response.ok) throw new Error("Frame unavailable");
        const blob = await response.blob();
        if (disposed) return;
        const next = URL.createObjectURL(blob);
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = next;
        setSrc(next);
        setWaiting(false);
      } catch {
        if (!disposed) setWaiting(true);
      } finally {
        clearTimeout(timeout);
        if (!disposed) timer = setTimeout(tick, 200);
      }
    };
    void tick();
    return () => {
      disposed = true;
      clearTimeout(timer);
      controller?.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [baseUrl, fetchWithHeaders, cameraName]);

  return (
    <>
      {src && <img src={src} alt={cameraName} className="h-full w-full object-contain" />}
      {waiting && (
        <div className="absolute inset-0 flex items-center justify-center bg-muted/80 p-3 text-center text-xs text-muted-foreground">
          {t("shared.camera.waiting")}
        </div>
      )}
    </>
  );
}
