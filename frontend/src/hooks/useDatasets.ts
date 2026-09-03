import { useCallback, useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { useSessionEvent } from "@/hooks/useActiveSession";
import { DatasetItem, listDatasets } from "@/lib/replayApi";

export const useDatasets = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setLoading(true);
    listDatasets(baseUrl, fetchWithHeaders)
      .then(setDatasets)
      .catch(() => setDatasets([]))
      .finally(() => setLoading(false));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // A recording or coaching (DAgger) session that just ended may have added a
  // local dataset — or removed one. `session_changed` is a droppable refetch
  // hint (see useActiveSession): re-pull the listing when one ends. Needed
  // because this hook's mount effect fires only once per page load and the
  // studio panels that call it never unmount within a visit, so nothing else
  // re-pulls after a session. The `seenAt` ref ignores the event already
  // present at mount, so this never double-fires with the mount effect.
  const sessionEvent = useSessionEvent();
  const seenAt = useRef(sessionEvent?.receivedAt ?? 0);
  useEffect(() => {
    if (!sessionEvent || sessionEvent.receivedAt === seenAt.current) return;
    seenAt.current = sessionEvent.receivedAt;
    if (sessionEvent.active) return;
    if (sessionEvent.kind !== "recording" && sessionEvent.kind !== "inference") {
      return;
    }
    refresh();
  }, [sessionEvent, refresh]);

  return { datasets, loading, refresh };
};
