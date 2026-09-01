import React, { createContext, useContext, ReactNode, useState, useCallback, useEffect, useMemo, useRef } from "react";
import { mockHubResponse } from "@/lib/mockHub";

interface ApiContextType {
  baseUrl: string;
  wsBaseUrl: string;
  fetchWithHeaders: (url: string, options?: RequestInit) => Promise<Response>;
}

const ApiContext = createContext<ApiContextType | undefined>(undefined);

const STORAGE_KEY = "makermodslab.apiBaseUrl";
const DEFAULT_LOCALHOST = "http://localhost:8000";

// In production the backend serves the UI, so the page origin is the API. In
// Vite dev the UI is on :8080 while the API stays on :8000, so the origin
// would be wrong there.
const defaultBaseUrl = (): string =>
  import.meta.env.DEV ? DEFAULT_LOCALHOST : window.location.origin;

const httpToWs = (url: string): string => url.replace(/^http(s?):/, "ws$1:");

const resolveInitialBaseUrl = (): string => {
  if (typeof window === "undefined") return DEFAULT_LOCALHOST;

  const fromQuery = new URLSearchParams(window.location.search).get("api");
  if (fromQuery) {
    try {
      new URL(fromQuery);
      const clean = fromQuery.replace(/\/$/, "");
      window.localStorage.setItem(STORAGE_KEY, clean);
      return clean;
    } catch {
      console.warn("Invalid `api` query param, ignoring:", fromQuery);
    }
  }

  return window.localStorage.getItem(STORAGE_KEY) || defaultBaseUrl();
};

export const ApiProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [baseUrl, setBaseUrl] = useState<string>(resolveInitialBaseUrl);
  const wsBaseUrl = httpToWs(baseUrl);

  // Self-heal a stale saved override. A `?api=` visit persists its URL to
  // localStorage forever — if that server is gone (temporary HTTPS/phone-cam
  // setup, a dead port), every request "Failed to fetch" with no way back
  // from the UI. Probe the override once on startup; if it's unreachable,
  // drop it and fall back to the default so plain http://localhost:8080/
  // always recovers on reload. An HTTP error status still counts as
  // reachable — only a network-level failure (or a 4s timeout) falls back.
  const probedRef = useRef(false);
  useEffect(() => {
    if (probedRef.current) return;
    probedRef.current = true;
    const fallback = defaultBaseUrl();
    if (baseUrl === fallback) return;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 4000);
    fetch(`${baseUrl}/api/v1/hf-auth-status`, { signal: controller.signal })
      .catch(() => {
        console.warn(
          `Saved API address ${baseUrl} is unreachable — falling back to ${fallback}.`,
        );
        try {
          window.localStorage.removeItem(STORAGE_KEY);
        } catch {
          // storage unavailable — the in-memory fallback still applies
        }
        setBaseUrl(fallback);
      })
      .finally(() => clearTimeout(timer));
    return () => {
      clearTimeout(timer);
    };
  }, [baseUrl]);

  const fetchWithHeaders = useCallback(async (url: string, options: RequestInit = {}): Promise<Response> => {
    // Dev-only Hub mock (?mockHub=1): serves canned jobs/models/auth so UI
    // work can continue through a huggingface.co outage. No-op in prod builds.
    if (import.meta.env.DEV) {
      const mocked = mockHubResponse(url, options);
      if (mocked) return mocked;
    }
    return fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });
  }, []);

  const value = useMemo(
    () => ({ baseUrl, wsBaseUrl, fetchWithHeaders }),
    [baseUrl, wsBaseUrl, fetchWithHeaders]
  );

  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
};

export const useApi = (): ApiContextType => {
  const context = useContext(ApiContext);
  if (context === undefined) {
    throw new Error("useApi must be used within an ApiProvider");
  }
  return context;
};
