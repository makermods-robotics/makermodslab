export type Fetcher = (
  url: string,
  options?: RequestInit
) => Promise<Response>;

export class ApiError extends Error {
  status: number;
  detail: string | null;
  /** Machine-readable error code (`<domain>.<condition>`, e.g. "session.held")
   * — an additive sibling of `detail` in the backend's error bodies. Null when
   * the response carried none. Branch on this, never on the prose. */
  code: string | null;
  /** Structured context beside the code (e.g. session.held's
   * `{ holder: { kind, session_id } }`). Shape depends on the code. */
  details: unknown;
  constructor(
    message: string,
    status: number,
    detail: string | null,
    code: string | null = null,
    details: unknown = null
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.code = code;
    this.details = details;
  }
}

export interface ApiRequestOptions {
  method?: string;
  body?: unknown;
  signal?: AbortSignal;
  /** Human-readable label for the error message, e.g. "Start training". */
  action?: string;
}

/**
 * Performs a request against the MakerMods Lab backend and parses the JSON response.
 * Throws ApiError with FastAPI's `detail` field on non-2xx, or on JSON parse
 * failure. Use this in place of ad-hoc `r.ok` / `r.json()` branching.
 */
export async function apiRequest<T = unknown>(
  baseUrl: string,
  fetcher: Fetcher,
  path: string,
  { method = "GET", body, signal, action }: ApiRequestOptions = {}
): Promise<T> {
  const init: RequestInit = { method, signal };
  if (body !== undefined) init.body = JSON.stringify(body);

  const url = `${baseUrl}${path}`;
  const r = await fetcher(url, init);
  if (!r.ok) {
    let detail: string | null = null;
    let code: string | null = null;
    let details: unknown = null;
    try {
      const errBody = await r.json();
      const raw = errBody?.detail ?? errBody?.message ?? null;
      // FastAPI 422s put an array of {loc,msg,type} in `detail`; most errors a
      // string. Normalize non-strings so they never render as "[object Object]".
      if (raw == null || typeof raw === "string") {
        detail = raw;
      } else if (Array.isArray(raw)) {
        detail = raw.map((d) => d?.msg ?? JSON.stringify(d)).join("; ");
      } else {
        detail = JSON.stringify(raw);
      }
      // Coded errors carry `code` (and sometimes `details`) beside `detail`.
      if (typeof errBody?.code === "string") code = errBody.code;
      if (errBody?.details != null) details = errBody.details;
    } catch {
      // body wasn't JSON
    }
    const label = action || `${method} ${path}`;
    throw new ApiError(
      `${label} failed: ${detail ?? r.status}`,
      r.status,
      detail,
      code,
      details
    );
  }
  // 204 No Content
  if (r.status === 204) return undefined as T;
  return r.json() as Promise<T>;
}
