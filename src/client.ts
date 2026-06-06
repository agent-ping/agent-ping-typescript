import { warnOnce } from "./warnings.js";

export interface HttpResult {
  ok: boolean;
  status: number;
  body: unknown;
  retryAfterMs?: number;
  errorClass?: "auth_error" | "network_error" | "server_error" | "client_error";
}

export interface ClientOptions {
  apiKey: string;
  baseUrl: string;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
}

const DEFAULT_TIMEOUT_MS = 2_000;

export class HttpClient {
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(opts: ClientOptions) {
    this.apiKey = opts.apiKey;
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.fetchImpl = opts.fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  async post(path: string, payload: unknown): Promise<HttpResult> {
    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await this.fetchImpl(url, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${this.apiKey}`,
          "user-agent": "agentping-ts/0.1",
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const text = await response.text();
      let body: unknown = null;
      if (text) {
        try {
          body = JSON.parse(text);
        } catch {
          body = text;
        }
      }

      if (response.ok) {
        return { ok: true, status: response.status, body };
      }

      const retryAfter = parseRetryAfter(response.headers.get("retry-after"));
      let errorClass: HttpResult["errorClass"];
      if (response.status === 401 || response.status === 403) {
        errorClass = "auth_error";
        warnOnce(
          "auth_error",
          "API key rejected; check AGENTPING_API_KEY. Events will be dropped.",
        );
      } else if (response.status >= 500) {
        errorClass = "server_error";
        warnOnce(
          "server_error",
          `AgentPing API returned ${response.status}; will retry with back-off.`,
        );
      } else if (response.status === 429) {
        errorClass = "server_error";
      } else {
        errorClass = "client_error";
      }

      const result: HttpResult = {
        ok: false,
        status: response.status,
        body,
        errorClass,
      };
      if (retryAfter !== undefined) result.retryAfterMs = retryAfter;
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      warnOnce(
        "network_error",
        `network error contacting AgentPing API: ${message}`,
      );
      return {
        ok: false,
        status: 0,
        body: null,
        errorClass: "network_error",
      };
    } finally {
      clearTimeout(timer);
    }
  }
}

function parseRetryAfter(header: string | null): number | undefined {
  if (!header) return undefined;
  const trimmed = header.trim();
  const asNumber = Number(trimmed);
  if (Number.isFinite(asNumber) && asNumber >= 0) {
    return Math.min(60_000, Math.round(asNumber * 1000));
  }
  const asDate = Date.parse(trimmed);
  if (!Number.isNaN(asDate)) {
    const delta = asDate - Date.now();
    return delta > 0 ? Math.min(60_000, delta) : 0;
  }
  return undefined;
}
