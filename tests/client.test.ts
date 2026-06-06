import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HttpClient } from "../src/client.js";
import { _resetWarningsForTests } from "../src/warnings.js";

function jsonResponse(
  status: number,
  body: unknown,
  headers: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

describe("HttpClient", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("sends bearer auth and JSON body", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse(202, { ok: true }));
    const client = new HttpClient({
      apiKey: "apk_eu_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      baseUrl: "https://api.example.com",
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const result = await client.post("/v1/runs", { id: "run_eu_x" });
    expect(result.ok).toBe(true);
    expect(result.status).toBe(202);

    const call = fetchMock.mock.calls[0]!;
    expect(call[0]).toBe("https://api.example.com/v1/runs");
    const init = call[1] as RequestInit;
    expect(init.method).toBe("POST");
    const headers = init.headers as Record<string, string>;
    expect(headers["authorization"]).toBe(
      "Bearer apk_eu_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    );
    expect(headers["content-type"]).toBe("application/json");
    expect(JSON.parse(init.body as string)).toEqual({ id: "run_eu_x" });
  });

  it("aborts after 2s timeout", async () => {
    let abortedSignal: AbortSignal | undefined;
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
      abortedSignal = init?.signal as AbortSignal | undefined;
      return new Promise<Response>((_resolve, reject) => {
        abortedSignal?.addEventListener("abort", () => {
          reject(new DOMException("aborted", "AbortError"));
        });
      });
    });

    const client = new HttpClient({
      apiKey: "apk_eu_a".padEnd(39, "a"),
      baseUrl: "https://api.example.com",
      timeoutMs: 20,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const result = await client.post("/v1/runs", {});
    expect(result.ok).toBe(false);
    expect(result.errorClass).toBe("network_error");
  });

  it("emits auth_error class on 401", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse(401, { error: "no" }));
    const client = new HttpClient({
      apiKey: "apk_eu_a".padEnd(39, "a"),
      baseUrl: "https://api.example.com",
      fetchImpl: fetchMock as unknown as typeof fetch,
    });
    const r = await client.post("/v1/runs", {});
    expect(r.ok).toBe(false);
    expect(r.errorClass).toBe("auth_error");
  });

  it("parses Retry-After header (seconds)", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse(429, { error: "slow" }, { "retry-after": "3" }));
    const client = new HttpClient({
      apiKey: "apk_eu_a".padEnd(39, "a"),
      baseUrl: "https://api.example.com",
      fetchImpl: fetchMock as unknown as typeof fetch,
    });
    const r = await client.post("/v1/runs", {});
    expect(r.ok).toBe(false);
    expect(r.retryAfterMs).toBe(3_000);
  });

  it("marks 500s as server_error", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse(503, { error: "boom" }));
    const client = new HttpClient({
      apiKey: "apk_eu_a".padEnd(39, "a"),
      baseUrl: "https://api.example.com",
      fetchImpl: fetchMock as unknown as typeof fetch,
    });
    const r = await client.post("/v1/runs", {});
    expect(r.errorClass).toBe("server_error");
  });

  it("warns at most once per error class", async () => {
    vi.restoreAllMocks();
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const fetchMock = vi.fn(async () => jsonResponse(401, { error: "no" }));
    const client = new HttpClient({
      apiKey: "apk_eu_a".padEnd(39, "a"),
      baseUrl: "https://api.example.com",
      fetchImpl: fetchMock as unknown as typeof fetch,
    });
    await client.post("/v1/runs", {});
    await client.post("/v1/runs", {});
    await client.post("/v1/runs", {});
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });
});
