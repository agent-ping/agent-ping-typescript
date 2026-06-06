import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

describe("heartbeat", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("posts to /v1/heartbeats with id and timestamps", async () => {
    const calls: Array<{ url: string; body: unknown }> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: JSON.parse(init!.body as string) });
      return new Response("{}", { status: 202 });
    });
    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 5,
      batchSize: 10,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const id = agentping.heartbeat("daily-summary", {
      status: "ok",
      costUsd: 0.084,
      durationMs: 12_300,
      metadata: { rows: 421 },
    });

    expect(id).toMatch(/^run_eu_[0-9a-f]{32}$/);

    await agentping.flush({ timeoutMs: 1_000 });

    const hb = calls.find((c) => c.url.endsWith("/v1/heartbeats"));
    expect(hb).toBeTruthy();
    const body = hb!.body as Record<string, unknown>;
    expect(body["id"]).toBe(id);
    expect(body["agent"]).toBe("daily-summary");
    expect(body["status"]).toBe("ok");
    expect(body["cost_usd"]).toBe(0.084);
    expect(body["started_at"]).toMatch(/T.*Z$/);
    expect(body["finished_at"]).toMatch(/T.*Z$/);
    expect(body["metadata"]).toEqual({ rows: 421 });
  });
});
