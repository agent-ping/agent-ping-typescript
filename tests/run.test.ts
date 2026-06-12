import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

describe("Run lifecycle", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("constructs id synchronously before HTTP", () => {
    let httpStarted = false;
    const fetchMock = vi.fn(async () => {
      httpStarted = true;
      return new Response("{}", { status: 202 });
    });
    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 60_000,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const run = agentping.run("a");
    expect(httpStarted).toBe(false);
    expect(run.id).toMatch(/^run_eu_[0-9a-f]{32}$/);
  });

  it("event ids follow the prefixed format", async () => {
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
    const run = agentping.run("a");
    run.event("log", { ok: true });
    await agentping.flush({ timeoutMs: 1_000 });

    const evtCall = calls.find((c) => c.url.includes("/events"));
    expect(evtCall).toBeTruthy();
    const body = evtCall!.body as {
      events: Array<{ external_id: string; type: string }>;
    };
    expect(body.events[0]!.external_id).toMatch(/^evt_eu_[0-9a-f]{32}$/);
    expect(body.events[0]!.type).toBe("log");
  });

  it("finish posts to /v1/runs/{id}/finish with status", async () => {
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
    const run = agentping.run("a");
    await run.finish({ status: "success", scores: { ok: 1 } });
    await agentping.flush({ timeoutMs: 1_000 });

    const finishCall = calls.find((c) => c.url.endsWith("/finish"));
    expect(finishCall).toBeTruthy();
    const body = finishCall!.body as Record<string, unknown>;
    expect(body["status"]).toBe("success");
    expect(body["finished_at"]).toMatch(/T.*Z$/);
    expect(body["scores"]).toEqual({ ok: 1 });
  });

  it("finish is idempotent: second call is a no-op", async () => {
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
    const run = agentping.run("a");
    await run.finish({ status: "success" });
    await run.finish({ status: "failed" });
    await agentping.flush({ timeoutMs: 1_000 });

    const finishCalls = calls.filter((c) => c.url.endsWith("/finish"));
    expect(finishCalls.length).toBe(1);
  });

  it("run start body carries customer_id, feature, metadata", async () => {
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
    agentping.run("a", {
      customerId: "cust_1",
      feature: "triage",
      metadata: { region: "uk" },
    });
    await agentping.flush({ timeoutMs: 1_000 });
    const startCall = calls.find(
      (c) => c.url.endsWith("/v1/runs") && !c.url.includes("events"),
    );
    expect(startCall).toBeTruthy();
    const body = startCall!.body as Record<string, unknown>;
    expect(body["agent"]).toBe("a");
    expect(body["customer_id"]).toBe("cust_1");
    expect(body["feature"]).toBe("triage");
    expect(body["metadata"]).toEqual({ region: "uk" });
  });

  it("run start body carries goal when set and omits it otherwise", async () => {
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
    agentping.run("digest", { goal: "Send the daily digest" });
    agentping.run("digest-two");
    await agentping.flush({ timeoutMs: 1_000 });

    const starts = calls.filter(
      (c) => c.url.endsWith("/v1/runs") && !c.url.includes("events"),
    );
    const withGoal = starts.find(
      (c) => (c.body as Record<string, unknown>)["agent"] === "digest",
    );
    const withoutGoal = starts.find(
      (c) => (c.body as Record<string, unknown>)["agent"] === "digest-two",
    );
    expect((withGoal!.body as Record<string, unknown>)["goal"]).toBe(
      "Send the daily digest",
    );
    expect(withoutGoal!.body as Record<string, unknown>).not.toHaveProperty(
      "goal",
    );
  });
});
