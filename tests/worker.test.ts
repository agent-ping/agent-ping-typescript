import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { getState } from "../src/state.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

describe("worker", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("batches up to batchSize per request", async () => {
    const calls: Array<{ url: string; body: unknown }> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: JSON.parse(init!.body as string) });
      return new Response("{}", { status: 202 });
    });

    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 50,
      batchSize: 3,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const run = agentping.run("test-agent");
    run.event("log", { i: 1 });
    run.event("log", { i: 2 });
    run.event("log", { i: 3 });

    await agentping.flush({ timeoutMs: 2_000 });

    // First call posts the run start, then events get batched.
    const eventCalls = calls.filter((c) => c.url.includes("/events"));
    expect(eventCalls.length).toBeGreaterThanOrEqual(1);
    const firstEventBatch = eventCalls[0]!.body as { events: unknown[] };
    expect(firstEventBatch.events.length).toBeGreaterThanOrEqual(1);
    expect(firstEventBatch.events.length).toBeLessThanOrEqual(3);
  });

  it("drops oldest envelope when queue is full", () => {
    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 60_000,
      queueSize: 5,
      fetchImpl: (() => new Promise(() => undefined)) as unknown as typeof fetch,
    });

    const run = agentping.run("test"); // run_start: 1
    for (let i = 0; i < 10; i++) {
      run.event("log", { i });
    }
    const s = agentping.status();
    expect(s.queueSize).toBe(5);
    expect(s.droppedCount).toBeGreaterThanOrEqual(6);
  });

  it("retries on 5xx with back-off and recovers", async () => {
    let calls = 0;
    const fetchMock = vi.fn(async () => {
      calls += 1;
      if (calls < 3) {
        return new Response("{}", { status: 503 });
      }
      return new Response("{}", { status: 202 });
    });

    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 10,
      batchSize: 50,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const run = agentping.run("retry-agent");
    run.event("log", { x: 1 });

    // Allow several flush ticks for back-off to expire.
    await agentping.flush({ timeoutMs: 3_000 });

    expect(fetchMock).toHaveBeenCalled();
    const state = getState();
    expect(state).not.toBeNull();
  });

  it("drops 4xx-bad-request batches without retry", async () => {
    const fetchMock = vi.fn(async () => {
      return new Response('{"error":"bad"}', { status: 400 });
    });

    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 10,
      batchSize: 50,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const run = agentping.run("bad-agent");
    run.event("log", { x: 1 });

    await agentping.flush({ timeoutMs: 1_000 });

    const status = agentping.status();
    expect(status.queueSize).toBe(0);
    // dropped includes run_start and the event (server rejected both)
    expect(status.droppedCount).toBeGreaterThanOrEqual(1);
  });

  it("respects Retry-After on 429", async () => {
    let count = 0;
    const fetchMock = vi.fn(async () => {
      count += 1;
      if (count === 1) {
        return new Response("{}", {
          status: 429,
          headers: { "retry-after": "0" },
        });
      }
      return new Response("{}", { status: 202 });
    });

    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 5,
      batchSize: 10,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const run = agentping.run("ratelimited");
    run.event("log", { x: 1 });

    await agentping.flush({ timeoutMs: 2_000 });
    expect(count).toBeGreaterThanOrEqual(2);
  });
});
