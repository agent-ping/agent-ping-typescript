import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

function collect() {
  const calls: Array<{ url: string; body: Record<string, unknown> }> = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, body: JSON.parse(init!.body as string) });
    return new Response("{}", { status: 202 });
  });
  agentping.init({
    apiKey: VALID_KEY,
    baseUrl: "https://api.example.com",
    flushIntervalMs: 5,
    batchSize: 50,
    fetchImpl: fetchMock as unknown as typeof fetch,
  });
  return calls;
}

describe("conversation", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });
  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("collects every turn on one run, across separate requests", async () => {
    const calls = collect();
    const session = agentping.newSessionId();

    // Three turns, each rebuilt from the session id alone, as a stateless
    // backend would on three unrelated HTTP requests.
    for (const tokens of [900, 1_700, 2_400]) {
      const chat = agentping.conversation("support-chat", session);
      chat.event("llm_call", {
        provider: "anthropic",
        model: "claude-sonnet-4-6",
        input_tokens: tokens,
        output_tokens: 40,
      });
    }
    await agentping.flush({ timeoutMs: 1_000 });

    const starts = calls.filter((c) => c.url.endsWith("/v1/runs"));
    const runIds = new Set(starts.map((c) => c.body["id"]));
    expect(runIds.size).toBe(1);
    expect([...runIds][0]).toBe(`run_eu_${session}`);

    // started_at must be identical every time or the server 409s.
    expect(new Set(starts.map((c) => c.body["started_at"])).size).toBe(1);

    const events = calls.filter((c) => c.url.includes("/events"));
    const total = events.flatMap(
      (c) => (c.body["events"] as Array<Record<string, unknown>>),
    );
    expect(total).toHaveLength(3);
  });

  it("derives the start time from the session id, not the clock", async () => {
    const calls = collect();
    const session = agentping.newSessionId();
    agentping.conversation("support-chat", session);
    await agentping.flush({ timeoutMs: 1_000 });

    const start = calls.find((c) => c.url.endsWith("/v1/runs"))!;
    const expected = new Date(parseInt(session.slice(0, 12), 16)).toISOString();
    expect(start.body["started_at"]).toBe(expected);
  });

  it("opts into the idle sweep, closing as success rather than timeout", async () => {
    const calls = collect();
    agentping.conversation("support-chat", agentping.newSessionId());
    await agentping.flush({ timeoutMs: 1_000 });

    const body = calls.find((c) => c.url.endsWith("/v1/runs"))!.body;
    expect(body["idle_timeout_seconds"]).toBe(900);
    expect(body["idle_timeout_status"]).toBe("success");
  });

  it("leaves an ordinary run out of the sweep entirely", async () => {
    const calls = collect();
    agentping.run("nightly-report");
    await agentping.flush({ timeoutMs: 1_000 });

    const body = calls.find((c) => c.url.endsWith("/v1/runs"))!.body;
    expect(body["idle_timeout_seconds"]).toBeUndefined();
  });

  it("rejects a session id that is not 32 hex", () => {
    collect();
    expect(() => agentping.conversation("support-chat", "not-a-session")).toThrow(
      /32 hex/,
    );
  });
});
