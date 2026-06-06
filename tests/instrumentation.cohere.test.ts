import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { instrumentCohere } from "../src/instrumentation/cohere.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

describe("instrumentCohere", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("captures billed_units on chat", async () => {
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

    const fakeClient = {
      chat: vi.fn(async (_args?: unknown) => ({
        usage: {
          billedUnits: { inputTokens: 320, outputTokens: 140 },
        },
      })),
    };

    const run = agentping.run("rfp");
    const wrapped = instrumentCohere(fakeClient, { run });
    await wrapped.chat({
      model: "command-r-plus-08-2024",
      messages: [{ role: "user", content: "classify" }],
    });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    expect(eventCall).toBeTruthy();
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("cohere");
    expect(llm.data["model"]).toBe("command-r-plus-08-2024");
    expect(llm.data["input_tokens"]).toBe(320);
    expect(llm.data["output_tokens"]).toBe(140);
    expect(llm.data["latency_ms"]).toBeTypeOf("number");
  });

  it("emits zero token fields when billed_units missing but still records the call", async () => {
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

    const fakeClient = {
      chat: vi.fn(async (_args?: unknown) => ({})),
    };

    const run = agentping.run("rfp");
    const wrapped = instrumentCohere(fakeClient, { run });
    await wrapped.chat({ model: "command-light", messages: [] });
    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("cohere");
    expect(llm.data["model"]).toBe("command-light");
    expect(llm.data["input_tokens"]).toBe(0);
    expect(llm.data["output_tokens"]).toBe(0);
  });
});
