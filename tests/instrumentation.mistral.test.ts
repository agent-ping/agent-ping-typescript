import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { instrumentMistral } from "../src/instrumentation/mistral.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

describe("instrumentMistral", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("captures token usage on chat.complete", async () => {
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
      chat: {
        complete: vi.fn(async (_args?: unknown) => ({
          model: "mistral-large-latest",
          usage: { promptTokens: 220, completionTokens: 95 },
        })),
      },
    };

    const run = agentping.run("doc-translator");
    const wrapped = instrumentMistral(fakeClient, { run });
    await wrapped.chat.complete({
      model: "mistral-large-latest",
      messages: [{ role: "user", content: "translate" }],
    });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    expect(eventCall).toBeTruthy();
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("mistral");
    expect(llm.data["model"]).toBe("mistral-large-latest");
    expect(llm.data["input_tokens"]).toBe(220);
    expect(llm.data["output_tokens"]).toBe(95);
    expect(llm.data["latency_ms"]).toBeTypeOf("number");
  });

  it("falls back to requested model when response omits one", async () => {
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
      chat: {
        complete: vi.fn(async (_args?: unknown) => ({
          usage: { promptTokens: 10, completionTokens: 5 },
        })),
      },
    };

    const run = agentping.run("classify");
    const wrapped = instrumentMistral(fakeClient, { run });
    await wrapped.chat.complete({ model: "mistral-small-latest", messages: [] });
    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["model"]).toBe("mistral-small-latest");
  });
});
