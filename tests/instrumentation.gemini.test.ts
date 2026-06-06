import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { instrumentGemini } from "../src/instrumentation/gemini.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

describe("instrumentGemini", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("captures token usage with cached subtraction", async () => {
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
      models: {
        generateContent: vi.fn(async (_args?: unknown) => ({
          modelVersion: "gemini-2.0-flash",
          usageMetadata: {
            promptTokenCount: 300,
            candidatesTokenCount: 90,
            cachedContentTokenCount: 120,
          },
        })),
      },
    };

    const run = agentping.run("research");
    const wrapped = instrumentGemini(fakeClient, { run });
    await wrapped.models.generateContent({
      model: "gemini-2.0-flash",
      contents: "hi",
    });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    expect(eventCall).toBeTruthy();
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call");
    expect(llm).toBeTruthy();
    expect(llm!.data["provider"]).toBe("gemini");
    expect(llm!.data["model"]).toBe("gemini-2.0-flash");
    expect(llm!.data["input_tokens"]).toBe(180);
    expect(llm!.data["cached_input_tokens"]).toBe(120);
    expect(llm!.data["output_tokens"]).toBe(90);
    expect(llm!.data["latency_ms"]).toBeTypeOf("number");
  });

  it("omits cached_input_tokens when response has no cache block", async () => {
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
      models: {
        generateContent: vi.fn(async (_args?: unknown) => ({
          modelVersion: "gemini-1.5-flash",
          usageMetadata: { promptTokenCount: 100, candidatesTokenCount: 40 },
        })),
      },
    };

    const run = agentping.run("research");
    const wrapped = instrumentGemini(fakeClient, { run });
    await wrapped.models.generateContent({ model: "gemini-1.5-flash", contents: "hi" });
    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["input_tokens"]).toBe(100);
    expect("cached_input_tokens" in llm.data).toBe(false);
  });

  it("emits llm_call_error on rejection", async () => {
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
      models: {
        generateContent: vi.fn(async (_args?: unknown) => {
          throw new Error("quota exhausted");
        }),
      },
    };

    const run = agentping.run("research");
    const wrapped = instrumentGemini(fakeClient, { run });
    await expect(
      wrapped.models.generateContent({ model: "gemini-2.0-flash", contents: "hi" }),
    ).rejects.toThrow("quota exhausted");
    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const err = body.events.find((e) => e.type === "llm_call_error")!;
    expect(err.data["provider"]).toBe("gemini");
    expect(err.data["error"]).toBe("quota exhausted");
  });
});
