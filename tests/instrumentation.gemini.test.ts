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

  it("captures gross token usage with the cached subset and thinking tokens", async () => {
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
          candidates: [
            {
              finishReason: "STOP",
              content: { parts: [{ text: "ok" }, { functionCall: { name: "lookup", args: {} } }] },
            },
          ],
          usageMetadata: {
            promptTokenCount: 300,
            candidatesTokenCount: 90,
            cachedContentTokenCount: 120,
            thoughtsTokenCount: 30,
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
    // promptTokenCount already includes the cached content; ingest wants gross.
    expect(llm!.data["input_tokens"]).toBe(300);
    expect(llm!.data["cached_input_tokens"]).toBe(120);
    // Gemini reports thinking separately from candidates; output is the sum.
    expect(llm!.data["output_tokens"]).toBe(120);
    expect(llm!.data["reasoning_tokens"]).toBe(30);
    expect(llm!.data["finish_reason"]).toBe("STOP");
    expect(llm!.data["tool_calls"]).toBe(1);
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

  it("emits an errored llm_call on rejection", async () => {
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
    const err = body.events.find((e) => e.type === "llm_call")!;
    expect(err.data["provider"]).toBe("gemini");
    expect(err.data["model"]).toBe("gemini-2.0-flash");
    expect(err.data["status"]).toBe("error");
    expect(err.data["error"]).toBe("quota exhausted");
    expect(err.data["exception"]).toBe("Error");
  });
});
