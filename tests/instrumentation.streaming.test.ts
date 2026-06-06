import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { instrumentCohere } from "../src/instrumentation/cohere.js";
import { instrumentGemini } from "../src/instrumentation/gemini.js";
import { instrumentMistral } from "../src/instrumentation/mistral.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

async function* asyncIter<T>(items: T[]): AsyncIterable<T> {
  for (const it of items) yield it;
}

describe("TypeScript provider streaming + embed", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  // --- Gemini ---

  it("instrumentGemini wraps generateContentStream and captures final usage", async () => {
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
        generateContent: vi.fn(async (_args?: unknown) => ({})),
        generateContentStream: vi.fn(async (_args?: unknown) =>
          asyncIter([
            { modelVersion: "gemini-2.0-flash" },
            { modelVersion: "gemini-2.0-flash" },
            {
              modelVersion: "gemini-2.0-flash",
              usageMetadata: {
                promptTokenCount: 250,
                candidatesTokenCount: 90,
                cachedContentTokenCount: 100,
              },
            },
          ]),
        ),
      },
    };

    const run = agentping.run("stream-research");
    const wrapped = instrumentGemini(fakeClient, { run });

    const stream = (await wrapped.models.generateContentStream({
      model: "gemini-2.0-flash",
      contents: "hi",
    })) as AsyncIterable<unknown>;

    let count = 0;
    for await (const _ of stream) count++;
    expect(count).toBe(3);

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("gemini");
    expect(llm.data["model"]).toBe("gemini-2.0-flash");
    expect(llm.data["input_tokens"]).toBe(150); // 250 - 100 cached
    expect(llm.data["cached_input_tokens"]).toBe(100);
    expect(llm.data["output_tokens"]).toBe(90);
  });

  it("instrumentGemini wraps embedContent and emits kind: embedding event", async () => {
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
        generateContent: vi.fn(async (_args?: unknown) => ({})),
        embedContent: vi.fn(async (_args?: unknown) => ({
          usageMetadata: { totalTokenCount: 220 },
        })),
      },
    };

    const run = agentping.run("rag-build");
    const wrapped = instrumentGemini(fakeClient, { run });
    await wrapped.models.embedContent!({ model: "text-embedding-004", contents: ["a"] });
    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["kind"]).toBe("embedding");
    expect(llm.data["input_tokens"]).toBe(220);
    expect(llm.data["output_tokens"]).toBe(0);
  });

  // --- Mistral ---

  it("instrumentMistral wraps chat.stream and captures final usage", async () => {
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
        complete: vi.fn(async (_args?: unknown) => ({})),
        stream: vi.fn(async (_args?: unknown) =>
          asyncIter([
            { data: { model: "mistral-large-latest" } },
            { data: { model: "mistral-large-latest", usage: { promptTokens: 320, completionTokens: 110 } } },
          ]),
        ),
      },
    };

    const run = agentping.run("translate-stream");
    const wrapped = instrumentMistral(fakeClient, { run });
    const stream = (await wrapped.chat.stream!({ model: "mistral-large-latest", messages: [] })) as AsyncIterable<unknown>;
    for await (const _ of stream) {
      void _;
    }

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["input_tokens"]).toBe(320);
    expect(llm.data["output_tokens"]).toBe(110);
    expect(llm.data["model"]).toBe("mistral-large-latest");
  });

  // --- Cohere ---

  it("instrumentCohere wraps chatStream and captures billed units from message-end", async () => {
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
      chatStream: vi.fn(async (_args?: unknown) =>
        asyncIter([
          { type: "content-delta" },
          { type: "content-delta" },
          { type: "message-end", delta: { usage: { billedUnits: { inputTokens: 280, outputTokens: 95 } } } },
        ]),
      ),
    };

    const run = agentping.run("rfp-stream");
    const wrapped = instrumentCohere(fakeClient, { run });
    const stream = (await wrapped.chatStream!({ model: "command-r-plus-08-2024", messages: [] })) as AsyncIterable<unknown>;
    for await (const _ of stream) {
      void _;
    }

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["input_tokens"]).toBe(280);
    expect(llm.data["output_tokens"]).toBe(95);
  });

  it("instrumentCohere wraps embed and emits kind: embedding event", async () => {
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
      embed: vi.fn(async (_args?: unknown) => ({
        meta: { billedUnits: { inputTokens: 540 } },
      })),
    };

    const run = agentping.run("rag-index");
    const wrapped = instrumentCohere(fakeClient, { run });
    await wrapped.embed!({ model: "embed-english-v3.0", texts: ["a", "b"] });
    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["kind"]).toBe("embedding");
    expect(llm.data["input_tokens"]).toBe(540);
    expect(llm.data["output_tokens"]).toBe(0);
  });
});
