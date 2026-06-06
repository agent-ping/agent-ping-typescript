import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { instrumentBedrock } from "../src/instrumentation/bedrock.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

/** Stand-ins for AWS SDK v3 command classes. Constructor.name is what
 *  the wrapper inspects. */
class ConverseCommand {
  constructor(public input: { modelId?: string }) {}
}
class ConverseStreamCommand {
  constructor(public input: { modelId?: string }) {}
}
class InvokeModelCommand {
  constructor(public input: { modelId?: string }) {}
}
class ListFoundationModelsCommand {
  constructor(public input: Record<string, unknown> = {}) {}
}

async function* asyncIter<T>(items: T[]): AsyncIterable<T> {
  for (const it of items) yield it;
}

describe("instrumentBedrock", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("ConverseCommand emits llm_call with usage and model", async () => {
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
      send: vi.fn(async (_command: unknown) => ({
        usage: { inputTokens: 420, outputTokens: 88 },
        output: { message: { role: "assistant", content: [{ text: "hi" }] } },
        stopReason: "end_turn",
      })),
    };

    const run = agentping.run("order-classifier");
    const wrapped = instrumentBedrock(fakeClient, { run });

    await wrapped.send(new ConverseCommand({ modelId: "eu.anthropic.claude-sonnet-4-5-20250929-v1:0" }));

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("bedrock");
    expect(llm.data["model"]).toBe("eu.anthropic.claude-sonnet-4-5-20250929-v1:0");
    expect(llm.data["input_tokens"]).toBe(420);
    expect(llm.data["output_tokens"]).toBe(88);
    expect(llm.data["latency_ms"]).toBeTypeOf("number");
  });

  it("ConverseCommand splits cache_read into cached_input_tokens", async () => {
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
      send: vi.fn(async (_command: unknown) => ({
        usage: { inputTokens: 1000, outputTokens: 100, cacheReadInputTokens: 400 },
      })),
    };

    const run = agentping.run("cache-test");
    const wrapped = instrumentBedrock(fakeClient, { run });
    await wrapped.send(new ConverseCommand({ modelId: "eu.anthropic.claude-sonnet-4-5-20250929-v1:0" }));

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["input_tokens"]).toBe(600); // 1000 - 400 cached
    expect(llm.data["cached_input_tokens"]).toBe(400);
    expect(llm.data["output_tokens"]).toBe(100);
  });

  it("InvokeModelCommand reads token counts from $metadata.httpHeaders", async () => {
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
      send: vi.fn(async (_command: unknown) => ({
        body: new Uint8Array(),
        $metadata: {
          httpHeaders: {
            "x-amzn-bedrock-input-token-count": "256",
            "x-amzn-bedrock-output-token-count": "44",
          },
        },
      })),
    };

    const run = agentping.run("invoke");
    const wrapped = instrumentBedrock(fakeClient, { run });
    await wrapped.send(new InvokeModelCommand({ modelId: "amazon.titan-text-lite-v1" }));

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["input_tokens"]).toBe(256);
    expect(llm.data["output_tokens"]).toBe(44);
  });

  it("non-bedrock commands are passed through without events", async () => {
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

    let nonChatCalls = 0;
    const fakeClient = {
      send: vi.fn(async (_command: unknown) => {
        nonChatCalls++;
        return { models: [] };
      }),
    };

    const run = agentping.run("listing");
    const wrapped = instrumentBedrock(fakeClient, { run });
    await wrapped.send(new ListFoundationModelsCommand());

    await agentping.flush({ timeoutMs: 1_000 });
    expect(nonChatCalls).toBe(1);
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const events = eventCall ? ((eventCall.body as { events: Array<{ type: string }> }).events) : [];
    expect(events.filter((e) => e.type === "llm_call")).toHaveLength(0);
  });

  it("ConverseStreamCommand captures final usage from metadata event", async () => {
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
      send: vi.fn(async (_command: unknown) => ({
        stream: asyncIter([
          { contentBlockDelta: { delta: { text: "Hello" } } },
          { contentBlockDelta: { delta: { text: " world" } } },
          { metadata: { usage: { inputTokens: 180, outputTokens: 24 } } },
        ]),
      })),
    };

    const run = agentping.run("stream-converse");
    const wrapped = instrumentBedrock(fakeClient, { run });
    const response = await wrapped.send(new ConverseStreamCommand({
      modelId: "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    })) as { stream: AsyncIterable<unknown> };
    for await (const _ of response.stream) {
      void _;
    }

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("bedrock");
    expect(llm.data["input_tokens"]).toBe(180);
    expect(llm.data["output_tokens"]).toBe(24);
  });

  it("resolves active run from runScopeAsync when no run option is passed", async () => {
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
      send: vi.fn(async (_command: unknown) => ({ usage: { inputTokens: 50, outputTokens: 10 } })),
    };

    const wrapped = instrumentBedrock(fakeClient); // no { run }
    const run = agentping.run("scoped-bedrock");
    await agentping.runScopeAsync(run, async () => {
      await wrapped.send(new ConverseCommand({ modelId: "us.anthropic.claude-3-5-sonnet-20240620-v1:0" }));
    });

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["input_tokens"]).toBe(50);
  });
});
